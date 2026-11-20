# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tool-calling harness for any OpenAI-compatible model endpoint.

Connects to `MODEL_BASE_URL` using `MODEL_API_KEY` and `MODEL_NAME`. The model
works in the task container through four built-in tools (`bash`, `read_file`,
`write_file`, `edit_file`) plus the task's own MCP tools, and finishes when it
provides a complete answer without calling a tool. That answer becomes the final
message. Calls go to the Chat Completions API, or to the Responses API for
OpenAI's GPT-6 models (`RESPONSES_MODELS`).

Optional environment variables: `MAX_OUTPUT_TOKENS` (default `32000`),
`COMPACT_AT_TOKENS` (default `150000`), and `REASONING_EFFORT` (default `max`
for Claude, GLM, GPT-6 and DeepSeek-V4, `high` for other models).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import posixpath
import re
import struct
import sys
import time
import types
from typing import Any

import openai
import taskbridge
import file_validation
import constraint_check

MODEL = os.environ.get("MODEL_NAME", "")
# The model's highest reasoning level: max for Claude, GLM, GPT-6 and
# DeepSeek-V4, high for the others (Gemini refuses max). REASONING_EFFORT
# overrides it.
MAX_EFFORT_MODELS = ("claude", "glm", "gpt-6", "deepseek-v4")
EFFORT = os.environ.get("REASONING_EFFORT") or (
    "max" if any(m in MODEL.lower() for m in MAX_EFFORT_MODELS) else "high"
)
# OpenAI's GPT-6 models only support function tools combined with
# reasoning_effort on the Responses API. Other models also switch to
# /v1/responses if /v1/chat/completions returns a 400 directing them there.
RESPONSES_MODELS = ("gpt-6",)
RESPONSES_ONLY = re.compile(
    r"/v1/responses|not supported.{0,80} in /v1/chat/completions"
)
# Claude reuses a cached prompt prefix only where a request marks one
# (marked()); other models cache on their own.
CACHE_MARK = "claude" in MODEL.lower()
COMPACT_AT_TOKENS = int(os.environ.get("COMPACT_AT_TOKENS", "150000"))
TOOL_RESULT_CHARS = 30_000
# MCP tools are offered as functions up to this many characters of JSON in all
DIRECT_MCP_CHARS = 40_000
# functions one request may offer (OpenAI's limit), the built-in four included
MAX_TOOLS = 128
KEEP_RECENT_RESULTS = 12
FINISH_MARGIN_SEC = 180
# MCP discovery must never exhaust the task time before the model can work.
MCP_INIT_TOTAL_SEC = 120
MCP_INIT_ATTEMPT_SEC = 30
MCP_INIT_WAITS = (1, 3)  # short retries for transient startup failures

# A tool-heavy solver can spend its entire window exploring input drawings
# without writing the graded deliverable. Prompt progress milestones while
# leaving the task deadline, tools, and engineering correctness unchanged.
OUTPUT_CHECKPOINT_STEPS = (8, 16, 24, 32)
MAX_IMAGE_BYTES = 3 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MAX_IMAGE_EDGE = 8192
MAX_TOTAL_IMAGE_BYTES = 12 * 1024 * 1024


def jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
  """Find JPEG SOF dimensions without importing an image decoder.

  Rejects files that have no supported SOF frame before the image scan.
  """
  if not data.startswith(b"\xff\xd8"):
    return None
  at = 2
  sof = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
         0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
  while at < len(data) - 1:
    if data[at] != 0xFF:
      return None
    at += 1
    while at < len(data) and data[at] == 0xFF:
      at += 1
    if at >= len(data):
      return None
    marker = data[at]
    at += 1
    if marker in (0xD8, 0xD9, 0x01) or 0xD0 <= marker <= 0xD7:
      continue
    if marker == 0xDA or at + 2 > len(data):
      return None
    size = int.from_bytes(data[at:at + 2], "big")
    if size < 2 or at + size > len(data):
      return None
    if marker in sof:
      if size < 7:
        return None
      height = int.from_bytes(data[at + 3:at + 5], "big")
      width = int.from_bytes(data[at + 5:at + 7], "big")
      return width, height
    at += size
  return None


def output_checkpoint(turns: int) -> str | None:
  """Return a generic deliverable checkpoint at sparse tool-use milestones.

  No task paths, answers, or rubric contents are hardcoded here.
  """
  if turns not in OUTPUT_CHECKPOINT_STEPS and not (
      turns > 32 and turns % 16 == 0
  ):
    return None
  return (
      f"OUTPUT CHECKPOINT after {turns} tool calls. Review the exact output "
      "file(s) and paths required by the task. Before further exploratory "
      "analysis, create or materially update a valid task-specific "
      "deliverable at those paths and use verify_file to check structure. "
      "If the task has drawings, prioritize information needed for a first "
      "working deliverable instead of long custom OCR experiments. "
      "Continue refining after a concrete output exists. Never invent "
      "engineering facts or claim that structural validation proves "
      "engineering correctness."
  )
RETRY_WAITS = (5, 15, 30, 60, 120)
# How endpoints word an error saying the conversation no longer fits (xAI's:
# "maximum prompt length is N but the request contains M tokens"). OpenAI's
# error for too many tools ("'tools': array too long") does not match.
CONTEXT_FULL = re.compile(
    r"context.?(length|window|size|limit)|maximum context|too many"
    r" tokens|prompt is too long|maximum prompt length"
    r"|(input|prompt|request|messages?)\b[^.]{0,40}(too long|exceed)"
    r"|exceed[^.]{0,40}(token|context)",
    re.I,
)
# The context window and the prompt's share of it, as vLLM, OpenAI, Anthropic,
# Gemini and xAI word them in such an error.
WINDOW_RE = re.compile(
    r"maximum context length (?:is |of )?(\d+)"
    r"|context (?:window|length|size)(?: limit)?(?: is| of)? (\d+)"
    r"|tokens? > (\d+) maximum|maximum number of tokens allowed \((\d+)\)"
    r"|maximum prompt length is (\d+)",
    re.I,
)
PROMPT_RE = re.compile(
    r"(\d+) input tokens|(\d+) in the messages|prompt is too long: (\d+)"
    r"|input token count \((\d+)\)|request contains (\d+) tokens",
    re.I,
)
MIN_OUTPUT = 1024  # the fewest output tokens to ask for
# characters an unread tool result keeps, however full the context
UNREAD_KEEP = 600
# tokens left free in the window, since tokenizers count a little differently
CONTEXT_MARGIN = 1024
# How endpoints word an error refusing a parameter or value outright, unlike one
# that names a smaller maximum.
UNSUPPORTED = re.compile(
    r"unsupported|unrecogni[sz]ed|unknown|not supported|does not support|not"
    r" (allowed|permitted)"
    r"|extra (inputs|fields)|unexpected",
    re.I,
)
# The standard fields of a streamed tool call, of its function, and of a reply's
# delta. Any other field an endpoint sends goes back with the message: Gemini
# refuses a tool call without its thought signature, which comes in the call's
# extra_content.
CALL_FIELDS = {"index", "id", "type", "function"}
FUNCTION_FIELDS = {"name", "arguments"}
DELTA_FIELDS = {
    "role",
    "content",
    "tool_calls",
    "function_call",
    "refusal",
    "audio",
    "annotations",
    "reasoning_content",
    "reasoning",
}

client = openai.AsyncOpenAI(
    base_url=os.environ.get("MODEL_BASE_URL"),
    api_key=os.environ.get("MODEL_API_KEY") or "none",
    max_retries=0,
    timeout=900,
)

SYSTEM = """You are an expert engineer working on a task in a Linux container. Your tools act inside it: bash (it \
starts in {workdir}), read_file, write_file and edit_file. The container has no internet access.{mcp}

How to work:
- Read the whole task, then look at the files you were given before choosing an approach.
- For PNG/JPEG engineering drawings, call view_image to inspect the actual image visually before coding OCR from scratch. If labels are small, use simple task-container cropping or enlargement with available tools and call view_image again. Never invent unreadable dimensions.
- Work in small, verified steps. Compute numbers with code rather than by hand, and check units.
- Before optimizing, enumerate every hard safety and feasibility constraint from the supplied task; keep an explicit pass/fail checklist.
- Independently recompute numerical claims using a second method or tool when feasible; validate dimensions, sign conventions, and units.
- For quantitative limits in the task instruction, use suggest_constraints as an INCOMPLETE reminder, then check_constraint with exact source text and independently chosen input values. Verify every input value against its original task files. A PASS checks only the submitted claim, never proves formula choice, input provenance, complete compliance or engineering safety.
- If task-specific MCP tools exist for structured artifacts, use them when appropriate rather than inventing file formats.
- Prioritize producing valid, constraint-compliant deliverables before optional optimization; do not claim successful checks without running them.
- When the task says how the result is checked (a time limit, a format, a test), check it that way yourself.
- Write each deliverable exactly where, under the name and in the format the task asks for. Write it early, then \
improve it.
- Produce office and binary formats (docx, xlsx, pdf, ...) with code, such as python3 and the libraries in the \
container, so they are real files of that format.
- Before you finish, call verify_file on each requested output path, including outputs created by bash or MCP. It checks file structure, not numerical correctness. Fix missing/corrupt files before the final answer.
- You have about {minutes} minutes in total; the time left is shown after the results of your tools.

When the work is done, answer without calling a tool: a short summary of what you delivered (the path of each file) \
and the key results. That answer is graded together with your files."""

MCP_DIRECT = """

The task also offers its own tools (the functions named <server>__<tool>)."""

MCP_CATALOG = """

The task also offers its own tools, listed below by server. Call mcp_find to see a tool's parameters, then mcp_call \
to run it.
{catalog}"""

SUMMARY_PROMPT = """The conversation is getting long, so the earlier turns will be removed. Write notes that let you \
continue without them: what the task requires, what you found in the files, every file you created or changed (with \
its path and state), the numbers and decisions so far, and what remains to do. Be specific."""


def tool(
    name: str, description: str, required: list[str], **properties: Any
) -> dict[str, Any]:
  """Builds an OpenAI function-tool definition dictionary.

  Args:
    name: Tool function name.
    description: Human-readable tool description.
    required: List of required parameter names.
    **properties: Parameter JSON Schema definitions keyed by parameter name.

  Returns:
    An OpenAI function-tool specification dictionary.
  """
  return {
      "type": "function",
      "function": {
          "name": name,
          "description": description,
          "parameters": {
              "type": "object",
              "properties": properties,
              "required": required,
          },
      },
  }


TOOLS = [
    tool(
        "bash",
        "Run a bash command in the task container and return its exit code and"
        " output. Each call starts a new shell in the working directory: only"
        " files persist between calls.",
        ["command"],
        command={"type": "string"},
        timeout_sec={
            "type": "integer",
            "description": (
                "Stop the command after this many seconds (default 120)."
            ),
        },
    ),
    tool(
        "read_file",
        "Read a text file from the task container.",
        ["path"],
        path={"type": "string"},
    ),
    tool(
        "view_image",
        "READ-ONLY VISUAL INPUT: show a PNG or JPEG drawing/image from the "
        "task container to the vision-capable AI model. Use this tool to "
        "inspect plan sheets, diagrams and labels directly BEFORE attempting "
        "custom OCR or text-only pixel analysis. Returns a short status and "
        "adds the image to the next model turn. Limit 4 MiB per image. "
        "Inspect several sheets in separate calls if necessary.",
        ["path"],
        path={"type": "string"},
    ),
    tool(
        "suggest_constraints",
        "List possible numeric constraints in the CURRENT task instruction "
        "using a simple keyword heuristic. Candidates may be incomplete, "
        "ambiguous or irrelevant; read the actual source before deciding. "
        "Does not access attachments or graders.",
        [],
    ),
    tool(
        "check_constraint",
        "Verify ONE model-proposed engineering numeric claim: confirm a "
        "verbatim requirement quote occurs in the CURRENT task instruction, "
        "recompute a restricted formula with dimensional units, and compare "
        "with a numeric limit present in the quote. PASS verifies ONLY this "
        "model-supplied formula and inputs; it DOES NOT establish actual "
        "engineering correctness, completeness, input provenance, "
        "comparand meaning, or correctness of the chosen formula.",
        ["quote", "limit_value", "limit_unit", "comparator", "expression", "variables"],
        quote={
            "type": "string",
            "description": "A verbatim passage from the evaluator task instruction (8-400 chars).",
        },
        limit_value={
            "type": "string",
            "description": "Numeric threshold token in the quote, e.g. 250 or 0.5.",
        },
        limit_unit={
            "type": "string",
            "description": "Explicit unit symbol such as MPa, mm, kN, A, kW or 1.",
        },
        comparator={
            "type": "string", "enum": ["<=", "<", ">=", ">", "=="],
            "description": "Direction chosen from the instruction (not independently verified).",
        },
        expression={
            "type": "string",
            "description": "Safe arithmetic expression using variables, pi, sqrt, abs, min or max.",
        },
        variables={
            "type": "object",
            "description": "Mapping: each name has {value: numeric string, unit: symbol}.",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "value": {"type": "string"},
                    "unit": {"type": "string"},
                },
                "required": ["value", "unit"],
            },
        },
        reported_value={
            "type": "string",
            "description": "Optional value you wrote to a deliverable (in limit_unit) for recomputation.",
        },
        rel_tol={
            "type": "string",
            "description": "Optional relative tolerance, 0..0.01. Default 0.000001.",
        },
    ),
    tool(
        "verify_file",
        "READ-ONLY preflight for a task deliverable file or directory. Checks "
        "existence and non-empty output, and structural validity for JSON, "
        "PDF, CSV, ZIP, DOCX, XLSX, PPTX, Python, XML and IFC. It does NOT "
        "check engineering correctness. Use it on each requested output "
        "before you finish, including files created through bash or MCP.",
        ["path"],
        path={"type": "string"},
        file_format={
            "type": "string",
            "description": "File type to validate, or auto to use its suffix.",
            "enum": sorted(file_validation.SUPPORTED),
        },
    ),
    tool(
        "write_file",
        "Create or overwrite a file in the task container; parent directories"
        " are created.",
        ["path", "content"],
        path={"type": "string"},
        content={"type": "string"},
        append={
            "type": "boolean",
            "description": (
                "Add to the end of the file instead (write a large file in"
                " parts)."
            ),
        },
    ),
    tool(
        "edit_file",
        "Replace exact text in a file. old_text must occur exactly once unless"
        " replace_all is true.",
        ["path", "old_text", "new_text"],
        path={"type": "string"},
        old_text={"type": "string"},
        new_text={"type": "string"},
        replace_all={"type": "boolean"},
    ),
]
MCP_TOOLS = [
    tool(
        "mcp_find",
        "Show the description and parameters of the task's own tools that match"
        " a query.",
        ["query"],
        query={
            "type": "string",
            "description": "A tool name, or words describing what you need.",
        },
    ),
    tool(
        "mcp_call",
        "Run one of the task's own tools.",
        ["tool", "arguments"],
        tool={"type": "string", "description": "The tool's name as listed."},
        arguments={"type": "object"},
    ),
]


def clip(text: str, limit: int = TOOL_RESULT_CHARS) -> str:
  """Truncates `text` in the middle if it exceeds `limit` characters.

  Args:
    text: String to clip.
    limit: Maximum character length before truncation.

  Returns:
    The original string if within `limit`, or a middle-truncated string.
  """
  if len(text) <= limit:
    return text
  head = limit * 2 // 3
  return (
      f"{text[:head]}\n[... {len(text) - limit} characters cut"
      f" ...]\n{text[-(limit - head):]}"
  )


def parse_arguments(call: dict[str, Any]) -> dict[str, Any] | str:
  """Parses a tool call's arguments into a dict or returns an error string.

  Rewrites `call["arguments"]` as canonical JSON (`"{}"` when invalid) because
  endpoints reject conversations containing malformed tool-call arguments.

  Args:
    call: Tool-call dictionary with an `arguments` JSON string field.

  Returns:
    The parsed arguments dictionary, or an error string for the model.
  """

  def reject(constant):
    raise ValueError(f"{constant} is not valid JSON")

  try:
    args = json.loads(call["arguments"] or "{}", parse_constant=reject)
    if not isinstance(args, dict):
      raise ValueError("the arguments must be a JSON object")
    call["arguments"] = (
        json.dumps(args, ensure_ascii=False)
        .encode("utf-8", "replace")
        .decode("utf-8")
    )
    return args
  except ValueError as exc:
    call["arguments"] = "{}"
    return (
        f"error: the arguments are not a valid JSON object ({exc}); send the"
        " call again"
    )


def unique_name(name: str, taken: Any) -> str:
  """Truncates `name` to 64 chars and appends `_2`, `_3`, ... if in `taken`.

  Args:
    name: Desired function name.
    taken: Collection or mapping of already-used function names.

  Returns:
    A unique function name of at most 64 characters.
  """
  candidate, n = name[:64], 1
  while candidate in taken:
    n += 1
    candidate = f"{name[:63 - len(str(n))]}_{n}"
  return candidate


def visible(text: str) -> str:
  """Strips `<think>...</think>` reasoning blocks from `text`.

  Args:
    text: Assistant response content string.

  Returns:
    The trimmed response text with inline `<think>` blocks removed.
  """
  return re.sub(r"<think>.*?(</think>|$)", "", text or "", flags=re.S).strip()


def extra_fields(part: Any, known: set[str]) -> dict[str, Any]:
  """Extracts non-null extra fields from a streamed SDK object `part`.

  The OpenAI SDK stores undeclared provider fields in `part.model_extra`.

  Args:
    part: Streamed Pydantic model object (delta, tool call, or function).
    known: Set of standard field names to exclude.

  Returns:
    Dictionary of extra fields not in `known` whose values are not `None`.
  """
  fields = getattr(part, "model_extra", None) or {}
  return {k: v for k, v in fields.items() if k not in known and v is not None}


def merged(old: Any, new: Any, key: str = "") -> Any:
  """Merges a single delta's extra field value `new` into `old`.

  Merges dictionaries key-by-key, concatenates lists, and replaces scalars.
  Delegates `thinking_blocks` lists to `thinking_blocks()`.

  Args:
    old: Accumulated field value so far.
    new: Incoming delta field value.
    key: Field name being merged.

  Returns:
    The merged field value.
  """
  if key == "thinking_blocks" and isinstance(new, list):
    return thinking_blocks(old if isinstance(old, list) else [], new)
  if isinstance(old, dict) and isinstance(new, dict):
    return {**old, **{k: merged(old.get(k), v, k) for k, v in new.items()}}
  if isinstance(old, list) and isinstance(new, list):
    return old + new
  return new


def thinking_blocks(
    blocks: list[dict[str, Any]], new: list[Any]
) -> list[dict[str, Any]]:
  """Accumulates streamed Claude `thinking_blocks` deltas.

  Joins consecutive deltas into the current block until a `signature` closes it;
  redacted blocks (containing `data`) arrive whole and are kept separate.

  Args:
    blocks: Existing list of accumulated thinking blocks.
    new: New list of thinking block deltas from the stream chunk.

  Returns:
    Updated list of thinking block dictionaries.
  """
  out = [dict(b) for b in blocks]
  for b in new:
    if not isinstance(b, dict):
      continue
    last = out[-1] if out else None
    if (
        last is None
        or last.get("signature")
        or last.get("data")
        or b.get("data")
        or last.get("type") != b.get("type")
    ):
      out.append(dict(b))
      continue
    last["thinking"] = (last.get("thinking") or "") + (b.get("thinking") or "")
    if b.get("signature"):
      last["signature"] = b["signature"]
  return out


def signed(block: Any) -> bool:
  """Returns whether `block` has a `signature` or redacted `data` field.

  Args:
    block: Candidate thinking block dictionary.

  Returns:
    `True` if Claude will accept `block` in a subsequent turn, else `False`.
  """
  return isinstance(block, dict) and bool(
      block.get("signature") or block.get("data")
  )


def drop_thinking(
    message: dict[str, Any], unless: Any = lambda block: False
) -> dict[str, Any]:
  """Filters `thinking_blocks` in `message`, keeping blocks matching `unless`.

  Args:
    message: Assistant message dictionary to modify in place.
    unless: Predicate `unless(block)` returning `True` for blocks to keep.

  Returns:
    The modified `message` dictionary.
  """
  if isinstance(message.get("thinking_blocks"), list):
    message["thinking_blocks"] = [
        b for b in message["thinking_blocks"] if unless(b)
    ]
  fields = message.get("provider_specific_fields")
  if isinstance(fields, dict) and isinstance(
      fields.get("thinking_blocks"), list
  ):
    message["provider_specific_fields"] = {
        **fields,
        "thinking_blocks": [b for b in fields["thinking_blocks"] if unless(b)],
    }
  return message


def marked(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
  """Returns the messages a Chat Completions call sends.

  For Claude, this is a copy whose system prompt and newest message carry
  `cache_control` (2 of Claude's 4 cache breakpoints), so every turn reads the
  conversation so far from the cache instead of paying for all of it again.
  The stored conversation stays as it is.

  Args:
    messages: The conversation so far.

  Returns:
    `messages` itself for other models, or the marked copy for Claude.
  """
  if not CACHE_MARK:
    return messages
  out = list(messages)
  for i in {0, len(out) - 1}:
    m, content = out[i], out[i].get("content")
    if isinstance(content, str) and content:
      out[i] = {
          **m,
          "content": [{
              "type": "text",
              "text": content,
              "cache_control": {"type": "ephemeral"},
          }],
      }
    elif isinstance(content, list):
      # A Claude cache breakpoint belongs on a content block, never the
      # multimodal message. Preserve original image and text parts by copy.
      copied = [dict(part) for part in content]
      for part in reversed(copied):
        if part.get("type") == "text":
          part["cache_control"] = {"type": "ephemeral"}
          break
      out[i] = {**m, "content": copied}
    else:
      out[i] = {**m, "cache_control": {"type": "ephemeral"}}
  return out


def response_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
  """Converts Chat Completions messages into Responses API input items.

  Assistant turns produced by the Responses API are replayed using their saved
  output items (including encrypted reasoning). Each function call takes its id
  and name from the turn's stored tool call, and its arguments as
  `parse_arguments` rewrote them.

  Args:
    messages: Conversation messages in Chat Completions format.

  Returns:
    The `input` list for a Responses API call.
  """
  items = []
  for m in messages:
    if m["role"] == "tool":
      items.append({
          "type": "function_call_output",
          "call_id": m["tool_call_id"],
          "output": m["content"],
      })
    elif m["role"] != "assistant":
      content = m["content"]
      if isinstance(content, list):
        # Responses API uses input_image/image_url instead of Chat
        # Completions image_url blocks. Keep unmodified text content.
        content = [
            {
                "type": "input_image",
                "image_url": part["image_url"]["url"],
                "detail": part["image_url"].get("detail", "auto"),
            }
            if part.get("type") == "image_url" else
            {"type": "input_text", "text": part["text"]}
            if part.get("type") == "text" else part
            for part in content
        ]
      items.append({"role": m["role"], "content": content})
    elif "output_items" in m:
      calls = iter(m.get("tool_calls") or [])
      for item in m["output_items"]:
        call = None
        if item.get("type") == "function_call":
          call = next(calls, None)
        if call:
          item = {
              **item,
              "call_id": call["id"],
              "name": call["function"]["name"],
              "arguments": call["function"]["arguments"],
          }
        items.append(item)
    else:
      if m.get("content"):
        items.append({"role": "assistant", "content": m["content"]})
      items += [
          {
              "type": "function_call",
              "call_id": c["id"],
              "name": c["function"]["name"],
              "arguments": c["function"]["arguments"],
          }
          for c in m.get("tool_calls") or []
      ]
  return items


def chat_usage(usage: Any) -> Any:
  """Returns Responses API `usage` in the shape `record` and `report` read.

  Args:
    usage: A Responses API usage object, or None.

  Returns:
    An object with `prompt_tokens`, `completion_tokens` and
    `prompt_tokens_details.cached_tokens`, as Chat Completions reports them, or
    None.
  """
  if usage is None:
    return None
  details = getattr(usage, "input_tokens_details", None)
  return types.SimpleNamespace(
      prompt_tokens=usage.input_tokens or 0,
      completion_tokens=usage.output_tokens or 0,
      prompt_tokens_details=types.SimpleNamespace(
          cached_tokens=getattr(details, "cached_tokens", 0) or 0
      ),
  )


class MCPServer(taskbridge.MCPClient):
  """Client wrapper for a single task MCP server and its discovered tools."""

  def __init__(self, session: taskbridge.Session, spec: dict[str, Any]):
    super().__init__(session, spec["url"], client="custom-harness")
    self.name = spec["name"]
    self.tools: list[dict[str, Any]] = []

  async def start(self) -> None:
    """Initializes the MCP session and fetches the server's tool list."""
    await self.initialize()
    catalog = await self.list_tools()
    self.tools = [
        tool for tool in catalog
        if isinstance(tool, dict)
        and isinstance(tool.get("name"), str)
        and bool(tool["name"])
    ]

  async def call(self, name: str, arguments: dict[str, Any]) -> str:
    """Calls tool `name` on the MCP server and formats the result as text.

    Args:
      name: Unprefixed MCP tool name on this server.
      arguments: Tool input arguments dictionary.

    Returns:
      Formatted text output (prefixed with `error: ` if `isError` is set).
    """
    result = await self.call_tool(name, arguments)
    parts = []
    content = (
        result.get("content") if isinstance(result.get("content"), list) else []
    )
    for item in content:
      kind = item.get("type") if isinstance(item, dict) else None
      if kind == "text":
        parts.append(str(item.get("text", "")))
      elif kind == "resource":
        parts.append(json.dumps(item.get("resource"))[:TOOL_RESULT_CHARS])
      elif kind == "resource_link":  # its uri, name and mimeType
        parts.append(json.dumps(item)[:TOOL_RESULT_CHARS])
      else:
        parts.append(f"[{kind or 'unknown'} content not shown]")
    if not parts and result.get("structuredContent") is not None:
      parts.append(json.dumps(result["structuredContent"]))
    text = "\n".join(parts) or "(no content)"
    return f"error: {text}" if result.get("isError") else text


class Reply:
  """Accumulates a streamed chat-completion response and its tool calls."""

  def __init__(self):
    self.text, self.reasoning, self.calls, self.usage, self.finish = (
        "",
        "",
        {},
        None,
        None,
    )
    self.extra = {}  # fields beyond DELTA_FIELDS, sent back with the message
    self._by_index, self._by_id, self._last = {}, {}, None

  def add_call(self, part: Any) -> None:
    """Adds one streamed tool-call delta to the accumulated calls.

    Groups deltas by `index`, or by `id` when an endpoint omits or reuses
    `index`. Deltas with neither continue the most recent call unless
    `_another_call` detects a new call boundary. Preserves provider-specific
    extra fields on both the call and its `function`.

    Args:
      part: Streamed tool-call delta object from a completion chunk.
    """
    index, call_id = (
        getattr(part, "index", None),
        getattr(part, "id", None) or "",
    )
    function = getattr(part, "function", None)
    name = (getattr(function, "name", None) or "") if function else ""
    key = self._by_index.get(index) if index is not None else None
    if (
        key is not None
        and call_id
        and self.calls[key]["id"] not in ("", call_id)
    ):
      key = None
    if (
        key is not None
        and not call_id
        and name
        and self._another_call(self.calls[key], name)
    ):
      key = None
    if key is None and call_id:
      key = self._by_id.get(call_id)
    if (
        key is None
        and index is None
        and not call_id
        and self._last is not None
        and not (name and self._another_call(self.calls[self._last], name))
    ):
      key = self._last
    if key is None:
      key = len(self.calls)
      self.calls[key] = {
          "id": "",
          "name": "",
          "arguments": "",
          "extra": {},
          "function_extra": {},
      }
    if index is not None:
      self._by_index[index] = key
    if call_id:
      self._by_id[call_id] = key
    call = self.calls[key]
    call["id"] = call["id"] or call_id
    call["name"] = call["name"] or name
    call["arguments"] += (
        (getattr(function, "arguments", None) or "") if function else ""
    )
    call["extra"] = merged(call["extra"], extra_fields(part, CALL_FIELDS))
    if function:
      call["function_extra"] = merged(
          call["function_extra"], extra_fields(function, FUNCTION_FIELDS)
      )
    self._last = key

  @staticmethod
  def _another_call(call: dict[str, Any], name: str) -> bool:
    """Checks whether a nameless-ID delta for `name` starts a new tool call.

    Returns `True` when `call` already has a function name and either `name`
    differs or `call["arguments"]` is already a complete JSON object.

    Args:
      call: Currently accumulating tool-call dictionary.
      name: Function name on the incoming delta.

    Returns:
      `True` if the delta begins a separate tool call, else `False`.
    """
    if not call["name"]:
      return False
    if name != call["name"]:
      return True
    try:
      return isinstance(json.loads(call["arguments"]), dict)
    except ValueError:
      return False

  def message(self) -> dict[str, Any]:
    """Builds the assistant message dict for the conversation history.

    Preserves provider-specific fields while stripping unsigned thinking blocks
    that Claude would reject.

    Returns:
      The formatted assistant message dictionary.
    """
    message = {**self.extra, "role": "assistant", "content": self.text or None}
    if self.calls:
      message["tool_calls"] = [
          {
              **c["extra"],
              "id": c["id"],
              "type": "function",
              "function": {
                  **c["function_extra"],
                  "name": c["name"],
                  "arguments": c["arguments"],
              },
          }
          for c in self.calls.values()
      ]
    if self.reasoning:
      message["reasoning_content"] = self.reasoning
    return drop_thinking(message, unless=signed)


class OutOfTime(Exception):  # pylint: disable=g-bad-exception-name
  """Raised when retrying a failed model call would exceed the time budget."""


class CutOff(asyncio.TimeoutError):  # pylint: disable=g-bad-exception-name
  """Raised when a streamed reply is cut off by its per-call deadline."""


class StreamBadRequest(openai.OpenAIError):  # pylint: disable=g-bad-exception-name
  """Raised when an SSE stream reports a non-retryable bad request error.

  OpenAI streams report errors via `type` and `code` strings (such as
  `invalid_request_error` or `context_length_exceeded`) instead of an HTTP 400
  status. Treating them as 400-equivalent allows `remedy()` to adjust the
  request rather than retrying it unchanged.
  """


# `type` or `code` values of an in-stream error that mark a bad request
BAD_REQUEST = frozenset({
    "invalid_request_error",
    "context_length_exceeded",
    "invalid_value",
    "invalid_prompt",
})


def stream_error(error: Any) -> openai.OpenAIError:
  """Classifies an in-stream error object as a bad request or a truncated stream.

  Args:
    error: Error object or dict reported inside an SSE stream.

  Returns:
    `StreamBadRequest` if the error type, code, or message indicates an invalid
    request or context overflow; otherwise `StreamCut` so the call is retried.
  """
  def field(name: str) -> Any:
    if isinstance(error, dict):
      return error.get(name)
    return getattr(error, name, None)

  message = str(field("message") or "")
  named = {field("type"), field("code")} & BAD_REQUEST
  if named or CONTEXT_FULL.search(message):
    return StreamBadRequest(message)
  return StreamCut(f"the stream reported an error: {message}")


class StreamCut(openai.OpenAIError):  # pylint: disable=g-bad-exception-name
  """Raised when an SSE stream ends before reporting a finish reason.

  Indicates a dropped connection, proxy timeout, or transient in-stream error.
  The partial response is discarded and the call is retried.
  """


class Agent:
  """Manages the conversation, tool execution, and context compaction loop."""

  def __init__(self, session: taskbridge.Session):
    self.session = session
    self.tools = list(TOOLS)
    self.mcp: dict[str, tuple[MCPServer, dict[str, Any]]] = {}
    self.messages: list[dict[str, Any]] = []
    self.prompt_tokens = 0
    # Each call asks for output_limit(): output_ceiling (MAX_OUTPUT_TOKENS,
    # lowered by shrink_output when the endpoint refuses that many), or less
    # when the context window has less room beside the prompt.
    self.output_ceiling = int(os.environ.get("MAX_OUTPUT_TOKENS", "32000"))
    self.context_window: int | None = None  # learned from a context-full error
    self.max_output = self.output_ceiling  # the limit the last call asked for
    # max_tokens once an endpoint refuses this one
    self.token_limit = "max_completion_tokens"
    # empty once an endpoint refuses it
    self.effort = {"reasoning_effort": EFFORT}
    # whether calls go to the Responses API rather than Chat Completions
    self.responses = MODEL.lower().startswith(RESPONSES_MODELS)
    self.finish_margin = FINISH_MARGIN_SEC
    self.tool_turns = 0
    self.pending_images: list[dict[str, Any]] = []
    self.seen_images: set[str] = set()
    self.image_bytes = 0

  async def run(self) -> str:
    """Runs the agent tool-calling loop until the task is complete.

    Returns:
      The final summary message produced by the agent.
    """
    # Set the summary reserve before MCP initialization, so discovery cannot
    # consume the time the agent needs to produce a final answer.
    left = self.session.time_left()
    self.finish_margin = (
        FINISH_MARGIN_SEC
        if left is None
        else min(FINISH_MARGIN_SEC, max(left, 0) / 4)
    )
    mcp = await self.connect_mcp()
    left = self.session.time_left()
    self.messages = [
        {
            "role": "system",
            "content": SYSTEM.format(
                workdir=self.session.workdir,
                mcp=mcp,
                minutes=int(left / 60) if left else "unlimited",
            ),
        },
        {"role": "user", "content": self.session.instruction},
    ]
    # head: the text of an answer cut off at the output limit
    nudges, head = 0, ""
    while True:
      left = self.session.time_left()
      if left is not None and left < self.finish_margin:
        return await self.finish("Time is almost up.")
      try:
        reply = await self.complete(head=head)
      except OutOfTime:
        return await self.finish(
            "Time is almost up, and the model is not answering."
        )
      # what it streamed is the answer if the summary call fails too
      except CutOff:
        return await self.finish("Time is almost up.")
      except openai.OpenAIError as exc:
        # Preserve deliverables already written even if the model endpoint
        # ultimately rejects or exhausts retries for a request.
        await self.session.log(f"model call failed: {type(exc).__name__}")
        summary = await self.finish("The model endpoint failed.")
        return summary or (
            "Model endpoint failed after partial execution; inspect existing "
            "task deliverables and logs for any completed work."
        )
      arguments = [parse_arguments(call) for call in reply.calls.values()]
      if reply.text or reply.calls:
        # A turn with neither (its output went to hidden reasoning, or a content
        # filter stopped it) is not sent back: endpoints refuse an assistant
        # message with null content and no tool calls.
        self.messages.append(reply.message())
      text = visible(head + reply.text)
      if text:
        self.session.final_message = text
      if not reply.calls:
        if visible(reply.text) and reply.finish == "length":
          # cut off at the output limit: ask for the rest, and reset the
          # empty-turn count
          nudges, head = 0, head + reply.text
          self.messages.append({
              "role": "user",
              "content": (
                  f"Your reply was cut off at the {self.max_output}-token"
                  " output limit. Continue where it stopped, and put long"
                  " content in files with write_file rather than in a reply."
              ),
          })
          continue
        if visible(reply.text) or nudges >= 2:
          return text or self.session.final_message
        nudges += 1  # only consecutive empty turns end the run
        self.messages.append({
            "role": "user",
            "content": (
                "Continue the task; when it is done, answer with "
                "your summary of the deliverables."
            ),
        })
        continue
      nudges, head = 0, ""
      for call, args in zip(reply.calls.values(), arguments):
        result = (
            await self.run_tool(call["name"], args)
            if isinstance(args, dict)
            else args
        )
        if reply.finish == "length" and isinstance(args, str):
          result += (
              f" Your reply hit the {self.max_output}-token output limit; write"
              " large files in parts."
          )
        self.messages.append(
            {"role": "tool", "tool_call_id": call["id"], "content": result}
        )
      left = self.session.time_left()
      if left is not None:
        self.messages[-1]["content"] += f"\n[{int(left // 60)} minutes left]"
      # Attach image in the next user turn AFTER replying to the tool call.
      # This ensures each assistant function_call still has its tool result.
      if self.pending_images:
        self.messages.extend(self.pending_images)
        self.pending_images.clear()
      previous_turns = self.tool_turns
      self.tool_turns += len(reply.calls)
      # Handle batched function calls that cross a checkpoint in one reply.
      for milestone in range(previous_turns + 1, self.tool_turns + 1):
        note = output_checkpoint(milestone)
        if note:
          self.messages.append({"role": "user", "content": note})
      if self.prompt_tokens > self.compact_at():
        await self.compact()

  async def finish(self, why: str) -> str:
    """Prompts the model for a final summary without tools when time is up.

    Args:
      why: Reason string prepended to the wrap-up prompt.

    Returns:
      The final summary text, falling back to `self.session.final_message`.
    """
    self.drop_images()
    self.messages.append({
        "role": "user",
        "content": (
            f"{why} Stop working now and answer with your final "
            "summary: what you delivered, where, and the key results."
        ),
    })
    left = self.session.time_left()
    # the summary call gets taskbridge.time_share(left, 30) seconds, and at
    # least 5
    reserve = (
        0 if left is None else left - max(5, taskbridge.time_share(left, 30))
    )
    try:
      text = visible(
          (
              await self.complete(tool_choice="none", reserve=reserve, head="")
          ).text
      )
    except (CutOff, openai.OpenAIError, OutOfTime):
      text = ""
    return text or self.session.final_message

  async def connect_mcp(self) -> str:
    """Connects to the task's MCP servers and registers their tools.

    Exposes a small set of MCP tools directly as OpenAI functions; for larger
    tool sets, lists their names in the system prompt and exposes `mcp_find`
    and `mcp_call`.

    Returns:
      The system-prompt section describing the available MCP tools.
    """
    left = self.session.time_left()
    # Cap startup to 10% of the remaining task and keep a work window and
    # the final-summary reserve. Untimed sessions use the fixed ceiling.
    startup_budget = (
        MCP_INIT_TOTAL_SEC
        if left is None
        else min(
            MCP_INIT_TOTAL_SEC,
            max(0.0, left * 0.10),
            max(0.0, left - self.finish_margin - min(30.0, left / 4)),
        )
    )
    startup_deadline = time.monotonic() + startup_budget
    unavailable = []
    # Fairly allocate discovery time so one hung app server cannot starve
    # all the other task integrations.
    count = len(self.session.mcp_servers)
    for index, spec in enumerate(self.session.mcp_servers):
      if not isinstance(spec, dict):
        unavailable.append("(invalid MCP specification)")
        await self.session.log("MCP server skipped: specification is not a dict")
        continue
      server_deadline = time.monotonic() + (
          max(0.0, startup_deadline - time.monotonic()) / max(1, count - index)
      )
      label = str(spec.get("name") or "(unnamed)")[:80]
      if (
          not isinstance(spec.get("name"), str)
          or not spec.get("name")
          or not isinstance(spec.get("url"), str)
          or not spec.get("url")
      ):
        unavailable.append(label)
        await self.session.log("MCP server skipped: invalid name/url")
        continue
      server = None
      last_error = "startup budget exhausted"
      for attempt, wait in enumerate((0, *MCP_INIT_WAITS)):
        remaining = min(startup_deadline, server_deadline) - time.monotonic()
        if remaining <= 0:
          break
        if wait:
          await asyncio.sleep(min(wait, max(0, remaining)))
        remaining = min(startup_deadline, server_deadline) - time.monotonic()
        if remaining <= 0:
          break
        left_now = self.session.time_left()
        if left_now is not None:
          remaining = min(
              remaining,
              max(0.0, left_now - self.finish_margin - min(30.0, left_now / 4)),
          )
        budget = min(MCP_INIT_ATTEMPT_SEC, remaining)
        if budget <= 0:
          break
        # Reconstruct on each attempt. A half-initialized client may have
        # an invalid MCP session ID and cannot be retried safely as-is.
        try:
          candidate = MCPServer(self.session, spec)
          previous_timeout = candidate.timeout_sec
          candidate.timeout_sec = max(1, min(previous_timeout, int(budget)))
          await asyncio.wait_for(candidate.start(), timeout=budget)
          candidate.timeout_sec = previous_timeout
          server = candidate
          break
        except (
            taskbridge.TaskError,
            taskbridge.MCPError,
            RuntimeError,
            KeyError,
            TypeError,
            ValueError,
            AttributeError,
            asyncio.TimeoutError,
        ) as exc:
          last_error = type(exc).__name__
          await self.session.log(
              f"MCP server {label} initialization attempt "
              f"{attempt + 1} failed: {last_error}: {str(exc)[:120]}"
          )
      if server is None:
        unavailable.append(label)
        await self.session.log(f"MCP server {label} unavailable: {last_error}")
        continue
      for t in server.tools:
        name = re.sub(r"[^A-Za-z0-9_-]", "_", f"{server.name}__{t['name']}")
        self.mcp[unique_name(name, self.mcp)] = (server, t)
      await self.session.log(
          f"MCP server {server.name}: {len(server.tools)} tools"
      )
    warning = (
        "\nWARNING: These task MCP integrations were unavailable at startup: "
        + ", ".join(unavailable)
        + ". Do not claim to have used them. Where feasible, inspect task "
        "files with bash/read_file, and report any unavailable app operation."
        if unavailable
        else ""
    )
    if not self.mcp:
      return warning
    functions = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": (t.get("description") or t["name"])[:1024],
                "parameters": (
                    {
                        k: v
                        for k, v in (t.get("inputSchema") or {}).items()
                        if k != "$schema"
                    }
                    or {"type": "object", "properties": {}}
                ),
            },
        }
        for name, (_, t) in self.mcp.items()
    ]
    if (
        len(json.dumps(functions)) <= DIRECT_MCP_CHARS
        and len(self.tools) + len(functions) <= MAX_TOOLS
    ):
      self.tools += functions
      return MCP_DIRECT + warning
    self.tools += MCP_TOOLS
    servers: dict[str, list[str]] = {}
    for name, (server, _) in self.mcp.items():
      servers.setdefault(server.name, []).append(name)
    return MCP_CATALOG.format(
        catalog="\n".join(
            f"{s}: {', '.join(names)}" for s, names in servers.items()
        )
    ) + warning

  def find_mcp(self, query: str) -> str:
    """Finds MCP tools matching `query` and formats their schemas.

    Returns as many matching tool schemas as fit within `TOOL_RESULT_CHARS`
    (always at least one, without truncating any individual schema).

    Args:
      query: Tool name or search keywords.

    Returns:
      Formatted descriptions and JSON parameter schemas of matching tools.
    """
    words = [w for w in re.split(r"[^a-z0-9]+", query.lower()) if w]

    def score(item):
      name, (_, t) = item
      text = f"{name} {t.get('description') or ''}".lower()
      return (
          (query == name) * 100
          + sum(w in name.lower() for w in words) * 3
          + sum(w in text for w in words)
      )

    best = [
        item
        for item in sorted(self.mcp.items(), key=score, reverse=True)[:8]
        if score(item) > 0
    ]
    if not best:
      return (
          f"no tool matches {query!r}; the names are listed in the system"
          " prompt"
      )
    found = [
        f"{name}: {(t.get('description') or '')[:400]}\nparameters:"
        f" {json.dumps(t.get('inputSchema') or {})}"
        for name, (_, t) in best
    ]
    while len(found) > 1 and len("\n\n".join(found)) > TOOL_RESULT_CHARS:
      found.pop()
    return "\n\n".join(found)

  async def complete(
      self,
      tool_choice: str = "auto",
      reserve: float | None = None,
      head: str | None = None,
  ) -> Reply:
    """Streams one model completion with retries and context-limit recovery.

    Stops streaming `reserve` seconds before the task deadline. Automatically
    adapts token limit parameters, shrinks output ceilings, or compacts the
    conversation when the endpoint reports a context-length error.

    Args:
      tool_choice: OpenAI `tool_choice` value (`"auto"` or `"none"`).
      reserve: Seconds before the task deadline at which to cut off the call.
      head: Prefix text from a prior length-truncated reply, if continuing.

    Returns:
      The completed `Reply` object.

    Raises:
      OutOfTime: If a transient error occurs and there is not enough time left
        to wait and retry.
      CutOff: If the response is still streaming when `reserve` is reached.
      openai.APIError: If a non-retryable API error occurs or retries are
        exhausted.
      StreamCut: If every stream, retries included, ends without a finish
        reason.
    """
    reserve = self.finish_margin if reserve is None else reserve
    waits = iter(RETRY_WAITS)
    while True:
      try:
        reply = await self.stream(self.messages, tool_choice, reserve, head)
        if reply.finish is None:
          raise StreamCut("the stream ended without a finish reason")
        await self.record(reply)
        return reply
      except openai.APIStatusError as exc:
        if exc.status_code == 400 and await self.remedy(str(exc)):
          continue
        if not (exc.status_code in (408, 409, 429) or exc.status_code >= 500):
          raise
        failure = exc
      except StreamBadRequest as exc:
        if await self.remedy(str(exc)):
          continue
        raise
      # a connection error, a timeout, or a stream that errored or ended early
      except (openai.APIError, StreamCut) as exc:
        failure = exc
      wait = next(waits, None)
      if wait is None:
        raise failure
      left = self.session.time_left()
      if left is not None and left - wait < reserve:
        await self.session.log(
            f"model call failed ({str(failure)[:300]}); no time left to wait"
            f" {wait}s"
        )
        raise OutOfTime(str(failure)[:300]) from failure
      await self.session.log(
          f"model call failed ({str(failure)[:300]}); again in {wait}s"
      )
      await asyncio.sleep(wait)

  async def remedy(self, error: str) -> bool:
    """Adjusts request parameters for the next call after a bad request.

    Args:
      error: The error message returned by a 400 response or stream failure.

    Returns:
      True if the request parameters were adjusted and the call can be retried.
    """
    if CONTEXT_FULL.search(error):
      # The conversation does not fit. Try in this order:
      # 1. fewer output tokens, when the error shows that alone makes room
      #    (restored as the prompt shrinks);
      # 2. compact() what the model has already read;
      # 3. fewer output tokens anyway, when the error did not give the
      #    prompt's size;
      # 4. cut_unread(), the tool results the model has not read yet.
      sent, measured = self.max_output, self.learn_context(error)
      if (
          measured
          and self.room() >= MIN_OUTPUT
          and self.output_limit() < sent
      ):
        return True
      return bool(
          await self.compact(force=True)
          or (not measured and self.output_limit() < sent)
          or self.cut_unread()
      )
    return (
        self.use_responses(error)
        or self.use_max_tokens(error)
        or self.drop_effort(error)
        or self.shrink_output(error)
        or self.drop_images()
    )

  def drop_images(self) -> bool:
    """Remove unsupported image blocks after an endpoint rejects a request.

    Without cleanup a permanent 400 would recur during the final summary.
    Inform the model that text-only inspection remains available.
    """
    removed = False
    for i, message in enumerate(self.messages):
      parts = message.get("content")
      if not isinstance(parts, list):
        continue
      if not any(part.get("type") in ("image_url", "input_image")
                 for part in parts if isinstance(part, dict)):
        continue
      text = " ".join(
          str(part.get("text", "")) for part in parts
          if isinstance(part, dict) and part.get("type") == "text"
      )
      self.messages[i] = {
          **message, "content": (
              text + "\n[Image unavailable to this model endpoint; use "
              "task tools to inspect the underlying file instead.]"
          ),
      }
      removed = True
    if removed:
      self.pending_images.clear()
      self.seen_images.clear()
      self.image_bytes = 0
      self.tools = [
          t for t in self.tools
          if t.get("function", {}).get("name") != "view_image"
      ]
    return removed

  def use_responses(self, error: str) -> bool:
    """Switches to the Responses API if a Chat Completions 400 points there.

    Args:
      error: Error message string from a 400 `APIStatusError`.

    Returns:
      `True` if calls now go to the Responses API.
    """
    if self.responses or not RESPONSES_ONLY.search(error):
      return False
    self.responses = True
    return True

  def use_max_tokens(self, error: str) -> bool:
    """Switches to `max_tokens` if `error` rejects `max_completion_tokens`.

    Args:
      error: Error message string from a 400 `APIStatusError`.

    Returns:
      `True` if the token limit parameter was switched to `max_tokens`.
    """
    if (
        self.token_limit != "max_completion_tokens"
        or "max_completion_tokens" not in error
        or not UNSUPPORTED.search(error)
    ):
      return False
    self.token_limit = "max_tokens"
    return True

  def drop_effort(self, error: str) -> bool:
    """Stops sending `reasoning_effort` once an endpoint refuses it.

    A model without reasoning, or a server that does not know the parameter,
    answers `400` naming it (`reasoning.effort` on the Responses API).

    Args:
      error: Error message string from a 400 `APIStatusError`.

    Returns:
      `True` if `reasoning_effort` was being sent and is now dropped.
    """
    if not self.effort or not re.search(r"reasoning[._]effort", error):
      return False
    self.effort = {}
    return True

  def shrink_output(self, error: str) -> bool:
    """Lowers `output_ceiling` when `error` indicates a smaller output limit.

    Args:
      error: Error message string from a 400 `APIStatusError`.

    Returns:
      `True` if `output_ceiling` was reduced, else `False`.
    """
    if UNSUPPORTED.search(error):
      return False
    limit = re.search(r"(\d+)\s*\(exclusive\)", error)
    stated = re.search(
        r"at most (\d+)|(\d+), which is the maximum|maximum (?:allowed"
        r" )?(?:number of )?(?:output|"
        r"completion) tokens (?:is |of )?(\d+)",
        error,
        re.I,
    )
    if limit:
      new = int(limit.group(1)) - 1
    elif stated:
      new = int(next(g for g in stated.groups() if g))
    elif re.search(
        r"max_?(completion_)?tokens|maxOutputTokens|output tokens", error, re.I
    ):
      new = self.max_output // 2
    else:
      return False
    if new >= self.max_output or new < MIN_OUTPUT:
      return False
    self.output_ceiling = new
    return True

  def learn_context(self, error: str) -> bool:
    """Extracts context window and prompt token counts from `error`.

    Args:
      error: Error message string from a context-length `APIStatusError`.

    Returns:
      `True` if both `context_window` and `prompt_tokens` are now known.
    """
    window, prompt = WINDOW_RE.search(error), PROMPT_RE.search(error)
    if window:
      self.context_window = int(next(g for g in window.groups() if g))
    if prompt:
      self.prompt_tokens = int(next(g for g in prompt.groups() if g))
    return bool(prompt and self.context_window)

  def room(self) -> int | None:
    """Returns remaining output token capacity in the known context window.

    Returns:
      Available tokens beside the prompt, or `None` if the window size is
      unknown.
    """
    return (
        self.context_window - self.prompt_tokens - CONTEXT_MARGIN
        if self.context_window
        else None
    )

  def output_limit(self) -> int:
    """Computes the max output tokens to request on the next completion call.

    Returns:
      `output_ceiling`, clamped down to `room()` when known (never below
      `MIN_OUTPUT`).
    """
    room = self.room()
    return (
        self.output_ceiling
        if room is None
        else max(MIN_OUTPUT, min(self.output_ceiling, room))
    )

  def compact_at(self) -> int:
    """Returns the prompt token threshold at which to trigger compaction.

    Returns:
      `COMPACT_AT_TOKENS`, or a lower threshold when the known context window
      requires more room for output.
    """
    if not self.context_window:
      return COMPACT_AT_TOKENS
    room = min(self.output_ceiling, self.context_window // 2)
    return max(
        self.context_window // 4,
        min(COMPACT_AT_TOKENS, self.context_window - room - CONTEXT_MARGIN),
    )

  def estimate_prompt_tokens(self) -> int:
    """Estimates prompt token count at four JSON characters per token.

    Skips `output_items`, which repeat a Responses API turn's text and calls
    plus encrypted reasoning whose length does not track its token count.

    Returns:
      Estimated number of tokens in `self.messages`.
    """
    total = 0
    for m in self.messages:
      parts = m.get("content")
      if isinstance(parts, list):
        header = {k: v for k, v in m.items()
                  if k not in ("content", "output_items")}
        total += len(json.dumps(header)) // 4
        for part in parts:
          if (isinstance(part, dict)
              and part.get("type") in ("image_url", "input_image")):
            total += 1500  # image tokens depend on tiles, not base64 length
          else:
            total += len(json.dumps(part)) // 4
      else:
        total += len(json.dumps(
            {k: v for k, v in m.items() if k != "output_items"
            })) // 4
    return total

  async def stream(
      self,
      messages: list[dict[str, Any]],
      tool_choice: str,
      reserve: float,
      head: str | None = None,
  ) -> Reply:
    """Streams a single chat completion call up to `reserve` seconds before end.

    If `head` is provided and the stream is cut off or cancelled, updates
    `self.session.final_message` with `visible(head + reply.text)`.

    Args:
      messages: Conversation message list to send.
      tool_choice: OpenAI `tool_choice` setting (`"auto"` or `"none"`).
      reserve: Seconds to hold back before the session deadline.
      head: Prefix text from an earlier truncated response, or `None`.

    Returns:
      The accumulated `Reply` instance.

    Raises:
      CutOff: If the stream times out at `left - reserve`.
      asyncio.CancelledError: If the task is cancelled during streaming.
    """
    left = self.session.time_left()
    self.max_output = self.output_limit()
    reply = Reply()

    async def read():
      if self.responses:
        await self.respond(reply, messages, tool_choice)
        return
      stream = await client.chat.completions.create(
          model=MODEL,
          messages=marked(messages),
          tools=self.tools,
          tool_choice=tool_choice,
          stream=True,
          stream_options={"include_usage": True},
          **self.effort,
          **{self.token_limit: self.max_output},
      )
      async for chunk in stream:
        reply.usage = chunk.usage or reply.usage
        if not chunk.choices:
          continue
        delta = chunk.choices[0].delta
        reply.finish = chunk.choices[0].finish_reason or reply.finish
        reply.text += delta.content or ""
        reply.reasoning += (
            getattr(delta, "reasoning_content", None)
            or getattr(delta, "reasoning", None)
            or ""
        )
        reply.extra = merged(reply.extra, extra_fields(delta, DELTA_FIELDS))
        for part in delta.tool_calls or []:
          reply.add_call(part)

    try:
      await asyncio.wait_for(read(), None if left is None else left - reserve)
    except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
      if head is not None:
        self.session.final_message = (
            visible(head + reply.text) or self.session.final_message
        )
      if isinstance(exc, asyncio.CancelledError):
        raise
      raise CutOff() from None
    except openai.APIError as exc:
      # the SDK raises an error sent inside a chat stream as an APIError
      said = {
          "message": getattr(exc, "message", None) or str(exc),
          "type": getattr(exc, "type", None),
          "code": getattr(exc, "code", None),
      }
      if isinstance(exc, openai.APIStatusError) or isinstance(
          stream_error(said), StreamCut
      ):
        raise
      raise StreamBadRequest(said["message"]) from exc
    for n, call in reply.calls.items():
      call["id"] = call["id"] or f"call_{n}_{len(self.messages)}"
      # some open models leak chat-template tokens into it
      call["name"] = call["name"].split("<|")[0].strip()
    return reply

  async def respond(
      self, reply: Reply, messages: list[dict[str, Any]], tool_choice: str
  ) -> None:
    """Streams a single Responses API call into `reply`.

    Keeps the conversation in Chat Completions format and converts it on each
    call via `response_input`. Because `store=False`, each response's output
    items (including encrypted reasoning) are saved and echoed back on
    subsequent turns. If the stream ends before `response.completed` or
    `response.incomplete`, `reply.finish` remains `None` so the call is retried.

    Args:
      reply: The `Reply` to populate as events arrive.
      messages: Conversation message list to send.
      tool_choice: `"auto"` or `"none"`.

    Raises:
      StreamCut: If the stream reports a transient error.
      StreamBadRequest: If the stream reports a non-retryable bad request.
    """
    effort = self.effort.get("reasoning_effort")
    stream = await client.responses.create(
        model=MODEL,
        input=response_input(messages),
        tools=[
            {"type": "function", **t["function"], "strict": False}
            for t in self.tools
        ],
        tool_choice=tool_choice,
        stream=True,
        store=False,
        include=["reasoning.encrypted_content"],
        max_output_tokens=self.max_output,
        **({"reasoning": {"effort": effort}} if effort else {}),
    )
    async for event in stream:
      kind = getattr(event, "type", "")
      if kind == "response.output_text.delta":
        reply.text += event.delta or ""
      elif kind in ("error", "response.failed"):
        raise stream_error(
            getattr(getattr(event, "response", None), "error", None) or event
        )
      elif kind in ("response.completed", "response.incomplete"):
        done = event.response
        items = [i.model_dump(exclude_none=True) for i in done.output or []]
        reply.extra = {"output_items": items}
        reply.text = "".join(
            part.get("text", "")
            for item in items
            if item.get("type") == "message"
            for part in item.get("content") or []
            if part.get("type") == "output_text"
        )
        for item in items:
          if item.get("type") == "function_call":
            reply.calls[len(reply.calls)] = {
                "id": item.get("call_id") or "",
                "name": item.get("name") or "",
                "arguments": item.get("arguments") or "",
                "extra": {},
                "function_extra": {},
            }
        reply.usage = chat_usage(done.usage)
        reason = getattr(done.incomplete_details, "reason", None)
        if kind == "response.completed":
          reply.finish = "tool_calls" if reply.calls else "stop"
        else:
          reply.finish = reason if reason == "content_filter" else "length"

  async def record(self, reply: Reply) -> None:
    """Updates `prompt_tokens` from `reply` and reports token usage.

    Args:
      reply: Completed `Reply` from `stream()`.
    """
    if reply.usage is None:
      self.prompt_tokens = self.estimate_prompt_tokens()
      return
    self.prompt_tokens = reply.usage.prompt_tokens or 0
    await self.report(reply)

  async def report(self, reply: Reply) -> None:
    """Reports token usage from `reply` to the evaluator session if present.

    Args:
      reply: Completed `Reply` whose `usage` should be reported.
    """
    usage = reply.usage
    if usage is None:
      return
    cached = (
        getattr(
            getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0
        )
        or 0
    )
    await self.session.usage(
        MODEL, usage.prompt_tokens or 0, usage.completion_tokens or 0, cached
    )

  async def run_tool(self, name: str, args: dict[str, Any]) -> str:
    """Dispatches a tool call by `name` and returns its clipped text output.

    Args:
      name: Tool function name (`bash`, `read_file`, `write_file`, `edit_file`,
        `mcp_find`, `mcp_call`, or a direct MCP tool name).
      args: Parsed tool arguments dictionary.

    Returns:
      The tool output or error string.
    """
    # Never log arguments: a tool's parameters could contain private/hidden
    # task instructions, source quotations, or document contents.
    permitted = {
        "bash", "read_file", "view_image", "write_file", "edit_file", "verify_file",
        "suggest_constraints", "check_constraint", "mcp_find", "mcp_call",
    }
    safe_tool_name = (
        name if name in permitted
        else ("mcp_tool" if name in self.mcp else "unknown_tool")
    )
    await self.session.log(
        f"tool call: {safe_tool_name}; argument count: {len(args)}"
    )
    try:
      if name == "suggest_constraints":
        candidates = constraint_check.numeric_constraint_candidates(
            self.session.instruction
        )
        return json.dumps({
            "candidate_quotes": candidates,
            "note": "Heuristic, incomplete candidates only; confirm source meaning."
        }, ensure_ascii=False)
      if name == "check_constraint":
        required = (
            "quote", "limit_value", "limit_unit", "comparator",
            "expression", "variables",
        )
        missing = [key for key in required if key not in args]
        if missing:
          return "error: check_constraint missing required fields: " + ", ".join(missing)
        vars_ = args["variables"]
        if isinstance(vars_, str):
          try:
            vars_ = json.loads(vars_)
          except ValueError:
            vars_ = None
        outcome = constraint_check.check_constraint(
            self.session.instruction,
            quote=args["quote"],
            limit_value=args["limit_value"],
            limit_unit=args["limit_unit"],
            comparator=args["comparator"],
            expression=args["expression"],
            variables=vars_,
            reported_value=args.get("reported_value"),
            rel_tol=args.get("rel_tol"),
        )
        return json.dumps(outcome, ensure_ascii=False)
      if name == "mcp_call":
        name, args = str(args.get("tool", "")), args.get("arguments") or {}
        args = json.loads(args) if isinstance(args, str) else args
        if name not in self.mcp:
          return f"error: no tool named {name!r}; use mcp_find"
      if name in self.mcp:
        server, t = self.mcp[name]
        # A single MCP request must not consume the time reserved for the
        # final summary. The upstream MCP client otherwise waits up to 300s.
        left = self.session.time_left()
        budget = 300.0 if left is None else min(300.0, left - self.finish_margin)
        if budget <= 0:
          return "error: task deadline approaching; reserve time for final summary"
        previous_timeout = server.timeout_sec
        try:
          # Bound the RPC's timeout as well as the total await, so that slow
          # MCP servers cannot overrun the final-summary reserve.
          server.timeout_sec = max(1, min(previous_timeout, int(budget)))
          return clip(
              await asyncio.wait_for(server.call(t["name"], args), timeout=budget)
          )
        except asyncio.TimeoutError:
          return (
              "error: MCP tool call timed out before the final-summary "
              "deadline. The remote tool MAY already have applied changes; "
              "inspect the resulting state before retrying to avoid duplicates."
          )
        except taskbridge.TaskError as exc:
          # The evaluator may have timed out an RPC before wait_for fired.
          # Some remote operations might still have side effects on failure.
          return (
              f"error: MCP RPC failed: {exc}. The remote tool MAY have applied "
              "changes; inspect its state before retrying."
          )
        finally:
          server.timeout_sec = previous_timeout
      if name == "mcp_find":
        return self.find_mcp(str(args.get("query", "")))
      handler = {
          "bash": self.bash,
          "read_file": self.read_file,
          "view_image": self.view_image,
          "verify_file": self.verify_file,
          "write_file": self.write_file,
          "edit_file": self.edit_file,
      }.get(name)
      return (
          clip(await handler(**args))
          if handler
          else f"error: there is no tool named {name}"
      )
    except taskbridge.TaskError as exc:
      return f"error: {exc}"
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as exc:
      return f"error: {type(exc).__name__}: {exc}"

  def path(self, path: str) -> str:
    """Resolves `path` relative to `self.session.workdir` if not absolute.

    Args:
      path: Absolute or workdir-relative file path.

    Returns:
      The resolved container path string.
    """
    return (
        path
        if path.startswith("/")
        else posixpath.join(self.session.workdir, path)
    )

  async def bash(self, command: str, timeout_sec: int = 120) -> str:
    """Executes `command` with bash in the task container.

    Args:
      command: Shell command string to run.
      timeout_sec: Timeout in seconds, cut to end before the time kept for the
        final summary.

    Returns:
      Formatted exit code and command output string.
    """
    cap = int(self.session.limits.get("exec_timeout_sec") or 600)
    left = self.session.time_left()
    if left is not None:
      cap = min(cap, max(1, int(left - self.finish_margin)))
    result = await self.session.exec(
        command, taskbridge.positive_int(timeout_sec, 120, cap)
    )
    note = " (timed out)" if result.get("timed_out") else ""
    truncated_note = (
        "\n[WARNING: the evaluator truncated command output. "
        "Inspect targeted ranges before relying on omitted content.]"
        if result.get("truncated")
        else ""
    )
    return (
        "exit code"
        f" {result['exit_code']}{note}\n{result['output'] or '(no output)'}"
        f"{truncated_note}"
    )

  async def read_file(self, path: str) -> str:
    """Reads a UTF-8 text file from `path` in the task container.

    Args:
      path: File path inside the task container.

    Returns:
      Decoded file contents, or a notice if the file is binary or empty.
    """
    data = await self.session.read_file(self.path(path))
    if b"\0" in data[:8192]:
      return (
          f"{path} is a binary file ({len(data)} bytes); inspect it with bash"
          " (python3, unzip -l, ...)"
      )
    return data.decode("utf-8", "replace") or "(empty file)"

  async def view_image(self, path: str) -> str:
    """Attach one checked task-local image to the next model request.

    Uses the evaluator's existing read_file RPC: no HTTP egress and no host
    filesystem access. Refuses oversized, invalid or non-image bytes.
    """
    left = self.session.time_left()
    if left is not None and left <= self.finish_margin + 20:
      return "error: too close to deadline for visual inspection"
    data = await self.session.read_file(self.path(path))
    if not 0 < len(data) <= MAX_IMAGE_BYTES:
      return f"error: image must be 1 to {MAX_IMAGE_BYTES} bytes"
    mime = None
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
      if (
          len(data) < 45
          or data[8:16] != b"\x00\x00\x00\x0dIHDR"
          or not data.endswith(b"\x00\x00\x00\x00IEND\xaeB\x60\x82")
      ):
        return "error: incomplete PNG image structure"
      width, height = struct.unpack(">II", data[16:24])
      if (width == 0 or height == 0 or
          width > MAX_IMAGE_EDGE or height > MAX_IMAGE_EDGE or
          width * height > MAX_IMAGE_PIXELS):
        return "error: invalid or too-large PNG dimensions"
      mime = "image/png"
    elif data.startswith(b"\xff\xd8\xff") and b"\xff\xd9" in data[-64:]:
      dimensions = jpeg_dimensions(data)
      if dimensions is None:
        return "error: missing or malformed JPEG dimensions"
      width, height = dimensions
      if (width == 0 or height == 0 or
          width > MAX_IMAGE_EDGE or height > MAX_IMAGE_EDGE or
          width * height > MAX_IMAGE_PIXELS):
        return "error: invalid or too-large JPEG dimensions"
      mime = "image/jpeg"
    if mime is None:
      return "error: not a PNG or JPEG file; convert using bash if needed"
    digest = hashlib.sha256(data).hexdigest()
    if digest in self.seen_images:
      return f"already attached image from {path}; review the earlier vision turn"
    if self.image_bytes + len(data) > MAX_TOTAL_IMAGE_BYTES:
      return "error: task image attachment budget exceeded"
    self.seen_images.add(digest)
    self.image_bytes += len(data)
    # Source path describes the user's supplied task file, not an external URL.
    # The model request uses the provider's native multimodal data URI.
    self.pending_images.append({
        "role": "user",
        "content": [
            {"type": "text", "text": (
                f"Visually inspect the task drawing at {path!r}. "
                "Read spatial layout, annotations, and measurements directly. "
                "If small text is not legible, say so rather than inventing it."
            )},
            {"type": "image_url", "image_url": {
                "url": "data:" + mime + ";base64,"
                       + base64.b64encode(data).decode("ascii"),
                "detail": "high",
            }},
        ],
    })
    return (
        f"image attached for the next model turn: {path} "
        f"({mime}, {len(data)} bytes); interpret it visually"
    )

  async def verify_file(self, path: str, file_format: str = "auto") -> str:
    """Read-only structural check for a deliverable in the task container.

    This does not assert engineering or safety compliance. It only detects
    absent files, empty outputs, and malformed common artifact formats.
    """
    remaining = self.session.time_left()
    if remaining is not None and remaining <= self.finish_margin + 2:
      return "error: too close to deadline for a file verification"
    timeout = (
        25 if remaining is None
        else max(1, min(25, int(remaining - self.finish_margin - 1)))
    )
    response = await self.session.exec(
        file_validation.probe_command(self.path(path), file_format or "auto"),
        timeout,
    )
    if response.get("timed_out") or response.get("truncated"):
      return "error: verification timed out or returned incomplete output"
    if response.get("exit_code") != 0:
      return (
          f"error: verification tool failed for {path}: "
          f"{str(response.get('output') or '')[:400]}"
      )
    try:
      # The evaluator merges stdout/stderr. Ignore incidental warnings and
      # parse only the final JSON result emitted by the isolated probe.
      lines = (response.get("output") or "").splitlines()
      outcome = json.loads(next(
          line for line in reversed(lines) if line.strip().startswith("{")
      ))
    except (ValueError, StopIteration):
      return f"error: verification produced invalid JSON for {path}"
    if outcome.get("ok"):
      return (
          f"PASS: {path}; {outcome.get('size_bytes', 'directory')} bytes; "
          f"check={outcome.get('check')} (not an engineering correctness check)"
      )
    return f"FAIL: {path}; {outcome.get('issue') or 'no valid output'}"

  async def write_file(
      self, path: str, content: str, append: bool = False
  ) -> str:
    """Writes or appends `content` to `path` in the task container.

    Args:
      path: Destination file path inside the task container.
      content: Text content to write.
      append: Whether to append to the file instead of overwriting.

    Returns:
      Confirmation string with the resulting file size in bytes.
    """
    size = await self.session.write_file(
        self.path(path), content, append=bool(append)
    )
    return f"{path} now holds {size} bytes"

  async def edit_file(
      self, path: str, old_text: str, new_text: str, replace_all: bool = False
  ) -> str:
    """Replaces `old_text` with `new_text` in `path` inside the task container.

    Args:
      path: Target file path inside the task container.
      old_text: Substring to search for.
      new_text: Replacement string.
      replace_all: Whether to replace all occurrences instead of requiring one.

    Returns:
      Confirmation string or an error message if `old_text` is missing or
      ambiguous.
    """
    text = (await self.session.read_file(self.path(path))).decode("utf-8")
    count = text.count(old_text) if old_text else 0
    if count == 0:
      return f"error: old_text was not found in {path}"
    if count > 1 and not replace_all:
      return (
          f"error: old_text occurs {count} times in {path}; add context to make"
          " it unique, or set replace_all"
      )
    await self.session.write_file(
        self.path(path),
        text.replace(old_text, new_text, -1 if replace_all else 1),
    )
    return f"replaced {count if replace_all else 1} occurrence(s) in {path}"

  def unread(self) -> list[dict[str, Any]]:
    """Returns tool result messages appended after the last assistant turn.

    Returns:
      List of unread `"tool"` role message dictionaries.
    """
    last = max(
        (n for n, m in enumerate(self.messages) if m["role"] == "assistant"),
        default=0,
    )
    return [m for m in self.messages[last:] if m["role"] == "tool"]

  async def compact(self, force: bool = False) -> bool:
    """Compacts conversation history without modifying unread tool results.

    First truncates older read tool outputs (keeping `KEEP_RECENT_RESULTS`
    unless `force` is `True`). If that is insufficient, summarizes older turns
    via a model call. Claude's thinking blocks stay on the kept turns: a block's
    signature does not depend on the turns before it, and Claude refuses a tool
    loop whose newest turn lost its block.

    Args:
      force: Whether compaction was triggered by a context-full error.

    Returns:
      `True` if any tool results were truncated or turns were summarized.
    """
    unread = {id(m) for m in self.unread()}
    results = [
        (n, m) for n, m in enumerate(self.messages) if m["role"] == "tool"
    ]
    # with `force`, the endpoint's own count of the prompt wins when it is
    # larger: the estimate can run low
    cuts, size = [], max(
        self.estimate_prompt_tokens(), self.prompt_tokens if force else 0
    )
    for n, m in results if force else results[:-KEEP_RECENT_RESULTS]:
      if force and self.context_window and size <= self.compact_at():
        break
      if len(m["content"]) > 600 and id(m) not in unread:
        size -= (len(m["content"]) - 400) // 4
        m["content"] = (
            m["content"][:400]
            + "\n[... removed to save context; run the tool again if needed"
            " ...]"
        )
        cuts.append(n)
    if cuts:
      await self.session.log(f"context: cut {len(cuts)} old tool results")
      self.prompt_tokens = self.estimate_prompt_tokens()
      if (
          not force
          or not self.context_window
          or self.prompt_tokens <= self.compact_at()
      ):
        return True
    cut = len(self.messages) - 6
    # the kept turns must not start with tool results whose call was replaced by
    # the notes
    while cut > 3 and self.messages[cut]["role"] == "tool":
      cut -= 1
    if cut < 10:  # too few turns to be worth a notes call
      return bool(cuts)
    try:
      notes = await self.stream(
          self.messages[:cut] + [{"role": "user", "content": SUMMARY_PROMPT}],
          "none",
          self.finish_margin,
      )
    except (openai.OpenAIError, asyncio.TimeoutError) as exc:
      await self.session.log(
          f"context: notes failed ({str(exc)[:200] or 'out of time'})"
      )
      return bool(cuts)
    await self.report(notes)
    if notes.finish is None:
      await self.session.log(
          "context: the notes were cut short, so the turns are kept"
      )
      return bool(cuts)
    text = visible(notes.text)
    if not text:
      await self.session.log(
          "context: the notes came back empty, so the turns are kept"
      )
      return bool(cuts)
    if notes.finish == "length":
      text += "\n[the notes were cut off at the output limit]"
    self.messages = (
        self.messages[:2]
        + [{
            "role": "user",
            "content": (
                "Your notes on the work so far (earlier "
                "turns were removed to save context):\n"
                + text
            ),
        }]
        + self.messages[cut:]
    )
    await self.session.log(f"context: replaced {cut - 2} messages with notes")
    self.prompt_tokens = self.estimate_prompt_tokens()
    return True

  def cut_unread(self) -> bool:
    """Truncates unread tool results as a last resort when context is full.

    Clips each unread tool output exceeding `2 * UNREAD_KEEP` to an equal share
    of the remaining context window (or halves it when the window size is
    unknown).

    Returns:
      `True` if at least one unread tool result was shortened, else `False`.
    """
    unread = [m for m in self.unread() if len(m["content"]) > 2 * UNREAD_KEEP]
    if not unread:
      return False
    share = None
    if self.context_window:
      rest = (
          self.estimate_prompt_tokens()
          - sum(len(m["content"]) for m in unread) // 4
      )
      share = (self.compact_at() - rest) * 4 // len(unread)
    for m in unread:
      size = len(m["content"])
      # a share within 100 characters of the size would barely shrink it
      keep = size // 2 if share is None or share >= size - 100 else share
      m["content"] = clip(m["content"], max(UNREAD_KEEP, keep))
    self.prompt_tokens = self.estimate_prompt_tokens()
    return True


async def solve(session: taskbridge.Session) -> str:
  """Runs `Agent` on `session` and returns the final answer.

  Args:
    session: Active task session.

  Returns:
    The final summary string returned by the agent.
  """
  print(
      f"task: workdir {session.workdir}, model {MODEL},"
      f" {len(session.mcp_servers)} MCP server(s)",
      file=sys.stderr,
  )
  return await Agent(session).run()


if __name__ == "__main__":
  taskbridge.run(solve)
