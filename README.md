# PLE Verified Engineering Harness

Entry under development for Google - Agent Harness on Physical-Engineering's Last Exam:
https://www.kaggle.com/competitions/physical-engineerings-last-exam

This is an independent fork of Google's Apache-2.0 google-research/ple sample_submissions/custom-openai reference harness with reliability improvements and offline regression tests.

## Model separation

- Development and code review on this VM: Claude Code subscription, model claude-fable-5-2. The project-local setting is .claude/settings.local.json (excluded from Git).
- Competition runtime model: google-cloud/claude-opus-5-5 in submission.yaml. Fable 5.2 is not an allowed submitted competition model.
- Claude Code's subscription login does not authenticate the Google-sponsored model proxy or pay for local Vertex model calls. Never embed subscription credentials, Hugging Face tokens, API keys or other secrets in this repository.

## Files

- serve.py: WebSocket agent harness, tool use, MCP discovery, retries, context compaction, deadlines
- taskbridge.py: Google's official PLE protocol implementation
- submission.yaml: model, installation, entrypoint
- tests/: offline reliability checks
- COMPETITION.md: Kaggle submission requirements
- LICENSE: Apache-2.0 license inherited from the reference implementation

## Current safety improvements

- MCP **discovery** is time-budgeted (maximum 10% of remaining task time, capped at 120 seconds), retries transient failures, protects other integrations from a hung server, and tells the model which integrations are unavailable.
- MCP **tool calls** respect the final-summary reserve and warn that a timed-out request may already have applied remote changes.
- Model endpoint errors trigger an attempt to produce a final answer rather than failing without deliverables.
- Truncated shell-command output is explicitly flagged so the agent does not mistake partial results for complete data.
- The read-only `verify_file` tool checks task deliverables before completion: missing/empty files, basic JSON/CSV/ZIP/Office/PDF/Python/XML/IFC structural validity. It cannot validate engineering calculations or guarantee a benchmark score.
- The file probe uses isolated Python with safely quoted paths, handles noisy evaluator output, and rejects symlinked outputs according to the competition artifact rules.

These are engineering changes; none has been shown to improve the official dev benchmark score yet.

## Reproducible offline tests

From the harness repo root, with the dependencies from requirements.txt in your Python environment:

    python -m unittest discover -s tests -p 'test_*.py' -v
    python -m py_compile serve.py taskbridge.py

From the root of the separately cloned Google PLE official kit:

    uv run python -m kit validate ../harness
    uv run python -m kit build ../harness

Actual official dev evaluation requires approved Hugging Face gated dataset access and an authenticated MODEL_PROXY_URL/MODEL_PROXY_KEY serving the declared sponsored model, potentially billed via Google Cloud. For example:

    uv run python eval/run.py ../harness --tasks data-hall-mep-routing --trials 1

Do not run this command until model proxy authorization, budget, and sufficient isolated CPU/RAM/disk are verified. Only eval/run.py may generate artifacts/ to be committed for Kaggle Writeup. Do not fabricate artifacts or hard-code dev task solutions.

## Numeric engineering constraint checks (model-audited, not ground truth)

Two additional local harness tools are available, with no paid external API
calls beyond the model request the competition already makes:

- `suggest_constraints`: an **incomplete heuristic** listing candidate numeric
  requirement phrases from the current evaluator task instruction. It is not
  a requirement extractor with guaranteed recall and does not read attachments.
- `check_constraint`: requires a verbatim quoted numeric requirement from
  the current task instruction, a number tied to its source unit, a comparator,
  a limited arithmetic expression and explicit variable values/units. It
  recomputes with safe AST operations and dimensional compatibility checks.

The result is `PASS`, `FAIL`, or `UNVERIFIED`. `PASS` only confirms the
*model-supplied* inputs/expression/bound comparison and exact quote context.
The checker cannot establish that the input values came from the correct
documents, that the mathematical model is appropriate, or that *all* task
constraints have been found. It checks basic comparator direction against
unambiguous source phrases such as "not exceed", "at least", "以下" and "以上",
but that heuristic is incomplete and does not replace human interpretation.
Unsupported units, decimal-comma or range ambiguity, ungrounded quotes,
conflicting comparator directions, or near-bound approximate calculations
produce `UNVERIFIED`, not a confident success.

Supported symbols are deliberately finite and explicit, including mm/m/ft/in,
N/kN, Pa/kPa/MPa/psi, W/kW, V/kV, A/mA, J/kWh, m2/m3, m/s and
other explicitly enumerated dimensions. Do not infer or silently convert
unsupported units, temperature offsets or decibel scales.

The constraint functions run entirely inside the harness process, do not use
`eval` or system commands, and do not need task files or verifier code.
Tool-call parameters, including hidden task source quotations, are never
persisted in evaluator logs by the harness.

Example based on synthetic numbers (NOT a benchmark task):

    Task text: "The steel stress shall not exceed 250 MPa."
    Formula:   F / (pi * d**2 / 4)
    Variables: F = 12.5 kN; d = 10 mm
    Limit:     250 MPa
    Result:    approx. 159.15 MPa => PASS for this supplied arithmetic claim

This remains unproven against Google's actual 29-task grading set.

## Free hosted CI for offline validation

This repository contains `.github/workflows/ple-harness-ci.yml`, intended
for a **public GitHub repository**. It runs 125+ synthetic/offline tests and
builds the harness with Google's official kit on an isolated hosted
`ubuntu-24.04` runner. Standard public GitHub Actions runners provide
4 CPU / 16 GB RAM at no charge. They have only ~14 GB disk and cannot be
assumed to fit the largest PLE tasks, which need substantial extra images.

This CI deliberately does NOT run `eval/run.py`, use any HF token, contact
paid model APIs, generate grading artifacts, or pretend to be a scored run.
The official PLE scoring requirement remains separate.
