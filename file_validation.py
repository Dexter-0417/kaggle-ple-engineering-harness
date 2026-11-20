# Copyright 2026.
# Licensed under the Apache License, Version 2.0.
"""Read-only file-structure probes for PLE task outputs.

Runs the probe in the evaluator task container through Session.exec, not in the
harness container. No network access, task IDs, graders, or reference answers.
"""

import shlex

SUPPORTED = frozenset(
    (
        "auto", "json", "jsonl", "csv", "tsv", "python", "pdf",
        "zip", "docx", "xlsx", "pptx", "ifc", "xml", "text",
    )
)
PROBE_PY = r'''
import ast
import csv
import json
import os
import pathlib
import sys
import zipfile
import zlib
import xml.etree.ElementTree as ET

path = pathlib.Path(sys.argv[1])
requested = sys.argv[2]
result = {"ok": False, "path": str(path), "format": requested}


def check():
    if path.is_symlink():
        raise ValueError("symlink outputs are not supported")
    if path.is_dir():
        files = 0
        for f in path.rglob("*"):
            if f.is_symlink():
                raise ValueError("output directory contains a symlink")
            if f.is_file():
                files += 1
            if files >= 5001:
                break
        if not files:
            raise ValueError("output directory is empty")
        result.update(ok=True, kind="directory", sampled_files=min(files, 5000))
        result["check"] = "directory_nonempty"
        return
    if not path.is_file():
        raise ValueError("file not found")
    size = path.stat().st_size
    if not size:
        raise ValueError("file is empty")
    result["size_bytes"] = size
    ext = path.suffix.lower()
    kind = requested if requested != "auto" else {
        ".json": "json", ".jsonl": "jsonl", ".csv": "csv",
        ".tsv": "tsv", ".py": "python", ".pdf": "pdf",
        ".zip": "zip", ".docx": "docx", ".xlsx": "xlsx",
        ".pptx": "pptx", ".ifc": "ifc", ".xml": "xml",
        ".txt": "text", ".md": "text",
    }.get(ext, "text")
    result["format"] = kind
    result["check"] = "existence_only"
    if kind in {"zip", "docx", "xlsx", "pptx"}:
        expected = {
            "docx": "word/document.xml",
            "xlsx": "xl/workbook.xml",
            "pptx": "ppt/presentation.xml",
        }.get(kind)
        with zipfile.ZipFile(path) as z:
            entries = z.namelist()
            if not entries:
                raise ValueError("archive has no members")
            if expected:
                if "[Content_Types].xml" not in entries or expected not in entries:
                    raise ValueError("missing required Office Open XML members")
                if z.getinfo(expected).file_size <= 8 * 1024 * 1024:
                    ET.fromstring(z.read(expected))
                else:
                    result["large_office_xml_not_parsed"] = True
            # Prevent compressed zip bombs from exhausting the verifier.
            uncompressed = sum(info.file_size for info in z.infolist())
            if size <= 2 * 1024 * 1024 and uncompressed <= 32 * 1024 * 1024:
                damaged = z.testzip()
                if damaged:
                    raise ValueError("damaged zip member")
                result["check"] = "archive_members_and_crc"
            else:
                result["check"] = "archive_structure_only"
            result["archive_members"] = len(entries)
    elif kind == "pdf":
        with path.open("rb") as f:
            head = f.read(1024)
            f.seek(max(0, size - 4096))
            tail = f.read()
        if not head.startswith(b"%PDF-") or b"%%EOF" not in tail:
            raise ValueError("not a complete PDF header/trailer")
        result["check"] = "pdf_header_and_trailer"
    elif kind == "json":
        if size <= 16 * 1024 * 1024:
            with path.open(encoding="utf-8") as f:
                json.load(f)
            result["check"] = "json_parse"
        else:
            result["check"] = "existence_only_large_json"
    elif kind == "jsonl":
        with path.open(encoding="utf-8") as f:
            n = 0
            for line in f:
                if line.strip():
                    json.loads(line)
                    n += 1
                if n >= 100:
                    break
        if not n:
            raise ValueError("no JSON Lines records")
        result.update(check="jsonl_sample_parse", sampled_rows=n)
    elif kind in {"csv", "tsv"}:
        csv.field_size_limit(2 ** 31 - 1)
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f, delimiter="\t" if kind == "tsv" else ",")
            samples = 0
            for row in reader:
                if row:
                    samples += 1
                if samples >= 100:
                    break
        if not samples:
            raise ValueError("no CSV/TSV rows")
        result.update(check="csv_sample_parse", sampled_rows=samples)
    elif kind == "python":
        if size <= 4 * 1024 * 1024:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            result["check"] = "python_ast_parse"
        else:
            result["check"] = "existence_only_large_python"
    elif kind == "ifc":
        with path.open("rb") as f:
            head = f.read(4096)
        if b"ISO-10303-21" not in head:
            raise ValueError("missing IFC STEP header")
        result["check"] = "ifc_header"
    elif kind == "xml":
        if size <= 16 * 1024 * 1024:
            ET.parse(path)
            result["check"] = "xml_parse"
        else:
            result["check"] = "existence_only_large_xml"
    result["ok"] = True


try:
    check()
except (OSError, ValueError, SyntaxError, json.JSONDecodeError,
        zipfile.BadZipFile, ET.ParseError, UnicodeError, RuntimeError,
        NotImplementedError, RecursionError, zlib.error, csv.Error) as exc:
    result["issue"] = type(exc).__name__ + ": " + str(exc)[:200]
print(json.dumps(result, ensure_ascii=True, separators=(",", ":")))
'''


def probe_command(path: str, file_format: str = "auto") -> str:
    """Return a single safely quoted, read-only Python process command."""
    if not isinstance(path, str) or not path or len(path) > 1024:
        raise ValueError("invalid or overlong output path")
    if any(ord(c) < 32 for c in path):
        raise ValueError("control character in output path")
    if file_format not in SUPPORTED:
        raise ValueError("unsupported file format")
    return (
        "python3 -I -W ignore -c " + shlex.quote(PROBE_PY)
        + " " + shlex.quote(path)
        + " " + shlex.quote(file_format)
    )
