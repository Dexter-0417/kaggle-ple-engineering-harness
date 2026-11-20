"""Offline, fake-file tests for the output-format verification tool."""
import asyncio
import json
import pathlib
import subprocess
import tempfile
import unittest
import zipfile
from unittest.mock import AsyncMock

from file_validation import probe_command
from serve import Agent


def probe(path, kind="auto"):
    result = subprocess.run(
        probe_command(str(path), kind),
        capture_output=True, text=True, shell=True,
        timeout=5,
        check=True,
    )
    return json.loads(result.stdout)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory(prefix="ple-output-check-")
        self.addCleanup(self.dir.cleanup)
        self.root = pathlib.Path(self.dir.name)

    def write(self, name, data):
        p = self.root / name
        p.write_bytes(data)
        return p

    def create_office(self, name, member):
        p = self.root / name
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("[Content_Types].xml", "<Types/>")
            z.writestr(member, "<root><test>OK</test></root>")
        return p

    def test_isolated_python_ignores_json_module_shadowing(self):
        # Task workdirs may contain user-created json.py; the verifier must
        # import only trusted stdlib modules instead of executing that file.
        fake = self.root / "json.py"
        fake.write_text('raise AssertionError("injected local module")')
        target = self.write("output.json", b'{"value":2}')
        cmd = probe_command(str(target))
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            cwd=self.root, timeout=5,
        )
        self.assertEqual(r.returncode, 0)
        self.assertTrue(json.loads(r.stdout)["ok"])

    def test_large_csv_field_does_not_false_fail(self):
        p = self.write("wide.csv", b"header\n" + b"x" * 150000 + b"\n")
        self.assertTrue(probe(p)["ok"])

    def test_plain_text_exists(self):
        self.assertTrue(probe(self.write("result.txt", b"some text"))["ok"])

    def test_missing_output_fails(self):
        x = probe(self.root / "missing.pdf")
        self.assertFalse(x["ok"])
        self.assertIn("not found", x["issue"])

    def test_empty_output_fails(self):
        x = probe(self.write("empty.md", b""))
        self.assertFalse(x["ok"])

    def test_valid_json_parses(self):
        x = probe(self.write("result.json", b'{"score": 1}'))
        self.assertTrue(x["ok"])
        self.assertEqual(x["check"], "json_parse")

    def test_invalid_json_fails(self):
        x = probe(self.write("result.json", b'{"score": }'))
        self.assertFalse(x["ok"])

    def test_jsonl_samples_records(self):
        x = probe(self.write("items.jsonl", b'{"a":1}\n{"a":2}\n'))
        self.assertEqual(x["sampled_rows"], 2)
        self.assertTrue(x["ok"])

    def test_csv_samples_rows(self):
        x = probe(self.write("items.csv", b"a,b\n1,2\n"))
        self.assertTrue(x["ok"])
        self.assertEqual(x["sampled_rows"], 2)

    def test_valid_pdf_header_and_trailer(self):
        x = probe(self.write("out.pdf", b"%PDF-1.7\nexample\n%%EOF\n"))
        self.assertTrue(x["ok"])

    def test_bad_pdf_fails(self):
        x = probe(self.write("out.pdf", b"hello"))
        self.assertFalse(x["ok"])

    def test_valid_zip_structure(self):
        p = self.root / "result.zip"
        with zipfile.ZipFile(p, "w") as f:
            f.writestr("item.txt", "good")
        x = probe(p)
        self.assertTrue(x["ok"])
        self.assertIn("archive_", x["check"])

    def test_bad_zip_fails(self):
        self.assertFalse(probe(self.write("data.zip", b"invalid zip"))["ok"])

    def test_docx_xml_integrity(self):
        x = probe(self.create_office("report.docx", "word/document.xml"))
        self.assertTrue(x["ok"])

    def test_xlsx_xml_integrity(self):
        x = probe(self.create_office("book.xlsx", "xl/workbook.xml"))
        self.assertTrue(x["ok"])

    def test_pptx_xml_integrity(self):
        x = probe(self.create_office("slides.pptx", "ppt/presentation.xml"))
        self.assertTrue(x["ok"])

    def test_broken_office_file_rejected(self):
        p = self.create_office("missing.docx", "other/file.xml")
        self.assertFalse(probe(p)["ok"])

    def test_python_valid_and_invalid(self):
        self.assertTrue(probe(self.write("good.py", b"x=1\n"))["ok"])
        self.assertFalse(probe(self.write("bad.py", b"def f(:\n"))["ok"])

    def test_ifc_header(self):
        self.assertTrue(probe(self.write("building.ifc", b"ISO-10303-21;\n"))["ok"])
        self.assertFalse(probe(self.write("broken.ifc", b"invalid"))["ok"])

    def test_nonempty_output_directory(self):
        p = self.root / "outputs"
        p.mkdir()
        self.write("outputs/out.txt", b"OK")
        self.assertTrue(probe(p)["ok"])

    def test_empty_directory_fails(self):
        p = self.root / "emptydir"
        p.mkdir()
        self.assertFalse(probe(p)["ok"])

    def test_symlinks_fail(self):
        target = self.write("target.txt", b"some data")
        link = self.root / "link.txt"
        link.symlink_to(target)
        self.assertFalse(probe(link)["ok"])

    def test_shell_metacharacters_cannot_execute(self):
        tricky = self.write("my;echo-hello$(touch BAD)", b"safe")
        self.assertTrue(probe(tricky)["ok"])
        self.assertFalse((self.root / "BAD").exists())

    def test_probe_rejects_invalid_format_or_control_characters(self):
        with self.assertRaises(ValueError):
            probe_command("/app/out.txt", "unsupported")
        with self.assertRaises(ValueError):
            probe_command("/app/out\nevil.txt")


class FakeSession:
    workdir = "/"
    limits = {"exec_timeout_sec": 600}

    def __init__(self, left=900):
        self.left = left
        self.count = 0

    def time_left(self):
        return self.left

    async def log(self, text):
        return None

    async def exec(self, command, timeout_sec=120):
        self.count += 1
        result = subprocess.run(
            command, shell=True, capture_output=True,
            text=True, timeout=timeout_sec,
        )
        return {
            "exit_code": result.returncode,
            "output": result.stdout + result.stderr,
            "timed_out": False, "truncated": False,
        }


class AgentFileToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_reports_pass_via_tool_dispatch(self):
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "report.json"
            p.write_text('{"done": true}')
            fake = FakeSession()
            agent = Agent(fake)
            result = await agent.run_tool(
                "verify_file", {"path": str(p), "file_format": "auto"}
            )
            self.assertIn("PASS:", result)
            self.assertIn("json_parse", result)
            self.assertEqual(fake.count, 1)

    async def test_agent_reports_missing_file_failure(self):
        fake = FakeSession()
        agent = Agent(fake)
        result = await agent.run_tool(
            "verify_file", {"path": "/tmp/__missing_file_for_PLE__.xlsx"}
        )
        self.assertIn("FAIL:", result)
        self.assertIn("not found", result)

    async def test_verifier_handles_null_optional_format(self):
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "out.json"
            p.write_text('{"ok":true}')
            fake = FakeSession()
            result = await Agent(fake).run_tool(
                "verify_file", {"path": str(p), "file_format": None}
            )
            self.assertIn("PASS:", result)

    async def test_verifier_parses_json_after_stderr_noise(self):
        class NoisySession(FakeSession):
            async def exec(self, command, timeout_sec=120):
                result = await super().exec(command, timeout_sec)
                result["output"] = "Python warning from task environment\n" + result["output"]
                return result

        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "result.json"
            p.write_text('{"works":true}')
            result = await Agent(NoisySession()).run_tool(
                "verify_file", {"path": str(p)}
            )
            self.assertIn("PASS:", result)

    async def test_tool_avoids_deadline_overrun(self):
        fake = FakeSession(left=1)
        agent = Agent(fake)
        agent.finish_margin = 1
        result = await agent.run_tool("verify_file", {"path": "/tmp/a.json"})
        self.assertIn("too close to deadline", result)
        self.assertEqual(fake.count, 0)


if __name__ == "__main__":
    unittest.main()
