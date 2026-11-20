"""In-memory evaluator protocol tests; no model server and no task dataset."""
import asyncio
import base64
import json
import pathlib
import subprocess
import tempfile
import unittest

import taskbridge
from serve import Agent


class EvaluatorSocket:
    """One mock evaluator stream speaking Google's PLE WebSocket frames."""

    def __init__(self, task=None, execute_real_commands=False):
        self.incoming = asyncio.Queue()
        self.sent = []
        self.files = {}
        self.exec_commands = []
        self.execute_real_commands = execute_real_commands
        self.incoming.put_nowait(json.dumps(task or {
            "type": "task", "protocol": 1,
            "instruction": "Write the requested deliverable",
            "workdir": "/app",
            "limits": {"agent_timeout_sec": 600, "exec_timeout_sec": 120},
            "mcp_servers": [],
        }))

    async def recv(self):
        return await self.incoming.get()

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.incoming.get()

    async def send(self, frame):
        msg = json.loads(frame)
        self.sent.append(msg)
        type_ = msg.get("type")
        rid = msg.get("id")
        if type_ == "exec":
            self.exec_commands.append(msg["command"])
            if self.execute_real_commands:
                r = subprocess.run(
                    msg["command"], shell=True,
                    capture_output=True, text=True, timeout=10,
                )
                answer = {
                    "exit_code": r.returncode,
                    "output": r.stdout + r.stderr,
                    "timed_out": False, "truncated": False,
                }
            else:
                answer = {
                    "exit_code": 0, "output": "mock-exec-ok",
                    "timed_out": False, "truncated": False,
                }
        elif type_ == "write_file":
            path = msg["path"]
            content = base64.b64decode(msg["content_b64"])
            if msg.get("append"):
                content = self.files.get(path, b"") + content
            self.files[path] = content
            answer = {"ok": True, "size": len(content)}
        elif type_ == "read_file":
            path = msg["path"]
            if path not in self.files:
                answer = {"ok": False, "error": "not found"}
            else:
                value = self.files[path]
                answer = {
                    "ok": True, "content_b64": base64.b64encode(value).decode(),
                    "size": len(value), "truncated": False,
                }
        else:
            return
        self.incoming.put_nowait(json.dumps({"type": "result", "id": rid, **answer}))


class ProtocolSmokeTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_protocol_round_trip(self):
        socket = EvaluatorSocket()

        async def handler(session):
            response = await session.exec("echo test")
            self.assertEqual(response["output"], "mock-exec-ok")
            await session.write_file("/app/output/report.json", '{"ok":true}')
            data = await session.read_file("/app/output/report.json")
            self.assertEqual(data, b'{"ok":true}')
            await session.log("completed mock workflow")
            return "Task completed"

        await asyncio.wait_for(taskbridge._session(socket, handler), timeout=5)
        sent_types = [msg["type"] for msg in socket.sent]
        self.assertIn("exec", sent_types)
        self.assertIn("write_file", sent_types)
        self.assertIn("read_file", sent_types)
        self.assertIn("log", sent_types)
        self.assertEqual(socket.sent[-1]["type"], "done")
        self.assertEqual(socket.sent[-1]["final_message"], "Task completed")
        self.assertEqual(socket.files["/app/output/report.json"], b'{"ok":true}')

    async def test_invalid_initial_frame_is_rejected(self):
        socket = EvaluatorSocket(task={"type": "invalid", "instruction": ""})

        async def handler(session):
            raise AssertionError("must never run")

        await asyncio.wait_for(taskbridge._session(socket, handler), timeout=5)
        self.assertEqual(socket.sent[-1]["type"], "error")

    async def test_partial_final_message_survives_handler_failure(self):
        socket = EvaluatorSocket()

        async def handler(session):
            session.final_message = "First deliverable was written"
            await session.write_file("/app/output/part.txt", "partial work")
            raise RuntimeError("simulated model failure")

        await asyncio.wait_for(taskbridge._session(socket, handler), timeout=5)
        self.assertEqual(socket.sent[-1]["type"], "done")
        self.assertIn("First deliverable", socket.sent[-1]["final_message"])

    async def test_failure_without_partial_answer_reports_error(self):
        socket = EvaluatorSocket()

        async def handler(session):
            raise RuntimeError("early failure")

        await asyncio.wait_for(taskbridge._session(socket, handler), timeout=5)
        self.assertEqual(socket.sent[-1]["type"], "error")

    async def test_late_cancelled_rpc_does_not_corrupt_next_response(self):
        class DelayedSocket(EvaluatorSocket):
            async def send(self, frame):
                msg = json.loads(frame)
                self.sent.append(msg)
                if msg.get("type") != "exec":
                    return
                rid = msg["id"]
                if msg["command"] == "slow":
                    async def late():
                        await asyncio.sleep(0.055)
                        await self.incoming.put(json.dumps({
                            "type": "result", "id": rid,
                            "ok": True, "exit_code": 0, "output": "OLD",
                        }))
                    asyncio.create_task(late())
                else:
                    await self.incoming.put(json.dumps({
                        "type": "result", "id": rid,
                        "ok": True, "exit_code": 0, "output": "NEW",
                    }))

        socket = DelayedSocket()
        task = json.loads(await socket.recv())
        session = taskbridge.Session(socket, task)
        reader = asyncio.create_task(session._read())
        try:
            with self.assertRaises(asyncio.TimeoutError):
                await asyncio.wait_for(session.exec("slow"), timeout=0.008)
            await asyncio.sleep(0.075)
            result = await asyncio.wait_for(session.exec("fast"), timeout=1)
            self.assertEqual(result["output"], "NEW")
            self.assertEqual(len(session._waiting), 0)
        finally:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)

    async def test_file_verification_tool_through_protocol(self):
        with tempfile.TemporaryDirectory(prefix="ple-protocol-") as td:
            output = pathlib.Path(td) / "actual.json"
            output.write_text('{"works":true}', encoding="utf-8")
            socket = EvaluatorSocket(execute_real_commands=True)

            async def handler(session):
                agent = Agent(session)
                answer = await agent.run_tool(
                    "verify_file", {"path": str(output), "file_format": "json"}
                )
                self.assertIn("PASS:", answer)
                return "Verified output via evaluator protocol"

            await asyncio.wait_for(taskbridge._session(socket, handler), timeout=5)
            self.assertEqual(socket.sent[-1]["type"], "done")
            self.assertTrue(any("python3 -I -W ignore -c" in x for x in socket.exec_commands))


if __name__ == "__main__":
    unittest.main()
