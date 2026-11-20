"""Offline tests for honest tool-result reporting in the PLE harness."""
import asyncio
import unittest
from serve import Agent
import taskbridge


class FakeSession:
    workdir = "/app"
    limits = {"exec_timeout_sec": 600}

    def __init__(self, result=None, remaining=900):
        self.result = result or {"exit_code": 0, "output": "", "truncated": False}
        self.remaining = remaining

    def time_left(self):
        return self.remaining

    async def log(self, message):
        return None

    async def exec(self, command, timeout):
        return self.result


class FakeMCP:
    timeout_sec = 300
    async def call(self, name, arguments):
        await asyncio.sleep(1)
        return "executed"


class ToolReliabilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_truncated_command_output_reports_data_loss(self):
        agent = Agent(FakeSession({"exit_code": 0, "output": "some output", "truncated": True}))
        answer = await agent.bash("echo test")
        self.assertIn("some output", answer)
        self.assertIn("WARNING", answer)
        self.assertIn("truncated", answer)

    async def test_full_output_has_no_warning(self):
        agent = Agent(FakeSession({"exit_code": 0, "output": "complete", "truncated": False}))
        answer = await agent.bash("echo test")
        self.assertIn("complete", answer)
        self.assertNotIn("WARNING", answer)

    async def test_evaluator_mcp_task_error_warns_about_side_effects(self):
        agent = Agent(FakeSession(remaining=20))
        agent.finish_margin = 2

        class FailedMCP:
            timeout_sec = 300

            async def call(self, name, arguments):
                raise taskbridge.TaskError("request timed out after dispatch")

        agent.mcp["x__mutate"] = (FailedMCP(), {"name": "mutate"})
        answer = await agent.run_tool("x__mutate", {"x": 1})
        self.assertIn("request timed out", answer)
        self.assertIn("MAY have applied", answer)

    async def test_mcp_call_task_error_uses_same_safety_warning(self):
        agent = Agent(FakeSession(remaining=20))
        agent.finish_margin = 2

        class FailedMCP:
            timeout_sec = 300

            async def call(self, name, arguments):
                raise taskbridge.TaskError("timeout after dispatch")

        agent.mcp["x__mutate"] = (FailedMCP(), {"name": "mutate"})
        answer = await agent.run_tool("mcp_call", {"tool": "x__mutate", "arguments": {}})
        self.assertIn("MAY have applied", answer)

    async def test_clipped_command_warning_survives_tool_dispatch(self):
        agent = Agent(FakeSession({"exit_code": 0, "output": "Z"*45000, "truncated": True}))
        answer = await agent.run_tool("bash", {"command": "echo example"})
        self.assertIn("WARNING", answer)
        self.assertIn("truncated", answer)

    async def test_mcp_timeout_warns_about_possible_side_effects(self):
        agent = Agent(FakeSession(remaining=0.07))
        agent.finish_margin = 0.02
        agent.mcp["x__mutate"] = (FakeMCP(), {"name": "mutate"})
        answer = await agent.run_tool("x__mutate", {"key": "value"})
        self.assertIn("timed out", answer)
        self.assertIn("already have applied", answer)
        self.assertIn("before retrying", answer)


if __name__ == "__main__":
    unittest.main()
