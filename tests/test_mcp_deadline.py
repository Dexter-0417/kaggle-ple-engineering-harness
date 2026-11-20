"""Offline deadline regression tests for the Google PLE agent."""
import asyncio
import unittest

from serve import Agent


class FakeSession:
    def __init__(self, remaining):
        self.remaining = remaining
        self.logs = []

    def time_left(self):
        return self.remaining

    async def log(self, message):
        self.logs.append(message)


class FakeServer:
    def __init__(self, delay=0):
        self.delay = delay
        self.timeout_sec = 300
        self.calls = []

    async def call(self, name, arguments):
        self.calls.append((name, arguments, self.timeout_sec))
        await asyncio.sleep(self.delay)
        return "DONE"


class MCPDeadlineTests(unittest.IsolatedAsyncioTestCase):
    def make_agent(self, remaining, margin=1.0, delay=0):
        session = FakeSession(remaining)
        agent = Agent(session)
        agent.finish_margin = margin
        server = FakeServer(delay)
        agent.mcp["some__tool"] = (server, {"name": "tool"})
        return agent, server

    async def test_timeout_preserves_final_summary_time(self):
        agent, server = self.make_agent(remaining=1.1, margin=1, delay=2)
        loop = asyncio.get_running_loop()
        start = loop.time()
        result = await agent.run_tool("some__tool", {"key": "value"})
        elapsed = loop.time() - start
        self.assertIn("timed out", result)
        self.assertLess(elapsed, 0.7)
        self.assertEqual(server.timeout_sec, 300)

    async def test_no_call_when_deadline_has_arrived(self):
        agent, server = self.make_agent(remaining=0.5, margin=1, delay=0)
        result = await agent.run_tool("some__tool", {})
        self.assertIn("deadline approaching", result)
        self.assertEqual(server.calls, [])

    async def test_fast_call_succeeds_and_timeout_is_restored(self):
        agent, server = self.make_agent(remaining=3, margin=1, delay=0)
        result = await agent.run_tool("some__tool", {"item": 1})
        self.assertEqual(result, "DONE")
        self.assertEqual(server.calls, [("tool", {"item": 1}, 2)])
        self.assertEqual(server.timeout_sec, 300)

    async def test_dynamic_mcp_call_uses_same_deadline(self):
        agent, server = self.make_agent(remaining=3, margin=1, delay=0)
        result = await agent.run_tool("mcp_call", {"tool": "some__tool", "arguments": {"x": True}})
        self.assertEqual(result, "DONE")
        self.assertEqual(server.calls[0][:2], ("tool", {"x": True}))


if __name__ == "__main__":
    unittest.main()
