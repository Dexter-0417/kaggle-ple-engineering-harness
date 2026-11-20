"""Offline MCP discovery, retry, cancellation, time-budget and fallback tests."""

import asyncio
import time
import types
import unittest
from collections import defaultdict
from unittest.mock import patch

import taskbridge
import serve


class FakeSession:
    workdir = "/app"
    instruction = "Do the engineering task"
    limits = {}

    def __init__(self, specs=(), remaining=3600):
        self.mcp_servers = list(specs)
        self.remaining = remaining
        self.logs = []
        self.final_message = ""

    def time_left(self):
        return self.remaining

    async def log(self, note):
        self.logs.append(note)


def spec(name):
    return {"name": name, "url": "http://127.0.0.1:8000/mcp/"}


class Factory:
    """Configure one fake startup action per attempt and observe new instances."""

    def __init__(self, behaviors=None):
        self.behaviors = behaviors or {}
        self.attempts = defaultdict(int)
        self.instances = []
        self.timeouts_seen = []

    def __call__(self, session, cfg):
        parent = self
        name = cfg["name"]
        attempt = self.attempts[name]
        self.attempts[name] += 1

        class FakeServer:
            timeout_sec = 300

            def __init__(self):
                self.name = name
                self.tools = [
                    {
                        "name": "act",
                        "description": "Example tool",
                        "inputSchema": {"type": "object", "properties": {}},
                    }
                ]

            async def start(self):
                parent.timeouts_seen.append(self.timeout_sec)
                actions = parent.behaviors.get(name, ["ok"])
                action = actions[min(attempt, len(actions) - 1)]
                if action == "hang":
                    await asyncio.sleep(30)
                elif action == "taskerror":
                    raise taskbridge.TaskError("unavailable")
                elif action == "keyerror":
                    raise KeyError("status")
                elif action == "typeerror":
                    raise TypeError("malformed frame")
                elif action == "cancel":
                    raise taskbridge.Cancelled()
                elif action == "async_cancel":
                    raise asyncio.CancelledError()
                elif action == "empty":
                    self.tools = []

        result = FakeServer()
        self.instances.append(result)
        return result


class StartupTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.patches = [
            patch.object(serve, "MCP_INIT_TOTAL_SEC", 0.12),
            patch.object(serve, "MCP_INIT_ATTEMPT_SEC", 0.035),
            patch.object(serve, "MCP_INIT_WAITS", (0, 0)),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    async def execute(self, server_specs, behaviors=None, remaining=3600):
        session = FakeSession(server_specs, remaining)
        factory = Factory(behaviors)
        agent = serve.Agent(session)
        if remaining is not None:
            # Agent.run() computes this reserve before discovery.
            agent.finish_margin = min(serve.FINISH_MARGIN_SEC, max(remaining, 0) / 4)
        with patch.object(serve, "MCPServer", factory):
            prompt = await agent.connect_mcp()
        return agent, session, factory, prompt

    async def test_no_servers_has_no_warning(self):
        agent, session, factory, prompt = await self.execute([])
        self.assertEqual(prompt, "")
        self.assertEqual(agent.mcp, {})
        self.assertEqual(factory.instances, [])

    async def test_success_registers_tool_restores_timeout(self):
        agent, _, factory, prompt = await self.execute([spec("apps")])
        self.assertIn("apps__act", agent.mcp)
        self.assertIn("task also offers its own tools", prompt)
        self.assertEqual(factory.instances[0].timeout_sec, 300)

    async def test_transient_failure_retries_with_fresh_instance(self):
        agent, _, factory, _ = await self.execute(
            [spec("apps")], {"apps": ["taskerror", "ok"]}
        )
        self.assertEqual(factory.attempts["apps"], 2)
        self.assertIsNot(factory.instances[0], factory.instances[1])
        self.assertIn("apps__act", agent.mcp)

    async def test_permanent_failure_warns_and_stops_at_three_attempts(self):
        agent, session, factory, prompt = await self.execute(
            [spec("apps")], {"apps": ["keyerror"]}
        )
        self.assertEqual(factory.attempts["apps"], 3)
        self.assertEqual(agent.mcp, {})
        self.assertIn("unavailable", prompt)
        self.assertTrue(any("KeyError" in msg for msg in session.logs))

    async def test_non_dict_server_spec_is_skipped(self):
        agent, session, factory, prompt = await self.execute(
            [None, spec("valid")]
        )
        self.assertIn("valid__act", agent.mcp)
        self.assertIn("invalid MCP specification", prompt)
        self.assertTrue(any("not a dict" in x for x in session.logs))

    async def test_real_mcp_server_filters_malformed_tool_definitions(self):
        import types
        from unittest.mock import AsyncMock

        server = serve.MCPServer(
            FakeSession(), spec("apps")
        )
        server.initialize = AsyncMock()
        server.list_tools = AsyncMock(return_value=[
            {}, {"name": 42}, {"name": ""}, {"name": "good", "inputSchema": {}}
        ])
        await server.start()
        self.assertEqual([t["name"] for t in server.tools], ["good"])

    async def test_malformed_spec_does_not_break_next_valid_server(self):
        agent, session, factory, prompt = await self.execute(
            [{"name": "broken"}, spec("valid")]
        )
        self.assertIn("valid__act", agent.mcp)
        self.assertIn("broken", prompt)
        self.assertEqual(factory.attempts["broken"], 0)
        self.assertTrue(any("invalid name/url" in x for x in session.logs))

    async def test_malformed_rpc_retries_and_recovers(self):
        agent, _, factory, _ = await self.execute(
            [spec("apps")], {"apps": ["typeerror", "ok"]}
        )
        self.assertEqual(factory.attempts["apps"], 2)
        self.assertIn("apps__act", agent.mcp)

    async def test_hung_first_server_does_not_starve_second(self):
        start = time.monotonic()
        agent, _, factory, prompt = await self.execute(
            [spec("slow"), spec("healthy")],
            {"slow": ["hang"], "healthy": ["ok"]},
        )
        self.assertLess(time.monotonic() - start, 0.45)
        self.assertIn("healthy__act", agent.mcp)
        self.assertIn("slow", prompt)
        self.assertGreaterEqual(factory.attempts["slow"], 1)

    async def test_short_task_caps_discovery_at_ten_percent(self):
        with patch.object(serve, "MCP_INIT_TOTAL_SEC", 120), patch.object(
            serve, "MCP_INIT_ATTEMPT_SEC", 40
        ):
            agent, session, factory, _ = await self.execute(
                [spec("apps")], remaining=20
            )
        self.assertEqual(factory.attempts["apps"], 1)
        self.assertLessEqual(factory.timeouts_seen[0], 2)

    async def test_no_time_left_gives_visible_warning_not_exception(self):
        agent, _, factory, prompt = await self.execute(
            [spec("apps")], remaining=0
        )
        self.assertEqual(factory.instances, [])
        self.assertIn("unavailable", prompt)

    async def test_task_cancellation_propagates(self):
        for action, expected in [
            ("cancel", taskbridge.Cancelled),
            ("async_cancel", asyncio.CancelledError),
        ]:
            with self.subTest(action=action):
                with self.assertRaises(expected):
                    await self.execute([spec("apps")], {"apps": [action]})

    async def test_no_tools_and_unavailable_server_are_reported(self):
        agent, _, _, prompt = await self.execute(
            [spec("empty"), spec("broken")],
            {"empty": ["empty"], "broken": ["taskerror"]},
        )
        self.assertEqual(agent.mcp, {})
        self.assertIn("broken", prompt)

    async def test_catalog_mode_retains_missing_server_warning(self):
        with patch.object(serve, "DIRECT_MCP_CHARS", 10):
            agent, _, _, prompt = await self.execute(
                [spec("ok"), spec("bad")],
                {"bad": ["taskerror"]},
            )
        self.assertIn("ok__act", agent.mcp)
        self.assertIn("bad", prompt)
        self.assertIn("ok__act", prompt)

    async def test_run_sets_summary_reserve_before_mcp_connection(self):
        session = FakeSession(remaining=20)
        agent = serve.Agent(session)
        observed = {}

        async def connect():
            observed["margin"] = agent.finish_margin
            return ""

        class Response:
            text = "Finished"
            calls = {}
            finish = "stop"

            def message(self):
                return {"role": "assistant", "content": self.text}

        async def reply(**kwargs):
            return Response()

        agent.connect_mcp = connect
        agent.complete = reply
        result = await agent.run()
        self.assertEqual(result, "Finished")
        self.assertEqual(observed["margin"], 5)


if __name__ == "__main__":
    unittest.main()
