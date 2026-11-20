"""Offline tests: recover from model API failures without losing task work."""
import asyncio
import unittest

import openai

from serve import Agent, StreamBadRequest


class FakeSession:
    instruction = "Write an engineering package"
    workdir = "/app"
    mcp_servers = []
    limits = {}

    def __init__(self):
        self.final_message = ""
        self.logs = []

    def time_left(self):
        return 3600

    async def log(self, item):
        self.logs.append(item)


class ModelErrorTests(unittest.IsolatedAsyncioTestCase):
    def make_agent(self):
        a=Agent(FakeSession())

        async def no_mcp():
            return ""

        a.connect_mcp=no_mcp
        return a

    async def test_recovers_model_error_with_summary(self):
        agent=self.make_agent()

        async def failed_complete(**kwargs):
            raise openai.OpenAIError("simulated 400")

        async def successful_finish(why):
            self.assertIn("endpoint failed", why)
            return "SUMMARY"

        agent.complete=failed_complete
        agent.finish=successful_finish
        self.assertEqual(await agent.run(), "SUMMARY")
        self.assertTrue(any("OpenAIError" in t for t in agent.session.logs))

    async def test_unrecoverable_model_error_returns_nonempty_done(self):
        agent=self.make_agent()

        async def failed_complete(**kwargs):
            raise StreamBadRequest("simulated invalid model request")

        async def failed_finish(why):
            return ""

        agent.complete=failed_complete
        agent.finish=failed_finish
        result=await agent.run()
        self.assertIn("Model endpoint failed", result)

    async def test_cancel_propagates(self):
        agent=self.make_agent()

        async def cancelled(**kwargs):
            raise asyncio.CancelledError()

        agent.complete=cancelled
        with self.assertRaises(asyncio.CancelledError):
            await agent.run()


if __name__ == "__main__":
    unittest.main()
