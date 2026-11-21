"""Regression tests for task-independent early-deliverable milestones."""
import unittest

from serve import output_checkpoint, Agent


class CheckpointPolicyTests(unittest.TestCase):
    def test_first_checkpoint_before_long_tool_detour(self):
        self.assertIsNone(output_checkpoint(7))
        text = output_checkpoint(8)
        self.assertIn("task's required entities", text)
        self.assertIn("implement a first substantive", text)
        self.assertIn("empty format-valid scaffold", text)
        self.assertIn("verify_file", text)

    def test_checkpoints_are_sparse_and_deterministic(self):
        expected = {8, 16, 24, 32, 48, 64}
        found = {n for n in range(1, 67) if output_checkpoint(n)}
        self.assertEqual(found, expected)

    def test_no_specific_task_solution_or_scoring_claim(self):
        note = output_checkpoint(16)
        self.assertNotIn("data-hall-mep-routing", note)
        self.assertNotIn("model.ifc", note)
        self.assertNotIn("solution.py", note)
        self.assertIn("Never fabricate", note)
        self.assertIn("engineering compliance", note)

    def test_zero_and_small_task_no_extra_prompt(self):
        for i in (0, 1, 2, 3, 6, 9):
            with self.subTest(i=i):
                self.assertIsNone(output_checkpoint(i))

    def test_agent_tracks_tool_turns_separately(self):
        class DummySession:
            pass
        agent = Agent(DummySession())
        self.assertEqual(agent.tool_turns, 0)


if __name__ == "__main__":
    unittest.main()
