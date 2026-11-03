"""CPU-only checks for causal information separation in the trial driver."""

import copy
import unittest

from benchmarks.agent_load_trace import build_trial, canonical_digest, policy_observation


class AgentTraceTests(unittest.TestCase):
    def test_fixed_seed_is_reproducible(self):
        trial = build_trial()
        self.assertEqual(trial, build_trial())
        digest = trial.pop("trace_digest")
        self.assertEqual(digest, canonical_digest(trial))
        self.assertNotEqual(build_trial(7)["trace_digest"], build_trial()["trace_digest"])

    def test_future_durations_abandonment_and_turns_cannot_change_policy_view(self):
        trial = build_trial()
        changed = copy.deepcopy(trial)
        for session in changed["sessions"]:
            session["load_generator_private"] = {
                "tool_wait_ms_after_turn": [999999] * 6,
                "abandon_after_turn": 0,
            }
            for turn in session["turns"][1:]:
                turn["user_instruction"] = "Unseen future instruction"
                turn["tool_result_for_this_turn"] = "Unseen future result"
        self.assertEqual(policy_observation(trial, "agent-0", 0),
                         policy_observation(changed, "agent-0", 0))

    def test_trial_has_competition_variable_waits_and_abandonment(self):
        trial = build_trial()
        self.assertEqual(len(trial["sessions"]), 6)
        waits = {wait for session in trial["sessions"]
                 for wait in session["load_generator_private"]["tool_wait_ms_after_turn"]}
        self.assertGreater(len(waits), 2)
        self.assertTrue(any(s["load_generator_private"]["abandon_after_turn"] is not None
                            for s in trial["sessions"]))
        self.assertEqual(len(trial["background"]), 12)


if __name__ == "__main__":
    unittest.main()
