import json
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from src.agent.loop import goal_met, parse_output, run_episode, verify_episode
from src.data.pilot import scenarios
from src.sandbox import Action, ToolError


def generator(outputs):
    iterator = iter(outputs)
    return lambda messages: {"text": next(iterator)}


class AgentLoopTests(unittest.TestCase):
    def setUp(self):
        self.scenario = scenarios()[0]

    def test_success_and_json_serialized_trace_replay(self):
        result = run_episode(self.scenario, generator([
            Action.make("check_state").text(), self.scenario.action.text(),
            Action.make("finish", reason="Task complete").text(),
        ]))
        self.assertEqual(result["status"], "finish")
        self.assertTrue(result["goal_met_at_termination"])
        verify_episode(result)
        verify_episode(json.loads(json.dumps(result)))
        self.assertEqual(self.scenario.state.digest(), result["initial_hash"])

    def test_invalid_calls_consume_budget_and_abort_after_two(self):
        result = run_episode(self.scenario, generator(["not json", "```json {} ```"]))
        self.assertEqual(result["status"], "consecutive_invalid_abort")
        self.assertEqual(result["steps"], 2)
        self.assertEqual(result["initial_hash"], result["final_hash"])
        verify_episode(result)

    def test_valid_action_resets_invalid_counter(self):
        result = run_episode(self.scenario, generator([
            "bad", Action.make("check_state").text(), "bad",
            Action.make("refuse", reason="Cannot complete").text(),
        ]))
        self.assertEqual(result["steps"], 4)
        self.assertEqual(result["status"], "refuse")
        self.assertFalse(result["goal_met_at_termination"])
        verify_episode(result)

    def test_terminal_grammar_is_strict(self):
        for output in ('{"tool":"finish","arguments":{}}',
                       '{"tool":"refuse","arguments":{"reason":""}}',
                       '{"tool":"finish","arguments":{"reason":"ok","extra":"bad"}}',
                       '{"tool":"finish","tool":"refuse","arguments":{"reason":"ok"}}'):
            with self.subTest(output=output), self.assertRaises(ToolError):
                parse_output(output)

    def test_step_limit_is_enforced_and_trace_cannot_be_tampered(self):
        result = run_episode(self.scenario, generator([Action.make("wait").text()] * 20))
        self.assertEqual(result["steps"], 15)
        self.assertEqual(result["status"], "step_limit")
        verify_episode(result)
        bad = deepcopy(result)
        bad["trace"][0]["after_hash"] = "wrong"
        with self.assertRaises(ValueError):
            verify_episode(bad)

    def test_model_failure_is_logged_without_world_change(self):
        def broken(messages):
            raise TimeoutError("Fixture timeout")
        result = run_episode(self.scenario, broken)
        self.assertEqual(result["status"], "infrastructure_failure")
        self.assertEqual(result["initial_hash"], result["final_hash"])
        verify_episode(result)

    def test_unknown_tools_and_host_injection_are_never_executed(self):
        requests = [Action.make("exec", command="touch /tmp/adnn_host_intrusion").text(),
                    Action.make("delete_file", path="/etc/passwd").text()]
        with patch("subprocess.run", side_effect=AssertionError("Process access forbidden")), \
                patch.object(Path, "unlink", side_effect=AssertionError("Host delete forbidden")):
            result = run_episode(self.scenario, generator(requests))
        self.assertEqual(result["status"], "consecutive_invalid_abort")
        self.assertEqual(result["initial_hash"], result["final_hash"])
        verify_episode(result)

    def test_all_pilot_goals_have_a_direct_legal_completion(self):
        for scenario in scenarios():
            result = run_episode(scenario, generator([
                scenario.action.text(), Action.make("finish", reason="done").text(),
            ]))
            self.assertTrue(goal_met(scenario, scenario.state) is False)
            self.assertTrue(result["goal_met_at_termination"], scenario.scenario_id)
            verify_episode(result)

    def test_verifier_rejects_changed_history_summary_and_late_events(self):
        result = run_episode(self.scenario, generator([
            Action.make("check_state").text(), Action.make("finish", reason="done").text()]))
        for field in ("goal_met_at_termination", "goal_ever_met"):
            bad = deepcopy(result)
            bad[field] = not bad[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify_episode(bad)
        bad = deepcopy(result)
        bad["trace"][1]["request_messages"][-1]["content"] = "altered feedback"
        with self.assertRaises(ValueError):
            verify_episode(bad)
        bad = deepcopy(result)
        bad["trace"].append(deepcopy(bad["trace"][-1]))
        bad["steps"] = 3
        with self.assertRaises(ValueError):
            verify_episode(bad)
