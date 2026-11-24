import unittest

from scripts.run_course_development_formats import CONFIG, read, summarize, system_prompt, with_system
from src.agent.loop import SYSTEM
from src.models.course_native import STATED_SYSTEM


class DevelopmentFormatTests(unittest.TestCase):
    def test_only_declared_system_text_changes(self):
        config = read(CONFIG)
        neutral = system_prompt(config, "stated", "candidate")
        self.assertEqual(neutral, STATED_SYSTEM.replace(config["stated_example_to_remove"], ""))
        changed = system_prompt(config, "agent", "candidate")
        self.assertEqual(changed, config["agent_prefix"] + SYSTEM)
        request = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "state"}]
        self.assertEqual(with_system(request, SYSTEM, changed)[1:], request[1:])
        self.assertEqual(request[0]["content"], SYSTEM)

    def test_selection_uses_legality_termination_and_parse_validity(self):
        config = read(CONFIG)
        assigned = [f"state-{number}" for number in range(6)]
        episodes = {
            "baseline": [{"id": key, "first_status": "executed", "status": "finish", "goal": True}
                         for key in assigned],
            "candidate": [{"id": key, "first_status": "rejected", "status": "finish", "goal": True}
                          for key in assigned]}
        stated = {condition: [{"id": key, "label": label, "truth": "reversible"} for key in assigned]
                  for condition, label in (("baseline", "reversible"), ("candidate", "irreversible"))}
        decision = summarize(config, episodes, stated, assigned)
        self.assertEqual(decision["agent_choice"], "baseline")
        self.assertEqual(decision["stated_choice"], "candidate")
        self.assertFalse(decision["authorizes_holdout"])
        episodes["baseline"][0]["status"] = "infrastructure_failure"
        self.assertEqual(summarize(config, episodes, stated, assigned)["status"],
                         "INCOMPLETE_COMPARISON_NO_FORMAT_SELECTED")
