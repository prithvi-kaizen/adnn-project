import unittest
from dataclasses import replace

from src.data.course_expansion import IRREVERSIBLE, MECHANISMS, PARTIAL, REVERSIBLE, pair_fixture
from src.data.course_paid_recovery import DESIGNS, paid_pair
from src.sandbox.core import legal_actions, transition


def exhaustive_fixture_labels(before, action):
    after, _ = transition(before, action)
    expected = before.object()
    criteria_matches = []
    pending = [(after, 0)]
    while pending:
        state, depth = pending.pop()
        current = state.object()
        if current == expected:
            criteria_matches.append((depth, REVERSIBLE))
        else:
            paid = expected["wallet"] - current["wallet"]
            if paid > 0 and current["fees_paid"] - expected["fees_paid"] == paid:
                adjusted = {**current, "wallet": expected["wallet"], "fees_paid": expected["fees_paid"]}
                if adjusted == expected:
                    criteria_matches.append((depth, PARTIAL))
        if depth < 3:
            pending.extend((transition(state, candidate)[0], depth + 1) for candidate in legal_actions(state))
    return tuple(next((label for label in (REVERSIBLE, PARTIAL) if any(
        depth <= horizon and matched == label for depth, matched in criteria_matches)), IRREVERSIBLE)
        for horizon in (1, 2, 3))


class SandboxExpansionTests(unittest.TestCase):
    def test_paid_recovery_designs_against_unmerged_search(self):
        for name, (field, first, second) in DESIGNS.items():
            states, action, _ = paid_pair(name, 0, 1234)
            left, right = (state.object() for state in states)
            self.assertEqual([key for key in left if left[key] != right[key]], [field], name)
            for state, expected in zip(states, (first, second), strict=True):
                with self.subTest(design=name, expected=expected):
                    self.assertEqual(exhaustive_fixture_labels(state, action), expected)

    def test_two_tick_ordered_mail_has_a_nonobvious_alternative_recovery(self):
        states, action, _ = pair_fixture("cancel_ordered_mail", 0, 1234)
        timers = {**states[0].pending_mail, "notice_1234": 2}
        intermediate = replace(states[0], pending_mail=timers)
        self.assertEqual(exhaustive_fixture_labels(intermediate, action),
                         (IRREVERSIBLE, IRREVERSIBLE, REVERSIBLE))

    def test_expected_horizon_labels_against_unmerged_full_state_search(self):
        for name, (field, first, second) in MECHANISMS.items():
            states, action, _ = pair_fixture(name, 0, 1234)
            left, right = (state.object() for state in states)
            self.assertEqual([key for key in left if left[key] != right[key]], [field], name)
            for state, expected in zip(states, (first, second), strict=True):
                with self.subTest(mechanism=name, expected=expected):
                    self.assertEqual(exhaustive_fixture_labels(state, action), expected)
