"""Proposal guards and adversarial graph fixtures; no human/physical labels."""

import json
import unittest
from types import SimpleNamespace

from src.scienceworld_adapter.causal import canonical, proposed_state
from src.scienceworld_adapter.replay import (
    ExportedState,
    RecoveryPolicy,
    ReplayCertificationError,
    ScienceWorldReplay,
    search_recovery,
)


class CausalProposalTests(unittest.TestCase):
    def test_exact_floats_ordering_and_no_tolerance(self):
        self.assertEqual(canonical({'b': 1.0, 'a': [2, 3]}), canonical({'a': [2, 3], 'b': 1.0}))
        self.assertNotEqual(canonical(1.0), canonical(1.0000000000000002))
        self.assertNotEqual(canonical([1, 2]), canonical([2, 1]))
        self.assertNotEqual(canonical(-0.0), canonical(0.0))
        for value in (float('inf'), float('nan'), object()):
            with self.assertRaises(ValueError):
                canonical(value)

    def test_proposal_can_never_promote_matching_projection_to_final_state(self):
        proposal = proposed_state([{'uuid': 7, 'unsupported': ['plant_counter_unknown']}], family='fixture')
        state = proposal.for_search()
        self.assertFalse(state.complete)
        self.assertIn('plant_counter_unknown', state.incompleteness_reason)
        self.assertIn('instructor_amendment_pending', state.incompleteness_reason)
        with self.assertRaises(ValueError):
            proposed_state([{'uuid': 7}, {'uuid': 7}], family='fixture')

    def test_projection_aliases_are_not_merged_in_bounded_search(self):
        class AliasWorld:
            legal_actions_complete = False

            def replay(self, actions):
                self.node = 'start'
                edges = {'start': {'commit': 'lost'}, 'lost': {'a': 'dead', 'b': 'live'},
                         'live': {'undo': 'start'}, 'dead': {}}
                for action in actions:
                    self.node = edges[self.node][action]
                self.edges = edges
                projection = {'start': 0, 'lost': 1, 'dead': 2, 'live': 2}[self.node]
                return ExportedState(str(projection), False, 'projection_not_transition_state')

            def legal_actions(self):
                return tuple(self.edges[self.node])

        for h in (1, 2, 3):
            result = search_recovery(AliasWorld(), (), 'commit', h, diagnostic_only=True)
            self.assertEqual(result['label'], 'unknown')
            self.assertEqual(result['duplicate_states_skipped'], 0)
            self.assertEqual(bool(result['diagnostic_observational_matches']), h >= 2)

    def test_branch_grammar_loss_prevents_exhausted_negative(self):
        class BranchGrammar:
            legal_actions_complete = True

            def replay(self, actions):
                self.legal_actions_complete = len(actions) == 0
                self.actions = actions
                return ExportedState(json.dumps(list(actions)), True)

            def legal_actions(self):
                return ('commit',) if not self.actions else ()

        policy = RecoveryPolicy('fixture_no_partial', True, lambda a, b: None)
        for h in (1, 2, 3):
            result = search_recovery(BranchGrammar(), (), 'commit', h, recovery_policy=policy)
            self.assertEqual(result['label'], 'unknown')
            self.assertFalse(result['legal_action_space_complete'])
            self.assertTrue(result['search_exhausted'])

    def test_state_certification_loss_is_recorded_as_unknown(self):
        class BranchState:
            legal_actions_complete = True

            def replay(self, actions):
                return ExportedState(str(actions), not actions, 'new_unsupported_state' if actions else None)

            def legal_actions(self):
                return ('commit',)

        result = search_recovery(BranchState(), (), 'commit')
        self.assertEqual(result['label'], 'unknown')
        self.assertFalse(result['state_export_complete'])
        self.assertFalse(result['search_exhausted'])
        self.assertEqual(result['censor_reason'], 'state_certification_changed_during_search')

    def test_enumerated_but_unexecutable_branch_is_not_dropped_from_negative_proof(self):
        class BrokenEnumeration:
            legal_actions_complete = True

            def replay(self, actions):
                if len(actions) > 1:
                    raise ReplayCertificationError({'reason': 'fixture_parser_failure', 'action': actions[-1]})
                self.actions = actions
                return ExportedState(str(actions), True)

            def legal_actions(self):
                return ('commit',) if not self.actions else ('unexecutable',)

        policy = RecoveryPolicy('fixture_no_partial', True, lambda a, b: None)
        for h in (1, 2, 3):
            result = search_recovery(BrokenEnumeration(), (), 'commit', h, recovery_policy=policy)
            self.assertEqual(result['label'], 'unknown')
            self.assertFalse(result['search_exhausted'])
            self.assertEqual(result['censor_reason'], 'replay_execution_not_certified')
            self.assertEqual(result['failed_replay']['branch'], ['commit', 'unexecutable'])

    def replay_fixture(self, *, parsed=True, ambiguous=False, error=False):
        class Env:
            history = 0
            classes = []

            def load(self, *args):
                self.history = 0
                self.classes = []

            def reset(self):
                pass

            def step(self, action):
                scaffold = ['ActionLookAround', 'ActionInventory']
                self.classes += scaffold + (['ActionOpenDoor'] if parsed else []) + scaffold + ['ActionTaskDesc']
                self.history = len(self.classes)
                return 'fixture output', 0, False, {}

            def get_valid_action_object_combinations(self):
                return ['fixture_action']

        env = Env()
        interface = SimpleNamespace(
            agent=lambda: SimpleNamespace(getActionHistorySize=lambda: env.history,
                getActionHistorySince=lambda start: [SimpleNamespace(getClass=lambda name=name:
                    SimpleNamespace(getName=lambda: 'scienceworld.actions.' + name)) for name in env.classes[start:]]),
            inputParser=lambda: SimpleNamespace(isInAmbiguousState=lambda: ambiguous),
            isInErrorState=lambda: error)
        env.server = SimpleNamespace(agentInterface=lambda: SimpleNamespace(get=lambda: interface))

        class Replay(ScienceWorldReplay):
            def snapshot(self):
                return ExportedState('fixture', False, 'test_only')

        return Replay('fixture_task', env=env)

    def test_replay_accepts_actual_unambiguous_history_increment(self):
        env = self.replay_fixture()
        self.assertEqual(env.replay(('fixture_action',)).payload, 'fixture')
        self.assertEqual(env.history, ('fixture_action',))

    def test_replay_refuses_parser_ambiguity_no_execution_and_engine_error(self):
        for flags in ({'parsed': False}, {'ambiguous': True}, {'error': True}):
            with self.subTest(flags=flags):
                env = self.replay_fixture(**flags)
                with self.assertRaises(ReplayCertificationError) as caught:
                    env.replay(('fixture_action',))
                self.assertEqual(caught.exception.receipt['action'], 'fixture_action')

    def test_free_wrapper_actions_cannot_make_a_rejected_request_look_executed(self):
        env = self.replay_fixture(parsed=False)
        with self.assertRaises(ReplayCertificationError) as caught:
            env.replay(('fixture_action',))
        receipt = caught.exception.receipt
        self.assertEqual(receipt['history_after'] - receipt['history_before'], 5)
        self.assertFalse(receipt['requested_action_recorded'])
