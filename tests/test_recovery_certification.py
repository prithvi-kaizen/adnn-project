"""Independent finite graph fixtures, never physical-world label evidence."""

import json
import unittest

from src.scienceworld_adapter.replay import ExportedState, RecoveryPolicy, search_recovery


class GraphWorld:
    legal_actions_complete = True

    def __init__(self, edges, payloads):
        self.edges, self.payloads, self.node = edges, payloads, 'start'

    def replay(self, actions):
        self.node = 'start'
        for action in actions:
            self.node = self.edges[self.node][action]
        return ExportedState(json.dumps(self.payloads[self.node], sort_keys=True), True)

    def legal_actions(self):
        return tuple(self.edges[self.node])


def paid_policy(target, state):
    a, b = json.loads(target.payload), json.loads(state.payload)
    cost = a['balance'] - b['balance']
    if a['contents'] == b['contents'] and cost > 0:
        return {'cost': cost, 'differences': {'balance': [a['balance'], b['balance']]}}
    return None


POLICY = RecoveryPolicy('fixture_paid_balance_only_v1', True, paid_policy)


class CertifiedRecoveryTests(unittest.TestCase):
    def world(self, free=False, approximate=False):
        edges = {'start': {'commit': 'lost'}, 'lost': {'pay': 'paid'}, 'paid': {}}
        payloads = {'start': {'contents': 'original', 'balance': 10},
                    'lost': {'contents': 'lost', 'balance': 10},
                    'paid': {'contents': 'original', 'balance': 8}}
        if free:
            edges['paid']['undo_fee'] = 'start'
        if approximate:
            payloads['paid'] = {'contents': 'original_with_loss', 'balance': 10}
        return GraphWorld(edges, payloads)

    def test_paid_recovery_requires_exhaustion_and_records_resource_cost(self):
        for horizon in (1, 2, 3):
            result = search_recovery(self.world(), (), 'commit', horizon, recovery_policy=POLICY)
            self.assertEqual(result['label'], 'partially_reversible')
            self.assertTrue(result['search_exhausted'])
            self.assertEqual(result['witness'], ['pay'])
            self.assertEqual(result['recovery_cost'], 2)

    def test_longer_free_witness_overrides_earlier_paid_witness(self):
        self.assertEqual(search_recovery(self.world(True), (), 'commit', 1, recovery_policy=POLICY)['label'],
                         'partially_reversible')
        for horizon in (2, 3):
            result = search_recovery(self.world(True), (), 'commit', horizon, recovery_policy=POLICY)
            self.assertEqual(result['label'], 'reversible')
            self.assertEqual(result['witness'], ['pay', 'undo_fee'])

    def test_exhausted_negative_needs_complete_grammar_and_policy(self):
        world = self.world(approximate=True)
        for horizon in (1, 2, 3):
            self.assertEqual(search_recovery(world, (), 'commit', horizon, recovery_policy=POLICY)['label'],
                             'irreversible')
        world.legal_actions_complete = False
        self.assertEqual(search_recovery(world, (), 'commit', recovery_policy=POLICY)['label'], 'unknown')
        self.assertEqual(search_recovery(self.world(), (), 'commit')['label'], 'unknown')

    def test_approximate_rule_is_explicit_and_not_inferred_from_text_similarity(self):
        world = self.world(approximate=True)
        policy = RecoveryPolicy('fixture_disclosed_content_loss', True,
            lambda target, state: {'cost': 0, 'differences': {'contents': 'disclosed suffix loss'}}
            if json.loads(state.payload)['contents'] == 'original_with_loss' else None)
        result = search_recovery(world, (), 'commit', recovery_policy=policy)
        self.assertEqual(result['label'], 'partially_reversible')
        self.assertEqual(result['recovery_cost'], 0)
        self.assertEqual(search_recovery(world, (), 'commit', recovery_policy=POLICY)['label'], 'irreversible')

    def test_budget_and_frontier_censoring_never_yield_partial_or_irreversible(self):
        result = search_recovery(self.world(), (), 'commit', max_nodes=1, recovery_policy=POLICY)
        self.assertEqual(result['label'], 'unknown')
        self.assertFalse(result['search_exhausted'])
        world = GraphWorld({'start': {'commit': 'lost'}, 'lost': {'a': 'x', 'b': 'y'}, 'x': {}, 'y': {}},
                           {k: {'contents': k, 'balance': 10} for k in ('start', 'lost', 'x', 'y')})
        result = search_recovery(world, (), 'commit', max_frontier=1, recovery_policy=POLICY)
        self.assertEqual(result['label'], 'unknown')
        self.assertEqual(result['censor_reason'], 'search_frontier_censored')

    def test_inconsistent_completeness_claim_is_rejected(self):
        with self.assertRaises(ValueError):
            ExportedState('{}', True, 'hidden_variable')
        with self.assertRaises(ValueError):
            ExportedState('{}', False)
