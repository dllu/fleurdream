import unittest
from dataclasses import replace
from unittest import mock

import guided
import progression
import solver


class GuidedTests(unittest.TestCase):
    def setUp(self):
        self.settings = solver.target, solver.tgt_abc, solver.abc, solver.max_len

    def tearDown(self):
        solver.target, solver.tgt_abc, solver.abc, solver.max_len = self.settings

    def roots(self, goal, spells, sources):
        solver.target = goal
        solver.tgt_abc = ''.join(c for c in solver.abc if c in goal)
        return solver.starting_states(spells, sources)

    def test_guided_findandreplace_without_using_itself_or_heedonic(self):
        spells = frozenset(solver.SPELL_NAMES) - {'find_and_replace', 'unlock_letter'}
        flowers = ('redpedalwalker', 'lilachangerbloom', 'cespyoped',
                   'tigerwishgrantbud', 'honeycombbmineraleaf')
        roots = self.roots('findandreplace', spells, flowers)
        result = guided.discover('findandreplace', roots, solver)
        self.assertIsNotNone(result.states)
        self.assertLessEqual(solver.spell_count(result.states[-1].hist), 12)
        state, required = progression.replay('findandreplace', result.states[-1].hist,
                                             progression.Technology(spells, flowers))
        self.assertIn('findandreplace', state.word)
        self.assertNotIn('find_and_replace', required.spells)
        self.assertNotIn('heedonicindigo', required.sources)
        for before, after in zip(result.states, result.states[1:]):
            self.assertTrue(any(child.key() == after.key() and child.hist == after.hist
                                for child in before.generate_states_unlock()))

    def test_cast_limit_and_work_budget_do_not_claim_infeasibility(self):
        roots = self.roots('ac', ('change',), ('ab',))
        short = guided.discover('ac', roots, solver, max_cost=0)
        self.assertIsNone(short.states)
        enough = guided.discover('ac', roots, solver, max_cost=1)
        self.assertIsNotNone(enough.states)
        self.assertEqual(solver.spell_count(enough.states[-1].hist), 1)
        roots = self.roots('zzzz', ('move', 'swap'), ('abc',))
        limited = guided.discover('zzzz', roots, solver, node_budget=0)
        self.assertIsNone(limited.states)
        self.assertEqual(limited.expanded, 0)

    def test_guided_successors_are_real_spells_with_frozen_constraints(self):
        roots = self.roots('bace', ('move', 'swap', 'swap_adjacent', 'change', 'add',
                                   'alter', 'adjacent', 'duplicate', 'combine'), ('abcd',))
        root = roots[0]
        root.frozen[1] = True
        legal = {child.hist[-1]: child.key() for child in root.generate_states_unlock()}
        for child in guided.successors(root, 'bace', solver):
            self.assertIn(child.hist[-1], legal)
            self.assertEqual(child.key(), legal[child.hist[-1]])

    def test_move_can_discard_a_letter_that_is_absent_from_the_target(self):
        roots = self.roots('abc', ('move',), ('apbc',))
        roots[0].frozen[0] = True
        result = guided.discover('abc', roots, solver, extended=True)
        self.assertIsNotNone(result.states)
        self.assertEqual(solver.spell_count(result.states[-1].hist), 1)
        self.assertIn('abc', result.states[-1].word)
        self.assertTrue(result.states[-1].hist[-1].startswith('move_1_'))

    def test_unlock_aware_alignment_only_proposes_real_casts(self):
        roots = self.roots('abg', ('unlock_letter', 'adjacent'), ('abh',))
        roots[0].frozen[2] = True
        result = guided.discover('abg', roots, solver, extended=True, unlock_ordering=True)
        self.assertIsNotNone(result.states)
        self.assertEqual(solver.spell_count(result.states[-1].hist), 2)
        self.assertTrue(result.states[-1].hist[1].startswith('unlock_2_'))
        for before, after in zip(result.states, result.states[1:]):
            self.assertTrue(any(child.key() == after.key() and child.hist == after.hist
                                for child in before.generate_states_unlock()))

    def test_most_frequent_and_unlock_do_not_construct_other_spell_successors(self):
        roots = self.roots('abac', ('most_frequent', 'unlock_letter'), ('aba',))
        roots[0].frozen[1] = True
        with mock.patch.object(solver.State, 'generate_states_unlock',
                               side_effect=AssertionError('unnecessary full expansion')):
            children = list(guided.successors(roots[0], 'abac', solver))
        self.assertTrue(any(s.hist[-1].startswith('mostf_') for s in children))
        self.assertTrue(any(s.hist[-1].startswith('unlock_') for s in children))

    def test_run_lock_counts_only_coalesce_discovery_and_keep_exact_replay_states(self):
        roots = self.roots('aaab', ('change',), ('aaaa',))
        wrong = replace(roots[0], frozen=[False, True, True, True])
        right = replace(roots[0], frozen=[True, True, True, False])
        self.assertNotEqual(wrong.key(), right.key())
        self.assertNotEqual(guided.discovery_key(wrong), guided.discovery_key(right))
        self.assertEqual(guided.discovery_key(wrong, True), guided.discovery_key(right, True))
        self.assertFalse(any('aaab' in child.word for child in wrong.change()))
        self.assertTrue(any('aaab' in child.word for child in right.change()))
        result = solver.solve('aaab', [wrong, right], guided_search=False)
        self.assertEqual(result.cost, 1)
        self.assertTrue(result.optimal)
        self.assertEqual(result.state.word, 'aaab')
        self.assertEqual(result.state.frozen, [True] * 4)

    def test_coalescing_preserves_exact_lock_positions_in_pairs_and_triplets(self):
        for word in ('aa', 'aaa', 'aabbbcccc'):
            roots = self.roots('ab', ('change',), (word,))
            left = replace(roots[0], frozen=[True] + [False] * (len(word) - 1))
            right = replace(roots[0], frozen=[False, True] + [False] * (len(word) - 2))
            self.assertNotEqual(guided.discovery_key(left, True), guided.discovery_key(right, True))
        root = self.roots('ab', ('change',), ('aaaa',))[0]
        left, right = replace(root, frozen=[True, False, False, False]), replace(
            root, frozen=[False, True, False, False])
        self.assertNotEqual(guided.discovery_key(left, True, min_lock_run=5),
                            guided.discovery_key(right, True, min_lock_run=5))

    def test_coalesced_discovery_returns_actual_legal_casts(self):
        roots = self.roots('aaaab', ('duplicate', 'move', 'change', 'unlock_letter'), ('aaaac',))
        roots[0].frozen[1] = True
        result = guided.discover('aaaab', roots, solver, coalesce_locks=True, extended=True)
        self.assertIsNotNone(result.states)
        for before, after in zip(result.states, result.states[1:]):
            self.assertTrue(any(child.key() == after.key() and child.hist == after.hist
                                for child in before.generate_states_unlock()))

    def test_newwatertower_with_two_flowers_and_early_spells(self):
        spells = frozenset(('add', 'combine', 'duplicate', 'subtract', 'move',
                            'swap', 'change', 'recast_spell'))
        flowers = ('redpedalwalker', 'tigerwishgrantbud')
        roots = self.roots('newwatertower', spells, flowers)
        result = guided.discover('newwatertower', roots, solver, width=256,
                                 family_width=16, extended=True,
                                 node_budget=8192, candidate_budget=500000)
        self.assertIsNotNone(result.states)
        self.assertLessEqual(solver.spell_count(result.states[-1].hist), 10)
        _, required = progression.replay('newwatertower', result.states[-1].hist,
                                         progression.Technology(spells, flowers))
        self.assertTrue(required.spells <= spells)
        self.assertEqual(required.sources, flowers)


if __name__ == '__main__':
    unittest.main()
