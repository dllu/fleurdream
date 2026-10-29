import itertools
import random
import shutil
import unittest
from pathlib import Path
from unittest import mock

import native_search
import solver


class NativeSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary = native_search.ensure_binary()
        if cls.binary is None:
            raise unittest.SkipTest('optional C++ compiler unavailable')

    def setUp(self):
        self.settings = solver.target, solver.tgt_abc, solver.abc, solver.max_len

    def tearDown(self):
        solver.target, solver.tgt_abc, solver.abc, solver.max_len = self.settings

    def roots(self, goal, sources=('ab', 'cde'), spells=solver.SPELL_NAMES):
        solver.target = goal
        solver.tgt_abc = ''.join(c for c in solver.abc if c in goal)
        return solver.starting_states(spells, sources)

    def test_all_single_cast_rules_match_python_on_real_histories(self):
        solver.abc = 'abcd'
        roots = self.roots('abc')
        rng = random.Random(11)
        examples = roots + [solver.State('x' * 35, [False] * 35, hist=['x' * 35])]
        for _ in range(60):
            state = rng.choice(roots)
            for _ in range(rng.randrange(6)):
                children = list(state.generate_states_unlock())
                if children:
                    state = rng.choice(children)
            examples.append(state)
        for state in examples:
            data, _ = native_search._payload('abc', [state], solver, 64, 16, 18, 1, True)
            actual = set()
            for line in native_search._run(data, self.binary).splitlines():
                word, frozen, used, pending, command = line.split(' ', 4)
                actual.add((word, int(frozen), int(used), command))
            expected = {(child.word or '-', sum(bool(f) << i for i, f in enumerate(child.frozen)),
                         sum(bool(f) << i for i, f in enumerate(child.useds())), child.hist[-1])
                        for child in state.generate_states_unlock()
                        if not child.hist[-1].startswith('combine_drop') and len(child.word) <= 64}
            self.assertEqual(actual, expected, state.hist)

    def test_bit_distances_match_python_including_one_unlock(self):
        words = [''.join(cs) for n in range(4) for cs in itertools.product('ab', repeat=n)]
        for goal in words:
            states = [solver.State(word, list(frozen), hist=[word]) for word in words
                      for frozen in itertools.product((False, True), repeat=len(word))]
            data, _ = native_search._payload(goal, states, solver, 64, 16, 18, 2, True)
            lines = native_search._run(data, self.binary).splitlines()
            self.assertEqual(len(lines), len(states))
            for state, line in zip(states, lines):
                plain, locked = map(int, line.split())
                self.assertEqual(plain, solver.bounded_substring_distance(
                    state.word, goal, [False] * len(state.word), len(goal)))
                self.assertEqual(locked, solver.bounded_substring_distance(
                    state.word, goal, state.frozen, len(goal), allow_one_unlock=True))

    def test_native_result_is_replayed_and_cast_limit_is_respected(self):
        roots = self.roots('abz', ('abc',), ('change',))
        result = native_search.discover('abz', roots, solver, binary=self.binary, width=64)
        self.assertEqual(result.states[-1].word, 'abz')
        self.assertEqual(solver.spell_count(result.states[-1].hist), 1)
        limited = native_search.discover('abz', roots, solver, binary=self.binary,
                                         width=64, max_cost=0)
        self.assertIsNone(limited.states)
        self.assertTrue(limited.available)

    def test_illegal_native_output_fails_instead_of_becoming_a_solution(self):
        roots = self.roots('abz', ('abc',))
        with mock.patch.object(native_search, '_run', return_value='FOUND 0 1 1 1\nchange_0_z_abz\n'):
            with self.assertRaisesRegex(RuntimeError, 'illegal cast'):
                native_search.discover('abz', roots, solver, binary=self.binary)

    def test_missing_compiler_and_unsupported_inputs_leave_python_available(self):
        roots = self.roots('abz', ('abc',))
        with mock.patch.object(native_search.shutil, 'which', return_value=None):
            result = native_search.discover('abz', roots, solver)
        self.assertFalse(result.available)
        self.assertIsNone(result.states)
        self.assertFalse(native_search.supported('z' * 64, roots, solver))


if __name__ == '__main__':
    unittest.main()
