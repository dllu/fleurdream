import itertools
import contextlib
import heapq
import io
import random
import unittest
from dataclasses import replace
from unittest import mock

import solver


USED_FIELDS = tuple(name for name in solver.State.__dataclass_fields__ if name.startswith('used_'))


def state(word, enabled=(), frozen=None):
    flags = {name: True for name in USED_FIELDS}
    flags.update({'used_' + name: False for name in enabled})
    return solver.State(
        word, [False] * len(word) if frozen is None else frozen,
        hist=[word], **flags,
    )


def set_goal(goal):
    solver.target = goal
    solver.tgt_abc = ''.join(c for c in solver.abc if c in goal)


def strings(alphabet, max_length):
    return [''.join(chars) for n in range(max_length + 1)
            for chars in itertools.product(alphabet, repeat=n)]


def reference_distance(source, goal, costs, insert=1, delete=True, substitute=True,
                       insertion_letters=None, substitution_letters=None):
    """Deliberately simple full-matrix reference, independent of the bit code."""
    inf = float('inf')
    dp = [[0] * (len(goal) + 1) for _ in range(len(source) + 1)]
    for j, t in enumerate(goal, 1):
        dp[0][j] = dp[0][j - 1] + (
            insert if insertion_letters is None or t in insertion_letters else inf
        )
    for i, c in enumerate(source, 1):
        for j, t in enumerate(goal, 1):
            dp[i][j] = min(
                dp[i][j - 1] + (
                    insert if insertion_letters is None or t in insertion_letters else inf
                ),
                dp[i - 1][j] + (costs[i - 1] if delete else inf),
                dp[i - 1][j - 1] + (
                    0 if c == t else costs[i - 1]
                    if substitute and (substitution_letters is None or t in substitution_letters[c])
                    else inf
                ),
            )
    return min(row[-1] for row in dp)


def reference_shortest(goal, roots):
    """Independent uniform-cost oracle over the legal transition generators."""
    set_goal(goal)
    queue = []
    order = itertools.count()
    best = {}
    for root in roots:
        cost = sum(2 if entry.startswith('combine_drop') else 1
                   for entry in root.hist if '_' in entry)
        heapq.heappush(queue, (cost, next(order), root))
    while queue:
        cost, _, current = heapq.heappop(queue)
        key = current.key()
        if key in best and best[key] <= cost:
            continue
        best[key] = cost
        if goal in current.word:
            return cost
        for child in current.generate_states_unlock():
            edge = 2 if child.hist[-1].startswith('combine_drop') else 1
            heapq.heappush(queue, (cost + edge, next(order), child))
    return None


class SolverTests(unittest.TestCase):
    def setUp(self):
        self.settings = solver.target, solver.tgt_abc, solver.abc, solver.max_len, solver.sources

    def tearDown(self):
        solver.target, solver.tgt_abc, solver.abc, solver.max_len, solver.sources = self.settings

    def test_bit_distance_against_reference_exhaustively(self):
        words = strings('ab', 3)
        for source in words:
            for goal in words:
                for frozen in itertools.product((False, True), repeat=len(source)):
                    costs = [float('inf') if f else 1 for f in frozen]
                    for insert, delete, substitute in itertools.product((False, True), repeat=3):
                        expected = reference_distance(
                            source, goal, costs, 1 if insert else float('inf'),
                            delete, substitute,
                        )
                        for cap in range(4):
                            actual = solver.bounded_substring_distance(
                                source, goal, frozen, cap,
                                allow_insert=insert, allow_delete=delete,
                                allow_substitute=substitute,
                            )
                            self.assertEqual(actual, min(cap + 1, expected),
                                             (source, goal, frozen, insert, delete, substitute, cap))

    def test_weighted_distance_against_reference(self):
        words = strings('ab', 3)
        for source in words:
            for goal in words:
                for costs in itertools.product((1, 2, 101), repeat=len(source)):
                    self.assertEqual(
                        solver.generalized_levenshtein_custom_cost_dp(source, goal, costs),
                        reference_distance(source, goal, costs),
                        (source, goal, costs),
                    )
        self.assertEqual(solver.generalized_levenshtein_distance('a', 'ab'), 1)

    def test_restricted_edit_alphabets_against_reference(self):
        words = strings('ab', 2)
        alphabets = ('', 'a', 'b', 'ab')
        for source in words:
            for goal in words:
                for frozen in itertools.product((False, True), repeat=len(source)):
                    costs = [float('inf') if f else 1 for f in frozen]
                    for inserted, sub_a, sub_b in itertools.product(alphabets, repeat=3):
                        substitutions = {'a': sub_a, 'b': sub_b}
                        insertion_mask = sum(1 << j for j, c in enumerate(goal, 1) if c in inserted)
                        substitution_masks = {
                            c: sum(1 << j for j, t in enumerate(goal, 1) if t in replacements)
                            for c, replacements in substitutions.items()
                        }
                        for delete in (False, True):
                            expected = reference_distance(
                                source, goal, costs, delete=delete,
                                insertion_letters=inserted, substitution_letters=substitutions,
                            )
                            for cap in range(3):
                                actual = solver.bounded_substring_distance(
                                    source, goal, frozen, cap, allow_delete=delete,
                                    insertion_mask=insertion_mask,
                                    substitution_masks=substitution_masks,
                                )
                                self.assertEqual(actual, min(cap + 1, expected),
                                                 (source, goal, frozen, inserted, substitutions, delete, cap))

    def test_pruning_preserves_all_small_reachable_paths(self):
        # Exhaust every ordering of these spells, including neutral setup moves,
        # then check every ancestor against each reachable terminal word.
        solver.abc = 'abcd'
        cases = [
            ('ab', ('alter', 'add')),
            ('ad', ('adjacent', 'alter')),
            ('ab', ('move', 'swap')),
            ('am', ('duplicate', 'alter')),
            ('bc', ('subtract', 'add')),
            ('mmm', ('most_frequent', 'chop')),
            ('cad', ('chop', 'add', 'alter')),
            ('abcd', ('drop', 'move', 'swap_adjacent')),
        ]
        checked = 0
        for word, enabled in cases:
            for frozen in ([False] * len(word), [i == 0 for i in range(len(word))]):
                root = state(word, enabled, frozen)
                stack = [(root,)]
                while stack:
                    path = stack.pop()
                    children = list(path[-1].generate_states_unlock())
                    if children:
                        stack.extend(path + (child,) for child in children)
                        continue
                    set_goal(path[-1].word)
                    for index, ancestor in enumerate(path):
                        fresh = replace(ancestor, score_=None)
                        feasible, lower_bound = fresh.search_bounds()
                        self.assertTrue(feasible,
                                        (enabled, path[-1].hist, ancestor.hist, fresh.dist))
                        self.assertLessEqual(lower_bound, len(path) - index - 1,
                                             (path[-1].hist, ancestor.hist, lower_bound))
                        checked += 1
        self.assertGreater(checked, 1000)

    def test_pruned_search_matches_exhaustive_search(self):
        solver.abc = 'abcd'
        for root in (state('ad', ('alter', 'add')), state('abc', ('move', 'chop'))):
            for goal in strings('ab', 3)[1:]:
                pruned = solver.solve(goal, [root])
                exhaustive = solver.solve(goal, [root], prune=False)
                self.assertEqual(pruned.state is not None, exhaustive.state is not None,
                                 (root.word, goal))

    def test_pruning_preserves_longer_paths_with_frozen_letters(self):
        solver.abc = 'abcdenmw'
        rng = random.Random(0)
        local_spells = (
            'add', 'alter', 'adjacent', 'change', 'chop', 'drop', 'duplicate',
            'most_frequent', 'move', 'subtract', 'swap', 'swap_adjacent',
        )
        for _ in range(300):
            word = ''.join(rng.choice(solver.abc) for _ in range(rng.randint(2, 7)))
            enabled = rng.sample(local_spells, rng.randint(4, 7))
            path = [state(word, enabled, [rng.random() < 0.25 for _ in word])]
            while True:
                children = list(path[-1].generate_states_unlock())
                if not children:
                    break
                path.append(rng.choice(children))
            set_goal(path[-1].word)
            for index, ancestor in enumerate(path):
                feasible, lower_bound = replace(ancestor, score_=None).search_bounds()
                self.assertTrue(feasible,
                                (path[-1].hist, ancestor.hist))
                self.assertLessEqual(lower_bound, len(path) - index - 1,
                                     (path[-1].hist, ancestor.hist, lower_bound))

    def test_bulk_spells_and_recast_disable_local_pruning(self):
        for spell in ('combine', 'spellcheck', 'find_and_replace', 'unlock_letter', 'recast_spell'):
            set_goal('zzzzzz')
            s = state('a', (spell,))
            self.assertIsNone(s.edit_budget())
            self.assertTrue(s.can_reach())
        recast = state('a', ('recast_spell',))
        recast.score()
        self.assertGreater(recast.dist, 2)
        self.assertTrue(any(not s.used_combine for s in recast.recast_spell()))

    def test_unlock_is_available_without_distance_gate_or_prior_move(self):
        set_goal('aaaaaaa')
        root = state('bbbb', ('unlock_letter',), [True] * 4)
        root.score()
        self.assertGreater(root.dist, 3)
        children = list(root.generate_states_unlock())
        self.assertEqual(len(children), 4)
        self.assertTrue(all(sum(s.frozen) == 3 and s.used_unlock_letter for s in children))

    def test_add_retains_letters_outside_the_target(self):
        set_goal('ab')
        root = state('a', ('add',))
        self.assertTrue(any(s.word == 'za' for s in root.generate_states_unlock()))

    def test_swap_pairs_are_unique(self):
        root = state('abcd', ('swap',))
        children = list(root.swap())
        self.assertEqual(len(children), 6)
        self.assertEqual(len({s.key() for s in children}), 6)
        self.assertEqual(len(list(state('abcd', ('swap', 'swap_adjacent')).swap())), 6)
        adjacent = list(state('abcd', ('swap_adjacent',)).swap_adjacent())
        self.assertEqual(len(adjacent), 3)
        self.assertEqual(len({s.key() for s in adjacent}), 3)

    def test_finishing_insertion_obeys_rules_and_position_limit(self):
        self.assertEqual(solver.finishing_insertion('ombine', 'combine', 33), (0, 'c'))
        self.assertIsNone(solver.finishing_insertion('combombine', 'combine', 33))
        self.assertIsNone(solver.finishing_insertion('xxxombine', 'combine', 2))
        self.assertIsNone(solver.finishing_insertion('combine', 'combine', 33))
        self.assertEqual(solver.finishing_insertion('', 'a', 0), (0, 'a'))

    def test_drop_and_chop_do_not_repair_internal_gaps(self):
        set_goal('ab')
        self.assertFalse(state('axb', ('drop', 'chop')).can_reach())
        set_goal('r')
        self.assertFalse(state('f', ('alter',)).can_reach())

    def test_pruning_does_not_use_the_ordering_hint(self):
        set_goal('a')
        s = state('b', ('change',))
        s.dist = 999
        s.score_ = -999
        self.assertTrue(s.can_reach())

    def test_heuristic_respects_frozen_letter_supply_and_change_restriction(self):
        set_goal('knock')
        s = state('knoc', ('duplicate',), [True, False, False, False])
        s.score()
        self.assertGreater(s.dist, 1)
        set_goal('bb')
        s = state('ab', ('change',))
        s.score()
        self.assertGreater(s.dist, 1)

    def test_bulk_and_local_expansions_preserve_all_successors(self):
        solver.sources = ['cd']
        set_goal('knock')
        s = state('ab', ('combine', 'spellcheck', 'find_and_replace', 'move', 'change'))
        full = {child.key() for child in s.generate_states_unlock()}
        split = {child.key() for child in s.generate_states_unlock(include_bulk=False)}
        split.update(child.key() for child in s.generate_bulk_states())
        self.assertEqual(full, split)

    def test_deferred_search_finds_bulk_only_solutions(self):
        solver.sources = ['cd']
        for word, spell, goal in (
            ('ab', 'combine', 'abcd'),
            ('amoveb', 'spellcheck', 'ab'),
            ('aaaa', 'find_and_replace', 'bbbb'),
        ):
            result = solver.solve(goal, [state(word, (spell,))])
            self.assertIsNotNone(result.state, (word, spell, goal))
            self.assertIn(goal, result.state.word)
            self.assertGreater(result.bulk_expanded, 0)

    def test_local_setup_before_a_deferred_bulk_spell(self):
        result = solver.solve(
            'axb', [state('axmovzb', ('change', 'spellcheck'))], preferred_rounds=1,
        )
        self.assertIsNotNone(result.state)
        self.assertEqual(len(result.state.hist) - 1, 2)
        self.assertTrue(result.state.hist[1].startswith('change_'))
        self.assertTrue(result.state.hist[2].startswith('spellcheck_'))

    def test_bulk_before_local_and_bulk_turn_before_preferred_turn(self):
        solver.sources = ['c']
        result = solver.solve(
            'abz', [state('ab', ('combine', 'change'))], preferred_rounds=0,
        )
        self.assertIsNotNone(result.state)
        self.assertEqual(len(result.state.hist) - 1, 2)
        self.assertTrue(result.state.hist[1].startswith('combine_'))
        self.assertTrue(result.state.hist[2].startswith('change_'))

    def test_bulk_turns_run_while_preferred_work_remains(self):
        solver.sources = ['cd']
        busy = state('aaaaaa', ('change', 'adjacent', 'move', 'swap', 'duplicate'))
        result = solver.solve(
            'abcd', [busy, state('ab', ('combine',))],
            prune=False, preferred_rounds=2,
        )
        self.assertIsNotNone(result.state)
        self.assertEqual(result.state.hist[0], 'ab')
        self.assertGreater(result.bulk_expanded, 0)
        self.assertLessEqual(result.expanded, 3)

    def test_source_words_get_fair_turns(self):
        result = solver.solve('z', [state('aaaa', ('combine', 'adjacent', 'duplicate')),
                                    state('q', ('change',))])
        self.assertIsNotNone(result.state)
        self.assertEqual(result.state.hist[0], 'q')
        self.assertLessEqual(result.expanded, 2)

    def replay(self, goal, solution):
        set_goal(goal)
        s = solver.State(solution.hist[0], [False] * len(solution.hist[0]), hist=[solution.hist[0]])
        for step in solution.hist[1:]:
            s = next(child for child in s.generate_states_unlock() if child.hist[-1] == step)
        self.assertEqual(s.key(), solution.key())
        self.assertIn(goal, s.word)

    def build_path(self, root, prefixes):
        path = [root]
        for prefix in prefixes:
            path.append(next(child for child in path[-1].generate_states_unlock()
                             if child.hist[-1].startswith(prefix)))
        return path

    def assert_legal_path(self, path):
        for before, after in zip(path, path[1:]):
            self.assertTrue(any(child.key() == after.key() and child.hist == after.hist
                                for child in before.generate_states_unlock()), after.hist)

    def test_cashmoney_removes_the_unused_add(self):
        result = solver.solve('cashmoney', first_solution=True)
        self.assertIsNotNone(result.state)
        self.assertEqual(len(result.state.hist) - 1, 4)
        self.assertEqual(result.removed_steps, 1)
        self.assertFalse(result.state.used_add)
        self.assertFalse(any(step.startswith('add_') for step in result.state.hist[1:]))
        self.replay('cashmoney', result.state)

    def test_shortening_rebases_indices_after_an_unused_insertion(self):
        set_goal('ac')
        path = self.build_path(state('ab', ('add', 'change')),
                               ('add_0_z_', 'change_2_c_'))
        shorter = solver._shorten_path(path, 'ac')
        self.assertEqual(shorter[-1].hist, ['ab', 'change_1_c_ac'])
        self.assertFalse(shorter[-1].used_add)
        self.assert_legal_path(shorter)

    def test_shortening_rebases_moves_and_swaps(self):
        set_goal('adcb')
        path = self.build_path(state('abcd', ('add', 'move', 'swap')),
                               ('add_0_z_', 'move_4_2_', 'swap_3b_4c_'))
        shorter = solver._shorten_path(path, 'adcb')
        self.assertEqual(shorter[-1].hist, ['abcd', 'move_3_1_adbc', 'swap_2b_3c_adcb'])
        self.assert_legal_path(shorter)

    def test_replay_reorders_swap_indices_after_rebasing(self):
        set_goal('bd')
        before = state('dabc', ('swap',))
        after = next(child for child in before.swap()
                     if child.hist[-1].startswith('swap_0d_3c_'))
        current = state('abcd', ('swap',))
        ids = [(0, i) for i in range(4)]
        replayed = solver._replay_step(before, after, [ids[3]] + ids[:3], current, ids, 1)
        self.assertIsNotNone(replayed)
        self.assertEqual(replayed[0].hist, ['abcd', 'swap_2c_3d_abdc'])
        self.assert_legal_path([current, replayed[0]])

    def test_shortening_tracks_the_unfrozen_original_after_duplicate(self):
        set_goal('cb')
        path = self.build_path(state('ab', ('duplicate', 'change')),
                               ('dup_0_a_', 'change_1_c_'))
        shorter = solver._shorten_path(path, 'cb')
        self.assertEqual(shorter[-1].hist, ['ab', 'change_0_c_cb'])
        self.assertFalse(shorter[-1].used_duplicate)
        self.assert_legal_path(shorter)

    def test_shortening_rebases_through_spellcheck(self):
        set_goal('ac')
        path = self.build_path(state('amoveb', ('add', 'spellcheck', 'change')),
                               ('add_0_z_', 'spellcheck_', 'change_2_c_'))
        shorter = solver._shorten_path(path, 'ac')
        self.assertEqual(shorter[-1].hist, ['amoveb', 'spellcheck_ab', 'change_1_c_ac'])
        self.assert_legal_path(shorter)

    def test_shortening_keeps_padding_needed_for_chop_and_absent_letter_rule(self):
        set_goal('xbcd')
        path = self.build_path(state('abcdxx', ('add', 'chop', 'change')),
                               ('add_6_z_', 'chop_0_', 'change_0_x_'))
        shorter = solver._shorten_path(path, 'xbcd')
        self.assertEqual(shorter[-1].hist, path[-1].hist)
        self.assert_legal_path(shorter)

    def test_shortening_keeps_noop_freezing_needed_for_find_and_replace(self):
        set_goal('abb')
        path = self.build_path(state('aaa', ('move', 'find_and_replace')),
                               ('move_0_0_', 'fandr_a_b_'))
        shorter = solver._shorten_path(path, 'abb')
        self.assertEqual(shorter[-1].hist, path[-1].hist)
        self.assert_legal_path(shorter)

    def test_shortening_keeps_the_combine_absent_letter_setup(self):
        set_goal('combine')
        path = self.build_path(state('honeycombbmineraleaf', ('swap', 'move', 'alter', 'add')),
                               ('swap_9b_10m_', 'move_6_8_', 'alter_5_n_', 'add_8_c_'))
        shorter = solver._shorten_path(path, 'combine')
        self.assertEqual(shorter[-1].hist, path[-1].hist)
        self.assert_legal_path(shorter)

    def test_replay_handles_all_spell_formats_and_ambiguous_drops(self):
        solver.sources = ['cd']
        solver.abc = 'abcde'
        set_goal('ace')
        enabled = [name[5:] for name in USED_FIELDS]
        roots = [state('ab', enabled), state('', enabled),
                 replace(state('aaa', enabled, [True, False, True]),
                         used_add=True, used_swap=True)]
        seen = set()
        for root in roots:
            ids = [(0, i) for i in range(len(root.word))]
            for child in root.generate_states_unlock():
                seen.add(solver._operation(child)[0])
                replayed = solver._replay_step(root, child, ids, root, ids, 1)
                self.assertIsNotNone(replayed, child.hist)
                self.assertEqual(replayed[0].key(), child.key(), child.hist)
                self.assertEqual(len(replayed[1]), len(child.word))
        self.assertEqual(seen, {'add', 'alter', 'adj+', 'adj-', 'change', 'chop', 'drop',
                                'dup', 'mostf', 'move', 'sub', 'swap', 'swap_adj',
                                'combine', 'spellcheck', 'fandr', 'recast', 'unlock'})

    def test_freedom_is_reproducible_and_legal(self):
        first = solver.solve('FREEDOM', first_solution=True)
        second = solver.solve('freedom', first_solution=True)
        self.assertIsNotNone(first.state)
        self.assertLessEqual(len(first.state.hist) - 1, 5)
        self.assertEqual(first.state.hist, second.state.hist)
        self.replay('freedom', first.state)

    def test_combine_is_found_and_legal(self):
        result = solver.solve('combine')
        self.assertIsNotNone(result.state)
        self.assertLessEqual(len(result.state.hist) - 1, 4)
        self.replay('combine', result.state)

    def test_combine_user_route_survives_pruning(self):
        set_goal('combine')
        s = state('honeycombbmineraleaf', ('swap', 'move', 'alter', 'add'))
        for prefix in ('swap_9b_10m_', 'move_6_8_', 'alter_5_n_', 'add_8_c_'):
            self.assertTrue(s.can_reach(), s.hist)
            s = next(child for child in s.generate_states_unlock()
                     if child.hist[-1].startswith(prefix))
        self.assertIn('combine', s.word)

    def test_knock_is_found_quickly_in_four_spells(self):
        result = solver.solve('knock', first_solution=True)
        self.assertIsNotNone(result.state)
        self.assertLessEqual(len(result.state.hist) - 1, 4)
        self.assertLess(result.expanded, 100)
        self.replay('knock', result.state)

    def test_knock_user_route_survives_pruning(self):
        set_goal('knock')
        s = state('lilachangerbloom', ('change', 'adjacent', 'move', 'swap'))
        for prefix in ('change_5_k_', 'adj-_2_l_', 'move_7_3_', 'swap_4a_13o_'):
            self.assertTrue(s.can_reach(), s.hist)
            s = next(child for child in s.generate_states_unlock()
                     if child.hist[-1].startswith(prefix))
        self.assertIn('knock', s.word)

    def test_spellcheck_preparations_create_new_removable_spells(self):
        set_goal('ab')
        root = state('ayopb', ('alter', 'spellcheck'))
        prepared = list(root.spellcheck_preparations())
        altered, checked = next((a, c) for a, c in prepared if c.word == 'ab')
        self.assertEqual(altered.word, 'ahopb')
        self.assert_legal_path([root, altered, checked])
        self.assertEqual(list(state('walkb', ('alter', 'spellcheck')).spellcheck_preparations()), [])

    def test_spellcheck_preparations_respect_availability_frozen_letters_and_positions(self):
        for root in (
            state('yop', ('alter',)),
            state('yop', ('spellcheck',)),
            state('yop', ('alter', 'spellcheck'), [True, False, False]),
            state('x' * (solver.max_len + 1) + 'yop', ('alter', 'spellcheck')),
        ):
            self.assertEqual(list(root.spellcheck_preparations()), [])

    def test_preparation_stops_at_a_goal_before_spellcheck_erases_it(self):
        set_goal('hop')
        root = state('yop', ('alter', 'spellcheck'))
        seeds = solver._spellcheck_seeds([solver._PendingState(root)])
        self.assertTrue(seeds)
        self.assertEqual(seeds[0].state.word, 'hop')
        self.assertFalse(seeds[0].state.used_spellcheck)
        self.assert_legal_path([root, seeds[0].state])

    def test_recastspell_user_route_survives_pruning(self):
        set_goal('recastspell')
        root = state('redpedalwalker', (
            'combine', 'alter', 'spellcheck', 'swap', 'move', 'duplicate',
            'add', 'adjacent', 'subtract',
        ))
        path = self.build_path(root, (
            'combine_10_cespyoped_', 'alter_14_h_', 'spellcheck_',
            'swap_9a_11e_', 'move_0_8_', 'dup_12_s_', 'add_13_t_',
            'adj+_19_k_', 'sub_17_d_',
        ))
        for ancestor in path:
            self.assertTrue(ancestor.can_reach(), ancestor.hist)
        self.assertIn('recastspell', path[-1].word)
        self.assert_legal_path(path)

    def test_recastspell_is_found_in_nine_spells(self):
        result = solver.solve('RECASTSPELL', first_solution=True)
        self.assertIsNotNone(result.state)
        self.assertLessEqual(len(result.state.hist) - 1, 9)
        self.assertLess(result.expanded, 500)
        self.replay('recastspell', result.state)

    def test_actual_cast_costs_include_recast_and_bundled_drop(self):
        self.assertEqual(solver.spell_count(['ab', 'recast_combine_ab',
                                            'combine_drop0_2_c_abc']), 3)
        root = state('ab', ('recast_spell',))
        recast = next(s for s in root.recast_spell() if not s.used_change)
        self.assertEqual(sum(root.useds()), sum(recast.useds()))
        node = solver._PendingState(recast, parent=solver._PendingState(root))
        self.assertEqual(node.cost, 1)

    def test_anytime_improves_a_macro_solution_and_proves_the_shorter_one(self):
        solver.sources = ['xc', 'c']
        root = state('ab', ('combine', 'drop'))
        events = []
        result = solver.solve('abc', [root], preferred_rounds=0,
                              on_solution=lambda s, c: events.append((c, s.hist.copy())))
        self.assertEqual([cost for cost, _ in events], [2, 1])
        self.assertTrue(events[0][1][-1].startswith('combine_drop0_'))
        self.assertTrue(events[1][1][-1].startswith('combine_'))
        self.assertEqual(result.cost, 1)
        self.assertTrue(result.optimal)
        self.assertEqual(result.stop_reason, 'optimal')
        self.assertEqual(result.improvements, 2)
        self.assertGreater(result.bound_pruned, 0)

    def test_first_solution_mode_does_not_claim_optimality(self):
        solver.sources = ['xc', 'c']
        result = solver.solve('abc', [state('ab', ('combine', 'drop'))],
                              preferred_rounds=0, first_solution=True)
        self.assertEqual(result.cost, 2)
        self.assertFalse(result.optimal)
        self.assertEqual(result.stop_reason, 'first_solution')

    def test_anytime_initializes_remaining_sources_after_a_finishing_add(self):
        events = []
        result = solver.solve('abc', [state('ab', ('add',)), state('abc')],
                              on_solution=lambda s, c: events.append(c))
        self.assertEqual(events, [1, 0])
        self.assertEqual(result.cost, 0)
        self.assertTrue(result.optimal)

    def test_cheaper_arrival_reopens_a_previously_seen_state(self):
        expensive = state('ab', ('change',))
        expensive.hist = ['cab', 'drop_n_ab']
        cheap = state('ab', ('change',))
        self.assertEqual(expensive.key(), cheap.key())
        result = solver.solve('az', [expensive, cheap])
        self.assertEqual(result.cost, 1)
        self.assertEqual(result.state.hist[0], 'ab')
        self.assertGreaterEqual(result.reopened, 1)
        self.assertTrue(result.optimal)

    def test_safe_bounds_are_independent_of_the_ordering_heuristic(self):
        solver.sources = ['xc', 'c']
        with mock.patch.object(solver.State, 'heuristic_distance', return_value=1000000):
            result = solver.solve('abc', [state('ab', ('combine', 'drop'))],
                                  preferred_rounds=0)
        self.assertEqual(result.cost, 1)
        self.assertTrue(result.optimal)

    def test_bulk_and_recast_use_conservative_remaining_cost_bounds(self):
        set_goal('zzzzzz')
        for spell in ('combine', 'spellcheck', 'find_and_replace', 'recast_spell', 'unlock_letter'):
            self.assertEqual(state('a', (spell,)).search_bounds(), (True, 1))
        set_goal('aaa')
        self.assertEqual(state('aaa').search_bounds(), (True, 0))

    def test_anytime_matches_uniform_cost_search_on_small_graphs(self):
        solver.abc = 'abcd'
        solver.sources = ['xc', 'c']
        roots = (state('ad', ('alter', 'add')), state('abc', ('move', 'chop')),
                 state('ab', ('combine', 'drop')), state('a', ('change', 'recast_spell')))
        for root in roots:
            for goal in ('a', 'ab', 'ba', 'abc', 'bc', 'dd', 'zz'):
                expected = reference_shortest(goal, [root])
                result = solver.solve(goal, [root])
                self.assertEqual(result.cost, expected, (root.word, goal))
                self.assertEqual(result.optimal, expected is not None, (root.word, goal))
                self.assertFalse(result.interrupted)

    def test_interrupt_keeps_an_incumbent_without_a_false_optimality_claim(self):
        solver.sources = ['xc', 'c']
        def stop(state, cost):
            raise KeyboardInterrupt
        result = solver.solve('abc', [state('ab', ('combine', 'drop'))],
                              preferred_rounds=0, on_solution=stop)
        self.assertEqual(result.cost, 2)
        self.assertIn('abc', result.state.word)
        self.assertTrue(result.interrupted)
        self.assertFalse(result.optimal)
        self.assertEqual(result.stop_reason, 'interrupted')

    def test_interrupt_during_cleanup_preserves_the_raw_legal_solution(self):
        with mock.patch.object(solver, '_shorten_path', side_effect=KeyboardInterrupt):
            result = solver.solve('ac', [state('ab', ('change',))])
        self.assertEqual(result.cost, 1)
        self.assertIn('ac', result.state.word)
        self.assertTrue(result.interrupted)
        self.assertFalse(result.optimal)

    def test_interrupt_before_a_solution_is_distinct_from_exhaustion(self):
        def stop(state):
            raise KeyboardInterrupt
        result = solver.solve('ac', [state('ab', ('change',))], on_progress=stop)
        self.assertIsNone(result.state)
        self.assertIsNone(result.cost)
        self.assertTrue(result.interrupted)
        self.assertEqual(result.stop_reason, 'interrupted')
        exhausted = solver.solve('z', [state('a')])
        self.assertIsNone(exhausted.state)
        self.assertFalse(exhausted.interrupted)
        self.assertEqual(exhausted.stop_reason, 'exhausted')

    def test_cli_continues_by_default_and_supports_first_solution(self):
        solver.sources = ['ab']
        with mock.patch.object(solver, 'solve', wraps=solver.solve) as search:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(solver.main(['ab']), 0)
            self.assertFalse(search.call_args.kwargs['first_solution'])
            self.assertIn('Proved shortest: 0 spell casts', output.getvalue())
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(solver.main(['ab', '--first']), 0)
            self.assertTrue(search.call_args.kwargs['first_solution'])

    def test_most_frequent_handles_empty_and_single_letter_words(self):
        self.assertEqual(list(state('', ('most_frequent',)).generate_states_unlock()), [])
        children = list(state('aa', ('most_frequent',)).generate_states_unlock())
        self.assertEqual(len(children), 3)
        self.assertTrue(all(s.word == 'aaa' for s in children))


if __name__ == '__main__':
    unittest.main()
