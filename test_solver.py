import itertools
import contextlib
import heapq
import io
import random
import unittest
from dataclasses import dataclass, replace
from unittest import mock

import solver


USED_FIELDS = tuple(name for name in solver.State.__dataclass_fields__ if name.startswith('used_'))


@dataclass(frozen=True)
class RegressionConfig:
    flowers: tuple
    spells: frozenset
    max_position: int = 33


# Snapshots: never derive these from mutable solver sources or State defaults.
REGRESSION_FLOWERS = (
    'honeycombbmineraleaf', 'redpedalwalker', 'lilachangerbloom', 'cespyoped',
    'tigerwishgrantbud',
)
REGRESSION_SPELLS = frozenset((
    'alter', 'move', 'add', 'adjacent', 'combine', 'change', 'chop', 'drop',
    'swap', 'subtract', 'duplicate', 'spellcheck', 'recast_spell',
))
REGRESSION_CONFIGS = {
    goal: RegressionConfig(
        REGRESSION_FLOWERS,
        REGRESSION_SPELLS | (frozenset(('swap_adjacent',))
                             if goal in ('mostfrequent', 'spikybetty') else frozenset()),
    )
    for goal in ('freedom', 'combine', 'knock', 'cashmoney', 'recastspell',
                 'mostfrequent', 'spikybetty')
}


def frozen_regression(goal):
    config = REGRESSION_CONFIGS[goal]
    return mock.patch.multiple(solver, sources=list(config.flowers),
                               max_len=config.max_position, abc='abcdefghijklmnopqrstuvwxyz')


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
                       insertion_letters=None, substitution_letters=None,
                       insertion_by_gap=None, free_delete=False, delete_frozen=False):
    """Deliberately simple full-matrix reference, independent of the bit code."""
    inf = float('inf')
    dp = [[0] * (len(goal) + 1) for _ in range(len(source) + 1)]
    initial_letters = insertion_letters if insertion_by_gap is None else insertion_by_gap[0]
    for j, t in enumerate(goal, 1):
        dp[0][j] = dp[0][j - 1] + (
            insert if initial_letters is None or t in initial_letters else inf
        )
    for i, c in enumerate(source, 1):
        letters = insertion_letters if insertion_by_gap is None else insertion_by_gap[i]
        for j, t in enumerate(goal, 1):
            dp[i][j] = min(
                dp[i][j - 1] + (
                    insert if letters is None or t in letters else inf
                ),
                dp[i - 1][j] + ((0 if free_delete else 1 if delete_frozen
                                 else costs[i - 1]) if delete else inf),
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


def reference_distance_with_one_unlock(source, goal, frozen, insert, delete, substitute):
    """Full matrix with an independent unlock-resource dimension."""
    inf = float('inf')
    dp = [[[inf, inf] for _ in range(len(goal) + 1)] for _ in range(len(source) + 1)]
    for i in range(len(source) + 1):
        dp[i][0] = [0, 0]
    for j in range(1, len(goal) + 1):
        dp[0][j] = [j if insert else inf] * 2
    for i, letter in enumerate(source, 1):
        for j, wanted in enumerate(goal, 1):
            for unlocked in range(2):
                choices = []
                if letter == wanted:
                    choices.append(dp[i - 1][j - 1][unlocked])
                elif not frozen[i - 1] and substitute:
                    choices.append(dp[i - 1][j - 1][unlocked] + 1)
                if insert:
                    choices.append(dp[i][j - 1][unlocked] + 1)
                if delete and not frozen[i - 1]:
                    choices.append(dp[i - 1][j][unlocked] + 1)
                if unlocked and frozen[i - 1]:
                    if delete:
                        choices.append(dp[i - 1][j][0] + 2)
                    if substitute:
                        choices.append(dp[i - 1][j - 1][0] + 2)
                dp[i][j][unlocked] = min(choices, default=inf)
    return min(value for row in dp for value in row[-1])


class SolverTests(unittest.TestCase):
    def setUp(self):
        self.settings = solver.target, solver.tgt_abc, solver.abc, solver.max_len, solver.sources

    def tearDown(self):
        solver.target, solver.tgt_abc, solver.abc, solver.max_len, solver.sources = self.settings

    @contextlib.contextmanager
    def regression_config(self, goal):
        config = REGRESSION_CONFIGS[goal.lower()]
        with mock.patch.object(solver, 'sources', list(config.flowers)), \
                mock.patch.object(solver, 'max_len', config.max_position), \
                mock.patch.object(solver, 'abc', 'abcdefghijklmnopqrstuvwxyz'):
            yield [replace(state(word, config.spells), allowed_spells=config.spells,
                           source_words=config.flowers) for word in config.flowers]

    def regression_solve(self, goal, **kwargs):
        with self.regression_config(goal) as roots:
            return solver.solve(goal, roots, **kwargs)

    def replay_regression(self, goal, solution):
        with self.regression_config(goal) as roots:
            root = next(s for s in roots if s.word == solution.hist[0])
            self.replay(goal, solution, root)

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

    def test_one_unlock_bit_distance_against_independent_resource_matrix(self):
        words = strings('ab', 3)
        for source, goal in itertools.product(words, repeat=2):
            for frozen in itertools.product((False, True), repeat=len(source)):
                for insert, delete, substitute in itertools.product((False, True), repeat=3):
                    expected = reference_distance_with_one_unlock(
                        source, goal, frozen, insert, delete, substitute)
                    for cap in range(4):
                        actual = solver.bounded_substring_distance(
                            source, goal, frozen, cap, allow_insert=insert,
                            allow_delete=delete, allow_substitute=substitute, allow_one_unlock=True)
                        self.assertEqual(actual, min(cap + 1, expected),
                                         (source, goal, frozen, insert, delete, substitute, cap))

    def test_single_unlock_hint_does_not_relax_multiple_frozen_positions(self):
        actual = solver.bounded_substring_distance(
            'aax', 'bbx', [True, True, False], 6, allow_insert=False,
            allow_delete=False, allow_one_unlock=True)
        self.assertEqual(actual, 7)
        self.assertEqual(solver.bounded_substring_distance(
            'xbc', 'abc', [True, False, False], 3,
            allow_insert=False, allow_one_unlock=True), 2)

    def test_free_deletion_relaxation_against_reference_exhaustively(self):
        words = strings('ab', 2)
        for source, goal in itertools.product(words, repeat=2):
            for frozen in itertools.product((False, True), repeat=len(source)):
                costs = [float('inf') if f else 1 for f in frozen]
                for insert, substitute in itertools.product((False, True), repeat=2):
                    expected = reference_distance(source, goal, costs,
                                                  insert=1 if insert else float('inf'),
                                                  substitute=substitute, free_delete=True)
                    for cap in range(3):
                        actual = solver.bounded_substring_distance(
                            source, goal, frozen, cap, allow_insert=insert,
                            allow_substitute=substitute, free_delete=True)
                        self.assertEqual(actual, min(cap + 1, expected),
                                         (source, goal, frozen, insert, substitute, cap))

    def test_spellcheck_bound_can_delete_frozen_letters_but_cannot_create_them(self):
        set_goal('ab')
        root = state('achangeb', ('spellcheck',), [True] * 8)
        self.assertEqual(next(root.spellcheck()).word, 'ab')
        self.assertEqual(root.search_bounds(), (True, 1))
        set_goal('z')
        self.assertFalse(state('a', ('spellcheck',)).can_reach())
        root = state('a', ('change', 'unlock_letter'), [True])
        self.assertEqual(root.search_bounds(), (True, 1))
        self.assertEqual(reference_shortest('z', [root]), 2)

    def test_drop_distance_relaxation_against_independent_matrix(self):
        words = strings('ab', 2)
        for source, goal in itertools.product(words, repeat=2):
            for frozen in itertools.product((False, True), repeat=len(source)):
                costs = [float('inf') if f else 1 for f in frozen]
                for insert, substitute in itertools.product((False, True), repeat=2):
                    expected = reference_distance(
                        source, goal, costs, insert=1 if insert else float('inf'),
                        substitute=substitute, delete_frozen=True)
                    for cap in range(3):
                        actual = solver.bounded_substring_distance(
                            source, goal, frozen, cap, allow_insert=insert,
                            allow_substitute=substitute, delete_frozen=True)
                        self.assertEqual(actual, min(cap + 1, expected),
                                         (source, goal, frozen, insert, substitute, cap))

    def test_most_frequent_rejects_ties_in_retracted_showyourworth_route(self):
        root = replace(state('tigerwishgrantbud', ('combine', 'drop', 'find_and_replace',
                                                  'most_frequent')),
                       source_words=('tigerwishgrantbud',))
        combined = next(child for child in root.combine()
                        if child.hist[-1].startswith('combine_drop0_9_'))
        replaced = next(combined.find_and_replace('i', 'o'))
        self.assertEqual({c: replaced.word.count(c) for c in 'ogr'},
                         {'o': 4, 'g': 4, 'r': 4})
        self.assertFalse(any(child.hist[-1].startswith('mostf_')
                             for child in replaced.generate_states_unlock()))

    def test_existing_letter_growth_accounts_for_add_and_change_absence_rule(self):
        set_goal('aaaa')
        root = state('aa', ('add', 'change'))
        self.assertFalse(root.can_reach())
        self.assertIsNone(reference_shortest('aaaa', [root]))
        set_goal('aaa')
        root = state('aa', ('add', 'change', 'duplicate'))
        self.assertTrue(root.can_reach())
        self.assertEqual(reference_shortest('aaa', [root]), 1)

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

    def test_gap_specific_insertions_against_reference(self):
        words = strings('ab', 2)
        for source in words:
            for goal in words:
                for frozen in itertools.product((False, True), repeat=len(source)):
                    costs = [float('inf') if f else 1 for f in frozen]
                    for letters in itertools.product(('', 'a', 'b', 'ab'), repeat=len(source) + 1):
                        masks = [sum(1 << j for j, c in enumerate(goal, 1) if c in alphabet)
                                 for alphabet in letters]
                        for delete, substitute in itertools.product((False, True), repeat=2):
                            expected = reference_distance(
                                source, goal, costs, delete=delete, substitute=substitute,
                                insertion_by_gap=letters,
                            )
                            for cap in range(3):
                                actual = solver.bounded_substring_distance(
                                    source, goal, frozen, cap, allow_delete=delete,
                                    allow_substitute=substitute, insertion_masks=masks,
                                )
                                self.assertEqual(actual, min(cap + 1, expected),
                                                 (source, goal, frozen, letters, delete, substitute, cap))

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
            set_goal('xyz')
            s = state('a', (spell,))
            self.assertIsNone(s.edit_budget())
            self.assertEqual(s.can_reach(), spell not in ('spellcheck', 'unlock_letter'))
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
        root = state('axb', ('drop', 'chop'))
        self.assertIsNone(reference_shortest('ab', [root]))
        self.assertIsNone(solver.solve('ab', [root]).state)
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

    def test_duplicate_hint_cannot_supply_a_remote_letter(self):
        set_goal('abe')
        remote = state('exab', ('duplicate',))
        self.assertGreater(remote.heuristic_distance(), 1)
        movable = state('exab', ('duplicate', 'move'))
        self.assertEqual(movable.heuristic_distance(), 1)
        recastable = state('exab', ('duplicate', 'recast_spell'))
        self.assertEqual(recastable.heuristic_distance(), 1)
        restored = next(s for s in recastable.recast_spell() if not s.used_move)
        self.assertEqual(solver.spell_count(restored.hist), 1)
        self.assertTrue(any('abe' in s.word for s in restored.generate_states_unlock()))
        set_goal('abee')
        self.assertEqual(state('abe', ('duplicate',)).heuristic_distance(), 1)
        self.assertGreater(state('abe', ('duplicate',), [False, False, True]).heuristic_distance(), 1)

    def test_regression_configs_do_not_follow_live_defaults_or_flowers(self):
        defaults = solver.State.__init__.__defaults__
        changed = (True,) * len(USED_FIELDS) + defaults[len(USED_FIELDS):]
        with mock.patch.object(solver, 'sources', ['z']), \
                mock.patch.object(solver.State.__init__, '__defaults__', changed):
            self.assertTrue(solver.State('a', [False]).used_move)
            with self.regression_config('spikybetty') as roots:
                config = REGRESSION_CONFIGS['spikybetty']
                self.assertEqual(tuple(solver.sources), config.flowers)
                self.assertEqual(tuple(s.word for s in roots), config.flowers)
                for root in roots:
                    self.assertEqual(root.allowed_spells, config.spells)
                    self.assertEqual(root.source_words, config.flowers)
                    for name in USED_FIELDS:
                        self.assertEqual(getattr(root, name), name[5:] not in config.spells)
                self.assertFalse(roots[0].used_move)
                self.assertTrue(roots[0].used_find_and_replace)
            with self.regression_config('knock') as roots:
                self.assertTrue(roots[0].used_swap_adjacent)

    def test_finishing_insertion_uses_a_real_copy_and_move(self):
        set_goal('abac')
        for enabled, cost in ((('duplicate', 'move'), 2),
                              (('duplicate', 'recast_spell'), 3)):
            root = state('abc', enabled)
            plan = root.finish_with_insertion()
            self.assertIsNotNone(plan)
            self.assertEqual(solver.spell_count(plan[-1].hist), cost)
            self.assertIn('abac', plan[-1].word)
            self.assert_legal_path([root] + plan)
            self.assertIsNone(root.finish_with_insertion(max_casts=cost - 1))

    def test_finishing_insertion_prefers_one_cast_and_respects_frozen_donors(self):
        set_goal('abee')
        root = state('abe', ('duplicate',))
        plan = root.finish_with_insertion(max_casts=1)
        self.assertIsNotNone(plan)
        self.assertEqual(solver.spell_count(plan[-1].hist), 1)
        self.assertTrue(plan[-1].used_duplicate)
        self.assert_legal_path([root] + plan)
        self.assertIsNone(state('abe', ('duplicate',), [False, False, True]).finish_with_insertion())
        set_goal('abe')
        root = state('exab', ('move',))
        plan = root.finish_with_insertion(max_casts=1)
        self.assertIsNotNone(plan)
        self.assertEqual(solver.spell_count(plan[-1].hist), 1)
        self.assert_legal_path([root] + plan)

    def test_finishing_insertion_respects_position_limit(self):
        set_goal('abac')
        solver.max_len = 0
        self.assertIsNone(state('abc', ('duplicate', 'move')).finish_with_insertion())
        set_goal('aa')
        root = state('a', ('duplicate',))
        plan = root.finish_with_insertion()
        self.assertIsNotNone(plan)
        self.assert_legal_path([root] + plan)

    def test_add_finish_can_first_remove_or_change_an_existing_letter(self):
        cases = (
            ('brigexxxxd', 'bridge', 'chop'),
            ('briged', 'bridge', 'drop'),
            ('cxxombine', 'combine', 'change'),
            ('cxxombine', 'combine', 'alter'),
            ('cxxombine', 'combine', 'subtract'),
            ('ccxxombine', 'combine', 'find_and_replace'),
            ('addxxbrige', 'bridge', 'spellcheck'),
        )
        for word, goal, preparation in cases:
            with self.subTest(preparation=preparation):
                set_goal(goal)
                root = state(word, ('add', preparation))
                self.assertIsNone(root.finish_with_add())
                plan = root.finish_with_insertion(max_casts=2)
                self.assertIsNotNone(plan)
                self.assertEqual(solver.spell_count(plan[-1].hist), 2)
                self.assertIn(goal, plan[-1].word)
                self.assert_legal_path([root] + plan)
                self.assertIsNone(root.finish_with_insertion(max_casts=1))

    def test_add_finish_can_restore_spent_add_but_respects_blacklists(self):
        set_goal('acb')
        root = state('ab', ('recast_spell',))
        plan = root.finish_with_insertion(max_casts=2)
        self.assertIsNotNone(plan)
        self.assertEqual(solver.spell_count(plan[-1].hist), 2)
        self.assert_legal_path([root] + plan)
        blocked = replace(root, allowed_spells=frozenset(('recast_spell',)))
        self.assertIsNone(blocked.finish_with_insertion())

    def test_finishing_copy_unlocks_a_frozen_repeated_letter(self):
        goal = 'd' * 11 + 'ie'
        set_goal(goal)
        root = state('d' * 10 + 'ie', ('duplicate', 'unlock_letter'), [True] * 10 + [False] * 2)
        plan = root.finish_with_insertion(max_casts=2)
        self.assertIsNotNone(plan)
        self.assertEqual([solver.command_spells(s.hist[-1])[0] for s in plan],
                         ['unlock_letter', 'duplicate'])
        self.assertIn(goal, plan[-1].word)
        self.assert_legal_path([root] + plan)
        self.assertIsNone(root.finish_with_insertion(max_casts=1))
        self.assertEqual(solver.solve(goal, [root]).cost, 2)

    def test_unlocked_finishing_move_and_one_cast_precedence(self):
        set_goal('aabc')
        root = state('abca', ('move', 'unlock_letter'), [True, False, False, True])
        plan = root.finish_with_insertion()
        self.assertEqual(len(plan), 2)
        self.assertEqual(plan[-1].word, 'aabc')
        self.assert_legal_path([root] + plan)
        root = state('abca', ('move', 'duplicate', 'unlock_letter'), [True, False, False, False])
        plan = root.finish_with_insertion()
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[-1].word, 'aabc')

    def test_bridge_progression_profile_does_not_need_later_spells(self):
        spells = frozenset(('alter', 'move', 'add', 'change', 'chop', 'drop', 'swap'))
        flowers = ('redpedalwalker', 'lilachangerbloom', 'cespyoped', 'tigerwishgrantbud')
        result = solver.solve('bridge', allowed_spells=spells, source_words=flowers,
                              first_solution=True)
        self.assertEqual(result.cost, 4)
        self.assertLessEqual(result.expanded, 30)
        root = next(s for s in solver.starting_states(spells, flowers)
                    if s.word == result.state.hist[0])
        self.replay('bridge', result.state, root)

    def test_bridge_user_route_is_legal_in_progression_profile(self):
        set_goal('bridge')
        spells = ('alter', 'move', 'add', 'change', 'chop', 'drop', 'swap')
        root = solver.starting_states(spells, ('tigerwishgrantbud',))[0]
        path = self.build_path(root, ('move_4_1_', 'swap_0t_14b_', 'chop_0_', 'add_3_d_'))
        self.assertIn('bridge', path[-1].word)
        self.assert_legal_path(path)

    def test_goal_predicate_keeps_required_unlock_during_shortening(self):
        root = state('a', ('unlock_letter',), [True])
        result = solver.solve('a', [root], accept_state=lambda s: not s.frozen[0])
        self.assertEqual(result.cost, 1)
        self.assertTrue(result.optimal)
        self.assertFalse(result.state.frozen[0])
        self.assertTrue(result.state.hist[-1].startswith('unlock_'))
        self.replay('a', result.state, root)
        impossible = solver.solve('a', [state('a', (), [True])],
                                  accept_state=lambda s: not s.frozen[0])
        self.assertIsNone(impossible.state)
        self.assertEqual(impossible.stop_reason, 'exhausted')

    def test_drop_can_remove_a_frozen_boundary_of_an_interior_flower(self):
        word = 'xredpedalwalkery'
        root = replace(state(word, ('drop',), [True] * len(word)), source_words=('cespyoped',))
        child = next(s for s in root.drop() if s.hist[-1].startswith('drop_1_r_'))
        self.assertEqual(child.word, 'xedpedalwalkery')
        set_goal(child.word)
        self.assertEqual(root.search_bounds(), (True, 1))
        result = solver.solve(child.word, [root])
        self.assertEqual(result.cost, 1)
        self.assertTrue(result.optimal)
        self.replay(child.word, result.state, root)

    def test_interior_drop_is_equivalent_to_the_existing_combine_drop_macro(self):
        root = solver.starting_states(('combine', 'drop'), ('ab', 'xy'))[0]
        combined = next(s for s in root.combine(include_drop=False, position=1, source='xy'))
        dropped = next(s for s in combined.drop() if s.hist[-1].startswith('drop_2_y_'))
        macro = next(s for s in root.combine(position=1, source='xy')
                     if s.hist[-1].startswith('combine_drop1_'))
        self.assertEqual(dropped.key(), macro.key())
        self.assertEqual(solver.spell_count(dropped.hist), solver.spell_count(macro.hist))
        set_goal('axb')
        self.assertEqual(solver._shorten_path([root, combined, dropped], 'axb')[-1].word, 'axb')

    def test_drop_and_combine_respect_the_gui_limit(self):
        root = solver.starting_states(('combine', 'drop', 'swap_adjacent'), ('x' * 35, 'redpedalwalker'))[0]
        self.assertTrue(all(s.hist[-1].startswith('drop_n_') and s.word == root.word[1:]
                            for s in root.drop()))
        self.assertEqual(list(root.combine(position=34)), [])
        self.assertEqual(list(root.swap_adjacent(position=33)), [])
        self.assertFalse(any(s.hist[-1].startswith('combine_drop1_21_')
                             for s in root.combine(position=21, source='redpedalwalker')))

    def test_replacement_finish_uses_real_casts_and_respects_resources(self):
        cases = (
            ('ba', 'ab', ('swap',), 1),
            ('abcd', 'xbce', ('change', 'adjacent'), 2),
            ('adda', 'abba', ('find_and_replace',), 1),
            ('abcd', 'axcy', ('change', 'recast_spell'), 3),
        )
        for goal, word, spells, cost in cases:
            with self.subTest(goal=goal, spells=spells):
                set_goal(goal)
                root = replace(state(word, spells), allowed_spells=frozenset(spells))
                plan = solver.finish_with_replacements(root)
                self.assertIsNotNone(plan)
                self.assertIn(goal, plan[-1].word)
                self.assertEqual(solver.spell_count(plan[-1].hist), cost)
                self.assert_legal_path([root] + plan)
                self.assertIsNone(solver.finish_with_replacements(root, max_casts=cost - 1))
                self.assertIsNone(solver.finish_with_replacements(root, node_budget=0))
        set_goal('abcd')
        frozen = state('xbce', ('change', 'adjacent'), [True, False, False, False])
        self.assertIsNone(solver.finish_with_replacements(frozen))
        solver.max_len = 1
        self.assertIsNone(solver.finish_with_replacements(state('xbce', ('change', 'adjacent'))))

    def test_targeted_generators_are_subsets_of_full_spell_generators(self):
        root = state('abcd', ('change', 'alter', 'adjacent', 'subtract', 'swap',
                              'swap_adjacent', 'find_and_replace'))
        calls = (
            ('change', {'position': 1, 'letter': 'z'}),
            ('alter', {'position': 0, 'letter': 'e'}),
            ('adjacent', {'position': 0, 'letter': 'b'}),
            ('subtract', {'position': 2}),
            ('swap', {'first': 0, 'second': 2}),
            ('swap_adjacent', {'position': 1}),
            ('find_and_replace', {'source': 'a', 'destination': 'z'}),
        )
        for name, kwargs in calls:
            with self.subTest(name=name):
                full = {child.hist[-1]: child.key() for child in getattr(root, name)()}
                selected = list(getattr(root, name)(**kwargs))
                self.assertEqual(len(selected), 1)
                for child in selected:
                    self.assertEqual(child.key(), full[child.hist[-1]])

    def test_combination_starts_have_legal_ancestry_and_respect_availability(self):
        solver.sources = ['cd']
        set_goal('abcd')
        root = state('ab', ('combine',))
        seeds = solver._combination_seeds([solver._PendingState(root)])
        self.assertTrue(seeds)
        for seed in seeds:
            self.assertEqual(seed.cost, 1)
            self.assert_legal_path([root, seed.state])
        self.assertEqual(solver._combination_seeds([solver._PendingState(state('ab'))]), [])
        self.assertEqual(solver._combination_seeds([solver._PendingState(root)], max_cost=1), [])

    def test_recombination_preparations_retain_every_real_intermediate_cast(self):
        set_goal('abcdcd')
        root = replace(state('ab', ('combine', 'recast_spell', 'spellcheck')),
                       allowed_spells=frozenset(('combine', 'recast_spell', 'spellcheck')),
                       source_words=('cd',))
        pending = solver._PendingState(root)
        for all_positions in (False, True):
            seeds = solver._recombination_seeds([pending], limit=8,
                                               all_positions=all_positions)
            if all_positions:
                self.assertTrue(seeds)
                self.assertTrue(any('abcdcd' in seed.state.word for seed in seeds))
            for seed in seeds:
                path, node = [], seed
                while node is not None:
                    path.append(node.state)
                    node = node.parent
                self.assert_legal_path(path[::-1])
                self.assertIn(seed.cost, (3, 4))
            self.assertEqual(solver._recombination_seeds(
                [pending], max_cost=3, all_positions=all_positions), [])

    def test_ordering_hint_accounts_for_remaining_replacement_and_unlock(self):
        set_goal('lele')
        root = state('rere', ('find_and_replace',))
        self.assertLessEqual(root.heuristic_distance(), 2)
        self.assertEqual(next(root.find_and_replace('r', 'l')).word, 'lele')
        set_goal('abg')
        frozen = state('abh', ('unlock_letter', 'adjacent'), [False, False, True])
        self.assertEqual(frozen.heuristic_distance(), 2)
        unlocked = next(frozen.unlock_letter(2))
        self.assertEqual(unlocked.heuristic_distance(), 1)
        self.assertIn('abg', next(unlocked.adjacent(2, 'g')).word)

    def test_full_preparation_ties_are_independent_of_source_list_order(self):
        set_goal('ababcd')
        histories = []
        for flowers in (('ab', 'cd'), ('cd', 'ab')):
            roots = [solver._PendingState(root) for root in solver.starting_states(
                ('combine', 'recast_spell', 'spellcheck'), flowers)]
            seeds = solver._recombination_seeds(roots, limit=16, all_positions=True)
            histories.append([node.state.hist for node in seeds])
        self.assertTrue(histories[0])
        self.assertEqual(histories[0], histories[1])

    def test_discovery_promotes_existing_nodes_without_reexpanding_them(self):
        # A finite graph fixture keeps an important node queued behind a long
        # attractive branch. Discovery must promote the existing pending node,
        # rather than discard the shortcut as a duplicate.
        root = state('r', ('combine',))
        expanded = []
        def children(current, include_bulk=True):
            expanded.append(current.word)
            if current.word == 'r':
                following = ('w', 's')
            elif current.word == 's':
                following = ('z',)
            elif current.word.startswith('w') and len(current.word) < 200:
                following = (current.word + 'a',)
            else:
                following = ()
            for word in following:
                yield replace(current, word=word, frozen=[False] * len(word), score_=None,
                              hist=current.hist + ['change_0_x_' + word])
        def seeds(roots, **kwargs):
            source = roots[0]
            prepared = replace(source.state, word='s', frozen=[False], score_=None,
                               hist=source.state.hist + ['change_0_x_s'])
            return [solver._PendingState(prepared, parent=source)]
        with mock.patch.object(solver.State, 'generate_states_unlock', children), \
                mock.patch.object(solver.State, 'generate_bulk_states', lambda s: iter(())), \
                mock.patch.object(solver.State, 'score', lambda s: 0 if s.word == 's' else 100), \
                mock.patch.object(solver, '_combination_seeds', seeds), \
                mock.patch.object(solver, '_spellcheck_seeds', return_value=[]):
            result = solver.solve('z', [root], simplify=False)
        self.assertEqual(result.cost, 2)
        self.assertTrue(result.optimal)
        self.assertLess(result.expanded, 50)
        self.assertEqual(len(expanded), len(set(expanded)))

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
            result = solver.solve(goal, [state(word, (spell,))], replacement_finish=False)
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

    def replay(self, goal, solution, initial=None):
        set_goal(goal)
        s = initial if initial is not None else solver.State(
            solution.hist[0], [False] * len(solution.hist[0]), hist=[solution.hist[0]],
        )
        for step in solution.hist[len(s.hist):]:
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
        result = self.regression_solve('cashmoney', first_solution=True)
        self.assertIsNotNone(result.state)
        self.assertLessEqual(result.cost, 4)
        self.assertFalse(result.state.used_add)
        self.assertFalse(any(step.startswith('add_') for step in result.state.hist[1:]))
        self.replay_regression('cashmoney', result.state)

    @frozen_regression('cashmoney')
    def test_shortening_removes_cashmoney_unused_y_from_the_original_route(self):
        set_goal('cashmoney')
        word = 'redpedalwalker'
        root = state(word, REGRESSION_CONFIGS['cashmoney'].spells)
        path = self.build_path(root, (
            'adj-_5_d_', 'add_13_y_', 'combine_8_honeycombbmineraleaf_',
            'change_7_s_', 'move_15_9_',
        ))
        shorter = solver._shorten_path(path, 'cashmoney')
        self.assertEqual(solver.spell_count(shorter[-1].hist), 4)
        self.assertFalse(shorter[-1].used_add)
        self.assert_legal_path(shorter)

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
        first = self.regression_solve('FREEDOM', first_solution=True)
        second = self.regression_solve('freedom', first_solution=True)
        self.assertIsNotNone(first.state)
        self.assertLessEqual(len(first.state.hist) - 1, 5)
        self.assertEqual(first.state.hist, second.state.hist)
        self.replay_regression('freedom', first.state)

    def test_combine_is_found_and_legal(self):
        result = self.regression_solve('combine')
        self.assertIsNotNone(result.state)
        self.assertLessEqual(len(result.state.hist) - 1, 4)
        self.replay_regression('combine', result.state)

    @frozen_regression('combine')
    def test_combine_user_route_survives_pruning(self):
        set_goal('combine')
        s = state('honeycombbmineraleaf', ('swap', 'move', 'alter', 'add'))
        for prefix in ('swap_9b_10m_', 'move_6_8_', 'alter_5_n_', 'add_8_c_'):
            self.assertTrue(s.can_reach(), s.hist)
            s = next(child for child in s.generate_states_unlock()
                     if child.hist[-1].startswith(prefix))
        self.assertIn('combine', s.word)

    def test_knock_is_found_quickly_in_four_spells(self):
        result = self.regression_solve('knock', first_solution=True)
        self.assertIsNotNone(result.state)
        self.assertLessEqual(len(result.state.hist) - 1, 4)
        self.assertLess(result.expanded, 100)
        self.replay_regression('knock', result.state)

    @frozen_regression('knock')
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

    @frozen_regression('recastspell')
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
        result = self.regression_solve('RECASTSPELL', first_solution=True)
        self.assertIsNotNone(result.state)
        self.assertLessEqual(len(result.state.hist) - 1, 9)
        self.assertLess(result.expanded, 500)
        self.replay_regression('recastspell', result.state)

    @frozen_regression('mostfrequent')
    def test_mostfrequent_screenshot_route_is_legal_and_passes_bounds(self):
        set_goal('mostfrequent')
        root = state('tigerwishgrantbud', (
            'combine', 'alter', 'adjacent', 'move', 'swap_adjacent', 'swap', 'add', 'recast_spell',
        ))
        path = self.build_path(root, (
            'combine_7_lilachangerbloom_', 'alter_27_e_', 'adj-_25_g_', 'move_3_26_',
            'swap_adj_20o_21m_', 'swap_0t_23h_', 'add_27_q_', 'recast_move_', 'move_32_28_',
        ))
        for index, ancestor in enumerate(path):
            feasible, lower_bound = ancestor.search_bounds()
            self.assertTrue(feasible, ancestor.hist)
            self.assertLessEqual(lower_bound, len(path) - index - 1)
        self.assertIn('mostfrequent', path[-1].word)
        self.assertEqual(solver.spell_count(path[-1].hist), 9)
        self.assert_legal_path(path)

    def test_anytime_discovers_mostfrequent_with_current_spells(self):
        costs = []
        def stop_after_first(state, cost):
            costs.append(cost)
            raise KeyboardInterrupt
        result = self.regression_solve('MOSTFREQUENT', on_solution=stop_after_first)
        self.assertTrue(result.interrupted)
        self.assertFalse(result.optimal)
        self.assertIsNotNone(result.state)
        self.assertLessEqual(result.cost, 9)
        self.assertEqual(costs, [result.cost])
        self.assertLess(result.expanded, 200)
        self.replay_regression('mostfrequent', result.state)

    def test_spikybetty_screenshot_route_is_legal_and_costs_ten_casts(self):
        with self.regression_config('spikybetty') as roots:
            set_goal('spikybetty')
            root = next(s for s in roots if s.word == 'tigerwishgrantbud')
            path = self.build_path(root, (
                'combine_13_cespyoped_', 'alter_19_b_', 'move_1_16_', 'sub_18_o_',
                'add_17_k_', 'swap_0t_21d_', 'recast_combine_',
                'combine_drop0_23_oneycombbmineraleaf_', 'spellcheck_',
            ))
            for ancestor in path:
                self.assertTrue(ancestor.can_reach(), ancestor.hist)
            self.assertIn('spikybetty', path[-1].word)
            self.assertEqual(solver.spell_count(path[-1].hist), 10)
            self.assert_legal_path(path)

    def test_anytime_discovers_spikybetty_with_frozen_configuration(self):
        def stop_after_first(state, cost):
            raise KeyboardInterrupt
        result = self.regression_solve('SPIKYBETTY', on_solution=stop_after_first)
        self.assertTrue(result.interrupted)
        self.assertFalse(result.optimal)
        self.assertIsNotNone(result.state)
        self.assertLessEqual(result.cost, 9)
        self.assertLess(result.expanded, 200)
        self.replay_regression('spikybetty', result.state)

    def test_actual_cast_costs_include_recast_and_bundled_drop(self):
        self.assertEqual(solver.spell_count(['ab', 'recast_combine_ab',
                                            'combine_drop0_2_c_abc']), 3)
        root = state('ab', ('recast_spell',))
        recast = next(s for s in root.recast_spell() if not s.used_change)
        self.assertEqual(sum(root.useds()), sum(recast.useds()))
        node = solver._PendingState(recast, parent=solver._PendingState(root))
        self.assertEqual(node.cost, 1)

    def test_recast_is_postponed_until_its_next_use_and_unused_recast_is_removed(self):
        root = solver.starting_states(('move', 'swap', 'recast_spell'), ('abcd',))[0]
        moved = next(root.move(1, 2))
        recast = next(moved.recast_spell())
        swapped = next(recast.swap(0, 3))
        final = next(swapped.move(1, 2))
        original = [root, moved, recast, swapped, final]
        normal = solver._normalize_recast(original)
        self.assertEqual(normal[-1].word, final.word)
        self.assertEqual(normal[-1].frozen, final.frozen)
        self.assertEqual(normal[-1].useds(), final.useds())
        self.assertEqual([solver.command_spells(s.hist[-1])[0] for s in normal[1:]],
                         ['move', 'swap', 'recast_spell', 'move'])
        self.assert_legal_path(normal)
        unused = solver._normalize_recast(original[:-1])
        self.assertEqual(solver.spell_count(unused[-1].hist), 2)
        self.assertFalse(unused[-1].used_recast_spell)
        self.assertEqual(unused[-1].word, swapped.word)
        self.assert_legal_path(unused)

    def test_recast_search_filters_other_casts_but_raw_replay_remains_legal(self):
        root = solver.starting_states(('move', 'swap', 'recast_spell'), ('abcd',))[0]
        moved = next(root.move(1, 2))
        recast = next(moved.recast_spell())
        all_children = list(recast.generate_states_unlock())
        self.assertTrue(any('swap' in solver.command_spells(s.hist[-1]) for s in all_children))
        canonical = list(recast.canonical_successors(all_children))
        self.assertTrue(canonical)
        self.assertTrue(all('move' in solver.command_spells(s.hist[-1]) for s in canonical))
        self.assertNotEqual(recast.key(), replace(recast, hist=moved.hist).key())
        with mock.patch.object(solver.State, 'swap', side_effect=AssertionError('unused cast')):
            self.assertEqual({s.key() for s in recast.search_successors()},
                             {s.key() for s in canonical})

    def test_recast_order_reduction_preserves_shortest_paths_against_raw_oracle(self):
        solver.abc = 'abc'
        for enabled in (('change', 'recast_spell'), ('move', 'swap', 'recast_spell'),
                        ('alter', 'add', 'recast_spell')):
            root = solver.starting_states(enabled, ('ab',))[0]
            for goal in ('ac', 'ba', 'cc', 'aba'):
                expected = reference_shortest(goal, [root])
                result = solver.solve(goal, [root], guided_search=False)
                self.assertEqual(result.cost, expected, (enabled, goal))
                self.assertEqual(result.optimal, expected is not None)
                if result.state:
                    for i, command in enumerate(result.state.hist[1:], 1):
                        if command.startswith('recast_'):
                            restored = command.rsplit('_', 1)[0][len('recast_'):]
                            self.assertIn(restored, solver.command_spells(result.state.hist[i + 1]))

    def test_recast_order_reduction_does_not_restrict_an_existing_current_position(self):
        root = solver.starting_states(('change', 'move', 'recast_spell'), ('ab',))[0]
        moved = next(root.move(0, 1))
        current = next(moved.recast_spell())
        self.assertEqual(current.pending_recast(), 'move')
        result = solver.solve('ca', [current], guided_search=False)
        self.assertEqual(result.cost, 3)
        self.assertTrue(result.optimal)
        self.assertEqual(result.state.word, 'ca')

    def test_search_discards_unchanged_spellcheck_but_replay_retains_it(self):
        root = state('ab', ('spellcheck',))
        checked = next(root.spellcheck())
        self.assertEqual(checked.word, root.word)
        self.assertTrue(checked.used_spellcheck)
        self.assertTrue(root.is_redundant_cast(checked))
        result = solver.solve('az', [root], prune=False, guided_search=False)
        self.assertIsNone(result.state)
        self.assertEqual(result.noop_pruned, 1)
        self.assertEqual(result.expanded, 1)

    def test_custom_goal_can_require_a_noop_cast_and_recast_is_not_discarded(self):
        root = state('ab', ('spellcheck',))
        result = solver.solve('ab', [root], accept_state=lambda s: s.used_spellcheck)
        self.assertEqual(result.cost, 1)
        self.assertTrue(result.state.used_spellcheck)
        self.assertEqual(result.noop_pruned, 0)
        root = solver.starting_states(('move', 'recast_spell'), ('ab',))[0]
        moved = next(root.move(0, 1))
        restored = next(moved.recast_spell())
        self.assertFalse(moved.is_redundant_cast(restored))

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
            expected = (False, 6) if spell in ('spellcheck', 'unlock_letter') else (False, 1)
            self.assertEqual(state('a', (spell,)).search_bounds(), expected)
        set_goal('aaa')
        self.assertEqual(state('aaa').search_bounds(), (True, 0))

    def test_repetition_bound_counts_unfrozen_replacement_supply_and_future_donors(self):
        set_goal('aaaaa')
        cases = (
            (('combine', 'recast_spell'), ('a', 'aa'), 'a', True),
            (('find_and_replace', 'most_frequent'), ('bbbb',), 'bbbb', True),
            (('find_and_replace',), ('bbbb',), 'bbbb', False),
            (('find_and_replace', 'unlock_letter'), ('bbbbb',), 'bbbbb', True),
        )
        for spells, flowers, word, expected in cases:
            with self.subTest(spells=spells, word=word):
                root = next(s for s in solver.starting_states(spells, flowers) if s.word == word)
                self.assertEqual(root.repetition_feasible(), expected)
                cost = reference_shortest('aaaaa', [root])
                self.assertEqual(cost is not None, expected)
        # Frozen donors cannot participate in replacement. One unlock only
        # restores one of them, so it cannot supply five new copies.
        root = solver.starting_states(('find_and_replace', 'unlock_letter'), ('bbbbb',))[0]
        root.frozen = [True] * 5
        self.assertFalse(root.repetition_feasible())

    def test_repetition_bound_preserves_random_real_bulk_spell_histories(self):
        solver.abc = 'ab'
        rng = random.Random(21)
        spell_sets = (
            ('combine', 'recast_spell', 'find_and_replace', 'unlock_letter', 'duplicate'),
            ('combine', 'recast_spell', 'find_and_replace', 'most_frequent', 'subtract'),
            ('find_and_replace', 'duplicate', 'most_frequent', 'unlock_letter', 'change'),
        )
        checked = 0
        for spells in spell_sets:
            for _ in range(100):
                path = [solver.starting_states(spells, ('aaa', 'bbbb'))[rng.randrange(2)]]
                for _ in range(7):
                    children = list(path[-1].generate_states_unlock())
                    if not children:
                        break
                    path.append(rng.choice(children))
                goal = path[-1].word
                if max((goal.count(c) for c in set(goal)), default=0) < 5:
                    continue
                set_goal(goal)
                for ancestor in path:
                    self.assertTrue(ancestor.repetition_feasible(),
                                    (spells, path[-1].hist, ancestor.hist))
                    self.assertTrue(ancestor.search_bounds()[0],
                                    (spells, path[-1].hist, ancestor.hist))
                    checked += 1
        self.assertGreater(checked, 100)

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
        result = solver.solve('ac', [state('ab', ('change',))], on_progress=stop,
                              replacement_finish=False)
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

    def test_spell_selections_accept_aliases_and_commas(self):
        self.assertEqual(solver.select_spells(['MOVE,adj', 'swap-adjacent', 'recastspell', 'fandr']),
                         frozenset(('move', 'adjacent', 'swap_adjacent', 'recast_spell',
                                    'find_and_replace')))
        self.assertEqual(solver.select_spells('none'), frozenset())
        self.assertEqual(solver.select_spells('all'), frozenset(solver.SPELL_NAMES))
        with self.assertRaisesRegex(ValueError, 'unknown spell'):
            solver.select_spells('moev')

    def test_default_starting_configuration_captures_current_settings(self):
        solver.sources = ['ab', 'cd']
        roots = solver.starting_states()
        defaults = solver.State('', [])
        self.assertEqual([root.word for root in roots], solver.sources)
        for root in roots:
            self.assertEqual(root.useds(), defaults.useds())
            self.assertEqual(root.source_words, ('ab', 'cd'))
            self.assertEqual(root.allowed_spells, frozenset(
                name for name in solver.SPELL_NAMES if defaults.available(name)))

    def test_blacklisted_spells_cannot_be_restored_by_recast_or_finishers(self):
        root = solver.starting_states(['recast_spell'], ['ab'])[0]
        self.assertEqual(list(root.recast_spell()), [])
        set_goal('ba')
        self.assertIsNone(root.finish_with_insertion())
        result = solver.solve('ba', allowed_spells=['recast_spell'], source_words=['ab'])
        self.assertIsNone(result.state)
        self.assertEqual(result.stop_reason, 'exhausted')
        # An explicitly blacklisted MOVE stays blocked even on an inconsistent
        # manually constructed state whose used flag says it is available.
        direct = replace(state('xabc', ('move', 'combine', 'recast_spell')),
                         allowed_spells=frozenset(('recast_spell',)))
        set_goal('abxc')
        self.assertEqual(list(direct.move()), [])
        self.assertEqual(list(direct.combine()), [])
        self.assertIsNone(direct.finish_with_insertion())

    def test_allowed_spent_spells_can_still_be_recast(self):
        root = solver.starting_states(['move', 'recast'], ['abc'])[0]
        moved = next(root.move(0, 2))
        restored = list(moved.recast_spell())
        self.assertEqual(len(restored), 1)
        self.assertFalse(restored[0].used_move)
        self.assertTrue(restored[0].used_recast_spell)
        self.assertEqual(restored[0].allowed_spells, root.allowed_spells)
        self.assertEqual(restored[0].source_words, root.source_words)

    def test_selected_sources_control_both_roots_and_combine_without_global_changes(self):
        solver.sources = ['zz']
        roots = solver.starting_states(['combine'], ['AB,cd', 'ab'])
        self.assertEqual([root.word for root in roots], ['ab', 'cd'])
        self.assertEqual({s.hist[-1].split('_')[2] for s in roots[0].combine()}, {'ab', 'cd'})
        result = solver.solve('abcd', allowed_spells=['combine'], source_words=['ab', 'cd'])
        self.assertTrue(result.optimal)
        self.assertEqual(result.cost, 1)
        self.assertEqual(result.state.source_words, ('ab', 'cd'))
        self.assertEqual(solver.sources, ['zz'])

    def test_initial_state_overrides_preserve_spent_spells(self):
        root = state('ab')  # CHANGE is spent, even if it is permitted.
        result = solver.solve('ac', [root], allowed_spells=['change'], source_words=['cd'])
        self.assertIsNone(result.state)
        result = solver.solve('ac', [state('ab', ('recast_spell',))],
                              allowed_spells=['change', 'recast_spell'], source_words=['cd'])
        self.assertEqual(result.cost, 2)
        self.assertEqual(result.state.source_words, ('cd',))
        self.assertEqual(result.state.allowed_spells, frozenset(('change', 'recast_spell')))

    def test_state_key_includes_recast_permissions_and_combine_sources(self):
        root = state('ab')
        keys = {root.key(), replace(root, allowed_spells=frozenset()).key(),
                replace(root, source_words=('cd',)).key()}
        self.assertEqual(len(keys), 3)

    def test_cli_allowlists_select_starting_spells_and_sources(self):
        with mock.patch.object(solver, 'solve', wraps=solver.solve) as search:
            with contextlib.redirect_stdout(io.StringIO()):
                code = solver.main(['ac', '--spells', 'add,change', 'move',
                                    '--sources', 'AB', '--first'])
        self.assertEqual(code, 0)
        self.assertEqual(search.call_args.kwargs['allowed_spells'],
                         frozenset(('add', 'change', 'move')))
        self.assertEqual(search.call_args.kwargs['source_words'], ('ab',))

    def test_cli_blacklists_start_from_all_implemented_choices(self):
        with mock.patch.object(solver, 'solve', wraps=solver.solve) as search:
            with contextlib.redirect_stdout(io.StringIO()):
                code = solver.main(['honey', '--blacklist-spells', 'adj,move',
                                    '--blacklist-sources', 'redpedalwalker,sakuya', '--first'])
        self.assertEqual(code, 0)
        self.assertEqual(search.call_args.kwargs['allowed_spells'],
                         frozenset(solver.SPELL_NAMES) - {'adjacent', 'move'})
        self.assertIn('find_and_replace', search.call_args.kwargs['allowed_spells'])
        self.assertIn('heedonicindigo', search.call_args.kwargs['source_words'])
        self.assertNotIn('sakuya', search.call_args.kwargs['source_words'])

    def test_cli_can_select_previously_disabled_or_custom_sources(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(solver.main(['sakuya', '--sources', 'sakuya', '--spells', 'none']), 0)
            self.assertEqual(solver.main(['custom', '--sources', 'custom', '--spells', 'none']), 0)

    def test_cli_rejects_invalid_or_conflicting_selections_before_search(self):
        cases = (
            ['--spells', 'moev'], ['--spells', 'add', '--blacklist-spells', 'move'],
            ['--sources', 'ab', '--blacklist-sources', 'sakuya'],
            ['--sources', 'not-a-word'], ['--sources', 'none'],
            ['--blacklist-sources', 'all'], ['--blacklist-sources', 'typo'],
        )
        with mock.patch.object(solver, 'solve') as search:
            for args in cases:
                with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        solver.main(['z'] + args)
                    self.assertEqual(error.exception.code, 2)
            search.assert_not_called()

    def test_most_frequent_handles_empty_and_single_letter_words(self):
        self.assertEqual(list(state('', ('most_frequent',)).generate_states_unlock()), [])
        children = list(state('aa', ('most_frequent',)).generate_states_unlock())
        self.assertEqual(len(children), 3)
        self.assertTrue(all(s.word == 'aaa' for s in children))


if __name__ == '__main__':
    unittest.main()
