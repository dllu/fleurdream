import contextlib
import io
import json
import os
import subprocess
import sys
import time
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import progression
import solver


CATALOG = json.loads((Path(__file__).parent / 'fixtures' / 'game_spells.json').read_text())
TARGETS = tuple(CATALOG['targets'])
BOOTSTRAP = ('change', 'hang', 'walk', 'alter', 'lie', 'drop', 'chop', 'add', 'move', 'swap')


class ProgressionTests(unittest.TestCase):
    def setUp(self):
        self.settings = solver.target, solver.tgt_abc, solver.abc, solver.max_len, solver.sources

    def tearDown(self):
        solver.target, solver.tgt_abc, solver.abc, solver.max_len, solver.sources = self.settings

    def test_catalog_matches_the_installed_game_snapshot(self):
        self.assertEqual(len(TARGETS), 100)
        self.assertEqual(tuple(solver.GAME_SPELLS), TARGETS)
        self.assertEqual(len(set(TARGETS)), len(TARGETS))
        self.assertTrue(set(progression.SPELL_UNLOCKS) <= set(TARGETS))

    def test_flower_gates_require_all_prerequisites(self):
        self.assertEqual(progression.unlocked_technology(()), progression.Technology())
        for words in (('explosion',), ('swim',), ('cashmoney',), ('violeepicturet',)):
            self.assertEqual(progression.unlocked_technology(words).sources,
                             progression.INITIAL_SOURCES)
        reached = ('lie', 'explosion', 'swim', 'bridge', 'breakandenter',
                   'cashmoney', 'violeepicturet')
        technology = progression.unlocked_technology(reached)
        self.assertEqual(set(technology.sources), {
            'redpedalwalker', 'lilachangerbloom', 'cespyoped', 'tigerwishgrantbud',
            'honeycombbmineraleaf', 'heedonicindigo', 'violeepicturet',
        })
        self.assertNotIn('sakuya', technology.sources)

    def test_walkthroughwalls_unlocks_every_source_without_free_spells(self):
        technology = progression.unlocked_technology(('walkthroughwalls',))
        self.assertEqual(set(technology.sources), set(solver.SOURCE_CATALOG))
        self.assertEqual(technology.spells, frozenset())

    def test_sakuya_requires_starting_the_festival_and_any_one_criterion(self):
        self.assertNotIn('sakuya', progression.unlocked_technology(('startthefestival',)).sources)
        for criterion in ('grabsakuyasvase', 'suitofarmor', 'newwatertower'):
            self.assertNotIn('sakuya', progression.unlocked_technology((criterion,)).sources)
            self.assertIn('sakuya', progression.unlocked_technology(
                ('startthefestival', criterion)).sources)

    def test_non_editing_spells_do_not_grant_unrelated_abilities(self):
        self.assertEqual(progression.unlocked_technology(('all', 'freedom')).spells, frozenset())
        self.assertEqual(progression.unlocked_technology(('change', 'swapadjacent')).spells,
                         frozenset(('change', 'swap_adjacent')))

    def test_bootstrap_progression_from_no_spells_and_two_flowers(self):
        report = progression.run(BOOTSTRAP, budgets=(0.1, 1), total_seconds=15)
        self.assertEqual(report.unresolved, (), progression.json_report(report))
        self.assertEqual(report.stopped, 'complete')
        levels = {'change': 0, 'hang': 0, 'walk': 0, 'alter': 1, 'lie': 1,
                  'drop': 2, 'chop': 3, 'add': 4, 'move': 5, 'swap': 5}
        for target, level in levels.items():
            with self.subTest(target=target):
                attempt = report.found[target]
                self.assertEqual(attempt.level, level)
                prior = {word for word, found in report.found.items() if found.level < level}
                self.assertEqual(attempt.technology, progression.unlocked_technology(prior))
                self.assertNotIn(progression.SPELL_UNLOCKS.get(target), attempt.technology.spells)
                state, required = progression.replay(target, attempt.history, attempt.technology)
                self.assertEqual(solver.spell_count(state.hist), attempt.cost)
                self.assertEqual(required, attempt.required)
                self.assertTrue(required.spells <= attempt.technology.spells)
                self.assertTrue(set(required.sources) <= set(attempt.technology.sources))
        self.assertEqual(report.found['change'].cost, 0)
        self.assertEqual(report.found['lie'].cost, 1)
        self.assertEqual(report.found['drop'].cost, 2)

    def test_audit_ignores_live_flags_sources_alphabet_and_position_limit(self):
        defaults = solver.State.__init__.__defaults__
        flags = tuple(name for name in solver.State.__dataclass_fields__ if name.startswith('used_'))
        changed = (True,) * len(flags) + defaults[len(flags):]
        with mock.patch.object(solver.State.__init__, '__defaults__', changed), \
                mock.patch.multiple(solver, sources=['z'], max_len=0, abc='z'):
            report = progression.run(('change', 'lie'), budgets=(1,), total_seconds=5)
            self.assertEqual(report.unresolved, ())
            self.assertEqual(report.found['change'].cost, 0)
            self.assertEqual(report.found['lie'].cost, 1)
            self.assertEqual(solver.sources, ['z'])
            self.assertEqual(solver.abc, 'z')
            self.assertEqual(solver.max_len, 0)

    def test_witness_dependencies_include_bundled_drop_and_combine_donor(self):
        technology = progression.Technology(frozenset(('combine', 'drop')), ('ab', 'xy'))
        attempt = progression.search('axb', technology, 3)
        self.assertEqual(attempt.status, 'found')
        self.assertEqual(attempt.cost, 2)
        self.assertTrue(attempt.history[-1].startswith('combine_drop1_'))
        self.assertEqual(attempt.required, technology)

    def test_witness_dependencies_include_recast(self):
        technology = progression.Technology(frozenset(('change', 'recast_spell')), ('aa',))
        attempt = progression.search('bc', technology, 3)
        self.assertEqual(attempt.status, 'found')
        self.assertEqual(attempt.cost, 3)
        self.assertEqual(attempt.required, technology)

    def test_replay_rejects_locked_spells_and_sources(self):
        with self.assertRaisesRegex(AssertionError, 'locked source'):
            progression.replay('lie', ['cespyoped'], progression.Technology())
        with self.assertRaisesRegex(AssertionError, 'illegal cast'):
            progression.replay('lie', ['lilachangerbloom', 'change_2_e_lieachangerbloom'],
                               progression.Technology())

    def test_timeout_is_not_reported_as_exhaustion(self):
        technology = progression.Technology(frozenset(solver.SPELL_NAMES))
        attempt = progression.search('z' * 1000, technology, 0.001)
        self.assertEqual(attempt.status, 'timeout')
        self.assertLess(attempt.seconds, 2)
        self.assertIsNone(attempt.cost)

    @unittest.skipUnless(sys.platform.startswith('linux'), 'process-group cleanup uses /proc')
    def test_timeout_also_stops_a_worker_subprocess(self):
        with tempfile.TemporaryDirectory() as directory:
            pidfile = Path(directory) / 'child.pid'
            def launch_child(*args, **kwargs):
                child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
                pidfile.write_text(str(child.pid))
                time.sleep(60)
            with mock.patch.object(solver, 'solve', side_effect=launch_child):
                attempt = progression.search('a', progression.Technology(), .5)
            self.assertEqual(attempt.status, 'timeout')
            pid = int(pidfile.read_text())
            for _ in range(20):
                try:
                    state = Path(f'/proc/{pid}/stat').read_text().split()[2]
                except FileNotFoundError:
                    break
                if state in ('Z', 'X'):
                    break
                time.sleep(.01)
            else:
                self.fail('timed-out worker left its subprocess running')

    def test_worker_errors_fail_instead_of_becoming_false_negatives(self):
        with self.assertRaisesRegex(RuntimeError, 'unknown spell'):
            progression.search('a', progression.Technology(frozenset(('typo',))), 3)

    def test_exhausted_profiles_are_cached_but_timeouts_are_retried(self):
        for status, expected in (('exhausted', 1), ('timeout', 2)):
            def searcher(target, technology, seconds, level):
                return progression.Attempt(target, level, technology, seconds, status, 0)
            searcher = mock.Mock(wraps=searcher)
            with self.subTest(status=status):
                report = progression.run(('bridge',), budgets=(0.01, 0.02),
                                         total_seconds=1, searcher=searcher)
                self.assertEqual(searcher.call_count, expected)
                self.assertEqual(report.unresolved, ('bridge',))
                self.assertEqual(report.stopped, 'stalled')

    def test_new_technology_caps_late_gate_trials_at_the_old_profile(self):
        calls = []
        def searcher(target, technology, seconds, level):
            calls.append((target, technology, seconds))
            if target == 'change' and seconds >= 1:
                return progression.Attempt(
                    target, level, technology, seconds, 'found', 0,
                    cost=0, history=['lilachangerbloom'],
                    required=progression.Technology(sources=('lilachangerbloom',)))
            return progression.Attempt(target, level, technology, seconds, 'timeout', 0)
        progression.run(('change', 'startthefestival'), budgets=(.01, 1),
                        total_seconds=5, searcher=searcher)
        index = next(i for i, (target, technology, seconds) in enumerate(calls)
                     if target == 'change' and seconds == 1)
        self.assertEqual(calls[index + 1], ('startthefestival', progression.Technology(), .01))
        self.assertIn('change', calls[index + 2][1].spells)

    def test_optional_words_get_small_early_probes_while_editing_unlocks_deepen(self):
        calls = []
        def searcher(target, technology, seconds, level):
            calls.append((target, technology, seconds))
            if target == 'change' and seconds >= 1:
                return progression.Attempt(
                    target, level, technology, seconds, 'found', 0,
                    cost=0, history=['lilachangerbloom'],
                    required=progression.Technology(sources=('lilachangerbloom',)))
            return progression.Attempt(target, level, technology, seconds, 'timeout', 0)
        progression.run(('change', 'balloon'), budgets=(.01, .25, 1),
                        total_seconds=5, searcher=searcher)
        early = [seconds for target, technology, seconds in calls
                 if target == 'balloon' and not technology.spells]
        self.assertTrue(early)
        self.assertEqual(set(early), {.01})
        self.assertTrue(any(target == 'balloon' and technology.spells and seconds == 1
                            for target, technology, seconds in calls))

    def test_dependency_minimization_can_certify_a_small_profile(self):
        technology = progression.Technology(frozenset(('change', 'alter')))
        attempt = progression.search('lie', technology, 3)
        self.assertEqual(attempt.status, 'found')
        self.assertEqual(attempt.required,
                         progression.Technology(frozenset(('change',)), ('redpedalwalker',)))
        smaller, trials, minimal = progression.minimize(attempt, seconds=1)
        self.assertTrue(minimal)
        self.assertEqual(smaller.required, attempt.required)
        self.assertTrue(all(trial.status == 'exhausted' for trial in trials))

    def test_minimization_does_not_claim_proof_after_a_timeout(self):
        technology = progression.Technology(frozenset(('change',)))
        attempt = progression.Attempt('lie', 1, technology, 1, 'found', 0,
                                      required=technology)
        def timeout(target, technology, seconds, level):
            return progression.Attempt(target, level, technology, seconds, 'timeout', seconds)
        _, trials, minimal = progression.minimize(attempt, searcher=timeout)
        self.assertFalse(minimal)
        self.assertTrue(trials)

    def test_json_report_preserves_failures_and_configuration(self):
        report = progression.run(('change', 'bridge'), budgets=(0.01,), total_seconds=2)
        decoded = json.loads(progression.json_report(report))
        self.assertEqual(decoded['unresolved'], ['bridge'])
        self.assertEqual(decoded['found']['change']['technology']['spells'], [])
        self.assertEqual(decoded['found']['change']['history'], ['lilachangerbloom'])
        self.assertEqual(decoded['assumptions']['max_position'], 33)
        self.assertEqual(decoded['unresolved_status']['bridge'], 'exhausted')

    def test_cli_writes_a_report_with_dependency_removal_trials(self):
        report = progression.run(('change', 'lie'), budgets=(1,), total_seconds=5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            with mock.patch.object(progression, 'run', return_value=report), \
                    contextlib.redirect_stdout(io.StringIO()):
                code = progression.main(['--minimize', '--report', str(path)])
            self.assertEqual(code, 0)
            decoded = json.loads(path.read_text())
        self.assertEqual(decoded['unresolved'], [])
        self.assertTrue(decoded['reductions']['lie']['inclusion_minimal'])
        self.assertEqual(decoded['reductions']['lie']['required']['spells'], ['change'])
        self.assertTrue(decoded['reductions']['lie']['trials'])

    def test_checkpoint_resumes_only_after_revalidating_saved_discoveries(self):
        report = progression.run(('change', 'lie'), budgets=(1,), total_seconds=5)
        del report.found['lie']
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            path.write_text(progression.json_report(report))
            saved = progression.load_report(path)
        progression.validate_report(saved)
        checkpoints = []
        resumed = progression.run(('change', 'lie'), budgets=(1,), total_seconds=5,
                                  initial_report=saved, on_checkpoint=checkpoints.append)
        self.assertEqual(resumed.unresolved, ())
        self.assertTrue(checkpoints)
        self.assertGreater(resumed.found['lie'].level, resumed.found['change'].level)
        resumed.found['change'].history = ['lilachangerbloom', 'change_0_x_xilachangerbloom']
        with self.assertRaisesRegex(AssertionError, 'illegal cast'):
            progression.validate_report(resumed)

    def test_dependency_graph_keeps_incomparable_routes_and_rejects_cycles(self):
        report = progression.run(('change', 'alter', 'lie', 'pet'), budgets=(1,), total_seconds=5)
        technology = progression.Technology(frozenset(('change',)), ('cespyoped',))
        other = progression.search('pet', technology, 1, level=2)
        self.assertEqual(other.status, 'found')
        self.assertTrue(progression.record_alternative(report, other))
        self.assertEqual({a.required.sources for a in report.alternatives['pet']},
                         {('redpedalwalker',), ('cespyoped',)})
        # ALTER is itself learned using CHANGE. It cannot be used as a new
        # bootstrap recipe for CHANGE, even though this spelling is legal.
        root = solver.starting_states(('alter',), ('lilachangerbloom',))[0]
        altered = next(root.alter(position=11, letter='p'))
        _, required = progression.replay('change', altered.hist,
                                         progression.Technology(frozenset(('alter',)),
                                                                ('lilachangerbloom',)))
        cyclic = progression.Attempt('change', 2, progression.Technology(frozenset(('alter',))),
                                     1, 'found', 0, cost=1, history=altered.hist, required=required)
        self.assertFalse(progression.record_alternative(report, cyclic))
        self.assertEqual(progression.dependency_technology(report, exclude=('change',)).spells,
                         frozenset())

    def test_dependency_graph_discards_a_dominated_legal_recipe(self):
        report = progression.run(('change', 'alter', 'lie'), budgets=(1,), total_seconds=5)
        technology = progression.Technology(frozenset(('change', 'alter')), ('redpedalwalker',))
        root = solver.starting_states(technology.spells, technology.sources)[0]
        altered = next(root.alter(position=2, letter='b'))
        solved = next(altered.change(position=11, letter='i'))
        _, required = progression.replay('lie', solved.hist, technology)
        extra = progression.Attempt('lie', 2, technology, 1, 'found', 0,
                                    cost=2, history=solved.hist, required=required)
        self.assertFalse(progression.record_alternative(report, extra))
        self.assertEqual(len(report.alternatives['lie']), 1)

    @unittest.skipUnless(os.environ.get('FLEURDREAM_FULL_PROGRESSION') == '1',
                         'opt-in 100-target benchmark: FLEURDREAM_FULL_PROGRESSION=1')
    def test_all_game_spells_are_reachable_with_progression(self):
        per_target = float(os.environ.get('FLEURDREAM_PROGRESSION_TARGET_SECONDS', '240'))
        budgets = tuple(sorted({min(value, per_target) for value in (.05, .25, 1, 5, 30)}
                               | {per_target}))
        report = progression.run(
            TARGETS, budgets=budgets,
            total_seconds=float(os.environ.get('FLEURDREAM_PROGRESSION_SECONDS', '1500')))
        output = os.environ.get('FLEURDREAM_PROGRESSION_REPORT')
        if output:
            Path(output).write_text(progression.json_report(report))
        self.assertEqual(report.unresolved, (),
                         f'{len(report.found)}/{len(TARGETS)} reached in {report.seconds:.1f}s; '
                         f'{report.stopped}; unresolved: {", ".join(report.unresolved)}')
        progression.validate_report(report)
        self.assertTrue(all(attempt.method == 'search' for attempt in report.found.values()))

    def test_spellcheck_dictionary_preserves_known_behavior_and_cascading_matches(self):
        def spellcheck(word):
            root = solver.starting_states(('spellcheck',), (word,))[0]
            return next(root.spellcheck())
        self.assertEqual(spellcheck('xxbiggervaseyy').word, 'xxyy')
        self.assertEqual(spellcheck('spikybetty').word, 'spikybetty')
        # Removing CHANGE exposes WATER. WATER must still be considered even
        # though it was not a substring of the original arrangement.
        result = spellcheck('xxwachangeteryy')
        self.assertEqual(result.word, 'xxyy')
        self.assertEqual(result.frozen, [False] * 4)
        for length in range(1, 25):
            self.assertEqual(solver.spellcheck_candidates(length), tuple(
                word for word in solver.SPELLCHECK_WORDS if len(word) <= length))


if __name__ == '__main__':
    unittest.main()
