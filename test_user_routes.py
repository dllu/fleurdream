import json
import os
import unittest
from pathlib import Path
from unittest import mock

import solver
import progression


ROUTES = json.loads((Path(__file__).parent / 'fixtures' / 'known_routes.json').read_text())


class UserRouteTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('FLEURDREAM_LONG_ROUTES') == '1',
                         'opt-in long-route discovery benchmark: FLEURDREAM_LONG_ROUTES=1')
    def test_solver_discovers_endgame_routes_with_frozen_unlocks(self):
        seconds = float(os.environ.get('FLEURDREAM_LONG_ROUTE_SECONDS', '180'))
        for target in ('startthefestival', 'walkthroughwalls', 'revealsakuyaexisted'):
            with self.subTest(target=target):
                route = ROUTES[target]
                technology = progression.Technology(frozenset(route['spells']), tuple(route['sources']))
                attempt = progression.search(target, technology, seconds)
                self.assertEqual(attempt.status, 'found',
                                 f'{target}: {attempt.status} after {attempt.seconds:.1f}s')
                progression.replay(target, attempt.history, technology)
                print(f'\n{target}: {attempt.cost} casts in {attempt.seconds:.3f}s', flush=True)

    def test_screenshot_routes_use_only_their_frozen_spells_and_sources(self):
        for case, route in ROUTES.items():
            target = route.get('target', case)
            with self.subTest(case=case), mock.patch.object(solver, 'target', target), mock.patch.multiple(
                    solver, max_len=route['max_position'],
                    tgt_abc=''.join(c for c in solver.abc if c in target)):
                state = next(root for root in solver.starting_states(route['spells'], route['sources'])
                             if root.word == route['root'])
                path = [state]
                for prefix in route['steps']:
                    state = next(child for child in state.generate_states_unlock()
                                 if child.hist[-1].startswith(prefix))
                    path.append(state)
                self.assertIn(target, state.word)
                self.assertEqual(solver.spell_count(state.hist), route['cost'])
                for i, ancestor in enumerate(path):
                    feasible, bound = ancestor.search_bounds()
                    self.assertTrue(feasible)
                    self.assertLessEqual(bound, route['cost'] - solver.spell_count(ancestor.hist))


if __name__ == '__main__':
    unittest.main()
