"""A bounded discovery pass over legal edits suggested by target alignments.

Discarding candidates here affects discovery only. The solver's complete
frontier and admissible optimality bounds remain independent of this pass.
"""

import collections
import heapq
import itertools
from dataclasses import dataclass


def alignment_edits(state, goal, all_alignments=False, relax_locks=False):
    word = state.word
    n, m = len(word), len(goal)
    impossible = n + m + 1
    rows = [[0] * (m + 1) for _ in range(n + 1)]
    rows[0] = list(range(m + 1))
    for i in range(1, n + 1):
        cost = (2 if relax_locks else impossible) if state.frozen[i - 1] else 1
        previous, current = rows[i - 1], rows[i]
        for j in range(1, m + 1):
            current[j] = min(previous[j - 1] + (0 if word[i - 1] == goal[j - 1] else cost),
                             previous[j] + cost, current[j - 1] + 1)
    replacements, insertions, deletions = set(), set(), set()
    pending = [(end, m) for end in sorted(range(n + 1), key=lambda i: rows[i][m])[:4]]
    visited = set()
    while pending:
        i, j = pending.pop()
        if not j or (i, j) in visited:
            continue
        visited.add((i, j))
        cost = (2 if relax_locks else impossible) if i and state.frozen[i - 1] else 1
        if i and rows[i][j] == rows[i - 1][j - 1] + (0 if word[i - 1] == goal[j - 1] else cost):
            if word[i - 1] != goal[j - 1]:
                replacements.add((i - 1, goal[j - 1]))
            pending.append((i - 1, j - 1))
            if not all_alignments:
                continue
        if rows[i][j] == rows[i][j - 1] + 1:
            insertions.add((i, goal[j - 1]))
            pending.append((i, j - 1))
            if not all_alignments:
                continue
        if i and rows[i][j] == rows[i - 1][j] + cost:
            deletions.add(i - 1)
            pending.append((i - 1, j))
    return replacements, insertions, deletions


def successors(state, goal, engine, extended=False, relax_locks=False):
    pending = state.pending_recast()
    if pending is not None:
        # The general search makes the same commuting-cast reduction. Keep
        # raw transitions available to trace replay and correctness oracles.
        yield from state.search_successors()
        return
    replacements, insertions, deletions = alignment_edits(
        state, goal, all_alignments=extended,
        relax_locks=relax_locks and state.available('unlock_letter'))
    for i, letter in sorted(replacements):
        yield from state.change(i, letter)
        yield from state.alter(position=i, letter=letter)
        yield from state.adjacent(i, letter)
        yield from state.find_and_replace(source=state.word[i], destination=letter)
        for j, donor in enumerate(state.word[:engine.max_len + 1]):
            if donor != letter or state.frozen[j]:
                continue
            first, second = sorted((i, j))
            if first != second:
                yield from state.swap(first, second)
                if second == first + 1:
                    yield from state.swap_adjacent(first)
                if extended:
                    yield from state.move(j, i)
                    if j < i:
                        yield from state.move(j, i - 1)
    for gap, letter in sorted(insertions):
        yield from state.add(gap, letter)
        for i, donor in enumerate(state.word[:engine.max_len + 1]):
            if letter != donor or state.frozen[i]:
                continue
            yield from state.move(i, gap - (i < gap))
            if gap in (i, i + 1):
                yield from state.duplicate(i)
    for i in sorted(deletions):
        yield from state.subtract(i)
        if extended:
            # MOVE can remove junk from the target window even when the
            # transported letter never appears in the target. Put it beyond
            # that window; matching donors are only one useful kind of move.
            for end in (0, min(len(state.word) - 1, engine.max_len)):
                yield from state.move(i, end)
    if extended:
        bigrams = {goal[i:i + 2] for i in range(len(goal) - 1)}
        trigrams = {goal[i:i + 3] for i in range(len(goal) - 2)}
        for i, letter in enumerate(state.word[:engine.max_len + 1]):
            if goal.count(letter) > state.word.count(letter):
                yield from state.duplicate(i)
            if state.word[i:i + 2][::-1] in bigrams:
                yield from state.swap_adjacent(i)
            shortened = state.word[:i] + state.word[i + 1:]
            for j in (i - 2, i - 1):
                if j < 0:
                    continue
                chunk = shortened[j:j + 3]
                if len(chunk) == 3 and (chunk in trigrams or chunk[1] + chunk[0] + chunk[2]
                                       in trigrams or chunk[0] + chunk[2] + chunk[1] in trigrams):
                    yield from state.subtract(i)
                    break
    yield from state.chop()
    for child in state.drop():
        if extended or child.hist[-1].startswith('drop_n_'):
            yield child
    yield from state.spellcheck()
    yield from state.recast_spell()
    yield from state.most_frequent()
    yield from state.unlock_letter()
    gaps = {i for i, _ in insertions} | {i for i, _ in replacements}
    if extended:
        gaps.update((0, len(state.word)))
        if len(state.hist) == 1:
            gaps.update(range(len(state.word) + 1))
    for gap in sorted(gaps):
        yield from state.combine(include_drop=False, position=gap)


@dataclass
class Discovery:
    states: list | None = None
    expanded: int = 0
    generated: int = 0


@dataclass
class _Node:
    state: object
    parent: object = None

    def path(self):
        result = []
        current = self
        while current is not None:
            result.append(current.state)
            current = current.parent
        return result[::-1]


def discovery_key(state, coalesce_locks=False, min_lock_run=4):
    """Optional approximate grouping, never the complete search's state key.

    Retain a real representative of each group. Counts within a repeated
    letter run do not preserve reachability, so replay must keep its exact
    frozen positions and failure in this discovery pass proves nothing.
    """
    key = state.key()
    if not coalesce_locks:
        return key
    runs = []
    for _, group in itertools.groupby(zip(state.word, state.frozen), key=lambda item: item[0]):
        locks = tuple(frozen for _, frozen in group)
        runs.append(sum(locks) if len(locks) >= min_lock_run else locks)
    return key[0], tuple(runs), *key[2:]


def discover(goal, roots, engine, *, width=64, node_budget=2048,
             candidate_budget=150000, max_cost=None, family_width=4, extended=False,
             accept_state=None, resource_ordering=False, unlock_ordering=False,
             coalesce_locks=False, min_lock_run=4):
    result = Discovery()
    if not roots or width <= 0:
        return result
    available_cost = max(engine.spell_count(root.hist) + sum(not flag for flag in root.useds())
                         + int(root.available("recast_spell")) for root in roots)
    max_cost = available_cost if max_cost is None else min(max_cost, available_cost)
    layers = collections.defaultdict(list)
    serial = itertools.count()
    seen = {}
    repeated = [(letter, count) for letter, count in collections.Counter(goal).items() if count >= 3]

    def push(node):
        state = node.state
        cost = engine.spell_count(state.hist)
        if cost > max_cost:
            return
        feasible, lower_bound = state.search_bounds()
        if not feasible or cost + lower_bound > max_cost:
            return
        key = discovery_key(state, coalesce_locks, min_lock_run)
        if seen.get(key, max_cost + 1) <= cost:
            return
        seen[key] = cost
        distance = (state.heuristic_distance() if resource_ordering else
                    engine.bounded_substring_distance(
                        state.word, goal, state.frozen, len(goal),
                        allow_one_unlock=unlock_ordering and state.available('unlock_letter')))
        order = distance + engine.missing_letter_count(state.word, goal) * 0.15
        if repeated and state.available("find_and_replace"):
            supply = collections.Counter(letter for letter, frozen in zip(state.word, state.frozen)
                                         if not frozen)
            for wanted, needed in repeated:
                if wanted in state.word:
                    continue
                for donor, count in supply.most_common(2):
                    if count < 3:
                        continue
                    preview = ''.join(wanted if letter == donor and not frozen else letter
                                      for letter, frozen in zip(state.word, state.frozen))
                    # Ordering only: preview a later replacement. Actual
                    # candidates still enforce every freeze and absence rule.
                    hinted = engine.bounded_substring_distance(preview, goal, state.frozen, len(goal))
                    order = min(order, 1 + hinted + engine.missing_letter_count(preview, goal) * 0.15)
        heap = layers[cost]
        entry = (-order, next(serial), node)
        if len(heap) < width * 4:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)

    for root in roots:
        push(_Node(root))
    for depth in range(max_cost + 1):
        group = sorted(layers.pop(depth, ()), reverse=True)
        chosen = []
        families = collections.Counter()
        for _, _, node in group:
            state = node.state
            family = state.hist[0], tuple(state.useds())
            if extended:
                structure = []
                for i, command in enumerate(state.hist[1:], 1):
                    if not command.startswith('combine_'):
                        continue
                    parts = command.split('_')
                    position = int(parts[2] if parts[1].startswith('drop') else parts[1])
                    donor = parts[3] if parts[1].startswith('drop') else parts[2]
                    length = len(state.hist[i - 1].rsplit('_', 1)[-1])
                    bucket = 0 if position == 0 else 5 if position == length else 1 + position * 4 // (length + 1)
                    structure.append((donor, bucket))
                family += (tuple(structure),)
            if families[family] >= family_width:
                continue
            families[family] += 1
            chosen.append(node)
            if len(chosen) >= width:
                break
        for node in chosen:
            state = node.state
            if goal in state.word and (accept_state is None or accept_state(state)):
                result.states = node.path()
                return result
            remaining = max_cost - engine.spell_count(state.hist)
            finish = state.finish_with_insertion(max_casts=remaining)
            if finish is None:
                finish = engine.finish_with_replacements(state, max_casts=min(4, remaining))
            if finish is not None and (accept_state is None or accept_state(finish[-1])):
                result.states = node.path() + finish
                return result
            if result.expanded >= node_budget:
                return result
            result.expanded += 1
            for child in successors(state, goal, engine, extended=extended,
                                    relax_locks=unlock_ordering):
                result.generated += 1
                if result.generated > candidate_budget:
                    return result
                if accept_state is None and state.is_redundant_cast(child):
                    continue
                following = _Node(child, parent=node)
                if engine.spell_count(child.hist) > max_cost:
                    continue
                if goal in child.word and (accept_state is None or accept_state(child)):
                    result.states = following.path()
                    return result
                push(following)
    return result
