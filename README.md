# fleurdream

Solver for [flowereverie](https://store.steampowered.com/app/2672070/Flowereverie/).

This uses deterministic anytime branch-and-bound search with fair turns across
source words. It reports a solution quickly, then keeps looking for strictly
shorter solutions. The source words and available spells are configured in
`solver.py` (`used_... = True` disables a spell initially).

Spoiler alert!!!!!!!

```
pypy3 solver.py startthefestival
```

Python 3 also works. Targets are case insensitive. Press **Ctrl+C** to stop and
return the best solution found so far. There is no automatic time limit.

To stop after the first solution:

```
python3 solver.py recastspell --first
```

`New best` reports each improvement. `Best found` means optimality is unproved;
`Proved shortest` means safe bounds have ruled out every shorter continuation.
Proving optimality can take much longer than finding a useful solution.

Lengths count actual spell casts. Recast costs one even though it resets another
spell's `used_` flag. A `combine_drop…` row bundles two casts, so printed history
rows are not always the same as solution length.

The search shares duplicate detection across all source words. Its ordering
heuristic accounts for frozen letters, the absent-letter rule for add/change,
and which spells can supply an insertion or substitution. Bit operations compute
this distance without allocating a dynamic-programming matrix for every state.

Local edits get preferred turns. Combine, spellcheck and find-and-replace expand
lazily on regular deferred turns, or immediately when preferred queues empty.
Every legal branch remains searchable, including local setup before a bulk
spell and adding letters outside the target. A legal finishing `add` is checked
after each generated state.

After initial exploration, a bounded lookahead also tries
alter→spellcheck preparations, with and without combine. It detects edits that
create removable spell words, such as turning `yop` into `hop`, and gives the
most promising prepared states extra turns. These shortcuts use legal spells
and retain their full histories; the ordinary search queues stay intact.

Before reporting an improvement, the solver tries removing steps and legally
replaying the remaining spells with adjusted positions. It keeps a deletion
only if the replay still reaches the target, then repeats. This removes unused
work while preserving padding, freezing, and absent-letter setup when needed.

Pruning uses separate, optimistic letter-count and edit bounds. A branch is
discarded when its casts so far plus a safe lower bound cannot beat the best
solution. When combine, spellcheck, find-and-replace, unlock, or recast could
invalidate the edit bound, only the universal one-more-cast bound is used.
The richer ordering heuristic never discards a state or proves optimality.

The search remembers the cheapest arrival at each state and reopens a state if
a cheaper path is found. A separate lower-bound queue includes deferred bulk
operations, so they cannot be overlooked when proving the result optimal.

For Python callers, `solve(..., on_solution=callback)` calls
`callback(state, cast_count)` for every improvement. The returned result includes
`state`, `cost`, `optimal`, `interrupted`, and `stop_reason`. Pass
`first_solution=True` for the quick-result mode.

Run the distance, shortest-path oracle, interruption, deferred-operation, and
FREEDOM/COMBINE/KNOCK/CASHMONEY/RECASTSPELL regressions with:

```
python3 -m unittest -v
```

![screenshot](screenshot.png)
