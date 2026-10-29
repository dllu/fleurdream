"""Bounded end-to-end progression audit; no installed game or save is needed."""

import argparse
import json
import multiprocessing
import os
import signal
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import solver


INITIAL_SOURCES = ("redpedalwalker", "lilachangerbloom")
SPELL_UNLOCKS = {name.replace("_", ""): name for name in solver.SPELL_NAMES}
# Alternatives are OR; the words within an alternative are AND. These rules
# are deliberately independent of the live CLI defaults and the player's save.
SOURCE_UNLOCKS = {
    "cespyoped": (frozenset(("lie",)),),
    "tigerwishgrantbud": (frozenset(("explosion", "swim")),),
    "honeycombbmineraleaf": (frozenset(("bridge",)),),
    "heedonicindigo": (frozenset(("breakandenter",)),),
    "violeepicturet": (frozenset(("cashmoney", "violeepicturet")),),
    "sakuya": tuple(frozenset(("startthefestival", criterion)) for criterion in (
        "grabsakuyasvase", "suitofarmor", "newwatertower")),
}
MAX_POSITION = 33
ALPHABET = "abcdefghijklmnopqrstuvwxyz"


@dataclass(frozen=True)
class Technology:
    spells: frozenset = frozenset()
    sources: tuple = INITIAL_SOURCES


def unlocked_technology(reached):
    reached = frozenset(reached)
    flowers = list(INITIAL_SOURCES)
    for flower, alternatives in SOURCE_UNLOCKS.items():
        if "walkthroughwalls" in reached or any(required <= reached for required in alternatives):
            flowers.append(flower)
    return Technology(frozenset(spell for word, spell in SPELL_UNLOCKS.items() if word in reached),
                      tuple(flowers))


@dataclass
class Attempt:
    target: str
    level: int
    technology: Technology
    budget: float
    status: str
    seconds: float
    cost: int | None = None
    history: list = field(default_factory=list)
    required: Technology | None = None
    expanded: int = 0
    method: str = "search"


def replay(target, history, technology):
    """Validate the actual solver trace and collect its direct dependencies."""
    solver.target = target
    solver.tgt_abc = "".join(c for c in ALPHABET if c in target)
    state = next((root for root in solver.starting_states(technology.spells, technology.sources)
                  if root.word == history[0]), None)
    if state is None:
        raise AssertionError("solution starts with a locked source")
    used_spells = set()
    used_sources = {state.word}
    for entry in history[1:]:
        child = next((s for s in state.generate_states_unlock() if s.hist[-1] == entry), None)
        if child is None:
            raise AssertionError(f"illegal cast: {entry}")
        for name in solver.SPELL_NAMES:
            if not getattr(state, "used_" + name) and getattr(child, "used_" + name):
                used_spells.add(name)
        kind, args = solver._operation(child)
        if kind == "combine":
            donor = args[-1]
            candidates = [word for word in technology.sources if (
                word[1:] if args[0] == "drop0" else word[:-1]
                if args[0] == "drop1" else word
            ) == donor]
            if not candidates:
                raise AssertionError("combine uses a locked donor")
            used_sources.add(next((word for word in candidates if word in used_sources), candidates[0]))
        state = child
    if target not in state.word:
        raise AssertionError("solution does not contain its target")
    required = Technology(frozenset(used_spells), tuple(
        word for word in technology.sources if word in used_sources))
    return state, required


def _search_worker(connection, target, technology):
    try:
        if os.name == 'posix':
            os.setsid()
        solver.max_len = MAX_POSITION
        solver.abc = ALPHABET
        result = solver.solve(target, allowed_spells=technology.spells,
                              source_words=technology.sources, first_solution=True)
        if result.interrupted:
            raise RuntimeError("worker search was interrupted")
        if result.state is None:
            connection.send(dict(status="exhausted", expanded=result.expanded))
        else:
            state, required = replay(target, result.state.hist, technology)
            # Verify that removing every unused spell/flower preserves this
            # witness. This gives a smaller reproducible technology profile.
            smaller, _ = replay(target, state.hist, required)
            if smaller.word != state.word or smaller.frozen != state.frozen:
                raise AssertionError("dependency reduction changed the solution")
            connection.send(dict(status="found", cost=solver.spell_count(state.hist),
                                 history=state.hist, required=required, expanded=result.expanded))
    except BaseException:
        connection.send(dict(status="error", error=traceback.format_exc()))
    finally:
        connection.close()


def search(target, technology, seconds, level=0):
    """Use a disposable process so a stalled search cannot hang the suite.

    A timeout says nothing about reachability. An exhausted search is a proof
    only for this exact technology profile. Worker exceptions remain failures.
    """
    if seconds <= 0:
        raise ValueError("search budget must be positive")
    method = "fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn"
    context = multiprocessing.get_context(method)
    receiving, sending = context.Pipe(duplex=False)
    process = context.Process(target=_search_worker, args=(sending, target, technology))
    start = time.perf_counter()
    process.start()
    sending.close()
    try:
        if not receiving.poll(seconds):
            payload = dict(status="timeout")
        else:
            try:
                payload = receiving.recv()
            except EOFError as error:
                raise RuntimeError(f"search worker for {target} exited without a result") from error
            if payload["status"] == "error":
                raise RuntimeError(payload["error"])
    finally:
        if process.is_alive():
            if os.name == 'posix':
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    process.terminate()
            else:
                process.terminate()
        process.join()
        receiving.close()
        process.close()
    return Attempt(target, level, technology, seconds, seconds=time.perf_counter() - start, **payload)


@dataclass
class ProgressionReport:
    targets: tuple
    attempts: list = field(default_factory=list)
    found: dict = field(default_factory=dict)
    reductions: dict = field(default_factory=dict)
    alternatives: dict = field(default_factory=dict)
    seconds: float = 0
    stopped: str = ""

    @property
    def unresolved(self):
        return tuple(target for target in self.targets if target not in self.found)

    @property
    def technology(self):
        return unlocked_technology(self.found)


def record_alternative(report, attempt):
    """Keep incomparable direct dependency sets, not one linear tech level."""
    if attempt.status != "found" or attempt.required is None:
        raise ValueError("a dependency alternative needs a verified solution")
    available = dependency_technology(report, exclude=(attempt.target,))
    if (not attempt.required.spells <= available.spells
            or not set(attempt.required.sources) <= set(available.sources)):
        # Relearning a word using itself, or tools obtainable only through it,
        # does not establish a new bootstrap route to that word.
        return False
    choices = report.alternatives.setdefault(attempt.target, [])
    def dominates(a, b):
        return (a.required.spells <= b.required.spells
                and set(a.required.sources) <= set(b.required.sources))
    for previous in choices:
        if dominates(previous, attempt):
            if not dominates(attempt, previous) or previous.cost <= attempt.cost:
                return False
    choices[:] = [previous for previous in choices if not dominates(attempt, previous)]
    choices.append(attempt)
    return True


def dependency_technology(report, exclude=()):
    """Close the recipe graph, respecting alternatives and avoiding cycles."""
    excluded = frozenset(exclude)
    reached = set()
    while True:
        available = unlocked_technology(reached)
        additions = set()
        for target, first in report.found.items():
            if target in reached or target in excluded:
                continue
            for recipe in report.alternatives.get(target, (first,)):
                required = recipe.required
                if (required is not None and required.spells <= available.spells
                        and set(required.sources) <= set(available.sources)):
                    additions.add(target)
                    break
        if not additions:
            return available
        reached.update(additions)


def run(targets=solver.GAME_SPELLS, *, budgets=(0.05, 0.25, 1, 5, 30, 240), total_seconds=1500,
        on_attempt=None, searcher=search, initial_report=None, on_checkpoint=None):
    """Try every target at each newly available technology, then deepen stalls.

    All attempts in a round use the previous round's discoveries. Thus a goal
    cannot use itself, a later discovery, or a source behind an unmet gate.
    Earlier profiles are tried first; timeouts prevent claiming minimum tech.
    """
    if not budgets or any(budget <= 0 for budget in budgets) or total_seconds <= 0:
        raise ValueError("budgets and total_seconds must be positive")
    targets = tuple(dict.fromkeys(targets))
    if any(target not in solver.GAME_SPELLS for target in targets):
        raise ValueError("target is not in the game spell catalog")
    if searcher is search and any(len(word) >= 16 or 'dddd' in word for word in targets):
        # Build once outside disposable workers, whose small bootstrap budgets
        # can otherwise expire during compilation rather than during search.
        import native_search
        native_search.ensure_binary()
    report = initial_report if initial_report is not None else ProgressionReport(targets)
    if report.targets != targets:
        raise ValueError("resumed report has a different target catalog")
    if initial_report is not None:
        validate_report(report)
    if not report.alternatives:
        for attempt in report.found.values():
            record_alternative(report, attempt)
    previous_seconds = report.seconds
    report.stopped = ""
    gates = frozenset().union(*(required for alternatives in SOURCE_UNLOCKS.values()
                               for required in alternatives), {"walkthroughwalls"})
    # Spending the remaining budget on a progression unlock is more useful
    # than an optional word. Both categories are still attempted every round.
    order = sorted(targets, key=lambda word: (
        0 if word in SPELL_UNLOCKS else 1 if word in gates else 2, word))
    start = time.perf_counter()
    level = max((attempt.level for attempt in report.found.values()), default=-1) + 1
    budget_index = 0
    exhausted = set()
    while report.unresolved:
        technology = report.technology
        pending_editors = any(word in SPELL_UNLOCKS for word in report.unresolved)
        for target in order:
            if target in report.found or (target, technology) in exhausted:
                continue
            remaining = total_seconds - (time.perf_counter() - start)
            if remaining <= 0:
                report.stopped = "total_budget"
                break
            budget = min(budgets[budget_index], remaining)
            if (report.technology != technology or (
                    pending_editors and target not in SPELL_UNLOCKS and target not in gates
                    and budget_index + 1 < len(budgets))):
                # Once this round has opened new technology, sample remaining
                # targets at the old profile, then use the new tools promptly.
                # Spending a long budget on each old-profile timeout delays
                # all subsequent unlocks without proving minimum technology.
                budget = min(budget, budgets[0], .05)
            attempt = searcher(target, technology, budget, level)
            if attempt.status not in ("found", "exhausted", "timeout"):
                raise AssertionError(f"invalid attempt status: {attempt.status}")
            report.attempts.append(attempt)
            if attempt.status == "found":
                report.found[target] = attempt
                record_alternative(report, attempt)
            elif attempt.status == "exhausted":
                exhausted.add((target, technology))
            if on_attempt is not None:
                on_attempt(attempt)
            if on_checkpoint is not None and attempt.status == "found":
                report.seconds = previous_seconds + time.perf_counter() - start
                on_checkpoint(report)
        if report.stopped:
            break
        if on_checkpoint is not None:
            report.seconds = previous_seconds + time.perf_counter() - start
            on_checkpoint(report)
        if report.technology != technology:
            level += 1
            budget_index = 0
        elif budget_index + 1 < len(budgets):
            budget_index += 1
        else:
            report.stopped = "stalled"
            break
    if not report.unresolved:
        report.stopped = "complete"
    report.seconds = previous_seconds + time.perf_counter() - start
    if on_checkpoint is not None:
        on_checkpoint(report)
    return report


def load_report(path):
    data = json.loads(Path(path).read_text())
    def technology(value):
        return Technology(frozenset(value["spells"]), tuple(value["sources"]))
    def attempt(value):
        value = dict(value)
        value["technology"] = technology(value["technology"])
        if value.get("required") is not None:
            value["required"] = technology(value["required"])
        return Attempt(**value)
    return ProgressionReport(
        tuple(data["targets"]), attempts=[attempt(value) for value in data["attempts"]],
        found={word: attempt(value) for word, value in data["found"].items()},
        reductions=data.get("reductions", {}), seconds=data["seconds"], stopped=data["stopped"],
        alternatives={word: [attempt(value) for value in values]
                      for word, values in data.get("alternatives", {}).items()},
    )


def validate_report(report):
    """Recheck every saved witness against strictly earlier unlocks."""
    prior = set()
    solver.max_len = MAX_POSITION
    solver.abc = ALPHABET
    levels = sorted({attempt.level for attempt in report.found.values()})
    for level in levels:
        available = unlocked_technology(prior)
        current = [attempt for attempt in report.found.values() if attempt.level == level]
        for attempt in current:
            if (attempt.target not in report.targets or report.found.get(attempt.target) is not attempt
                    or attempt.status != "found"):
                raise ValueError("invalid saved discovery")
            if (not attempt.technology.spells <= available.spells
                    or not set(attempt.technology.sources) <= set(available.sources)):
                raise ValueError(f"saved discovery uses a locked resource: {attempt.target}")
            state, required = replay(attempt.target, attempt.history, attempt.technology)
            if solver.spell_count(state.hist) != attempt.cost or required != attempt.required:
                raise ValueError(f"saved discovery has incorrect dependencies: {attempt.target}")
        prior.update(attempt.target for attempt in current)


def minimize(attempt, *, seconds=0.25, searcher=search):
    """Greedily remove direct requirements, retaining a solver-found witness.

    This proves inclusion-minimality only when every remaining single-resource
    removal was exhaustively ruled out. It does not prove minimum cardinality
    or a minimum transitive progression cost. Timeouts stay explicitly unknown.
    """
    current = attempt
    trials = []
    while True:
        technology = current.required
        candidates = [Technology(technology.spells - {spell}, technology.sources)
                      for spell in sorted(technology.spells)]
        if len(technology.sources) > 1:
            candidates += [Technology(technology.spells, tuple(
                word for word in technology.sources if word != source))
                for source in reversed(technology.sources)]
        for candidate in candidates:
            trial = searcher(current.target, candidate, seconds, current.level)
            trials.append(trial)
            if trial.status == "found":
                current = trial
                break
        else:
            minimal = all(trial.status == "exhausted" for trial in trials[-len(candidates):]) \
                if candidates else True
            return current, trials, minimal


def explore_dependencies(report, *, seconds=0.1, total_seconds=60, searcher=search):
    """Search smaller and incomparable toolsets, retaining their real recipes.

    The bootstrap graph excludes the target while constructing each portfolio,
    so a later spell that depends on the target cannot make a circular shortcut.
    Neither failed time-limited trials nor this finite portfolio prove a global
    minimum dependency set.
    """
    start = time.perf_counter()
    heavy = {'spellcheck', 'recast_spell', 'find_and_replace', 'unlock_letter',
             'most_frequent', 'swap_adjacent'}
    trials = []
    for target, original in tuple(report.found.items()):
        available = dependency_technology(report, exclude=(target,))
        known = report.alternatives.get(target, (original,))
        requirements = [attempt.required for attempt in known]
        portfolios = []
        for required in requirements:
            portfolios.extend(Technology(required.spells, (source,))
                              for source in available.sources)
            portfolios.extend((Technology(available.spells - heavy, required.sources),
                               Technology(available.spells - heavy, available.sources)))
        seen = set()
        for technology in portfolios:
            if technology in seen:
                continue
            seen.add(technology)
            remaining = total_seconds - (time.perf_counter() - start)
            if remaining <= 0:
                return trials
            trial = searcher(target, technology, min(seconds, remaining),
                             max(attempt.level for attempt in report.found.values()) + 1)
            trials.append(trial)
            if trial.status == 'found':
                smaller, reductions, minimal = minimize(trial, seconds=min(seconds, remaining),
                                                        searcher=searcher)
                trials.extend(reductions)
                for candidate in reductions:
                    if candidate.status == 'found':
                        record_alternative(report, candidate)
                record_alternative(report, smaller)
    return trials


def json_report(report):
    def encode(value):
        if isinstance(value, (frozenset, set)):
            return sorted(value)
        raise TypeError(type(value).__name__)
    data = asdict(report)
    data.update(unresolved=report.unresolved, unlocked=asdict(report.technology),
                assumptions={"max_position": MAX_POSITION, "alphabet": ALPHABET,
                             "sakuya": "STARTTHEFESTIVAL and any one festival criterion, or WALKTHROUGHWALLS"},
                minimum_tech="incomparable direct dependency sets; discovery rounds are timestamps, not tech levels")
    data["unresolved_status"] = {
        target: next((attempt.status for attempt in reversed(report.attempts)
                      if attempt.target == target), "not_attempted")
        for target in report.unresolved
    }
    return json.dumps(data, default=encode, indent=2) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds-per-target", type=float, default=240)
    parser.add_argument("--initial-seconds", type=float, default=0.05,
                        help="starting per-attempt budget, useful when resuming difficult targets")
    parser.add_argument("--total-seconds", type=float, default=1500)
    parser.add_argument("--resume", type=Path, help="resume a report after validating all saved witnesses")
    parser.add_argument("--report", type=Path, help="write all attempts and replayed witnesses as JSON")
    parser.add_argument("--minimize", action="store_true",
                        help="after discovery, try removing each required spell or flower")
    parser.add_argument("--minimize-seconds", type=float, default=0.05,
                        help="additional time budget per dependency-removal trial (default: 0.05)")
    parser.add_argument("--dependency-seconds", type=float, default=0,
                        help="additional total budget for alternative dependency profiles")
    args = parser.parse_args(argv)
    if (args.seconds_per_target <= 0 or args.total_seconds <= 0 or args.minimize_seconds <= 0
            or args.initial_seconds <= 0 or args.initial_seconds > args.seconds_per_target
            or args.dependency_seconds < 0):
        parser.error("time budgets must be positive")

    def progress(attempt):
        if attempt.status == "found":
            print(f"Discovery round {attempt.level}: {attempt.target}: {attempt.cost} casts "
                  f"({attempt.seconds:.3f}s)", flush=True)

    budgets = tuple(sorted({value for value in (0.05, 0.25, 1, 5, 30)
                            if args.initial_seconds <= value <= args.seconds_per_target}
                           | {args.initial_seconds, args.seconds_per_target}))
    def checkpoint(report):
        if args.report:
            temporary = args.report.with_name(args.report.name + ".tmp")
            temporary.write_text(json_report(report))
            temporary.replace(args.report)
    report = run(budgets=budgets, total_seconds=args.total_seconds, on_attempt=progress,
                 initial_report=load_report(args.resume) if args.resume else None,
                 on_checkpoint=checkpoint)
    if args.minimize:
        start = time.perf_counter()
        for target, attempt in report.found.items():
            smaller, trials, minimal = minimize(attempt, seconds=args.minimize_seconds)
            report.reductions[target] = dict(
                required=asdict(smaller.required), history=smaller.history, cost=smaller.cost,
                inclusion_minimal=minimal, trials=[asdict(trial) for trial in trials])
            record_alternative(report, smaller)
        report.seconds += time.perf_counter() - start
    if args.dependency_seconds:
        start = time.perf_counter()
        explore_dependencies(report, total_seconds=args.dependency_seconds)
        report.seconds += time.perf_counter() - start
    if args.report:
        args.report.write_text(json_report(report))
    print(f"Reached {len(report.found)}/{len(report.targets)} targets in {report.seconds:.1f}s; "
          f"stopped: {report.stopped}.")
    if report.unresolved:
        print("Unresolved: " + ", ".join(report.unresolved))
    return int(bool(report.unresolved))


if __name__ == "__main__":
    raise SystemExit(main())
