import argparse
import collections
import difflib
import heapq
import itertools
import sys
import time
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import List, Optional

# usage: pypy3 solver.py revealsakuyaexisted

target = ""

sources = [
    "honeycombbmineraleaf",
    "redpedalwalker",
    "lilachangerbloom",
    "cespyoped",
    "tigerwishgrantbud",
    "heedonicindigo",
    "violeepicturet",
    "sakuya",
]

SOURCE_CATALOG = (
    "honeycombbmineraleaf", "redpedalwalker", "lilachangerbloom", "cespyoped",
    "tigerwishgrantbud", "heedonicindigo", "violeepicturet", "sakuya",
)
SPELL_NAMES = (
    "alter", "move", "add", "adjacent", "combine", "change", "chop", "drop",
    "swap", "swap_adjacent", "subtract", "duplicate", "spellcheck", "recast_spell",
    "find_and_replace", "most_frequent", "unlock_letter",
)

# limits the max position to avoid infeasible actions due to long words being cut off
max_len = 33
abc = "abcdefghijklmnopqrstuvwxyz"
tgt_abc = abc
alter_map = {
    "d": "pbq",
    "b": "pdq",
    "p": "bdq",
    "q": "bdp",
    "a": "e",
    "e": "a",
    "n": "uc",
    "u": "cn",
    "c": "un",
    "m": "w",
    "w": "m",
    "h": "y",
    "y": "h",
}

# Primary spell catalog from the installed game, not just the editing spells.
# Flowereverie.exe: gml_Object_global_setter_Create_0 (100 entries).
# SHA-256: 8f22b3c71fa041e88fb4e723fb4e2bdb186bf2925b06bb957544ce95b50d9f66
GAME_SPELLS = (
    "add",
    "adjacent",
    "all",
    "alter",
    "balloon",
    "becomesakuya",
    "biggervase",
    "breakandenter",
    "bridge",
    "cash",
    "cashmoney",
    "change",
    "chat",
    "chop",
    "climb",
    "combine",
    "compost",
    "controltemperature",
    "crystalball",
    "dddddddddddie",
    "drop",
    "duplicate",
    "ebb",
    "explode",
    "explosion",
    "feed",
    "fertilizer",
    "findandreplace",
    "fish",
    "flow",
    "flowereverie",
    "freedom",
    "getout",
    "grab",
    "grabsakuyasvase",
    "greenthumb",
    "hang",
    "hide",
    "hint",
    "hop",
    "jumptheshark",
    "knock",
    "lie",
    "light",
    "magnetize",
    "map",
    "mine",
    "mirror",
    "money",
    "mostfrequent",
    "mostfrequentletter",
    "move",
    "mulch",
    "newwatertower",
    "one",
    "pesticides",
    "pet",
    "phone",
    "picture",
    "plant",
    "play",
    "proveinnocence",
    "readchalkboard",
    "recastspell",
    "redballoon",
    "remember",
    "revealsakuyaexisted",
    "revealsakuyawasntreal",
    "rob",
    "run",
    "sale",
    "seek",
    "shield",
    "shop",
    "showyourworth",
    "silhouette",
    "spellcheck",
    "spikybetty",
    "startthefestival",
    "subtract",
    "suitofarmor",
    "sunlight",
    "swap",
    "swapadjacent",
    "swim",
    "sword",
    "talk",
    "telephone",
    "teleport",
    "ticket",
    "two",
    "unlockletter",
    "violeepicturet",
    "wakeup",
    "walk",
    "walkthroughwalls",
    "wardrobe",
    "water",
    "watertower",
    "wheelbarrows",
)
# SPELLCHECK has a separate dictionary. Keep the established behavior: adding
# every target here would erase SPIKYBETTY in the known in-game solution.
SPELLCHECK_WORDS = (
    "add",
    "adjacent",
    "all",
    "alter",
    "balloon",
    "biggervase",
    "breakandenter",
    "bridge",
    "cash",
    "cashmoney",
    "change",
    "chat",
    "chop",
    "climb",
    "combine",
    "compost",
    "drop",
    "duplicate",
    "ebb",
    "explode",
    "explosion",
    "fertilizer",
    "fish",
    "freedom",
    "getout",
    "grab",
    "greenthumb",
    "hang",
    "hide",
    "hint",
    "hop",
    "lie",
    "light",
    "map",
    "mine",
    "money",
    "move",
    "mulch",
    "one",
    "pet",
    "phone",
    "picture",
    "plant",
    "redballoon",
    "rob",
    "sale",
    "seek",
    "shield",
    "shop",
    "silhouette",
    "spellcheck",
    "steal",
    "subtract",
    "sunlight",
    "swap",
    "swim",
    "sword",
    "talk",
    "telephone",
    "teleport",
    "ticket",
    "two",
    "violeepicturet",
    "wakeup",
    "walk",
    "water",
)
# Public target catalog, independent of the SPELLCHECK dictionary.
spells = GAME_SPELLS


def diff_strings(old, new):
    """
    Diffs two strings and returns the additions in green and deletions in red as a single string.

    Parameters:
    old (str): The old string to compare.
    new (str): The new string to compare.

    Returns:
    str: A string with colored additions and deletions.
    """
    # ANSI escape sequences for colors
    RED = "\033[91m"
    GREEN = "\033[92m"
    RESET = "\033[0m"

    # Generate the diff
    diff = difflib.ndiff(old, new)

    # Use a list to build the output string
    output_parts = []

    # Process the diff to create the output string
    for d in diff:
        if d.startswith("-"):
            output_parts.append(f"{RED}{d[2:]}{RESET}")
        elif d.startswith("+"):
            output_parts.append(f"{GREEN}{d[2:]}{RESET}")
        elif d.startswith(" "):
            output_parts.append(d[2:])

    # Join the list into a single string
    return "".join(output_parts)


def command_cost(entry):
    """Count actual casts, including the two-cast COMBINE/DROP shortcut."""
    if "_" not in entry:
        return 0  # Initial source word.
    return 2 if entry.startswith("combine_drop") else 1


def spell_count(history):
    return sum(command_cost(entry) for entry in history)


@dataclass
class State:
    word: str
    frozen: List
    used_alter: bool = False
    used_move: bool = False
    used_add: bool = False
    used_adjacent: bool = False
    used_combine: bool = False
    used_change: bool = False
    used_chop: bool = False
    used_drop: bool = False
    used_swap: bool = False
    used_swap_adjacent: bool = False
    used_subtract: bool = False
    used_duplicate: bool = False
    used_spellcheck: bool = False
    used_recast_spell: bool = False
    used_find_and_replace: bool = False
    used_most_frequent: bool = False
    used_unlock_letter: bool = False
    hist: List = field(default_factory=list)
    score_: Optional[float] = None
    dist: float = 0.0
    allowed_spells: Optional[frozenset] = field(default=None, repr=False)
    source_words: Optional[tuple] = field(default=None, repr=False)
    canonical_recast: Optional[str] = field(default=None, repr=False)

    def spell_allowed(self, name):
        return self.allowed_spells is None or name in self.allowed_spells

    def available(self, name):
        return not getattr(self, "used_" + name) and self.spell_allowed(name)

    def useds(self):
        return [
            self.used_alter,
            self.used_move,
            self.used_add,
            self.used_adjacent,
            self.used_combine,
            self.used_change,
            self.used_chop,
            self.used_drop,
            self.used_swap,
            self.used_swap_adjacent,
            self.used_subtract,
            self.used_duplicate,
            self.used_spellcheck,
            self.used_recast_spell,
            self.used_find_and_replace,
            self.used_most_frequent,
            self.used_unlock_letter,
        ]

    def frozen_str(self):
        return "".join("x" if x else "-" for x in self.frozen)

    def used_str(self):
        return "".join("1" if x else "0" for x in self.useds())

    def __repr__(self):
        lines = [self.word, self.frozen_str(), f"{self.score()} {self.dist}"]
        last_hist = None
        for i, c in enumerate(self.hist):
            hist = c.split("_")
            if last_hist is not None:
                colordiff = diff_strings(last_hist, hist[-1])
            else:
                colordiff = hist[-1]
            command = " ".join(hist[:-1])
            padding = " " * max(0, 40 - len(command))
            lines.append(f"    {i}: {command} {padding} {colordiff}")
            last_hist = hist[-1]
        return "\n".join(lines)

    def stringify(self):
        return self.word + "_" + self.frozen_str() + "_" + self.used_str()

    def key(self):
        return (self.word, tuple(self.frozen), tuple(self.useds()),
                self.allowed_spells, self.source_words, self.pending_recast())

    def pending_recast(self):
        """The spell to use next in the search's canonical cast ordering.

        This is a search restriction, not a game rule. Raw generators and
        replay still accept older solutions with a delayed use of that spell.
        """
        if (not self.canonical_recast or len(self.hist) < 2
                or not self.hist[-1].startswith("recast_")):
            return None
        return self.canonical_recast

    def canonical_successors(self, successors):
        pending = self.pending_recast()
        for child in successors:
            if pending is None or pending in command_spells(child.hist[-1]):
                yield child

    def is_redundant_cast(self, child):
        """Spending a spell without changing letters or locks cannot help.

        Keep raw generators unchanged for legal trace replay. Recast resets
        availability and unlock changes locks, so neither is rejected here.
        A pending recast has a separate search-order restriction; leave those
        states alone rather than assuming the broader continuation set.
        """
        return (self.pending_recast() is None and self.word == child.word
                and self.frozen == child.frozen
                and self.allowed_spells == child.allowed_spells
                and self.source_words == child.source_words
                and all(not before or after for before, after in zip(self.useds(), child.useds())))

    def search_successors(self, include_bulk=True, bulk_only=False):
        pending = self.pending_recast()
        bulk = ("combine", "spellcheck", "find_and_replace")
        method = getattr(self, pending, None) if pending else None
        if method is not None:
            if pending in bulk:
                if include_bulk:
                    yield from method()
            elif not bulk_only:
                yield from method()
            if pending == "drop" and include_bulk:
                # Donor preparation can bundle DROP with COMBINE.
                for child in self.combine():
                    if child.hist[-1].startswith("combine_drop"):
                        yield child
            return
        raw = (self.generate_bulk_states() if bulk_only else
               self.generate_states_unlock(include_bulk=include_bulk))
        yield from self.canonical_successors(raw)

    def finish_with_add(self):
        if not self.available("add"):
            return None
        insertion = finishing_insertion(self.word, target, max_len)
        if insertion is None:
            return None
        i, c = insertion
        new_word = self.word[:i] + c + self.word[i:]
        return replace(
            self,
            word=new_word,
            frozen=self.frozen[:i] + [True] + self.frozen[i:],
            used_add=True,
            hist=self.hist + [f"add_{i}_{c}_{new_word}"],
            score_=None,
        )

    def finish_with_insertion(self, max_casts=None):
        """Complete a one-letter gap by ADD, MOVE, or DUPLICATE then MOVE.

        A spent MOVE can first be recast. Every returned intermediate state
        comes from the real spell generators and remains in the solution trace.
        """
        if max_casts is not None and max_casts < 1:
            return None
        added = self.finish_with_add()
        if added is not None:
            return [added]
        gaps = finishing_existing_insertions(self.word, target, max_len)
        if not gaps:
            return self.finish_with_prepared_add(gaps, max_casts)
        if self.available("duplicate"):
            for gap, letter in gaps:
                for i in (gap - 1, gap):
                    if (0 <= i < len(self.word) and i <= max_len
                            and self.word[i] == letter and not self.frozen[i]):
                        copied = next(self.duplicate(i), None)
                        if copied is not None and target in copied.word:
                            return [copied]
        if not self.available("move"):
            unlocked = self.finish_with_unlocked_insertion(gaps, max_casts)
            if unlocked is not None:
                return unlocked
            prepared = self.finish_with_prepared_add(gaps, max_casts)
            if prepared is not None:
                return prepared
            if (not self.available("recast_spell") or not self.spell_allowed("move")
                    or (max_casts is not None and max_casts < 2)):
                return None
            movable = next(s for s in self.recast_spell() if not s.used_move)
            prefix = [movable]
        else:
            movable = self
            prefix = []
        # An existing donor outside the goal can simply be moved into the gap.
        for gap, letter in gaps:
            for i, c in enumerate(self.word[:max_len + 1]):
                if c != letter or self.frozen[i]:
                    continue
                destination = gap - (i < gap)
                without = self.word[:i] + self.word[i + 1:]
                word = without[:destination] + c + without[destination:]
                if target in word:
                    moved = next(movable.move(i, destination), None)
                    if moved is not None:
                        return prefix + [moved]

        if not prefix:
            unlocked = self.finish_with_unlocked_insertion(gaps, max_casts)
            if unlocked is not None:
                return unlocked
            prepared = self.finish_with_prepared_add(gaps, max_casts)
            if prepared is not None:
                return prepared

        if not self.available("duplicate") or (max_casts is not None and len(prefix) + 2 > max_casts):
            return None
        # Preserve the donor in the goal by duplicating it. The new first copy
        # freezes; the unfrozen original is now at i+1 and can be transported.
        for gap, letter in gaps:
            for i, c in enumerate(self.word[:max_len]):
                if c != letter or self.frozen[i]:
                    continue
                copied = next(movable.duplicate(i), None)
                if copied is None:
                    continue
                moved = next(copied.move(i + 1, gap), None)
                if moved is not None and target in moved.word:
                    return prefix + [copied, moved]
        return None

    def finish_with_unlocked_insertion(self, gaps, max_casts=None):
        """Unlock a frozen donor, then copy or transport it into the gap."""
        if not self.available("unlock_letter") or max_casts is not None and max_casts < 2:
            return None
        if not (self.available("duplicate") or self.available("move")):
            return None
        for gap, letter in gaps:
            for i, char in enumerate(self.word[:max_len + 1]):
                if char != letter or not self.frozen[i]:
                    continue
                unlocked = next(self.unlock_letter(i))
                if i in (gap - 1, gap):
                    for copied in unlocked.duplicate(i):
                        if target in copied.word:
                            return [unlocked, copied]
                for moved in unlocked.move(i, gap - (i < gap)):
                    if target in moved.word:
                        return [unlocked, moved]
        return None

    def finish_with_prepared_add(self, gaps, max_casts=None):
        """Cross a neutral setup cast that makes the final ADD legal.

        Removing or changing an existing letter can enable ADD's absent-letter
        rule. Recasting a spent ADD can also fill a gap. Both actual casts are
        retained and checked with the ordinary generators.
        """
        if max_casts is not None and max_casts < 2:
            return None
        if (self.used_add and self.spell_allowed("add")
                and self.available("recast_spell")):
            restored = next(s for s in self.recast_spell() if not s.used_add)
            added = restored.finish_with_add()
            if added is not None:
                return [restored, added]
        if not self.available("add") or not gaps:
            return None
        candidates = itertools.chain(self.chop(), self.drop(), self.spellcheck())
        for prepared in candidates:
            added = prepared.finish_with_add()
            if added is not None:
                return [prepared, added]
        for letter in dict.fromkeys(letter for _, letter in gaps):
            if self.word.count(letter) == 1:
                i = self.word.index(letter)
                for prepared in itertools.chain(self.subtract(i), self.alter(position=i),
                                                 self.change(position=i)):
                    added = prepared.finish_with_add()
                    if added is not None:
                        return [prepared, added]
            for prepared in self.find_and_replace(source=letter):
                added = prepared.finish_with_add()
                if added is not None:
                    return [prepared, added]
        return None

    def edit_budget(self, relax_deletion_and_locks=False):
        # These spells can change many letters or restore an arbitrary spell.
        # The local edit bound below is only valid once all of them are spent.
        if not (
            self.used_combine
            and self.used_find_and_replace
            and self.used_recast_spell
        ):
            return None
        if not relax_deletion_and_locks and not (self.used_spellcheck and self.used_unlock_letter):
            return None
        insertions = sum(not used for used in (
            self.used_add, self.used_duplicate, self.used_most_frequent, self.used_move
        ))
        deletions = sum(not used for used in (self.used_subtract, self.used_move, self.used_drop))
        substitutions = sum(not used for used in (
            self.used_change, self.used_alter, self.used_adjacent
        )) + 2 * sum(not used for used in (self.used_swap, self.used_swap_adjacent))
        return insertions, deletions, substitutions

    def search_bounds(self):
        """Return feasibility and an admissible lower bound on remaining casts.

        Bulk spells, unlock and recast can invalidate the edit relaxation. In
        that case, only the universal one-cast bound for a non-goal is used.
        """
        if target in self.word:
            return True, 0
        budget = self.edit_budget()
        free_delete = False
        relaxed_locks = False
        if budget is None:
            budget = self.edit_budget(relax_deletion_and_locks=True)
            if budget is None:
                return self.repetition_feasible(), 1
            # SPELLCHECK only deletes: granting arbitrary free deletions is a
            # safe relaxation, including deletion of frozen letters. UNLOCK
            # is relaxed by treating all letters as editable at no extra cost.
            free_delete = not self.used_spellcheck
            relaxed_locks = not self.used_unlock_letter
        # Moving, swapping, chopping and dropping cannot increase letter counts.
        growth = sum(not used for used in (
            self.used_add, self.used_duplicate, self.used_most_frequent,
            self.used_change, self.used_alter, self.used_adjacent,
        ))
        missing = missing_letter_count(self.word, target)
        if missing > growth:
            return False, missing
        preserving_growth = sum(not used for used in (
            self.used_duplicate, self.used_most_frequent, self.used_alter, self.used_adjacent))
        for letter, needed in target_counts(target):
            present = self.word.count(letter)
            # If this letter never disappears, ADD/CHANGE cannot create it.
            # If it does disappear, every final copy needs a remaining growth
            # cast. Deletions and unlocking cannot increase letter counts.
            if needed > max(present + preserving_growth, growth):
                return False, max(1, needed - present)
        insertions, deletions, substitutions = budget
        substitution_masks = None
        if not relaxed_locks and self.used_change and self.used_swap and self.used_swap_adjacent:
            # These direct substitutions form a safe relaxation once all bulk
            # spells, unlock and recast are spent: edited letters freeze.
            _, alters, adjacent = target_edit_masks(target)
            substitution_masks = {
                c: (0 if self.used_alter else alters.get(c, 0))
                | (0 if self.used_adjacent else adjacent.get(c, 0))
                for c in set(self.word)
            }
        bound = bounded_substring_distance(
            self.word, target, [False] * len(self.word) if relaxed_locks else self.frozen, sum(budget),
            allow_insert=insertions > 0,
            allow_delete=free_delete or deletions > 0,
            allow_substitute=substitutions > 0,
            substitution_masks=substitution_masks,
            free_delete=free_delete,
            delete_frozen=not self.used_drop,
        )
        per_cast = 2 if not (self.used_move and self.used_swap and self.used_swap_adjacent) else 1
        lower_bound = max(1, missing, (bound + per_cast - 1) // per_cast)
        return bound <= sum(budget), lower_bound

    def repetition_feasible(self):
        """Optimistic count caps while bulk operations still remain.

        New letters from local edits are frozen. Only COMBINE and UNLOCK can
        increase the supply of unfrozen letters for a future FINDANDREPLACE.
        Position, deletion and resource conflicts are deliberately relaxed.
        """
        repeated = tuple((c, n) for c, n in target_counts(target) if n >= 5)
        if not repeated:
            return True
        recast = self.available("recast_spell")
        combines = (int(self.available("combine"))
                    + int(recast and self.spell_allowed("combine")))
        donors = self.source_words if self.source_words is not None else tuple(sources)
        donor_counts = source_letter_maxima(tuple(donors)) if combines else {}
        preserving = ("duplicate", "most_frequent", "alter", "adjacent")
        growth = preserving + ("add", "change")
        preserve_growth = sum(self.available(name) for name in preserving) + int(
            recast and any(self.spell_allowed(name) for name in preserving))
        reset_growth = sum(self.available(name) for name in growth) + int(
            recast and any(self.spell_allowed(name) for name in growth))
        replacement = self.available("find_and_replace") or (
            recast and self.spell_allowed("find_and_replace"))
        unfrozen = collections.Counter(c for c, frozen in zip(self.word, self.frozen) if not frozen)
        for letter, needed in repeated:
            supplied = combines * donor_counts.get(letter, 0)
            # ADD/CHANGE cannot increase a letter while it remains present.
            # If it disappears, grant every growth cast and every new donor.
            maximum = max(self.word.count(letter) + supplied + preserve_growth,
                          supplied + reset_growth)
            if replacement:
                # Consider the last replacement creating this letter. Earlier
                # copies must have disappeared because its destination must
                # be absent. All new copies freeze, so ADD/CHANGE cannot add
                # to that batch. Grant all combines both before and after it,
                # and reuse recast optimistically across conflicting choices.
                batch = max((unfrozen[c] + combines * donor_counts.get(c, 0)
                             for c in set(unfrozen) | set(donor_counts) if c != letter), default=0)
                maximum = max(maximum, batch + int(self.available("unlock_letter"))
                              + supplied + preserve_growth)
            if needed > maximum:
                return False
        return True

    def can_reach(self):
        return self.search_bounds()[0]

    def heuristic_distance(self):
        """Estimate local edits using letters each remaining spell can supply.

        This deliberately ignores some setup moves and bulk spells. It is an
        ordering hint, never a pruning bound; can_reach uses its own relaxation.
        """
        masks, alters, adjacent = target_edit_masks(target)
        missing = 0
        for c, mask in masks.items():
            if c not in self.word:
                missing |= mask
        movable = 0
        for c, frozen in zip(self.word, self.frozen):
            if not frozen:
                movable |= masks.get(c, 0)

        # Recast can restore a spent MOVE. Include that transport possibility
        # in the ordering hint; the real recast still costs a cast in search.
        can_move = self.available("move") or (
            self.available("recast_spell") and self.spell_allowed("move")
        )
        insertion_mask = 0 if self.used_add else missing
        if can_move:
            insertion_mask |= movable
        if not self.used_most_frequent:
            # Ignore frequency ties here: an optimistic supply is useful for
            # ordering, and later edits may change which letter is most common.
            for c in set(self.word):
                insertion_mask |= masks.get(c, 0)
        insertion_masks = None
        if not self.used_duplicate and not can_move:
            # DUPLICATE supplies a copy beside its source letter, not at an
            # arbitrary gap elsewhere in the word. MOVE has its own global
            # supply above and can only be restored by an actual recast.
            insertion_masks = [insertion_mask] * (len(self.word) + 1)
            for i, (c, frozen) in enumerate(zip(self.word, self.frozen)):
                if not frozen:
                    insertion_masks[i] |= masks.get(c, 0)
                    insertion_masks[i + 1] |= masks.get(c, 0)

        # FINDANDREPLACE can supply an absent letter even after CHANGE was
        # spent. Ignoring it makes useful preparations appear impossible.
        generic = missing if self.available("change") or self.available("find_and_replace") else 0
        if not self.used_swap or not self.used_swap_adjacent:
            generic |= movable
        substitution_masks = {
            c: generic | (0 if self.used_alter else alters.get(c, 0))
            | (0 if self.used_adjacent else adjacent.get(c, 0))
            for c in set(self.word)
        }
        distance = bounded_substring_distance(
            self.word, target, self.frozen, len(target),
            allow_delete=can_move or not self.used_subtract,
            insertion_mask=insertion_mask,
            insertion_masks=insertion_masks,
            substitution_masks=substitution_masks,
            allow_one_unlock=self.available("unlock_letter"),
        )
        return distance

    def score(self):
        if self.score_ is None:
            self.dist = self.heuristic_distance()
            self.score_ = (
                100
                - self.dist
                - spell_count(self.hist) * 0.4
                - missing_letter_count(self.word, target) * 0.2
            )
        return self.score_

    def __lt__(self, other) -> bool:
        return self.score() > other.score()

    def recast_spell(self):
        if not self.available("recast_spell"):
            return
        for name in SPELL_NAMES:
            if name in ("recast_spell", "unlock_letter") or not self.spell_allowed(name):
                continue
            flag = "used_" + name
            if not getattr(self, flag):
                continue
            label = "mostf" if name == "most_frequent" else name
            yield replace(
                self, **{flag: False}, used_recast_spell=True,
                canonical_recast=name,
                hist=self.hist + [f"recast_{label}_{self.word}"], score_=None,
            )

    def generate_states_unlock(self, include_bulk=True):
        yield from self.generate_states(include_bulk=include_bulk)
        yield from self.unlock_letter()

    def unlock_letter(self, position=None):
        if not self.available("unlock_letter"):
            return
        for i, frozen in enumerate(self.frozen):
            if i > max_len:
                break
            if not frozen or position is not None and i != position:
                continue
            new_frozen = self.frozen.copy()
            new_frozen[i] = False
            yield replace(
                self,
                frozen=new_frozen,
                used_unlock_letter=True,
                hist=self.hist + [f"unlock_{i}_{self.word[i]}_{self.word}"],
                score_=None,
            )

    def generate_bulk_states(self):
        yield from self.combine()
        yield from self.spellcheck()
        yield from self.find_and_replace()

    def generate_states(self, include_bulk=True):
        if include_bulk:
            yield from self.generate_bulk_states()

        if self.available("recast_spell"):
            yield from self.recast_spell()

        yield from self.chop()

        yield from self.drop()

        yield from self.alter()

        yield from self.move()

        yield from self.subtract()

        yield from self.duplicate()

        yield from self.adjacent()

        yield from self.add()
        yield from self.most_frequent()
        yield from self.change()
        yield from self.swap()
        yield from self.swap_adjacent()

    def most_frequent(self, position=None):
        if self.available("most_frequent"):
            most_common = collections.Counter(self.word).most_common(2)
            if most_common and (len(most_common) == 1 or most_common[0][1] > most_common[1][1]):
                c = most_common[0][0]
                for i in range(len(self.word) + 1):
                    if i > max_len:
                        break
                    if position is not None and i != position:
                        continue
                    new_word = self.word[:i] + c + self.word[i:]
                    new_frozen = self.frozen[:i] + [True] + self.frozen[i:]
                    new_self = replace(
                        self,
                        word=new_word,
                        frozen=new_frozen,
                        used_most_frequent=True,
                        hist=self.hist + [f"mostf_{i}_{c}_{new_word}"],
                        score_=None,
                    )
                    yield new_self

    def add(self, position=None, letter=None):
        n = len(self.word)
        if self.available("add"):
            # Prefer target letters, but retain setup moves involving others.
            for c in tgt_abc + "".join(c for c in abc if c not in tgt_abc):
                if letter is not None and c != letter:
                    continue
                if c in self.word:
                    continue
                for i in range(n + 1):
                    if position is not None and i != position:
                        continue
                    if i > max_len:
                        break
                    new_word = self.word[:i] + c + self.word[i:]
                    new_frozen = self.frozen[:i] + [True] + self.frozen[i:]
                    new_self = replace(
                        self,
                        word=new_word,
                        frozen=new_frozen,
                        used_add=True,
                        hist=self.hist + [f"add_{i}_{c}_{new_word}"],
                        score_=None,
                    )
                    yield new_self


    def adjacent(self, position=None, letter=None):
        if self.available("adjacent"):
            for i, ci in enumerate(self.word):
                if position is not None and i != position:
                    continue
                if i > max_len:
                    break
                if self.frozen[i]:
                    continue
                if ci < "z" and (letter is None or letter == chr(ord(ci) + 1)):
                    ci0 = chr(ord(ci) + 1)
                    new_word = self.word[:i] + ci0 + self.word[i + 1 :]
                    new_frozen = self.frozen.copy()
                    new_frozen[i] = True
                    new_self = replace(
                        self,
                        word=new_word,
                        frozen=new_frozen,
                        used_adjacent=True,
                        hist=self.hist + [f"adj+_{i}_{ci}_{new_word}"],
                        score_=None,
                    )
                    yield new_self

                if ci > "a" and (letter is None or letter == chr(ord(ci) - 1)):
                    ci0 = chr(ord(ci) - 1)
                    new_word = self.word[:i] + ci0 + self.word[i + 1 :]
                    new_frozen = self.frozen.copy()
                    new_frozen[i] = True
                    new_self = replace(
                        self,
                        word=new_word,
                        frozen=new_frozen,
                        used_adjacent=True,
                        hist=self.hist + [f"adj-_{i}_{ci}_{new_word}"],
                        score_=None,
                    )
                    yield new_self


    def move(self, source_index=None, destination_index=None):
        n = len(self.word)
        if self.available("move"):
            for i, ci in enumerate(self.word):
                if source_index is not None and i != source_index:
                    continue
                if i > max_len:
                    break
                if self.frozen[i]:
                    continue
                word_without_i = self.word[:i] + self.word[i + 1 :]
                frozen_without_i = self.frozen[:i] + self.frozen[i + 1 :]
                for j in range(n):
                    if destination_index is not None and j != destination_index:
                        continue
                    if j > max_len:
                        break
                    new_word = word_without_i[:j] + ci + word_without_i[j:]
                    new_frozen = frozen_without_i[:j] + [True] + frozen_without_i[j:]
                    new_self = replace(
                        self,
                        word=new_word,
                        frozen=new_frozen,
                        used_move=True,
                        hist=self.hist + [f"move_{i}_{j}_{new_word}"],
                        score_=None,
                    )
                    yield new_self

    def duplicate(self, position=None):
        if self.available("duplicate"):
            for i, ci in enumerate(self.word):
                if position is not None and i != position:
                    continue
                if i > max_len:
                    break
                if self.frozen[i]:
                    continue
                new_word = self.word[:i] + ci + ci + self.word[i + 1 :]
                new_frozen = self.frozen[:i] + [True, False] + self.frozen[i + 1 :]
                new_self = replace(
                    self,
                    word=new_word,
                    frozen=new_frozen,
                    used_duplicate=True,
                    hist=self.hist + [f"dup_{i}_{ci}_{new_word}"],
                    score_=None,
                )
                yield new_self

    def drop(self):
        if not self.available("drop"):
            return
        seen = set()
        for i in (0, len(self.word) - 1):
            if i in seen or i < 0 or i > max_len:
                continue
            seen.add(i)
            word, frozen = self.word[:i] + self.word[i + 1:], self.frozen[:i] + self.frozen[i + 1:]
            yield replace(self, word=word, frozen=frozen, used_drop=True,
                          hist=self.hist + [f"drop_n_{word}"], score_=None)
        # A source name can survive a combine or be recreated by other casts.
        # Its boundary letters can be dropped even inside the arrangement.
        donor_names = self.source_words if self.source_words is not None else tuple(sources)
        for flower in drop_flower_names(tuple(donor_names)):
            start = self.word.find(flower)
            while start >= 0:
                for i in (start, start + len(flower) - 1):
                    if i in seen or i > max_len:
                        continue
                    seen.add(i)
                    word = self.word[:i] + self.word[i + 1:]
                    yield replace(self, word=word, frozen=self.frozen[:i] + self.frozen[i + 1:],
                                  used_drop=True, hist=self.hist + [f"drop_{i}_{self.word[i]}_{word}"],
                                  score_=None)
                start = self.word.find(flower, start + 1)

    def subtract(self, position=None):
        if not self.available("subtract"):
            return
        for i, c in enumerate(self.word[:max_len + 1]):
            if (position is not None and i != position) or self.frozen[i]:
                continue
            word = self.word[:i] + self.word[i + 1:]
            yield replace(self, word=word, frozen=self.frozen[:i] + self.frozen[i + 1:],
                          used_subtract=True, hist=self.hist + [f"sub_{i}_{c}_{word}"], score_=None)

    def alter(self, for_spellcheck=False, position=None, letter=None):
        if self.available("alter"):
            for i, ci in enumerate(self.word):
                if position is not None and i != position:
                    continue
                if i > max_len:
                    break
                if self.frozen[i]:
                    continue
                if ci not in alter_map:
                    continue
                for ca in alter_map[ci]:
                    if letter is not None and ca != letter:
                        continue
                    if for_spellcheck and not creates_spell_at(self.word, i, ca):
                        continue
                    new_word = self.word[:i] + ca + self.word[i + 1 :]
                    new_frozen = self.frozen.copy()
                    new_frozen[i] = True
                    new_self = replace(
                        self,
                        word=new_word,
                        frozen=new_frozen,
                        used_alter=True,
                        hist=self.hist + [f"alter_{i}_{ca}_{new_word}"],
                        score_=None,
                    )
                    yield new_self

    def spellcheck_preparations(self):
        """Legal ALTER -> SPELLCHECK pairs that create a removable spell."""
        if not self.available("spellcheck"):
            return
        for altered in self.alter(for_spellcheck=True):
            for checked in altered.spellcheck():
                if checked.word != altered.word:
                    yield altered, checked

    def find_and_replace(self, source=None, destination=None):
        if self.available("find_and_replace"):
            src_chars = sorted(set(self.word))
            tgt_chars = sorted(set(abc) - set(self.word))

            for src in src_chars:
                if source is not None and src != source:
                    continue
                for tgt in tgt_chars:
                    if destination is not None and tgt != destination:
                        continue
                    new_word = list(self.word)
                    new_frozen = self.frozen.copy()
                    for i, ci in enumerate(self.word):
                        if ci == src and not self.frozen[i]:
                            new_word[i] = tgt
                            new_frozen[i] = True
                    new_word = "".join(new_word)
                    new_self = replace(
                        self,
                        word=new_word,
                        frozen=new_frozen,
                        used_find_and_replace=True,
                        hist=self.hist + [f"fandr_{src}_{tgt}_{new_word}"],
                        score_=None,
                    )
                    yield new_self

    def chop(self):
        n = len(self.word)
        if not self.available("chop"):
            return
        if n % 2 == 0:
            new_word = self.word[: n // 2]
            yield replace(
                self,
                word=new_word,
                frozen=self.frozen.copy()[: n // 2],
                used_chop=True,
                hist=self.hist + [f"chop_0_{new_word}"],
                score_=None,
            )

            new_word = self.word[n // 2 :]
            yield replace(
                self,
                word=new_word,
                frozen=self.frozen.copy()[n // 2 :],
                used_chop=True,
                hist=self.hist + [f"chop_1_{new_word}"],
                score_=None,
            )
        else:
            new_word = self.word[: n // 2 + 1]
            yield replace(
                self,
                word=new_word,
                frozen=self.frozen.copy()[: n // 2 + 1],
                used_chop=True,
                hist=self.hist + [f"chop_0_{new_word}"],
                score_=None,
            )

            new_word = self.word[n // 2 :]
            yield replace(
                self,
                word=new_word,
                frozen=self.frozen.copy()[n // 2 :],
                used_chop=True,
                hist=self.hist + [f"chop_1_{new_word}"],
                score_=None,
            )

    def change(self, position=None, letter=None):
        n = len(self.word)
        if not self.available("change"):
            return
        for c in abc:
            if letter is not None and c != letter:
                continue
            if c in self.word:
                continue
            for i in range(n):
                if position is not None and i != position:
                    continue
                if i > max_len:
                    break
                if self.frozen[i]:
                    continue
                new_word = self.word[:i] + c + self.word[i + 1 :]
                new_frozen = self.frozen.copy()
                new_frozen[i] = True
                new_self = replace(
                    self,
                    word=new_word,
                    frozen=new_frozen,
                    used_change=True,
                    hist=self.hist + [f"change_{i}_{c}_{new_word}"],
                    score_=None,
                )
                yield new_self

    def swap(self, first=None, second=None):
        if not self.available("swap"):
            return
        for i, ci in enumerate(self.word):
            if first is not None and i != first:
                continue
            if i > max_len:
                break
            if self.frozen[i]:
                continue
            for j in range(i + 1, min(len(self.word), max_len + 1)):
                if second is not None and j != second:
                    continue
                cj = self.word[j]
                if self.frozen[j]:
                    continue
                lw = list(self.word)
                lw[j] = ci
                lw[i] = cj
                new_frozen = self.frozen.copy()
                new_frozen[i] = True
                new_frozen[j] = True
                new_word = "".join(lw)
                new_self = replace(
                    self,
                    word=new_word,
                    frozen=new_frozen,
                    used_swap=True,
                    hist=self.hist + [f"swap_{i}{ci}_{j}{cj}_{new_word}"],
                    score_=None,
                )
                yield new_self

    def swap_adjacent(self, position=None):
        if not self.available("swap_adjacent"):
            return
        for i, ci in enumerate(self.word):
            if position is not None and i != position:
                continue
            if i > max_len:
                break
            if self.frozen[i]:
                continue
            j = i + 1
            if j >= len(self.word) or j > max_len or self.frozen[j]:
                continue
            cj = self.word[j]
            lw = list(self.word)
            lw[j] = ci
            lw[i] = cj
            new_frozen = self.frozen.copy()
            new_frozen[i] = True
            new_frozen[j] = True
            new_word = "".join(lw)
            yield replace(
                self,
                word=new_word,
                frozen=new_frozen,
                used_swap_adjacent=True,
                hist=self.hist + [f"swap_adj_{i}{ci}_{j}{cj}_{new_word}"],
                score_=None,
            )

    def combine(self, include_drop=True, position=None, source=None):
        if not self.available("combine"):
            return
        n = len(self.word)
        for c in self.source_words if self.source_words is not None else sources:
            if source is not None and c != source:
                continue
            m = len(c)
            for i in range(n + 1):
                if i > max_len:
                    break
                if position is not None and i != position:
                    continue
                new_word = self.word[:i] + c + self.word[i:]
                new_frozen = self.frozen[:i] + [False] * m + self.frozen[i:]
                new_self = replace(
                    self,
                    word=new_word,
                    frozen=new_frozen,
                    used_combine=True,
                    hist=self.hist + [f"combine_{i}_{c}_{new_word}"],
                    score_=None,
                )
                yield new_self
            if include_drop and self.available("drop"):
                c1 = c[:-1]
                m = len(c1)
                for i in range(n + 1):
                    if i + len(c) - 1 > max_len:
                        break
                    if position is not None and i != position:
                        continue
                    new_word = self.word[:i] + c1 + self.word[i:]
                    new_frozen = self.frozen[:i] + [False] * m + self.frozen[i:]
                    new_self = replace(
                        self,
                        word=new_word,
                        frozen=new_frozen,
                        used_combine=True,
                        used_drop=True,
                        hist=self.hist + [f"combine_drop1_{i}_{c1}_{new_word}"],
                        score_=None,
                    )
                    yield new_self
                c2 = c[1:]
                for i in range(n + 1):
                    if i > max_len:
                        break
                    if position is not None and i != position:
                        continue
                    new_word = self.word[:i] + c2 + self.word[i:]
                    new_frozen = self.frozen[:i] + [False] * m + self.frozen[i:]
                    new_self = replace(
                        self,
                        word=new_word,
                        frozen=new_frozen,
                        used_combine=True,
                        used_drop=True,
                        hist=self.hist + [f"combine_drop0_{i}_{c2}_{new_word}"],
                        score_=None,
                    )
                    yield new_self

    def spellcheck(self):
        if not self.available("spellcheck"):
            return
        new_word = self.word
        new_frozen = self.frozen.copy()
        for s in spellcheck_candidates(len(self.word)):
            new_word, new_frozen = remove_substring_and_filter_list(
                new_word, new_frozen, s
            )

        new_self = replace(
            self,
            word=new_word,
            frozen=new_frozen,
            used_spellcheck=True,
            hist=self.hist + [f"spellcheck_{new_word}"],
            score_=None,
        )
        yield new_self


@lru_cache(maxsize=128)
def target_masks(goal):
    # Bit j means that the first j letters of the target have matched.
    masks = {}
    for j, c in enumerate(goal, 1):
        masks[c] = masks.get(c, 0) | (1 << j)
    return masks


@lru_cache(maxsize=128)
def drop_flower_names(donors):
    return tuple(dict.fromkeys((*SOURCE_CATALOG, *donors)))


@lru_cache(maxsize=256)
def spellcheck_candidates(length):
    # SPELLCHECK only deletes. A longer spell cannot occur initially or become
    # exposed later in this cast. Preserve catalog order for cascading matches.
    return tuple(spell for spell in SPELLCHECK_WORDS if len(spell) <= length)


@lru_cache(maxsize=26)
def spell_creation_patterns(letter):
    return tuple(
        (spell[:i], spell[i + 1:], i)
        for spell in SPELLCHECK_WORDS for i, c in enumerate(spell) if c == letter
    )


def creates_spell_at(word, position, replacement):
    """Whether replacing this character forms a known spell across the edit."""
    if word[position] == replacement:
        return False
    return any(
        position >= offset
        and word.startswith(prefix, position - offset)
        and word.startswith(suffix, position + 1)
        for prefix, suffix, offset in spell_creation_patterns(replacement)
    )


@lru_cache(maxsize=128)
def target_edit_masks(goal):
    masks = target_masks(goal)
    alters = {}
    adjacent = {}
    for c in "abcdefghijklmnopqrstuvwxyz":
        alters[c] = 0
        for replacement in alter_map.get(c, ""):
            alters[c] |= masks.get(replacement, 0)
        adjacent[c] = masks.get(chr(ord(c) - 1), 0) | masks.get(chr(ord(c) + 1), 0)
    return masks, alters, adjacent


@lru_cache(maxsize=128)
def target_counts(goal):
    return tuple(collections.Counter(goal).items())


@lru_cache(maxsize=128)
def source_letter_maxima(flowers):
    counts = {}
    for flower in flowers:
        for letter, count in collections.Counter(flower).items():
            counts[letter] = max(counts.get(letter, 0), count)
    return counts


@lru_cache(maxsize=65536)
def missing_letter_count(word, goal):
    return sum(max(0, count - word.count(c)) for c, count in target_counts(goal))


@lru_cache(maxsize=65536)
def finishing_insertion(word, goal, max_position):
    """Return a legal insertion completing the goal, if there is one."""
    for j, c in enumerate(goal):
        if c in word:
            continue
        start = word.find(goal[:j] + goal[j + 1:])
        if start >= 0 and start + j <= max_position:
            return start + j, c
    return None


def finish_with_replacements(state, max_casts=None, node_budget=256):
    """Try correcting a target window using actual replacement/swap casts.

    Every edited position inside the window must become correct immediately.
    This small discovery search is deliberately bounded; failing here never
    prunes the ordinary search or proves infeasibility.
    """
    names = ("change", "alter", "adjacent", "swap", "swap_adjacent", "find_and_replace")
    width = len(target)
    if width > len(state.word) or not width or max_casts is not None and max_casts < 1:
        return None
    max_casts = min(7, max_casts) if max_casts is not None else 7

    def capacity(current):
        value = sum(current.available(name) * (width if name == "find_and_replace"
                    else 2 if name in ("swap", "swap_adjacent") else 1) for name in names)
        if current.available("recast_spell"):
            value += max((width if name == "find_and_replace" else 2
                          if name in ("swap", "swap_adjacent") else 1
                          for name in names if current.spell_allowed(name)), default=0)
        return value

    def mismatches(current, start, limit):
        bad = []
        for j, wanted in enumerate(target):
            i = start + j
            if current.word[i] != wanted:
                if current.frozen[i] or i > max_len:
                    return None
                bad.append(i)
                if len(bad) > limit:
                    return None
        return bad

    possible = capacity(state)
    if possible == 0:
        return None
    windows = []
    for start in range(len(state.word) - width + 1):
        bad = mismatches(state, start, possible)
        if bad:
            windows.append((len(bad), start))
    windows.sort()
    remaining = node_budget

    def visit(current, start, path, seen):
        nonlocal remaining
        remaining -= 1
        if remaining < 0:
            return None
        bad = mismatches(current, start, capacity(current))
        if bad is None:
            return None
        if not bad:
            return path
        if len(path) >= max_casts:
            return None
        key = current.word, tuple(current.frozen), tuple(current.useds())
        if seen.get(key, max_casts + 1) <= len(path):
            return None
        seen[key] = len(path)

        def corrections():
            for i in bad:
                wanted = target[i - start]
                c = current.word[i]
                for j, donor in enumerate(current.word[:max_len + 1]):
                    if donor != wanted or current.frozen[j]:
                        continue
                    if start <= j < start + width and target[j - start] != c:
                        continue
                    first, second = sorted((i, j))
                    yield from current.swap(first, second)
                    if second == first + 1:
                        yield from current.swap_adjacent(first)
                yield from current.alter(position=i, letter=wanted)
                yield from current.adjacent(position=i, letter=wanted)
                yield from current.change(position=i, letter=wanted)
                if (current.available("find_and_replace") and wanted not in current.word
                        and all(current.word[k] != c or target[k - start] == wanted
                                for k in range(start, start + width))):
                    yield from current.find_and_replace(source=c, destination=wanted)

        for child in corrections():
            answer = visit(child, start, path + [child], seen)
            if answer is not None:
                return answer
        if current.available("recast_spell"):
            for child in current.recast_spell():
                if any(getattr(current, "used_" + name) and child.available(name)
                       for name in names):
                    answer = visit(child, start, path + [child], seen)
                    if answer is not None:
                        return answer
        return None

    for _, start in windows:
        answer = visit(state, start, [], {})
        if answer is not None:
            return answer
        if remaining <= 0:
            break
    return None


@lru_cache(maxsize=65536)
def finishing_existing_insertions(word, goal, max_position):
    """Insertion gaps that would finish the target using an existing letter."""
    gaps = {}
    for j, c in enumerate(goal):
        if c not in word:
            continue
        needle = goal[:j] + goal[j + 1:]
        start = word.find(needle)
        while start >= 0 and start + j <= max_position:
            gaps[start + j, c] = None
            start = word.find(needle, start + 1)
    return tuple(gaps)


def bounded_substring_distance(
    source, goal, frozen, max_cost, *,
    allow_insert=True, allow_delete=True, allow_substitute=True,
    substitution_masks=None, insertion_mask=-1, insertion_masks=None,
    free_delete=False,
    delete_frozen=False,
    allow_one_unlock=False,
):
    """Relaxed edit distance to a substring, capped at max_cost + 1.

    Frozen source letters can only match. Source prefixes/suffixes are free.
    Each integer stores all target positions reachable at a given edit cost,
    so one bit operation replaces a loop over target positions.
    insertion_masks optionally supplies a target-letter mask per source gap.
    """
    m = len(goal)
    if not m:
        return 0
    masks = target_masks(goal)
    end_bit = 1 << m
    first_insertion = insertion_mask if insertion_masks is None else insertion_masks[0]
    # A restricted insertion alphabet may not form the whole target from an
    # empty substring. Keep that case distinct from unrestricted insertions.
    best = min(m, max_cost + 1) if allow_insert and first_insertion == -1 else max_cost + 1
    previous = [1]
    for d in range(1, min(max_cost + 1, best)):
        previous.append(
            previous[-1] | ((previous[-1] << 1) & first_insertion) if allow_insert else 1
        )
        if previous[-1] & end_bit:
            best = d
            previous.pop()
            break
    for i, (c, locked) in enumerate(zip(source, frozen), 1):
        match = masks.get(c, 0)
        substitute = -1 if substitution_masks is None else substitution_masks[c]
        insert = insertion_mask if insertion_masks is None else insertion_masks[i]
        current = []
        for d in range(len(previous)):
            # Bit 0 permits starting a new substring at every source position.
            value = 1 | ((previous[d] << 1) & match)
            if free_delete and allow_delete:
                value |= previous[d]
            if d:
                if allow_delete and (not locked or delete_frozen):
                    value |= previous[d - 1]
                if not locked:
                    if allow_substitute:
                        value |= (previous[d - 1] << 1) & substitute
                if allow_insert:
                    value |= (current[d - 1] << 1) & insert
            if value & end_bit:
                best = d
                if not best:
                    return 0
                break
            current.append(value)
        previous = current
    if allow_one_unlock and best > 2 and any(frozen):
        best = min(best, _bounded_one_unlock(
            source, goal, frozen, min(max_cost, best - 1),
            allow_insert, allow_delete, allow_substitute,
            substitution_masks, insertion_mask, insertion_masks, free_delete, delete_frozen))
    return best


def _bounded_one_unlock(source, goal, frozen, max_cost, allow_insert, allow_delete,
                        allow_substitute, substitution_masks, insertion_mask,
                        insertion_masks, free_delete, delete_frozen):
    """Two bit-vector bands track whether the single unlock was consumed.

    Editing a frozen source letter costs two: unlock, then edit. A path can
    take that transition only once. Source prefixes and suffixes remain free.
    """
    masks, end = target_masks(goal), 1 << len(goal)
    first_insert = insertion_mask if insertion_masks is None else insertion_masks[0]
    best = min(len(goal), max_cost + 1) if allow_insert and first_insert == -1 else max_cost + 1
    unused = [1]
    for d in range(1, min(max_cost + 1, best)):
        unused.append(unused[-1] | ((unused[-1] << 1) & first_insert) if allow_insert else 1)
    used = unused.copy()
    for i, (letter, locked) in enumerate(zip(source, frozen), 1):
        match = masks.get(letter, 0)
        substitute = -1 if substitution_masks is None else substitution_masks[letter]
        insert = insertion_mask if insertion_masks is None else insertion_masks[i]
        next_unused, next_used = [], []
        for d in range(len(unused)):
            a = 1 | ((unused[d] << 1) & match)
            b = 1 | ((used[d] << 1) & match)
            if free_delete and allow_delete:
                a |= unused[d]
                b |= used[d]
            if d:
                if allow_delete and (not locked or delete_frozen):
                    a |= unused[d - 1]
                    b |= used[d - 1]
                if not locked and allow_substitute:
                    a |= (unused[d - 1] << 1) & substitute
                    b |= (used[d - 1] << 1) & substitute
                if allow_insert:
                    a |= (next_unused[d - 1] << 1) & insert
                    b |= (next_used[d - 1] << 1) & insert
            if locked and d >= 2:
                if allow_delete:
                    b |= unused[d - 2]
                if allow_substitute:
                    b |= (unused[d - 2] << 1) & substitute
            if (a | b) & end:
                best = d
                if not best:
                    return 0
                break
            next_unused.append(a)
            next_used.append(b)
        unused, used = next_unused, next_used
    return best


def generalized_levenshtein_distance(source: str, target: str) -> int:
    return generalized_levenshtein_custom_cost_dp(source, target, [1] * len(source))


def generalized_levenshtein_custom_cost_dp(source: str, target: str, costs: list):
    """Weighted edit distance from any source substring to the target.

    Deletion/substitution use the source letter's cost; insertion costs one.
    Keep only the previous row, including an empty-substring match.
    """
    previous = list(range(len(target) + 1))
    best = len(target)
    for c, cost in zip(source, costs):
        current = [0]
        for j, t in enumerate(target):
            if c == t:
                current.append(previous[j])
            else:
                current.append(min(
                    previous[j + 1] + cost,
                    current[j] + 1,
                    previous[j] + cost,
                ))
        best = min(best, current[-1])
        previous = current
    return best


def remove_substring_and_filter_list(s, bool_list, substring):
    # Find all occurrences of the substring
    start = 0
    while start < len(s):
        # Find the start index of the next occurrence of the substring
        start = s.find(substring, start)
        if start == -1:  # No more occurrences found
            break

        # Calculate the end index of the substring
        end = start + len(substring)

        # Remove the characters from the string
        s = s[:start] + s[end:]

        # Remove the corresponding elements from the list
        del bool_list[start:end]

    return s, bool_list


def _operation(state):
    parts = state.hist[-1].split("_")[:-1]
    if parts[:2] == ["swap", "adj"]:
        return "swap_adj", parts[2:]
    return parts[0], parts[1:]


def command_spells(command):
    """Editing spells consumed by one history row, including bundled DROP."""
    if command.startswith("combine_drop"):
        return ("combine", "drop")
    if command.startswith("swap_adj_"):
        return ("swap_adjacent",)
    kind = command.split("_", 1)[0]
    return ({"sub": "subtract", "dup": "duplicate", "adj+": "adjacent",
             "adj-": "adjacent", "fandr": "find_and_replace",
             "mostf": "most_frequent", "recast": "recast_spell",
             "unlock": "unlock_letter"}.get(kind, kind),)


def _drops_left(before, after):
    return before.word[1:] == after.word and before.frozen[1:] == after.frozen


def _letter_ids_after(ids, before, after, step):
    """Track letters across a legal step, including identical-letter copies."""
    kind, args = _operation(after)
    ids = list(ids)
    if kind in ("add", "mostf", "dup"):
        # Duplicate inserts a frozen copy before the unfrozen original.
        ids.insert(int(args[0]), (step, 0))
    elif kind == "combine":
        i = int(args[-2])
        count = len(after.word) - len(before.word)
        ids[i:i] = [(step, j) for j in range(count)]
    elif kind == "move":
        letter = ids.pop(int(args[0]))
        ids.insert(int(args[1]), letter)
    elif kind in ("swap", "swap_adj"):
        i, j = (int(arg[:-1]) for arg in args)
        ids[i], ids[j] = ids[j], ids[i]
    elif kind == "sub":
        ids.pop(int(args[0]))
    elif kind == "drop":
        if args[0] == "n":
            ids = ids[1:] if _drops_left(before, after) else ids[:-1]
        else:
            ids.pop(int(args[0]))
    elif kind == "chop":
        n = len(ids)
        ids = ids[: (n + 1) // 2] if args[0] == "0" else ids[n // 2 :]
    elif kind == "spellcheck":
        word = before.word
        for spell in spellcheck_candidates(len(before.word)):
            word, ids = remove_substring_and_filter_list(word, ids, spell)
    elif kind not in ("alter", "change", "adj+", "adj-", "fandr", "recast", "unlock"):
        return None
    return ids


def _rebase_gap(gap, old_ids, ids):
    """Keep insertion boundaries attached to surviving neighboring letters."""
    if gap == 0:
        return 0
    if gap == len(old_ids):
        return len(ids)
    positions = {letter: i for i, letter in enumerate(ids)}
    for letter in old_ids[gap:]:
        if letter in positions:
            return positions[letter]
    for letter in reversed(old_ids[:gap]):
        if letter in positions:
            return positions[letter] + 1
    return None


def _replay_step(before, after, old_ids, current, ids, step):
    """Rebase a recorded command, then let the real generators validate it."""
    kind, args = _operation(after)
    args = args.copy()
    positions = {letter: i for i, letter in enumerate(ids)}
    expected_drop = None

    if kind in ("alter", "change", "adj+", "adj-", "sub", "dup", "unlock") or (
            kind == "drop" and args[0] != "n"):
        i = positions.get(old_ids[int(args[0])])
        if i is None:
            return None
        args[0] = str(i)
        if kind in ("adj+", "adj-", "sub", "dup", "unlock", "drop"):
            args[1] = current.word[i]
        if kind == "drop":
            expected_drop = current.word[:i] + current.word[i + 1:], current.frozen[:i] + current.frozen[i + 1:]
    elif kind in ("add", "mostf", "combine"):
        position_arg = len(args) - 2 if kind == "combine" else 0
        i = _rebase_gap(int(args[position_arg]), old_ids, ids)
        if i is None:
            return None
        args[position_arg] = str(i)
    elif kind == "move":
        old_i, old_j = map(int, args)
        i = positions.get(old_ids[old_i])
        if i is None:
            return None
        j = _rebase_gap(
            old_j, old_ids[:old_i] + old_ids[old_i + 1 :], ids[:i] + ids[i + 1 :]
        )
        if j is None:
            return None
        args = [str(i), str(j)]
    elif kind in ("swap", "swap_adj"):
        pair = [positions.get(old_ids[int(arg[:-1])]) for arg in args]
        if None in pair:
            return None
        args = [f"{i}{current.word[i]}" for i in sorted(pair)]
    elif kind == "drop":
        if _drops_left(before, after):
            expected_drop = current.word[1:], current.frozen[1:]
        else:
            expected_drop = current.word[:-1], current.frozen[:-1]

    method = {
        "combine": "combine", "spellcheck": "spellcheck",
        "fandr": "find_and_replace", "recast": "recast_spell",
        "change": "change", "swap": "swap", "swap_adj": "swap_adjacent",
        "chop": "chop", "drop": "drop",
    }.get(kind)
    candidates = (getattr(current, method)() if method is not None
                  else current.generate_states_unlock(include_bulk=False))
    prefix = "_".join([kind] + args) + "_"
    for candidate in candidates:
        if expected_drop is None and not candidate.hist[-1].startswith(prefix):
            continue
        if expected_drop is not None and (candidate.word, candidate.frozen) != expected_drop:
            continue
        new_ids = _letter_ids_after(ids, current, candidate, step)
        if new_ids is not None:
            return candidate, new_ids
    return None


def _normalize_recast(path):
    """Postpone RECAST to immediately before the restored spell is used.

    Moving it past other casts changes neither their preconditions nor their
    results: it only resets a different spell's availability. Remove it if
    that spell is never used. Real generators validate the reordered trace.
    """
    index = next((i for i in range(1, len(path)) if path[i].pending_recast()), None)
    if index is None:
        return path
    name = path[index].pending_recast()
    following = next((i for i in range(index + 1, len(path))
                      if name in command_spells(path[i].hist[-1])), len(path))
    if following == index + 1:
        return path
    order = list(range(1, len(path)))
    order.remove(index)
    if following < len(path):
        order.insert(order.index(following), index)
    ids = [[(0, i) for i in range(len(path[0].word))]]
    for step in range(1, len(path)):
        letters = _letter_ids_after(ids[-1], path[step - 1], path[step], step)
        if letters is None:
            return path
        ids.append(letters)
    replay, letters = [path[0]], ids[0]
    for step in order:
        result = _replay_step(path[step - 1], path[step], ids[step - 1],
                              replay[-1], letters, step)
        if result is None:
            return path
        state, letters = result
        replay.append(state)
    return replay


def _shorten_path(path, goal, accept_state=None):
    """Greedily remove steps only when a legal replay still reaches the goal.

    Stable letter identities rebase later indices after an omitted insertion,
    deletion, move, etc. The spell generators enforce all availability, frozen
    letter, position and absent-letter rules during each replay.
    """
    while len(path) > 1:
        if goal in path[0].word and (accept_state is None or accept_state(path[0])):
            return path[:1]
        ids = [[(0, i) for i in range(len(path[0].word))]]
        for step in range(1, len(path)):
            following = _letter_ids_after(ids[-1], path[step - 1], path[step], step)
            if following is None:
                return path
            ids.append(following)

        for omitted in range(1, len(path)):
            replay = [path[0]]
            letters = ids[0]
            for step in range(1, len(path)):
                if step == omitted:
                    continue
                following = _replay_step(
                    path[step - 1], path[step], ids[step - 1],
                    replay[-1], letters, step,
                )
                if following is None:
                    break
                state, letters = following
                replay.append(state)
                if goal in state.word and (accept_state is None or accept_state(state)):
                    break
            if goal in replay[-1].word and (accept_state is None or accept_state(replay[-1])):
                path = replay
                break
        else:
            return path
    return path


@dataclass
class SearchResult:
    state: Optional[State] = None
    cost: Optional[int] = None
    expanded: int = 0
    generated: int = 0
    duplicates: int = 0
    pruned: int = 0
    bulk_expanded: int = 0
    guided_expanded: int = 0
    guided_generated: int = 0
    noop_pruned: int = 0
    native_expanded: int = 0
    native_generated: int = 0
    bound_pruned: int = 0
    reopened: int = 0
    improvements: int = 0
    removed_steps: int = 0
    first_solution_seconds: Optional[float] = None
    optimal: bool = False
    interrupted: bool = False
    stop_reason: str = ""
    seconds: float = 0.0


def _arrival_cost(state, parent):
    if parent is None:
        return spell_count(state.hist)
    return parent.cost + spell_count(state.hist[len(parent.state.hist):])


@dataclass
class _PendingState:
    state: State
    local_expanded: bool = False
    parent: Optional["_PendingState"] = None
    bulk_expanded: bool = False
    cache_key: Optional[tuple] = None
    lower_bound: int = 0
    cost: int = field(init=False)

    def __post_init__(self):
        self.cost = _arrival_cost(self.state, self.parent)

    @property
    def pending(self):
        return not self.local_expanded or (
            not self.bulk_expanded
            and not (self.state.used_combine and self.state.used_spellcheck
                     and self.state.used_find_and_replace)
        )


def _spellcheck_seeds(roots, limit=16, max_cost=None):
    """Rank real setup paths without changing or pruning the normal frontier."""
    seeds = []
    seen = set()
    for root in roots:
        if root.state.used_alter or root.state.used_spellcheck:
            continue
        bases = itertools.chain(
            [root], (_PendingState(s, parent=root) for s in root.state.combine())
        )
        for base in bases:
            for altered, checked in base.state.spellcheck_preparations():
                edited_node = _PendingState(altered, parent=base)
                node = (edited_node if target in altered.word else
                        _PendingState(checked, parent=edited_node))
                if max_cost is not None and node.cost >= max_cost:
                    continue
                key = node.state.key()
                if key in seen:
                    continue
                seen.add(key)
                if node.state.can_reach():
                    seeds.append(node)
    seeds.sort(key=lambda node: -node.state.score())
    return seeds[:limit]


def _combination_seeds(roots, limit=4, max_cost=None):
    """Give promising early combinations turns before local edits spend spells."""
    seeds = []
    seen = set()
    for root in roots:
        for state in root.state.combine(include_drop=False):
            node = _PendingState(state, parent=root)
            if max_cost is not None and node.cost >= max_cost:
                continue
            key = state.key()
            if key in seen:
                continue
            seen.add(key)
            if state.can_reach():
                seeds.append(node)
    seeds.sort(key=lambda node: -node.state.score())
    selected = []
    families = collections.Counter()
    for node in seeds:
        _, args = _operation(node.state)
        family = (node.parent.state.word, args[-1])
        if families[family] >= 2:
            continue
        selected.append(node)
        families[family] += 1
        if len(selected) == limit:
            break
    return selected


def _recombination_seeds(roots, limit=128, max_cost=None, all_positions=False):
    """Prepare two donors before committing scarce local editing spells.

    A second flower may supply only letters needed later for transport, so
    greedily judging each individual combine can discard the useful setup.
    Try this bounded portfolio separately; every node retains its real casts.
    """
    seeds, seen = [], set()
    heaps, serial = collections.defaultdict(list), itertools.count()
    catalog_order = {word: i for i, word in enumerate(SOURCE_CATALOG)}
    def source_order(word):
        return catalog_order.get(word, len(catalog_order)), word
    def ordered_combinations(state):
        donors = state.source_words if state.source_words is not None else sources
        for donor in sorted(donors, key=source_order):
            yield from state.combine(include_drop=False, source=donor)
    # Discovery tie-breaking should not depend on how a CLI list or the
    # progression audit orders the same available flowers.
    bases = ((_PendingState(state, parent=root)
              for root in sorted(roots, key=lambda node: source_order(node.state.hist[0]))
              for state in ordered_combinations(root.state)) if all_positions else
             _combination_seeds(roots, limit=64, max_cost=max_cost))
    for base in bases:
        for restored in base.state.recast_spell():
            if restored.used_combine:
                continue
            reset = _PendingState(restored, parent=base)
            combinations = (ordered_combinations(restored) if all_positions else
                            restored.combine(include_drop=False))
            for combined in combinations:
                parent = _PendingState(combined, parent=reset)
                for state in itertools.chain((combined,), combined.spellcheck()):
                    node = parent if state is combined else _PendingState(state, parent=parent)
                    if max_cost is not None and node.cost >= max_cost:
                        continue
                    if all_positions:
                        if not state.can_reach():
                            continue
                        family = (node.state.hist[0], base.state.hist[-1].split("_")[2],
                                  combined.hist[-1].split("_")[2])
                        heap = heaps[family]
                        entry = (state.score(), next(serial), node)
                        if len(heap) < 8:
                            heapq.heappush(heap, entry)
                        elif entry > heap[0]:
                            heapq.heapreplace(heap, entry)
                        continue
                    key = state.key()
                    if key in seen:
                        continue
                    seen.add(key)
                    if state.can_reach():
                        seeds.append(node)
    if all_positions:
        # Keep only a small heap per donor family instead of retaining the
        # hundreds of thousands of possible arrangements in memory.
        return [entry[2] for entry in sorted(
            (entry for heap in heaps.values() for entry in heap), reverse=True)[:limit]]
    seeds.sort(key=lambda node: -node.state.score())
    selected, families = [], collections.Counter()
    for node in seeds:
        donors = tuple(command.split("_")[2] for command in node.state.hist[1:]
                       if command.startswith("combine_"))
        family = node.state.hist[0], donors
        if families[family] >= 2:
            continue
        families[family] += 1
        selected.append(node)
        if len(selected) == limit:
            break
    return selected


def _selection_tokens(values):
    if isinstance(values, str):
        values = [values]
    return [token.lower() for value in values for token in value.replace(",", " ").split()]


def select_spells(values):
    """Resolve CLI or API spell names, including comma-separated names."""
    tokens = _selection_tokens(values)
    if tokens == ["all"]:
        return frozenset(SPELL_NAMES)
    if tokens == ["none"]:
        return frozenset()
    aliases = {name.replace("_", ""): name for name in SPELL_NAMES}
    aliases.update(adj="adjacent", sub="subtract", dup="duplicate",
                   fandr="find_and_replace", recast="recast_spell",
                   mostf="most_frequent", unlock="unlock_letter", swapadj="swap_adjacent")
    selected = set()
    for token in tokens:
        name = aliases.get(token.replace("-", "").replace("_", ""))
        if name is None:
            raise ValueError(f"unknown spell {token!r}; choose from: {', '.join(SPELL_NAMES)}")
        selected.add(name)
    return frozenset(selected)


def known_sources():
    return tuple(dict.fromkeys((*SOURCE_CATALOG, *sources)))


def select_sources(values):
    tokens = _selection_tokens(values)
    if tokens == ["all"]:
        return known_sources()
    if tokens == ["none"]:
        return ()
    for token in tokens:
        if not token or any(c not in "abcdefghijklmnopqrstuvwxyz" for c in token):
            raise ValueError(f"invalid source {token!r}; use a word containing only letters a-z")
    return tuple(dict.fromkeys(tokens))


def starting_states(allowed_spells=None, source_words=None):
    """Capture initial availability separately from spells spent during search."""
    if allowed_spells is None:
        defaults = State("", [])
        allowed_spells = frozenset(name for name in SPELL_NAMES if defaults.available(name))
    else:
        allowed_spells = select_spells(allowed_spells)
    source_words = tuple(sources) if source_words is None else select_sources(source_words)
    if not source_words:
        raise ValueError("at least one source is required")
    flags = {"used_" + name: name not in allowed_spells for name in SPELL_NAMES}
    return [State(word, [False] * len(word), hist=[word], allowed_spells=allowed_spells,
                  source_words=source_words, **flags) for word in source_words]


def solve(goal, initial_states=None, on_progress=None, on_solution=None, *, prune=True,
          preferred_rounds=8, simplify=True, first_solution=False,
          allowed_spells=None, source_words=None, replacement_finish=True,
          guided_search=True, accept_state=None, compiled_search=True):
    """Anytime branch-and-bound search, with fast heuristic discovery queues.

    on_solution(state, cost) reports each strictly shorter, legally replayed
    solution. Search continues until interrupted or the remaining frontier's
    admissible bounds prove optimality. first_solution=True stops after the
    first result. The ordering heuristic is never used for pruning or proof.
    Early combinations and spellcheck preparations get separate discovery
    turns, including promotion of matching states already queued elsewhere.
    allowed_spells fixes which spells may be used or recast; source_words fixes
    starting flowers and combine donors. Omitted selections use current defaults.
    With supplied initial_states, their spent flags and configuration are retained
    unless explicitly overridden; allowing a spell does not refresh a spent cast.
    """
    global target, tgt_abc
    if preferred_rounds < 0:
        raise ValueError("preferred_rounds must be nonnegative")
    if allowed_spells is not None:
        allowed_spells = select_spells(allowed_spells)
    if source_words is not None:
        source_words = select_sources(source_words)
        if not source_words:
            raise ValueError("at least one source is required")
    target = goal.lower()
    tgt_abc = "".join(c for c in abc if c in target)
    coalesce_discovery_locks = any(sum(1 for _ in group) >= 4
                                  for _, group in itertools.groupby(target))
    start = time.perf_counter()
    result = SearchResult()
    local_queues = []
    bulk_queues = []
    bound_queue = []
    serial = itertools.count()
    best_g = {}
    best_score = float("-inf")
    initial_roots = []
    prepared = False
    discovery_batches = None
    next_discovery = 32
    guided_tried = False
    done = False

    def consider_solution(node):
        nonlocal done
        if accept_state is not None and not accept_state(node.state):
            return False
        if result.cost is not None and node.cost >= result.cost:
            return False
        # Save a valid incumbent before cleanup/callbacks so Ctrl+C cannot lose
        # a just-found solution, even if it interrupts the replay itself.
        result.state = node.state
        result.cost = node.cost
        result.removed_steps = 0
        result.improvements += 1
        result.seconds = time.perf_counter() - start
        if result.first_solution_seconds is None:
            result.first_solution_seconds = result.seconds
        path = []
        current = node
        while current is not None:
            path.append(current.state)
            current = current.parent
        path.reverse()
        normalized = _normalize_recast(path)
        if accept_state is None or accept_state(normalized[-1]):
            path = normalized
            result.state = path[-1]
            result.cost = spell_count(result.state.hist)
        if simplify:
            shortened = (_shorten_path(path, target) if accept_state is None else
                         _shorten_path(path, target, accept_state))
            normalized = _normalize_recast(shortened)
            if accept_state is None or accept_state(normalized[-1]):
                shortened = normalized
            result.removed_steps = len(path) - len(shortened)
            result.state = shortened[-1]
            result.cost = spell_count(result.state.hist)
        result.seconds = time.perf_counter() - start
        if result.cost == 0:
            result.optimal = True
            result.stop_reason = "optimal"
            done = True
        elif first_solution:
            result.stop_reason = "first_solution"
            done = True
        if on_solution is not None:
            on_solution(result.state, result.cost)
        return done

    def enqueue(state, source_index, parent=None):
        result.generated += 1
        if accept_state is None and parent is not None and parent.state.is_redundant_cast(state):
            result.noop_pruned += 1
            return False
        cost = _arrival_cost(state, parent)
        if result.cost is not None and cost >= result.cost:
            result.bound_pruned += 1
            return False
        if target in state.word and (accept_state is None or accept_state(state)):
            return consider_solution(_PendingState(state, parent=parent))
        key = state.key()
        previous = best_g.get(key)
        if previous is not None and cost >= previous:
            result.duplicates += 1
            return False
        if previous is not None:
            result.reopened += 1
        best_g[key] = cost

        # Exact finishing sequences can cross neutral copy/recast setup moves.
        # They add incumbents without terminating the anytime search.
        allowance = None if result.cost is None else result.cost - cost - 1
        finish = state.finish_with_insertion(max_casts=allowance)
        if finish is not None:
            node = _PendingState(state, parent=parent)
            for step in finish:
                node = _PendingState(step, parent=node)
            if consider_solution(node):
                return True

        feasible, lower_bound = state.search_bounds() if prune else (True, 1)
        if not feasible:
            result.pruned += 1
            return False
        if result.cost is not None and cost + lower_bound >= result.cost:
            result.bound_pruned += 1
            return False
        state.score()
        if replacement_finish and (parent is None or state.dist <= 2):
            allowance = 2 if result.cost is None else min(2, result.cost - cost - 1)
            finish = finish_with_replacements(state, max_casts=allowance)
            if finish is not None:
                node = _PendingState(state, parent=parent)
                for step in finish:
                    node = _PendingState(step, parent=node)
                if consider_solution(node):
                    return True
                if result.cost is not None and cost + lower_bound >= result.cost:
                    result.bound_pruned += 1
                    return False
        node = _PendingState(state, parent=parent, cache_key=key, lower_bound=lower_bound)
        order = next(serial)
        entry = (-state.score(), order, node)
        heapq.heappush(local_queues[source_index], entry)
        if not (state.used_combine and state.used_spellcheck and state.used_find_and_replace):
            heapq.heappush(bulk_queues[source_index], entry)
        if not first_solution:
            heapq.heappush(bound_queue, (cost + lower_bound, order, node))
        return False

    def frontier_lower_bound():
        while bound_queue:
            bound, _, node = bound_queue[0]
            if best_g.get(node.cache_key) != node.cost or not node.pending:
                heapq.heappop(bound_queue)
                continue
            return bound
        return float("inf")

    def clean_front(queue, bulk_turn):
        while queue:
            node = queue[0][2]
            closed = node.bulk_expanded if bulk_turn else node.local_expanded
            if closed or best_g.get(node.cache_key) != node.cost:
                heapq.heappop(queue)
                continue
            if result.cost is not None and node.cost + node.lower_bound >= result.cost:
                heapq.heappop(queue)
                node.local_expanded = node.bulk_expanded = True
                result.bound_pruned += 1
                continue
            break

    try:
        if initial_states is None:
            initial_states = starting_states(allowed_spells, source_words)
        for state in initial_states:
            overrides = {}
            if allowed_spells is not None:
                overrides["allowed_spells"] = allowed_spells
            if source_words is not None:
                overrides["source_words"] = source_words
            allowed = overrides.get("allowed_spells", state.allowed_spells)
            if allowed is not None:
                overrides.update({"used_" + name: True for name in SPELL_NAMES if name not in allowed})
            # A supplied current position may already follow a historical
            # recast. Only recasts made during this search can be postponed.
            state = replace(state, **overrides, canonical_recast=None, score_=None, dist=0)
            initial_roots.append(_PendingState(state))
            source_index = len(local_queues)
            local_queues.append([])
            bulk_queues.append([])
            if enqueue(state, source_index):
                break

        local_rounds = 0
        while not done and (any(local_queues) or any(bulk_queues)):
            # Check only BETWEEN complete expansion batches. During generation,
            # the popped node still has unseen successors even if its heaps
            # are empty. Deferred bulk work also keeps a node's bound active.
            if result.cost is not None and frontier_lower_bound() >= result.cost:
                result.optimal = True
                result.stop_reason = "optimal"
                done = True
                break
            work = result.expanded + result.bulk_expanded
            if guided_search and not guided_tried and work >= (32 if len(target) >= 13 else 256):
                from guided import discover
                guided_tried = True
                if compiled_search and accept_state is None and (
                        len(target) >= 16 or coalesce_discovery_locks):
                    import native_search
                    for width in (8192, 32768):
                        native = native_search.discover(
                            target, [root.state for root in initial_roots], sys.modules[__name__],
                            width=width, max_cost=None if result.cost is None else result.cost - 1)
                        result.native_expanded += native.expanded
                        result.native_generated += native.generated
                        if native.states:
                            node = next(root for root in initial_roots if root.state is native.states[0])
                            for state in native.states[1:]:
                                node = _PendingState(state, parent=node)
                            consider_solution(node)
                            break
                        if not native.available:
                            break
                    if done:
                        break
                guidance_roots = initial_roots + _combination_seeds(
                    initial_roots, limit=8, max_cost=result.cost) + _spellcheck_seeds(
                    initial_roots, limit=24, max_cost=result.cost)
                discovery = discover(target, [node.state for node in guidance_roots],
                                     sys.modules[__name__],
                                     coalesce_locks=coalesce_discovery_locks,
                                     accept_state=accept_state,
                                     max_cost=None if result.cost is None else result.cost - 1)
                result.guided_expanded += discovery.expanded
                result.guided_generated += discovery.generated
                if discovery.states is None:
                    discovery = discover(target, [node.state for node in guidance_roots],
                                         sys.modules[__name__], width=256, family_width=16,
                                         extended=True, unlock_ordering=True,
                                         coalesce_locks=coalesce_discovery_locks,
                                         node_budget=8192, candidate_budget=500000,
                                         accept_state=accept_state,
                                         max_cost=None if result.cost is None else result.cost - 1)
                    result.guided_expanded += discovery.expanded
                    result.guided_generated += discovery.generated
                if (discovery.states is None and len(target) >= 13 and any(
                        (len(root.state.word) >= 4 and root.state.word in target)
                        or len(root.state.word) * 2 <= len(target)
                        for root in initial_roots)):
                    # A short source or one anchoring the target makes the
                    # narrow setup ranking useful. Otherwise go straight to
                    # the pass trying every first insertion position.
                    preparations = _recombination_seeds(initial_roots, max_cost=result.cost)
                    if preparations:
                        guidance_roots = preparations
                        discovery = discover(
                            target, [node.state for node in guidance_roots], sys.modules[__name__],
                            width=512, family_width=16, extended=True, unlock_ordering=True,
                            coalesce_locks=coalesce_discovery_locks,
                            node_budget=16384, candidate_budget=1000000,
                            accept_state=accept_state,
                            max_cost=None if result.cost is None else result.cost - 1)
                        result.guided_expanded += discovery.expanded
                        result.guided_generated += discovery.generated
                if discovery.states is None and len(target) >= 13:
                    preparations = _recombination_seeds(
                        initial_roots, limit=512, max_cost=result.cost, all_positions=True)
                    if preparations:
                        guidance_roots = preparations
                        discovery = discover(
                            target, [node.state for node in guidance_roots], sys.modules[__name__],
                            width=1024, family_width=16, extended=True, unlock_ordering=True,
                            coalesce_locks=coalesce_discovery_locks,
                            node_budget=32768, candidate_budget=2000000,
                            accept_state=accept_state,
                            max_cost=None if result.cost is None else result.cost - 1)
                        result.guided_expanded += discovery.expanded
                        result.guided_generated += discovery.generated
                if discovery.states:
                    node = next(root for root in guidance_roots if root.state is discovery.states[0])
                    for state in discovery.states[1:]:
                        node = _PendingState(state, parent=node)
                    if consider_solution(node):
                        break
            if not prepared and work >= next_discovery:
                if discovery_batches is None:
                    discovery_batches = [batch for batch in (
                        _combination_seeds(initial_roots, max_cost=result.cost),
                        _spellcheck_seeds(initial_roots, max_cost=result.cost),
                    ) if batch]
                    discovery_batches.sort(key=lambda batch: -batch[0].state.score())
                batch = discovery_batches.pop(0) if discovery_batches else []
                prepared = not discovery_batches
                next_discovery = work + 128

                # A good setup may already be queued behind a misleading
                # partial solution. Promote the SAME node into a fair lane,
                # preserving shared expansion flags and its proof-queue entry.
                wanted = {seed.state.key() for seed in batch}
                words = {key[0] for key in wanted}
                queued = {}
                for queues in (local_queues, bulk_queues):
                    for queue in queues:
                        for _, _, node in queue:
                            if (node.state.word in words and node.cache_key in wanted
                                    and best_g.get(node.cache_key) == node.cost and node.pending):
                                queued[node.cache_key] = node
                for seed in batch:
                    source_index = len(local_queues)
                    local_queues.append([])
                    bulk_queues.append([])
                    existing = queued.get(seed.state.key())
                    if existing is not None and existing.cost <= seed.cost:
                        entry = (-existing.state.score(), next(serial), existing)
                        if not existing.local_expanded:
                            heapq.heappush(local_queues[source_index], entry)
                        if (not existing.bulk_expanded and not (
                                existing.state.used_combine and existing.state.used_spellcheck
                                and existing.state.used_find_and_replace)):
                            heapq.heappush(bulk_queues[source_index], entry)
                    elif enqueue(seed.state, source_index, parent=seed.parent):
                        break
                if done:
                    break
            bulk_turn = any(bulk_queues) and (
                local_rounds >= preferred_rounds or not any(local_queues)
            )
            queues = bulk_queues if bulk_turn else local_queues
            local_rounds = 0 if bulk_turn else local_rounds + 1
            for source_index, queue in enumerate(queues):
                clean_front(queue, bulk_turn)
                if not queue:
                    continue
                _, _, node = heapq.heappop(queue)
                state = node.state
                if state.score() > best_score:
                    best_score = state.score()
                    if on_progress is not None:
                        on_progress(state)
                if bulk_turn:
                    result.bulk_expanded += 1
                    node.bulk_expanded = True
                if node.local_expanded:
                    successors = state.search_successors(bulk_only=True)
                else:
                    result.expanded += 1
                    node.local_expanded = True
                    successors = state.search_successors(include_bulk=bulk_turn)
                for successor in successors:
                    if enqueue(successor, source_index, parent=node):
                        break
                if done:
                    break
        if not done:
            result.optimal = result.state is not None
            result.stop_reason = "optimal" if result.optimal else "exhausted"
    except KeyboardInterrupt:
        result.interrupted = True
        result.stop_reason = "interrupted"
    result.seconds = time.perf_counter() - start
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Find and improve a Flowereverie spell solution.",
        epilog=("Spell names: " + ", ".join(SPELL_NAMES) + ". Known sources: "
                + ", ".join(known_sources()) + ". Put the target before selection options."),
    )
    parser.add_argument("target", help="word to find (case insensitive)")
    parser.add_argument("--first", action="store_true", help="stop after the first solution")
    parser.add_argument("--python-only", action="store_true",
                        help="disable optional compiled discovery")
    spell_group = parser.add_mutually_exclusive_group()
    spell_group.add_argument(
        "--spells", "--allowed-spells", nargs="+", metavar="SPELL",
        help="starting spells, separated by spaces or commas; 'all' or 'none' also accepted",
    )
    spell_group.add_argument(
        "--blacklist-spells", "--exclude-spells", nargs="+", metavar="SPELL",
        help="allow every implemented spell except these (not just the current defaults)",
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--sources", "--allowed-sources", nargs="+", metavar="WORD",
        help="starting flowers and combine donors, separated by spaces or commas; 'all' uses the catalog",
    )
    source_group.add_argument(
        "--blacklist-sources", "--exclude-sources", nargs="+", metavar="WORD",
        help="use every known flower except these, for starts and combine donors",
    )
    args = parser.parse_args(argv)
    try:
        allowed = None
        if args.spells is not None:
            allowed = select_spells(args.spells)
        elif args.blacklist_spells is not None:
            allowed = frozenset(SPELL_NAMES) - select_spells(args.blacklist_spells)
        selected_sources = None
        if args.sources is not None:
            selected_sources = select_sources(args.sources)
        elif args.blacklist_sources is not None:
            excluded = select_sources(args.blacklist_sources)
            unknown = set(excluded) - set(known_sources())
            if unknown:
                raise ValueError("unknown source(s): " + ", ".join(sorted(unknown)))
            selected_sources = tuple(word for word in known_sources() if word not in excluded)
        if selected_sources == ():
            raise ValueError("at least one source is required")
    except ValueError as error:
        parser.error(str(error))
    start = time.perf_counter()

    def report_solution(state, cost):
        noun = "cast" if cost == 1 else "casts"
        print(f"New best: {cost} spell {noun} after {time.perf_counter() - start:.3f}s", flush=True)
        print(state, flush=True)

    selections = {}
    if allowed is not None:
        selections["allowed_spells"] = allowed
    if selected_sources is not None:
        selections["source_words"] = selected_sources
    result = solve(args.target, on_solution=report_solution, first_solution=args.first,
                   compiled_search=not args.python_only, **selections)
    if result.interrupted:
        print("\nSearch interrupted.")
    if result.state is None:
        if result.interrupted:
            print("Stopped before finding a solution.")
        else:
            print("No feasible solution: exhausted the search.")
    else:
        label = "Proved shortest" if result.optimal else "Best found"
        noun = "cast" if result.cost == 1 else "casts"
        print(f"{label}: {result.cost} spell {noun} ({result.seconds:.3f}s).")
        if result.interrupted:
            # Repeat the complete incumbent in case Ctrl+C interrupted its
            # earlier output or a later solution's cleanup.
            print(result.state)
        elif args.first and not result.optimal:
            print("Stopped after the first solution; optimality has not been proved.")
    print(
        f"Expanded {result.expanded:,} states; pruned {result.pruned:,} as infeasible; "
        f"pruned {result.bound_pruned:,} by length; skipped {result.duplicates:,} duplicates; "
        f"reopened {result.reopened:,}; bulk expansions {result.bulk_expanded:,}."
    )
    if result.guided_expanded:
        print(f"Guided discovery expanded {result.guided_expanded:,} candidates; "
              f"generated {result.guided_generated:,} legal successors.")
    if result.native_expanded:
        print(f"Compiled discovery expanded {result.native_expanded:,} states; "
              f"generated {result.native_generated:,} successors.")
    if result.removed_steps:
        noun = "step" if result.removed_steps == 1 else "steps"
        print(f"Removed {result.removed_steps} redundant {noun} after legal replay.")
    if result.state is not None:
        return 0
    return 130 if result.interrupted else 1


if __name__ == "__main__":
    raise SystemExit(main())
