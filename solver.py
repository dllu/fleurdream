import argparse
import collections
import difflib
import heapq
import itertools
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
    #"heedonicindigo",
    #"violeepicturet",
    #"sakuya",
]

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

spells = [
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
]


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
    used_swap_adjacent: bool = True
    used_subtract: bool = False
    used_duplicate: bool = False
    used_spellcheck: bool = False
    used_recast_spell: bool = False
    used_find_and_replace: bool = True
    used_most_frequent: bool = True
    used_unlock_letter: bool = True
    hist: List = field(default_factory=list)
    score_: Optional[float] = None
    dist: float = 0.0

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
        return self.word, tuple(self.frozen), tuple(self.useds())

    def finish_with_add(self):
        if self.used_add:
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

    def edit_budget(self):
        # These spells can change many letters or restore an arbitrary spell.
        # The local edit bound below is only valid once all of them are spent.
        if not (
            self.used_combine
            and self.used_spellcheck
            and self.used_find_and_replace
            and self.used_unlock_letter
            and self.used_recast_spell
        ):
            return None
        insertions = sum(not used for used in (
            self.used_add, self.used_duplicate, self.used_most_frequent, self.used_move
        ))
        deletions = sum(not used for used in (self.used_subtract, self.used_move))
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
        if budget is None:
            return True, 1
        # Moving, swapping, chopping and dropping cannot increase letter counts.
        growth = sum(not used for used in (
            self.used_add, self.used_duplicate, self.used_most_frequent,
            self.used_change, self.used_alter, self.used_adjacent,
        ))
        missing = missing_letter_count(self.word, target)
        if missing > growth:
            return False, missing
        insertions, deletions, substitutions = budget
        substitution_masks = None
        if self.used_change and self.used_swap and self.used_swap_adjacent:
            # These direct substitutions form a safe relaxation once all bulk
            # spells, unlock and recast are spent: edited letters freeze.
            _, alters, adjacent = target_edit_masks(target)
            substitution_masks = {
                c: (0 if self.used_alter else alters.get(c, 0))
                | (0 if self.used_adjacent else adjacent.get(c, 0))
                for c in set(self.word)
            }
        bound = bounded_substring_distance(
            self.word, target, self.frozen, sum(budget),
            allow_insert=insertions > 0,
            allow_delete=deletions > 0,
            allow_substitute=substitutions > 0,
            substitution_masks=substitution_masks,
        )
        per_cast = 2 if not (self.used_move and self.used_swap and self.used_swap_adjacent) else 1
        lower_bound = max(1, missing, (bound + per_cast - 1) // per_cast)
        return bound <= sum(budget), lower_bound

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

        insertion_mask = 0 if self.used_add else missing
        if not self.used_move or not self.used_duplicate:
            insertion_mask |= movable
        if not self.used_most_frequent:
            # Ignore frequency ties here: an optimistic supply is useful for
            # ordering, and later edits may change which letter is most common.
            for c in set(self.word):
                insertion_mask |= masks.get(c, 0)

        generic = 0 if self.used_change else missing
        if not self.used_swap or not self.used_swap_adjacent:
            generic |= movable
        substitution_masks = {
            c: generic | (0 if self.used_alter else alters.get(c, 0))
            | (0 if self.used_adjacent else adjacent.get(c, 0))
            for c in set(self.word)
        }
        return bounded_substring_distance(
            self.word, target, self.frozen, len(target),
            allow_delete=not (self.used_move and self.used_subtract),
            insertion_mask=insertion_mask,
            substitution_masks=substitution_masks,
        )

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
        if self.used_recast_spell:
            return
        if self.used_alter:
            yield replace(
                self,
                used_alter=False,
                used_recast_spell=True,
                hist=self.hist + [f"recast_alter_{self.word}"],
                score_=None,
            )
        if self.used_move:
            yield replace(
                self,
                used_move=False,
                used_recast_spell=True,
                hist=self.hist + [f"recast_move_{self.word}"],
                score_=None,
            )
        if self.used_add:
            yield replace(
                self,
                used_add=False,
                used_recast_spell=True,
                hist=self.hist + [f"recast_add_{self.word}"],
                score_=None,
            )
        if self.used_adjacent:
            yield replace(
                self,
                used_adjacent=False,
                used_recast_spell=True,
                hist=self.hist + [f"recast_adjacent_{self.word}"],
                score_=None,
            )
        if self.used_combine:
            yield replace(
                self,
                used_combine=False,
                used_recast_spell=True,
                hist=self.hist + [f"recast_combine_{self.word}"],
                score_=None,
            )
        if self.used_change:
            yield replace(
                self,
                used_change=False,
                used_recast_spell=True,
                hist=self.hist + [f"recast_change_{self.word}"],
                score_=None,
            )
        if self.used_chop:
            yield replace(
                self,
                used_chop=False,
                used_recast_spell=True,
                hist=self.hist + [f"recast_chop_{self.word}"],
                score_=None,
            )
        if self.used_drop:
            yield replace(
                self,
                used_drop=False,
                used_recast_spell=True,
                hist=self.hist + [f"recast_drop_{self.word}"],
                score_=None,
            )
        if self.used_swap:
            yield replace(
                self,
                used_swap=False,
                used_recast_spell=True,
                hist=self.hist + [f"recast_swap_{self.word}"],
                score_=None,
            )
        if self.used_swap_adjacent:
            yield replace(
                self,
                used_swap_adjacent=False,
                used_recast_spell=True,
                hist=self.hist + [f"recast_swap_adjacent_{self.word}"],
                score_=None,
            )
        if self.used_subtract:
            yield replace(
                self,
                used_subtract=False,
                used_recast_spell=True,
                hist=self.hist + [f"recast_subtract_{self.word}"],
                score_=None,
            )
        if self.used_duplicate:
            yield replace(
                self,
                used_duplicate=False,
                used_recast_spell=True,
                hist=self.hist + [f"recast_duplicate_{self.word}"],
                score_=None,
            )
        if self.used_spellcheck:
            yield replace(
                self,
                used_spellcheck=False,
                used_recast_spell=True,
                hist=self.hist + [f"recast_spellcheck_{self.word}"],
                score_=None,
            )
        if self.used_find_and_replace:
            yield replace(
                self,
                used_find_and_replace=False,
                used_recast_spell=True,
                hist=self.hist + [f"recast_find_and_replace_{self.word}"],
                score_=None,
            )
        if self.used_most_frequent:
            yield replace(
                self,
                used_most_frequent=False,
                used_recast_spell=True,
                hist=self.hist + [f"recast_mostf_{self.word}"],
                score_=None,
            )

    def generate_states_unlock(self, include_bulk=True):
        yield from self.generate_states(include_bulk=include_bulk)
        if self.used_unlock_letter:
            return
        for i, frozen in enumerate(self.frozen):
            if i > max_len:
                break
            if not frozen:
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
        n = len(self.word)
        if include_bulk:
            yield from self.generate_bulk_states()

        if not self.used_recast_spell:
            yield from self.recast_spell()

        yield from self.chop()

        if not self.used_drop:
            new_self = replace(
                self,
                word=self.word[1:],
                frozen=self.frozen.copy()[1:],
                used_drop=True,
                hist=self.hist + [f"drop_n_{self.word[1:]}"],
                score_=None,
            )
            yield new_self
            new_self = replace(
                self,
                word=self.word[:-1],
                frozen=self.frozen.copy()[:-1],
                used_drop=True,
                hist=self.hist + [f"drop_n_{self.word[:-1]}"],
                score_=None,
            )
            yield new_self

        yield from self.alter()

        if not self.used_move:
            for i, ci in enumerate(self.word):
                if i > max_len:
                    break
                if self.frozen[i]:
                    continue
                word_without_i = self.word[:i] + self.word[i + 1 :]
                frozen_without_i = self.frozen[:i] + self.frozen[i + 1 :]
                for j in range(n):
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

        if not self.used_subtract:
            for i, ci in enumerate(self.word):
                if i > max_len:
                    break
                if self.frozen[i]:
                    continue
                new_word = self.word[:i] + self.word[i + 1 :]
                new_frozen = self.frozen[:i] + self.frozen[i + 1 :]
                new_self = replace(
                    self,
                    word=new_word,
                    frozen=new_frozen,
                    used_subtract=True,
                    hist=self.hist + [f"sub_{i}_{ci}_{new_word}"],
                    score_=None,
                )
                yield new_self

        if not self.used_duplicate:
            for i, ci in enumerate(self.word):
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

        if not self.used_adjacent:
            for i, ci in enumerate(self.word):
                if i > max_len:
                    break
                if self.frozen[i]:
                    continue
                if ci < "z":
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

                if ci > "a":
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

        if not self.used_add:
            # Prefer target letters, but retain setup moves involving others.
            for c in tgt_abc + "".join(c for c in abc if c not in tgt_abc):
                if c in self.word:
                    continue
                for i in range(n + 1):
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

        if not self.used_most_frequent:
            most_common = collections.Counter(self.word).most_common(2)
            if most_common and (len(most_common) == 1 or most_common[0][1] > most_common[1][1]):
                c = most_common[0][0]
                for i in range(n + 1):
                    if i > max_len:
                        break
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

        yield from self.change()
        yield from self.swap()
        yield from self.swap_adjacent()

    def alter(self, for_spellcheck=False):
        if not self.used_alter:
            for i, ci in enumerate(self.word):
                if i > max_len:
                    break
                if self.frozen[i]:
                    continue
                if ci not in alter_map:
                    continue
                for ca in alter_map[ci]:
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
        if self.used_spellcheck:
            return
        for altered in self.alter(for_spellcheck=True):
            for checked in altered.spellcheck():
                if checked.word != altered.word:
                    yield altered, checked

    def find_and_replace(self):
        if not self.used_find_and_replace:
            src_chars = sorted(set(self.word))
            tgt_chars = sorted(set(abc) - set(self.word))

            for src in src_chars:
                for tgt in tgt_chars:
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
        if self.used_chop:
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

    def change(self):
        n = len(self.word)
        if self.used_change:
            return
        for c in abc:
            if c in self.word:
                continue
            for i in range(n):
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

    def swap(self):
        if self.used_swap:
            return
        for i, ci in enumerate(self.word):
            if i > max_len:
                break
            if self.frozen[i]:
                continue
            for j in range(i + 1, min(len(self.word), max_len + 1)):
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

    def swap_adjacent(self):
        if self.used_swap_adjacent:
            return
        for i, ci in enumerate(self.word):
            if i > max_len:
                break
            if self.frozen[i]:
                continue
            j = i + 1
            if j >= len(self.word) or self.frozen[j]:
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

    def combine(self):
        if self.used_combine:
            return
        n = len(self.word)
        for c in sources:
            m = len(c)
            for i in range(n + 1):
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
            if not self.used_drop:
                c1 = c[:-1]
                m = len(c1)
                for i in range(n + 1):
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
        if self.used_spellcheck:
            return
        new_word = self.word
        new_frozen = self.frozen.copy()
        for s in spells:
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


@lru_cache(maxsize=26)
def spell_creation_patterns(letter):
    return tuple(
        (spell[:i], spell[i + 1:], i)
        for spell in spells for i, c in enumerate(spell) if c == letter
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


def bounded_substring_distance(
    source, goal, frozen, max_cost, *,
    allow_insert=True, allow_delete=True, allow_substitute=True,
    substitution_masks=None, insertion_mask=-1,
):
    """Relaxed edit distance to a substring, capped at max_cost + 1.

    Frozen source letters can only match. Source prefixes/suffixes are free.
    Each integer stores all target positions reachable at a given edit cost,
    so one bit operation replaces a loop over target positions.
    """
    m = len(goal)
    if not m:
        return 0
    masks = target_masks(goal)
    end_bit = 1 << m
    # A restricted insertion alphabet may not form the whole target from an
    # empty substring. Keep that case distinct from unrestricted insertions.
    best = min(m, max_cost + 1) if allow_insert and insertion_mask == -1 else max_cost + 1
    previous = [1]
    for d in range(1, min(max_cost + 1, best)):
        previous.append(
            previous[-1] | ((previous[-1] << 1) & insertion_mask) if allow_insert else 1
        )
        if previous[-1] & end_bit:
            best = d
            previous.pop()
            break
    for c, locked in zip(source, frozen):
        match = masks.get(c, 0)
        substitute = -1 if substitution_masks is None else substitution_masks[c]
        current = []
        for d in range(len(previous)):
            # Bit 0 permits starting a new substring at every source position.
            value = 1 | ((previous[d] << 1) & match)
            if d:
                if not locked:
                    if allow_delete:
                        value |= previous[d - 1]
                    if allow_substitute:
                        value |= (previous[d - 1] << 1) & substitute
                if allow_insert:
                    value |= (current[d - 1] << 1) & insertion_mask
            if value & end_bit:
                best = d
                if not best:
                    return 0
                break
            current.append(value)
        previous = current
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
        ids = ids[1:] if _drops_left(before, after) else ids[:-1]
    elif kind == "chop":
        n = len(ids)
        ids = ids[: (n + 1) // 2] if args[0] == "0" else ids[n // 2 :]
    elif kind == "spellcheck":
        word = before.word
        for spell in spells:
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

    if kind in ("alter", "change", "adj+", "adj-", "sub", "dup", "unlock"):
        i = positions.get(old_ids[int(args[0])])
        if i is None:
            return None
        args[0] = str(i)
        if kind in ("adj+", "adj-", "sub", "dup", "unlock"):
            args[1] = current.word[i]
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
        "chop": "chop",
    }.get(kind)
    candidates = (getattr(current, method)() if method is not None
                  else current.generate_states_unlock(include_bulk=False))
    prefix = "_".join([kind] + args) + "_"
    for candidate in candidates:
        if not candidate.hist[-1].startswith(prefix):
            continue
        if expected_drop is not None and (candidate.word, candidate.frozen) != expected_drop:
            continue
        new_ids = _letter_ids_after(ids, current, candidate, step)
        if new_ids is not None:
            return candidate, new_ids
    return None


def _shorten_path(path, goal):
    """Greedily remove steps only when a legal replay still reaches the goal.

    Stable letter identities rebase later indices after an omitted insertion,
    deletion, move, etc. The spell generators enforce all availability, frozen
    letter, position and absent-letter rules during each replay.
    """
    while len(path) > 1:
        if goal in path[0].word:
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
                if goal in state.word:
                    break
            if goal in replay[-1].word:
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


def solve(goal, initial_states=None, on_progress=None, on_solution=None, *, prune=True,
          preferred_rounds=8, simplify=True, first_solution=False):
    """Anytime branch-and-bound search, with fast heuristic discovery queues.

    on_solution(state, cost) reports each strictly shorter, legally replayed
    solution. Search continues until interrupted or the remaining frontier's
    admissible bounds prove optimality. first_solution=True stops after the
    first result. The ordering heuristic is never used for pruning or proof.
    """
    global target, tgt_abc
    if preferred_rounds < 0:
        raise ValueError("preferred_rounds must be nonnegative")
    target = goal.lower()
    tgt_abc = "".join(c for c in abc if c in target)
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
    done = False

    def consider_solution(node):
        nonlocal done
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
        if simplify:
            path = []
            current = node
            while current is not None:
                path.append(current.state)
                current = current.parent
            path.reverse()
            shortened = _shorten_path(path, target)
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
        cost = _arrival_cost(state, parent)
        if result.cost is not None and cost >= result.cost:
            result.bound_pruned += 1
            return False
        if target in state.word:
            return consider_solution(_PendingState(state, parent=parent))
        key = state.key()
        previous = best_g.get(key)
        if previous is not None and cost >= previous:
            result.duplicates += 1
            return False
        if previous is not None:
            result.reopened += 1
        best_g[key] = cost

        # Preserve the exact finishing-ADD shortcut, but do not terminate an
        # anytime search when it supplies an incumbent.
        finish = state.finish_with_add()
        if finish is not None:
            node = _PendingState(state, parent=parent)
            if consider_solution(_PendingState(finish, parent=node)):
                return True

        feasible, lower_bound = state.search_bounds() if prune else (True, 1)
        if not feasible:
            result.pruned += 1
            return False
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
            initial_states = [
                State(word=src, frozen=[False] * len(src), hist=[src]) for src in sources
            ]
        for state in initial_states:
            state = replace(state, score_=None, dist=0)
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
            if not prepared and result.expanded + result.bulk_expanded >= 64:
                prepared = True
                for seed in _spellcheck_seeds(initial_roots, max_cost=result.cost):
                    source_index = len(local_queues)
                    local_queues.append([])
                    bulk_queues.append([])
                    if enqueue(seed.state, source_index, parent=seed.parent):
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
                    successors = state.generate_bulk_states()
                else:
                    result.expanded += 1
                    node.local_expanded = True
                    successors = state.generate_states_unlock(include_bulk=bulk_turn)
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
    parser = argparse.ArgumentParser(description="Find and improve a Flowereverie spell solution.")
    parser.add_argument("target", help="word to find (case insensitive)")
    parser.add_argument("--first", action="store_true", help="stop after the first solution")
    args = parser.parse_args(argv)
    start = time.perf_counter()

    def report_solution(state, cost):
        noun = "cast" if cost == 1 else "casts"
        print(f"New best: {cost} spell {noun} after {time.perf_counter() - start:.3f}s", flush=True)
        print(state, flush=True)

    result = solve(args.target, on_solution=report_solution, first_solution=args.first)
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
    if result.removed_steps:
        noun = "step" if result.removed_steps == 1 else "steps"
        print(f"Removed {result.removed_steps} redundant {noun} after legal replay.")
    if result.state is not None:
        return 0
    return 130 if result.interrupted else 1


if __name__ == "__main__":
    raise SystemExit(main())
