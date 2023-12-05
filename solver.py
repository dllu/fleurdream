import collections
import difflib
import heapq
import random
import sys
import time
from dataclasses import dataclass, field, replace
from typing import List, Optional

# usage: pypy3 solver.py revealsakuyaexisted

if len(sys.argv) != 2:
    print("usage: pypy3 solver.py revealsakuyaexisted")
target = sys.argv[1]

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

# limits the max position to avoid infeasible actions due to long words being cut off
max_len = 33
max_search = 1000000
# max_search = 600000000
tgt_abc = "".join(a for a in "abcdefghijklmnopqrstuvwxyz" if a in target)
abc = "abcdefghijklmnopqrstuvwxyz"
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

    def score(self):
        if self.score_ is None:
            self.dist = generalized_levenshtein_custom_cost_dp(
                self.word, target, [int(x) * 100 + 1 for x in self.frozen]
            )
            self.score_ = (
                100
                - self.dist
                - sum(int(x) for x in self.useds()) * 0.4
                + random.random() * 0.01
            )
            if self.used_adjacent:
                self.score_ += 0.1
            if self.used_alter:
                self.score_ += 0.11
            if self.used_combine:
                self.score_ += 0.2
            if self.used_spellcheck:
                self.score_ += 0.25
            if self.used_recast_spell:
                self.score_ -= 0.2
            if self.used_unlock_letter:
                self.score_ += 0.35

            """
            # debug hack
            for i in range(len(self.hist) - 2):
                if (
                    self.used_alter
                    and self.used_spellcheck
                    and self.hist[i + 1][:5] == "alter"
                    and self.hist[i + 2][:5] == "spell"
                    and 'cespe' in self.hist[i + 2]
                ):
                    self.score_ += 10
            """
        return self.score_

    def __lt__(self, other) -> bool:
        return self.score() > other.score()

    def recast_spell(self, max_dist=2):
        if self.dist > max_dist:
            return
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

    def generate_states_unlock(self):
        if self.used_unlock_letter:
            yield from self.generate_states()
            return
        for state in self.generate_states():
            state.score()
            yield state
            if state.dist > 3:
                continue
            for i, f in enumerate(state.frozen):
                if i > max_len:
                    break
                if not f:
                    continue
                new_frozen = state.frozen.copy()
                new_frozen[i] = False
                yield replace(
                    state,
                    frozen=new_frozen,
                    used_unlock_letter=True,
                    hist=state.hist + [f"unlock_{i}_{state.word[i]}_{state.word}"],
                    score_=None,
                )

    def generate_states(self):
        n = len(self.word)
        if (
            self.used_combine
            and self.used_spellcheck
            and self.used_find_and_replace
            and self.used_unlock_letter
        ):
            max_improvement = 0
            if not self.used_drop:
                max_improvement += 1
            if not self.used_change:
                max_improvement += 1
            if not self.used_subtract:
                max_improvement += 1
            if not self.used_duplicate:
                max_improvement += 1
            if not self.used_adjacent:
                max_improvement += 1
            if not self.used_move:
                max_improvement += 2
            if not self.used_swap:
                max_improvement += 2
            if not self.used_swap_adjacent:
                max_improvement += 2
            if not self.used_add:
                max_improvement += 1
            if not self.used_alter:
                max_improvement += 1
            if not self.used_recast_spell:
                max_improvement += 2
            if not self.used_most_frequent:
                max_improvement += 1
            if self.dist > max_improvement:
                return

        yield from self.combine()

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

        if not self.used_alter:
            for i, ci in enumerate(self.word):
                if i > max_len:
                    break
                if self.frozen[i]:
                    continue
                if ci not in alter_map:
                    continue
                for ca in alter_map[ci]:
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
            for c in tgt_abc:
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
            if most_common[0][1] > most_common[1][1]:
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

        if not self.used_find_and_replace:
            src_chars = set(self.word)
            tgt_chars = set(abc) - set(self.word)

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

        yield from self.change()
        yield from self.swap()
        yield from self.swap_adjacent()

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
            for j, cj in enumerate(self.word):
                if i == j or (abs(i - j) == 1 and not self.used_swap_adjacent):
                    continue
                if j > max_len:
                    break

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
            for j, cj in enumerate(self.word):
                if abs(j - i) != 1:
                    continue
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
                    used_swap_adjacent=True,
                    hist=self.hist + [f"swap_adj_{i}{ci}_{j}{cj}_{new_word}"],
                    score_=None,
                )
                yield new_self

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


def generalized_levenshtein_distance(source: str, target: str) -> int:
    """
    Given strings source and target, find the minimum Levenshtein distance between any
    substring of source and target using dynamic programming to avoid redundant computations.
    """
    n = len(source)
    m = len(target)

    # If target is longer than source, it's impossible to match
    if m > n:
        return m

    # Initialize a matrix where dp[i][j] represents the Levenshtein distance
    # between the first i characters of source and the first j characters of target
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # Source prefixes can be transformed into empty string by dropping all characters
    for i in range(1, n + 1):
        dp[i][0] = 0  # Cost of dropping source characters is not counted

    # Target prefixes can be reached from empty source prefix by inserting every character
    for j in range(1, m + 1):
        dp[0][j] = j

    # Compute the distance
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if source[i - 1] == target[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # Deletion
                dp[i][j - 1] + 1,  # Insertion
                dp[i - 1][j - 1] + cost,  # Substitution
            )

    # Find the minimum distance in the last column
    min_distance = min(dp[i][m] for i in range(1, n + 1))

    return min_distance


def generalized_levenshtein_custom_cost_dp(source: str, target: str, costs: list):
    """
    Compute the generalized Levenshtein distance where the costs for deletion and substitution
    of characters in the source are given by the costs list, and the cost for insertion is 1.
    """
    n = len(source)
    m = len(target)

    # Initialize the DP table for distances
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # Initialize the first column with 0 as we are looking for substring matching
    for i in range(1, n + 1):
        dp[i][0] = 0

    # Initialize the first row with insertions (cost is always 1)
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + 1

    # Fill the DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if source[i - 1] == target[j - 1]:
                # No cost if characters are the same
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # Calculate costs for each operation
                insertion = dp[i][j - 1] + 1
                deletion = dp[i - 1][j] + costs[i - 1]
                substitution = dp[i - 1][j - 1] + costs[i - 1]
                dp[i][j] = min(insertion, deletion, substitution)

    # The minimum distance is the smallest distance at the end of the target
    # considering any end point in the source
    min_distance = min(dp[i][m] for i in range(1, n + 1))

    return min_distance


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


def main():
    global max_search

    states = [State(word=src, frozen=[False] * len(src), hist=[src]) for src in sources]
    combinations = []
    for s in states:
        s.score()
        combinations.append(s)
        for ss in s.combine():
            ss.score()
            combinations.append(ss)
            for ssr in ss.recast_spell(max_dist=99999):
                for sss in ssr.combine():
                    sss.score()
                    combinations.append(sss)
    combinations2 = combinations.copy()
    for c in combinations2:
        for cc in c.spellcheck():
            combinations.append(cc)

    combinations.sort()

    max_depth = 0
    best_dist = 9999
    best_score = 0
    print(f"starting out with {len(combinations)} combinations")
    start_time = time.time()

    for i in range(10):
        for initial_s in combinations:
            states = [initial_s]
            visited = set()
            heapq.heapify(states)

            while states:
                s = heapq.heappop(states)
                if s.dist < best_dist or (
                    s.dist == best_dist and s.score() > best_score
                ):
                    best_dist = s.dist
                    best_score = s.score()
                    print("New best found:", time.time() - start_time)
                    print(s)
                    if target in s.word:
                        print("TARGET FOUND!!!!!", time.time() - start_time)
                for ns in s.generate_states_unlock():
                    if ns.stringify() in visited:
                        continue
                    visited.add(ns.stringify())
                    heapq.heappush(states, ns)
                    if len(ns.hist) > max_depth:
                        max_depth = len(ns.hist)
                        print("depth", max_depth)
                    for ns2 in ns.spellcheck():
                        if ns2.stringify() in visited:
                            continue
                        visited.add(ns2.stringify())
                        heapq.heappush(states, ns2)
                        if len(ns2.hist) > max_depth:
                            max_depth = len(ns2.hist)
                            print("depth", max_depth)

                if len(visited) > max_search:
                    break
        print(f"Failed to find with {max_search=}")
        max_search *= 2


main()


def test_dist():
    wtf = False

    def t(a, b, c, e):
        d = generalized_levenshtein_custom_cost_dp(a, b, c)
        print(a, b, d, e, d == e)
        if d != e:
            wtf = True

    t("foobarfoobar", "barfoo", [1, 2, 3] * 4, 0)
    t("foobarfoobar", "baroo", [1, 2, 3] * 4, 1)
    t("abcdefghijklmnopqrs", "abcdefglmnopqrs", [1] * len("abcdefghijklmnopqrs"), 4)
    t("foobarfoobar", "barqfoo", [1, 2, 3] * 4, 1)
    t("foobarqwerqwer", "foobar", [1, 2] * 7, 0)
    t("asdfasdffoobar", "foobar", [1, 2] * 7, 0)
    t("fooqwerbar", "foobar", [1, 2] * 5, 3)
    t("foomar", "foobar", [1, 2] * 3, 2)
    if wtf:
        exit()


def test_remove_substring():
    s = "lilachangerbloom"
    f = [c > "i" for c in s]

    def bs(f):
        return "".join(str(int(x)) for x in f)

    print(s + "\n" + bs(f))
    for spell in spells:
        s, f = remove_substring_and_filter_list(s, f, spell)

    print(s + "\n" + bs(f))


def test():
    test_remove_substring()
    test_dist()


# test()
