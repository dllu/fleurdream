"""Optional compiled discovery; every result is checked by Python spell rules.

This module supplies no reachability or optimality proof. Failure or absence
of a C++ compiler leaves the ordinary Python search responsible for those.
"""

import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
from dataclasses import dataclass


SOURCE = Path(__file__).with_suffix('.cpp')


@dataclass
class Discovery:
    states: list | None = None
    expanded: int = 0
    generated: int = 0
    available: bool = False


def ensure_binary(cache_dir=None):
    """Compile once per source version; no installation or package is needed."""
    compiler = shutil.which(os.environ.get('CXX', 'c++'))
    if compiler is None:
        return None
    source = SOURCE.read_bytes()
    digest = hashlib.sha256(source + compiler.encode() + platform.machine().encode()).hexdigest()[:20]
    cache = Path(cache_dir) if cache_dir is not None else Path(tempfile.gettempdir()) / (
        'fleurdream-native-' + str(os.getuid() if hasattr(os, 'getuid') else 'local'))
    cache.mkdir(mode=0o700, parents=True, exist_ok=True)
    executable = cache / ('discovery-' + digest + ('.exe' if os.name == 'nt' else ''))
    if executable.is_file():
        return executable
    with tempfile.TemporaryDirectory(prefix='build-', dir=cache) as build:
        output = Path(build) / executable.name
        process = subprocess.Popen(
            [compiler, '-O3', '-std=c++17', '-DNDEBUG', str(SOURCE), '-o', str(output)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            stdout, stderr = process.communicate()
        except BaseException:
            process.terminate()
            process.wait()
            raise
        if process.returncode:
            raise RuntimeError('Native discovery compilation failed:\n' + stderr)
        output.replace(executable)
    return executable


def _payload(goal, roots, engine, width, family_width, max_cost, mode, structural_families):
    first = roots[0]
    spells = set(engine.SPELL_NAMES) if first.allowed_spells is None else first.allowed_spells
    donors = first.source_words if first.source_words is not None else tuple(engine.sources)
    order = {word: i for i, word in enumerate(engine.SOURCE_CATALOG)}
    donors = tuple(sorted(donors, key=lambda word: (order.get(word, len(order)), word)))
    allowed = sum(1 << i for i, name in enumerate(engine.SPELL_NAMES) if name in spells)
    offset = min(engine.spell_count(state.hist) for state in roots)
    lines = ['FLEURDREAM_NATIVE_1',
             f'{goal or "-"} {engine.abc} {engine.max_len} {allowed} {width} '
             f'{family_width} {max_cost - offset} {mode} {int(structural_families)}']
    flowers = tuple(dict.fromkeys(engine.SOURCE_CATALOG + tuple(donors)))
    for words in (donors, engine.SPELLCHECK_WORDS, flowers):
        lines.extend((str(len(words)), *words))
    lines.extend(engine.alter_map.get(chr(i + 97), '') or '-' for i in range(26))
    lines.append(str(len(roots)))
    for state in roots:
        frozen = sum(bool(flag) << i for i, flag in enumerate(state.frozen))
        used = sum(bool(flag) << i for i, flag in enumerate(state.useds()))
        pending = engine.SPELL_NAMES.index(state.pending_recast()) if state.pending_recast() else -1
        lines.append(f'{state.word or "-"} {frozen} {used} '
                     f'{engine.spell_count(state.hist) - offset} {pending}')
    return '\n'.join(lines) + '\n', offset


def supported(goal, roots, engine):
    if not roots or len(goal) > 63 or not 0 <= engine.max_len <= 63:
        return False
    first = roots[0]
    donors = first.source_words if first.source_words is not None else tuple(engine.sources)
    words = [goal, engine.abc, *donors, *engine.SPELLCHECK_WORDS, *engine.SOURCE_CATALOG]
    if len(donors) > 32 or not engine.abc or any(not word for word in donors):
        return False
    for state in roots:
        if (len(state.word) > 64 or len(state.frozen) != len(state.word)
                or state.allowed_spells != first.allowed_spells
                or state.source_words != first.source_words):
            return False
        words.append(state.word)
    words.extend(engine.alter_map.get(chr(i + 97), '') for i in range(26))
    return all(all('a' <= c <= 'z' for c in word) for word in words)


def _run(data, binary):
    process = subprocess.Popen([str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = process.communicate(data)
    except BaseException:
        process.terminate()
        process.wait()
        raise
    if process.returncode:
        raise RuntimeError(f'Native discovery exited with {process.returncode}:\n{stderr}')
    return stdout


def discover(goal, roots, engine, *, width=8192, family_width=64, max_cost=None,
             structural_families=True, binary=None):
    result = Discovery()
    if not supported(goal, roots, engine) or width <= 0:
        return result
    limit = max(engine.spell_count(state.hist) + sum(not flag for flag in state.useds())
                + int(state.available('recast_spell')) for state in roots)
    max_cost = limit if max_cost is None else min(max_cost, limit)
    roots = [state for state in roots if engine.spell_count(state.hist) <= max_cost]
    if not roots:
        return result
    order = {word: i for i, word in enumerate(engine.SOURCE_CATALOG)}
    roots.sort(key=lambda state: (order.get(state.hist[0], len(order)), state.hist[0],
                                 engine.spell_count(state.hist), state.word, tuple(state.frozen)))
    binary = ensure_binary() if binary is None else binary
    if binary is None:
        return result
    result.available = True
    data, offset = _payload(goal, roots, engine, width, family_width, max_cost, 0, structural_families)
    lines = _run(data, binary).splitlines()
    if not lines:
        raise RuntimeError('Native discovery returned no status')
    status = lines[0].split()
    if status[0] == 'MISS' and len(status) == 3:
        result.expanded, result.generated = map(int, status[1:])
        return result
    if status[0] != 'FOUND' or len(status) != 5:
        raise RuntimeError('Invalid native discovery status: ' + lines[0])
    root, cost, result.expanded, result.generated = map(int, status[1:])
    if not 0 <= root < len(roots):
        raise RuntimeError('Native discovery returned an invalid root')
    path = [roots[root]]
    for entry in lines[1:]:
        child = next((state for state in path[-1].generate_states_unlock()
                      if state.hist[-1] == entry), None)
        if child is None:
            raise RuntimeError('Native discovery returned an illegal cast: ' + entry)
        path.append(child)
    if goal not in path[-1].word or engine.spell_count(path[-1].hist) != cost + offset:
        raise RuntimeError('Native discovery returned an invalid solution')
    result.states = path
    return result
