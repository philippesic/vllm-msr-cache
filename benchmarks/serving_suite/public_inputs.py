"""Pinned public source preparation and fresh, auditable task construction.

No performance selection occurs here. Historical task lists define exclusions,
not timing data. Corpus partitions and all source IDs are retained explicitly.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

from .campaign_contracts import atomic_json, file_hash, load
from .contracts import validate_corpus

SOURCE_PINS = {
    'more-itertools': ('https://github.com/more-itertools/more-itertools.git', 'f42daddf205f335fea6e200dad46049f33e71221'),
    'peps': ('https://github.com/python/peps.git', 'd728748f4a99ec08e3f4c101c1cb07ba16b140f1'),
}
DOCUMENT_FRESH = [241, 279, 282, 289, 305, 307, 310, 311, 314, 315, 318, 322, 327, 328, 342, 352]
DOCUMENT_REPAIR = [353, 357, 358, 366, 370, 371, 372, 376, 386, 389, 391, 393, 405, 409, 411, 414]


def prepare_sources(cache):
    cache = Path(cache)
    result = {}
    for name, (repository, revision) in SOURCE_PINS.items():
        target = cache / 'sources' / name
        if not target.exists():
            target.mkdir(parents=True)
            subprocess.run(['git', 'init', '-q', str(target)], check=True)
            subprocess.run(['git', '-C', str(target), 'remote', 'add', 'origin', repository], check=True)
            subprocess.run(['git', '-C', str(target), 'fetch', '--depth', '1', 'origin', revision], check=True)
            subprocess.run(['git', '-C', str(target), 'checkout', '--detach', revision], check=True)
        actual = subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != revision or subprocess.check_output(['git', '-C', str(target), 'status', '--porcelain'], text=True).strip():
            raise ValueError(f'Prepared public source changed: {name}')
        licenses = {str(p.relative_to(target)): file_hash(p) for p in target.glob('*LICENSE*') if p.is_file()}
        if not licenses:
            licenses = {str(p.relative_to(target)): file_hash(p) for p in target.glob('*COPYING*') if p.is_file()}
        result[name] = dict(repository=repository, revision=revision, path=str(target.resolve()), licenses=licenses)
    atomic_json(cache / 'sources.json', result)
    return result


def excerpt_symbol(path, node):
    lines = Path(path).read_text().splitlines(True)
    first = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])])
    return ''.join(lines[first - 1:node.end_lineno])


def repository_tasks(root, partition, python):
    from benchmarks.research_checks.qualification_inputs import REPOSITORY_TASKS, REPOSITORY_RESERVE
    root = Path(root).resolve()
    source, tests = root / 'more_itertools/more.py', root / 'tests/test_more.py'
    functions = {n.name: n for n in ast.parse(source.read_text()).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and not n.name.startswith('_')}
    classes = {n.name: n for n in ast.parse(tests.read_text()).body if isinstance(n, ast.ClassDef)}
    old = {name for name, _ in REPOSITORY_TASKS + REPOSITORY_RESERVE}
    normalized = lambda value: re.sub('[^a-z0-9]', '', value.lower()).removeprefix('test').removesuffix('tests').removesuffix('test')
    candidates = []
    for name, function in sorted(functions.items()):
        matches = [test for test in classes if normalized(test) == normalized(name)]
        if name not in old and len(matches) == 1:
            # Whole symbol excerpts, never arbitrary character truncation.
            text = excerpt_symbol(source, function) + '\n\n' + excerpt_symbol(tests, classes[matches[0]])
            if len(text) <= 18000:
                candidates.append((name, matches[0], text))
    if partition == 'development':
        selected = [(name, test, excerpt_symbol(source, functions[name]) + '\n\n' + excerpt_symbol(tests, classes[test])) for name, test in REPOSITORY_TASKS]
    elif partition in ('confirmation', 'repair'):
        offset = 0 if partition == 'confirmation' else 16
        selected = candidates[offset:offset + 16]
    else:
        raise ValueError('Unknown task partition')
    if len(selected) != 16:
        raise ValueError(f'Insufficient fresh repository tasks for {partition}: {len(selected)}')
    sessions = []
    for name, test, context in selected:
        sessions.append(dict(id='more-' + name, repository=str(root), context=f'PUBLIC SOURCE more_itertools.{name} AND TEST CLASS {test}\n' + context,
            input_files={str(source): file_hash(source), str(tests): file_hash(tests)}, evidence=name,
            quality_reference=dict(symbol=name, test_class=test, source_symbol=excerpt_symbol(source, functions[name]), test_symbol=excerpt_symbol(tests, classes[test])),
            turns=[f'Investigate {name}: describe its purpose and plan a boundary-case check.',
                   f'Explain one concrete {name} behavior supported by the actual search results.',
                   f'Interpret the observed tests for {name}. Distinguish evidence from assumptions.',
                   f'Report {name}: one supported behavior, test_status=PASS or FAIL from the executed tests, and a concrete limitation.'],
            tools=[dict(kind='search', argv=['rg', '-n', '--max-count', '30', name, str(source)], valid_exit_codes=[0]),
                   dict(kind='test', argv=[python, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', f'tests/test_more.py::{test}'], valid_exit_codes=[0]),
                   dict(kind='evidence', argv=['rg', '-n', '--max-count', '35', 'assert|raise|def test_', str(tests)], valid_exit_codes=[0])]))
    return corpus(sessions, 'repository', partition, SOURCE_PINS['more-itertools'])


def document_tasks(root, partition, python, references):
    from benchmarks.research_checks.qualification_inputs import DOCUMENT_TASKS
    from benchmarks.research_checks.qualification_inputs import parse_document
    root = Path(root).resolve()
    numbers = {'development': DOCUMENT_TASKS, 'confirmation': DOCUMENT_FRESH, 'repair': DOCUMENT_REPAIR}[partition]
    refs, sessions = {}, []
    for number in numbers:
        path = root / 'peps' / f'pep-{number:04d}.rst'
        text = path.read_text()
        expected = {}
        for key in ('PEP', 'Title', 'Status', 'Type', 'Created'):
            match = re.search(r'^' + re.escape(key) + r':\s*(.+)$', text, re.M)
            if match is None:
                raise ValueError(f'Public PEP missing {key}: {number}')
            expected[key] = match.group(1)
        refs[str(number)] = expected
        # Keep the header and whole opening sections. The full original remains
        # available to the bounded real tools and the quality source audit.
        chunks = re.split(r'(?m)(?=^[^\n]+\n[=~-]{3,}\s*$)', text)
        shown = ''
        for chunk in chunks:
            if shown and len(shown) + len(chunk) > 14000:
                break
            shown += chunk
        common = [python, '-m', 'benchmarks.research_checks.qualification_inputs', 'document', '--file', str(path), '--reference', str(Path(references).resolve())]
        sessions.append(dict(id=f'pep-{number:04d}', repository=str(Path(__file__).resolve().parents[2]),
            context=f'PUBLIC PEP {number}; complete opening sections, {len(shown)}/{len(text)} characters\n' + shown,
            input_files={str(path): file_hash(path)}, evidence=f'PEP {number}', quality_reference=expected,
            turns=[f'Investigate PEP {number}: identify its purpose and plan how to verify its status and a rule.',
                   'Use the actual document search results to identify its specification and restrictions.',
                   'Interpret the actual metadata validation; distinguish proposal status from runtime support.',
                   f'Report PEP {number}: its Status and Type, one supported rule, test_status=PASS or FAIL from validation, and a concrete limitation.'],
            tools=[dict(kind='search', argv=['rg', '-n', '--max-count', '35', '^Status:|^Type:|^Title:|^Abstract|^Specification|^Rationale|^Backwards|^Security|^Reference', str(path)], valid_exit_codes=[0]),
                   dict(kind='test', argv=common + ['--action', 'validate'], valid_exit_codes=[0]),
                   dict(kind='evidence', argv=common + ['--action', 'summarize'], valid_exit_codes=[0])]))
    atomic_json(references, refs)
    return corpus(sessions, 'documents', partition, SOURCE_PINS['peps'])


def corpus(sessions, adapter, partition, source):
    return validate_corpus(dict(schema='semantic-serving-corpus-v1', synthetic=False,
        provenance=f'Authored {adapter} workflows over pinned public source {source[0]} at {source[1]}; {partition} partition. Real read-only tools, authored arrivals, no production/autonomous-agent claim.',
        workload_kind=adapter, partition=partition, cohort_mode='fixed', sessions=sessions))
