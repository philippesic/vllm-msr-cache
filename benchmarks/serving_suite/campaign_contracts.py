"""Portable campaign identities, strict contracts and crash-safe attempt ledger.

No torch, transformers, CUDA or network imports belong in this module.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .contracts import digest, positive

SCHEMA = 'serving-campaign-v1'
PHASES = ('preflight', 'screen', 'pilot', 'confirm', 'overhead', 'operational')
STATES = {'planned', 'running', 'interrupted', 'failed', 'verified', 'superseded', 'unsupported', 'blocked'}
ID = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_.-]*$')


def atomic_json(path, value):
    """Unique staging name, durable contents, atomic replacement and parent fsync."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        parent = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load(path):
    def duplicate(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=duplicate,
                      parse_constant=lambda v: (_ for _ in ()).throw(ValueError(f'Nonfinite JSON: {v}')))


def fields(value, required, optional=(), *, label='record'):
    if not isinstance(value, dict):
        raise ValueError(f'{label}: expected object')
    missing = set(required) - value.keys()
    unknown = value.keys() - set(required) - set(optional)
    if missing or unknown:
        raise ValueError(f'{label}: missing {sorted(missing)}, unknown {sorted(unknown)}')


def identifier(value):
    if not isinstance(value, str) or not ID.fullmatch(value):
        raise ValueError(f'Invalid logical identifier: {value!r}')
    return value


def local_path(root, relative):
    """Reject traversal and symlink escapes, including nonexistent output paths."""
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError('Expected nonempty relative artifact path')
    if '..' in Path(relative).parts:
        raise ValueError('Artifact path traversal')
    path = Path(root) / relative
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('Artifact escapes root')
    return path


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_protocol(root, record):
    path = Path(root)/'protocol.lock.json'
    if path.exists():
        raise ValueError('Protocol already frozen')
    atomic_json(path,record)
    atomic_json(Path(root)/'protocol-seal.json',dict(sha256=file_hash(path)))
    path.chmod(0o444)


def read_protocol(root):
    path = Path(root)/'protocol.lock.json'
    receipt = load(Path(root)/'protocol-seal.json')
    if file_hash(path)!=receipt['sha256']:
        raise ValueError('Frozen protocol changed')
    return load(path)


def validate_campaign(data):
    fields(data, ('schema', 'id', 'models', 'workloads', 'systems', 'matrix', 'policy', 'execution', 'seed', 'precision'))
    if data['schema'] != SCHEMA or data['execution'] != 'eager':
        raise ValueError('Expected serving-campaign-v1, eager execution')
    identifier(data['id'])
    positive(data['seed'], 'seed')
    for catalog in ('models', 'workloads', 'systems'):
        if not isinstance(data[catalog], str) or not data[catalog]:
            raise ValueError(f'{catalog}: catalog path required')
    if not isinstance(data['matrix'], dict) or not data['matrix']:
        raise ValueError('Empty matrix')
    for model, cells in data['matrix'].items():
        identifier(model)
        if not isinstance(cells, dict) or not cells:
            raise ValueError('Empty model cells')
        for cell, systems in cells.items():
            identifier(cell)
            if (not isinstance(systems, list) or len(systems) != len(set(systems))
                    or not systems or set(systems) - {'arc', 'wait', 'continuum', 'upstream'}):
                raise ValueError('Invalid system selection')
            if not {'arc', 'wait'} <= set(systems):
                raise ValueError('Every comparison requires ARC/wait')
    fields(data['precision'], ('allowed_repeats', 'target_log_half_width', 'throughput_benefit', 'no_regression_ratio', 'unhinted_latency_ratio', 'overhead_repeats'))
    if data['precision']['allowed_repeats'] != [6, 8, 10, 12] or data['precision']['overhead_repeats'] != 6:
        raise ValueError('Stage 6 repetition choices changed')
    for name in ('target_log_half_width', 'throughput_benefit', 'no_regression_ratio', 'unhinted_latency_ratio'):
        positive(data['precision'][name], name, False)
    return data


def validate_host(data):
    fields(data, ('schema', 'id', 'device', 'runtime_python', 'asset_cache', 'models', 'runtime_bindings', 'limits'), ('model_profiles', 'workload_bindings', 'published_runtimes', 'resolved_published_runtimes', 'confirmation_bindings'))
    if data['schema'] != 'serving-campaign-host-v1' or data['device'] != 'cuda:0':
        raise ValueError('Unsupported host schema/device')
    identifier(data['id'])
    for key in ('runtime_python', 'asset_cache'):
        if not isinstance(data[key], str) or not Path(data[key]).is_absolute():
            raise ValueError(f'{key}: absolute local path required')
    fields(data['limits'], ('startup_timeout_s', 'request_timeout_s', 'tool_timeout_s', 'case_timeout_s'))
    for key, value in data['limits'].items():
        positive(value, key, False)
    for key in ('models', 'runtime_bindings', 'model_profiles', 'workload_bindings'):
        if not isinstance(data.get(key, {}), dict):
            raise ValueError(f'{key}: expected mapping')
    return data


def cell_inputs(cell):
    return dict(model=cell['model'],cell=cell['cell'],systems=cell['systems'],
        profile={k:v for k,v in (cell.get('profile') or {}).items() if k not in ('runtime_python','allocation_receipts')},
        development={k:cell['workload_binding'].get(k) for k in ('plan_digest','corpus_digest')} if cell.get('workload_binding') else None,
        confirmation={k:cell['confirmation_binding'].get(k) for k in ('plan_digest','corpus_digest')} if cell.get('confirmation_binding') else None,
        driver_runtime=cell.get('runtime_identity'),external_runtimes=cell.get('system_identities',{}))


def seal_tree(root):
    root = Path(root)
    files = {}
    for path in sorted(root.rglob('*')):
        if path.name == 'seal.json' and path.parent == root:
            continue
        if path.is_symlink():
            raise ValueError('Sealed artifacts must be materialized, not symlinks')
        if path.is_file():
            files[str(path.relative_to(root))] = file_hash(path)
    record = {'schema': 'serving-campaign-seal-v1', 'files': files}
    atomic_json(root / 'seal.json', record)
    return digest(record)


def verify_tree(root, expected=None):
    root = Path(root)
    seal = load(root / 'seal.json')
    fields(seal, ('schema', 'files'))
    if seal['schema'] != 'serving-campaign-seal-v1' or not isinstance(seal['files'], dict):
        raise ValueError('Unknown seal schema')
    if expected is not None and digest(seal) != expected:
        raise ValueError('Seal identity changed')
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p != root / 'seal.json'}
    if actual != set(seal['files']):
        raise ValueError('Missing or unsealed artifacts')
    for name, expected_hash in seal['files'].items():
        path = local_path(root, name)
        if path.is_symlink() or file_hash(path) != expected_hash:
            raise ValueError(f'Changed artifact: {name}')
    return len(actual)


class Ledger:
    """Single writer per campaign; replay is authoritative after an index crash.

    A torn final event is retained in a recovery artifact before truncation. No
    earlier malformed line is repaired. The lock inode is never removed.
    """
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def writer(self):
        with (self.root / '.writer.lock').open('a+') as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('Campaign already has an active writer') from exc
            yield self

    def events(self):
        path = self.root / 'ledger.jsonl'
        if not path.exists():
            return []
        raw = path.read_bytes()
        if raw and not raw.endswith(b'\n'):
            raise ValueError('Torn ledger tail; run explicit recovery')
        result = []
        previous = None
        for line in raw.splitlines():
            row = json.loads(line)
            fingerprint = row.pop('digest')
            if row.get('sequence') != len(result) or row.get('previous') != previous or digest(row) != fingerprint:
                raise ValueError('Broken ledger event chain')
            row['digest'] = fingerprint
            previous = fingerprint
            result.append(row)
        return result

    def index(self):
        result = {}
        for event in self.events():
            result[event['attempt_id']] = event
        return result

    def event(self, case_id, attempt_id, state, **details):
        identifier(case_id)
        identifier(attempt_id)
        if state not in STATES:
            raise ValueError('Unknown attempt state')
        events = self.events()
        old = self.index().get(attempt_id)
        if old and (old['case_id'] != case_id or old['state'] in {'verified', 'failed', 'interrupted', 'unsupported', 'blocked', 'superseded'}):
            raise ValueError('Attempt identity/terminal state is immutable')
        if not old and state != 'planned':
            raise ValueError('Attempt must start planned')
        if old and state == 'planned':
            raise ValueError('Attempt already planned')
        row = dict(sequence=len(events), previous=events[-1]['digest'] if events else None,
                   timestamp_unix=time.time(), case_id=case_id, attempt_id=attempt_id, state=state, details=details)
        row['digest'] = digest(row)
        with (self.root / 'ledger.jsonl').open('ab') as stream:
            stream.write((json.dumps(row, sort_keys=True, allow_nan=False) + '\n').encode())
            stream.flush()
            os.fsync(stream.fileno())
        atomic_json(self.root / 'index.json', {'schema': SCHEMA, 'attempts': self.index()})
        return row

    def begin(self, identity, **details):
        case_id = digest(identity)
        attempt_id = uuid.uuid4().hex
        relative = f'attempts/{case_id}/{attempt_id}'
        self.event(case_id, attempt_id, 'planned', identity=identity, path=relative, **details)
        directory = local_path(self.root, relative)
        directory.mkdir(parents=True, exist_ok=False)
        atomic_json(directory / 'identity.json', identity)
        self.event(case_id, attempt_id, 'running', identity=identity, path=relative, **details)
        return attempt_id, directory

    def reusable(self, identity):
        case_id = digest(identity)
        for row in reversed(list(self.index().values())):
            if row['case_id'] == case_id and row['state'] == 'verified':
                verify_tree(local_path(self.root, row['details']['path']), row['details']['seal'])
                if row['details']['identity'] != identity:
                    raise ValueError('Attempt identity mismatch')
                return row
        return None

    def recover(self):
        path = self.root / 'ledger.jsonl'
        if path.exists():
            raw = path.read_bytes()
            if raw and not raw.endswith(b'\n'):
                tail = raw.rfind(b'\n') + 1
                (self.root / f'torn-ledger-{uuid.uuid4().hex}.bin').write_bytes(raw[tail:])
                with path.open('r+b') as stream:
                    stream.truncate(tail)
                    stream.flush()
                    os.fsync(stream.fileno())
        for row in list(self.index().values()):
            if row['state'] in {'planned', 'running'}:
                self.event(row['case_id'], row['attempt_id'], 'interrupted', **row['details'], recovery='New whole-case attempt required')
        return self.index()
