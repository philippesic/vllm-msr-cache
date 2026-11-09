"""Derive the authorized 5080 primary source without changing transfer inputs."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from benchmarks.serving_suite.campaign_contracts import (
    atomic_json, digest, load, seal_tree, verify_tree,
)

CHECKPOINT = '1319f09bafac46bab89df465dd433c67441f4b26'
COORDINATOR = 'benchmarks/serving_suite/campaign.py'


def prepare(bundle: Path, destination: Path, receipt: Path):
    bundle, destination = bundle.resolve(), destination.resolve()
    verify_tree(bundle)
    if destination.exists() or receipt.exists():
        raise ValueError('Preserve existing source revisions; select a new destination')
    portable = load(bundle / 'portable.json')
    original = copy.deepcopy(portable)
    project = Path(__file__).resolve().parents[2]
    coordinator = subprocess.check_output(
        ['git', '-C', str(project), 'show', CHECKPOINT + ':' + COORDINATOR])
    previous = (bundle / 'source' / COORDINATOR).read_bytes()
    # Primary already used explicit runtime binding. The checkpoint also brings
    # the bounded health checker into the source manifest, giving this execution
    # a new source identity with the repaired current coordinator.
    expected = previous.replace(
        b"('qualification_inputs','wait_hint_runtime_probe'):",
        b"('qualification_inputs','wait_hint_runtime_probe','stage6_healthcheck'):")
    if coordinator != expected or coordinator == previous:
        raise ValueError('Coordinator differs beyond the declared primary revision')
    shutil.copytree(bundle, destination)
    (destination / 'source' / COORDINATOR).write_bytes(coordinator)
    identity = json.loads(subprocess.check_output(
        [sys.executable, '-c', 'import json; from benchmarks.serving_suite.campaign import source_snapshot; print(json.dumps(source_snapshot()))'],
        cwd=destination / 'source',
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1')))
    parent_source = portable['protocol']['source']
    changed = sorted(k for k in set(identity['files']) | set(parent_source['files'])
                     if identity['files'].get(k) != parent_source['files'].get(k))
    if set(changed) != {COORDINATOR, 'benchmarks/research_checks/stage6_healthcheck.py'}:
        raise ValueError('Unexpected executed source changes: ' + ', '.join(changed))
    revision = dict(
        schema='stage6-5080-primary-source-revision-v1',
        parent_bundle_seal=digest(load(bundle / 'seal.json')),
        parent_source=parent_source, source=identity,
        coordinator_checkpoint=CHECKPOINT, changed_files=changed,
        authority='2026-11-08 authorized 5080 primary replication with repaired coordinator and new source identity',
        worker_interpreter='cell.profile.runtime_python or host.runtime_python explicitly',
        frozen_order_repeats_policy_models_workloads_preserved=True,
        parent_transfer_unchanged=True,
    )
    portable['protocol']['source'] = identity
    portable['replication_source_revision'] = revision
    if {k: v for k, v in portable.items() if k not in ('protocol', 'replication_source_revision')} != {
        k: v for k, v in original.items() if k != 'protocol'}:
        raise ValueError('Non-source transfer field changed')
    if {k: v for k, v in portable['protocol'].items() if k != 'source'} != {
        k: v for k, v in original['protocol'].items() if k != 'source'}:
        raise ValueError('Frozen schedule or non-source protocol changed')
    atomic_json(destination / 'portable.json', portable)
    revision['execution_bundle_seal'] = seal_tree(destination)
    revision['execution_bundle_files'] = verify_tree(destination)
    verify_tree(bundle)
    atomic_json(receipt, revision)
    return dict(source=identity['digest'], changed_files=changed,
                execution_bundle_seal=revision['execution_bundle_seal'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'out', 'receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.bundle, args.out, args.receipt), indent=2))
