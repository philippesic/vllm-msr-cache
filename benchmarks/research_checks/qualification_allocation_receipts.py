"""Check startup observations of actual CPU KV backing, not requested pin flags."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PREFIX = 'SEMANTIC_OFFLOAD_ALLOCATION '
SCHEMA = 'semantic-offload-allocation-v1'


def check_observation(observation):
    buffers = observation.get('cpu_buffers', [])
    chunk = observation.get('kv_bytes_per_chunk', 0)
    requested = observation.get('requested_cpu_bytes', 0)
    blocks = observation.get('num_cpu_blocks', 0)
    valid_geometry = all(type(x) is int and x > 0 for x in (chunk, requested, blocks))
    allocated = sum(b.get('nbytes', 0) for b in buffers)
    checks = dict(
        observed_supported_buffers=observation.get('schema') == SCHEMA and observation.get('available') is True and bool(buffers),
        all_cpu=bool(buffers) and all(b.get('device') == 'cpu' for b in buffers),
        all_pinned=bool(buffers) and all(b.get('pinned') is True for b in buffers),
        load_store_share_backing=bool(buffers) and all(b.get('load_uses_same_storage') is True for b in buffers),
        actual_bytes_match_geometry=valid_geometry and allocated == blocks*chunk,
        actual_bytes_match_rounded_budget=valid_geometry and allocated == (requested//chunk)*chunk,
        single_block_chunks=observation.get('blocks_per_chunk') == 1,
    )
    return dict(checks=checks, passed=all(checks.values()), actual_cpu_bytes=allocated,
                observation=observation)


def report(directory):
    directory = Path(directory)
    result = json.loads((directory/'result.json').read_text())
    names = [c['name'] for c in result['cases']] if 'cases' in result else ['.']
    cases = []
    for name in names:
        path = directory/name/'server.log'
        observations = []
        malformed = []
        for line in path.read_text(errors='replace').splitlines() if path.exists() else []:
            if PREFIX not in line:
                continue
            try:
                observations.append(check_observation(json.loads(line.split(PREFIX, 1)[1])))
            except (TypeError, ValueError, AttributeError) as exc:
                malformed.append(str(exc))
        cases.append(dict(case=name, observations=observations, malformed=malformed,
            actual_pinned_path_established=len(observations) == 1 and not malformed and observations[0]['passed']))
    return dict(schema='wait-hint-actual-allocation-receipts-v1', cases=cases,
        all_actual_pinned_paths_established=bool(cases) and all(c['actual_pinned_path_established'] for c in cases),
        scope='One startup observation per single-worker selected cell: actual CPU device, Tensor.is_pinned(), backing shared by load/store, and allocated byte geometry. No inference from environment flags, no retroactive proof for older runs, and no KV-content or transfer-overlap claim.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    value = report(args.directory)
    args.output.write_text(json.dumps(value, indent=2)+'\n')
    print('Actual pinned CPU paths established:', value['all_actual_pinned_paths_established'])
    return 0 if value['all_actual_pinned_paths_established'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
