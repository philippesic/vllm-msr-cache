"""Prepare a bounded competing-prefix trace from frozen local token pools.

This writes only prepared/competing-{seed}.json. It never imports a model,
downloads data, initializes CUDA, or alters another experiment's frozen manifest.
Policy-visible observed prefixes and driver-private future appends stay separate.
"""
import hashlib
import json
from pathlib import Path

H = Path(__file__).resolve().parent
ROOT = H.parents[1]
OLD = ROOT / 'output/scoring-overhead-20261101/prepared'
SEEDS = (1103, 2207, 3301)
BLOCK_BYTES = 458752


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(seed):
    source_path = OLD / f'multi-{seed}.json'
    old = json.loads(source_path.read_text())
    pools = [session['initial_tokens'] for session in old['sessions']]
    assert len(pools) >= 6
    first_tokens = set()

    def context(pool_index, length, start):
        pool = pools[pool_index]
        # Skip the common "Session-" heading. Use an existing, distinct first
        # token for every request context so not even the initial cache block
        # can accidentally share a chain with another warm/peer context.
        while pool[start] in first_tokens:
            start += 1
            assert start < min(128, len(pool))
        first_tokens.add(pool[start])
        repeated = pool * ((length + start + len(pool) - 1) // len(pool))
        tokens = repeated[start:start + length]
        assert len(tokens) == length
        return tokens, start

    sources = [context(index, 1536, 3) for index in range(6)]
    pressures = [context((index + 2) % 6, 3072, 5) for index in range(6)]
    rounds = []
    for index in range(3):
        sessions = []
        for slot in range(2):
            source_index = index * 2 + slot
            append = old['sessions'][source_index]['turn_append_tokens'][1][:16]
            assert len(append) == 16
            sessions.append({'source': sources[source_index][0], 'append': append,
                             'token_pool_index': source_index,
                             'token_pool_offset': sources[source_index][1]})
        rounds.append({'sessions': sessions,
                       'pressure_a': pressures[index * 2][0],
                       'pressure_b': pressures[index * 2 + 1][0]})
    p = {
        'model': old['model'], 'namespace': old['namespace'],
        'trace': {'seed': seed}, 'rounds': rounds,
        'geometry': {'gpu_blocks': 640, 'gpu_kv_bytes': 640 * BLOCK_BYTES,
                     'cpu_blocks': 1024, 'cpu_kv_bytes': 1024 * BLOCK_BYTES,
                     'block_bytes': BLOCK_BYTES, 'max_model_len': 3328,
                     'max_num_seqs': 2, 'max_num_batched_tokens': 1024},
        'driver_private': {'post_peer_gap_ms': 250, 'return_delays_ms': [0, 0]},
        'dataset_contract': {
            'schema': 'causal-competing-peer-tools-v1', 'rounds': 3,
            'requests': 18, 'output_tokens': 768,
            'source_tokens': 1536, 'source_output_tokens': 32,
            'peer_tokens': 3072, 'peer_output_tokens': 32,
            'return_append_tokens': 16, 'return_output_tokens': 64,
            'default_return_release': 'both concurrent after both peers complete plus fixed 250ms',
            'peer_requests_are_serial': True,
            'source_warms_are_serial': True,
            'policy_constant_wait_estimate_ms': 250, 'policy_refresh_ms': 200,
            'policy_receives_future_input_or_release': False,
            'policy_observed_prefix': 'only the immutable 1536-token completed warm input',
            'binding': 'canonical_event_id from already received accepted acknowledgments',
            'private_followup': 'EXPERIMENT_RETURN_SKEW_MS and EXPERIMENT_RETURN_FIRST affect driver timers only',
            'public_phase_followup': 'EXPERIMENT_HINT_START=last_peer waits for actual second peer request start',
            'candidate_gpu_block_grid': [208, 256, 320], 'reference_gpu_blocks': 640,
            'minimum_simultaneous_return_blocks_including_null': 207,
            'maximum_peer_blocks': 194,
            'context_first_tokens_unique': True,
            'source_dataset': str(source_path.relative_to(ROOT)),
            'source_dataset_sha256': sha(source_path),
            'workload_sha256': sha(H / 'competing_workload.py'),
            'preparer_sha256': sha(Path(__file__)),
        },
    }
    validate(p)
    return p


def validate(p):
    assert len(p['rounds']) == 3
    contexts = []
    for round_data in p['rounds']:
        assert len(round_data['sessions']) == 2
        for session in round_data['sessions']:
            assert len(session['source']) == 1536 and len(session['append']) == 16
            assert 1536 + 32 + 16 + 64 <= p['geometry']['max_model_len']
            contexts.append(session['source'])
        for name in ('pressure_a', 'pressure_b'):
            assert len(round_data[name]) == 3072
            assert 3072 + 32 <= p['geometry']['max_model_len']
            contexts.append(round_data[name])
    assert len({context[0] for context in contexts}) == len(contexts) == 12
    assert len({tuple(context[:16]) for context in contexts}) == 12
    assert all(isinstance(token, int) and token >= 0 for context in contexts for token in context)
    assert 2 * ((1536 + 32 + 16 + 64 + 15) // 16) + 1 <= 208
    assert (3072 + 32 + 15) // 16 < 208
    assert p['geometry']['cpu_blocks'] == 1024


def main():
    destination = H / 'prepared'
    destination.mkdir(exist_ok=True)
    receipts = []
    for seed in SEEDS:
        p = build(seed)
        path = destination / f'competing-{seed}.json'
        content = json.dumps(p, indent=2) + '\n'
        if path.exists():
            assert path.read_text() == content, f'preserve existing differing dataset: {path}'
        else:
            path.write_text(content)
        receipts.append({'seed': seed, 'path': str(path), 'bytes': path.stat().st_size,
                         'requests': 18, 'output_tokens': 768,
                         'minimum_return_blocks_including_null': 207,
                         'sha256': sha(path)})
    print(json.dumps({'prepared': receipts, 'no_tokenizer_or_cuda_import': True}, indent=2))


if __name__ == '__main__':
    main()
