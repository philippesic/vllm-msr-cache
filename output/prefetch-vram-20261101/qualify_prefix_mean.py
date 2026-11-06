"""Root-owned standalone CUDA qualification; does not start a vLLM server.

Run with the intended source first on PYTHONPATH. Prints JSON; --output saves
identities and measured numerical differences. No timing/performance claims.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from vllm.v1.kv_offload.base import make_offload_key
import semantic_offload.worker as worker_module
from semantic_offload.worker import SemanticOffloadingWorker


def pin(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def make_worker():
    worker = SemanticOffloadingWorker.__new__(SemanticOffloadingWorker)
    worker._method = 'mean';worker._head_aggregation = 'mean'
    worker._scoring_mode = 'synchronous';worker._scoring_schedule = 'opportunity'
    worker._summary_generation = 0;worker._summary_stream = torch.cuda.Stream()
    worker._query_snapshots = {};worker._query_snapshot_events = {}
    worker._query_snapshot_max_age_ns = 10_000_000_000;worker._query_snapshot_max_requests = 8
    worker._score_transport_context = {'epoch': 'cuda-qualification:0',
        'prefix_mean_prefetch_scoring': True, 'sources': {'active-a': 11, 'active-b': 12}}
    worker.durable_summaries = {}
    return worker


def qualify():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA qualification requires the root-owned idle GPU')
    torch.manual_seed(1103)
    worker = make_worker()
    lengths = (96, 127, 193)
    keys = [make_offload_key(str(i).encode(), 0) for i in range(sum(lengths))]
    # Use the real summary builder and publication machinery on a separate
    # producer stream. No producer synchronization precedes the scoring helper.
    with torch.cuda.stream(worker._summary_stream):
        kv = torch.randn(sum(lengths), 2, 16, 128, device='cuda', dtype=torch.float16)
        summaries = worker._summarize_mean_key_batches(kv.float())
        worker.durable_summaries = dict(zip(keys, summaries))
        worker._register_summary_publications(101, keys)
    producer = torch.cuda.Stream()
    with torch.cuda.stream(producer):
        query = torch.randn(2, 2, 128, device='cuda')
        worker._on_queries_captured(['active-a', 'active-b'], query)
    request = {'active_query_ids': ['active-a', 'active-b'], 'candidates': {}}
    offset = 0
    for i, length in enumerate(lengths):
        selected = tuple(keys[offset:offset + length]);offset += length
        request['candidates'][str(i)] = {'event_id': str(i), 'keys': selected,
                                        'residency_versions': {k: 1 for k in selected}}
    worker.receive_prefix_score_requests(request)
    observation, = worker.pop_prefix_score_observations()
    torch.cuda.synchronize()  # Only after exercising production wait ordering.
    common_query = query.mean(0)
    offsets, results, dense_values = 0, [], []
    for i, length in enumerate(lengths):
        selected_means = torch.stack([torch.stack([s.mean for s in worker.durable_summaries[k]])
                                      for k in keys[offsets:offsets + length]])
        dense = (selected_means * common_query).sum(-1).mean().item()
        # Also compare a CPU reduction of the same already-published FP32 inputs.
        cpu = (selected_means.cpu() * query.cpu().mean(0)).sum(-1).mean().item()
        actual = observation['candidates'][str(i)]['score']
        torch.testing.assert_close(torch.tensor(actual), torch.tensor(dense), atol=2e-5, rtol=2e-5)
        dense_values.append(dense)
        results.append({'keys': length, 'centroid_gpu': actual, 'dense_gpu': dense,
                        'dense_cpu': cpu, 'gpu_absolute_error': abs(actual - dense),
                        'cpu_absolute_error': abs(actual - cpu)})
        offsets += length
    centroid_values = [observation['candidates'][str(i)]['score'] for i in range(len(lengths))]
    rank = lambda xs: sorted(range(len(xs)), key=lambda i: (-xs[i], i))
    assert rank(centroid_values) == rank(dense_values)
    # Centroids must not alias summary or KV storage, including cached hits.
    original_centroids = [v.clone() for v in worker._prefix_mean_centroid_cache.values()]
    old_scores = {k: v['score'] for k, v in observation['candidates'].items()}
    with torch.cuda.stream(worker._summary_stream):
        kv.fill_(777)
        for heads in worker.durable_summaries.values():
            for summary in heads: summary.mean.fill_(999)
    worker.receive_prefix_score_requests(request)
    cached, = worker.pop_prefix_score_observations()
    assert {k: v['score'] for k, v in cached['candidates'].items()} == old_scores
    for actual, original in zip(worker._prefix_mean_centroid_cache.values(), original_centroids):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
    # A real republication changes the signature and recomputes the centroid.
    worker._register_summary_publications(102, keys)
    worker.receive_prefix_score_requests(request)
    changed, = worker.pop_prefix_score_observations()
    assert changed['candidates']['0']['score'] != old_scores['0']
    worker._durable_summary_publications.clear()
    worker.receive_prefix_score_requests(request)
    assert not worker.pop_prefix_score_observations()
    worker._register_summary_publications(103, keys)
    for job in range(104, 128):
        worker._register_summary_publications(job, keys)
        worker.receive_prefix_score_requests(request)
        worker.pop_prefix_score_observations()
    cache = worker._prefix_mean_centroid_cache
    cache_bytes = sum(t.untyped_storage().nbytes() for t in cache.values())
    assert len(cache) <= 16 and cache_bytes <= 1024 * 1024
    assert all(not t.requires_grad and t.grad_fn is None for t in cache.values())
    torch.cuda.synchronize()
    return {'status': 'passed', 'device': torch.cuda.get_device_name(),
            'script_pin': pin(__file__), 'worker_pin': pin(worker_module.__file__),
            'metric': observation['metric'], 'comparisons': results,
            'rank_equal': True, 'producer_stream_and_query_event_waits_exercised': True,
            'centroid_owned_after_kv_and_summary_overwrite': True,
            'changed_publication_rebuilt': True, 'missing_publication_rejected': True,
            'bounded_cache_entries': len(cache), 'bounded_cache_storage_bytes': cache_bytes,
            'worker_counters': worker._prefix_mean_scoring_stats,
            'limitations': ['Same-query raw metric qualification; does not establish old EMA ranking equivalence or production ranking quality.',
                            'CPU consumer identity/residency/lease rejection contracts are separate pytest checks.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = qualify()
    if args.output: args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
