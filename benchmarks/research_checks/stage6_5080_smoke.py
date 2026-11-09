"""Diagnostic GPU-transfer round trip and short Continuum serving, outside timing."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from harness.gpu_lock import GPULock


def kernel():
    import torch
    import lmcache.c_ops as ops
    if torch.cuda.get_device_capability() != (12, 0):
        raise ValueError('This diagnostic is specifically for the RTX 5080')
    torch.manual_seed(17011)
    layers, pages, block, heads, dim = 3, 16, 16, 2, 64
    tokens = pages * block
    source = [torch.randn(2, pages, block, heads, dim, device='cuda', dtype=torch.float16)
              for _ in range(layers)]
    target = [torch.zeros_like(t) for t in source]
    slots = torch.randperm(tokens, device='cuda')
    host = torch.empty(2, layers, tokens, heads * dim, dtype=torch.float16, pin_memory=True)
    pointers = torch.tensor([t.data_ptr() for t in source], dtype=torch.int64, pin_memory=True)
    ops.multi_layer_kv_transfer(host, pointers, slots, source[0].device, tokens, True, False)
    torch.cuda.synchronize()
    for index, tensor in enumerate(source):
        torch.testing.assert_close(host[:, index], tensor.reshape(2, tokens, heads * dim)[:, slots].cpu(), rtol=0, atol=0)
    restored = torch.tensor([t.data_ptr() for t in target], dtype=torch.int64, pin_memory=True)
    ops.multi_layer_kv_transfer(host, restored, slots, target[0].device, tokens, False, False)
    torch.cuda.synchronize()
    for before, after in zip(source, target, strict=True):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    extension = Path(ops.__file__)
    return dict(passed=True, operation='multi_layer_kv_transfer', dtype='float16',
        layers=layers, tokens=tokens, pinned_cpu_bytes=host.numel() * host.element_size(),
        gpu_to_cpu_exact=True, cpu_to_gpu_exact=True,
        torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
        capability=list(torch.cuda.get_device_capability()), extension=str(extension),
        extension_sha256=hashlib.sha256(extension.read_bytes()).hexdigest(),
        scope='Exact pinned CPU/GPU KV round trip; no performance or model-correctness claim')


def serve(campaign, output):
    from benchmarks.serving_suite.campaign_contracts import load, atomic_json
    from benchmarks.serving_suite.runner import plan_for_variant, run_case
    cells = load(campaign / 'resolved.json')['cells']
    cell = next(c for c in cells if c['model'] == 'qwen3_8b_awq' and c['cell'] == 'B06')
    binding = cell['workload_binding']
    plan = plan_for_variant(copy.deepcopy(load(binding['plan'])), 'continuum')
    corpus = copy.deepcopy(load(binding['corpus']))
    corpus['sessions'] = corpus['sessions'][:1]
    nodes = corpus['sessions'][0]['graph']['nodes']
    plan.update(sessions=1, concurrency=1, client_concurrency=1, max_tokens=16,
        expected_requests_per_wave=sum(n['kind'] == 'request' for n in nodes),
        expected_tools_per_wave=sum(n['kind'] == 'tool' for n in nodes))
    atomic_json(output / 'plan.json', plan)
    atomic_json(output / 'corpus.json', corpus)
    case = run_case(plan, corpus['sessions'], 'candidate', 0, output, variant='continuum')
    atomic_json(output / 'case.json', case)
    passed = (case['evaluation']['status'] == 'passed' and case['summary']['completed'] ==
              plan['expected_requests_per_wave'] and case['cleanup']['verified'])
    return dict(passed=passed, evaluation=case['evaluation'], completed=case['summary']['completed'],
        geometry=case.get('effective_geometry'), activation=case.get('campaign', {}).get('policy_activation'),
        cleanup=case['cleanup'],
        scope='One-session 16-output-token diagnostic from the pinned development workload; excluded from campaign measurements')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('kernel', 'serve'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--campaign', type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Use a fresh smoke directory')
    args.out.mkdir(parents=True)
    with GPULock():
        result = kernel() if args.kind == 'kernel' else serve(args.campaign.resolve(), args.out.resolve())
    (args.out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 2)
