"""Experimental FP16 KV-page SVD transport. Disabled unless kv_svd_rank > 0.

Each physical page is factored as consecutive 16 x 256 FP16 matrices (the
supported dense Triton head layout). Both K and V values are compressed. Only
factors are retained on CPU; no full-page shadow exists. Truncated SVD is
computed through the small Gram eigensystem, in FP32. Transfers are synchronous
for a conservative ownership fence; their reported time includes all codec work.
"""
import json
import time
from pathlib import Path

import torch
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.kv_offload.base import TransferResult


def encode(x, rank):
    # cuSOLVER workspace grows sharply above 256 small matrices on this pin.
    # Bound solver batches while preserving exact matrix order and factor rank.
    shape = x.shape[:-2]
    matrices = x.reshape(-1, 16, 256)
    left, projected = [], []
    for part in matrices.split(256):
        part = part.float()
        _, u = torch.linalg.eigh(part @ part.transpose(-1, -2))
        u = u[..., -rank:]
        left.append(u.half())
        projected.append((u.transpose(-1, -2) @ part).half())
    return (torch.cat(left).reshape(*shape, 16, rank),
            torch.cat(projected).reshape(*shape, rank, 256))


def decode(u, projected):
    return (u.float() @ projected.float()).half()


class SVDTransport:
    def __init__(self, caches, count, rank, trace=None):
        if type(rank) is not int or not 1 <= rank <= 15:
            raise ValueError('kv_svd_rank must be an integer in 1..15 (0 disables)')
        if len(caches.group_data_refs) != 1:
            raise ValueError('KV SVD supports one dense KV group only')
        self.rank = rank
        self.gpu = []
        self.factors = []
        self.ready = torch.zeros(count, dtype=torch.bool)
        self.results = []
        self.trace = Path(trace) if trace else None
        for cache in caches.tensors:
            page = cache.page_size_bytes
            if page % (2 * 16 * 256):
                raise ValueError('KV SVD requires FP16 dense 16x256 head pages')
            gpu = cache.tensor.view(torch.int8).view(-1, page).view(torch.float16)
            heads = page // (2 * 16 * 256)
            self.gpu.append(gpu)
            self.factors.append((
                torch.empty((count, heads, 16, rank), dtype=torch.float16, pin_memory=PIN_MEMORY),
                torch.empty((count, heads, rank, 256), dtype=torch.float16, pin_memory=PIN_MEMORY),
            ))
        self.allocated_bytes = sum(t.numel()*t.element_size() for pair in self.factors for t in pair)
        self.emit({'event': 'allocate', 'factor_bytes': self.allocated_bytes,
                   'uncompressed_bytes': sum(t.shape[1]*2*count for t in self.gpu), 'rank': rank})

    def emit(self, record):
        if self.trace:
            with self.trace.open('a') as f:
                f.write(json.dumps(record) + '\n')

    def transfer(self, job, src, dst, store):
        start = time.perf_counter()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record(torch.cuda.current_stream())
        gpu_spec, cpu_spec = (src, dst) if store else (dst, src)
        if len(gpu_spec.block_ids) != len(cpu_spec.block_ids):
            raise ValueError('KV SVD requires blocks_per_chunk=1')
        ids = [int(i) for i in cpu_spec.block_ids]
        if not store and not self.ready[ids].all():
            raise RuntimeError('KV SVD load from unpublished CPU slot')
        gi = torch.as_tensor(gpu_spec.block_ids, device=self.gpu[0].device, dtype=torch.long)
        # Uniform dense layers are batched into one eigensolver launch and one
        # host copy per factor. Full pages exist only as bounded job workspace.
        if len({tuple(g.shape[1:]) for g in self.gpu}) != 1:
            raise ValueError("KV SVD requires uniform dense layer pages")
        if store:
            x = torch.stack([g.index_select(0, gi) for g in self.gpu])
            x = x.reshape(len(self.gpu), len(ids), -1, 16, 256)
            u, p = encode(x, self.rank)
            error = energy = 0.0
            if self.trace:
                error = float((x.float()-decode(u,p).float()).square().sum())
                energy = float(x.float().square().sum())
            host_u, host_p = u.cpu(), p.cpu()
            for layer, (left, right) in enumerate(self.factors):
                left[ids] = host_u[layer]
                right[ids] = host_p[layer]
        else:
            host_u = torch.stack([left[ids] for left, _ in self.factors])
            host_p = torch.stack([right[ids] for _, right in self.factors])
            x = decode(host_u.to(self.gpu[0].device), host_p.to(self.gpu[0].device))
            for layer, gpu in enumerate(self.gpu):
                gpu.index_copy_(0, gi, x[layer].reshape(len(ids), -1))
            error = energy = 0.0
        total = sum(len(ids)*(left[0].numel()+right[0].numel())*2
                    for left,right in self.factors)
        end_event.record(torch.cuda.current_stream())
        end_event.synchronize()
        stream_interval = start_event.elapsed_time(end_event) / 1000
        if store:
            self.ready[ids] = True
        elapsed = time.perf_counter()-start
        self.emit({'event': 'store' if store else 'load', 'job': job,
                   'blocks': len(ids), 'factor_bytes': total, 'seconds': elapsed,
                   'cuda_stream_interval_s': stream_interval,
                   'relative_squared_error': error/energy if energy else None})
        self.results.append(TransferResult(job_id=job, success=True, transfer_size=total, transfer_time=stream_interval))
        return True

    def shutdown(self):
        self.gpu.clear()
        self.factors.clear()
        self.results.clear()


class SVDDirection:
    def __init__(self, transport, store):
        self.transport, self.store = transport, store
    def transfer_async(self, job, src, dst):
        return self.transport.transfer(job, src, dst, self.store)
    def get_finished(self):
        results, self.transport.results = self.transport.results, []
        return results
    def wait(self, jobs):
        pass  # submit completed and fenced the whole transfer
    def shutdown(self):
        self.transport.shutdown()
