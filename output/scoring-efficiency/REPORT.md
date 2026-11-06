# Scoring efficiency trial — completed

Work reduction helped modestly in this real-serving trial, but ARC remained faster overall. Indexed search did not help the naturally tiny pool. The largest measured feature cost surrounds capture's GPU query-index construction, rather than vector comparisons.

## Implementation

Both controls are opt-in; defaults remain continuous, flat scoring.

- **Work reduction:** sample queries every eight captures into bounded snapshots; score eligible blocks only when multiple paused sessions compete for insufficient staging room. Skip all-fit ranking without inventing scores. Preserve feature age and use a bounded missing-score fallback.
- **Smarter search:** cluster block summaries, route queries to a few groups, compute exact scores for visited blocks, and rotate exploration. Pools below 128 blocks use flat search. Candidate selection is approximate; exact KV identity/reuse rules are unchanged.
- **Instrumentation:** optional host wall/thread-CPU aggregates, counters and sampled CUDA stream intervals cover capture, search, scoring, readback, ranking, EMA, metadata, decisions, logging and transfer submission. Diagnostic hooks add scheduler/runner, queue, RPC and JSON/serialization measurements. Explicit pre-teardown export closes the initial incomplete shutdown snapshots.

## Real serving comparison

Cached dense **Qwen3-4B**, BF16, one GPU, eager execution; six simulated closed-loop agents plus 12 background requests. Each arm served 45 requests / 3,264 output tokens. Initial contexts: 2,048–2,560 tokens; private waits: 75–2,400 ms. GPU KV pool: 1,440 MiB; DRAM tier: 448 MiB; maximum three running sequences. This is generated agent traffic, not a production trace or Qwen3.5.

| Observer-off configuration | Median release→first token, ms | Background median, ms | Output tokens/s | Executed prompt tokens |
|---|---:|---:|---:|---:|
| ARC demand | 2742.5 | 7212.7 | 82.342 | 80515 |
| Current semantic | 3149.5 | 7435.3 | 81.257 | 82435 |
| Work reduction | 2738.5 | 7203.1 | 81.892 | 80563 |
| Indexed search enabled | 3168.6 | 7435.8 | 81.153 | 82435 |
| Both enabled | 2839.4 | 7219.0 | 81.843 | 80563 |

Work reduction improved throughput **0.78%** and median first-token time **13.05%** versus current semantic, while throughput remained **0.55% below ARC**. These are single-run observations in fixed order without significance estimates. Closed-loop arrivals and cache histories differ; current/index execute 1,920 more prompt tokens than ARC versus 48 more for work/combined. The differences cannot all be attributed to scoring cost. Stride reduction and opportunity-only scoring were measured together, not independently.

Only **11 blocks / 24.75 MiB** naturally entered DRAM, usually 1–2 blocks per eligible source. There was **no constrained multi-session ranking**. Production indexed search stayed on its small-pool flat fallback; opportunity mode skipped scoring when allocation did not need ranking. Competitive fallback/freshness behavior was CPU-tested, not naturally exercised here.

Current semantic staged 18 MiB with 15.75 MiB consumed on time; work reduction staged 13.5 MiB with 6.75 MiB consumed on time. Both needed 27 MiB demand H2D versus ARC's 38.25 MiB. On-time staging covered less than 0.2% of returning context bytes. All consumed prefetched prefixes were partial, so actual GPU splice ran and native adoption did not. No capacity or semantic-ranking advantage is established.

## Complete component attribution

The ninth lifecycle was an observer-on diagnostic. These are cumulative **inclusive host wall** spans across its roughly 40-second timed load. Nested scopes overlap: do not add them or interpret them as exclusive/reclaimable latency.

| Component | Calls | Host wall total, ms |
|---|---:|---:|
| Model execution, including nested capture/scoring | 1180 | 39657.8 |
| Capture extraction | 1139 | 13615.3 |
| GPU query-index tensor creation/upload, inside capture | 1139 | 13216.8 |
| Query gather / head-group reduction | 1139 each | 44.2 / 98.7 |
| Scoring total | 1139 | 195.2 |
| Vector comparisons, inside scoring | 1000 | 73.1 |
| Score readback, inside scoring | 1000 | 59.3 |
| Ranking/metadata, inside scoring | 1000 | 23.0 |
| Scheduler schedule | 1175 | 167.0 |
| Prefetch decision | 1175 | 7.2 |
| Worker metadata construction | 1175 | 56.7 |
| Consume worker metadata | 2350 | 49.2 |
| JSON logging encode / stdout flush | 206 each | 2.7 / 9.7 |
| Transfer store / load submission | 11 / 15 | 1117.0 / 5.6 |
| Partial-prefix splice | 3 | 12.6 |

The 13.217-second boundary wraps `torch.tensor(last_indices, device=GPU)`. It can include driver waiting/spinning for earlier GPU work; **it is not 13 seconds of extra GPU computation or proven removable delay**. Its 72 sampled CUDA stream ranges total 4.262 ms; samples do not cover every call. Thread CPU may include driver spin. The next debugging target is query-index creation and synchronization with model execution, before additional vector-search tuning.

Client JSON encoding totaled 3.587 ms; SSE JSON decoding 24.862 ms; server request JSON parsing 5.661 ms. API hint RPC envelopes totaled 4.479 seconds across 109 calls versus 21.929 ms inside the engine handler; their difference is not pure network time. Actual msgpack counts/bytes, per-request scheduler/worker queue boundaries, capture-to-score provenance and transfer/fence envelopes are saved. Worker prepare-host is not device-ready time. Complete ninth-run profiles supersede initial diagnostic coverage, without rewriting initial raw observations.

## Real-tensor indexed-search replay

Post-load export preserved eight real query batches / 24 correlated rows and the final actual 11-block summary bank. Forced-index replay used these tensors without feedback into serving. Earlier queries against final-bank summaries form a static computational replay, not a causal policy test.

The index reranked 226 pairs versus 264 flat comparisons (14.39% fewer), **plus 264 routing and 484 build comparisons**. Top-1/top-3 recall was 100%; mean top-8 recall was **88.02%**, ranging from 75–100%. Visited exact scores matched flat within 1e-3.

Even excluding build, indexed routing/gather/exact-score/readback spans totaled about 5.8 ms versus about 1.0 ms for flat math/readback. First cold build took 8.081 seconds and may include lazy initialization/host launch gaps. Warmed rebuild and larger-pool crossover are unmeasured. Keep indexing disabled for this workload.

## Verification and limits

**405/405 real requests across nine serving lifecycles matched exact input/output token lists and ordered native block keys**, without serving failures/OOM/allocation failures. Native CPU validation passed 263 tests, then 266 including diagnostic checks, with CUDA initialization forbidden. Mac independently passed six index/profiler tests and five extracted production-worker checks.

Production source matches frozen measured v3. After measurement, two test-only helper imports were qualified with `tests.` for normal package discovery. Static resolution and Mac checks passed; that import-only correction was not rerun in native vLLM. The original frozen test manifest remains preserved.

Still unmeasured: isolated DMA device duration, pure network time, actual hint/SSE response JSON encoding, Python index memory peak, independent cold-build initialization costs, warmed/large-pool crossover and repeated-trial noise. Complete exclusive timing attribution is not claimed.

## Saved evidence and cleanup

- [Detailed serving report](results/REPORT.md), [per-request CSV](results/measurements.csv), [validity checks](results/evidence/validity.json).
- [Complete diagnostic](results/diagnostic/REPORT.md), [component CSV](results/diagnostic/components.csv), [scheduler queues](results/diagnostic/scheduler-queue.csv), [per-query replay](results/diagnostic/replay.csv).
- [Frozen source pins](candidate-pins.json), `candidate-source-v3.tar.gz`; reproduction archives and compressed raw evidence under `results/`.
- [PC handback](results/evidence/final-handback.json), [verification](results/evidence/final-combined-audit.json), [storage inventory](STORAGE.json).

All owned servers exited, descendant accounting is empty, project lock is free, custody is released and stale offers are cancelled. No other job was stopped or resumed. Useful models, environments, rollback material and results remain. No commit or push was made.
