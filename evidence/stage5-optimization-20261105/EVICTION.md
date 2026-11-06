# Why faster CPU eviction did not establish faster serving

The algorithmic speedup reproduces on exact states captured from serving, but
native victim selection consumes too little total time in the primary and tight
workloads to explain a material end-to-end improvement. This is measured
attribution, not an assumption that the CPU microbenchmark was irrelevant.

| Workload / policy | Eviction calls | Eviction body, total | Workflow | Body / workflow |
| --- | ---: | ---: | ---: | ---: |
| Primary ARC | 222 | 9.97 ms | 111.13 s | 0.0090% |
| Primary wait | 213 | 4.75 ms | 105.60 s | 0.0045% |
| Primary linear wait | 214 | 4.16 ms | 103.84 s | 0.0040% |
| Tight ARC | 604 | 82.15 ms | 184.70 s | 0.0445% |
| Tight wait | 561 | 45.57 ms | 184.97 s | 0.0246% |
| Tight linear wait | 556 | 16.05 ms | 177.22 s | 0.0091% |
| Long documents, batch 4096, ARC | 465 | 159.42 ms | 175.04 s | 0.0911% |
| Long documents, batch 4096, linear ARC | 465 | 17.86 ms | 178.72 s | 0.0100% |

These are diagnostic runs, not a paired performance promotion. Their scheduling,
cache trajectories and transferred work differ, and profiling itself has cost.
The tight pair's several-second workflow difference cannot be attributed to its
roughly 30 ms difference in eviction body time. Earlier unprofiled screens also did
not establish a repeatable eviction improvement. Host body time is direct time
at stake, not a strict end-to-end upper bound: scheduling and GPU overlap may
amplify or hide changes.

## Actual-state replay

All 47 bounded before-state samples produced identical victim lists, block IDs,
reference counts, resident and ghost order, and adaptive target after eviction.
The offline replay retains actual byte keys and excludes state construction from
its clocks. Iterator counting uses a separate run; timed runs use ordinary
OrderedDict queues.

In real tight-cache samples, selecting 39 victims took 3074 iterator visits in
native ARC versus 101 in linear ARC and was about 7× faster in isolated replay.
A 64-victim sample took 2017 versus 64 visits and was about 5× faster. Thus the
repeated scanning found by the CPU experiment really exists in serving.

However, 186 of 213 primary wait-policy calls evicted one block. Single-victim
replay was slightly slower with the linear implementation because its setup has
no repeated scan to eliminate. The scheduler stores newly eligible chunks as
prefill and decoding advance; it does not normally evict 128–256 victims against
a 3584-entry protected queue prefix on every step. Queue capacity, victim count,
reference protection and the position of eligible entries all matter. The
synthetic 44 ms native call is not representative of the normal call distribution.

## Measurement and limits

The opt-in wrapper clocks only the existing policy body, without CUDA fences.
It records wall and thread CPU time, requested victims and queue/protection sizes.
Snapshot capture and call-record overhead are separate. Flush time is separate
and excludes final summary serialization. The measured native prepare_store
interval contains eviction and its wrapper overhead, and must not be added to
eviction body time. Clock overhead is not subtracted, which is relevant for
tiny calls. Final flushed summaries reconcile exactly with raw call
records; every request completed. The observer is disabled in ordinary runs.

Samples include the first call in each victim-count magnitude and periodic
calls, capped at 12 per engine. They are not an unbiased distribution or proof
about every possible ARC state. Random differential tests independently cover
1,000 generated states with five sequential eviction attempts each, including
atomic failure and protected/referenced blocks.

The real long-document follow-up increases prefill batch size to 4,096 tokens to
seek larger eviction batches without fabricating cache contents. It produced
calls requesting up to 250 victims. Native eviction took 159 ms
in a 175-second run; linear eviction took 18 ms in a 179-second run. Natural
output counts were 7,240 and 7,243, so these are not equal-output timing estimates.
The absolute body saving is about 142 ms; even this deliberately larger-batch
cell keeps native eviction below 0.1% of workflow wall time. All ten additional
document-state replays matched exactly. A 197-victim
sample required 34,380 native iterator visits versus 368 linear visits and was
about 23× faster in isolated replay; another 197-victim sample was about 8×
faster. The pathological CPU pattern therefore has a real serving counterpart,
but it still contributes little total time on this host and workload.

Future hardware or workloads with larger batches, much larger resident queues
or faster GPU execution may warrant another
profile; these local measurements do not reject that possibility.

[Compact accounting and replay results](eviction-serving-profile.json);
[original CPU profile](cpu-profile.json) and
[protected-prefix stress](cpu-profile-protected.json).
Raw states, calls, frozen source and offline verification are under
`output/stage5-optimization-20261105/eviction-round8/` and
`output/stage5-optimization-20261105/closeout-round9/00-live-documents/`.
The reusable collector/replayer is
[arc_serving_profile.py](../../benchmarks/research_checks/arc_serving_profile.py).
