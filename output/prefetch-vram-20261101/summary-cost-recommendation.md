# Summary construction: measured cost and bounded next steps

Read-only investigation, 2026-11-01. No source changes, imports, tests, GPU work, or new background processes. Current confirmation and quality runs remain controlled by the root/capture agents.

## Recommendation

Do not add source-finalization jobs or permanently skip undeclared requests' summaries on the present evidence. First finish the balanced short repetition. The existing data support a small, exact gather optimization if another candidate is needed; they do not establish that source-summary lifecycle changes will recover the short outlier.

The lowest-risk candidate is to replace per-block tensor indexing plus FP16 stacking in `worker._build_durable_summaries_for_job` with an ordered, consecutive-run gather. This keeps every raw store, summary, publication, version check, and audit. It removes Python tensor views and potentially an intermediate FP16 allocation; it does not remove the FP32 reduction. Its benefit is unmeasured and probably modest. Freeze and measure it separately only after current confirmation, preferably with prefix-path component timings.

## Current short workload evidence

The current seed-1103 candidate took 25.393912 s versus large ARC 24.748708 s and matched-budget ARC 25.046073 s. Reaching 99% of their throughputs would require reducing candidate elapsed time by 395.216 ms and 94.848 ms respectively (fixed token count; target elapsed = ARC elapsed / 0.99).

All three candidate seeds perform the same extra store work, although their elapsed-time residuals differ substantially:

| Seed | Candidate elapsed s | Large ARC elapsed s | Small ARC elapsed s | Candidate CPU s | Large ARC CPU s |
|---|---:|---:|---:|---:|---:|
| 1103 | 25.393912 | 24.748708 | 25.046073 | 27.10 | 26.31 |
| 2207 | 25.043068 | 24.856240 | 25.185072 | 26.42 | 26.13 |
| 3301 | 25.088302 | 24.987900 | 25.212701 | 26.46 | 26.42 |

Each candidate has 66 prepare-store jobs and 1,770 accepted store keys, versus 36 jobs and 1,746 keys in either ARC control. Each transfers 811,991,040 D2H bytes versus ARC's 800,980,992: an extra 10.5 MiB. Each has 33 query captures and consumes 582 native prefetched blocks. Candidate and large ARC have zero demand H2D bytes; small ARC has 264,241,152 demand H2D bytes. Identical extra job/key/byte counts do not explain why seed 1103 is much farther behind. This is a reason to repeat the pair, not evidence that fixed work is free.

Sources: the `summary.json` files under [short candidate 1103](runs/confirm-short-r0-candidate-competing-1103-semantic/summary.json), [large ARC 1103](runs/confirm-short-r0-arc_full-competing-1103-arc/summary.json), [small ARC 1103](runs/confirm-short-r0-arc_small-competing-1103-arc/summary.json), and the corresponding `2207` and `3301` run directories.

Passive manager wrappers isolate only a small part of that fixed work. For seed 1103, candidate prepare+complete-store total 61.288 ms host / 60.863 ms thread CPU versus large ARC 34.332 / 34.103 ms: about 26.955 ms host / 26.760 ms CPU extra. These are inclusive manager-method measurements, not total DMA/summary/audit cost. They cannot explain the 395 ms recovery requirement alone.

Sources: [candidate worker](runs/confirm-short-r0-candidate-competing-1103-semantic/worker-13099.json), [large ARC worker](runs/confirm-short-r0-arc_full-competing-1103-arc/worker-12947.json). Candidate seed 3301 similarly records 29.218 ms prepare and 30.399 ms complete across 66 jobs in [its worker record](runs/confirm-short-r0-candidate-competing-3301-semantic/worker-14392.json).

## Existing component profiles: useful limits, not a current attribution

The earlier same-host long-workload continuous-dense diagnostic recorded:

| Component | Calls | Host ms | Interpretation |
|---|---:|---:|---|
| summary.build_job | 114 | 159.542 | Includes host launch/gather/reduction submission; 156.284 ms thread CPU |
| transfer.submit_store | 114 | 414.164 | Includes summary work; do not add the preceding row |
| transfer.poll | 995 | 14.871 | Host polling |
| metadata.bind_worker | 995 | 42.011 | Per-step worker binding |
| metadata.build_worker | 995 | 2,152.625 | Old dense rank/EMA transport; already removed by prefix scalar path |
| scheduler.build_metadata | 995 | 1,568.025 | Inclusive scheduler work |
| scheduler.consume_worker | 1,992 | 5,388.589 | Old dense scorer updates; already removed |
| logging.preservation | 18 | 593.452 | Inclusive audit/logging path |
| logging.json_encode | 359 | 532.970 | Overlaps higher-level logging measurements |

Only eight sampled summary GPU intervals were recorded (1.038 ms combined), so that figure is not total summary GPU time. Logging emitted 58.7 MB. Dense scorer readback timings in this diagnostic also must not be attributed to the current prefix scorer. Source: [long diagnostic components](runs/diagnostic-competing_long_dram-1103-semantic/components/components-96062.json).

An older, differently configured multi-workload semantic diagnostic had 451.058 ms summary host cost across 115 calls and 648.066 ms submit-store cost, versus ARC's 0.632 ms no-op summary and 180.284 ms submit-store across 109 calls. It establishes that old full-field summary construction could matter, not that the present mean-only prefix candidate spends 451 ms there. Sources: [semantic components](../scoring-overhead-20261101/runs/diagnostic-multi-1103-semantic/components/components-56357.json), [ARC components](../scoring-overhead-20261101/runs/diagnostic-multi-1103-arc/components/components-56990.json).

Thus there is no current-prefix component measurement showing that eliminating peer summaries could recover the outlier. Historical 159 ms total summary host work is smaller than 395 ms, but the workloads/configurations differ, so this is an order-of-magnitude comparison rather than a formal bound.

## Why permanent undeclared-source skipping is unsafe

`connector._register_observed_prefix` accepts only explicit `observed_prefix_id` and boundary declarations, validates native key binding and namespace, and runs at request finish. The competing driver intentionally gives peers no declaration. That is causal evidence that this request is not presently a nominated source; it is not evidence that its cached native keys can never belong to a later declared source.

A later declared request can reuse those exact cached CPU keys without a new raw store. If the earlier store permanently omitted summaries, the later source then has no complete numerical feature set. Existing prefix scoring deliberately refuses incomplete key coverage; silently ranking it using fewer keys would change the metric, and fabricating block scores is invalid. Re-stores must also retire old publications, so skipping a new reduction cannot leave an old summary accidentally valid.

Exact backfill would require new ownership/readiness work: retain or reacquire the precise GPU source layout, own it through summary DMA/reduction fences, bind it to current generations/publications, acknowledge completion before nomination, and handle eviction, abort, overwrite, and later declaration. That lifecycle is substantially broader than a conditional around a reduction. The estimated benefit does not justify it now.

Relevant seams: [observed source registration](../../semantic_offload/connector.py:1921), [store layout construction](../../semantic_offload/connector.py:8132), [summary building](../../semantic_offload/worker.py:1962), [store submission and fencing](../../semantic_offload/worker.py:2040), [eviction invalidation](../../semantic_offload/worker.py:908).

## Exact smaller candidates and their limits

1. **Ordered contiguous-run gather.** Current summary gathering creates one tensor view per source block, stacks FP16, then casts to FP32. Group the selected block IDs into consecutive basic-slice runs while preserving every order, duplicate, shape group, and bounds check. For one run, make an owned contiguous FP32 copy directly; for multiple runs, concatenate slices in the existing order before conversion. The reduction must receive identical ordered, contiguous FP32 data, including FP32 source inputs; keep existing per-key grouping/publication/failure cleanup. Test negative indices, nonconsecutive/duplicate IDs, mixed block counts, last-write publication order, ownership after source overwrite, and failure fences. This saves dispatch/allocation work without a new lifecycle.

2. **Memoize immutable audit rendering, if profiling confirms it.** Native pool snapshots repeatedly SHA256 `repr(block_hash)` and CPU snapshots sort/render every resident key. Native block hashes are immutable bytes; a bounded content-keyed digest/key-render cache could avoid repeated serialization while still reading all current physical IDs, readiness, refcounts, counters, and membership. Cache neither whole snapshots nor by block ID alone. A sorted-membership cache additionally needs a reliable membership generation; `_score_residency_sequence` stamps stores but is not by itself a complete membership revision because removals also matter. Apply any observer optimization equally to controls, preserve audit output semantics, re-freeze and repeat. This is audit CPU reduction, not evidence of better scoring itself.

3. **Existing centroid caching already addresses repeat math.** Prefix centroids are already owned and cached by exact key/publication signature, bounded at 16 entries / 1 MiB. They cannot safely be reused merely by request ID after publication changes. Another durable centroid cache has no measured repeated work to remove here.

4. **Do not casually defer decode stores.** Native store jobs represent newly computed chunk keys, not duplicate metadata that can simply be memoized. Storing only at request finish could reduce fixed jobs but changes CPU availability, pressure/preemption recovery, and transfer timing. It requires pressure-aware flushing and separate controls, even if eventual bytes are identical. It is not a drop-in correctness-neutral optimization.

No implementation is recommended during the frozen confirmation. If the repeated short result still misses the target, first capture an equally instrumented prefix-path candidate/ARC component pair; then try the gather-only change if summary dispatch is material. Keep source-finalization as an explicitly unimplemented alternative, and retain the unsuccessful short pair rather than selecting favorable runs.
