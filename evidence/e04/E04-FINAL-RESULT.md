# E04 final result

**Status:** scientifically final negative result for incremental semantic/cost-aware
CPU admission; positive result for the policy-agnostic native store threshold.

## Execution and integrity

- Authorized device: NVIDIA GeForce RTX 5080,
  `GPU-44ceb8b5-b67a-a317-fee3-f01c9201994e`.
- Frozen implementation: revision
  `01d3eaa5a9bd90095e6f710c3c1ca07f77103f72`, tree
  `3f7264533602bdd013deaec3a600d0735d28108a`.
- Frozen runtime: revision `f3f036805c886c562d1b5397da40a898f163e93a`,
  tree `c4b6cd9a94426340256f0957215e7bbb03ff1bc9`.
- 420/420 cells passed and validated: 100 semantic/native, 160 ARC and
  160 LRU; 10,500/10,500 request records; five independent seeds and four
  workload profiles per arm. There were no failed, blocked, invalid or missing
  cells.
- All 10,500 request/preservation events had complete request, ordered-key and
  physical-block identities. Exact output-token digests agreed across all arms
  for every matched profile/seed/request group. This is the retained exact-KV
  identity and observable-output correctness proof.
- Completion, capture and cleanup all returned zero. The local archive digest
  is `15ffc92056a02fa9abb981289add2c819446b3386ee9cc583db1a94dbbf1b21e`.
  Postcheck found no owned process and no GPU-lock holder.

## Main matched result

The table reports means over the 20 semantic cells and duplicate-averaged matched
ARC/LRU cells. Percentage intervals below are paired by seed after averaging the
four workload strata; brackets are 95% t intervals across the five independent
seeds.

| Threshold-2 family | TTFT ms | ITL ms | Throughput tok/s | Makespan ms | Store MB/cell | Load MB/cell | Reused | Recomputed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Semantic/native | 216.89 | 27.16 | 57.17 | 11,246 | 2.96 | 0 | 16,598 | 49,449 |
| Matched ARC | 210.18 | 25.83 | 58.72 | 10,631 | 2.98 | 0 | 16,420 | 49,628 |
| Matched LRU | 209.55 | 25.81 | 58.78 | 10,660 | 3.02 | 0 | 16,475 | 49,573 |

Semantic threshold-2 versus matched ARC:

- TTFT +3.17% [-1.82%, +8.17%]
- ITL +4.76% [-4.27%, +13.80%]
- throughput -2.70% [-7.39%, +2.00%]
- makespan +5.12% [-6.56%, +16.80%]
- store bytes -0.77% [-6.26%, +4.72%]
- reused tokens +1.09% [-2.35%, +4.53%]
- recomputed tokens -0.36% [-1.50%, +0.78%]

Semantic threshold-2 versus matched LRU:

- TTFT +3.51% [+0.13%, +6.88%]
- ITL +4.80% [-4.23%, +13.84%]
- throughput -2.83% [-7.70%, +2.05%]
- makespan +4.80% [-7.79%, +17.39%]
- store bytes -1.89% [-7.66%, +3.88%]
- reused tokens +0.76% [-3.89%, +5.42%]
- recomputed tokens -0.25% [-1.79%, +1.30%]

Threshold-2 therefore does not establish an incremental semantic advantage over
matched ARC or LRU. Its reuse, recomputation, write and serving intervals mostly
include equality; TTFT is measurably worse than LRU by a small amount.

## Mechanism benefit versus neutral

Within the semantic/native family, threshold-2 versus neutral reduced store bytes
by 99.790% [99.777%, 99.804%], improved mean throughput by 14.93%
[14.57%, 15.30%], reduced makespan by 12.00% [11.02%, 12.97%], reduced TTFT by
22.50% [19.50%, 25.50%], and reduced ITL by 19.78% [18.79%, 20.77%]. It did not
materially change reuse (+0.47% [-4.73%, +5.67%]) or recomputation (-0.14%
[-1.84%, +1.56%]).

Matched ARC and LRU received the same approximately 99.78% write reduction
without a clear serving-latency penalty. Across the threshold cells, semantic,
ARC and LRU wrote only 129, 260 and 263 chunks respectively; there were no loads
or restores. The benefit is avoidance of speculative CPU persistence, not useful
restore or prefetch.

Pooled request percentiles for semantic threshold-2 were TTFT
96.79/2,068.01/5,200.43 ms and ITL 21.23/60.25/113.05 ms at p50/p95/p99.
The high tails are workload-mixture summaries and are not substituted for the
paired seed-level comparisons above.

## Other E04 mechanisms

Semantic historical-frequency admission was decisively worse than its matched
controls: versus ARC, TTFT +32.39% [+30.38%, +34.40%], ITL +26.99%
[+24.95%, +29.02%], throughput -14.42% [-15.66%, -13.18%], and makespan
+15.00% [+12.53%, +17.48%]. Results versus LRU were similar. It produced no
reuse or recomputation gain and did not reduce writes relative to LRU.

Frequency admission emitted 300,413 causally bound decisions across its five
arm families. Of these, 300,266 admitted and only 147 rejected. There were
241,751 compulsory fallbacks for unavailable features (80.47%), 58,500 free-slot
admissions, and only 162 utility comparisons. Native frequency alone made 61,579
decisions and rejected 15. Candidate cost was the unit-cost frequency control;
there were no asynchronous scorer events. Direct CPU scorer/admission duration
was not measured and is unavailable, not zero.

Semantic known-prefix-cap was also worse than matched ARC: TTFT +23.15%
[+19.11%, +27.19%], ITL +17.99% [+11.16%, +24.81%], throughput -10.33%
[-13.46%, -7.19%], makespan +8.13% [+0.77%, +15.49%], 21.93% fewer reused
tokens, 10.43% more recomputation, and 12.59% more writes. ARC known-prefix-cap
was the only arm with a material reuse tradeoff versus its own neutral control:
+9.26% reused tokens and -3.86% recomputation, while reducing writes 27.01%.

Thus two configurations remain nondominated for different objectives:

- threshold-2 for minimum writes and near-control serving performance;
- ARC known-prefix-cap where additional restored reuse is worth hundreds of
  times more CPU-write traffic than threshold-2.

No semantic configuration is nondominated against its matched ARC/LRU control.

## Prefetch, occupancy and cost evidence

- The wait-hint runtime was bound and 3,360 known-prefix registration events
  occurred in the causal-tool-resume stratum, but publications and terminal hint
  actions were zero.
- Speculative/prefetch jobs, submitted/completed/consumed/discarded bytes and
  asynchronous scorer events were all zero. Prefetch usefulness is therefore an
  explicit absence finding, not a successful prefetch result.
- Capacity was fixed at 256 MiB CPU cache and 640 GPU blocks. Phase-boundary
  occupied CPU bytes/GPU blocks were observed as zero, so useful-residency/TTL
  occupancy is unavailable from those samples. Peak active allocator bytes were
  3,496,378,368 for semantic and 3,493,383,168 for ARC/LRU.
- Semantic threshold-2 wrote 59,179,008 bytes total across 20 cells in 129 chunks,
  with 0.0171 aggregate measured store seconds. It restored zero bytes/chunks.
  The same cells retained 331,968 reused and 988,987 recomputed tokens.
- Missing measurements (direct scorer CPU duration, useful byte-seconds/TTL,
  and prefetch precision with an empty publication denominator) remain
  unavailable rather than being reported as zero.

## Conclusion and stopping rule

E04 is final. Five independent repetitions in every relevant workload/arm,
complete matched ARC/LRU controls, exact correctness, readable serving/transfer
costs and clean termination provide stable evidence. Additional cells are
unlikely to reverse the decisive frequency/known-prefix penalties or create a
semantic advantage in threshold-2 where decisions and preservation outcomes are
effectively matched.

Adopt native `store_threshold=2` as a configurable write-avoidance guard when
first-reuse recomputation is acceptable. Do not claim it as a semantic-policy
win, and do not enable or invest in prefetch based on E04: no hint was published
and no prefetch transfer occurred. Prefer matched ARC behind the admission guard
until a workload with real semantic/ARC disagreements and complete cost features
demonstrates incremental value.

## Retained evidence

- `captured-v4/e04-native-frequency-full-grid-repair-20260909-v4/e04-screen-result-v1.json`
- `captured-v4/e04-native-frequency-full-grid-repair-20260909-v4/completion-validation.json`
- `captured-v4/e04-native-frequency-full-grid-repair-20260909-v4/supervisor-evidence/`
- `captured-v4/execution-root.tar.gz`
- `captured-v4/execution-root.tar.gz.sha256`
- `captured-v4/postcheck.stdout`

