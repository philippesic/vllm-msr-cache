# Project history

This is the compact timeline retained while the project is paused. It records
what was tried and what was learned; it is not a release claim or a work queue.

## July 2026 — prototype and measurement foundation

- Built the first exact KV offload/eviction/prefetch extension and a latency
  benchmark harness.
- Added selectable semantic scoring, partial-splice correctness checks, mixed
  workloads, cached candidate tensors, and vectorized query scoring.
- Early profiling established that semantic bookkeeping can dominate latency if
  summaries, candidate stacks, or debug paths are rebuilt on the hot path.

## August 2026 — lifecycle, policy, and preservation studies

- Extended lifecycle accounting for load/store traffic, preemption, exact key
  identity, and preservation outcomes.
- Compared LRU, ARC, semantic-mean, and semantic-minmax under pressured Chat/RAG
  workloads. ARC remained the strongest serving baseline; semantic policies were
  slower without a preservation advantage in that run.
- Tested chain-aware, prefix-aware, and future-reuse ideas. Future-reuse labeling
  was kept offline and leakage-safe; it did not become a serving feature.
- Built an exact MTP offload round-trip validator and cache-affinity scheduling
  experiments. These remained bounded research paths.

## Early September 2026 — integrated MSR and prefetch work

- Repaired the real vLLM execution path, wait-aware staging, transfer attribution,
  block-table provenance, worker fences, and failure cleanup.
- Evaluated nine eviction/prefetch improvements, including async scoring,
  admission controls, prefetch protection, scheduling, and accounting changes.
- Rejected the correction-cache oracle after its matched-error cold-byte gate
  failed. Related approximate-KV work was parked rather than presented as exact
  cache reuse.

## September 9 — E04 admission campaign

- Completed 420/420 RTX 5080 cells and 10,500/10,500 exact request records across
  five seeds and four workload profiles.
- Native `store_threshold=2` reduced writes by about 99.79% and improved the
  semantic arm versus its neutral configuration: TTFT −22.50%, ITL −19.78%,
  throughput +14.93%, and makespan −12.00%.
- Matched ARC remained ahead at threshold 2 by point estimate: semantic TTFT
  +3.17%, ITL +4.76%, throughput −2.70%, and makespan +5.12%.
- No restore or prefetch occurred in these cells. The result supports a generic
  admission guard, not an MSR/prefetch provider benefit.

## September 10–11 — combined MSR and semantic-prefetch campaign

- Progressed from one-block activation diagnostics to exact multi-source causal
  selection, full transfer/staging/bind/install/fence/consumption attribution,
  pressure profiles, horizons, burst/recovery, three reference seeds, and a
  genuinely disjoint input.
- The final accepted set contained 40 valid cells and 1,944/1,944 successful
  exact requests. A broader historical execution ledger retained 69 valid cells
  and 3,980 valid request rows.
- On three repeated reference traces, MSR+semantic executed 120,174 prompt tokens
  versus 119,918 for ARC/LRU/FIFO controls in every trace (+0.2135%). It also used
  3.56% more instrumented model-runner time and 12.40% more H2D traffic than ARC.
- On one disjoint trace, semantic prefetch selected and consumed the returning
  source, but executed work tied the controls. Against ARC, model-runner time was
  +4.94%, median TTFT +6.23%, and throughput −6.37%.
- The campaign therefore closed as a well-supported negative for the tested
  configurations. It does not establish universal MSR inferiority.

## Pause point — September 2026

The code, exactness work, negative results, and compact evidence record are worth
preserving. A production release remains open because no repeatable provider
benefit has been demonstrated against matched ARC/LRU. Future work should start
from the ranked backlog, use independent inputs, and begin with a small exact
matched comparison rather than repeating the full historical campaign.
