# Admission thresholds helped; semantic admission did not beat matched caches

2026-11-10 · historical · negative

The policy-agnostic store threshold sharply reduced writes. Adding semantic/cost-aware admission did not establish an incremental serving benefit over matched ARC or LRU.

![Throughput change (%)](chart.svg)

Reported paired seed-level 95% intervals. Different comparisons answer different questions; higher is faster.

## Scope

420 valid cells, 10,500 requests, five seeds and four workload profiles on RTX 5080. Matched output-token digests and block identities agreed.

## Limits

No loads, restores or prefetches occurred. Threshold effects cannot be interpreted as a retention or prefetch win; results belong to the earlier implementation.

## Decision

Preserve threshold admission as a separate hypothesis. Do not revive semantic admission solely from its improvement over an unguarded baseline.

## Sources

- [evidence/e04/E04-FINAL-RESULT.md](../sources/ef2be4894e8e03aa1628a9b6c2026308c334e60b240bb5a4a1b4789f6bb2dcd6.json)
- [evidence/september-2026.md](../sources/0a832a8589b351d8203936f1abc453be2f35b6f043716039caedad3c5be1b982.json)

Exact source hashes and figure inputs: [result.json](result.json).
