# The suspected pending-prefetch guard did not delay the observed requests

2026-11-10 · historical · functional

Direct observations found zero returns from the semantic pending/in-flight early-return guard across 975 calls. This particular guard did not explain the two measured slowdowns; native transfer deferrals were separate.

## Scope

Two active-scoring/prefetch executions at 1,024 and 2,048 GPU blocks, 144 strict request proofs. The smaller case had eight preemptions, four pending binds and three native transfer deferrals.

## Limits

No in-flight binding occurred in the new runs, though one existed in historical data. Retrospective bind-to-ready intervals are potential bounds, not measured guard waits. Instrumented timings do not isolate causal throughput loss.

## Decision

Retain the falsified explanation with its exposure limits. Do not infer scheduler stalls from binding state or treat unobserved cases as measured zeros.

## Sources

- [output/demand-prefetch-stall-20261102/REPORT.md](../sources/9b8f4b6f84a85f65510287ecc5aa4bff5ff510271c5e1f1f60c3e35a769546cf.json)

Exact source hashes and figure inputs: [result.json](result.json).
