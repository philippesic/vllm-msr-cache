# Faster eviction did not translate into a qualified serving optimization

2026-11-10 · historical · unresolved

No new optimization qualified beyond existing wait hints. Large eviction calls sped up substantially, but eviction occupied less than 0.05% of primary/tight workflow time and only 0.09% in the larger-document diagnostic.

![Workflow throughput change (%)](chart.svg)

Three-repetition development panels. Different candidate comparisons are shown separately; higher is faster.

## Scope

144 valid development cases plus two invalid configurations. Selected timing and pressure panels used three repetitions; exact-state CPU replay examined native decisions.

## Limits

Development-selected inputs are not independent confirmation. Some mechanisms never activated, so their value remains unresolved rather than disproved.

## Decision

Keep the simple candidate. Preserve microbenchmark gains as diagnostics and require end-to-end value before importing faster eviction or inactive bridges.

## Sources

- [evidence/stage5-optimization-20261105/README.md](../sources/de155d9906f4d65afa27e5f53d86d3f580b85504accc2aa1ce086a5d0020a801.json)
- [evidence/stage5-optimization-20261105/EVICTION.md](../sources/0066306277320122470ad5897e12acdac14db95a79294def4b3911ad3fcd35e1.json)
- [evidence/stage5-optimization-20261105/EXHAUSTION.md](../sources/95bedbcb51694b53c5730549e1412f9f6633100a27324b34fae540b2adf72fc6.json)

Exact source hashes and figure inputs: [result.json](result.json).
