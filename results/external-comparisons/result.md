# Single-pass external comparisons did not establish a winner

2026-11-10 · historical · unresolved

The five-system exploration and corrected four-system comparison exposed configuration differences and similar wait/upstream batch times. Neither establishes a repeatable system ranking.

![Workflow batch time (seconds)](chart.svg)

Corrected four-system comparison only. Lower is faster; one run per system, no uncertainty intervals.

## Scope

Corrected four-system study: one run per system, sixteen workflows, 64 responses and 48 tools each, Qwen2.5-7B-AWQ on RTX 2080 Ti.

## Limits

Runtimes, chunking, paging, output lengths and instrumentation differ. Continuum decode-publication correction was predeclared; the original false row remains diagnostic. Strict output-format compliance mostly failed.

## Decision

Keep published-system comparisons and their configuration corrections; do not treat token rate as equal-work throughput or label parity as quality equivalence.

## Sources

- [evidence/fair-four-20261104/README.md](../sources/ad2c579a666970958e932f2d4f4e9a487cae8b0ac25853abef3c8596f05daf74.json)
- [evidence/fair-four-20261104/FAIRNESS.md](../sources/166ab5af84086633b621be3721f099d762cad2a1dc9cced48099ef252b6eda51.json)
- [evidence/five-way-20261104/README.md](../sources/0b4becee3ce003905b0465a76f065d3d8b61e8e28e76602a34db9df0aeea85fe.json)

Exact source hashes and figure inputs: [result.json](result.json).
