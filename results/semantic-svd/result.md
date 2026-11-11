# Semantic and SVD variants added cost without established serving value

2026-11-10 · historical · negative

The measured semantic variants were slower than wait-hint ARC. Lossy SVD changed outputs: rank 4 failed 36 of 72 authored QA checks, while rank 8 passing the narrow fixture did not establish general equivalence.

## Scope

Fifty-four serving cases, 972 responses on RTX 2080 Ti and Qwen2.5-1.5B.

## Limits

Scripted public waits provided no advance prefetch window. The study measures cost and narrow quality effects; it cannot reject every semantic or prefetch opportunity.

## Decision

Keep exact KV as the current behavior and retain the failed/inconclusive alternatives as evidence, without importing their implementation.

## Sources

- [evidence/semantic-svd-serving-20261104/README.md](https://github.com/philippesic/vllm-msr-cache/blob/1319f09bafac46bab89df465dd433c67441f4b26/evidence/semantic-svd-serving-20261104/README.md)
- [evidence/semantic-svd-serving-20261104/results.json](https://github.com/philippesic/vllm-msr-cache/blob/1319f09bafac46bab89df465dd433c67441f4b26/evidence/semantic-svd-serving-20261104/results.json)

Exact source hashes and figure inputs: [result.json](result.json).
