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

- [evidence/semantic-svd-serving-20261104/README.md](../sources/0fa7deadbd1964345364cb99e4435497dc4aacaf48eed825319fe11b0ae1736d.json)
- [evidence/semantic-svd-serving-20261104/results.json](../sources/8c7a76ce273ef3bbd70a042cc45ebdb8a07dad38f2fbae010c59b1048a129319.json)

Exact source hashes and figure inputs: [result.json](result.json).
