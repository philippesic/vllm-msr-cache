# Lower asynchronous scoring CPU cost did not establish ARC parity

2026-11-10 · historical · mixed

Cooperative readiness waits reduced measured readiness-plus-executor CPU from 17.885 to 5.221 ms per job, about 71%. The final serving pairs still ran 2.71% and 2.27% below ARC with roughly 12% worse median first-token latency.

## Scope

Qwen3-4B BF16 eager, 640 GPU blocks and 448 MiB CPU KV. Separate continuous-scoring diagnostics exercised the wait path; final ARC/candidate/candidate/ARC product runs used the simulated-agent workload.

## Limits

Final product runs submitted zero asynchronous score jobs and no constrained ranking, so their regression cannot be blamed on the new wait code. Earlier favorable revisions did not establish dependable parity. Diagnostic histories/coverage differ and readback is already included in executor cost.

## Decision

Retain lifecycle safeguards and the distinction between less CPU spinning and less GPU work. Do not promote asynchronous scoring on a microbenchmark alone.

## Sources

- [output/scoring-audit/REPORT.md](../sources/ed295773e7fb30b7724d4041cb8bc9483400ae86385a9ab60b895bcbaac8d10e.json)
- [output/scoring-audit/astra-acceptance.md](../sources/40d04469fbc0bc94e07e1859755556080d6bcbbe8d792a42e5a423c7de78ea99.json)

Exact source hashes and figure inputs: [result.json](result.json).
