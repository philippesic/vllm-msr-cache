# Isolated semantic cache factorial, November 4

User authorization: experimental implementation and real serving, bounded hill
climb; no deployment. Preserve previous qualification and paused AWQ work.

Recovered design: `semantic_offload/config.py`, `policy.py`, `worker.py`,
`connector.py`, `docs/CONFIGURATION.md`, and retained CPU-hint configuration.
The extant base is mean semantic relevance for CPU eviction, rolling rank
weights, exact prefix identity for reuse, and causal known-prefix prefetch with
owned staging and native adoption. No KV SVD implementation or original SVD
parameter specification was found in current records. The new codec is an
explicit experimental extension rather than a recovered historical algorithm.
It compresses both K and V page values, not predictor features. Attention reads
reconstructed FP16 pages. Exact prefix identity is preserved, but values are
lossy; prior exact-KV qualification does not transfer.

Six arms: instrumented stock ARC, GPU-gated CPU-recency wait-hint ARC, semantic
base with both additions off, base + KV SVD, base + semantic prefetch, both.
All use identical model, GPU capacity, concurrency and nominal CPU budget. The
first codec keeps the same slot count and reduces allocated host storage; a
capacity-expansion follow-up is warranted if retention misses dominate. Rank 8
retains 53.125% of each page's bytes as FP16 factors. Codec transfers initially
synchronize to establish a conservative ownership fence. Timings include codec,
residual diagnostics and synchronization, not only PCIe copies.

Initial fixture: Qwen2.5-1.5B FP16 on RTX 2080 Ti; six distinct warehouse retrieval
sessions, three live turns and scripted waits, 128 GPU blocks, 64 MiB nominal CPU
budget. Assertions test recovery code and badge color. This establishes narrow
retrieval fidelity only. Raw prompts, output tokens, SSE and resource samples
are retained through the existing serving-suite run_case API. The supervisor
reserves GPU ownership across startup and idle gaps; no server engine methods
are replaced. Failed setup attempts remain in output with their errors.

Bound: diagnose failures/inactivity and repair before inference. Tune rank,
observation stride, staging budget and capacity only where cost/activity supports
a mechanism. Repeat promising variants in reversed order and use public corpus
where available. Stop if gains depend on answer damage, codec overhead exceeds
credible avoided work, or realistic activation destroys the apparent benefit.
No single inactive/failed screen is a verdict. Report outstanding coverage.

Raw evidence: `output/semantic-svd-serving-20261104/`. Preserve each source
identity, coordinator version, configuration and result before editing executed
code. Do not edit serving dependencies while a campaign runs.
