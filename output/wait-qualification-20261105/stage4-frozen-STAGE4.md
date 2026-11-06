# Stage 4: available compatibility and reliability cells (pending)

Preparation is separate from qualification. Every local cell uses the patched
pinned 0.26 runtime on the RTX 2080 Ti / WSL2 host. See
[the integration scope](../../docs/WAIT_HINT_SUPPORT.md) and the frozen
[protocol](PROTOCOL.md). No cross-product or other-hardware claim follows.

| Selected cell | Required evidence |
| --- | --- |
| 7B AWQ, eager | Three pressure-probe waves, actual CPU restores, fresh-salt full recomputation, equal greedy output IDs |
| 1.5B unquantized Qwen2.5 | Same functional checks with byte-converted cache; separate three paired real-workflow batches at full and predeclared half-deployment byte budgets |
| OPT-125M, MHA | Raw-token functional pressure check; same-byte conversion rounded down; no chat/performance claim |
| Minimal bounded audit-off profile | Pressure correctness plus direct paired real-workflow performance, after-timing cancellation/recovery |
| Abandoned advice | One initial hint followed by distinct-prefix churn; late diagnostic return must recompute fully |
| 7B AWQ decode graphs | Three paired real-workflow batches, explicit graph-path receipts |
| Minimal client | Standalone documented two-turn client with actual SHA-256 subprocess, positive CPU-ready refresh, owned cleanup |
| Isolated install/reproduction | Follow pinned instructions in isolation; keep author-run and independent reproduction distinct |
| Newer available runtime | Read-only API compatibility probe; do not silently adapt published baselines |
| Other GPU/native Linux | Untested unless actual access and execution become available |

The base functional geometry uses 84 MiB GPU KV and 168 MiB CPU KV; additional
models convert by actual bytes per block and round down. OPT loses a fractional
block and keeps its supported 1,024-token probe context rather than borrowing the
8,192-token chat workload. Probe texts are actual public document excerpts. Each
fresh reference must recompute its entire prompt with zero external hits;
matching a baseline that shares the same cache path is insufficient. Exact greedy
IDs test observed generation, not direct KV byte equality.

Malformed/future advice, namespace mismatch, duplicate idempotence and stale
causal timestamps are explicit checks. Operational-tier probes also verify a
real disconnect's native abort receipt, request drain and recovery. Every fresh
server exercises startup/restart; all owned processes must be reaped. Functional
activation cannot establish retained performance, and neutral all-fit workloads
must remain visible.

The finite Stage 3 churn exposure and these correctness probes are bounded tests.
Neither establishes unlimited session handling or a production failure rate.
The minimal profile caps new identities at 4,096 and may refuse further advice
while continuing ordinary serving. A selected default has no performance claim
until its direct measurements and answer checks finish.

Raw specs and all attempts are retained in
`output/wait-qualification-20261105/`; executed source, actual allocations,
sampling, model/package identity and path receipts attach to each measured cell.
