# Stage 6 local execution

Status: implementation and preparation in progress; no new performance claim.

Decision: establish correctness, workload-specific candidate/ARC/Continuum/upstream
comparisons and a reproducible RTX 5080 handoff under the
[locked execution scope](../../docs/STAGE6_EXECUTION_PLAN.md).

The candidate is the unchanged Stage 5 CPU-only wait-retention behavior.
The selected models and 54-configuration matrix remain mandatory. Preparation,
development, confirmation and diagnostics remain separate. Raw attempts are
retained under `output/stage6-2080ti-20261106/`; reusable assets live in a separate
local cache. Runtime, model and corpus identities will be recorded before serving.

Initial host inspection: RTX 2080 Ti idle with no compute processes, about 823 GiB
free disk, 15 GiB total host RAM and 4 GiB swap. This is preparation evidence,
not qualification. Existing environments and historical outputs are preserved.

## First implementation checkpoint

The catalog encodes all 54 required configurations. Initial implementation covers
strict campaign/host contracts, durable hash-chained attempts, artifact seals,
writer exclusion, interrupted-attempt recovery, comparison order, bounded pilot
precision planning, offline verification and transfer interfaces. The graph
scheduler demonstrates actual fan-out/dependency result propagation and sibling
cancellation in CPU tests. These components are not a completed campaign pipeline;
serving adapters, phase completion gates and portable transfer integration still
require further implementation and native validation.

Initial checks: 169 maintained benchmark tests passed through the patched runtime,
31 portable checks passed, and the runtime import check passed. A later focused
set including graph checks passed 24 tests. These counts describe their respective
source checkpoints, not final qualification.

The first Qwen3-8B-AWQ smoke completed 8/8 requests with matching repeated greedy
outputs and verified cleanup. Actual FP16/Marlin execution and pinned CPU backing
were observed. CPU restoration was inactive; the result remains **inconclusive**
for the required restore activity. It is a synthetic functional fixture and must
not enter Stage 6 performance comparisons. [Raw report](../../output/stage6-2080ti-20261106/qwen3-smoke-01/REPORT.md).
