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


## Native model and live-adapter checkpoint

All three selected models completed 11/11 deterministic native requests, actual
CPU restore and a full fresh-reference computation with equal greedy token IDs.
Duplicate, wrong-namespace, future-information and stale-clock advice checks
passed. Cleanup was verified. See the compact
[functional receipts](model-functional-checkpoint.json); raw probe files retain
scoped metrics and exact tokens. These are functional fixtures, not speed results.

Ministral uses the publisher BF16 artifact with FP16 execution, the supported
text-only flag, and explicit HF/safetensors loading. An isolated copy of the local
runtime uses Transformers 5.5.3, huggingface-hub 1.33.0 and tokenizers 0.22.2 to
match the pinned runtime's model implementation. The original environment remains
unchanged. Qwen3-4B also passed in FP16. Its measured KV allowance in the one-sequence
probe was about 1.57 GiB; proposed multi-sequence envelopes are still development
settings, not qualified capacities.

Pinned public sources and disjoint development/confirmation/repair inputs are
prepared. Repository tools execute full test classes even when the prompt shows
explicit complete-method excerpts. Document workflows execute real search,
metadata parsing and validation. Team graphs propagate actual specialist results;
mixed traffic has a fixed interleaved 50/50 offered-unit assignment. Live streams,
causal hint publication, bounded tool subprocesses and cancellation have focused
CPU integration coverage. The latest focused set passed 130 tests.

Initial development blocks are retained as failed attempts. One exposed an
oversized historical test-class prompt; another caught a changed token prefix
from decoded-text re-templating. The adapter now preserves observed prompt/output
IDs and compiles only the new native turn suffix. No confirmation result has been
used to tune the policy, which remains the selected Stage 5 configuration.
