# Stage 6 local execution

Status: implementation and preparation in progress; no new performance claim.

The two-wave operational development probe caught a serving defect before pilot
or confirmation: CPU-only wait retention still allowed the older preemption
staging path to reserve GPU blocks. All requests, malformed-advice checks,
cancellation, recovery and cleanup completed, but native registry validation
correctly rejected the candidate case. The repair makes the existing retention-only
and no-dispatch settings cover preemption enqueue, retry and direct reservation.
Selected configuration, budgets and workloads remain unchanged. Focused CPU
checks pass 315 tests, including regression coverage; all native capability and
development comparisons must now use a fresh source snapshot. Earlier development
timings under preemption cannot establish a pure retention-only effect. See the
[finding and retained observations](preemption-dispatch-repair.json).

The corrected campaign is `development-09`. The original 180-second request
deadline expired for four queued long summaries before their first token, after
12/16 requests completed. New immutable bindings use 900 seconds per request and
3,600 seconds per case across all arms. Tasks, prompts, output caps, cache budgets
and policy are unchanged. All three native fixtures and the registry probe passed
again. Long summaries now completed 16/16 requests per arm, with no request
failures; the measured ARC load took 219.3 seconds, explaining why the old
180-second deadline was insufficient. Two final answers per arm repeated text,
exceeded the requested word limit and reached their fixed output cap. Their
[source spot audit](../../output/stage6-2080ti-20261106/development-09/analysis/quality-screen/development-summary-spot-audit-01.json)
retains these model-quality failures; no prompt or policy was tuned to remove
them. The full screen remains in progress. See the
[deadline finding](deadline-checkpoint.json). Maintained benchmark checks now pass
196 tests at that checkpoint. A later focused transfer check passes six tests,
including rejection of both model snapshot symlinks and direct blob reads during
offline verification. The final real relocation proof remains due.

The corrected campaign also passed its own real interruption and whole-block
replacement check: the first arm completed, the second was interrupted after
resource acquisition, owned cleanup passed, and resume reran both arms. See the
[replacement receipt](../../output/stage6-2080ti-20261106/development-09/analysis/recovery-proof.json).
The separate two-wave persistent-operation probe cannot count as confirmation
evidence; its retained failure triggered the repair described above.

The repaired development chat completed six requests per arm without failures or
output caps; the retained source review supports both final answers. Short QA
completed 64 requests per arm with valid JSON throughout and no caps. Each arm
matched 20/64 references exactly and contained the full normalized reference span
in 54/64 answers. A targeted source audit found correct paraphrases and inconsistent
source questions/references, so those mechanical scores are not an answer-correctness
rate. Original questions and references remain unchanged. These are development
diagnostics, not fresh confirmation quality or speed estimates.

A second clean reconstruction now matches the first for all three installed
environment identities and both external executed source identities. The source
snapshot builds an extension wheel without the original checkout. A real measured
screen was interrupted after its first arm completed; owned cleanup passed and
resume replaced the entire comparison block. The partial attempt remains sealed.
See [durability checkpoint](durability-checkpoint.json). The complete 54-configuration
development screen is queued behind fresh native checks; confirmation remains due.
The [pipeline and 5080 commands](../../docs/STAGE6_PIPELINE.md) now describe public
runtime reconstruction, frozen import and mandatory hardware qualification.

The first source review found an inappropriate investigation instruction in
ordinary chat: both arms invented a test-status claim despite running no tools.
Those outputs and explicit audit judgments are retained. Before confirmation,
the shared prompt was repaired uniformly: ordinary tasks receive no investigation
report requirement, and tool workflows may report validation only after an actual
tool result. Native checks and the development screen are rerunning from the
corrected source. This is a development prompt repair, not policy optimization.
Full quality summaries keep format failures, reference accuracy, capped outputs
and mechanical pass rates separate from source-audited usefulness and raw speed.

The isolated runtimes have now been independently rebuilt from public source and
native-wheel pins. All three rebuilt native models passed restore versus fresh
greedy-reference checks. Rebuilt Continuum and upstream each completed 64/64 real
workflow requests with actual policy activation (44 Continuum pins; 48 upstream
priority applications and 531 selections). These are development capability
checks, not paired speed results. See [reconstruction checkpoint](runtime-reconstruction-checkpoint.json).

The dependency recipe now includes the full CUDA component closure. Runtime
fingerprints ignore checkout-local package metadata; installed environments are
identified independently of the caller's directory. Independent reconstruction
matched the external native binaries; upstream's generated version metadata is
explicitly different from the historical environment. Historical environments
and failed development attempts remain intact.

The short-QA cohort uses article-disjoint SQuAD v1.1 paragraphs/questions/references,
with an authored JSON answer instruction. It replaces the insufficient pool of
eligible LongBench QA examples before any QA serving. LongBench still supplies
intact long-summary inputs; filtered WildChat supplies chat starts. Real repository
tools execute whole test classes with their full tracked source dependency closure.
All 48 development/confirmation/repair classes passed: 334 tests and 7,979 subtests.

The maintained benchmark checks pass 190 tests, native CPU checks 100, focused
registry checks 35 and portable checks 31 at this checkpoint. Recovery tests reap
timed-out, crashed and detached owned children while preserving pre-existing work.
Transfer tests retain both cohorts and complete tool imports after relocation and
removal of original paths. The complete GPU matrix, bounded pilot/confirmation,
source audit, operational runs and real interruption/relocation proof remain due.

Development envelope rule, recorded before the complete matrix screen: B06 must
show ARC returning requests whose CPU prefixes were evicted. B07 must at least
halve that count by changing capacity on identical tasks. This tests the pressure
contrast; candidate speed does not select the envelope. Native backing and host
paging observations remain mandatory.

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
