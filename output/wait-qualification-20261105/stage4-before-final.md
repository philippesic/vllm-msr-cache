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
| Long-prefix trusted references | Base 7B eager/graphs and 1.5B eager, up to 6,000-token real-document targets, 8,192 context, actual CPU restore versus fresh recomputation |
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

Before any Stage 4 measurement, three additional long-prefix functional cells
were selected to cover actual workflow-length positions and graph execution.
They use 448 MiB GPU KV (512 base-model blocks / 1,024 smaller-model blocks) and
2,688 MiB CPU KV, with up to 6,000-token target excerpts and 8,192 context.
Real document churn must evict the GPU copy, and the cached return must actually
restore CPU KV before its greedy output is compared with fully fresh computation.
These are correctness diagnostics, not added performance repetitions. The byte
choice supports one full base-model context and is not a search for a speed win.
Frozen additions are in `stage4-premeasurement-addendum.json`; original short
pressure probes and all selected performance cells remain unchanged.

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

## Evidence collection repairs prepared before measurement

Stage3 source inspection found that the selected V2 model runner does not
construct or propagate the graph statistics emitted by the older runner. The
native `--cudagraph-metrics` flag alone is therefore insufficient. An opt-in
V2 statistics repair is prepared: record the actual dispatched mode after a real
forward, exclude dummy/capture and profiling work, and carry the record through
the existing output, scheduler and logger. It changes no graph dispatch,
allocation, kernel or retention policy. The ignored prepared patch passes a
read-only application check; its five CPU tests have
passed, while actual GPU graph receipts are still pending. The repair was applied after Stage3 completed and before any Stage4
measurement, and is retained as an additive native patch. Actual GPU receipts remain required.

The prepared six-file evidence patch additionally enables that flag only for
graph cells, records actual CPU backing tensors at startup when benchmark
telemetry is enabled, and strengthens protocol probes. Pinning must be observed
with `Tensor.is_pinned()` on the allocated buffers and matching load/store
backing, with byte accounting independently checked against the rounded budget.
Requested environment flags alone do not prove this property. Stage2 and Stage3 published-variant campaigns already have independent `native_worker_geometry` / LMCache buffer observations and pre-timing byte/pinning checks; Stage1 has no such observer receipt. The selected Stage4 serving probes also retain the existing observer. The new emitter additionally checks matching load/store backing and covers the standalone example supervisor, which explicitly clears the observer path. It does not retroactively add a tensor observation to Stage1.
Eleven synthetic allocation-report checks and five synthetic graph-report checks
passed; they establish parser behavior, not GPU execution.

Namespace/stale rejections must have a successful protocol response, the exact
expected rejection reason and matching identity. Generic server errors, unrelated
bad-request responses and a changed duplicate acknowledgment must fail. The
future-information probe requires the pinned API's specific validation error.
These are stricter evidence assertions selected before any Stage4 result.
Both guarded repairs are now applied. The focused60, benchmark131 and qualification/V2-telemetry28 tests passed through the pinned launcher. The applied source, test logs and all13selected spec hashes are frozen in `stage4-source-and-tests-freeze.json` before any Stage4 GPU measurement. The unchanged selected GPU cells still require actual execution.

## Read-only runtime boundary executed at the mixed/soak quiet boundary

The actual pinned0.26.1rc1.dev75+gdc1be7903 overlay imports the extension's manager,
spec, worker and connector, exposes all required native publication/allocation
methods and constructs a four-entry ARC manager. The separately pinned newer
published retention runtime (`0.1.dev1+g7966f717d.precompiled` distribution identity)
imports its own native classes but cannot import the current extension because
`BlockStatus` was renamed/replaced in its chunk-based API. Its required native
completed-prefix publication and wait-hint scheduler hooks are absent, and the
CPU allocation hooks changed. The four-entry constructor attempt fails at import
with that actual error. This is a measured unsupported-version boundary, not a
GPU port or mechanism rejection. No published source/environment was modified.
[Positive receipt](../../output/wait-qualification-20261105/runtime-pinned-interface.json),
[negative receipt](../../output/wait-qualification-20261105/runtime-newer-interface.json),
and retained static-interface/patch-dry-run evidence explain the scope.

The actual pinned native graph logger also passed a CPU aggregation, table-parser
and disjoint-interval-reset check. This supplements synthetic fixtures; it still
does not establish GPU graph dispatch, which remains required in Stage4.

## First functional diagnostic retained and evidence adapter repaired

The initial `runtime-probe` attempt is sealed/verified and failed after a complete
ARC control plus one wait wave. The wait wave actually restored29360128bytes
(native completed-transfer observer), had512external cached tokens, matched all24
greedy output IDs against its fully recomputed fresh reference, and passed
namespace/stale/duplicate checks. Two probe assertions were wrong: vLLM wraps
the expected forbidden-field message in its canonical ErrorResponse rather than
FastAPI's bare `detail`, and eager exports directional byte counters while the
probe selected only the modern flat series. The server correctly returned400
with the exact expected message; the absent flat sample was not zero transfer.
Both allocated-buffer receipts passed, and owned processes were reaped.

The v4 probe accepts only the exact400 validation reason in the two known
envelopes, uses the maintained canonical modern/eager parser and monotonic diff
(no missing/reset-as-zero), and retains every settled request's before/after raw
metric text plus source provenance. Nine new regression cases cover these
formats; all37 qualification/V2 cases pass. This changes no policy, corpus,
budgets, conditions or expected restore/fresh-computation/equality checks.
`stage4-probe-repair-freeze.json` records the source/tests and identical selected
spec before the complete fresh `runtime-probe-repaired` control/candidate run.
`compatibility-replacements.json` explicitly links both attempts; the original
failure remains visible and has no eligible timing comparison. Other selected
Stage4 cells have not run or influenced this repair.

The complete repaired short7B functional cell passed all six waves and after-timing
operational checks. Every candidate return really restored CPU KV; every fresh
reference computed its entire prompt with zero external hits; all greedy IDs
matched. Actual pinned/shared-backing byte receipts pass for both arms. The raw
diagnostic span is0.735%longer with wait (33.735s versus33.489s), retained as an
ARC-loss alarm; candidate additionally executes four protocol-negative/duplicate
RPCs per wave, so this is not a matched application-performance comparison or
isolated retention overhead measurement. [Complete decision and native scoped deltas](../../output/wait-qualification-20261105/runtime-probe-repaired-decision.json).

The unquantized Qwen2.5-1.5B short functional cell also passes all six waves,
actual pinned/shared-backing geometry, CPU restores, fresh full recomputation,
exact greedy generation and after-timing abort/drain/recovery. Its raw diagnostic
span is3.769%longer with wait, retained in the alarm record; extra candidate
protocol RPC checks prevent a matched application-performance interpretation.
The separately selected real-workflow full/half-budget cells remain required,
with no setting chosen from these diagnostic timings. [Functional decision](../../output/wait-qualification-20261105/runtime-probe-qwen15-functional-decision.json).

The bounded audit-off profile passes all six short functional waves, actual
pinned/shared CPU backing, CPU restores/fresh full computation, exact greedy
generation and native abort/drain/recovery. Its tiny diagnostic timing favors
wait by1.48%, with no performance conclusion from one unmatched-protocol pair.
[Functional decision](../../output/wait-qualification-20261105/runtime-probe-minimal-functional-decision.json).

The first OPT125M control failed before any requests because the preselected local
snapshot contained only config.json. The same pinned public revision was completed
with weights/tokenizer files; configuration hash and entire selected specification
remain unchanged. The zero-request startup failure is sealed/verified and retained;
`opt125-input-repair.json` / `opt125-download-receipt.json` explain the complete
fresh control/candidate retry, linked explicitly in compatibility replacements.
This is an input-availability repair, not a model/geometry or outcome-driven choice.
Reusable model weights stay in the shared cache and out of Git/release artifacts.

The completed-snapshot OPT125M cell passes six functional waves, actual MHA byte
geometry/pinned shared backing, CPU restores/fresh full recomputation, exact
greedy output and native abort/drain/recovery. Its2.664%longer wait diagnostic
span is retained, with the same unmatched-protocol timing limitation.
[Functional decision](../../output/wait-qualification-20261105/runtime-probe-opt125-complete-functional-decision.json).

The initial abandoned-advice campaign completed all six behavior/protocol waves
and operational/cleanup/allocation checks. Each late return fully recomputed541
tokens with zero external hits/load bytes, exactly as specified. Its raw status is
**inconclusive** solely because the inherited generic required-activity list still
demanded a restore, contradicting explicit `require_restore=False` and required
no-cache-after-abandon. The corrected declaration removes only generic `restore`;
store/CPU-recency activity and every actual no-restore/fresh/equality/protocol
assertion remain. Both fresh arms rerun under the new declared spec; the original
inconclusive result remains sealed and linked. [Pre-retry repair receipt](../../output/wait-qualification-20261105/abandon-activity-declaration-repair.json).

The base-only activity correction missed the condition-level wait override. Its
second complete attempt again passed all six actual functional/operational and
allocation checks, but remained inconclusive because restore was still required
in the effective wait plan. This implementation mistake is retained explicitly.
The final correction changes only `conditions.wait.required_activity`; both base
and merged effective declarations now omit restore, retain store/CPU recency,
and are verified through the serving runner's actual activity-selection helper
before the fresh `runtime-probe-abandon-effective` invocation. No behavioral
assertion or candidate setting changes. Both earlier inconclusive attempts remain
in the explicit compatibility attempt chain. [Effective declaration proof](../../output/wait-qualification-20261105/abandon-effective-activity-repair.json).

The corrected effective abandoned-advice declaration passes all six waves and
operational/allocation checks. Late returns recompute fully without CPU restores,
as required. Both earlier declaration failures remain in the attempt chain.
[Complete receipt](../../output/wait-qualification-20261105/runtime-probe-abandon-effective-functional-decision.json).

The long-context 7B cell passes all six waves, actual CPU restoration and fresh
full recomputation, exact greedy IDs, pinning/shared backing and native
abort/drain/recovery. Real targets extend to6,000 tokens. This closes the short
probe's positional coverage gap without establishing all-position/model support
or application speed. [Receipt](../../output/wait-qualification-20261105/runtime-probe-long-functional-decision.json).

The long-context 1.5B cell likewise passes six waves and all restore/fresh/greedy,
allocation and operational checks. The one-pair diagnostic span is0.607%longer
with wait and remains an alarm, with extra candidate protocol RPCs and no matched
application-performance interpretation. [Complete receipt](../../output/wait-qualification-20261105/runtime-probe-qwen15-long-functional-decision.json).

Graph long-context correctness passes all six waves and operational/allocation
checks. Native disjoint interval dispatch records establish actual timed FULL
graph execution after subtracting a conservative allowance for all untimed
requests. The raw diagnostic wait span is0.247%longer; extra protocol RPCs remain
unmatched and this is not application timing. [Functional receipt](../../output/wait-qualification-20261105/runtime-probe-graphs-functional-decision.json),
[actual graph receipt](../../output/wait-qualification-20261105/runtime-probe-graphs-graph-receipts.json).

The public standalone client passes its actual SHA-256 tool, positive ready-CPU
refresh, actual pinned/shared backing, request drain and owned cleanup.
[Result](../../output/wait-qualification-20261105/example-client/result.json).

Both standalone client runs pass, including actual GPU serving from the separate
2,874-file source copy and own environment. Installation followed the published
commands offline, native import and60focused tests pass, all module/binary paths
are owned, staged native files match the copied source, and the source manifest
remains unchanged. Actual pinning/shared backing, positive CPU refresh, drain
and cleanup pass. The observed dependency closure and seven transitive-version
differences are disclosed; this is isolated author integration reproduction, not
an independent human or identical-environment performance reproduction.
[Isolation result](../../output/wait-qualification-20261105/isolated-reproduction/result.json),
[GPU client](../../output/wait-qualification-20261105/isolated-example-client/result.json).

## Completed graph-mode real-workflow comparison

Three rotated ARC/wait pairs complete384responses. Wait uses8.8707%less batch
time, conditional95%interval8.0846–9.6501%; every pair favors wait. All six
actual pinned/shared-backing observations and actual timed FULL-graph lower-bound
receipts pass. First-pair wait executes53,673prefill tokens versus66,457ARC,
generates7,446output tokens versus7,399, and both have zero preemptions. CPU-load
bytes increase, so lower transfer volume is not the explanation. These are
within-mode complete-system timings, not a graph-versus-eager comparison.
Narrow test-status checks pass while known factual errors and strict formatting
failures remain; useful-answer parity is unestablished.
[Summary](../../output/wait-qualification-20261105/stage4-graphs-summary.json),
[graph execution](../../output/wait-qualification-20261105/stage4-graphs-graph-receipts.json),
[first-pair inspection](../../output/wait-qualification-20261105/stage4-graphs-first-pair.json).

## Completed1.5B full-budget workflow comparison

All six measurements pass; wait's point estimate is3.0629%less batch time,
conditional95%interval−5.2830 to+10.7472%. Every pair favors wait, but neither
useful gain nor the2%neutral-overhead bound is established. All six have zero
CPU loads, prior CPU-eviction returns and preemptions, with essentially identical
native prefill. The third candidate generates8,219tokens versus8,738ARC and
has a larger timing advantage; real tool/paging differences are retained.
No CPU-retention contribution is demonstrated in this all-fit cell. The first
pair also has substantive answer-quality limitations, including an incorrect
candidate per-test outcome claim; timing is not equal-quality successful work.
No extension or tuning is selected. The constrained-byte cell remains separate.
[Summary](../../output/wait-qualification-20261105/stage4-qwen15-performance-summary.json),
[native-work inspection](../../output/wait-qualification-20261105/stage4-qwen15-full-inspection.json).

## Completed1.5B constrained-byte workflow comparison

All six measurements and actual allocation receipts pass. Every pair favors wait;
the mean is4.6652%less batch time, conditional95%interval−1.1809 to+10.1735%.
The2%neutral-overhead bound is met, but a useful timing advantage is unresolved.
First-pair wait executes54,958prefill tokens versus67,498ARC and generates
8,649output tokens versus8,610, with zero preemptions in both. CPU demand loads
increase while recomputation falls. These covariates support work saving, not
an isolated causal or equal-useful-quality claim. The model's answer weaknesses
remain part of the result; no extension or tuning follows.
[Summary](../../output/wait-qualification-20261105/stage4-qwen15-cohost-summary.json),
[first-pair work](../../output/wait-qualification-20261105/stage4-qwen15-cohost-first-pair.json).
