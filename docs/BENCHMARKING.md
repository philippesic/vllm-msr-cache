# Development and benchmarking

Use the [pinned runtime](../vendor/VLLM.md) and run from the repository root.
The maintained suite lives in `benchmarks/serving_suite/`. It starts a real local
vLLM HTTP endpoint and generates live responses through the vendored launcher.
It never downloads models or imports run-local scripts from `output/`.

## Choose a tier

- **Smoke:** one candidate server, four sessions, two identical requests per
  session, concurrency one, up to eight output tokens. Checks startup, streaming,
  deterministic repeated outputs, offload/restore activity, telemetry, and cleanup.
  This catches setup failures; it is not a performance or KV byte-integrity proof.
- **Screen:** one matched native ARC/candidate pair, six sessions, two turns,
  concurrency two, up to 32 output tokens. A cheap indication of cost and activity.
  It never automatically rejects a development. Noisy, inactive, capped, or
  unrepresentative results need a targeted follow-up. There is no guarantee a small
  workload will reveal every useful improvement.
- **Confirm:** three matched pairs in alternating arm order, with 12 disjoint
  sessions per pair, three turns, concurrency four, up to 128 output tokens. Reports
  per-pair results and a paired 95% interval for throughput. Require representative
  inputs and more repetitions when the interval does not resolve the question.
- **Operational:** each arm keeps one endpoint alive for four waves of 12 sessions,
  three turns, concurrency four per wave. Generates to EOS or the context limit,
  checks queue drain, disconnects an isolated request, requires its exact native
  scheduler abort receipt, then verifies generation recovery. Positive wave spacing
  can overlap waves. This is real serving with scripted traffic, not real users,
  executed external tools, an indefinite soak, or a production certification.

Defaults are deliberately bounded and configurable. Smoke/screen use an authored
cache-pressure fixture unless a corpus is supplied. Confirm/operational require a
supplied corpus; `--allow-synthetic` explicitly permits fixture qualification only.
The default candidate is semantic-mean eviction with speculative prefetch disabled.
It is not the separately qualified CPU-advice research profile.

## Real repository-agent workflow

### Explicit four-system comparison

The published adapter also accepts `--published-variants arc,wait,continuum,upstream`
and a predeclared `--published-order-seed`. This removes the semantic arm and
makes ARC a true no-hint control: no client observed-prefix digest/metadata,
hint RPCs, plugin or source registration. Its remaining connector settings match
active wait ARC except CPU retention and the CPU-advice enable budget. Both keep
the same application continuation priority; upstream now receives that priority
through its AsyncLLM bridge and uses its native priority scheduler. Explicitly
select native upstream `eviction_policy: arc` in its runtime config. Set
`LMCACHE_SAVE_DECODE_CACHE=true` in Continuum's runtime environment to match
native decode KV publication; LMCache's default false is a supported control,
not a requirement of published Continuum. The corrected row was selected before
timing irrespective of its performance.

An upstream compatibility copy is supported only through explicit runtime fields
`execution_source`, `compatibility_patch`, and `compatibility_patch_sha256`.
Preflight reconstructs its one permitted changed module from published source
plus patch, checks every other Python file, verifies the imported checkout, and
freezes actual Python (including submodule support) and native content hashes.
The published checkout must remain clean. Label these results as compatibility
patched; never relabel them as unmodified published behavior.

The four-system path checks actual backing allocations and pinned CPU memory
before measured requests. Native CPU chunks remain 16 tokens; Continuum's
LMCache chunks are 256 tokens. Continuum retains its scheduler, published runner
and supported WSL input-buffer behavior. These checks control specified settings;
they do not remove runtime/backend differences or establish policy superiority.
See the [predeclared fairness manifest](../evidence/fair-four-20261104/FAIRNESS.md).

### Five-system comparison

`make bench-agent-five PY="$PY" MODEL="$MODEL" OUT=output/my-five-way RUNTIMES=/path/to/runtimes.json`
selects one pass through ARC, wait-hint, semantic, Continuum public preview, and
upstream vLLM PR #59714. It retains the sixteen-session real search/test/evidence
workload and cache budgets. See the [runtime manifest example](../benchmarks/serving_suite/published-runtimes.example.json).
This path requires **two separately installed published runtimes**; the existing
project interpreter cannot substitute for them. The adapters have CPU/HTTP tests and both external runtimes have passed a
one-session real-tool probe on this host. A full comparison is still a separate
measurement; these probes do not establish a performance ranking. The completed
[one-pass comparison](../evidence/five-way-20261104/README.md) records measured
results and the host-specific limits.

Pins are in `benchmarks/serving_suite/published.py`. Install each exact checkout
editable in its own environment with its own native dependencies (Continuum's
build requires torch 2.8.0; this upstream draft requires torch 2.13.0). Follow each
checkout's build instructions; do not mix their Python sources into our installed
wheel. Continuum also needs its LMCache dependency. Preflight rejects a different
revision, dirty checkout, interpreter importing another wheel, or unequal declared
CPU budget. Dependencies, commands and identities are retained in the run.

All five arms use eager execution for a common execution mode across versions.
The local three keep their existing scheduling credit; Continuum uses its own
scheduler, upstream its default scheduler. This compares **systems**, not isolated
eviction logic. The external HTTP bridge only forwards requests to the published
AsyncLLM implementations: upstream's draft exposes `kv_hints` there but not in its
normal completions schema. The upstream request policy assigns priority 10 for
60 seconds after completion, then clears the session claim on the final turn;
this follows the published example, not an upstream-tuned optimum.

Continuum's preview omits the paper's automatic timeout estimator and parses
generated bash blocks to recognize tools. Every arm therefore gets the same
additional instruction to echo the next fixed tool command in one bash block.
The application still executes only the original declared argv, never generated
shell. This prompt change means **rerun all five arms**; do not reuse historical
three-way numbers. Whether Continuum actually pins must be checked in its retained
`scheduler_timestamps`; an HTTP success alone is not activation evidence.

External native accounting is unavailable where it lacks our instrumentation;
it is never filled with zero or treated as our receipt protocol. Continuum activation is checked from its published scheduler recorder; upstream
activation uses read-only Python return observation of priority application and
selection, without replacing engine methods. Missing activation makes a case
inconclusive. Measured request/workflow timings, memory and tool outcomes are
reported independently of missing native source accounting. One pass is an exploratory comparison, without a confidence claim.
On this WSL host, the upstream runtime needs `VLLM_WSL2_ENABLE_PIN_MEMORY=0`
and uses `VLLM_USE_V2_MODEL_RUNNER=0`; pinned-memory probes failed with illegal
CUDA memory access. These explicit compatibility settings belong in its manifest
and limit conclusions to this setup. Historical `bench-agent-pressure`
behavior and results remain unchanged.

For wait-retention development on the local 7B AWQ / 2080 Ti setup, use the
server-queue pressure profile:

```bash
make bench-agent-pressure PY="$PY" MODEL="$MODEL" OUT=output/my-pressure-pair
# Three rotated repetitions, including semantic retention:
make bench-agent-pressure PY="$PY" MODEL="$MODEL" OUT=output/my-three-way \
  BENCH_ARGS='--repeats 3 --semantic-config benchmarks/research_checks/fixtures/semantic_wait_prefix.json'
make bench-verify PY="$PY" OUT=output/my-three-way
```

This profile offers sixteen distinct repository investigations to eight server
sequences, with all ready requests submitted directly to the server. It allocates
2,560 GPU KV blocks (2,240 MiB) and 3,072 CPU KV blocks (2,688 MiB): enough GPU
headroom for active contexts, while the full growing histories compete for cache.
Tools execute live through four subprocess workers; tool queue and process time
are separate, with causal hints covering both. This bounds host import-memory
pressure independently of HTTP concurrency. Both arms get identical priority scheduling and 8-second
continuation credit. Causal hints start before tool launch, refresh during actual
tools, and continue while a returning request queues, stopping at first token.
The explicit CPU-only wait configuration protects ready CPU prefixes even while
they remain on GPU. This removes the optional GPU-residency gate; identity,
ownership, audits and no-speculative-dispatch safeguards remain enabled.
The normal `bench-agent` defaults below remain a smaller mechanics screen.

`REPORT.md` reports completed request/workflow rate, average offered first-token
latency including client queueing, tool-ready latency, token rate, and paired
whole-run intervals. With four requests per completed workflow, request-rate and
workflow-rate ratios coincide. Startup is separate. Compact native output keeps
counters/audits but omits verbose key/transport histories; it is not a byte-lifetime
proof. Exact tool stdout is retained; irrelevant pytest elapsed-time text is
omitted from model input. All costs inside the measured application are included.
A single pair is an exploratory screen; use at least three repetitions and read
prompt/output fingerprints, activation and quality limits with the interval.

`--semantic-config` adds a third arm. It uses the existing SemanticPolicy in place
of ARC, composed with wait-hint recency; it is not an additive ARC-preserving
semantic score. An explicit `--candidate-config` in a three-way comparison must
remain exact-KV CPU-only wait-hint ARC. The supplied semantic profile enables
native relevance updates at capture stride 2 with prefix/chain protection and
disables KV SVD and GPU prefetch.
Those techniques require separate configurations and their own activation checks.


`make bench-agent` runs a read-only repository investigation application against
the real HTTP model endpoint. Eight concurrent sessions each make four generated
requests, accumulating actual assistant history and outputs from three executed
tools: repository search, the relevant pytest file through the pinned runtime,
and a final evidence search. The application selects the fixed tool pipeline;
model-generated shell text is never executed. Initial context is actual source
and tests, with explicit excerpt limits and hashes. There is no synthetic padding,
recorded tool-output replay, or sleep standing in for tool execution. These are
authored maintenance tasks about this repository, not production user traffic,
SWE-bench task success, or an autonomous patch-writing agent.

```bash
make bench-agent PY="$PY" MODEL="$MODEL" OUT=output/my-agent-eval \
  BENCH_ARGS='--execution graphs --require-activity store,restore,cpu_recency'
make bench-verify PY="$PY" OUT=output/my-agent-eval
```

The default comparison is ARC versus GPU-gated CPU-retention wait-hint ARC.
Both arms receive the same causal hint protocol and instrumentation. Hints start
after each tool process spawns and refresh once per second only while it runs;
the producer knows the completed prefix and that a tool is pending, never future
output or completion time. Short searches may finish before an acknowledgement;
reports count those explicitly rather than claiming useful advance notice.
The test subprocess provides a genuine longer wait while other sessions use the
GPU. Wait-hint CPU refresh is advice, not speculative prefetch or a guarantee of
subsequent reuse. Default caches are 1,024 GPU blocks and 896 MiB CPU KV with an
8,192-token context. Use the cached 7B AWQ model for the intended 2080 Ti pressure
profile. This does not reproduce the historical frozen priority configuration.
The existing runtime uses a positive prefetch enable budget before entering its
CPU-only advice branch; that branch plus `wait_hint_diagnostic_no_dispatch` keeps
the wait arm CPU-only. Native staging counters must remain zero. Setting that
budget to zero suppresses CPU advice too, and the required `cpu_recency` check
will report the arm inactive.

Use `--sessions`, `--concurrency`, `--gpu-blocks`, `--cpu-bytes`, `--context-chars`,
`--arrival-interval-s`, `--waves`, and `--repeats` to vary offered load and capacity.
Paired order reverses on alternate repetitions. `--task-file` accepts
`repository-investigation-tasks-v1`, using the versioned
[task file](../benchmarks/serving_suite/repo_tasks.json) as an example. `--repo-root`
changes the actual tool working directory; that repository must provide the
declared files and runtime/test entrypoint. Test failures, tool timeouts, context
overflow, request failures, and missing telemetry are retained as failures. No
hidden retry or truncation repairs them. Outputs must use a new directory.
`--candidate-config` supplies another semantic/prefetch/SVD arm while retaining
the same workload; its configuration and required activation must match the claim.

The normal `REPORT.md`, plan/source/model identities, raw streams, metrics,
resources, requests, process cleanup, and offline artifact sealing still apply.
The agent tier also freezes executed source in `executed-source.tar.gz`, including
dirty and untracked Python inputs. Temporary `runtime-staging/` import trees are
owned by the launcher and excluded from artifact sealing; source and wheel
identities are retained independently.
Each case additionally emits `agent.json`, `request-diagnostics.csv`,
`workflow-*.jsonl`, raw tool stdout under `tools/`, and `kv-qualification.jsonl`:

| Measurement | Meaning and limits |
| --- | --- |
| Workflow completion/rate/duration | Full four-request session including actual tool waits and client queueing. Startup is reported separately. |
| Initial/return TTFT, p50/p95/max | Separates first requests from continuations. Tool-completion-to-first-token includes client queueing. Small-sample tails are descriptive. |
| Tool duration, exit, bytes, truncation | Actual command, working directory, stdout hash and lifetime. Excerpt limits affect model input only; raw output remains. |
| Hint RTT, reasons, lead time, windows | Actual acknowledgements before tool completion, GPU-gate skips, ready CPU refreshes and errors. Repeated key refreshes are observations, not unique saved keys. |
| Prefill, reuse, recomputation | Per-request native receipts and endpoint counter deltas. CPU eviction observations on returning requests show cache competition; neither allocation nor refresh activity alone proves useful work saved. |
| Transfers/prefetch | Demand bytes and native worker intervals; speculative submission/completion/consumption/reclamation are separate. Missing critical-path evidence stays unavailable. |
| Pressure and queueing | Native scheduler steps, running/waiting counts, reclaimable GPU pages; sampled GPU/CPU cache gauges. Reclaimable pages include idle cached prefixes. |
| Memory/utilization/power/cost | GPU samples, server RSS/CPU, separate client-plus-tool RSS/CPU samples, scrape RTT and sampled device energy. RSS can double-count shared pages; short child lifetimes and allocator peaks can be missed. |
| Answer checks | Final claimed test result versus actual pytest exit status. Wrong claims, unparseable status and strict-format violations are separate. Exact answers remain for review; code-reasoning accuracy is ungraded. |
| Comparison integrity | Exact prompt/output fingerprints, finish reasons, incomplete cases, configured geometry, provenance and cleanup. Different live outputs prevent identical-work causal claims. |

Detailed native tracing and HTTP scraping have cost, enabled equally and charged
to both arms. Gauge updates can lag the server; native scheduler/key records
provide finer observations. A pair without observed competition for returning
CPU prefixes is inconclusive for the retention hypothesis. One arm avoiding all
such evictions can be the desired treatment outcome; zero baseline CPU restores
can likewise be the outcome that advice improves, not a broken baseline. One complete pair
accepts the harness, not a policy performance claim; use repetitions and inspect
whether advice both activates and has a useful opportunity before drawing one.
Native request counts, prompt tokens, computed prefill tokens and loaded tokens
at first admission are cross-checked against endpoint counters. Cumulative
original-prompt execution and engine-wide physical transfer traffic are separate;
request-attributed traffic can leave a reported gap under preemption. Disagreement
or unavailable required
accounting fails measurement validation. `bench-verify` recomputes the diagnostic
report from retained requests, workflow events, server records and resource samples.

For prefetch qualification, `--qualification-trace` enables a default-off native
queue/load trace in each case's `kv-qualification.jsonl`. Analyze it with
`python -m benchmarks.research_checks.hybrid_qualification PATH`. It separates
arrival-to-load, async-load issue-to-receipt, and receipt-to-admission intervals,
and samples reclaimable GPU blocks. These are diagnostic observations with I/O
overhead. Reclaimable space is not necessarily empty; CPU readiness and bandwidth
are not observed. Reported candidate overlap is optimistic and cannot establish
exposed restore stalls, throughput gains, or an MSR qualification by itself.

## Commands

```bash
PY=/home/philip/.local/share/venvs/vllm-semantic-cache-ext/bin/python
MODEL=/absolute/path/to/cached/model/snapshot
make runtime-check PY="$PY"
make test PY="$PY"
make bench-tests PY="$PY"
make benchmark-help PY="$PY"

make bench-smoke PY="$PY" MODEL="$MODEL" OUT=output/my-change/smoke
make bench-screen PY="$PY" MODEL="$MODEL" OUT=output/my-change/screen
make bench-confirm PY="$PY" MODEL="$MODEL" OUT=output/my-change/confirm \
  BENCH_ARGS='--corpus /absolute/path/to/corpus.json'
make bench-operational PY="$PY" MODEL="$MODEL" OUT=output/my-change/operational \
  BENCH_ARGS='--corpus /absolute/path/to/corpus.json --arrival-interval-s 0.1 --wave-interval-s 5'
make bench-verify PY="$PY" OUT=output/my-change/operational
```

Each `OUT` must be new; existing results are never overwritten. Add `--plan` to
`BENCH_ARGS` for GPU-free input/plan validation; use a separate output directory
for the actual run. It checks the local model config exists but does not load a
model, tokenize inputs, or establish that a workload fits memory. Normal startup
checks prompt lengths; later context overflow is a recorded failure, never silent
truncation. Model and corpus paths can contain spaces when passed directly to the
Python CLI or correctly quoted within `BENCH_ARGS`.

The equivalent CLI is `"$PY" -m benchmarks.serving_suite.runner TIER --model
"$MODEL" --output NEW_DIR ...`. Do not wrap the suite itself in the runtime
launcher: it launches each server through that launcher. CPU checks requiring
patched vLLM APIs use `make test` / `make bench-tests` through the launcher.

Exit code zero means measurement checks passed; code two means failed or
inconclusive execution. Invalid CLI inputs also return nonzero. Read `REPORT.md`
and `result.json`, not only the exit code. Offline verification checks integrity
and aggregate consistency; it can successfully verify an honestly failed run.

## Match configuration to the question

Use `--candidate-config FILE` for a full SemanticOffloadingConnector KV-transfer
JSON, as described in [configuration](CONFIGURATION.md). The suite forces equal
CPU budgets for both arms and enables the native scoring receipt on the candidate.
Use `--gpu-blocks`, `--cpu-bytes`, `--max-model-len`, and `--gpu-memory-utilization`
explicitly when changing geometry. Defaults are 128 GPU blocks of 16 tokens,
256 MiB offload space, 2,048 context tokens, and 0.75 memory utilization. Configured
KV capacity is not total device VRAM. Longer public prompts need a larger context
and sufficient cache capacity; do not silently shorten them.

`--require-activity store,restore` is the default. Add `scoring` when applied
semantic relevance updates are required. Add `hint_accepted` with `--publish-hints`
and a candidate configuration containing an explicit `admission_namespace` when
checking speculative wait-hint admission. CPU-retention-only advice instead uses
`--require-activity store,restore,cpu_recency`: this requires a positive count of
ready CPU keys actually refreshed. Its native acknowledgement is deliberately
`no_action` for GPU prefetch, so it must not be counted as an accepted GPU lease.
`--publish-hints` enables the endpoint plugin and declares the observed input prefix
and full session identity on the initial request as well as continuations.
The hint producer uses only the completed prefix and a constant 750 ms declared
wait-budget estimate; actual scripted delays remain in the corpus. It does not
provide future user text or measured future completion times. Acceptance does not
prove native prefetch adoption;
aggregate transfer counters cannot establish speculative-copy attribution. Store/restore/scoring activity also does not prove
that CPU eviction candidates competed under capacity pressure. Size the working
set and cache budget for the retention hypothesis and retain specialized eviction
proofs before attributing a benefit to ranking. Use the
specialized preservation/attribution checks for those claims. Empty required
activity (`--require-activity ''`) is only appropriate for an explicitly declared
overhead-only measurement. Missing counters remain unavailable; observed zero
activity is inconclusive when that activity is required.

`--execution eager` is the default. `--execution graphs` selects the same
FULL_DECODE_ONLY CUDA graph mode for both arms, with compilation disabled. The
host uses FP16/TRITON_ATTN and mapped pinned memory; BF16/FlashAttention 2 are not
supported on the 2080 Ti. The model must fit this geometry. `--natural-eos` removes
the small output cap up to the remaining context; `--no-natural-eos` restores the
cap. Finish reasons and actual output lengths are always retained.

`--concurrency` limits active client requests **per wave** and configures vLLM's
maximum running sequences. With overlapping waves, total offered concurrency can
exceed that value and queue at the server. Zero `--wave-interval-s` drains each
wave before the next; positive spacing schedules waves against their start times.
`--arrival-interval-s` spaces sessions within each wave. Session waits are scripted
and recorded; no external tool is executed. Warmup, startup, cancellation, and
recovery are excluded from normal request throughput. Load duration includes
scripted waits and drains, so throughput describes the offered workflow, not a
maximum GPU kernel rate.

Defaults bound startup to 180 seconds, individual requests to 90 seconds, and the
normal workload of each case to 900 seconds. These are separate budgets; model
hashing, startup, drain, cancellation receipt collection, and shutdown also take
time. A stopped/failed case preserves partial records and owned-process cleanup.
There are no hidden request retries. SIGTERM and Ctrl-C produce retained failure
records and cleanup; SIGKILL cannot run cleanup code.

## Corpus contract

Supply JSON with this shape. `turns` are user messages; later turns use the actual
previous generated text. `context` is attached only to the first user message.
`wait_s` contains delays between turns. Optional `expect_contains` supplies
case-insensitive substring checks for successive responses; these are simple
content assertions, not a general quality judge.

```json
{
  "schema": "semantic-serving-corpus-v1",
  "synthetic": false,
  "provenance": "Describe the source, selection, version, and permission to retain it",
  "sessions": [
    {
      "id": "source-1",
      "context": "Evidence available to this session",
      "turns": ["First question", "Follow-up question"],
      "wait_s": [0.2],
      "expect_contains": [["expected first answer"], ["expected follow-up"]]
    }
  ]
}
```

Provide at least `sessions × repeats` unique sessions for confirmation and enough
turns per selected session. Selection is deterministic from the seed and disjoint
across confirmation repetitions. The suite stores the full corpus and exact
prompts/responses locally, so use an appropriate artifact location for sensitive
inputs. In smoke, subsequent requests intentionally repeat the first prompt;
content assertions must reflect that smoke behavior.

The existing public corpus format can be imported without its old runner:

```bash
"$PY" -m benchmarks.serving_suite.import_corpus \
  /path/to/session-corpus.json output/my-corpus.json --cohort agent
```

The importer supports `quantized-realworld-sessions-v1`, preserves original user
text and source IDs, and records the source hash plus scripted waits. A previously
used public corpus is useful for harness qualification; label it honestly and use
fresh held-out inputs for a new policy claim.

## Evidence and interpretation

Every executed run retains the validated plan/corpus, model-file and executed-source
hashes, package/hardware identity, exact server commands/configurations, raw logs,
per-request payloads/token IDs/decoded text, raw SSE, native Prometheus snapshots,
and CPU/RSS/GPU resource samples. `case.json` / `result.json` report completion and
error counts, input/output tokens, finish reasons, client queue delay, time to first
token, end-to-end latency, chunk gaps, throughput, required activity, operational
checks, and cleanup. The artifact manifest seals files for offline verification.

Pair reports say whether exact prompts and outputs agree. If live continuations
diverge, throughput compares different live workflows and cannot by itself isolate
a cache-policy speedup. Three pairs provide an initial conditional interval, not
population-wide confidence; do not treat individual requests as independent paired
repetitions. Small-sample p95 values are descriptive, not tail-latency guarantees.

Resource samples default to 250 ms. GPU memory/utilization/power cover the whole
device; summed process RSS can double-count shared pages. CPU deltas cover the
load window. They are sampled observations, not allocator peaks or an indefinite
leak test. Raw streams preserve transport evidence; completed HTTP responses and
matched smoke outputs do not directly inspect transported KV bytes. Answer quality
is unassessed unless explicit content assertions are supplied, and even those do
not establish general quality. The suite must be supplemented with application
quality and production-load evaluation before deployment decisions.

The [suite qualification record](../evidence/benchmark-suite-20261103/README.md)
records actual host turnaround, failure controls, and tested configurations. Its
harness passes are not new policy-performance wins and do not transfer the frozen
research campaign's qualification to current source.

When upgrading vLLM or changing the protocol, run `bench-tests`, the four GPU
tiers, and the affected failure controls before calling the new combination
qualified. Keep instrumentation in versioned source with contract tests; do not
inject replacement engine methods from a run directory. The abort receipt and
scoring counter are opt-in native interfaces tied to this pinned runtime.

## Shared GPU and repository conventions

Check [STATUS](STATUS.md) and live process state before GPU work. Idle gaps between
cases do not release an owner's reservation. Maintained runners share `harness.gpu_lock`: a supervisor holds the common
GPU-0 reservation across its entire invocation, including gaps between cases.
Direct suite runs also hold the historical `/tmp/vllm-semantic-cache-gpu.lock`;
frozen research children keep their original leaf lock under the supervisor.
The suite rejects existing compute processes. Unwrapped historical launchers may
not honor the supervisor reservation. Do not overlap measurements with another campaign. It shuts down
only its spawned server tree and verifies cleanup before another case.

Keep reusable runners/validators in `benchmarks/`, shared helpers in `harness/`,
raw runs in ignored `output/`, and compact portable findings in `evidence/`.
Keep only the current handoff in STATUS and completed outcomes in
[PROJECT_HISTORY](PROJECT_HISTORY.md). Preserve failed/negative attempts that
explain conclusions. Ignored output may contain frozen dependencies, not disposable
scratch. New run freezes track executed sources; unexecuted documentation changes
do not invalidate serving. Do not edit executed Python while a suite is running.

Existing specialized runners remain available: `benchmarks.run_latency_suite`,
`eviction_prefetch_campaign`, `wait_hint_serving_diagnostic`, and the staged
preflight tool. They are not interchangeable with frozen 2080 Ti protocols. Follow
the selected result's original manifest for reproduction. For focused old checks:

```bash
make test-harness PY="$PY"
make test PY="$PY" TESTS='tests/test_wait_hint.py tests/test_wait_hint_cpu_recency.py'
```

## Portable automation and historical reproduction

`make check-light PY=python` runs dependency-light contracts and syntax checks.
Install `requirements-checks.txt` in an isolated Python 3.12 environment. The
GitHub workflow runs this target on pushes and pull requests, without torch,
vLLM, model downloads, or a GPU. It is not the full runtime test suite. Run the
pinned full tests and relevant GPU tier locally before runtime-sensitive claims.

Reusable HTTP, cadence, priority, GPU-gate and operational validators now live in
`benchmarks/research_checks/`. They take an explicit artifact directory and never
import code from ignored output. Their agreement was checked against retained
real campaign results. The frontier/AWQ coordinators remain **historical frozen
reproduction tools**, with original runtime, source and corpus requirements. Their
old drivers and archived helpers remain unchanged in output for reproduction;
they are not a fresh-checkout alternative to the maintained four-tier suite.
Changing a coordinator invalidates its old source-pinned execution plan: prepare
a new plan for an explicitly authorized new experiment, never rewrite old pins.
Paused AWQ experiments remain paused.

For a compact inventory of all retained operating points (including failed and
interrupted attempts), run after measurements:

```bash
"$PY" -m benchmarks.research_checks.wait_opportunity_report output/my-campaign \
  --output output/my-campaign-summary.json
```

This creates JSON and CSV without changing source runs. Its completed-prefix
transcript bound distinguishes new prompt work from remaining recomputation;
above-bound work can include preemptions. Do not pool different operating points
or selecting screens into confirmation intervals. See the
[validated wait-opportunity comparison](../evidence/wait-opportunity-20261104/README.md)
for results, controls, limitations and retained failures.
