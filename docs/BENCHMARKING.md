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
