# Benchmark implementation

The [evidence contract](benchmark-contract.md) defines the decisions and validity requirements.
release-01 stopped after panel 0/7 at Philip's requested boundary; its existing queue
is cancelled. Do not resume or launch the old schedule. Compare and release
acquisition are not qualified.
Do not use a successful model probe as release or performance evidence.

The maintained entry point is `uv run --locked python -m tools.benchmark`:

- `functional --runtime-python artifacts/venvs/runtime/bin/python --model /local/dense-model
  --out artifacts/runs/new-attempt --timeout 600` runs the existing dense GPU gate through
  the shared benchmark entry point. Its receipt uses the runtime verifier format;
  verify it with `python -m tools.runtime_verify --verify artifacts/runs/attempt`.
- `models` prints immutable model candidates and their qualification state.
- `plan --profile functional|compare|release` prints the allocated cases, repetitions
  and budget. The output explicitly says `unqualified_schedule`.
- `probe-model --runtime-python artifacts/venvs/runtime/bin/python --model /local/model
  --out artifacts/runs/new-attempt --system gpu-only|native-arc|wait-hint|lmcache --timeout 900`
  launches one localhost server, checks generation, and for CPU-offload modes tests
  GPU-only displacement and restored versus fresh token IDs. Native ARC and wait-hint
  also require registered readiness. New LMCache probes require external restoration;
  older generation-only receipts remain limited to generation. LMCache on the current
  WSL host has successful Qwen and Gemma restoration probes using the opt-in
  VMM/staged-copy repair, plus one graded Qwen coding pilot. Broader application
  and reliability qualification remains pending.
- `verify-probe artifacts/runs/attempt` checks a v3/v4/v5/v6 receipt, frozen source/model hashes,
  retained artifacts, token equality and original grader outcomes without GPU work. Earlier investigation
  attempts retain their original schemas; this reader does not silently upgrade them.
- New traces use `benchmark-trace-v3`: the sustained operating check sends each
  returning prompt (1025, 3073 or 6145 tokens) once more while it is GPU-resident,
  displaces the GPU cache and compares the CPU-restored return with that resident
  return when both reused the same number of cached tokens. A fresh run is not the
  reference, because GPU prefix reuse alone changes Gemma's output. v1/v2 receipts
  keep their original fresh-run comparison.
- A focused controlled probe adds `--trace-case T1|T2 --trace-seconds 180
  --pressure-prefixes 24` to `probe-model`. T1 checks resident/restored continuation;
  T2 adds competition, abandoned advice and unhinted return. Use 48 pressure prefixes
  for the admitted Qwen configuration and 24 for Gemma. This is supplementary
  mechanism evidence, not a replacement for a complete application campaign.
- `probe-model --trace-case I1 --trace-seconds 180` is a bounded continuation
  diagnostic. It uses the failed Gemma prompt (token IDs 5000 through 8071, 32 output
  tokens), recording three fresh executions, a GPU-resident return and a return after
  GPU displacement, all with the same parameters and top-five logprobs. CPU arms
  require restored reuse; the GPU-only displaced control requires zero cached tokens.
  The reader checks actual
  cache conditions and recomputes token comparisons. A completed diagnostic can report
  unequal tokens; that is an observed correctness question, not a passing invariance
  gate. It does not repeat or replace the failed operating cell. Native and GPU-only
  diagnostics completed and were independently verified: fresh outputs were stable,
  while GPU-resident outputs differed even without CPU offload. Native restoration
  also reproduced the final-token mismatch. Correctness qualification remains
  unresolved by fresh-run comparison. The GPU lane case
  `test_restored_kv_bytes_match_stored_and_resident_boundary_output` settles it:
  after a GPU-only reset and clearing the GPU KV memory, every block a CPU restore
  wrote must be byte-identical to a block stored before displacement, and the
  restored output must equal a GPU-resident return at the same cached boundary.
  On Gemma every block of the restored 1,536-token prefix matched; the only
  differing blocks are the partially filled pages holding the recomputed boundary
  token and decode positions, and outputs matched; see the
  [settle result](../results/promote-hold-settle/result.md).

- `simulate --runtime-python PY --out artifacts/runs/NEW` replays concurrent
  sessions on the CPU in about a minute and writes `simulation.json`. It re-runs
  itself under the runtime interpreter as `simulate-native`, which loads
  `wait_hint_offload/vllm_adapter/replay.py`: the real pinned `CPUOffloadingManager`
  and ARC policy plus the product's reachable-hold code. The tools-side
  model reproduces the offloading scheduler's call order (lookup, whole-request touch
  at start and after each store, store completion delay, final store at computed
  tokens), a GPU prefix cache that frees a request's tail first, a running-request
  limit and GPU admission. Recurrent-state groups (window of one chunk) store only
  the last chunk of each scheduling step, as the engine checkpoints them; other
  groups store every chunk. Arms are `gpu-only`, `native` (advice disabled),
  `reachable-hold` and `oracle` (hold until return without cap or deadline). It reports recomputed prefix tokens on return for hinted
  and unhinted cohorts, store failures, evictions, restores, GPU displacement,
  abandoned hints and hold yields, plus working-set ratios. Scenarios live in
  `benchmarks/replay/scenarios.json`; `--scenario NAME` selects some. Session shapes
  in `benchmarks/replay/release-01-shapes.json` come from release-01 block-0
  application requests (prompt, generated, unpressured cached tokens as the shared
  prefix, and gaps). A scenario whose `gpu-only` and `oracle` hinted recomputed shares
  differ by less than 10 points is reported `non-discriminating` and excluded from
  comparison. Results are simulated direction: arrival times are arm-independent and
  loads are instantaneous. Geometries declare cache groups (Gemma: one 64-token
  full-attention group and five sliding groups of 16-token chunks with a 64-chunk
  window, as measured) and whether GPU prefix reuse serves returns
  (`gpu_prefix_reuse`). Attention-style GPU reuse overstated GPU hits for both hybrid
  models in the engine; the `-no-gpu-reuse` scenarios are the calibrated variants.
- `pressure --profile smoke|advice|full --runtime-python PY --lmcache-python LMPY
  --model ID=PATH [--model ...] [--arm NAME ...] --out artifacts/runs/NEW` is the
  multi-session pressure benchmark defined in `benchmarks/pressure.json`. Before any
  GPU work it refuses seeds whose schedule lacks hinted, unhinted or abandoned
  sessions, and runs each model's workload on the run's exact seeds through the
  simulator (`precheck.json`):
  a model whose `gpu-only` and oracle-pool hinted recomputed shares differ by less
  than 10 points is refused. Each model/arm gets one server; repetitions reuse it
  after a full cache reset. A cell first restores a separate 2112-token prefix
  after a GPU-only reset (the measured sessions are never pre-promoted), measures
  CPU keys per token from native store bytes, then replays the seeded schedule:
  five concurrent sessions as token-ID prompts (each turn reuses the previous
  prompt and actual output tokens up to the declared shared prefix), four one-shot
  competitors, real gaps, `ignore_eos` fixed-length outputs, and one hint per wait
  published at completion (abandoning sessions hint after their last turn). Pressure
  cells use `--max-model-len 6144`, conversations of at most 5000 tokens, two
  running requests, Qwen's GPU cache limited to 36 blocks (12,288 tokens) and
  Gemma's native 12,352 tokens, against a working set near 21K tokens. CPU pools
  are about 0.6 of the working set at measured keys per token (Qwen 60 keys of 16.5 MiB, Gemma 2560 MiB);
  LMCache gets the same nominal bytes. Arms: `gpu-only` (known worse),
  `native-arc` and `native-arc-repeat` (advice-disabled control and its A/A repeat),
  `wait-hint` (reachable-hold), `lmcache`, and
  `oracle` (native ARC with a pool larger than every stored token, the known-better
  control; an engine hold-until-return would need a product-only setting).
  `verify-pressure DIR [DIR ...]` recomputes everything from raw request records.
  Recomputed tokens on return are the shared prefix floored to the model's hit
  alignment (528 Qwen, 64 Gemma) minus cached tokens; TTFT is the engine's.
  A cell is `valid` only if its activity contracts hold: hinted and unhinted
  returns and an abandoned hint exist; the separate-prefix restore worked; native
  store volume exceeded the pool (evictions) and load bytes were nonzero (restores);
  the pool holds the largest conversation at the measured keys per token; advice
  arms acknowledged every hint with the expected mechanism and applied it; the
  gpu-only arm shows displaced returns. Otherwise the cell is `unexercised`
  (LMCache eviction/restore counters are not observable and are left unknown).
  A model is `discriminating` only if valid known-worse and known-better controls
  differ by at least 10 points and the A/A difference is smaller than that gap;
  otherwise comparisons are withheld. Several directories (for example smoke
  controls plus `advice` repetitions) can be verified together if they share one
  model definition (workload, engine settings, arms, controls). Candidate rules are
  fixed per round in its task file and recorded in its result. Gemma uses a model workload override (4 sessions, 4000-token
  conversations, 2 one-shots) because native offload stores 21 one-MiB keys per 64
  tokens (one full-attention chunk and five sliding-window groups of four 16-token
  chunks), about 336 KiB per token: its pool is 2560 MiB and its oracle pool
  7680 MiB. The oracle checks that its pool never filled instead of evictions.
  Pressure definitions set the simulator's `gpu_prefix_reuse` to false because the
  engine's gpu-only arm recomputed nearly every returning prefix on both hybrid
  models; simulator-only fields do not change a model definition's identity.
  A profile lists seeds, optionally per model; `--seed N` (repeatable) replaces a
  profile's seeds for one invocation and is recorded in the admission.
  `plan --profile pressure-smoke|pressure-advice|pressure-full [--calibration DIR]`
  projects GPU time from measured cell costs.
- `report-probe artifacts/runs/attempt` verifies first, then summarizes task quality,
  request counts, context compaction, hint activity and timing. Missing request captures
  refuse. It makes no comparative performance or release claim.
- `grade-coding --harness-python artifacts/venvs/harness/bin/python --dataset artifacts/assets/test.json
  --predictions artifacts/runs/preds.json --instance INSTANCE --out artifacts/runs/new-grade`
  grades an existing prediction without rerunning generation. It creates a new receipt
  linked to the original dataset and prediction hashes; never rewrite the original run.

A coding pilot adds `--coding-dataset artifacts/assets/test.json`,
`--harness-python artifacts/venvs/harness/bin/python` and `--instance INSTANCE_ID`.
The default --coding-kind repair uses the original issue and grader.
--coding-kind regression instead asks for one new benchmark_generated_test.py,
keeps the reference fix hidden, and checks the same test before and after that fix.
It is a project-authored rubric, not an official SWE-bench test-generation score.
`--coding-kind navigation` instead investigates a pinned project and submits only
`benchmark_navigation.json`. The answer names implementation/callers, current
override behavior and affected upstream tests. Grading checks that answer and runs
the selected upstream tests plus hidden executable behavior checks in a fresh image.

It executes mini-swe-agent in a disposable Docker container, captures every completed
model request and causal tool-wait hint, then invokes SWE-bench 4.1.0's original grader for the classic Verified dataset schema.
Resolved/unresolved task quality is distinct from whether acquisition and grading
completed. Per-request timing is engine timing, not client streaming TTFT; the
application uses non-streaming responses to retain registration acknowledgements.
The proxy allows 90 seconds per generation and the harness client 120 seconds;
these request deadlines do not increase the 240-second task acquisition limit or
campaign budgets. Deadline cleanup retains the latest available trajectory and
patch and grades that patch separately from task completion. Timed-out acquisition
cannot pass. A single pilot does not establish pressure coverage, competitive
benefit or a score for the full source benchmark.
Validation uses at most 12 tool turns within each 120-second application budget;
release retains 32 turns within 240 seconds. Reaching the turn cap is a retained
task outcome. Reaching a wall deadline is censored acquisition. All arms receive
the same caps. Terminus command responses allow 1024 tokens: the earlier 512-token
cap repeatedly truncated its JSON command envelopes. Original task validators and
wall budgets remain fixed.

## Environments and inputs

Portable checks need no serving or harness dependencies. Runtime installation is in
[runtime](runtime.md). The Gemma feasibility environment first installs that exact
runtime, then applies the three hash-pinned packages in
[the compatibility override](../benchmarks/runtime-overrides.txt) with
`uv pip install --python artifacts/venvs/runtime/bin/python --no-deps --require-hashes -r benchmarks/runtime-overrides.txt`.
This is a separate environment: do not alter a runtime used by active work.
The override restores the Transformers configuration interface used by the pinned
vLLM. The separately applied, hash-pinned `gemma-turing` patch addresses the attention
kernel's shared-memory limit on Turing. Loading and initial restored/fresh probes
passed, but the first Gemma native-ARC sustained check stopped after a continuation
token mismatch. This remains an unresolved correctness/operating qualification
failure; it does not count as one hour of exposure. Preserve the failed cell and
diagnose it after the existing queue, without replacing its result.

[Harness requirements](../benchmarks/harness-requirements.txt) record the isolated
mini-swe-agent/Harbor/SWE-bench environment. Install into a separate Python 3.12.3
venv with `uv pip install --python artifacts/venvs/harness/bin/python --require-hashes
--no-deps -r benchmarks/harness-requirements.txt`, then `uv pip check --python
artifacts/venvs/harness/bin/python`. A clean installation and dependency check passed locally.
Docker is required for coding tasks; model commands
must execute inside task containers, without host mounts or network access.
C1 image IDs are pinned in benchmarks/workloads.json. Acquisition uses the image
ID, and grading uses a local tag containing the complete image hash; image changes
between acquisition and grading refuse. The grader retains instance images and
does not clean unrelated image caches. Other source assets and validators still
need admission before confirmation.
The C3 Click and HTTPX definitions, revisions, image identities and hidden reference
checks live in [navigation definitions](../benchmarks/workflows/navigation.json).
Their [Dockerfile](../benchmarks/workflows/Dockerfile) uses a pinned base and
[hash-locked dependencies](../benchmarks/workflows/requirements.txt). Build with the
cloned pinned project as `source/`, beside that Dockerfile and requirements file;
retain the final image ID and exact build inputs before admitting a new task. These
are task-specific development images, not serving environments. Do not replace
an admitted image tag or change the reference checks after observing candidate outputs.
Downloaded datasets and weights live outside Git; authored workload definitions,
source ancestry, selection rationale and compact evidence belong in Git.

## Shared-server campaigns

`run --profile compare|release --runtime-python CORE_PY --lmcache-python LMCACHE_PY
--harness-python HARNESS_PY --dataset ASSET_DIRECTORY --model ID=MODEL_PATH
--out artifacts/runs/NEW_DIRECTORY` admits every required workload before starting a GPU.
Repeat `--model` for the pinned model IDs; release also requires `dense-fixture`.
The asset directory contains the source-specific `swebench-c1`, `swebench-c2`,
`navigation-c3`, `terminal-c4`, `branching-c5` and `grounded-b1` directories.
All use retained `test.json` datasets, admitted hashes and content-addressed images.

Current v3 campaigns keep one server per model/comparator/block and reset both GPU
and external cache state between tasks. Each application keeps its own transcript,
requests, grading and deadline. Validation adds a separately accounted pressure,
abandoned-hint and unhinted-return diagnostic to its last application slot.
Release also attaches this diagnostic to LMCache's B1-1 slot on each model/start,
so external pressure coverage does not depend on incidental application traffic.
These four embedded diagnostics are charged separately within the campaign budget;
they do not increase the 180 application / 52 standalone trace counts.
`--case ID` selects a labelled partial acquisition; it cannot qualify the full suite.
The earlier single-dataset runner remains available for old partial reproductions.

`--resume` with the same directory and inputs appends a new acquisition segment;
completed and failed tasks are retained, while never-started work can continue.
Source/model/runtime drift refuses. Resume does not restore a server's lost cache.
`verify-campaign DIRECTORY` checks original observations and full denominators.
Its measurement, release and benefit fields are distinct. Missing operating gates,
fresh confirmation or application evidence block the relevant claim. Neither a
successful panel nor a report with complete measurements implies a policy benefit.
Campaign execution, interruption recovery and release reporting remain under qualification.
A full release first runs the mandatory pinned CPU integration and real GPU
functional cases, then native-only load calibration. Their proof snapshots must match
the campaign. This bounded preparation is reported separately from recurring
acquisition (at most 3 + 10 + 10 minutes plus cleanup).

Each invocation records its owner and admitted launches before acquisition. Recovery
refuses a live owner; after interruption it retains completed receipts and marks
incomplete admitted panels interrupted rather than retrying them. Their full admitted
budgets are charged conservatively. Only never-started work resumes.

Release preparation separately runs bounded native-ARC-only calibration for each
model. Four cold 3072-token/32-output-token requests establish common low/near/overload
arrival intervals at 0.3/0.8 times inverse median service time, followed by a four-request burst. Low/near/burst phases offer two/three/four requests respectively. Those intervals
are frozen before comparator acquisition; every arm receives the same independently
offered requests, with offer/start/completion accounting and drain/recovery. These
small probes characterize fixed-load behavior and cannot establish maximum capacity.

Pressure diagnostics use 1K prefixes so multiple independent prefixes can coexist
inside the small Qwen pool; the resident/restore and offered-load probes retain 3K
prefixes. Preserve earlier capacity-limited negative results. Admission never
replaces natural application traffic after observing candidate outcomes.

Each new campaign records source-group exposure in the asset directory and its own
receipt before acquisition. Previously admitted groups stay exposed even after a
failure; repeating starts does not create independent source tasks. Initial known
pilot exposure lives in workloads.json. Copy this small generated exposure registry
with assets when moving machines; omitting it loses confirmation provenance.
At closeout, retain exposure receipts under tracked `benchmarks/exposure/` before
pruning raw runs or assets. New admission reads both that retained registry and
the active asset registry, so cleanup cannot make an exposed source appear fresh.

Panels record serving-session RSS and whole-device memory/temperature/power/clocks.
Traces retain periodic windows and actual cache-reset responses. Summed RSS includes
shared-page duplication; metadata counts/charged bytes in acknowledgements are the
logical-state observations. Native chunk layout gives a conservative working-set
ratio, separate from asynchronously exported native byte counters. Neither readiness
acknowledgements nor raw cached-token totals alone prove useful retention.

The frozen engineering policy in benchmarks/claims.json uses both-arm task success
for every claimed application, two independent blocks, at least 5% latency reduction
above a conservative measured noise allowance, and max(20 ms, 5%) no-op/unhinted
latency tolerances. These are descriptive engineering gates, not statistical margins.
The report exposes paired regressions, raw protected-cohort delays and all failed
cells; fresh confirmation requires a previously unexposed source group. It never
turns repeated starts into more source tasks or publishes an overall competitor ranking.
Application reports identify their timing scope: mini-swe acquisition excludes
grading; Harbor trial time includes setup, verification and cleanup. Compare only
matched tasks/harnesses. Engine first-token timing is reported separately and is
not client streaming latency.

## Extending and checking

`tools/benchmarking/plan.py` owns schedule counts and allocation. `server.py` owns
serving arguments; each arm for a given model must use the same resource limits.
`probe.py` owns bounded feasibility attempts and verification. `proxy.py` captures
application requests and tool-wait publication. `harness.py` integrates the original
coding harness and grader. `report.py` independently checks capture completeness and
summarizes verified pilot observations. Keep lifecycle failures and quality failures separate.

Use `tools/check.py --test tests/test_benchmark.py` for focused portable feedback,
the runtime lane for adapter/group changes, and `--full` before checkpoints.
New architectures require real restore and failure-path checks; new workloads require
original validators or a documented executable rubric. Update this guide and the
system map when a command becomes usable or its meaning changes. Expensive acquisition
must enforce the reviewed budgets and preserve failed or incomplete attempts.

## Local compatibility patches

`patch-runtime --runtime-python artifacts/venvs/runtime/bin/python lmcache-peer lmcache-copy lmcache-allocator`
applies the LMCache 0.5.6 WSL compatibility repair. `gemma-turing` applies the narrow
Turing attention tile/stage change to the pinned vLLM. The command checks complete
original source hashes, preserves `.benchmark-original` backups and refuses source
that differs. Run only against an idle isolated runtime. Patch definitions are in
`benchmarks/runtime-patches/manifest.json`.

The LMCache benchmark environment opts into `LMCACHE_WSL_VMM=1` and enables PyTorch expandable
segments for KV allocation after loading weights. Permission is granted only to the owned receiving child process. LMCache
uses its existing staged-copy path instead of the failing direct batch-copy path.
This choice and its extra staging cost belong in comparator configuration reporting.
New v4/v5 model probes retain hashes of the relevant installed runtime source files;
older receipts do not retroactively acquire this provenance.

V5 requires an external restoration result for LMCache and reports restoration status.
Earlier generation-status receipts keep their original meaning even where they retain
an additional restoration observation.

## Legacy campaigns

Giving `run` a single dataset file selects the older v1/v2 partial reproduction path.
New work uses the asset directory and shared-server campaigns above. Legacy readers
preserve their original provenance and measurement meaning; they do not gain the
new lifecycle, exposure, resource or load guarantees.

## Regression-test grading

`grade-regression --dataset artifacts/assets/swebench-c2/test.json
--prediction artifacts/runs/submitted.patch --instance INSTANCE --out artifacts/runs/new-grade`
checks that a submission only adds the designated test file. It runs that exact test
in the pinned source image before and after applying the hidden reference fix.
A successful result needs matching nonempty test identities, an assertion failure
before the fix and all tests passing afterward, with no errors or skips. An empty
or out-of-scope submission is an unsuccessful task, not a grader infrastructure
failure. Images run without networking, with one CPU and 2 GiB RAM; grading has a
165-second bound including cleanup.

`verify-regression artifacts/runs/grade` checks retained source/patch hashes and recomputes
the outcome from the JUnit reports. It also independently checks rubric refusals.
Real checker qualification accepted a known Requests regression and rejected a
test that passed on both versions, plus an empty submission. This checks the
instrument; it does not qualify agent performance, broad test usefulness or release
coverage. The first investigation receipt lacks the later retained full dataset;
its original limited provenance remains readable.

## Repository-navigation grading

`grade-navigation --dataset artifacts/assets/navigation-c3/test.json
--prediction artifacts/runs/submitted.patch --instance INSTANCE --out artifacts/runs/new-grade`
requires a single added JSON answer file; source edits and extra answer fields fail
the task. The rubric fixes the relevant symbols and semantics before acquisition.
Matching answers must also pass upstream tests and hidden API behavior checks in a
fresh content-addressed image, without network access. `verify-navigation
artifacts/runs/grade` checks the retained rubric, patch and observations and derives the
outcome again. Known answers for both projects passed the real container checker;
an incorrect precedence answer was rejected. This qualifies those rubric checks,
not broad repository-understanding quality.

## Terminal workflows

`probe-model` with `--coding-kind terminal` uses Harbor 0.25.0 and its host-side
Terminus-2 agent. The task container has no network or host project mounts. The
original Terminal-Bench instruction and tests are retained byte for byte; the
derived task changes its image to a prepared content-addressed image and disables
container networking. All compared systems use the same harness and task limits.
These two tasks produce task-specific outcomes, not an official Terminal-Bench score.

[Terminal definitions](../benchmarks/workflows/terminal.json) pin every source file
and the final images. The [preparation Dockerfile](../benchmarks/workflows/terminal.Dockerfile)
adds tmux and caches the exact pytest invocation used by the original verifier.
Preparation uses networking; acquisition and original verification run offline.
Keep the original image and source task recoverable. The external `terminal-c4`
dataset directory sits beside `harbor/terminal-bench`, matching its relative source
paths. Docker Compose is required; this host uses the checksum-verified v5.6.0
plugin installed using [Docker's documented method](https://docs.docker.com/compose/install/linux/).

For CPU instrument qualification, `terminal --harness-python artifacts/venvs/harness/bin/python
--dataset artifacts/assets/terminal-c4/test.json --instance INSTANCE --agent oracle|nop
--out artifacts/runs/new-attempt` runs the original known solution or no-op control.
`verify-terminal artifacts/runs/attempt` checks retained source, original test observations
and reward agreement. Current v2 receipts also retain the executed driver source;
earlier v1 instrument receipts keep their earlier provenance scope. Both original
solutions passed and the Git no-op control failed as expected. Model acquisition
still requires a live probe. Owned cleanup removes only the trial's containers and
retains image caches. Terminus JSON tool commands have their own conservative causal
hint detection; prose, malformed actions and completed-task messages receive no hint.

## Grounded document workflows

`probe-model --coding-kind grounded` uses the common mini-swe-agent path to search
and read a frozen document corpus. The independent Python queue task and shared-source
RFC comparison are defined in [grounded definitions](../benchmarks/workflows/grounded.json).
Sources are public immutable revisions; their URLs and byte hashes are retained.
The [document image recipe](../benchmarks/workflows/grounded.Dockerfile) supplies
search/read tools and a Git checkout of those exact documents, without host mounts
or network access during acquisition. The external dataset directory contains
`test.json` beside its admitted document files.

`grade-grounded --dataset artifacts/assets/grounded-b1/test.json --prediction artifacts/runs/answer.patch
--instance INSTANCE --out artifacts/runs/new-grade` accepts only a newly added
`benchmark_grounded.json`. The frozen rubric requires correct structured facts and
supporting one-based source line ranges, with at most 12 lines per citation.
`verify-grounded artifacts/runs/grade` derives the outcome from retained document bytes,
facts and citations. The checker rejects correct facts with unrelated citations.
Both known answers passed; this evaluates the named tasks, not broad summary quality
or a general QA benchmark score. Original documents stay in ignored artifacts/assets and
receipts; only authored definitions and provenance are tracked here.

## Branching workflows

`probe-model --coding-kind branching` forks one real Requests issue into a production
repair and an independently authored regression test. Both branches start with the
same actual retrieved source transcript in separate offline containers. The fixed
A/B/A schedule returns to the original patch after competing tool work. Named proxy
sessions keep observations separate; the common context policy applies to every turn.
The original SWE-bench grader evaluates the repair; the regression test must fail
before and pass after the hidden reference fix. `verify-branching DIRECTORY` checks
branch isolation, executed transcripts, patch/input/image identities and both graders.
The live path and required compaction activity remain unqualified.

## Frozen-main feasibility

`pr-feasibility --runtime-python artifacts/venvs/pr-evidence/bin/python
--upstream artifacts/checkouts/vllm-pr-evidence --out artifacts/runs/NEW`
runs the CPU callback reproduction against the frozen-main source. It snapshots
project inputs, compares installed upstream files with the exact checkout, hides
CUDA and uses a bounded owned process. `verify-pr-feasibility artifacts/runs/NEW`
checks retained hashes and observations. The internal `pr-feasibility-native`
operation is used by that wrapper. This diagnoses interface feasibility only;
observer identity is not hold-plugin identity or an efficacy result. See
[the execution boundary](pr-evidence-execution.md). The 2080 Ti screen is not
admitted when neither planned route meets its lifecycle contract.

Add `--private-state` to `pr-feasibility` for the separate concrete-context
probe. It uses real frozen context/state methods with CPU request/scheduler fixtures
to test complete group-key access and return detection, without claiming serving
or hold-policy qualification. The verifier distinguishes its receipt schema.

Add `--hold-policy` instead of `--private-state` for the native hold prototype CPU
probe. It also compares registered, advice-disabled eviction sequences against stock
LRU/ARC. Successful policy-method injection is not qualification of the pending
post-response delivery patch.

The same frozen-main CPU receipt command accepts `--bookkeeping` for the full
ARC/LRU chunk/session cost matrix. It runs with CUDA hidden, retains raw latency
samples and verifies all 18 cells. See [execution details](pr-evidence-execution.md)
for the workload and its adversarial oldest-key limitation.

`pr-feasibility --post-response` requires the separately patched runtime and checks
its delivery hash, native engine dispatch, lifecycle and A3p/A3 identity. It retains
raw CPU delivery-cost samples; this is not network/serving qualification.

Admit the pinned semi-synthetic W1 corpus with `pr-admit-w1 --runtime-python ENV/bin/python
--dataset artifacts/assets/pr-evidence-w1 --model D8=SNAPSHOT [--model HM=SNAPSHOT
--model HS=SNAPSHOT] --out artifacts/runs/NEW`. `verify-pr-w1 DIRECTORY` validates its
receipts, selected rows and stored token episodes. The source definition is
`benchmarks/pr-evidence/w1-source.json`; model revisions and the rendering recipe are
fixed in the admission implementation before any GPU benefit is measured.

`pr-freeze-w1 ADMISSION --out artifacts/runs/NEW` verifies a W1 admission and writes
its compact freeze plus explicit excluded task IDs. The tracked W1 freeze identifies
the full immutable manifest, token data, recipe, source mix and stratum weights.

`pr-feasibility --instrument` runs CPU native-manager identity/cost checks plus real
loopback streaming fault and endpoint checks. `verify-pr-feasibility` checks its
source/log hashes, event journals and stream-derived metrics. This is instrument
qualification, not GPU model admission. Keep all failed and corrected attempts.

`pr-plan-w1 --data ADMISSION/data --model D8 --partition pilot --seed 12101
--arrival-rate RATE --one-shot-rate RATE --horizon SECONDS --out artifacts/runs/FILE.json`
creates a paired content/arrival schedule after verifying the frozen corpus. Rates
and horizon must come from the Stage 2 calibration protocol for real acquisition;
CPU fixture schedules do not freeze performance settings. `pr-client-native
--schedule FILE --arm ARM --url http://127.0.0.1:PORT --out artifacts/runs/NEW`
is the bounded internal streaming client. Starting a GPU server still requires the
hardware admission, authorization, idle check and budget lifecycle; this client
alone does not certify a campaign.


`pr-reconcile-events EVENTS --out artifacts/runs/FILE.json` reconstructs native
CPU residency, evictions and transfer acknowledgements from the common observer.
It refuses inconsistent lifecycle records; an unclosed stream remains explicitly
partial. Request token-source counts additionally require the paired schedule and
client completion state, so this journal command alone is not an efficacy report.


Frozen-main job entry points:

- `pr-gpu-job --campaign artifacts/runs/pr-evidence-2080ti --name UNIQUE
  --runtime-python ENV/bin/python --model-path SNAPSHOT --model D8 --arm A2
  --phase admission --pool-gib 1 --seconds 600` requires a clean source and the
  committed Stage 1 go record. Screen jobs add `--phase screen --schedule FILE`.
  The command enforces the authorized device, fresh idle observation, model/runtime
  identities and cumulative admission/screen budgets. It keeps all attempts and
  refuses to pass an unfinished prior attempt.
- `verify-pr-gpu-job DIRECTORY` validates the immutable receipt, including failed
  attempts; receipt integrity alone does not establish model admission.
- `verify-pr-model-admission DIRECTORY` additionally checks poisoned-destination
  restored bytes in every group and identical deterministic continuation at the
  same actual cached boundary. Fresh-versus-cached differences remain diagnostic.
- `pr-measure-cell DIRECTORY --out artifacts/runs/FILE.json` joins the paired W1
  schedule, raw streaming captures and native events. It retains failed sessions,
  unoffered turns, censored latency and unknown cache-source counts.
- `pr-compare-cells --treatment METRICS --baseline METRICS --cohort hinted_return
  --metric ttft_ns --contrast benefit --label MODEL-WORKLOAD-STATISTIC --out FILE`
  accepts repeated ordered treatment/baseline pairs. It refuses duplicate seeds or
  mixed model/partition inputs, recomputes the population-weighted pooled statistic
  inside 10,000 paired seed resamples, and reports BCa plus clustered delta intervals.
  Other metrics are `client_itl_ns` and `recompute`; other contrasts are
  `relative_harm` and `absolute_change`. Undefined/censored/degenerate evidence stays
  unqualified. Raw counts, unweighted and weighted complete-response descriptions
  retain failure counts alongside inference. Use one deterministic label per
  reported model/workload/statistic.
- `pr-route-difference --post METRICS --ttl METRICS --baseline METRICS --label LABEL
  --out FILE` takes the same repeated paired seeds and reports post-response benefit
  minus TTL benefit, using their common baseline. This whole-route contrast is
  descriptive and cannot isolate the delivery hook.
- `pr-calibration INPUT --step sizing|pools|baseline|quote --out FILE` calculates the
  frozen sizing settings, pool ranking, better baseline or complete-block quote.
  Sizing takes the three A2 metric records from the 240-second/no-one-shot pass;
  baseline selection takes all 21 calibration cells. Pool inputs are
  `working_sets_bytes` (three observations) and `largest_conversation_bytes` from
  admitted native geometry. Quote inputs are `horizon_seconds`,
  `startup_cleanup_seconds` and `remaining_screen_seconds`. These calculations do
  not replace receipt/resource validation or the committed calibration freeze.
- `pr-screen-verdict EVIDENCE --out FILE` applies the frozen per-route/model/workload
  rule to `pr-screen-decision-input-v1`: benefit, recomputation, five harm and A/A
  estimates, opportunity and explicit validity/identity/diagnosis qualifications.
  Estimates contain `point`, `interval` and `delta_interval`; missing values are
  null. Blocking reasons take precedence, followed by efficacy STOP, CONTINUE and
  INCONCLUSIVE. Interval-method threshold disagreement is INCONCLUSIVE. This is
  decision arithmetic, not automatic certification of supplied qualifications;
  review the linked evidence before adopting its conclusion.

The admission driver uses a fixed 4,097-token prefix and 32 generated tokens, with
prime, GPU-resident and CPU-restored requests. The synchronous byte detector is
admission-only. Performance cells use normal asynchronous native copies.
