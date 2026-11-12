# Benchmark implementation

The benchmark suite is under implementation in [the active task](../tasks/benchmark-design.md).
Schedule counts are implemented; compare and release acquisition are not yet qualified.
Do not use a successful model probe as release or performance evidence.

The maintained entry point is `uv run --locked python -m tools.benchmark`:

- `functional --runtime-python /external/venv/bin/python --model /external/dense-model
  --out /external/new-attempt --timeout 600` runs the existing dense GPU gate through
  the shared benchmark entry point. Its receipt uses the runtime verifier format;
  verify it with `python -m tools.runtime_verify --verify /external/attempt`.
- `models` prints immutable model candidates and their qualification state.
- `plan --profile functional|compare|release` prints the allocated cases, repetitions
  and budget. The output explicitly says `unqualified_schedule`.
- `probe-model --runtime-python /external/venv/bin/python --model /external/model
  --out /external/new-attempt --system gpu-only|native-arc|wait-hint|lmcache --timeout 900`
  launches one localhost server, checks generation, and for CPU-offload modes tests
  GPU-only displacement and restored versus fresh token IDs. Native ARC and wait-hint
  also require registered readiness. New LMCache probes require external restoration;
  older generation-only receipts remain limited to generation. LMCache on the current
  WSL host has successful Qwen and Gemma restoration probes using the opt-in
  VMM/staged-copy repair, plus one graded Qwen coding pilot. Broader application
  and reliability qualification remains pending.
  See the active benchmark task for evidence.
- `verify-probe /external/attempt` checks a v3/v4/v5/v6 receipt, frozen source/model hashes,
  retained artifacts, token equality and original grader outcomes without GPU work. Earlier investigation
  attempts retain their original schemas; this reader does not silently upgrade them.

- `report-probe /external/attempt` verifies first, then summarizes task quality,
  request counts, context compaction, hint activity and timing. Missing request captures
  refuse. It makes no comparative performance or release claim.
- `grade-coding --harness-python /external/harness/bin/python --dataset /external/test.json
  --predictions /external/preds.json --instance INSTANCE --out /external/new-grade`
  grades an existing prediction without rerunning generation. It creates a new receipt
  linked to the original dataset and prediction hashes; never rewrite the original run.

A coding pilot adds `--coding-dataset /external/dataset/test.json`,
`--harness-python /external/harness/bin/python` and `--instance INSTANCE_ID`.
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
`uv pip install --python /external/venv/bin/python --no-deps --require-hashes -r benchmarks/runtime-overrides.txt`.
This is a separate environment: do not alter a runtime used by active work.
The override restores the Transformers configuration interface used by the pinned
vLLM; it does not fix the Gemma attention kernel limitation on Turing.

[Harness requirements](../benchmarks/harness-requirements.txt) record the isolated
mini-swe-agent/Harbor/SWE-bench environment. Install into a separate Python 3.12.3
venv with `uv pip install --python /external/harness/bin/python --require-hashes
--no-deps -r benchmarks/harness-requirements.txt`, then `uv pip check --python
/external/harness/bin/python`. A clean installation and dependency check passed locally.
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
--out NEW_EXTERNAL_DIRECTORY` admits every required workload before starting a GPU.
Repeat `--model` for the pinned model IDs; release also requires `dense-fixture`.
The asset directory contains the source-specific `swebench-c1`, `swebench-c2`,
`navigation-c3`, `terminal-c4`, `branching-c5` and `grounded-b1` directories.
All use retained `test.json` datasets, admitted hashes and content-addressed images.

Current v3 campaigns keep one server per model/comparator/block and reset both GPU
and external cache state between tasks. Each application keeps its own transcript,
requests, grading and deadline. Validation adds a separately accounted pressure,
abandoned-hint and unhinted-return diagnostic to its last application slot.
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

`patch-runtime --runtime-python /external/venv/bin/python lmcache-peer lmcache-copy lmcache-allocator`
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

`grade-regression --dataset /external/swebench-c2/test.json
--prediction /external/submitted.patch --instance INSTANCE --out /external/new-grade`
checks that a submission only adds the designated test file. It runs that exact test
in the pinned source image before and after applying the hidden reference fix.
A successful result needs matching nonempty test identities, an assertion failure
before the fix and all tests passing afterward, with no errors or skips. An empty
or out-of-scope submission is an unsuccessful task, not a grader infrastructure
failure. Images run without networking, with one CPU and 2 GiB RAM; grading has a
165-second bound including cleanup.

`verify-regression /external/grade` checks retained source/patch hashes and recomputes
the outcome from the JUnit reports. It also independently checks rubric refusals.
Real checker qualification accepted a known Requests regression and rejected a
test that passed on both versions, plus an empty submission. This checks the
instrument; it does not qualify agent performance, broad test usefulness or release
coverage. The first investigation receipt lacks the later retained full dataset;
its original limited provenance remains readable.

## Repository-navigation grading

`grade-navigation --dataset /external/navigation-c3/test.json
--prediction /external/submitted.patch --instance INSTANCE --out /external/new-grade`
requires a single added JSON answer file; source edits and extra answer fields fail
the task. The rubric fixes the relevant symbols and semantics before acquisition.
Matching answers must also pass upstream tests and hidden API behavior checks in a
fresh content-addressed image, without network access. `verify-navigation
/external/grade` checks the retained rubric, patch and observations and derives the
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

For CPU instrument qualification, `terminal --harness-python /external/harness/bin/python
--dataset /external/terminal-c4/test.json --instance INSTANCE --agent oracle|nop
--out /external/new-attempt` runs the original known solution or no-op control.
`verify-terminal /external/attempt` checks retained source, original test observations
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

`grade-grounded --dataset /external/grounded-b1/test.json --prediction /external/answer.patch
--instance INSTANCE --out /external/new-grade` accepts only a newly added
`benchmark_grounded.json`. The frozen rubric requires correct structured facts and
supporting one-based source line ranges, with at most 12 lines per citation.
`verify-grounded /external/grade` derives the outcome from retained document bytes,
facts and citations. The checker rejects correct facts with unrelated citations.
Both known answers passed; this evaluates the named tasks, not broad summary quality
or a general QA benchmark score. Original documents stay in external assets and
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
