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
- `verify-probe /external/attempt` checks a v3/v4/v5 receipt, frozen source/model hashes,
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
Downloaded datasets and weights live outside Git; authored workload definitions,
source ancestry, selection rationale and compact evidence belong in Git.

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

## Campaign acquisition under implementation

`run --profile release --case C1-1 --model qwen35-4b-int4=/external/qwen
--model gemma4-12b-int4=/external/gemma --runtime-python /external/core/bin/python
--lmcache-python /external/lmcache/bin/python --harness-python /external/harness/bin/python
--dataset /external/swebench-c1/test.json --out /external/new-campaign` executes the
selected C1 case across the scheduled arms, models and repetitions. Explicit case
selection always produces a partial campaign. `verify-campaign /external/campaign`
checks coverage, identities and each child receipt; failed cells stay in coverage.
V2 campaigns freeze task image identities across all cells; older v1 receipts retain
their earlier provenance scope.
Source changes during acquisition refuse further cells. Use a stable checkpoint.
C1 repair and C2 test-authoring acquisition are implemented. A partial C2 command
uses --case C2-1 and its admitted dataset. Mixed-dataset campaigns and full
compare/release commands still refuse missing integration, workloads and traces. Resume and release verdicts remain unimplemented.
This command is being qualified and does not yet satisfy the complete suite contract.

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
