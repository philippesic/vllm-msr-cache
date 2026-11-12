# Benchmark implementation

The benchmark suite is under implementation in [the active task](../tasks/benchmark-design.md).
Schedule counts are implemented; compare and release acquisition are not yet qualified.
Do not use a successful model probe as release or performance evidence.

The maintained entry point is `uv run --locked python -m tools.benchmark`:

- `models` prints immutable model candidates and their qualification state.
- `plan --profile functional|compare|release` prints the allocated cases, repetitions
  and budget. The output explicitly says `unqualified_schedule`.
- `probe-model --runtime-python /external/venv/bin/python --model /external/model
  --out /external/new-attempt --system gpu-only|native-arc|wait-hint --timeout 900`
  launches one localhost server, checks generation, and for CPU-offload modes tests
  registered readiness, GPU-only displacement and restored versus fresh token IDs.
- `verify-probe /external/attempt` checks a v3 receipt, frozen source/model hashes,
  retained artifacts and observed outcomes without GPU work. Earlier investigation
  attempts retain their original schemas; this reader does not silently upgrade them.

A coding pilot adds `--coding-dataset /external/dataset/test.json`,
`--harness-python /external/harness/bin/python` and `--instance INSTANCE_ID`.
It executes mini-swe-agent in a disposable Docker container, captures every completed
model request and causal tool-wait hint, then invokes SWE-bench's original grader.
Resolved/unresolved task quality is distinct from whether acquisition and grading
completed. Per-request timing is engine timing, not client streaming TTFT; the
application uses non-streaming responses to retain registration acknowledgements.
A single pilot does not establish pressure coverage, competitive benefit or a score
for the full source benchmark.

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
mini-swe-agent/Harbor/SWE-bench environment. Full hash locking and clean installation
qualification remain pending. Docker is required for coding tasks; model commands
must execute inside task containers, without host mounts or network access.
Images, original dataset records and graders must be pinned before confirmation.
Downloaded datasets and weights live outside Git; authored workload definitions,
source ancestry, selection rationale and compact evidence belong in Git.

## Extending and checking

`tools/benchmarking/plan.py` owns schedule counts and allocation. `server.py` owns
serving arguments; each arm for a given model must use the same resource limits.
`probe.py` owns bounded feasibility attempts and verification. `proxy.py` captures
application requests and tool-wait publication. `harness.py` integrates the original
coding harness and grader. Keep lifecycle failures and quality failures separate.

Use `tools/check.py --test tests/test_benchmark.py` for focused portable feedback,
the runtime lane for adapter/group changes, and `--full` before checkpoints.
New architectures require real restore and failure-path checks; new workloads require
original validators or a documented executable rubric. Update this guide and the
system map when a command becomes usable or its meaning changes. Expensive acquisition
must enforce the reviewed budgets and preserve failed or incomplete attempts.
