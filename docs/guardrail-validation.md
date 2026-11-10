# Guardrail validation, 2026-11-09

Implementation revision: `5cda8eccc9a6e7e9f5f9a238dc29fb59e86e7b5f`. Local-only work on DESKTOP-T4JKGJB;
temporary project name retained. The following documentation-only evidence commit
does not alter the validated tools or contracts.

## Final host runs

| Command | Passed groups | Skipped groups | Failed groups | Validator wall time | Pytest passed |
| --- | ---: | ---: | ---: | ---: | ---: |
| `uv run --locked python tools/check.py` | 9 | 2 | 0 | 44.66 s | 151 |
| `uv run --locked python tools/check.py --self-test` | 1 | 0 | 0 | 43.71 s | 140 |

Both commands exited 0. No pytest tests skipped or failed. Native and runtime
qualification are the two explicit aggregate skips; neither counts as passed.
The standalone `uv run --locked python -m tools.runtime_verify` printed the
not-implemented-until-source-port message and exited 2.

Each final run printed 133 named bad-specimen rejections with accepted positive
controls. Relative to the incoming foundation, 61 bad specimens and 61 paired
positive controls were added (133 pairs total). The registry, specimens and
fixed collection manifests are committed. Seven self-test integrity controls and
eleven direct CPU/contract tests account for the other collected tests. The
exceptions file has 15 reasoned entries; AGENTS remains 42 lines.

## Development failures that informed the final checks

Comment removal initially exposed an import-spacing lint failure. The isolated
runtime proof initially resolved the real repository's stub, accidentally accepting
mutated stubs. Giving the specimen its own tools package, controlling checker
import precedence and copying the pytest gate fixed that isolation error; both
runtime mutations then failed for the named contract diagnostic.

An import-time NewType factory violated the existing inert-contract boundary;
a nominal scalar class preserves type separation without weakening that boundary.
Missing test directories and contract tests outside the development root caused
Vulture/deptry positive-control failures. Fixtures now use the real tests layout.
Identity metadata is read directly in its constructor control, avoiding unused
field exceptions that depended on unrelated names elsewhere in Vulture's scan.
The final complete commands above passed after these repairs.

## Deferred scope and decisions

Full GPU runtime verification, reference-count mutation ownership enforcement,
CUDA timing/synchronization enforcement and full serving correctness/quality tests
require actual runtime/measurement modules and separately authorized hardware work.
Filename conventions are checked, but no discovery loader, adapter or benchmark
runner is implemented. Skills, blind evaluations and an automated recurring-
correction detector remain deferred as requested. Result-path ownership is a
static tripwire, with its limits recorded in decision 005.

No new decision blocks this stage. Philip reviews future exceptions and check
configuration changes. Existing future decisions about external evidence storage
and confirmation margins remain in the benchmark protocol. Docker/hosted CI and
GPU execution were not run or qualified by this host validation.

The old repository was untouched. FINDINGS and the CLAUDE pointer are unchanged.
No remote was added or used, no push was attempted, no GPU job ran, and no runtime,
CUDA or benchmark implementation was ported. Disposable proof repositories were
removed by their fixture; reusable validator caches were preserved.

## Changed paths

38 paths across this task, including this record:

- [.github/workflows/checks.yml](../.github/workflows/checks.yml)
- [.gitignore](../.gitignore)
- [AGENTS.md](../AGENTS.md)
- [README.md](../README.md)
- [benchmarks/__init__.py](../benchmarks/__init__.py)
- [docs/benchmarking.md](../docs/benchmarking.md)
- [docs/checks.md](../docs/checks.md)
- [docs/decisions/005-agent-guardrails.md](../docs/decisions/005-agent-guardrails.md)
- [docs/decisions/README.md](../docs/decisions/README.md)
- [docs/guardrail-validation.md](../docs/guardrail-validation.md)
- [docs/runtime.md](../docs/runtime.md)
- [docs/system-map.md](../docs/system-map.md)
- [pyproject.toml](../pyproject.toml)
- [semantic_offload/__init__.py](../wait_hint_offload/__init__.py)
- [semantic_offload/contracts.py](../wait_hint_offload/contracts.py)
- [semantic_offload/vllm_adapter/__init__.py](../wait_hint_offload/vllm_adapter/__init__.py)
- [tests/__init__.py](../tests/__init__.py)
- [tests/test_check_cases.py](../tests/test_check_cases.py)
- [tests/test_contracts.py](../tests/test_contracts.py)
- [tests/test_cpu.py](../tests/test_cpu.py)
- [tools/__init__.py](../tools/__init__.py)
- [tools/agent_policy.py](../tools/agent_policy.py)
- [tools/bootstrap-uv.txt](../tools/bootstrap-uv.txt)
- [tools/check-cases.json](../tools/check-cases.json)
- [tools/check.py](../tools/check.py)
- [tools/config.py](../tools/config.py)
- [tools/cpu-tests.json](../tools/cpu-tests.json)
- [tools/exceptions.toml](../tools/exceptions.toml)
- [tools/native.py](../tools/native.py)
- [tools/pytest_gate.py](../tools/pytest_gate.py)
- [tools/repo_policy.py](../tools/repo_policy.py)
- [tools/rules.json](../tools/rules.json)
- [tools/runtime_policy.py](../tools/runtime_policy.py)
- [tools/runtime_verify.py](../tools/runtime_verify.py)
- [tools/self-tests.json](../tools/self-tests.json)
- [tools/system_map.py](../tools/system_map.py)
- [tools/unused.py](../tools/unused.py)
- [uv.lock](../uv.lock)
