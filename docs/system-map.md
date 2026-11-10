# System map

| Flow | Command | Target | Availability |
| --- | --- | --- | --- |
| CPU checks | `uv run --locked python tools/check.py` | `tools/check.py:main` | implemented |
| Runtime verify | `uv run --locked python -m tools.runtime_verify` | `tools/runtime_verify.py:main` | unavailable; exits 2 |
| Benchmarks | `uv run --locked python -m benchmarks` | `benchmarks/__main__.py` | future; absent |

The CPU command runs formatting, lint, strict types, imports, repository policy,
Vulture, deptry, native/runtime availability gates and CPU tests including all
bad specimens and positive controls. It has no GPU or serving side effects.
`uv run --locked python tools/check.py --self-test` runs just the rule proofs.
See [checks](checks.md) for exact scope and skip accounting.

The runtime command prints “not implemented until source port” and exits 2.
The aggregate verifies this stub behavior and reports a skip. Runtime execution,
building a binary and completed research remain separate claims; see
[runtime](runtime.md). The map check fails on a missing entry function, a removed
command row, an availability change or an unqualified benchmark entry appearing.

The future benchmark command has plan/run/verify/report modes. It will own result
records and process cleanup, writing only to an explicitly external run root.
Definitions will be isolated `benchmarks/workloads/*_workload.py` files. Future
retention policies use `semantic_offload/policies/*_policy.py`. Discovery uses
filename suffixes, with no name-hardcoded registries or per-host launch scripts.
No discovery loader, extension implementation or benchmark runner exists now.
See [benchmark protocol](benchmarking.md) and
[guardrail decisions](decisions/005-agent-guardrails.md).
