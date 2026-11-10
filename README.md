# vllm-cache-next

A fresh foundation for research on **exact KV reuse** with CPU offload and
causal tool-wait advice, compared with matched native ARC. Semantic scores rank
retention or prefetch candidates; they never replace answers or permit approximate
KV reuse. The temporary project name does not imply a public release.

This repository contains packaging, CPU validation and source-linked documentation.
There is no cache runtime, CUDA extension or benchmark harness yet. It installs
no vLLM, torch or CUDA dependency and makes no hardware support claim.

## Install and check

Use uv **0.13.6** and CPython **3.12.3** (Linux x86-64 is the verified host).
With Python/pip available, the hash-pinned bootstrap is optional:

```sh
python3 -m pip install --require-hashes -r tools/bootstrap-uv.txt
uv python install 3.12.3
uv sync --locked
uv run --locked python tools/check.py
```

The validator checks without editing files. It runs format, lint, strict typing,
import boundaries, repository policy, a separately reported native group, and
CPU pytest including deliberate-violation proofs. Native currently reports
`skipped: no native sources yet`. [Checks and proofs](docs/checks.md) explain
exact coverage and limitations. CI runs the identical command in the pinned
[CPU image](Dockerfile.checks); the same image can be built locally with Docker.

The optional pre-commit hook runs that same command (no GPU/build work):

```sh
git config --local core.hooksPath .githooks
```

It is not enabled automatically. Typical timing is recorded in the handoff;
this hook includes the proofs rather than maintaining a weaker check path.

## Navigation

- [Research instructions](AGENTS.md) and [durable findings](FINDINGS.md).
- [Runtime assumptions](docs/runtime.md): distinct historical environments and GPUs.
- [Benchmark protocol](docs/benchmarking.md): future invocation and comparison rules.
- [Decisions](docs/decisions/README.md): boundaries, default candidate and evidence policy.

Raw runs, execution snapshots, transient plans and scratch live outside Git.
The old repository remains read-only evidence. No remote is configured; no
license has been selected for publication and no distribution is authorized.
