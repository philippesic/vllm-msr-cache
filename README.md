# wait-hint-kv

Exact CPU KV offload for vLLM with causal wait-hint advice. A completed request can
register its prefix; a later hint protects that prefix's ready CPU blocks in vLLM's
native ARC cache until the session returns, and lets keys the return can never use
go first. Running with advice disabled is the matched control: same code, same
safety guards, no advice.

What has been tested, and its limits, is in [FINDINGS](FINDINGS.md).

## Development

Use uv **0.13.6** and CPython **3.12.3**:

```sh
uv sync --locked
git config core.hooksPath .githooks
uv run --locked python tools/check.py
uv run --locked python tools/check.py --full
```

The plain check is the fast lane for use while working. `--full` adds the rule
proofs; it is acceptance and what the pre-commit hook runs. After adding or renaming tests, run
`--update-tests` and review the printed diff. See [checks](docs/checks.md).

The pinned vLLM runtime is a separate, hash-locked environment used for integration
tests and serving. Setup and commands are in [runtime](docs/runtime.md). It never
uses a GPU unless explicitly requested.

## Navigation

- [Architecture and hint protocol](docs/architecture.md)
- [Runtime, serving and GPU gate](docs/runtime.md)
- [Checks](docs/checks.md)
- [System map](docs/system-map.md)
- [Findings](FINDINGS.md), [result format](docs/results.md) and [recovery](docs/recovery.md)
- [Open tasks and handoff format](tasks/README.md)
- [Agent working rules](AGENTS.md)

Temporary runs and execution snapshots live under ignored `artifacts/` until curated
into tracked results and pruned. The public upstream is
[`philippesic/wait-hint-kv`](https://github.com/philippesic/wait-hint-kv).
The grafted main retains the original project history; `next-pre-graft-main` preserves
the original NEXT history. Other local archive refs remain local. See
[recovery](docs/recovery.md) for retention and verification.

Benchmark implementation and bounded feasibility commands: [benchmark guide](docs/benchmarks.md). Comparison campaigns remain unqualified.
