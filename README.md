# vllm-semantic-cache-ext-next

Exact CPU KV offload with causal wait-hint ARC recency advice. Semantic ranking
and GPU prefetch stay in the historical repository. Advice only reorders ready
native CPU residents; exact native block keys remain the reuse boundary.

The minimal port includes typed hint contracts, bounded session state, a native
vLLM adapter, authenticated HTTP publication and separate portable/runtime/hardware
checks. Benchmark redesign and high-level agent operations remain deferred.
Functional qualification is recorded in [FINDINGS](FINDINGS.md); it is not evidence
of a speedup, answer quality or support for another GPU/model/runtime.

## Development

Use uv **0.13.6** and CPython **3.12.3**:

```sh
uv sync --locked
uv run --locked python tools/check.py
uv run --locked python tools/check.py --test tests/test_advice.py
```

The full portable check includes deliberate bad/control specimens and hides CUDA.
The targeted command reports partial coverage. After changing test discovery, use
`--update-tests` and review its diff. See [checks](docs/checks.md).

Install the independent hash-locked runtime only when needed:

```sh
uv venv --python 3.12.3 /external/venvs/wait-hint
uv pip install --python /external/venvs/wait-hint/bin/python --require-hashes -r runtime/requirements.txt
uv pip install --python /external/venvs/wait-hint/bin/python --no-deps -e .
uv run --locked python tools/check.py --runtime-python /external/venvs/wait-hint/bin/python --out /external/new-cpu-attempt
```

This runs strict runtime types and real pinned CPU integration. It snapshots
source and retains logs and receipts. It never acquires or uses a GPU.
[Runtime instructions](docs/runtime.md) explain the supported serving configuration,
HTTP contract and separately authorized hardware gate. No GPU reservation machinery
is included here. The optional full portable commit hook is enabled with
`git config --local core.hooksPath .githooks`; setup does not enable it silently.

## Navigation

- [System map](docs/system-map.md), [architecture and protocol](docs/source-port.md).
- [Port plan and disposition](docs/porting-overhaul.md).
- [Research instructions](AGENTS.md), [findings](FINDINGS.md).
- [Historical failure analysis](docs/issue-analysis.md), [decisions](docs/decisions/README.md).
- [Deferred benchmark interface](docs/benchmarking.md).

Track code, tests, workflow definitions and compact findings here. Raw attempts,
execution snapshots and retained wheels live outside Git. The historical repo
remains intact. This checkout has no remote; no public release is authorized.
