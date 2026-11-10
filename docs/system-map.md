# System map

| Flow | Command | Target | Availability |
| --- | --- | --- | --- |
| CPU checks | `uv run --locked python tools/check.py` | `tools/check.py:main` | implemented |
| Runtime verify | `uv run --locked python -m tools.runtime_verify` | `tools/runtime_verify.py:main` | implemented; explicit runtime and output required |
| Benchmarks | `uv run --locked python -m benchmarks` | `benchmarks/__main__.py` | future; absent |

The check enforces these rows against `tools/system_map.py` and the actual targets.
See [checks](checks.md) for check modes and [runtime](runtime.md) for the runtime lane.

Product flow: HTTP `/v1/wait-hints` → endpoint authentication and schema validation →
engine utility RPC → scheduler-owned observation validation → ready native ARC queue
reordering. Registration arrives through `kv_transfer_params.wait_hint` on an
ordinary completed request. Native workers own transfer bytes; the adapter reports
completion and failure to the native scheduler. See [architecture](architecture.md).

The benchmark entry is not implemented yet.
