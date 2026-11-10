# System map

| Flow | Command | Target | Availability |
| --- | --- | --- | --- |
| CPU checks | `uv run --locked python tools/check.py` | `tools/check.py:main` | implemented |
| Runtime verify | `uv run --locked python -m tools.runtime_verify` | `tools/runtime_verify.py:main` | implemented; explicit runtime and output required |
| Benchmarks | `uv run --locked python -m benchmarks` | `benchmarks/__main__.py` | future; absent |

Portable checks never qualify runtime or GPU behavior. Target a portable test with
`--test tests/PATH.py`; `--update-tests` explicitly refreshes its discovery manifest.

The supported full runtime CPU command is
`uv run --locked python tools/check.py --runtime-python /external/venv/bin/python --out /external/new-attempt`.
It runs strict adapter/test types against the installed pin and then isolated
native CPU integration. No-argument runtime verification exits 2, asking for the
explicit environment/output; this is an interface check, counted as skipped in
the portable aggregate. `--verify /external/attempt` checks retained functional
receipts without rerunning hardware. See [runtime](runtime.md).

Product flow: HTTP `/v1/wait-hints` → endpoint authentication/schema validation →
engine utility RPC → scheduler-owned observation validation → ready native ARC
queue reordering. Registration arrives through `kv_transfer_params.wait_hint` on
an ordinary completed request. Native workers own transfer bytes; the adapter
reports completion/failure to the native scheduler. See [source port](source-port.md).

The benchmark entry remains absent. Future workload/policy discovery uses suffixes,
without introducing an extension framework merely for this port. No skills,
benchmark runner, scheduler or GPU reservation service has been added.
