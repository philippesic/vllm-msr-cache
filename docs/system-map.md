# System map

| Flow | Command | Target | Availability |
| --- | --- | --- | --- |
| Benchmark qualification | `uv run --locked python -m tools.benchmark` | `tools/benchmark.py:main` | model/coding probes implemented; comparison campaigns unqualified |
| CPU checks | `uv run --locked python tools/check.py` | `tools/check.py:main` | implemented |
| Runtime verify | `uv run --locked python -m tools.runtime_verify` | `tools/runtime_verify.py:main` | implemented; explicit runtime and output required |
The check enforces these rows against `tools/system_map.py` and the actual targets.
See [checks](checks.md) for check modes and [runtime](runtime.md) for the runtime lane. See [benchmarks](benchmarks.md) for implementation status and model/coding probes.

Product flow: HTTP `/v1/wait-hints` → endpoint authentication and schema validation →
engine utility RPC → scheduler-owned observation validation → ready native ARC queue
reordering. Registration arrives through `kv_transfer_params.wait_hint` on an
ordinary completed request. Native workers own transfer bytes; the adapter reports
completion and failure to the native scheduler. See [architecture](architecture.md).

## Maintenance

When adding, renaming, moving or retiring a supported entry point, update
`tools/system_map.py` and this table together. Keep commands, target symbols and
availability accurate; planned commands stay in the active task until implemented.
Update affected README/topic links and examples in the same change. Internal helpers
need no table row. When the product's routing or ownership changes, update the flow
above and its explanation in architecture.md.

Exercise the affected command through its documented entry point within the task's
authorization, then run the normal checks. The map check validates listed entries;
the author must also check for missing or obsolete entries. Do not launch a GPU run
merely to maintain this document; report any qualification that remains pending.
