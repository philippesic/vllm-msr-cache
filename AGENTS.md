# Purpose

Port exact CPU KV offload with causal wait-hint CPU ARC recency advice.
Semantic ranking and GPU prefetch stay in the historical repository.
A hint never pins KV, manufactures readiness or authorizes approximate reuse.
Use the same runtime with advice disabled as the matched functional control.

# Work and validation

Follow [the port plan](docs/porting-overhaul.md) within the authorized task's scope
and stopping conditions. Proceed through routine reversible repairs autonomously.
Use a separate worktree for parallel implementation and integrate serially.

Run `uv run --locked python tools/check.py` for full portable acceptance.
Use `--test tests/PATH.py` for explicitly partial feedback.
Use `--update-tests` explicitly after adding tests; review additions/removals.
Runtime and hardware qualification remain distinct from portable checks.
See [system map](docs/system-map.md) and [checks](docs/checks.md).

No code comments or commentary docstrings. Keep exactly one root AGENTS.md.
Do not add skills or GPU reservation machinery. GPU execution needs explicit
user authorization and externally established ownership of the host/device.

Changes weakening guarantees or adding substantive exceptions need Philip's
review unless the task explicitly authorizes them. Routine strengthening and
mechanical test-manifest maintenance proceed within scope.

# Evidence and communication

Use [FINDINGS](FINDINGS.md) for scoped conclusions with immutable evidence.
Preserve failed, adverse and incomplete attempts. Report actual mechanism,
executed versus reused checks, unavailable observations and qualification scope.
A healthy process, passing CPU checks and useful research are separate facts.
Keep the historical repository and sealed runs intact.
Benchmark methodology and high-level agent features are deferred.
