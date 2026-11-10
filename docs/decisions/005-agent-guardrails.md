# 005: Agent guardrails

Accepted for the non-source stage. Based on the garden audit summarized in the
approved task: contradictory status writers, ad-hoc launchers, silent fallbacks,
unused generated programs, interpreter drift and unreasoned suppressions.
Historical evidence remains indexed in [004](004-evidence.md) and
[FINDINGS](../../FINDINGS.md); no runtime completion follows from this decision.

## G005-COMMENTS

Philip's correction replaces the proposed middle ground with a full ban on code
comments and commentary docstrings. Python string expression statements are also
rejected, including attribute and function docstrings. Shebangs, encoding headers
and legally needed license headers are the only header allowances. Tool directives
need exact path/text membership and a reason in
[exceptions.toml](../../tools/exceptions.toml). No broad noqa is implied.
Derivations, mathematics and assumptions belong in decisions or FINDINGS with a
stable ID, referenced through a named constant or test name when useful.

The removed foundation comments described inert package placeholders, native
fail-closed dispatch, the pytest collection/report gate, script bootstrap and
AST scanning of tokens instead of specimen strings. Those explanations are now
recorded here and in [checks](../checks.md). The pytest gate changes a skipped
report before pytest counts outcomes. The specimen registry is independent of
the specimen list so deleting a case cannot quietly reduce coverage.

Exceptions are exact and reasoned. Missing files, unknown rules, duplicate entries,
empty reasons and unused directives fail. Philip reviews exceptions and check
configuration changes; this task authorizes the initial implementation and entries.

## G005-LAYOUT

Future extension definitions use isolated `benchmarks/workloads/*_workload.py`
and `semantic_offload/policies/*_policy.py` modules, discovered by suffix. There
are no adapters, discovery loader or registry implementation in this stage.
Static named registries, dictionaries of local functions and hardcoded imports
into those extension modules are rejected. A future loader needs a separately reviewed,
path-specific dynamic-import allowance and its own discovery behavior proofs.
New top-level Python files, executable shell files, main guards and import-time
script calls require exact entry membership. Module helpers remain ordinary
isolated files. This prevents repeating the old launcher proliferation.

## G005-OWNERSHIP

Only the future `benchmarks/runner.py`, reached through `python -m benchmarks`,
will write durable result records. Reporting reads those records and derives
views; it cannot rewrite outcomes or maintain a second status ledger. Result,
results and summary JSON path literals in other library modules fail a cheap
path check. This is a tripwire, not proof against dynamically constructed paths
or arbitrary filesystem APIs. The runner is not implemented or authorized here.
Future CPU/GPU refcounts and job identities must each have one owning component;
readers receive immutable snapshots/events. Enforce ownership with AST checks
when actual modules establish the writer and mutation APIs.

## G005-FOOTGUNS

Bare excepts fail everywhere. Library packages cannot print. Environment access
is centralized in `tools/config.py` (future runtime config: `semantic_offload/config.py`).
Tests cannot use time.sleep. Import aliases and simple assignment aliases are
resolved for these checks; arbitrary reflection and runtime rebinding are not.
Benchmarks use local `random.Random`, `numpy.random.default_rng` or `RandomState`
with explicit integer literal seeds; global RNG calls and implicit/None seeds
fail. Torch random calls remain blocked pending a seeded measurement module.
These static constraints do not establish determinism of a future workload.
CUDA synchronization/timing belongs in that future measurement module, where
CUDA events and completion fences must be qualified with actual execution.
No CUDA timing check body exists before that module.

## G005-UNUSED

Vulture runs at 60% confidence across library, tools and tests. Exact externally
consumed pytest hooks/report fields are reasoned exceptions; stale entries fail.
Deptry checks missing, unused, transitive and development-only dependencies.
The tools and tests roots are explicit development-root exceptions; serving and
benchmark imports cannot use that exemption. Dev CLI dependencies need not have
Python imports, so deptry's development-dependency behavior is preserved. Neither
tool proves a public API has no external caller. No runtime API is removed here.

## G005-RUNTIME

`uv run --locked python -m tools.runtime_verify` exits 2 with an exact unavailable
message. The aggregate checks that stub contract and reports runtime as skipped,
never as a passed group. A rebuilt binary, prose summary or CPU check cannot
satisfy future runtime verification. The future body requires lifecycle,
restore/fresh equality, architecture and failure-path observations described in
[runtime](../runtime.md); there is no GPU execution in this stage.

## G005-IDENTITY

The standard-library-only contract is a frozen `ExactKVIdentity`: identical
prefix tokens, model, adapter, positions and layout are all necessary. Its
identity predicate takes identities, never retention scores. `RetentionScore`
is a distinct static type; `RetentionCandidate` associates ranking information
with an identity without substituting for it. The CPU suite mutates each field,
checks immutability and shows a higher-ranked mismatched candidate being refused.
These are contract examples, not a cache implementation: no tensor storage,
byte proof, allocation ownership, event fences, serving behavior or quality is
qualified. A future runtime must also validate actual digests, token/position
geometry, adapters, layouts and lifecycle before using a matching identity.

## G005-MAP

[The system map](../system-map.md) lists command, target function and availability.
A deterministic check verifies those rows and actual targets. The benchmark
entry must remain absent while documented as future; adding it requires updating
the map and qualification together. This prevents inert placeholders being
reported as completed runtime or benchmark flows.
