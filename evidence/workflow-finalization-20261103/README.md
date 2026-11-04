# Workflow finalization

The shared GPU reservation now covers complete maintained-suite and research
coordinator invocations, including idle gaps. Leaf runs remain compatible with
the historical lock. Cross-process exclusion, partial-acquisition rollback,
and supervisor/legacy-child composition passed portable tests. Frozen historical
scripts are preserved; unwrapped historical tools are not automatically migrated.

A new Python 3.12 environment installed the exact runtime requirements and editable
project independently of the existing environment. Runtime imports, 1,381 tests
(26 missing historical-artifact skips), and a real eight-request GPU smoke passed.
The smoke checked generation, offload/restore, repeated outputs and cleanup. Cached
model files were reused; no model download or research experiment was resumed.
The validated environment is retained at the path in [validation.json](validation.json).

HTTP, cadence, priority, gate and operational validators were promoted into
`benchmarks/research_checks/`, with explicit artifact paths instead of importing
code from ignored output. Results agreed exactly with original validators on ten
real screen cases and three operational cases. Historical source-pinned drivers
still require their original runtime/data; the maintained four-tier suite is the
fresh-checkout path. Changing coordinators requires a new execution plan, not
rewriting old hashes. The paused AWQ prototype remains opt-in and unqualified.

GitHub Actions now runs `make check-light` on pushes and pull requests. Its local
equivalent passed 31 tests in a separate pytest-only environment, followed by
syntax checks. No GPU or full runtime install is required by CI. The remote
workflow has not run because these changes have not been pushed.

Raw install/test logs, the sealed smoke result, and cleanup inventory are in
ignored `output/workflow-finalization/`. The disposable lightweight environment
was removed; the verified runtime and reusable caches are retained. See
[validation.json](validation.json) for measured cleanup and log identities.
