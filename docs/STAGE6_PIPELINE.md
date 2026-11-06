# Stage 6 campaign pipeline

Implementation is in progress. This page documents implemented boundaries, not
completed qualification. The authoritative scope remains the
[execution plan](STAGE6_EXECUTION_PLAN.md).

The campaign layer extends `benchmarks.serving_suite`; existing four-tier commands
remain supported. It separates public logical identities from host-local bindings.
Start with `benchmarks/serving_suite/templates/host.example.json`, copied to a
private location, and the checked-in `campaigns/stage6.json` specification.

```bash
PY=/home/philip/.local/share/venvs/vllm-semantic-cache-ext/bin/python
CAMPAIGN=benchmarks/serving_suite/campaigns/stage6.json
HOST=/absolute/private/path/host.json
OUT=output/new-stage6-campaign
"$PY" -m benchmarks.serving_suite.campaign prepare --campaign "$CAMPAIGN" --host "$HOST"
"$PY" -m benchmarks.serving_suite.campaign compile --campaign "$CAMPAIGN" --host "$HOST" --out "$OUT"
"$PY" -m benchmarks.serving_suite.campaign status --out "$OUT"
"$PY" -m benchmarks.serving_suite.campaign verify --out "$OUT"
```

Preparation resolves publisher references once to immutable revisions and retains
content hashes. Compilation performs no GPU work and lists every unresolved
binding; it does not label an incomplete plan runnable. Measurement phases refuse
unresolved prerequisites. The selected campaign has 54 system/cell configurations
per repetition; this excludes overhead and operational work.

Each case has a content-derived identity and a fresh attempt identifier. The
append-only ledger uses a hash chain and durable writes; its index is rebuildable.
Completed attempts are sealed. Resume checks the seal and identity, and interrupted
cases become new whole-process attempts. Partial comparison blocks are replaced
as complete blocks. The campaign writer and GPU supervisor have separate locks;
children take only the legacy leaf GPU lock.

The dependency graph contract permits fixed request/tool graphs, validates
acyclic dependencies and allowlisted typed tools, forwards actual dependency
results, and cancels owned sibling work on failure. This is fixed orchestration,
not an autonomous agent choosing arbitrary tools or shell commands.

Confirmation requires correctness capability receipts, measured pilot precision
and cost, a locked task cohort/configuration/source closure, and balanced order.
Whole-process paired log-ratio intervals distinguish throughput gains from time
reductions. Requests within a run are not independent statistical repetitions.
Missing measurements, quality checks and operational checks remain explicit.

Raw local artifacts belong in ignored `output/`; reusable implementation and
compact findings belong in Git. A push does not back up raw attempts or downloaded
weights. Export must retain an explicit transfer/reacquisition inventory. The
5080 needs its own build and native correctness preflight before confirmation;
local export/import checks cannot establish second-machine performance.
