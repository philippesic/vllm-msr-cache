# Stage 6 campaign pipeline

The pipeline implements preparation, native capability checks, development screens,
bounded pilots, frozen confirmation, controls, persistent operation, source review,
and portable replication. Campaign completion is a derived result, not an install
claim. See the [execution plan](STAGE6_EXECUTION_PLAN.md) and
[current execution evidence](../evidence/stage6-2080ti-20261106/README.md).

The campaign layer extends `benchmarks.serving_suite`; existing four-tier commands
remain supported. Public logical identities are separate from private host bindings.
Use a new output directory for a new campaign. Keep the model artifacts, public
source trees, isolated runtimes, and immutable source snapshots in a reusable cache.

## Prepare a new local campaign

The public reconstruction recipe creates three isolated Python 3.12 environments.
It uses `uv`, exact dependency files, pinned native wheels, and pinned public source
checkouts. It installs the extension only into the local environment. The upstream
execution copy receives the recorded mapped-host-pointer compatibility patch;
its original checkout remains intact. Preparation does not qualify a GPU.

```bash
python3 -m benchmarks.serving_suite.runtime_setup --out /absolute/cache/runtimes
```

Copy `benchmarks/serving_suite/templates/host.example.json` to a private file.
Bind `runtime_python` to the new `local-env/bin/python`, the published runtime
manifest and resolved identities to the recipe's outputs, and `asset_cache` to
the reusable asset directory. Choose proposed resource profiles before screening;
the template is a contract, not measured capacity. Run from the project root.

```bash
PY=/absolute/cache/runtimes/local-env/bin/python
CAMPAIGN=benchmarks/serving_suite/campaigns/stage6.json
HOST=/absolute/private/path/host.json
OUT=output/new-stage6-campaign
"$PY" -m benchmarks.serving_suite.campaign prepare --campaign "$CAMPAIGN" --host "$HOST"
"$PY" -m benchmarks.serving_suite.public_inputs prepare --cache /absolute/cache/assets
```

`public_inputs chat-sample` filters the pinned WildChat shard. `public_inputs build`
constructs disjoint development, confirmation and repair cohorts from the pinned
SQuAD JSON, LongBench archive, chat sample, more-itertools and PEP sources. Its
`--python` must name the isolated tool interpreter used for actual tests; `--model`
names the prepared anchor tokenizer. See each subcommand's `--help` for the explicit
local input paths. These are authored tasks over public material, not official
dataset scores. Selected source text, reference answers and complete tool source
dependencies are retained with provenance. No downloads occur during measurement.

```bash
"$PY" -m benchmarks.serving_suite.campaign_setup --campaign "$CAMPAIGN" --host "$HOST" --out /absolute/new-inputs
HOST=/absolute/new-inputs/host.private.json
"$PY" -m benchmarks.serving_suite.campaign compile --campaign "$CAMPAIGN" --host "$HOST" --out "$OUT"
"$PY" -m benchmarks.serving_suite.campaign status --out "$OUT"
```

Preparation resolves publisher references once to immutable revisions and retains
content hashes. Compilation performs no GPU work and lists every unresolved
binding; it does not label an incomplete plan runnable. Measurement phases refuse
unresolved prerequisites. The selected campaign has 54 system/cell configurations
per repetition; this excludes overhead and operational work.

## Acquire and freeze

```bash
"$PY" -m benchmarks.serving_suite.campaign run --out "$OUT" --phase preflight
"$PY" -m benchmarks.serving_suite.campaign run --out "$OUT" --phase screen
"$PY" -m benchmarks.serving_suite.campaign qualify --out "$OUT"
"$PY" -m benchmarks.serving_suite.campaign run --out "$OUT" --phase pilot
"$PY" -m benchmarks.serving_suite.campaign pilot-plan --out "$OUT"
"$PY" -m benchmarks.serving_suite.campaign freeze --out "$OUT"
"$PY" -m benchmarks.serving_suite.campaign run --out "$OUT" --phase confirm
"$PY" -m benchmarks.serving_suite.campaign run --out "$OUT" --phase overhead
"$PY" -m benchmarks.serving_suite.campaign run --out "$OUT" --phase operational
```

Preflight checks native CPU restoration against fresh greedy generation on every
model, actual pinning/geometry, invalid advice and registry capacity. The complete
screen checks all selected configurations, actual external activation, a pressured
B06 and the matched roomy B07 contrast. A screen does not select policy winners.
Inspect objective quality and actual outputs during development before freezing.
The policy remains the selected Stage 5 CPU wait-retention behavior.

Pilots run three independent process blocks in the representative cells. Their
variance chooses from 6/8/10/12 confirmation repetitions (8/12 for four systems),
with a declared within-model propagation rule. Freeze seals task/resource/runtime
identities, executed source, repetitions, balanced order, and supplementary
schedules. Later confirmation, overhead and operational execution refuses changed
source or bindings. A new immutable source snapshot can run independently of
working-tree documentation or analysis edits during development. Freeze the final
source before confirmation; do not edit its executed files afterward.

Each case has a content-derived identity and a fresh attempt identifier. The
append-only ledger uses a hash chain and durable writes; its index is rebuildable.
Completed attempts are sealed. Resume checks the seal and identity, and interrupted
cases become new whole-process attempts. Partial comparison blocks are replaced
as complete blocks. The campaign writer and GPU supervisor have separate locks;
children take only the legacy leaf GPU lock.

Use `resume --out "$OUT" --phase PHASE` after an interruption. It seals interrupted
attempts and replaces an incomplete matched block as a whole. `--only MODEL:CELL`
selects work without removing other required cells from completion. `--dry-run`
shows the schedule without starting a server. Failed and interrupted artifacts
remain in the ledger; new attempts never overwrite them.

The dependency graph contract permits fixed request/tool graphs, validates
acyclic dependencies and allowlisted typed tools, forwards actual dependency
results, and cancels owned sibling work on failure. This is fixed orchestration,
not an autonomous agent choosing arbitrary tools or shell commands.

Confirmation requires correctness capability receipts, measured pilot precision
and cost, a locked task cohort/configuration/source closure, and balanced order.
Whole-process paired log-ratio intervals distinguish throughput gains from time
reductions. Requests within a run are not independent statistical repetitions.
Missing measurements, quality checks and operational checks remain explicit.

## Quality, recovery, verification and transfer

```bash
"$PY" -m benchmarks.serving_suite.campaign quality --out "$OUT"
```

This writes objective checks for every final response and a blinded review packet
under `analysis/quality/`. Review up to 32 responses per model/cell/system across
repetitions and traffic classes, plus every automatic failure and every capped
response. Capped intermediate steps are reviewed separately from final-answer
automatic scores, using only their preceding graph dependencies and tools. Inspect actual
source, dependency outputs and tools using the retained rubric. An exact duplicate
reference permits reuse of a previously made judgment; it is not an automatic
pass. Submit explicit source-supported, instruction-following and usefulness
judgments, unsupported claims and source notes using the versioned audit schema.

The archive-wide cap audit separately covers ordinary capped responses from all
campaigns immediately inside the acquisition root, including earlier failed or
interrupted attempts and the pilot/control/operational phases. It excludes native
fixtures and isolated warmup/cancellation/recovery probes. Use a new destination;
the packet retains the archived system-instruction source and verifies its hash.
It refuses finalization during acquisition or after its ledger inventory
changes. Exact duplicate contexts permit judgment reuse, with explicit ratings
still required for every response.

```bash
"$PY" -m benchmarks.serving_suite.campaign_capped_review --root /absolute/acquisition-root --out /absolute/new-cap-audit
"$PY" -m benchmarks.serving_suite.campaign_capped_review --out /absolute/new-cap-audit --ratings /absolute/cap-source-audit.json
```

```bash
"$PY" -m benchmarks.serving_suite.campaign quality --out "$OUT" --ratings /absolute/source-audit.json
"$PY" -m benchmarks.serving_suite.campaign verify --out "$OUT"
"$PY" -m benchmarks.serving_suite.campaign_portability --out "$OUT" --destination /absolute/new-portability-proof
"$PY" -m benchmarks.serving_suite.campaign report --out "$OUT"
"$PY" -m benchmarks.serving_suite.campaign export --out "$OUT" --destination /absolute/new-transfer-bundle
```

The campaign must also retain a real interrupted measured process followed by a
verified whole-block replacement. `campaign_recovery` provides the bounded
development exercise; it verifies the exact owned coordinator/worker before
signalling. Do not interrupt unrelated work or deliberately perturb confirmation.
`campaign_completion.recovery` derives its proof from the sealed ledger.

The portability proof relocates the bundle and an evidence copy, temporarily hides
the original evidence path under its writer lock, and runs the exported verifier
and report with model-file access and GPU/model imports forbidden. The relocated
comparison records must match the original report. It restores the original path
even on failure. It then starts a fresh process from the relocated exported source,
binds that source as the project root, imports a separately identified host campaign
and checks GPU-free preflight compilation and the confirmation guard. Pass `--runtimes` with
a second clean reconstruction to exercise different actual interpreter and external
runtime bindings; pinned cached weights are checked offline. Preserve the proof
destination and original evidence until verification succeeds.

Before freeze, `campaign_development --parent "$OUT" --out NEW_DIAGNOSTIC --model
qwen3_8b_awq --waves 2` prepares a separate development operation. Execute its
`screen` phase through the normal campaign command. It uses the persistent runner,
real advice perturbations and native cancellation without counting as the parent's
required screen or operational confirmation. This catches operational runner
defects before the long frozen acquisition.

## Replicate on the RTX 5080

Copy the sealed transfer bundle to the new host. Start in `BUNDLE/source`, the
exported source closure, using a bootstrap Python with `uv` installed. Use new
runtime, cache, host-binding and output paths. No local 2080 Ti environment or
historical `output/` tree is needed to reconstruct the runtimes.

```bash
BUNDLE=/absolute/relocated-transfer-bundle
RUNTIMES=/absolute/new-cache/stage6-runtimes
CACHE=/absolute/new-cache/stage6-assets
HOST=/absolute/private/stage6-5080.json
OUT=/absolute/new-output/stage6-5080
cd "$BUNDLE/source"
python3 -m benchmarks.serving_suite.runtime_setup --out "$RUNTIMES"
PY="$RUNTIMES/local-env/bin/python"
"$PY" -m benchmarks.serving_suite.transfer_setup --bundle "$BUNDLE" --runtimes "$RUNTIMES" --cache "$CACHE" --out "$HOST" --id rtx5080
"$PY" -m benchmarks.serving_suite.campaign import --bundle "$BUNDLE" --host "$HOST" --out "$OUT"
make bench-tests PY="$PY"
make check-light PY="$PY"
"$PY" -m benchmarks.serving_suite.campaign run --out "$OUT" --phase preflight
"$PY" -m benchmarks.serving_suite.campaign run --out "$OUT" --phase screen
"$PY" -m benchmarks.serving_suite.campaign qualify --out "$OUT"
"$PY" -m benchmarks.serving_suite.campaign activate-import --out "$OUT"
"$PY" -m benchmarks.serving_suite.campaign run --out "$OUT" --phase confirm
"$PY" -m benchmarks.serving_suite.campaign run --out "$OUT" --phase overhead
"$PY" -m benchmarks.serving_suite.campaign run --out "$OUT" --phase operational
```

`transfer_setup --offline` reuses already cached pinned weights. The bundle includes
both selected cohorts, reference answers, complete executed public tool inputs,
licenses, source/patch pins and exact dependency recipes. Full model files are
reacquired at their frozen revisions and their complete content identity must
match. Actual runtime/native identities are recorded afresh on the new host.
Tools receive the new interpreter and source paths; model-visible tool declarations
use stable public names, so changing host paths does not change prompts.

Import preserves the original tasks, precision, repetitions and balanced order,
while creating a linked hardware-specific campaign identity. It cannot acquire
confirmation before its own native checks and complete screens pass.
Host deadline limits are derived from the frozen plans rather than setup defaults.
`activate-import` activates that existing schedule; do not run `freeze` or replan
repetitions on the replication host. If the pinned native stack cannot execute
correctly on the new GPU, retain the capability failure and investigate it before
making hardware or speed claims. Compiled architecture support alone is not proof
of successful serving. Repeat the quality and offline verification steps above for
the new acquisition.

Raw local artifacts belong in ignored `output/`; reusable implementation and
compact findings belong in Git. A push does not back up raw attempts or downloaded
weights. Export must retain an explicit transfer/reacquisition inventory. The
5080 needs its own build and native correctness preflight before confirmation;
local export/import checks cannot establish second-machine performance.
