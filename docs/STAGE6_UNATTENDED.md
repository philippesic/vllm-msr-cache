# Stage 6 unattended operation

The durable GPU coordinator owns acquisition, including idle gaps between cases.
Agent work is a bounded health check every 20 minutes, plus repairs, analysis and
completion work. Do not keep an active turn polling or waiting for the next check.
The user explicitly selected this operating mode on 2026-11-07. Routine healthy
checks are silent; report material changes, repairs, blockers and completion.

The running campaign is `output/stage6-2080ti-20261106/development-15`, phase
`confirm`: 524 configurations in 210 whole matched blocks. It runs from immutable
source 21. Never edit that snapshot, its protocol, order, inputs or results. The
mutable checkout has a separately recorded derived validity/transfer revision;
its serving files are unchanged. Preserve the paused AWQ work and all useful runs.

## Scheduled check

The supported T3 task is bound to this thread, every 20 minutes. Its saved receipt
is `output/stage6-2080ti-20261106/unattended-health/schedule.json`; use its task ID
to update/disable it, without searching for a replacement. Scheduling runs only
when no agent turn is active. Do not use an active sleep/poll loop to wait for it.
Completion or a stopped coordinator is discovered on the next scheduled check,
within approximately 20 minutes while the application/scheduler is available.
Machine/app outages cannot guarantee delivery; the persisted ledger and exact
resume command below provide recovery when the scheduler returns.

Run one check from the project checkout:

```bash
PY=/home/philip/.local/share/venvs/vllm-semantic-cache-ext/bin/python
OUT=/home/philip/vllm-semantic-cache-ext/output/stage6-2080ti-20261106/development-15
STATE=/home/philip/vllm-semantic-cache-ext/output/stage6-2080ti-20261106/unattended-health
"$PY" -m benchmarks.research_checks.stage6_healthcheck --out "$OUT" --state-dir "$STATE"
```

The bounded check reads the ledger and completed case receipts, request/stream
progress, native request/tool coverage, GPU process ownership and reservation,
GPU telemetry and host memory. It retains raw health receipts outside the
campaign. Completed cases must form whole matched blocks before completion.
Zero GPU utilization during server startup or cleanup is normal. Persistent
absence of request/stream/ledger progress, a supervised deadline overrun,
missing telemetry, foreign GPU clients or severe memory exhaustion need review.
Normal swap use or an unfavorable throughput estimate alone is not a failure.

Fast worker polling (currently 0.5 seconds), per-request/case deadlines, descendant
tracking and cleanup stay unchanged. These are inexpensive process supervision,
not agent-level inspection. A healthy check ends the turn silently.

## Repair and recovery

If attention is required, inspect the linked attempt's result, evaluation,
worker/server logs and cleanup. Preserve failed attempts. Identify runner/setup,
correctness, resource or workload-purpose failures before repairing. Derived
analysis may be recomputed from sealed results; serving/prompt/resource changes
require separately declared affected comparison blocks. Never tune the frozen
candidate or repeat until a favorable result appears.

If the coordinator stopped, first inspect the saved process birth identities,
actual GPU clients, lock ownership and any still-running worker. Do not launch a
duplicate, kill foreign processes or unlink lock files. Allow a surviving owned
worker to finish or use the existing owned cleanup path. Once resources are free:

```bash
RECEIPT=/home/philip/vllm-semantic-cache-ext/output/stage6-2080ti-20261106/derived-analysis-revision-22.json
"$PY" -m benchmarks.research_checks.stage6_resume --out "$OUT" --analysis-receipt "$RECEIPT" --check
"$PY" -m benchmarks.research_checks.stage6_resume --out "$OUT" --analysis-receipt "$RECEIPT" > "$STATE/resume.log" 2>&1
```

Launch the actual resume in a persistent execution session, then end the active
turn once its ownership/progress is established. The wrapper binds the exact
frozen coordinator and checks its complete source identity. It overlays only the
hash-attested derived measurement-validity predicate in the coordinator process;
new worker processes still execute the unchanged frozen source. The live `--check`
proof reuses all eight completed restore-inactive B06 candidate attempts without
changing their original inconclusive ledger states. This avoids repeating the
finished four-system blocks under the old treatment-only restore requirement.
An interrupted comparison is still replaced as a whole, with its original attempt
preserved. Update ownership receipts after recovery and return to scheduled checks.

## Repository pressure coverage

The primary anchor B06 first-eight-session confirmation cohort has no CPU
restoration or CPU-eviction-affected returns in either ARC or candidate. Retain its
four-system throughput results as a GPU-reuse control. It does not satisfy B06's
intended pressure purpose, and B07 cannot establish pressure removal from this
cohort. This is a coverage defect, not an adverse policy result.

Existing full-sixteen-session operational evidence establishes local pressure:
ARC restored 141,726 tokens, with prior CPU evictions on 265/288 returning
requests; candidate restored 273,296 tokens, with prior CPU evictions on 270/288.
Both completed 384 main requests. These are one sustained process per arm across
six waves, not independent repeated four-system throughput confirmation. Secondary
repository and four-system document results must also retain their actual native
activity; do not infer activity from the workload label.

The remaining precise gap is anchor **four-system repository performance under
pressure**, not all local CPU-offload activity. Philip subsequently authorized the
[bounded PR-evidence addendum](STAGE6_ADDENDUM.md). It supersedes the unlaunched
nine-configuration supplement: do not run both. The original 524 cases are intact.

The frozen addendum is `output/stage6-2080ti-20261106/addendum-02`; its queue receipt
is `unattended-health/addendum-queue.json` beside this campaign. It schedules 18
repository configurations (held-out ARC pressure gate, four four-system blocks,
one roomy ARC), 48 new QA O0/O2 configurations (24 fixed pairs), and 12 prediction
stress configurations (four matched three-arm blocks with abandoned/late sessions
and unhinted traffic). Maximum 85 attempts, six elapsed hours overall and two per
section. Inputs, orders, estimators, purpose gates, retries and stopping rules are
prespecified in the addendum. Preparation/CPU checks do not authorize an early GPU
launch. No recent-upstream-base investigation; SAE PR #50422 feasibility only,
no integration or measurements.

## Addendum launch, completion and recovery

Once the primary check reports `acquisition_complete`, verify the primary
coordinator and worker have exited, GPU clients are absent and both reservations
are free. Verify the queued protocol hash and source manifest. Read `snapshot`
from the addendum `protocol.lock.json` and use it as the command's working
directory; the launcher refuses a different source. Then launch in a persistent
execution session (not a background shell without a receipt):

```bash
ADDENDUM=/home/philip/vllm-semantic-cache-ext/output/stage6-2080ti-20261106/addendum-02
ADDENDUM_PY=/home/philip/.cache/semantic-offload/stage6-assets/runtimes-rebuilt-01/local-env/bin/python
"$ADDENDUM_PY" -m benchmarks.serving_suite.campaign_addendum run --out "$ADDENDUM" > "$STATE/addendum-coordinator.log" 2>&1
```

Use the host receipt's pinned `runtime_python` for launch/resume: workers inherit
`sys.executable`. The project test venv is suitable for bounded health checks but
has a different package identity. The initial launch used that test interpreter
and the repository gate failed before server startup. Its failed attempt remains
sealed; repository is closed because gate replacement is forbidden. The independent
QA/advice sections resumed with the verified unchanged pinned runtime and original
start times. See [diagnosis and launch](../evidence/stage6-2080ti-20261106/primary-acquired-addendum-launch.json).

The entry point independently refuses incomplete primary whole blocks, a live
primary coordinator, occupied GPU/locks, changed inputs/source/policy or exceeded
persisted budgets. It keeps the fast existing worker supervision and fresh
processes. Record the returned execution-session/owner identity in the queue
receipt, establish the first attempt, and end the turn; do not wait for acquisition.

During addendum acquisition run **one** bounded check instead of repeatedly
checking the finished primary:

```bash
"$PY" -m benchmarks.serving_suite.campaign_addendum status --out "$ADDENDUM"
```

Healthy ends silently until the next existing 20-minute wakeup. `attention`
requires preserved diagnosis and ownership/resource checks. A stopped coordinator
is actionable even if its last state says running. Reuse complete sealed whole
blocks with `"$ADDENDUM_PY" -m benchmarks.serving_suite.campaign_addendum run --out "$ADDENDUM" --resume` from the same attested snapshot;
unfinished blocks need a linked `repair` authorization (CLI `--help`), at most one
four-arm repository and one three-arm prediction replacement. There are no QA
pair replacements. If repair is outside scope/budget, use `close-section` with
retained evidence and a reason, then resume the independent sections. No favorable
timing/quality retries or resetting start times. An exhausted cap is a retained
coverage limitation; finish independent audits and report it.

After `complete` or `budget_exhausted`, verify addendum attempts with
`benchmarks.serving_suite.campaign_verify.verify_campaign`, generate the actual
`benchmarks.serving_suite.addendum_report --out "$ADDENDUM"` comparisons and
secondary pressure audit, and prepare its source review with
`campaign_quality.prepare_review(Path(ADDENDUM), 'screen')`. The ordinary all-attempt
capped inventory automatically includes the addendum ledger. Complete its final
answer/blinded review too; acquisition does not establish quality equivalence.

The pre-addendum primary analysis/validation closure is preserved separately in
`output/stage6-2080ti-20261106/preserved-primary-analysis.json` (`closure` field).
Only the four already allowed derived analysis/transfer modules differ there
from primary source 21. Run the original primary report/export/relocation proof
from that attested closure, not the new addendum driver: do not broaden the
primary derived-source allowlist or relabel its timed inputs. The original
`derived-analysis-revision-22.json` still attests the unchanged recovery predicate.
Package the distinct addendum source/input/schedule closure and its findings
alongside the primary reproduction bundle; prove imports/relocation for both.

## After acquisition

A completed GPU phase is not completion of the user's task. Verify all 524 logical
configurations in 210 whole blocks; run the bounded addendum above;
finish objective/blinded source quality and every archived capped response; verify
retained attempts; prove relocation and GPU-free import against runtime rebuild 02;
retain model/workload/system comparisons, CIs and adverse findings; render the
actual comparison HTML with T3 html_preview then html_render. Finish the compact
findings, compatibility/quality/results and exact 5080 commands. Do not run 5080
jobs here. Readiness is for separately gated replication, not a 5080 performance or
production claim. See docs/STAGE6_EXECUTION_PLAN.md and the evidence README.

A scheduled run detecting acquisition completion continues this work autonomously.
Do not remain in a polling loop, defer the audit indefinitely or declare completion
from process exit alone. Preserve raw artifacts, update STATUS, commit coherent
checkpoints and push the user-owned branch. Disable this scheduled task only after
all requested work is complete or the user explicitly stops/pauses it. A required
external blocker must be reported with a concrete recoverable state.
