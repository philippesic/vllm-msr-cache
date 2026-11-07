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
pressure**, not all local CPU-offload activity. Fill it after the healthy frozen
acquisition releases the GPU using a separately declared bounded supplement:

- Use the existing complete sixteen-session confirmation repository cohort used
  by the operational runs, with the same real tools, model, output budgets and
  frozen candidate. No padding, task selection by speed or policy changes.
- Base GPU/CPU budgets remain 1,024 blocks / 2,793,406,464 CPU bytes. Declare sixteen
  active sessions/concurrency uniformly for all four systems. Freeze the executed
  supplement inputs and code before launch in a new raw directory.
- Two predeclared whole four-system blocks, using the existing balanced `order`
  function for repetitions 0 and 1, plus one ARC roomy control on the identical
  cohort at 4,831,838,208 CPU bytes and the unchanged GPU budget: nine scheduled
  configurations. Run the roomy case between the two base blocks.
- The first base block must complete with real native ARC restores and at least
  one CPU-eviction-affected returning request, all declared tool/request identities,
  active external policies and clean owned shutdown. If purpose/resource fit fails,
  preserve the block and report the gap; do not change capacities or chase activity.
- Allow at most one whole four-system block replacement for a diagnosed setup or
  execution defect, capped at thirteen total attempts. No optional stopping based
  on timing, no broad-matrix restart. Retain unfavorable or inconclusive results.
- Two repetitions establish bounded coverage and conditional observations, not the
  primary precision target or a new policy win. Keep this supplement separate from
  the frozen primary estimates and from the sustained-operation measurements.

Do not interrupt primary acquisition to run or prepare a new GPU diagnostic.
Implement the supplement through the existing serving runner/locks, with an
explicit manifest and receipts; avoid another campaign framework. If this small
scope cannot run faithfully, disclose the remaining coverage limitation.

## After acquisition

A completed GPU phase is not completion of the user's task. Verify all 524 logical
configurations in 210 whole blocks; run the bounded coverage supplement above;
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
