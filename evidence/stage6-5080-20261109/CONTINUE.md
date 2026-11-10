# RTX 5080 rebuild and full Stage 6 pass

Continue the authorized 2026-11-09 job autonomously through final audits and
pushed results. Branch `stage6-5080-20261109`, user-owned origin. Preserve the
closed November 8 campaign and all 2080 Ti history. No other machines/projects,
training, Stage 7 or credentials in tracked files. A setup/harness/storage failure
requires repair and continuation, not a capability-failure closeout.

## Current state

**Confirmation recovery is running** under independent user service
`stage6-5080-confirm-20261109-03.service`. Driver ownership is
`$BASE/confirm-driver-owner-03.private.json` (PID, birth, command, cwd, unit),
with driver log `$BASE/confirm-driver-03.log`. Individual cell coordinators have
`$BASE/confirm-owner-03-MODEL-CELL.private.json` and
`$BASE/confirm-03-MODEL-CELL.log`. Revalidate both service/driver and the current
frozen coordinator before any action; never launch a duplicate. The initial
cell was Qwen3-8B B11; all eight B11 blocks now have valid complete comparisons.
The latest post-repair health check has **370/524 valid configurations,
132/210 complete matched blocks**, with Ministral B02 ARC repeat 3 active.
All twelve Qwen3-8B B12 blocks are complete. Review 15 captures all completed
Qwen3-8B blocks; secondary-model acquisition is outside that static snapshot.
The [terminal anchor inventory](confirmation-qwen8b-attempts-complete.json)
reconciles all 364 planned Qwen3-8B configurations: 363 valid logical cases,
377 retained attempts / 35,530 verified files and 129/130 matched blocks.
The sole missing block is the known first B06 activity failure; its 32 attempts
remain unchanged. Execution is complete for that model, not analytical
completion or a complete campaign. Continue the remaining model cells.
B10 completed all twelve native pairs; B08 and B09 completed all twelve frozen four-system blocks;
B05 and B07 completed all twelve frozen pairs.
Health reports attention for the known first-block B06 candidate activity failure
below, plus original interrupted/setup-failed B11 arms until verified replacements
resolve those logical cases. Recheck actual live progress; these counts are a
checkpoint, not the current truth. Startup checks all three model files before
creating the first attempt; bounded disk-sleep/I/O with no child or log line
is not by itself evidence that the job stopped. Never launch a duplicate.

The first independent service (02) omitted the existing host-provided `rg` from
its default PATH. Eight attempts failed and a ninth was deliberately interrupted;
all report FileNotFoundError for `rg`, have verified cleanup/seals and no timing
credit. [Tool-PATH repair](confirmation-tool-path-repair-03.json) retains every
attempt, the stopped service exit state, tool version/hash and exact source/order.
Service 02 was reset after preserving that evidence. The corrected
[driver 03](../../benchmarks/research_checks/stage6_5080_resume_confirm_03.sh)
requires `STAGE6_CAMPAIGN_PATH` supplied explicitly at service creation and checks
`rg` and pinned `pytest` before any GPU acquisition. An independent service
environment probe passed; `rg` also resolves in the actual frozen coordinator's
environment. Current service 03 is outside the tool host's process tree.
At its first serving checkpoint, upstream and Continuum each completed all 80
requests and 48 tools, with no measurement error and verified cleanup/seals.
The full four-system replacement block remains pending at that checkpoint.
The subsequent [replacement proof](confirmation-b11-replacement-03.json)
verifies all four systems: 320 requests and 192 tools, common workload units,
clean cleanup and valid seals. The frozen block selector chooses only the new
shared block; the original partial comparison and failed service replacements
remain outside matched timing credit. Qwen3-8B B06 still has exactly 32 attempts.
Never restart scripts 02/03 with their existing names/logs; choose fresh recovery names
only after proving the current driver stopped and checking actual failures.

Original execution session 66082 and its recorded coordinator stopped during the
tool-host interruption around 02:09 UTC on November 10. No owned workers or GPU
compute clients remained; both GPU locks and the writer lock were available.
[Recovery receipt](confirmation-host-interruption-02.json) preserves the original
partial B11 arm, its verified seal, unchanged protocol/source and exact remaining
cell suffix. The [initial recovery driver](../../benchmarks/research_checks/stage6_5080_resume_confirm_02.sh)
runs the original frozen `resume --only MODEL:CELL` CLI for each remaining cell,
in original order. The interrupted B11 comparison is replaced as a whole; its
three earlier completed arms and interrupted fourth arm remain without matched
timing credit. Existing complete blocks are reused. Do not restart session 66082
or run a full-phase resume: that would retry the activity-invalid B06 block.
The selected suffix deliberately excludes earlier Qwen3-8B B06. The independent
service is outside the tool host's process tree; its exact driver and frozen
coordinator descendants were checked live. No study-design change.

The service has `Restart=no` and retains its exit state. If it stops, inspect the
failed cell and preserve logs/receipts before selecting another recovery suffix;
never blindly restart the existing script because its launch names already exist.
Zero exit means all selected acquisition commands returned, not all cases passed.
Reconcile the entire frozen identity inventory before overhead/operational.

Screen session 65450 is **finished**. All 54 required configurations / 22 whole
blocks completed; [screen receipt](screen-complete.json) verifies 58 verified,
one interrupted and one failed/inconclusive attempt across screen + preflight,
6,160 sealed files. The count retains the unmatched original B01 ARC. B10
candidate served all 20 requests with clean cleanup and no CPU restores/evictions;
the frozen validity rules accept it as a GPU-resident control, while preserving
its original inconclusive status. [Control receipt](screen-team-control.json).
Never retry merely to obtain restore activity.

The [first confirmation B06 candidate activity receipt](confirmation-b06-inactive-restore-01.json)
verifies all 81 sealed files: all 32 requests / 24 tools / eight units completed,
zero caps and clean cleanup, but **zero CPU restores and eight actual scheduler
preemptions**. Both native receipts and the scheduler metric record eight. The
frozen v3 validity rules reject this attempt; their zero-preemption control
exception does not apply. Its four-system block remains incomplete for analysis.
Do not relabel it, change capacity or retry to obtain favorable activity. The
second scheduled candidate process also has zero restores, but zero preemptions
and evictions, so the existing frozen control exception accepts it while keeping
its raw inconclusive status. The first ARC process has zero restores/affected
returns: confirmation pressure is not established by the passing screen gate.
Continue the remaining frozen measurements and audits, preserving this adverse
outcome. It can prevent a complete final matrix; never force `complete=true` or
silently reduce the frozen repetition requirement. Scheduled attention for this
same reviewed attempt is not evidence of a dead coordinator or authorization
for a replacement. Investigate distinct new issues normally.

The [terminal B06 attempt inventory](confirmation-b06-attempts-complete.json)
verifies all **32 scheduled attempts / 2,648 sealed files**: 1,024 completed
responses, zero caps, every declared request/tool/unit and cleanup check passed.
There are **31 valid measurements / seven complete matched blocks**, not the
required eight. All eight planned repetitions ran exactly once; the first remains
excluded. B06 execution is finished and must not be resumed to replace it.

[Qualification](qualification.json) **passes**: all native correctness fixtures,
registry, external activation/pinned geometry, all required screens, all three
ARC pressure gates and the roomy reduction (one to zero affected returns).
[Activation](activation.json) preserves exact imported schedule/supplementary
order/precision/source, without replanning: 524 confirmation configurations in
210 matched blocks. This is functional qualification, not performance or
production qualification. Confirm, overhead, operational and final audits remain.

Completed prerequisites:

- [Build](BUILD.md), [source/toolchain/wheel receipt](lmcache-build.json): same
  LMCache 0.3.6 release, all 178 Python files unchanged, sm_120 plus PTX, separate
  Continuum environment, same package versions. Exact pinned CPU/GPU round trip.
- [Short serving smoke](continuum-smoke.json): warmup and four requests succeeded
  with GPU KV stores/pinned geometry/cleanup. Original one-session policy result
  stays inconclusive; this diagnostic is excluded from campaign measurements.
- [Full Continuum screen](continuum-full-screen.json): B06 passed all 32 requests
  and 24 scheduler pins, with sealed rebuilt-native identity and clean release.
- [All five Continuum screens](continuum-all-screen.json) now pass: B06/B08/B09/
  B11/B12 completed 32/64/64/80/32 requests and 24/30/35/42/24 pins respectively;
  all seals/cleanup pass. This fixes the original blocker across every frozen
  Continuum screening fixture, without claiming repeated performance yet.
- [Native preflight](preflight.json): all three restore/fresh-reference equality
  fixtures and anchor registry probe passed; four attempts / 223 files verify.
- [Import](import.json): all 22 cells resolved; unchanged profiles/corpora and
  frozen order/repeats. Plan differences only the new Continuum interpreter and
  rebuilt LMCache wheel reference. All workers explicitly bind the pinned PY.
- [Recovery](recovery-complete.json): deliberate B01 interruption sealed and
  cleaned, both replacement arms passed as a whole block. The watcher is done;
  never restart it or repeat preflight. [Original interruption](recovery-interruption.json).
- [Anchor four-system screen](anchor-screen.json): all B06 arms passed geometry,
  cleanup/activation and ARC CPU-pressure purpose. ARC had one eviction-affected
  return; B07 roomy ARC had zero, satisfying the prespecified roomy reduction
  check at the latest health snapshot. This is not whole-campaign qualification.
- [Ministral pressure screen](ministral-pressure-screen.json): both native B06
  arms passed. ARC had 35 eviction-affected returns out of 48 returning requests,
  candidate 29/48. This meets that model's ARC pressure-purpose gate.
- [Qwen4B pressure screen](qwen4b-pressure-screen.json): both native B06 arms
  passed. ARC had 41 eviction-affected returns out of 48 returning requests,
  candidate 38/48. All three models now have positive ARC pressure evidence;
  whole-matrix qualification subsequently passed.
- [Transfer seals](transfer-seals.json) and [old-history integrity](history-integrity.json)
  pass. All original raw measurements and the original LMCache binary remain.
- Exported CPU checks passed: 240 benchmark tests, 31 lightweight tests and
  compilation. Validation-generated caches were moved outside the old bundle;
  all 4,606 execution-bundle files verify. Do not write bytecode/caches there.

## Paths and safe launch

```bash
REPO=/home/philip/vllm-semantic-cache-ext
BASE=$REPO/output/stage6-5080-20261109
OLD=$REPO/output/stage6-5080-20261108
VOLUME=$OLD/linux-volume-01
BUNDLE=$OLD/primary-execution-bundle-01
RUNTIMES=/home/philip/.cache/semantic-offload/stage6-5080-runtimes-02
PY=/home/philip/.cache/semantic-offload/stage6-5080-runtimes-01/local-env/bin/python
OUT=$VOLUME/primary-20261109
export PATH=/usr/lib/wsl/lib:$PATH
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/mnt/c/vllm-semantic-cache-ext-stage6-5080/huggingface
```

`$BASE/primary-01` is an alias; use canonical OUT for process matching. Verify
VOLUME is mounted before writes/resume. Never format its existing image. The
old FILESYSTEM_REPAIR.md contains the image/remount recipe. Latest free space:
about 9.0 GiB ext4, 3.26 GiB WSL root and 11.19 GiB Windows at the last bounded
health check; monitor all three. Root includes the latest derived review snapshot.
Private disk observation 03 retains recent host-owned app-server
binary/catalog sizes: these account for part of the recent root growth, but the
full cause is not established. Runtime/tmp had no large recent writes. Shared
application files, configuration and all raw/derived evidence were preserved.
Keep useful caches/raw evidence. Reconsider storage for later relocation copies
using actual sizes, not an assumption that a second raw tree will fit.
Private storage projection 02 models about 2.86 GB for the then-remaining
174 confirmation configurations using the largest observed same-arm attempt;
38 overhead and six operational cases remain unmodeled. This supports continued
acquisition, not a guarantee that the final materialized relocation copy fits.
Size that copy after acquisition and expand existing storage if necessary.

After the entire confirmation driver is terminal and the frozen identity inventory
is reconciled, launch another phase from an independent user service with fresh
unit/log/receipt names, using the retained exec helper, for example:

```bash
"$PY" -m benchmarks.research_checks.stage6_5080_launch --phase overhead \
  --log "$BASE/overhead-01.log" --receipt "$BASE/overhead-owner-01.private.json"
```

It validates the mount, exact exported source and literal pinned interpreter,
then execs the existing coordinator with its normal reservation and cleanup.
Run that helper under `systemd-run --user` with the mutable repository as working
directory, explicit HF_HOME, the verified tool PATH (including existing `rg`) and
pinned interpreter. Verify tool availability in that service environment first.
Retain unit exit state and
preserve logs/receipts. Add `--resume` only after verifying stopped owned resources
and selecting the actual unfinished scope. Stop/reset owned transient services
during final verified cleanup. Other campaign
commands must run after **cd into `$BUNDLE/source`** with `$PY`. Initial wrong-cwd
setup/import launches were stopped before any acquisition and preserved; do not
repeat that mistake. Never acquire from mutable source or a test interpreter.

## Remaining work

1. Supervise the independent confirmation service through all 524 configurations /
   210 matched blocks. Inspect real failures and measurement_validity; retained
   restore-inactive GPU controls are not CPU-pressure evidence. Preserve every
   original status and whole-block repairs. Do not repeat screen, qualification
   or activation merely because a monitoring turn was interrupted.
2. Once confirm is complete, run overhead (38 configurations), then operational
   (six) through the helper with fresh logs/receipts. No freeze, replanning,
   policy/model/workload/geometry changes or favorable-result retries.
3. Verify every sealed attempt and campaign identity. Finish objective/blinded
   source and every ordinary capped-output audit, recovery and portability proofs,
   campaign report/completion, and standalone comparison chart. Render requested
   comparisons with T3 html_preview then html_render; do not start a server.
4. Report geometric throughput ratios with paired whole-process 95% CIs and
   actual latency figures. Candidate=wait. The report's Continuum/ wait ratio and
   interval can be inverted to give candidate/Continuum (invert bounds in reverse
   order). Retain uncertainty, adverse results and actual pressure. No quality,
   production or GPU-only cross-host attribution claims without evidence.
5. Consolidate RESULTS.md and STATUS, commit/push all findings, release owned
   servers/GPU/locks, disable the health schedule only at pushed completion or an
   irreparable genuine-failure closeout. An ordinary process exit is not done.

## First completed-cell estimates

[First two cells](confirmation-first-cells.json) now retain all frozen native
Qwen3-8B repetitions: B01 twelve pairs and B02 six. Candidate/ARC ratios are
1.0013538578 [0.9989260363, 1.0037875799] and
1.0077593363 [0.9961581821, 1.0194955964], using the exact frozen whole-process
log Student-t estimator. Both intervals include parity. Per-process serving
seconds and end-user mean/p95 TTFT/completion summaries are in the receipt and
RESULTS.md. All 456 matched request texts and finish reasons agree between
native arms, but no quality-equivalence claim follows. Returning CPU restores
and eviction-affected-return counters are zero; do not claim pressured-offload
benefit. This checkpoint is complete only for these two cells, not the campaign.
[Interim display](early-controls.html) was previewed/rendered with T3, with
[bound data/render proof](early-controls-render.json); it covers only these two
cells. Do not mistake it for the pending final full-matrix chart.
The newer [three-cell checkpoint](confirmation-three-cells.json) adds complete
B03 document repetitions: candidate/ARC **0.9972007675
[0.9896228337, 1.0048367286]**, twelve pairs. Candidate/ARC mean serving times
are 45.6411976 / 45.5122017 seconds; mean end-user TTFT is 20.8078097 /
20.7544478 seconds and mean completion 25.5310667 / 25.4610581 seconds.
Both arms have 192 requests, 63,768 output tokens, zero caps and no returning
CPU restores/eviction-affected returns. Its interval also includes parity; do not
claim a resolved regression or gain. All 192 new matched texts/finish reasons
agree, bringing the three-cell total to 648 pairs / 1,296 recorded responses.
That does not waive source/word-limit findings or establish quality equivalence.
The [new display](early-three-cells.html) was previewed/rendered with T3;
[render binding](early-three-cells-render.json) and
[actual extraction/guard validation](early-cells-validation.json) are retained.
It covers three of 22 native cells; all older files above remain unchanged.

The job-specific extraction code is tracked in
`benchmarks/research_checks/stage6_5080_early_cells.py`. It was invoked as an
absolute script with the pinned PY from BUNDLE/source, imports the frozen report
module explicitly, waits without writing until every B03 pair is present and
refuses an existing receipt or wrong working directory. It is derived reporting,
not acquisition source. Current receipt exists: do not rerun to overwrite it.

Final report extraction must reproduce all three estimates unchanged and include
all remaining cells, external systems, overhead, operational and audits.

## Final report preparation

The primary-only reporting helper is
`benchmarks/research_checks/stage6_primary_findings.py` in the mutable repo.
[Latest validation](reporting-validation-03.json): 28 focused reporting/estimator tests,
with the earlier 31 portable checks/compilation separately retained; chart JavaScript parses in V8. The current helper uses transparent roots and host theme variables with standalone fallbacks.
The original `stage6_findings.py` requires an addendum and contains historical
2080 conclusions; do not use it to present this primary-only campaign.

After all acquisition, confirmation source review and portability readiness,
regenerate the frozen campaign report from SOURCE. Only when it has
`complete=true`, use the helper from mutable REPO:

```bash
"$PY" -m benchmarks.research_checks.stage6_primary_findings \
  --report "$OUT/analysis/current/results.json" \
  --out evidence/stage6-5080-20261109/results
```

It refuses incomplete reports or an existing destination, validates estimates
and whole-process work, and produces `comparisons.json`, `tables.md` and a
standalone interactive `comparisons.html`. The derived display consistently
puts candidate above external systems by reciprocal transformation with reversed
CI-bound order; original traffic-class/classification orientation remains
explicit. Latency figures are means of per-process statistics, not pooled p95s.
External missing native counters stay unavailable. Actual full-report extraction
and final T3 html_preview/html_render remain untested/pending until data exist.
This is derived presentation; acquisition source/design stays unchanged.

## Terminal audit order

These steps are pending. They were checked against the exported CLI and current
audit implementations, not executed as a claim of completion. After confirmation,
overhead and operational have all finished, verify the actual owner processes
have exited and no ordinary ledger attempt remains running/planned. Successful
campaign completion requires all
524 confirmation configurations / 210 matched blocks, 38 overhead configurations
and six operational configurations, with complete matched supplementary blocks
and their purpose/cleanup receipts. A zero CLI exit alone is insufficient.

Execution of a phase and analytical completion are distinct. If the known first
B06 activity failure remains excluded, **do not wait forever for 524 valid cases
or resume completed confirmation to replace it**. After the recorded coordinator
exits, reconcile every frozen scheduled model/cell/repeat/system identity against
its terminal sealed attempt and recheck owned resources. The unchanged frozen
CLI permits the existing overhead and operational schedules after confirmation's
planned attempts have ended; acquire them and finish the audits. The all-cap
inventory requires stable terminal ledgers, not a favorable measurement verdict.
Portability can preserve an incomplete throughput report. Retain missing blocks,
all adverse outcomes and `complete=false`; if the full requirements remain
unattainable without changing the study, document a failed full campaign rather
than asserting success. The success-only findings helper above must keep its
guard; any failure presentation must explicitly retain incomplete scope and the
actual frozen estimates/repetition counts, never fabricate the missing pair.

The reporting helper now supports **explicit `--retain-incomplete`** for that
terminal outcome. Its default still rejects incomplete reports. The option
preserves the frozen report's complete/acquisition flags, completion evidence and
every missing model/cell/repeat identity. It labels estimates conditional and
shows measured pairs / (measured pairs + reported missing blocks), including in
the chart; no missing repetition is inserted or requirement reduced. Existing
estimator, matched-work and no-overwrite checks remain. All comparisons must
still contain at least two matched pairs. Only use this after the remaining
acquisition and audits finish, with an explicitly incomplete RESULTS.md outcome.
Synthetic CLI validation covers rejection, explicit retention and existing-output
preservation; actual terminal extraction and final chart rendering remain pending.

From **`$BUNDLE/source`**, with the literal `$PY` and bytecode disabled:

```bash
"$PY" -m benchmarks.serving_suite.campaign verify --out "$OUT"
"$PY" -m benchmarks.serving_suite.campaign quality --out "$OUT"
```

The second command prepares the **final confirmation** packet in
`$OUT/analysis/quality`; it is not a source-review pass. Preserve an immutable
copy of its packet, objective, summary and private labels under a fresh BASE
snapshot. Explicitly read every new complete context before rating. Reuse only
exact eight-field plus archived instruction-source matches, rebind current IDs,
and cover every prescribed item, automatic failure and capped intermediate.
Then finish with the absolute path to the completed final ratings:

```bash
"$PY" -m benchmarks.serving_suite.campaign quality --out "$OUT" \
  --ratings "$REPO/evidence/stage6-5080-20261109/confirm-source-judgments-final.json"
"$PY" -m benchmarks.serving_suite.campaign_capped_review \
  --root "$BASE" --out "$BASE/caps-all-final-01"
```

Those destination names are proposed fresh paths; preserve any existing path
and choose a new numbered one if already present. The all-cap inventory must
have `complete_inventory=true`, no pending attempts and unchanged input ledger
digests. Its root is BASE, whose immediate campaign child is this campaign;
VOLUME would also include November 8 history. Review all new extended contexts,
including actual preceding dependencies/tools and archived instruction/prompt
bindings, then finish with the complete cap ratings:

```bash
"$PY" -m benchmarks.serving_suite.campaign_capped_review \
  --out "$BASE/caps-all-final-01" \
  --ratings "$REPO/evidence/stage6-5080-20261109/caps-all-source-judgments-final.json"
```

Next check disk using actual tree sizes. Portability materializes another raw
tree and exported bundle, holds the writer lock and temporarily parks the
original campaign path; never run it during active acquisition or another audit.
Use a new destination on the mounted ext4 image, outside OUT:

```bash
"$PY" -m benchmarks.serving_suite.campaign_portability --out "$OUT" \
  --destination "$VOLUME/portability-20261109-final-01" --runtimes "$RUNTIMES"
"$PY" -m benchmarks.serving_suite.campaign report --out "$OUT"
"$PY" -m benchmarks.serving_suite.campaign verify --out "$OUT"
```

Require the final report's `complete=true`, all 42 comparisons, complete
quality/overhead/operational/recovery/portability gates, and every seal. The
pre-proof sealed export can correctly retain an incomplete readiness snapshot;
preserve it and link the adjacent final proof/report rather than rewriting its
seal. The portability proof's inherited text saying no 5080 execution describes
the GPU-free proof only. Its `separately_rebuilt_runtime_bindings=true` does not
mean all environments were rebuilt: only Continuum was rebuilt; local/upstream
were reused at their pins. State both scopes explicitly in final evidence.

Only then run the primary findings helper above, compare its first B01/B02
estimates to `confirmation-first-cells.json`, preview/render the full chart and
write the final plain-language results. Retain final source/cap judgments,
integrity/transfer/portability receipts and a sealed compact completion packet.
Release owned resources and disable only this job's schedule at actual closeout,
record the verified release, and push the final checkpoint. Keep goal active until
the full requested outcome and push are verified; the present document is a recipe.

The latest read-only [confirmation integrity checkpoint](confirmation-integrity-checkpoint-05.json)
verifies 26,096 sealed files / 279 verified, one interrupted and nine raw failed
attempts at its snapshot, plus one active unsealed attempt. It checks attempt,
input and protocol identities, retaining the ledger observations before/after
the frozen verifier. Raw failed states include accepted inactive-restore controls
and the excluded B06 attempt; integrity does not change their validity.
Coordinator ownership was checked live, and all 2,192 exact frozen acquisition
source files still match the locked manifest. The
[previous checkpoint](confirmation-integrity-checkpoint-02.json) separately
reverified the exact 2,192-file acquisition source against its locked manifest
and all 4,606 execution-bundle files unchanged. All earlier checkpoints remain.
These are explicitly not final completion audits: active unsealed work and
later attempts are outside each snapshot's sealed inventory.

## Answer-review checkpoint

Earlier completed partial screen review: [index 06](screen-review-index-06.json) and
[ratings 06](screen-source-judgments-06.json), raw snapshot
`$BASE/quality-screen-checkpoint-06/`. All 781 prescribed selected items have
explicit judgments, covering 289 full contexts. Of 142 novel contexts relative
to index 05, 138 were explicitly read and four exactly matched the separate
cap review. There are four QA parts, ten repository parts and one cap-reuse
part; each binds the immutable snapshot packet and context signature. Full new
source/test bodies and additional NumericRangeTests reference content were
read. Snapshot objective coverage is 778 finals. Selected-item findings: 190
source-unsupported, 424 instruction failures, 53 not useful, with duplicates
counted separately. This retains malformed/fenced QA JSON, unsupported additions,
wrong answers, invented limitations, false validation denials caused by broad rg
snippets, and overlong/unfinished intermediates. Correct useful fragments are
separately recognized where warranted.
These are selected sample counts, not population rates or quality equivalence.
The archived executed system instruction also requires the literal validation
status format for tool-using team syntheses and limits intermediates to 70 words;
the latest ratings/index bind its digest. Do not waive those requirements based
on the shorter current node prompt. Original scores/timings/gates are unchanged.

[Partial all-cap review](caps-screen-review-index-01.json), raw
`$BASE/caps-screen-checkpoint-01/`, covers all 13 sealed capped responses present
at that snapshot (10 extended contexts; four novel): four source-unsupported,
all 13 instruction failures, 11 not useful. Correct useful fragments in two
duplicate capped responses are retained as useful. This inventory includes arms
of unfinished whole blocks and is explicitly incomplete during acquisition.
It checks the archived executed system instruction plus the current node and
only preceding dependencies/tools. Final all-phase cap inventory/finish remains
pending; do not mark the partial inventory complete. Its expanded signatures
include system-instruction and prompt-context bindings; ordinary quality ratings
are reused only after separately checking those additional fields.

Snapshot 06 is fully rated for its partial inventory; preserve it.
**Final screen source audit 07 is complete**, with
[index 07](screen-review-index-07.json), [ratings 07](screen-source-judgments-07.json)
and [frozen-validator finish receipt](screen-quality-complete.json).
Raw immutable input: `$BASE/quality-screen-final-07/`; finished isolated audit:
`$BASE/quality-screen-final-audit-07/analysis/quality/`. It covers 1,194 objective
finals and all 1,248 prescribed selected items across 522 complete contexts.
All 233 novel contexts versus canonical 06 were explicitly read and judged:
54 QA finals, 131 investigation finals and 48 capped intermediates. Four QA
parts and 25 investigation parts retain explicit context bindings. Exact prior
reuse checks all eight fields and the same archived system-instruction digest;
all sealed screen manifests share the verified instruction-source file hash.
Selected counts, including duplicate contexts: **401 source-unsupported,
725 instruction failures, 111 not useful**. These are not population rates or
quality equivalence. Useful correct fragments remain useful where warranted.
Reread pinned source/tests confirmed `windowed` rejects n<=0 and
`PeekableTests.test_slicing_reset` exists before final consolidation; judge the
pinned evidence rather than a remembered library implementation.

The frozen `finish_review` hard-codes `analysis/quality`, so it was run on a
materialized copy of the final screen snapshot in the isolated audit workspace,
with exact packet/objective/private mapping and `phase=screen` preserved.
It validated all 1,248 judgments and produced audited status. The live campaign's
confirmation quality destination was not touched. Do not reuse this isolated
screen status as a confirmation audit.

The earlier expanded **partial all-attempt cap review 02** covers every sealed
ordinary cap through all screening and the first twelve confirmation B01 blocks:
[index](caps-all-review-index-02.json), [ratings](caps-all-source-judgments-02.json),
[extra instruction bindings](caps-instruction-bindings-02.json), raw
`$BASE/caps-all-checkpoint-02/`. It includes **156 responses / 79 extended
contexts**: 132 screen and 24 confirmation, 142 capped intermediates and 14
capped finals. All contexts exactly match existing explicit judgments; additional
archived system-instruction source, tools/no-tools branches and token-prompt
binding digests were separately checked. Source-unsupported 101, instruction
failures 156, not useful 40, with duplicates counted. All are selected cap counts,
not population rates. `complete_inventory=false`; do not call final cap finish.
Regenerate after every acquisition phase is terminal, preserving snapshot 02.
The entire mutable campaign_workload module differs from the frozen module in
historical addendum branches; the prompt-construction prefix matches, and the
actual archived frozen source/hash is authoritative. Never acquire from mutable
source or claim entire module equality from the shared instruction snippet.

The earlier **partial all-attempt cap review 03** covers **163 responses / 86
extended contexts** through early B08: 132 screen and 31 confirmation responses,
148 capped intermediates and 15 capped finals. [Index](caps-all-review-index-03.json),
[ratings](caps-all-source-judgments-03.json),
[instruction/corpus/prompt bindings](caps-instruction-bindings-03.json), raw
`$BASE/caps-all-checkpoint-03/`. All 156 earlier responses exactly reuse their
extended-context judgments. Five new responses match explicit confirmation08
judgments; [two new contexts](caps-all-explicit-judgments-03-part-01.json) were
explicitly read. One PEP 310 intermediate invents validation PASS before the
actual validation tool, but retains useful correct source excerpts. One initial
PEP 327 response instead gives CSV content and is not useful. Both exceed the
70-word intermediate limit and end unfinished.

Selected cap counts are **108 source failures, 163 instruction failures and
46 not useful**, including duplicates. Archived instruction sources in all 51
cap-bearing attempts match the previously inspected source/snippet. Each response's source,
reference, task/current instruction, tool-branch flag, node and role match its
sealed corpus; objective fields and the raw prompt/workload-source digest were
independently recomputed. Each of the 86 quality contexts has one prompt-binding
digest in this snapshot. Raw token prompts were bound, not individually decoded.
The inventory has `complete_inventory=false` and one pending attempt; do not
call final cap finish or treat this as all-phase completion. Preserve earlier
snapshots and regenerate after every phase is terminal. This checkpoint used cap03
for exact expanded-context reuse and is extended by cap04 below.

The historical **partial all-attempt cap review 04** covers **196 responses / 119
extended contexts**: 132 screen and 64 confirmation, 177 intermediates and nineteen
finals. [Index](caps-all-review-index-04.json),
[ratings](caps-all-source-judgments-04.json),
[bindings](caps-instruction-bindings-04.json), raw `$BASE/caps-all-checkpoint-04/`.
All 163 earlier responses exactly reuse cap03 extended-context ratings. Twenty-nine
added responses match prior explicit confirmation judgments; [four new complete
contexts](caps-all-explicit-judgments-04-part-01.json) were explicitly read.
Three are repetitive reverse-iteration intermediates: one correctly copies actual
PASS but still lacks useful analysis, while another invents PASS before validation.
The fourth is an eleven-word malformed repeated-token/wrong-metadata Decimal
response; its cap is not a whitespace-word-limit excess. All four are not useful.

Selected duplicate-inclusive counts are **136 source failures, 196 instruction
failures and 73 not useful**. All sixty cap-bearing archived instruction sources
verify. Every sealed source/reference/instruction/node/role/tool-branch and objective
binding, plus raw prompt/workload-source digest, was independently recomputed.
Each quality context has one prompt-binding digest in this snapshot; tokens were
bound rather than individually decoded. Includes ordinary individual arms outside
valid matched blocks, preserving all timing/objective exclusions. The snapshot
has `complete_inventory=false` and one pending attempt; do not call frozen finish.
Regenerate after all phases are terminal, retaining every earlier snapshot. Use
cap04 for future exact expanded-context reuse, with novel contexts explicitly read.

The historical **partial all-attempt cap review 05** covers **215 responses / 135
extended contexts**: 132 screen and 83 confirmation, 194 intermediates and 21
finals. [Index](caps-all-review-index-05.json),
[ratings](caps-all-source-judgments-05.json),
[bindings](caps-instruction-bindings-05.json), raw `$BASE/caps-all-checkpoint-05/`.
Nineteen response occurrences were added: three repeat an old extended signature,
and sixteen new extended items exactly match full quality contexts already
explicitly read in confirm12. Thus 199 occurrences reuse cap04 and sixteen reuse
confirm12, with zero novel full quality contexts or automatic novel judgments.
Selected duplicate-inclusive counts are **152 source failures, 214 instruction
failures and 79 not useful**. Source support, requested status/rule/limitation
format and usefulness remain separate judgments; a capped final may fit the
whitespace-word instructions while making an unsupported additional claim.
All 68 cap-bearing archived instruction sources verify; sealed corpus/node,
source/reference/role/tool branches, objective and raw prompt/workload digests
were independently rechecked. Tokens were bound, not individually decoded.
Includes failed/interrupted/unmatched ordinary arms. One attempt was pending,
so `complete_inventory=false`; no final finish was called. Regenerate after all
phases are terminal. Use cap05 for exact expanded-context reuse, retaining history.

The latest **partial all-attempt cap review 06** covers **217 responses / 137
extended contexts**: 132 screen and 85 confirmation, 196 intermediates and 21
finals. [Index](caps-all-review-index-06.json),
[ratings](caps-all-source-judgments-06.json),
[bindings](caps-instruction-bindings-06.json), raw `$BASE/caps-all-checkpoint-06/`.
All 215 prior occurrences exactly reuse cap05 extended-context judgments; both
new occurrences exactly match the explicitly read confirm14 capped intermediates
(indices 8 and 51). No new full quality context or automatic novel judgment.
Both contain useful rank/validation facts but unsupported performance-test claims,
exceed the intermediate word limit and omit the required next tool declaration.

Selected duplicate-inclusive counts are **154 source failures, 216 instruction
failures and 79 not useful**. All 69 cap-bearing archived instruction sources
verify. In addition to complete sealed source/reference/instruction/node/role/tool
branches and objective/prompt bindings, every actual preceding dependency and
tool excerpt was independently recomputed from sealed requests and ancestor tool
files. Each quality context has one prompt-binding digest; tokens were bound,
not individually decoded. All ordinary failed/interrupted/unmatched arms were
included. One attempt was pending, so `complete_inventory=false`; no frozen final
finish was called. Regenerate after all phases are terminal. Use cap06 for future
exact expanded-context reuse and preserve earlier snapshots.

The earlier **partial confirmation** review 01 is also retained:
[index 01](confirm-review-index-01.json), [ratings 01](confirm-source-judgments-01.json)
and [two explicit contexts](confirm-explicit-judgments-01.json), raw
`$BASE/quality-confirm-checkpoint-01/`. At seven completed B01 blocks, it contains
14 objective finals (zero automatic failures) and 14 capped intermediates:
28 selected items, two distinct full contexts, both explicitly read.
All 28 lack support for precise fictional landmark names/measurements and repeat
an inaccurate factual-provenance claim; all fail the relevant instruction, while
all retain useful creative content. This is fictional worldbuilding and a source
support limitation, not a real-world geography error. Duplicate contexts are
repeated fixed-cohort observations, not independent quality samples. The index
is explicitly acquisition_complete=false. Do not finish this partial audit as
though all confirmation were captured; regenerate at terminal acquisition and
rebind canonical IDs to the new first item for exact context reuse.

Earlier **partial confirmation review 02**:
[index](confirm-review-index-02.json), [ratings](confirm-source-judgments-02.json),
raw `$BASE/quality-confirm-checkpoint-02/`. It captures twelve B01 matched blocks
and two B02 blocks: 280 objective finals, 168 automatic reference failures,
24 capped intermediates, **237 selected outputs / 59 full contexts**. All 57 new
QA full passages/questions/references/answers were explicitly read and recorded
in four `confirm-qa-judgments-02-part-*.json` files. Most mechanical QA failures
are valid, source-supported full sentences that differ from shorter references.
One new QA adds unsupported Persian etymology; its five-waters answer stays
useful. With earlier fictional provenance/cap issues, selected counts are
**52 source failures, 52 instruction failures, zero not useful**, including
duplicate contexts. These are selected repeated-cohort counts, not population
rates. Creative usefulness is separated from grounding. No original objective,
timing or workload change. This is explicitly incomplete acquisition; preserve
all earlier snapshots. Its exact-context reuse has since been extended by the
newer snapshots below; use the latest completed canonical index for future work.

**Partial confirmation snapshot 03 is fully reviewed**, raw
`$BASE/quality-confirm-checkpoint-03/`. It captures full B01/B02 plus two B03
matched blocks: 856 objective finals, 635 prescribed selected outputs and
24 capped intermediates, **74 distinct full contexts**. All **20 new contexts**
were explicitly read: sixteen full source/reference/answer document summaries
in eight `confirm-summary-judgments-03-part-*.json` files and four QA contexts in
[QA judgments 03](confirm-qa-judgments-03.json). Summary instructions require
at most 300 words, key facts/uncertainties and no unsupported additions; no tools
or dependencies. Four summaries fail source fidelity: treating disputed gun and
police accounts as settled, overstating a regulator's investigation status,
reversing Taylor Swift's response and misidentifying an evacuee as staying.
Three summaries exceed 300 words in total: two additional source-supported
summaries and one source-failing summary. All retain useful content, separately
from these failures. Earlier prose incorrectly said three additional summaries;
the explicit ratings and counts were correct and remain unchanged. Index 04
records the correction while preserving index 03 as history.
[Canonical index 03](confirm-review-index-03.json) and
[ratings 03](confirm-source-judgments-03.json) cover all 635 selected items:
**76 source failures, 84 instruction failures, zero not useful**, including
duplicate fixed-cohort contexts. These selected counts are not population rates
or quality equivalence. [Validator receipt](confirm-quality-checkpoint-03.json)
checks all judgments in an isolated materialized copy, using the identical
archived quality module; live acquisition/analysis is untouched. This validates
the partial snapshot, not final confirmation. Context reuse checks eight fields
and equality of the entire archived executed instruction source; canonical IDs
are rebound to the first matching item in this packet. This base is now extended
by canonical confirm04 below. Old 2080 explicit judgment
lists alone lack those bindings and cannot supply unread-context judgments.

Earlier **partial confirmation review 04**:
[index](confirm-review-index-04.json), [ratings](confirm-source-judgments-04.json),
[validator receipt](confirm-quality-checkpoint-04.json), raw
`$BASE/quality-confirm-checkpoint-04/`. It captures all B01/B02/B03 and three B04
matched blocks: **1,464 objective finals, 750 automatic failures, 24 capped
intermediates, 892 selected outputs / 75 full contexts**. One new Burke birthplace
QA was fully read and correctly answers Dublin, Ireland in valid JSON;
[explicit judgment](confirm-qa-judgments-04.json). All other contexts exactly
reuse prior complete source/instruction/answer judgments. Archived executed
instruction snippets and file hashes were directly verified in all 66 included
attempts, with canonical IDs rebound to this packet. Selected counts:
**77 source failures, 85 instruction failures, zero not useful**. These include
duplicate contexts and certify neither population rates nor quality equivalence.
The frozen preparation/finish algorithms ran against isolated derived output;
the live campaign's analysis/quality inventory was checked unchanged. Index 04
also records the earlier summary-word-limit prose correction: three overlength
summaries total, including one source failure; individual ratings unchanged.
Its canonical base is extended by snapshot 05 below. Acquisition and final
confirmation review remain incomplete.

Earlier **partial confirmation review 05**:
[index](confirm-review-index-05.json), [ratings](confirm-source-judgments-05.json),
[validator receipt](confirm-quality-checkpoint-05.json), raw
`$BASE/quality-confirm-checkpoint-05/`. It captures all B01/B02/B03/B04 and the
first B05 matched block: **2,344 objective finals, 1,502 automatic failures,
24 capped intermediates, 1,644 selected outputs / 83 full contexts**. All eight
new repository final contexts were explicitly read, with complete source/test
references and every actual preceding tool/dependency output. Four
`confirm-repository-judgments-05-part-*.json` files retain those judgments.
All eight contain useful supported content. Seven miss the required underscore
status format; the remaining final uses the accepted format but invents an
empty-input limitation for chunked_even, contradicted by its source and passing
empty-input test. Actual test passes, formatting compliance and source fidelity
are judged separately; irrelevant broad rg results do not erase full test-class
evidence. Selected counts are **79 source failures, 101 instruction failures,
zero not useful**, with duplicate contexts counted. All 86 included archived
instruction sources verify; frozen finish validates all 1,644 judgments in an
isolated materialized workspace. The live quality directory and acquisition
remain unchanged. A post-preparation diagnostic assumed reference.kind existed;
repository references omit that optional field. The diagnostic was corrected
and the already-complete snapshot retained, without rerunning acquisition or
rewriting snapshots. That checkpoint used canonical screen07 plus confirm05 for exact reuse.
This is partial acquisition, not the final confirmation audit or population
quality/equivalence evidence.

Earlier **partial confirmation review 06** covers all B01/B02/B03/B04/B05
and the first valid B06 four-system block:
[index](confirm-review-index-06.json), [ratings](confirm-source-judgments-06.json),
[isolated frozen-validator receipt](confirm-quality-checkpoint-06.json), raw
`$BASE/quality-confirm-checkpoint-06/`. It contains **2,552 objective finals,
1,682 automatic final failures, 24 capped intermediates and 1,833 selected items
across 93 complete contexts**. All ten novel repository final contexts were
explicitly read, including full code/tests, instructions and actual preceding
dependencies/tools. Six `confirm-repository-judgments-06-part-*.json` files retain
the judgments. All ten retain useful content; three have unsupported claims
(noninteger chunk-size infinite loops/incorrect chunking, None wrapping allegedly
confirmed by test_none, and an empty-input limitation contradicted by a test).
Eight fail instructions; supported final behavior and actual test outcomes are
assessed separately from preceding errors and literal status formatting.
Selected counts: **87 source failures, 287 instruction failures, zero not useful**,
including duplicates. All 112 included archived instruction sources verify and
the isolated frozen validator accepts all 1,833 judgments. The live quality
directory is unchanged. The first B06 block remains excluded by frozen selection;
this review does not repair its missing activity. Use canonical screen07 plus
confirm06 for that checkpoint's exact-context and instruction-binding reuse. Final
confirmation review still needs a terminal inventory; no population quality or
equivalence claim follows.

Earlier **partial confirmation review 07** includes all seven valid B06
blocks after all 32 scheduled B06 attempts became terminal:
[index](confirm-review-index-07.json), [ratings](confirm-source-judgments-07.json),
[isolated validator](confirm-quality-checkpoint-07.json), raw
`$BASE/quality-confirm-checkpoint-07/`. It covers **2,744 objective finals,
1,845 automatic final failures, 24 capped intermediates, 2,011 selected items
and 123 complete contexts**. All 30 novel dependency chains/answers were
explicitly read and judged. [Component bindings](confirm-review-components-07.json)
verify exact equality of the already explicitly read complete source/reference,
all task instructions, current instruction, role and actual tool results; no
judgment was automatically assigned. Six new manual parts retain full eight-field
context signatures. Three novel finals make unsupported limitation claims
(noninteger chunk-size loops/incorrect chunking, zero-size resource exhaustion,
or an unexplained complex-iterable restriction); 27 fail instructions. All 30
retain useful content. A status clause citing a genuinely passing test_none was
distinguished from claiming test_none proves singleton wrapping.
Selected duplicate-inclusive counts are **101 source failures, 459 instruction
failures, zero not useful**. All 136 included archived instruction sources verify
and frozen finish validates all 2,011 judgments in isolation. Original timings,
objective results, frozen exclusions and live quality directory remain unchanged.
That checkpoint used canonical screen07 plus confirm07 for exact-context reuse. This
is still partial acquisition, not final audit or population quality evidence.

The earlier **partial confirmation review 08** adds all B07 repetitions and the
first B08 four-system block:
[index](confirm-review-index-08.json), [ratings](confirm-source-judgments-08.json),
[isolated frozen validator](confirm-quality-checkpoint-08.json), raw
`$BASE/quality-confirm-checkpoint-08/`. It covers **3,000 objective finals,
2,017 automatic final failures, 28 capped intermediates, 2,243 selected items
and 162 complete contexts**. All 39 novel PEP contexts were explicitly judged:
35 finals and four capped intermediates. All sixteen distinct supplied PEP source
texts were read in full within their supplied boundaries (ten whole documents,
six opening excerpts). [Source bindings](confirm-source-components-08.json)
link exact repeated source text to its first read context; every novel reference,
task/current instruction, actual dependency/tool chain and answer was read.
Fourteen `confirm-investigation-judgments-08-part-*.json` files retain the explicit
judgments. No novel source/instruction/usefulness rating was automatic.

Among novel contexts, 18 lack source support, 25 fail instructions and six are
not useful. Findings retain an unfinished enormous-count search command, three
210-word repetitive capped intermediates, a repetitive final with a wrong
creation year and no limitation, CSV/pickle subject confusion, unsupported
whole-document absence/runtime claims and loss of a retracted-idea distinction.
Supported historical rules and useful fragments are retained. Five-field
metadata validation does not verify feature implementations; source support
here is confined to supplied evidence, not all knowledge outside the packet.
Selected duplicate-inclusive counts are **133 source failures, 658 instruction
failures and six not useful**. All 164 included archived instruction sources
verify; frozen finish accepts all 2,243 judgments in an isolated workspace.
Live quality state, measurement timings, objective scores and the B06 exclusion
remain unchanged. This checkpoint used canonical screen07 plus confirm08 for exact full
context/instruction reuse and is extended by confirm09 below. This is still a partial audit, not final campaign
completion or a population quality/equivalence result.

The earlier **partial confirmation review 09** includes four completed B08
four-system blocks: [index](confirm-review-index-09.json),
[ratings](confirm-source-judgments-09.json),
[isolated validator](confirm-quality-checkpoint-09.json), raw
`$BASE/quality-confirm-checkpoint-09/`. It has **3,192 objective finals, 2,060
automatic final failures, 34 capped intermediates and 2,337 selected outputs /
181 complete contexts**. All 24 novel reference/instruction/dependency/tool/answer
contexts were explicitly read and judged in six new manual parts. Their thirteen
complete supplied source strings exactly match prior explicitly read confirm08
sources, with [component bindings](confirm-source-components-09.json) preserving
full/opening boundaries. No novel rating was assigned automatically.

Novel counts are **15 source-support failures, 19 instruction failures and six
not useful**. Findings retain fake tool summaries/validation, wrong-subject
Decimal/logging/C API restrictions, a false FAIL inferred from rejection,
unsupported Unicode/compatibility claims, repetitive output and an empty final.
Correct source fragments and useful partial answers remain separate from errors.
A correctly copied PASS/PEP number can be source-supported yet not useful;
historical transition plans are assessed as source accounts, not independent
runtime proof. Ambiguous compatibility caveats are read narrowly as empirical
verification limits, not claims that the PEP lacks a compatibility discussion.

Selected duplicate-inclusive counts are **159 source failures, 725 instruction
failures and twelve not useful**. All 176 included archived instruction sources
verify; frozen finish accepts all 2,337 judgments in isolation. Live quality and
original timing/objective/acquisition state are unchanged. The first B06 block
remains excluded, and this is not a final audit or population quality/equivalence
result. This checkpoint used canonical screen07 plus confirm09 for exact full-context
and archived-instruction reuse and is extended by confirm10 below.

The earlier **partial confirmation review 10** includes **all twelve B08
four-system blocks**: [index](confirm-review-index-10.json),
[ratings](confirm-source-judgments-10.json),
[isolated validator](confirm-quality-checkpoint-10.json), raw
`$BASE/quality-confirm-checkpoint-10/`. It covers **3,704 objective finals,
2,174 automatic final failures, 53 capped intermediates, 2,479 selected outputs /
220 complete contexts**. All fifty new contexts (31 finals, 19 capped
intermediates) were explicitly judged in ten manual parts. Their eleven supplied
source strings exactly match earlier explicit readings; all unmatched reference,
instruction, actual dependency/tool and answer components were read. The
[component receipt](confirm-review-components-10.json) binds complete repeated
fields and individual preceding dependencies to prior explicit readings, including
earlier contexts within this snapshot. No novel rating is automatic.

Novel counts are **39 source failures, 45 instruction failures and 28 not
useful**. Findings retain wrong PEP/subject answers, repetitive/truncated output,
false FAIL inferred from rejection, validation invented before tools, unsupported
version/Unicode/compatibility restrictions and incorrect required API methods.
Correct sourced facts can still fail instruction/usefulness, and useful fragments
remain separate from errors. Bold markup containing literal `test_status=PASS`
is accepted; a bold status label followed by a colon does not supply that literal.
One capped response has five words and mostly whitespace; cap does not imply a
word-limit failure. Historical source accounts and header-only validation are
kept distinct from runtime proof.

Selected duplicate-inclusive counts are **193 source failures, 860 instruction
failures and 38 not useful**. All 208 included archived instruction sources verify;
frozen finish accepts all 2,479 judgments in an isolated materialized workspace.
Live quality/objectives/timings remain unchanged; the first B06 block stays
excluded. This is still partial campaign acquisition, not population quality or
equivalence. This checkpoint used canonical screen07 plus confirm10 for exact complete
context/instruction reuse and is extended by confirm11 below.

The historical **partial confirmation review 11** adds the first complete B09
four-system block: [index](confirm-review-index-11.json),
[ratings](confirm-source-judgments-11.json),
[isolated validator](confirm-quality-checkpoint-11.json), raw
`$BASE/quality-confirm-checkpoint-11/`. B09 is a PEP-investigation fixture;
the transient chat description calling it team synthesis was incorrect.
It covers **3,768 objective finals, 2,190 automatic final failures, 55 capped
intermediates, 2,545 selected outputs / 234 complete contexts**. Fourteen contexts
absent from confirm10 (twelve finals, two capped intermediates) were explicitly
read and judged in four new parts; one exactly repeats an older confirm09
context, with identical boolean judgments, and thirteen are new across earlier
confirmation snapshots. Eleven supplied source strings and repeated complete
fields/dependency components exactly match earlier explicit readings;
[bindings](confirm-review-components-11.json) retain this without automatic ratings.

The group has **four source failures, eight instruction failures and one not
useful**: fake tool-output source excerpts, an unsupported modern comparison,
a vague complex-package exclusion and repetitive output, with formatting failures
separate from source fidelity. Supported historical enumerate start/stop and
class-decorator proposal scope remain. Correct metadata can be useful partial
content while still omitting a specification rule/concrete limitation. Selected
duplicate-inclusive counts are **212 source failures, 894 instruction failures
and 39 not useful**. All 212 included archived instruction sources verify and
the frozen validator accepts all 2,545 judgments in isolation. Live quality,
objectives and timings are unchanged; the excluded first B06 block remains.
This is not final acquisition or a population accuracy/equivalence result. Use
canonical screen07 plus confirm11 for future exact full-context/instruction reuse;
retain all older snapshots, including contexts temporarily outside the selected
sample. A reappearing exact historical context may reuse its explicit judgment
after full binding and current-ID checks.

The historical **partial confirmation review 12** includes all twelve B09 four-system
blocks and the first B10 native pair (98 selected blocks):
[index](confirm-review-index-12.json), [ratings](confirm-source-judgments-12.json),
[component bindings](confirm-review-components-12.json),
[isolated validator](confirm-quality-checkpoint-12.json), raw
`$BASE/quality-confirm-checkpoint-12/`. It covers **4,480 objective finals,
2,337 automatic final failures, 76 capped intermediates and 2,777 selected
outputs / 291 complete contexts**. All 59 contexts new versus the union of
screen07 and confirm01 through confirm11 were explicitly read and judged across
eighteen manual parts. Fifteen supplied source strings match prior explicit
readings; four team-source/test bodies first appear in this snapshot and were
read in full. Repeated fields and individual preceding dependencies are exactly
bound. Four contexts independently reviewed in cap04 have identical boolean
judgments. No novel rating was automatic.

The new group has **36 source failures, 47 instruction failures and 14 not
useful**. Findings include wrong proposals, fabricated validation transcripts,
unsupported runtime or whole-document absence claims, word caps and missing
literal status. B10 synthesis is checked against the actual specialist findings
and executed tests; claims that no tests exist, or that rejected non-Sequence
input is untested, contradict the supplied evidence. Correct source facts may
remain useful despite formatting failures. Historical proposal scope and
source/test coverage limits remain explicit.

Selected duplicate-inclusive counts are **264 source failures, 1,085 instruction
failures and 54 not useful**. All 258 archived instruction sources verify, and
the frozen validator accepts all 2,777 judgments in isolation. Objectives,
timings and live quality are unchanged; first B06 remains excluded. This is
partial acquisition, not a population accuracy/equivalence result. Future reuse
should bind the union of screen07 and all completed confirmation indices to
avoid losing older contexts temporarily absent from the selected sample.

The historical **partial confirmation review 13** includes all twelve B10 native
pairs and two B11 four-system blocks (111 selected blocks):
[index](confirm-review-index-13.json), [ratings](confirm-source-judgments-13.json),
[component bindings](confirm-review-components-13.json),
[isolated validator](confirm-quality-checkpoint-13.json), raw
`$BASE/quality-confirm-checkpoint-13/`. It has **4,824 objective finals, 2,542
automatic final failures, 76 capped intermediates and 3,062 selected outputs /
341 complete contexts**. All fifty new contexts were explicitly read in thirty
manual parts: 39 investigation finals, eight team syntheses and three QA finals.
New unmatched source/test components were read in full; prior complete fields
and individual dependencies exactly bind confirm08 through confirm12 or preceding
confirm13 readings. Earlier static reference preparation does not replace these
actual-answer reviews. No novel judgment was automatic.

The new group has **21 source failures, 45 instruction failures and zero not
useful**. Findings include copied restrictions from unrelated broad search output,
blanket custom-type exclusions, duplicate/empty-input claims contradicted by tests,
test-private executor state confused with implementation, and missing literal
status. Narrow TypeError/coverage/storage limitations retain credit where supported.
Three QA answers use the required one-string-field JSON and supported full-sentence
answers; their automatic normalized exact-match failures remain unchanged.

Selected duplicate-inclusive counts are **332 source failures, 1,244 instruction
failures and 65 not useful**. All 288 archived instruction sources verify; the
frozen validator accepts all 3,062 judgments in isolation. No timing/objective or
live quality change, no first-B06 comparison credit, no final acquisition or
population accuracy/equivalence claim. Reuse only exact complete contexts and
archived instructions from the union of completed indices, retaining all history.

The historical **partial confirmation review 14** includes all eight B11 four-system
blocks and the first B12 block (118 selected blocks):
[index](confirm-review-index-14.json), [ratings](confirm-source-judgments-14.json),
[component bindings](confirm-review-components-14.json),
[isolated validator](confirm-quality-checkpoint-14.json), raw
`$BASE/quality-confirm-checkpoint-14/`. It covers **5,624 objective finals,
3,191 automatic final failures, 78 capped intermediates and 3,720 selected
outputs / 398 complete contexts**. All 61 new contexts were explicitly read and
judged in twenty manual parts: 58 investigation finals, two capped intermediates
and one QA final. All fourteen supplied source strings exactly match earlier
explicit readings; every new reference/instruction/dependency/tool/answer was
separately read, with full-field and individual-dependency bindings checked.
No novel judgment was automatic.

The added group has **18 source failures, 60 instruction failures and one not
useful**. Unsupported duplicate/custom-type/hashability restrictions, performance
validation and infinite-loop claims remain distinct from source-supported type,
coverage and tracking-memory limitations. A claim of unique locks contradicts
the explicit shared-lock test and is not useful; another final correctly resolves
prior lock errors against actual evidence. Both capped intermediates retain useful
source facts but exceed the under-70 limit and invent performance-test coverage.
The JSON QA answer is supported and follows format while retaining its automatic
short-reference exact-match failure.

Selected duplicate-inclusive counts are **440 source failures, 1,612 instruction
failures and 67 not useful**. All 316 included archived instruction sources verify;
the isolated frozen validator accepts all 3,720 judgments. The selected B11
recovery blocks follow the unchanged frozen selector; original partial/PATH-failed
attempts remain without comparison credit. First B06 remains excluded. No original
objective, timing or live quality change; no population accuracy/equivalence or
final-acquisition claim. Continue exact full-context/instruction reuse from the
union of all completed indices, retaining every historical snapshot.

The latest **partial confirmation review 15** includes all twelve B12 blocks
(129 selected Qwen3-8B blocks): [index](confirm-review-index-15.json),
[ratings](confirm-source-judgments-15.json),
[component bindings](confirm-review-components-15.json),
[isolated validator](confirm-quality-checkpoint-15.json), raw
`$BASE/quality-confirm-checkpoint-15/`. It covers **5,976 objective finals,
3,476 automatic final failures, 78 capped intermediates and 4,024 selected
outputs / 429 complete contexts**. All 32 new investigation finals were read
and judged across five manual parts. Their five complete supplied sources exactly
bind prior explicit readings; every new dependency and answer was separately read.
Bindings include confirm05 through confirm14 and earlier confirm15 contexts.

The new group has **five source failures, 31 instruction failures and zero not
useful**. Two answers wrongly cite test_none as proof of singleton wrapping;
another denies an explicit error message. Unsupported infinite-input failure and
custom-iterable restrictions remain visible. Documented callback_iter deprecation
and abort behavior receive source credit. Thirty finals omit literal status; one
has it but supplies an unsupported limitation. Useful partial facts remain distinct.
Selected duplicate-inclusive counts are **459 source failures, 1,908 instruction
failures and 67 not useful**. All 360 included archived instruction sources verify;
the frozen validator accepts all 4,024 judgments in isolation. Objectives,
timings, live quality and first-B06 exclusion remain unchanged. No population
quality/equivalence or final-acquisition claim. Reuse the union of screen07 and
confirm01 through confirm15, with exact complete-context/instruction binding.

The [anchor inventory helper](../../benchmarks/research_checks/stage6_5080_qwen8b_complete.py)
was run from SOURCE with literal PY; its receipt already exists, so do not rerun
to overwrite it. [Validation](anchor-inventory-validation.json) records complete
inventory assertions and bytecode/no-overwrite guards. A misspelled audit-only
bytecode flag generated 16 disposable caches; they were moved to
`$BASE/generated-validation-cache-02`. The original 4,606-file bundle and
2,192-file frozen source identity reverify unchanged; no acquisition change.

[Repository reference preparation](confirmation-repository-reference-review.json)
checks all eight B05 source/test reference snippets against their pinned files
and AST bodies. Complete bodies, task turns and tool definitions were explicitly
read: SequenceView, adjacent, all_unique, always_iterable, always_reversible,
callback_iter, chunked_even and circular_shifts. Notes retain boundary behavior
and actual test coverage to avoid inventing limitations from the broad search
listing. This is static preparation, not generated-answer review or a new test
execution. In particular, callback_iter has an explicitly gated running-future
test, and chunked_even tests zero/negative sizes without consuming the iterator.
Actual generated answers and preceding tool/dependency outputs must still be
read before any future source judgment; do not assign ratings from this note.

[PEP reference preparation](confirmation-pep-reference-review.json) explicitly
reads all sixteen B08 supplied contexts, reference headers, task turns and tool
definitions. Ten contain the complete original document; six are opening-section
excerpts. All file hashes, prefix boundaries and five reference header fields
match the pinned files. The actual frozen document tool source was also read:
validation compares PEP/Title/Status/Type/Created to references; summarization
returns metadata, length/word counts and heading names. Neither executes the
specified Python feature or verifies its semantic rules. No campaign tool was
run or generated answer rated by this preparation. Notes retain historical
rejected/superseded status, excerpt scope and specification/version distinctions;
do not substitute modern Python knowledge for this supplied evidence. Future
generated answers and actual preceding tool/dependency outputs still need review.

Confirmation and final all-phase cap audits remain pending. Review any new
confirmation contexts explicitly, reusing screen07 and the union of completed confirmation indices only on exact full context
and system-instruction equality. Prepare the final confirmation packet after
acquisition is terminal; do not invent unread judgments. Final all-cap inventory
must be regenerated after confirm/overhead/operational are terminal and cover
every ordinary sealed attempt, including interrupted/failed arms. Its root is
BASE (containing only this campaign alias), not VOLUME (which also contains the
closed November 8 campaign). Expanded cap signatures additionally bind executed
system instructions and prompt context. All timings/objective results are
unchanged. Preserve every earlier review snapshot as history.

## Supervision

Health schedule: `scheduled-task:d1de183b-4ef5-498a-aafe-ba015a2805ae`, every
15 minutes in this thread. Reverify ownership, mount, disk, GPU and request/stream/
ledger progress. Run `benchmarks.research_checks.stage6_healthcheck` from mutable
REPO with canonical OUT and `--state-dir "$BASE/confirm-health-01"`; the first
post-activation check passed. Its source uses the now-present protocol lock and
primarily assesses confirmation. Healthy long
acquisition returns to unattended operation; a finished phase advances immediately.
Interrupted checks are not completion. Keep the whole original objective active.
