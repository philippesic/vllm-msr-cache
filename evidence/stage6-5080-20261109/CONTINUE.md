# RTX 5080 rebuild and full Stage 6 pass

Continue the authorized 2026-11-09 job autonomously through final audits and
pushed results. Branch `stage6-5080-20261109`, user-owned origin. Preserve the
closed November 8 campaign and all 2080 Ti history. No other machines/projects,
training, Stage 7 or credentials in tracked files. A setup/harness/storage failure
requires repair and continuation, not a capability-failure closeout.

## Current state

**Confirmation is running**, persistent execution session **66082**. Ownership
is `$BASE/confirm-owner-01.private.json` (PID, birth, command, cwd, source), log
`$BASE/confirm-01.log`. Revalidate the actual process before any resume; never
launch a duplicate. The latest bounded confirmation health check was healthy: **16/524 valid
configurations, eight/210 complete matched blocks**, with Qwen3-8B B01 ARC repeat
eight active and no issues. Recheck actual live progress; these counts are a
checkpoint, not the current truth. Startup checks all three model files before
creating the first attempt; bounded disk-sleep/I/O with no child or log line
is not by itself evidence that the job stopped. Never launch a duplicate.

Screen session 65450 is **finished**. All 54 required configurations / 22 whole
blocks completed; [screen receipt](screen-complete.json) verifies 58 verified,
one interrupted and one failed/inconclusive attempt across screen + preflight,
6,160 sealed files. The count retains the unmatched original B01 ARC. B10
candidate served all 20 requests with clean cleanup and no CPU restores/evictions;
the frozen validity rules accept it as a GPU-resident control, while preserving
its original inconclusive status. [Control receipt](screen-team-control.json).
Never retry merely to obtain restore activity.

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
about 16.5 GiB ext4, 6.8 GiB WSL root and 8.6 GiB Windows; monitor all three.
Keep useful caches/raw evidence. Reconsider storage for later relocation copies
using actual sizes, not an assumption that a second raw tree will fit.

To launch another phase from the mutable project root, use the retained exec
helper with new log/receipt names, e.g. after screen qualification/activation:

```bash
"$PY" -m benchmarks.research_checks.stage6_5080_launch --phase confirm \
  --log "$BASE/confirm-01.log" --receipt "$BASE/confirm-owner-01.private.json"
```

It validates the mount, exact exported source and literal pinned interpreter,
then execs the existing coordinator with its normal reservation and cleanup.
Add `--resume` only after verifying stopped owned resources. Other campaign
commands must run after **cd into `$BUNDLE/source`** with `$PY`. Initial wrong-cwd
setup/import launches were stopped before any acquisition and preserved; do not
repeat that mistake. Never acquire from mutable source or a test interpreter.

## Remaining work

1. Supervise the existing confirmation session through all 524 configurations /
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

## Answer-review checkpoint

Latest completed partial review: [index 06](screen-review-index-06.json) and
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

A first **partial confirmation** review is also retained:
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

Confirmation and final all-phase cap audits remain pending. Review any new
confirmation contexts explicitly, reusing canonical 07 only on exact full context
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
