# RTX 5080 rebuild and full Stage 6 pass

Continue the authorized 2026-11-09 job autonomously through final audits and
pushed results. Branch `stage6-5080-20261109`, user-owned origin. Preserve the
closed November 8 campaign and all 2080 Ti history. No other machines/projects,
training, Stage 7 or credentials in tracked files. A setup/harness/storage failure
requires repair and continuation, not a capability-failure closeout.

## Current state

**Screen is running**, persistent execution session 65450. Its authoritative
ownership is in `$BASE/screen-owner-02.private.json` (PID, birth, command, cwd,
source); log `$BASE/screen-02.log`. Revalidate the actual process and GPU before
any resume. Never launch a duplicate. At 2026-11-09 17:40 UTC, 16 of 22 whole
screen blocks were complete (all Qwen cells and Ministral B02/B06/B08/B09);
Ministral B11 ARC was active. Forty-two screen attempts were verified, one
deliberately interrupted, one raw failed/inconclusive
B10 candidate control, one active. The verified count includes the original
unmatched B01 ARC, which receives no matched-comparison credit. The B10 control
served all 20 requests with clean cleanup and no errors, CPU evictions or
restores; its seal verifies. The existing frozen validity rules accept it as a
GPU-resident control, not CPU-pressure evidence. See
[control receipt](screen-team-control.json). Never retry merely to obtain restore
activity. Check the live ledger rather than treating this snapshot as current.

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
  candidate 29/48. This meets that model's ARC pressure-purpose gate; Qwen4B
  and whole-matrix qualification remain pending.
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
about 17.2 GiB ext4, 6.9 GiB WSL root and 8.5 GiB Windows; monitor all three.
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

1. Finish all 54 logical screen configurations / 22 blocks. Inspect real failures
   and measurement_validity; retained restore-inactive GPU controls are not CPU
   pressure evidence. Preserve every original status and all whole-block repairs.
2. From exported source, `campaign qualify --out "$OUT"`. Require
   `all_required_correctness_passed`; inspect all secondary pressure/roomy/native/
   external gates. Diagnose failures; do not activate a failed gate.
3. `campaign activate-import --out "$OUT"`, then confirm (524 configurations /
   210 blocks), overhead (38) and operational (six) with the helper. No freeze,
   replanning, policy/model/workload/geometry changes or favorable-result retries.
4. Verify every sealed attempt and campaign identity. Finish objective/blinded
   source and every ordinary capped-output audit, recovery and portability proofs,
   campaign report/completion, and standalone comparison chart. Render requested
   comparisons with T3 html_preview then html_render; do not start a server.
5. Report geometric throughput ratios with paired whole-process 95% CIs and
   actual latency figures. Candidate=wait. The report's Continuum/ wait ratio and
   interval can be inverted to give candidate/Continuum (invert bounds in reverse
   order). Retain uncertainty, adverse results and actual pressure. No quality,
   production or GPU-only cross-host attribution claims without evidence.
6. Consolidate RESULTS.md and STATUS, commit/push all findings, release owned
   servers/GPU/locks, disable the health schedule only at pushed completion or an
   irreparable genuine-failure closeout. An ordinary process exit is not done.

## Answer-review checkpoint

Latest completed partial review: [index 05](screen-review-index-05.json) and
[ratings 05](screen-source-judgments-05.json), raw snapshot
`$BASE/quality-screen-checkpoint-05/`. All 435 prescribed selected items have
explicit judgments, covering 147 full contexts. All 16 novel contexts were read,
including four team syntheses and one capped intermediate; additional full
PeekableTests reference content was inspected. Snapshot objective coverage is
458 finals. Selected-item findings: 86 source-unsupported, 155 instruction
failures, 16 not useful, with complete-context duplicates counted separately.
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

**Snapshot 06 is already in progress; do not regenerate or overwrite it.** Raw
`$BASE/quality-screen-checkpoint-06/` has 778 objective finals, 781 selected items,
and 142 novel full contexts relative to canonical index 05. All 65 novel QA
contexts were explicitly read and rated in `screen-qa-judgments-06-part-01..04`.
Four repository intermediates exactly match the explicitly read cap review and
are in `screen-cap-reused-judgments-06.json`. Four novel partitions finals were
read in `screen-repository-judgments-06-part-01.json`. Thus 73/142 novel contexts
are recorded and **69 repository contexts remain unread/unjudged**; the last
part lists their exact indices into `novel-contexts.json`. No index 06 or
consolidated ratings 06 exists yet. Source IDs needing review: set_partitions,
peekable, sort_together, split_when, replace, seekable, ichunked, chunked,
map_reduce, split_at, bucket, collapse, numeric_range, windowed and
distinct_permutations. Preserve source-model/arm blinding. Read each actual
dependency/tool context; broad rg snippets do not erase actual pytest results.

Finish those explicit judgments, then consolidate against canonical 05 plus the
new manual parts (exact eight-field reuse only), binding every packet review ID.
After that use a fresh checkpoint directory (07 next), regenerate
`prepare_review(OUT,'screen')` from exported source and copy its derived packet.
Verify the same executed system-instruction digest and explicitly read/judge
every novel required context. Do not invent missing
judgments or mark a partial inventory final. After acquisition regenerate final
screen/confirmation and all-capped inventories. Capped inventory root should be
BASE (which contains only this campaign alias), not VOLUME (which also contains
the closed November 8 campaign). Full required audits remain pending.

## Supervision

Health schedule: `scheduled-task:d1de183b-4ef5-498a-aafe-ba015a2805ae`, every
15 minutes in this thread. Reverify ownership, mount, disk, GPU and request/stream/
ledger progress. Before activation inspect directly; stage6_healthcheck requires
protocol.lock.json and primarily assesses confirmation. After activation run its
bounded check with canonical OUT and a state directory under BASE. Healthy long
acquisition returns to unattended operation; a finished phase advances immediately.
Interrupted checks are not completion. Keep the whole original objective active.
