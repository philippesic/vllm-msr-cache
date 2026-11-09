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
any resume. Never launch a duplicate. At 2026-11-09 17:01 UTC, seven of 22 whole
screen blocks were complete; B08 upstream was active, with Continuum/candidate
already verified. Nineteen screen attempts were verified, one deliberately
interrupted, one active. The verified count includes the original unmatched B01
ARC, which receives no matched-comparison credit. No additional raw failed
attempts were present. Check the live ledger rather than treating this snapshot
as a current monitor.

Completed prerequisites:

- [Build](BUILD.md), [source/toolchain/wheel receipt](lmcache-build.json): same
  LMCache 0.3.6 release, all 178 Python files unchanged, sm_120 plus PTX, separate
  Continuum environment, same package versions. Exact pinned CPU/GPU round trip.
- [Short serving smoke](continuum-smoke.json): warmup and four requests succeeded
  with GPU KV stores/pinned geometry/cleanup. Original one-session policy result
  stays inconclusive; this diagnostic is excluded from campaign measurements.
- [Full Continuum screen](continuum-full-screen.json): B06 passed all 32 requests
  and 24 scheduler pins, with sealed rebuilt-native identity and clean release.
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

Latest completed partial review: [index 03](screen-review-index-03.json) and
[ratings 03](screen-source-judgments-03.json), raw snapshot
`$BASE/quality-screen-checkpoint-03/`. All 231 prescribed selected B01–B04 items
have explicit judgments, covering 76 full contexts; 16 novel summary contexts
were read in full in parts 01–04. Snapshot objective coverage is 258 finals.
Selected-item findings: 30 source-unsupported, 32 instruction failures, ten not
useful, with complete-context duplicates counted separately. These are selected
sample counts, not population rates or quality equivalence. Original objective
scores, timing and gates are unchanged. Earlier snapshots remain preserved.

For the next snapshot use a fresh checkpoint directory (04 next), regenerate
`prepare_review(OUT,'screen')` from exported source and copy its derived packet.
Reuse judgments only on exact equality of all signature fields listed in index
03; explicitly read/judge every novel required context. Do not invent missing
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
