# RTX 5080 rebuild and full Stage 6 pass

Authorized 2026-11-09: rebuild the same pinned LMCache 0.3.6 source for
`12.0+PTX`, prove Continuum serving, then rerun the entire primary campaign
with unchanged design. Continue autonomously through audits and pushed results.
Preserve the closed November 8 campaign and all 2080 Ti artifacts. No training,
Stage 7, unrelated projects or other machines; no credentials in tracked output.

Repository: `/home/philip/vllm-semantic-cache-ext`; branch
`stage6-5080-20261109`; push its established user-owned origin at checkpoints.

Current phase: **whole screen resumed**, persistent session 65450, log `output/stage6-5080-20261109/screen-02.log`,
ownership receipt `screen-owner-02.private.json`. Use its recorded PID/birth/cwd
and actual process state; never duplicate a live coordinator.

The deliberate recovery audit is now **complete**: the first B01 ARC arm and
interrupted candidate arm remain sealed, and both replacement arms passed in
one new matched block. [Whole-block recovery proof](recovery-complete.json)
verifies owned cleanup and full replacement. The coordinator advanced to B02.
The [original interruption proof](recovery-interruption.json) remains the
point-in-time receipt before replacement. No original partial-block measurement
is credited. The watcher is finished; do not restart it or repeat preflight.

The first preflight launch was stopped before any ledger attempt because its
cwd was mutable source. The retained launch helper now validates exact exported
source, mount and interpreter before exec into the unchanged coordinator.

## Established evidence

- [Transfer seal audit](transfer-seals.json): original primary/addendum/completion
  packet, execution bundle and new imported bundle all match their original
  seal receipts and materialized file hashes. No historical artifacts changed.

- [Native preflight](preflight.json): all three restore/fresh-reference fixtures
  and the anchor registry probe passed; all four attempts / 223 files verify.

- [Build](BUILD.md) and [receipt](lmcache-build.json): same release revision,
  all 178 Python files unchanged, separate Continuum environment, CUDA 12.8,
  sm_120 plus PTX. Exact FP16 pinned CPU/GPU round trip passed.
- [Live Continuum smoke](continuum-smoke.json): warmup, four requests, actual
  GPU KV stores, pinned geometry and cleanup succeeded. Generic policy verdict
  remains inconclusive because the one-session diagnostic produced no scheduler
  pins; unchanged full-screen activation gates still apply. Diagnostic excluded
  from campaign timing/qualification.
- [Import](import.json): all 22 cells resolved; every worker explicitly binds
  the original pinned local interpreter. Same parent protocol/order/repeats,
  all profiles/corpora identical; plan differences only the new Continuum
  interpreter and rebuilt LMCache wheel reference. 54 screen configurations,
  524 confirmation configurations / 210 blocks, 38 overhead, six operational.
- Exported CPU validation passed 240 benchmark tests, 31 lightweight tests and
  compilation. Generated caches from validation were moved outside the sealed
  bundle; every original sealed content hash verifies. Future validation uses
  separate PYTHONPYCACHEPREFIX/pytest cache paths.
- [History integrity](history-integrity.json): all old 41 attempts / 3,142 files,
  4,606 execution-bundle files, original LMCache binary unchanged. Both original
  and rebuilt Continuum environments have identical package versions.

## Paths and commands

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

`$BASE/primary-01` is a convenience alias. Use canonical `$OUT` for process
matching. Verify the existing ext4 mount at every check/resume; never format it.
The old FILESYSTEM_REPAIR.md records the image/remount recipe. Monitor free
space on ext4, WSL root and Windows. At import: roughly 17 GiB, 6.9 GiB, 8.6 GiB
free respectively. Build/toolkit caches and raw attempts are retained on ext4.

For each phase launch, run from the project root using the retained exec helper
with **new** log/receipt filenames, e.g. (do not launch while another is active):

```bash
"$PY" -m benchmarks.research_checks.stage6_5080_launch --phase screen   --log "$BASE/screen-01.log" --receipt "$BASE/screen-owner-01.private.json"
```

The helper forces cwd `$BUNDLE/source`, pinned `$PY`, exact source identity and
mounted ext4, then execs the existing campaign CLI with its normal GPU reservation
and owned-child cleanup. Add `--resume` only after checking stopped resources.
To call other campaign commands, **cd into `$BUNDLE/source` first** and use `$PY`.
Do not run acquisition from mutable checkout or inherit a test interpreter.

## Remaining sequence

1. Preflight is complete; do not repeat it.
2. Whole-block B01 recovery is complete and verified. Continue all remaining
   screen cells in their frozen order; do not restart the recovery watcher.
3. After all 54 screens, `campaign qualify --out "$OUT"`; inspect
   `all_required_correctness_passed`. Diagnose every genuine failure. Do not
   activate with failed correctness/activation/capacity/workload-purpose gates.
   Known native restore-inactive GPU-reuse controls retain original status and
   use attested measurement_validity; never relabel them pressure evidence.
4. `campaign activate-import`, then confirm, overhead and operational using
   the existing frozen order/repeats. Never freeze/replan, tune policy or swap
   models/workloads/geometry. The full 5080 pass remains unfinished until all
   phases and audits finish, or an irreparable genuine failure is documented.
5. Verify every sealed attempt and transfer. Run campaign report/completion,
   objective/blinded source and every capped-output audit, recovery/portability
   proofs. Retain adverse results. Report geometric mean throughput ratios with
   paired whole-process 95% CIs, candidate vs ARC and vs Continuum, and actual
   latency figures. No quality or production equivalence claim without evidence.
6. Update RESULTS.md/STATUS, commit/push, release owned servers/GPU locks, then
   disable this health schedule. Do not stop at a normal process exit or launch.

Health schedule: `scheduled-task:d1de183b-4ef5-498a-aafe-ba015a2805ae`, every
15 minutes in this thread. A healthy long GPU phase returns to unattended
operation. Before protocol activation inspect the ledger/owned processes and
raw request/stream progress directly; the existing stage6_healthcheck requires
an activated protocol. After activation, use its bounded check with canonical
OUT and a state directory under BASE. Fix setup/harness/disk/filesystem failures
and continue. A scheduler interruption is not completion; reverify and resume.

Incremental answer review began during screening: `screen-review-index-01.json`
and `screen-source-judgments-01.json` here cover two identical-context B01 finals
(one distinct complete context), read without system/speed labels. The retained
raw packet is `$BASE/quality-screen-checkpoint-01/`. This is a partial review,
not a finalized screen or confirmation audit. Later inventories must be regenerated
from completed acquisition; reuse these judgments only on exact full-context
signatures, and explicitly review every novel required item.

The next incremental screen review is retained in `screen-review-index-02.json`
and `screen-source-judgments-02.json`: 112 selected finals from completed B01/B02
blocks, 60 distinct complete contexts (59 novel QA contexts explicitly read).
The objective packet covers 130 finals. The raw snapshot is
`$BASE/quality-screen-checkpoint-02/`. Four selected responses (two duplicated
contexts) have unsupported content; two also fail semantic instruction adherence.
The elapsed-year conversion and source-inconsistent mandolin count remain
flagged. Exact-match scores, all timings and gates are unchanged. These remain
partial screen judgments; final confirmation/all-cap audits are still required.

Screen advanced through B05 with no additional failed executions; its first full
Continuum arm is now B06 (inspect actual ledger at next check). The source-review
snapshot `$BASE/quality-screen-checkpoint-03/` covers completed B01–B04 blocks:
258 objective finals, 231 selected items and 16 novel summary contexts beyond
checkpoint 02. `screen-summary-judgments-03-part-01.json` retains explicit full-source
judgments for novel indices 0–3; **indices 4–15 remain unread/pending**. Continue
those from the retained novel-contexts.json; never silently infer judgments for
missing contexts or finalize the 231-item packet until they are reviewed. The
first four include overlength/repetitive output and material attribution errors.
All original objective scores/timings/gates remain unchanged.
