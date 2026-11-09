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
launch a duplicate. The latest bounded confirmation health check was healthy: **76/524 valid
configurations, 38/210 complete matched blocks**, with Qwen3-8B B04 ARC repeat
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
[Latest validation](reporting-validation-02.json): 21 focused reporting/estimator tests,
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
have exited and no ordinary ledger attempt remains running/planned. Require all
524 confirmation configurations / 210 matched blocks, 38 overhead configurations
and six operational configurations, with complete matched supplementary blocks
and their purpose/cleanup receipts. A zero CLI exit alone is insufficient.

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

The latest read-only [confirmation integrity checkpoint](confirmation-integrity-checkpoint-02.json)
verifies 8,563 sealed files / 111 verified, one interrupted and one raw failed
attempt at its snapshot, plus one active unsealed attempt. It checks attempt,
input and protocol identities. The exact 2,192-file acquisition source still
equals the locked manifest, all 4,606 execution-bundle files reverify unchanged,
and the coordinator ownership was checked live. This is explicitly not a final
completion audit. [Earlier checkpoint](confirmation-integrity-checkpoint-01.json)
is preserved; recording time and later ledger observation are distinct from the
CLI's sealed-attempt snapshot, not an atomic whole-campaign completion proof.

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

The expanded **partial all-attempt cap review 02** now covers every sealed
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

Latest **partial confirmation review 04**:
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
Use canonical screen07 plus confirm04 for subsequent exact-context reuse.
Acquisition and final confirmation review remain incomplete.

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
confirmation contexts explicitly, reusing canonical screen07/confirm04 only on exact full context
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
