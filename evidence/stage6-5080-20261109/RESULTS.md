# RTX 5080 Stage 6 — confirmation terminal; supplementary phases running

The pinned LMCache rebuild works on the RTX5080. All five Continuum screens,
all 54 screening configurations / 22 complete blocks, and functional qualification
pass. [Terminal confirmation inventory](confirmation-attempts-complete.json)
reconciles **all 524 frozen configurations**, retaining 537 attempts and verifying
**56,362 sealed files**, inner artifacts and the complete declared request/tool/
unit/cleanup coverage. There are **521 valid logical measurements and 207/210
complete matched blocks**. Only the three documented original B06 activity failures
are excluded: Qwen3-8B repeat0 and Ministral repeats0/3. No activity retry,
imputation, relabeling or repeat reduction was performed. The frozen full
throughput-acquisition completion flag remains **false**.

The independent confirmation service exited successfully and its original driver
and final coordinator are gone. The next [independent overhead service](overhead-launch-01.json)
is running the 38 frozen control configurations, using the same literal interpreter
and frozen source. Operational has six configurations after that. Final source/cap,
control/operational, integrity, campaign and portability audits/results are still
pending; the campaign and GPU allocation are **not released**. Existing 32 anchor
conditional estimates below remain measured evidence, not a final campaign pass.

Confirmation now has a retained [B06 activity failure](confirmation-b06-inactive-restore-01.json):
the first candidate process served all 32 requests and 24 tools with no caps and
clean cleanup, but recorded zero CPU restores and eight scheduler preemptions.
The frozen analysis rejects it; the zero-preemption GPU-resident-control exception
does not apply. Its four-system block therefore cannot receive comparison credit.
This is not an LMCache kernel-loading failure. All planned confirmation measurements have now executed without a favorable-activity
retry, capacity change or analysis-rule change. The three excluded blocks remain
explicitly missing; a complete required matrix has not passed.
The [finished B06 attempt inventory](confirmation-b06-attempts-complete.json)
verifies all 32 scheduled attempts and 2,648 sealed files: 1,024 completed
responses, zero caps and all coverage/cleanup checks passed. It has 31 valid
measurements and seven complete matched blocks, short of the required eight.
All planned repetitions ran once; the excluded block was not replaced.

The [reporting validation](reporting-validation-03.json) covers 28 focused tests
and a synthetic CLI check for an explicitly incomplete presentation. The helper
keeps its default completion guard; its explicit option preserves missing frozen
blocks, actual pair counts, conditional estimates and failed completion flags.
This is preparation for reporting the terminal outcome, not final measured
extraction or a repair of the B06 activity failure.

This is a new campaign. The failed November 8 campaign and all 2080 Ti history
remain unchanged. The sole declared setup deviation is rebuilding the same
LMCache 0.3.6 source for `sm_120` plus PTX. Source and algorithm are unchanged;
the rebuilt Continuum environment retains the same package versions. The study
keeps native ARC, candidate, Continuum and upstream vLLM PR #59714, with the
exported models, workloads, resource geometry, order, repeat counts and analysis.
Cross-host differences cannot be attributed to the GPU alone.

The [build receipt](lmcache-build.json) records source/toolchain/architecture and
wheel/binary identities, setup repairs and exact pinned CPU/GPU round-trip proof:
three FP16 layers, 256 tokens and 393,216 bytes, equal in both directions. The
[short serving diagnostic](continuum-smoke.json) completed warmup and four requests
with GPU KV stores and pinned geometry/cleanup. Its original one-session policy
verdict remains inconclusive and it is excluded from performance measurements.
Exported CPU checks passed 240 benchmark tests, 31 lightweight tests and compilation.

The [five Continuum screen receipts](continuum-all-screen.json) record B06/B08/
B09/B11/B12 completing 32/64/64/80/32 requests with 24/30/35/42/24 scheduler pins.
Every seal and cleanup passes. This repairs the original warmup kernel blocker
across all frozen Continuum screen fixtures; it does not establish a speed gain.

[Native preflight](preflight.json) passed restore/fresh-reference token equality
and namespace/idempotence/future/stale checks for every model plus the registry
probe. [Qualification](qualification.json) passes native correctness, external
activation and pinned geometry, screen completeness, all three ARC pressure
gates, and the roomy reduction. [Activation](activation.json) preserves the exact
imported order, supplementary schedules, repeat/precision choices and acquisition
source, without replanning. This is functional, not production qualification.

The [completed-cell checkpoint](confirmation-three-cells.json) retains
all frozen repetitions for Qwen3-8B chat (B01), independent QA (B02) and document
summaries (B03). The [earlier two-cell receipt](confirmation-first-cells.json)
remains unchanged. The
candidate/ARC throughput ratio is a geometric mean of whole-process paired
ratios, with log Student-t 95% CIs; serving seconds exclude startup. All three
intervals include parity, so none resolves a throughput gain or loss. The
[current interim display](early-three-cells.html) and
[render receipt](early-three-cells-render.json) cover these three cells only.
[Extraction validation](early-cells-validation.json) records actual complete-cell
execution, exact prior-data preservation and guards against overwriting history
or importing from the wrong source directory.

| Cell | Process pairs | Candidate mean s | ARC mean s | Candidate/ARC throughput [95% CI] |
| --- | ---: | ---: | ---: | --- |
| B01 | 12 | 33.457 | 33.502 | 1.0014 [0.9989, 1.0038] |
| B02 | 6 | 6.220 | 6.268 | 1.0078 [0.9962, 1.0195] |
| B03 | 12 | 45.641 | 45.512 | 0.9972 [0.9896, 1.0048] |

Latency figures are milliseconds, averaged over per-process statistics.
Mean process p95 is not a pooled request p95 or a population-tail guarantee.

| Cell / system | Mean end-user TTFT | Mean process p95 TTFT | Mean completion | Mean process p95 completion |
| --- | ---: | ---: | ---: | ---: |
| B01 / candidate | 76.241 | 164.217 | 1068.861 | 1521.612 |
| B01 / ARC | 81.246 | 187.266 | 1076.579 | 1525.527 |
| B02 / candidate | 2822.843 | 5447.739 | 3137.215 | 5656.632 |
| B02 / ARC | 2837.121 | 5463.019 | 3153.108 | 5672.586 |
| B03 / candidate | 20807.810 | 39655.551 | 25531.067 | 44184.825 |
| B03 / ARC | 20754.448 | 39535.872 | 25461.058 | 44050.939 |

All 648 matched request texts and finish reasons agree between native arms
within these three fixed cohorts (1,296 recorded responses total). Each B01 arm
has 72 responses / 12 caps; each B02 arm has 384 responses / zero caps; each B03
arm has 192 responses / zero caps and 63,768 output tokens.
None of these cells records returning CPU restoration or eviction-affected returns;
these are controls, not evidence of pressured CPU-offload benefit. Output
identity is conditional on these requests and does not establish answer
correctness, population quality equivalence or results for other cells/systems.
The full 524-configuration confirmation and later phases remain incomplete.

Development pressure observations are retained with their scope:

- [Anchor Qwen3-8B](anchor-screen.json): ARC had one CPU-eviction-affected return;
  roomy ARC had zero, satisfying the prespecified reduction check.
- [Ministral3-3B](ministral-pressure-screen.json): ARC had 35 affected returns
  among 48 returning requests; candidate had 29/48.
- [Qwen3-4B](qwen4b-pressure-screen.json): ARC had 41/48; candidate had 38/48.
- [B10 candidate control](screen-team-control.json): all 20 requests completed
  without CPU restores/evictions. Its raw inconclusive status remains. The frozen
  analysis accepts the complete GPU-resident control, without pressure credit.

[Screen integrity](screen-complete.json) verifies all 6,160 sealed files across
58 verified, one deliberately interrupted and one raw inconclusive/failed attempt
in screen plus preflight. The unmatched original B01 ARC receives no comparison
credit. [Whole-block recovery](recovery-complete.json) passed owned cleanup and
replacement of both B01 arms. [Transfer seals](transfer-seals.json) and
[old-history integrity](history-integrity.json) pass; original evidence is retained.

The ongoing [confirmation integrity snapshot](confirmation-integrity-checkpoint-05.json)
verifies 26,096 sealed files, including 279 verified attempts, one interrupted
attempt and nine raw failed states (including accepted controls and the excluded
B06 attempt). All 2,192 frozen acquisition-source files still match the lock.
The [earlier checkpoint](confirmation-integrity-checkpoint-02.json)
also verified the 4,606-file execution bundle
unchanged. Active unsealed work and later attempts are outside each snapshot;
the ledger was unchanged during the latest check and the exact coordinator
remained live. All older checkpoints remain; final campaign completion is pending.

The [final screening source audit](screen-quality-complete.json) validates all
1,248 prescribed selected outputs across 522 distinct full contexts, alongside
objective checks on all 1,194 screening finals. It retains 401 source-support
failures, 725 instruction failures and 111 outputs judged not useful, counting
duplicate contexts separately. These are selected-review counts, not population
error rates. Failures include malformed/fenced JSON, wrong or incomplete answers,
unsupported additions/limitations, invented or denied validation, and overlong or
unfinished intermediates. Correct useful fragments are separately recognized.
[Explicit judgments](screen-source-judgments-07.json) bind the actual preceding
dependencies/tools and archived executed system instruction. The frozen audit
validator passed on an isolated materialized screen snapshot; the active
confirmation analysis was not changed. All timings and original objective results
remain, and quality equivalence is not established.

The [partial all-attempt cap review](caps-all-review-index-04.json) covers
**196 sealed capped responses / 119 extended contexts**: 132 screen, 64
confirmation; 177 intermediates and nineteen finals. It includes ordinary
individual arms outside matched blocks. Thirty-three responses are added since
[cap03](caps-all-review-index-03.json): 29 exactly match prior explicit confirmation
judgments, and four complete new contexts were explicitly read. One correctly
copies actual PASS but is not useful; another invents PASS before validation.
An eleven-word malformed repeated-token response is capped without exceeding
the whitespace-word limit. All four new contexts are not useful.

Selected duplicate-inclusive counts are **136 source-support failures, 196
instruction failures and 73 not useful**. Archived instructions in all sixty
cap-bearing attempts, sealed corpus/node/source/reference/role/tool branches,
objective fields and raw prompt/workload-source digests were separately checked.
Tokens were bound, not individually decoded. These are selected cap counts,
not population error rates. The inventory still has a pending attempt and is
not finalized; it must be regenerated after every acquisition phase finishes.
Timings, objective results and frozen exclusions remain unchanged.

The historical [partial cap review](caps-all-review-index-05.json) contains **215
responses / 135 contexts**: 132 screen and 83 confirmation, 194 intermediate and
21 final responses. Nineteen occurrences were added; three repeat an earlier
extended signature and sixteen exactly match full contexts already explicitly
judged in confirm12. No new answer rating was automatic. Selected counts are
**152 source-support failures, 214 instruction failures and 79 not useful**.
Source fidelity, requested format and usefulness are assessed separately.
[Bindings](caps-instruction-bindings-05.json) check all 68 cap-bearing archived
instruction sources and sealed corpus/node/prompt identities. Raw tokens were
bound, not individually decoded. Acquisition remains active and one attempt was
pending; this is not a final cap inventory or population error rate.

The historical [partial cap review](caps-all-review-index-06.json) contains **217
responses / 137 contexts**: 132 screen and 85 confirmation, 196 intermediate and
21 final responses. All earlier judgments exactly reuse cap05; the two additions
exactly match the explicitly reviewed confirm14 intermediates. They retain useful
facts but unsupported performance-test claims and intermediate word/command failures.
Selected counts are **154 source-support failures, 216 instruction failures and
79 not useful**. [Bindings](caps-instruction-bindings-06.json) check all 69
cap-bearing archived instruction sources and independently reconstruct actual
preceding dependencies/tools from sealed requests and tool files. Raw prompts
were bound, not individually decoded. Failed/interrupted/unmatched arms remain
included. One attempt was pending; the all-phase final inventory is still required.

The historical [partial all-attempt cap review](caps-all-review-index-07.json) covers
**418 responses / 175 extended contexts**: 132 screen and 286 confirmation,
370 intermediates and 48 finals. All 24 novel full contexts (19 intermediates,
five finals) were explicitly read and judged; 217 occurrences exactly reuse
cap06 and 59 exactly reuse completed confirm16 contexts. The 24 manual judgments
apply to 142 repeated occurrences. No unread or automatically rated novel answer
is included. [Component bindings](caps-review-components-07.json) retain exact
prior readings and every newly read dependency; [raw bindings](caps-instruction-bindings-07.json)
independently reconstruct corpus, node, instruction, objective, preceding requests,
tools and prompt/source identity for all 106 cap-bearing archived attempts.

Duplicate-inclusive findings are **264 source-support failures, 417 instruction
failures and 91 not useful**. Among the 24 novel contexts the respective counts
are 16, 24 and three. Incorrect test-result claims and boundary predictions remain
visible alongside useful source-grounded fragments. The supplied PEP 307 explicitly
supports its classic-class pickling restriction; that point is not charged as an
error. Source failure there is the incorrect mapping-population operation.
Ordinary failed/interrupted/unmatched responses are included, including the invalid
Ministral B06 arms. [Independent reproduction](cap-review-validation-07.json)
passes. Tokens were bound, not individually decoded. This is a partial snapshot
with one pending attempt, not the final all-phase inventory or a population error
rate; timing, objective results and frozen exclusions remain unchanged.

The latest [partial all-attempt cap review](caps-all-review-index-08.json) contains
**480 responses / 182 extended contexts**: 132 screen/348 confirmation,
418 intermediates/62 finals. All 418 prior occurrences are retained. There are 473
exact prior extended-context occurrences (55 added repetitions) and seven new
extended contexts exactly matching explicitly reviewed confirm17 quality contexts;
**no new full quality context or automatic novel rating**. [Raw bindings](caps-instruction-bindings-08.json)
independently verify all 117 cap-bearing archived instructions, sealed corpus/node/
role/reference/tool branches, actual dependencies/tools, objective and prompt/source
identities. A separate [reproduction](cap-review-validation-08.json) regenerates
all three records exactly. Duplicate-inclusive findings are **315 source failures,
479 instruction failures and 99 not useful**, not population rates. Ordinary
failed/interrupted/unmatched arms remain included; tokens were bound, not decoded.
Acquisition was active with one pending attempt. This remains a partial snapshot;
final all-phase cap inventory and finish are required. Timing and objective results
remain unchanged.

The [historical partial confirmation review](confirm-review-index-11.json) adds the
first B09 four-system block after all twelve B08 blocks: **3,768 objective finals,
2,190 automatic final failures, 55 capped intermediates and 2,545 selected outputs
/ 234 complete contexts**. B09 is another PEP-investigation fixture. Fourteen
contexts absent from confirm10 (twelve finals, two capped intermediates) were
explicitly reviewed: thirteen new across prior confirmation snapshots and one
exact older context with the same judgments. Eleven supplied source strings,
repeated complete fields and individual preceding dependencies exactly match
prior explicit readings; [bindings](confirm-review-components-11.json) retain
this, with every new component/answer read separately and no automatic rating.
The [earlier completed-B08 snapshot](confirm-review-index-10.json) remains.

The added group has four source-support failures, eight instruction failures
and one output not useful. Findings retain fake tool-output excerpts, unsupported
modern comparisons, vague package restrictions and repetitive output. Historical
source scope, actual header validation, literal status formatting and usefulness
are assessed separately. Correct partial metadata can remain useful while
omitting the requested specification rule or concrete limitation. Earlier
wrong-subject, invented-validation, cap, QA, fictional-provenance and summary
findings remain in their snapshots, including the summary-word-count correction.

Selected duplicate-inclusive counts are **212 source-support failures, 894
instruction failures and 39 outputs not useful**. The
[isolated frozen validator](confirm-quality-checkpoint-11.json) accepts all 2,545
judgments; all 212 included archived instruction sources verify. These are
selected partial-review counts, not population error rates or quality equivalence.
The first B06 block remains excluded. Original objectives, timings and live
quality state are unchanged. Final confirmation audit is still pending.

The [historical partial confirmation review](confirm-review-index-12.json) includes
all twelve B09 blocks and the first B10 native pair: **4,480 objective finals,
2,337 automatic final failures, 76 capped intermediates, 2,777 selected outputs
/ 291 complete contexts**. All 59 new contexts versus the union of completed
earlier reviews were explicitly read. Fifteen supplied source strings match
earlier readings; four new team-source/test bodies were read in full.
[Bindings](confirm-review-components-12.json) retain exact repeated fields and
individual dependencies. Four contexts also match earlier explicit cap ratings.

The new group has **36 source failures, 47 instruction failures and 14 not
useful**. Wrong proposals, fabricated tool outcomes, unsupported runtime or
whole-document absence claims, caps and missing literal status remain visible.
B10 synthesis sometimes repeats claims that tests are absent despite actual
passes, or misses a boundary case already tested. Supported facts and legitimate
coverage limits retain credit independently of formatting failures.

Selected duplicate-inclusive counts are **264 source failures, 1,085 instruction
failures and 54 not useful**. All 258 archived instruction sources verify; the
[isolated frozen validator](confirm-quality-checkpoint-12.json) accepts all 2,777
judgments. These are partial selected-cohort counts, not population error rates
or quality equivalence. Original objectives, timings, live quality and the first
B06 exclusion remain unchanged. Final confirmation and all-phase cap audits
remain pending.

The [historical partial confirmation review](confirm-review-index-13.json) includes
all twelve B10 native pairs and two B11 four-system blocks: **4,824 objective
finals, 2,542 automatic final failures, 76 capped intermediates and 3,062 selected
outputs / 341 complete contexts**. Fifty new contexts—39 investigation finals,
eight syntheses and three QA finals—were explicitly read. Exact repeated fields
and preceding dependencies are retained in [bindings](confirm-review-components-13.json).

The new group has **21 source failures, 45 instruction failures and zero not
useful**. Incorrect restrictions copied from unrelated tests, unsupported custom
type exclusions, duplicate/empty-input claims contradicted by tests and test
internals confused with implementation remain visible. Correct behavior and
narrow coverage/storage limits retain usefulness independently of missing literal
status. Three supported QA answers obey JSON instructions but differ from the
short exact references; those automatic failures remain recorded.

Selected duplicate-inclusive counts are **332 source failures, 1,244 instruction
failures and 65 not useful**. The [isolated validator](confirm-quality-checkpoint-13.json)
accepts all 3,062 judgments and all 288 archived instruction sources verify.
This is a partial selected-cohort audit, not a population error rate or quality
equivalence. Timings, objectives, live quality and first-B06 exclusion are unchanged.
Final confirmation and all-phase audits remain pending.

The [historical partial confirmation review](confirm-review-index-14.json) includes
all eight B11 four-system blocks and the first B12 block: **5,624 objective
finals, 3,191 automatic final failures, 78 capped intermediates and 3,720 selected
outputs / 398 complete contexts**. All 61 new contexts (58 investigation finals,
two capped intermediates and one QA final) were explicitly read and judged.
Fourteen supplied source strings match prior readings; exact full-field and
individual-dependency [bindings](confirm-review-components-14.json) preserve
those matches, with every new component and answer separately reviewed.

The new group has **18 source failures, 60 instruction failures and one not
useful**. Unsupported input exclusions, performance validation, infinite-loop
predictions and a unique-lock claim remain visible. Source-supported type,
coverage and memory limits retain credit. Both capped intermediates contain
useful source facts but exceed the intermediate word limit. A supported JSON
answer differs from the short exact reference, whose automatic failure remains.

Selected duplicate-inclusive counts are **440 source failures, 1,612 instruction
failures and 67 not useful**. All 316 archived instruction sources verify; the
[isolated frozen validator](confirm-quality-checkpoint-14.json) accepts all 3,720
judgments. These are partial selected-cohort counts, not population rates or
quality equivalence. Original objectives, timings and live quality are unchanged;
failed/partial B11 setup attempts and first B06 remain outside comparison credit.
Final confirmation and all-phase audits remain pending.

A [declared orchestration repair](confirmation-host-interruption-02.json) records
the tool-host interruption after 298 valid confirmation configurations / 112
matched blocks. The recorded coordinator and all workers were absent, with no
GPU compute clients and free reservations. The interrupted B11 arm was sealed
and verified; its partial comparison has no timing credit. An independent user
service resumes the complete B11 block and remaining cells through the unchanged
frozen CLI, in original order. Earlier complete blocks are retained, and the
activity-invalid B06 block is deliberately not retried. This is setup recovery,
not a capability failure or study-design change. Acquisition remains in progress.

The first independent service exposed a [tool-PATH setup error](confirmation-tool-path-repair-03.json):
its default environment could not find the existing ripgrep executable. Eight
attempts failed and a ninth was deliberately interrupted after partial serving;
all nine seals and cleanup verify, and none receives timing credit. The corrected
service inherits the verified tool PATH and checks ripgrep/pinned pytest before
GPU acquisition. The service environment probe and live coordinator PATH checks
pass. After repair, upstream and Continuum each completed all 80 requests and 48
tools with clean cleanup and verified seals; the complete replacement block was
still pending at that checkpoint. Affected B11 comparisons are replaced as whole
blocks through the frozen CLI; all earlier evidence and the B06 activity
exclusion remain unchanged.

The subsequent [whole-block recovery proof](confirmation-b11-replacement-03.json)
passes: all four systems complete **320 requests and 192 tools** in total, with
common workload units, valid measurements, cleanup and seals. The frozen selector
chooses the new block and excludes the original partial and PATH-failed attempts.
Qwen3-8B B06 remains at its original 32 attempts. At that checkpoint health had **300/524 valid
configurations and 113/210 matched blocks**, with no new issue beyond the retained
activity exclusion and setup attempts awaiting valid replacement. This proves
recovery of that block, not completion of confirmation or the campaign.

All **364 planned Qwen3-8B confirmation configurations have executed**.
The [terminal model inventory](confirmation-qwen8b-attempts-complete.json)
verifies every retained seal: **377 attempts / 35,530 files**, including original
partial and setup-failed arms. There are **363 valid logical configurations and
129 of 130 required matched blocks**. The first B06 activity failure remains the
sole missing anchor block; all 32 B06 attempts ran once, without an activity retry.
Three valid original arms remain unmatched after whole-block setup recovery and
receive no extra comparison credit. Secondary-model and later phases continue;
this is model execution completion, not a completed confirmation matrix.

The [latest partial confirmation review](confirm-review-index-15.json) covers all
twelve B12 blocks: **5,976 objective finals, 3,476 automatic final failures,
78 capped intermediates and 4,024 selected outputs / 429 complete contexts**.
All 32 new investigation finals were explicitly reviewed; their five supplied
source strings exactly match earlier readings, with every new dependency and
answer separately read. [Bindings](confirm-review-components-15.json) retain
full-field and individual-dependency matches.

The new group has **five source failures, 31 instruction failures and zero not
useful**. Wrong test_none evidence, denial of an explicit error message and
unsupported infinite/custom-iterable restrictions remain visible. Explicitly
documented deprecation and abort behavior receive credit. Selected
duplicate-inclusive counts are **459 source failures, 1,908 instruction failures
and 67 not useful**. All 360 archived instruction sources verify, and the
[isolated frozen validator](confirm-quality-checkpoint-15.json) accepts all 4,024
judgments. These selected counts are not population quality rates or equivalence.
Original objectives, timings, live quality and first-B06 exclusion are unchanged.
The [inventory validation](anchor-inventory-validation.json) also reverifies the
original source and execution bundle after audit-generated caches were moved
outside the frozen source. Final confirmation and all-phase audits remain pending.

Ministral B06 has an additional [activity-validity failure](confirmation-ministral-b06-inactive-restore-01.json):
candidate repeat 0 completes **64 requests / 48 tools / 16 units** with clean
cleanup, but zero CPU restores and 15 eviction-affected returning requests. Its
ARC peer has 14 affected returns and also zero returning restored tokens. Both
137-file seals verify. The sixteen-session secondary-model case does not satisfy
the frozen Qwen3-8B GPU-control exception; no rule/capacity change or favorable
activity retry is made. Remaining original repetitions and phases continue.
Five candidate length caps remain subject to the final all-attempt source audit.
This additional missing comparison can further prevent full-matrix completion.
The [next incremental receipt](confirmation-ministral-b06-inactive-restore-02.json)
finds the same failure in original repeat 3, with another 137-file seal verified
and four caps retained. Repeats 1 and 2 have valid complete blocks; no original
repetition is relabeled or retried. The subsequent
[terminal cell inventory](confirmation-ministral-b06-attempts-complete.json)
verifies all **twelve planned attempts / 1,644 sealed files**, with 768 completed
requests, 576 tools and clean graph/unit coverage and cleanup. Ten measurements
are valid; **four of six matched pairs** are complete. Repeats 0 and 3 remain
excluded with their valid unmatched ARC peers. All 62 raw caps await the final
all-attempt audit. The cell is finished without retrying activity failures.

The [completed partial review 16](confirm-review-index-16.json) covers **6,872
objective finals, 4,359 automatic final failures, 108 capped intermediates and
4,942 selected outputs / 561 contexts**. All 132 new contexts were explicitly
read/judged across nineteen parts: 64 QA finals, 61 investigation finals and seven
capped intermediates. Exact [component bindings](confirm-review-components-16.json)
retain 78 previously read source matches and two newly read unmatched source
strings, with every new dependency and answer reviewed.

The new group has **77 source failures, 131 instruction failures and twelve not
useful**. Fenced/malformed QA format remains separate from correct supplied facts;
unsupported elaborations and entity/date mistakes remain visible. Investigation
errors include denying actual passed tests, importing unrelated broad-search
tests, false divisibility/type constraints and invented cache/concurrency defects.
Supported fragments and narrow test-coverage limits retain credit. A final with
correct required PASS/fact/limitation but an extra false opening retains instruction
credit while failing source fidelity.

Selected duplicate-inclusive counts are **721 source failures, 2,825 instruction
failures and 99 not useful**. All 380 archived instruction sources verify, and the
[isolated frozen validator](confirm-quality-checkpoint-16.json) accepts all 4,942
judgments. The [initial 64-context progress](confirm-review-progress-16-01.json)
is preserved as history. No objective, timing, live-quality or exclusion change;
these are not population rates or quality equivalence. Later acquisition and
final all-attempt audits remain pending.

The interim Qwen3-8B extraction now retains **all 32 frozen comparisons** for
the completed anchor model: [data](anchor-results-01/comparisons.json),
[throughput and latency tables](anchor-results-01/tables.md),
[interactive display](anchor-results-01/comparisons.html),
[extraction checks](anchor-results-01/validation.json) and
[render binding](anchor-results-01/render.json). This is a model subset of the
42 planned campaign comparisons; secondary models, later phases and final audits
remain pending. The actual incomplete flags and every globally missing block
remain preserved. The frozen report writes were isolated; live report files
were unchanged. The first three published ratios, CIs, serving times and latency
summaries reproduce unchanged. All 28 focused reporting tests pass.

Candidate throughput divided by each baseline is shown below. Each ratio is a
geometric mean with a paired whole-process Student-t 95% CI, without multiplicity
adjustment. Below one means slower. B06 uses only seven valid blocks of eight;
the failed first block is not imputed, replaced or counted through its other arms.

| Qwen3-8B cell | Valid / required blocks | Candidate / ARC [95% CI] | Candidate / Continuum [95% CI] | Candidate / upstream [95% CI] |
| --- | ---: | --- | --- | --- |
| B06 | 7 / 8 | 0.9781 [0.9546, 1.0021] | 0.9477 [0.9246, 0.9714] | 1.0015 [0.9716, 1.0323] |
| B08 | 12 / 12 | 0.9857 [0.9646, 1.0072] | 0.6851 [0.6397, 0.7336] | 0.9003 [0.8690, 0.9327] |
| B09 | 12 / 12 | 0.9733 [0.9617, 0.9850] | 0.6826 [0.6558, 0.7106] | 0.9395 [0.9277, 0.9514] |
| B11 | 8 / 8 | 1.0726 [0.9478, 1.2139] | 1.0689 [0.9304, 1.2280] | 1.0014 [0.8952, 1.1203] |
| B12 | 12 / 12 | 0.9759 [0.9227, 1.0321] | 0.9248 [0.8867, 0.9645] | 0.9869 [0.9504, 1.0248] |

The candidate is slower than ARC on B09: about 2.7%, with its CI below parity.
Its apparent B11 advantage remains unresolved because the interval includes one.
Against Continuum, B06/B08/B09/B12 are slower with intervals below one; B11 is
unresolved. All twelve native-cell results, including the small adverse B07
roomy-control result, remain in the linked table. No new across-workload aggregate
is introduced. B06's native CPU-pressure evidence remains inconclusive, while
both native B09 arms have observed pressure and active CPU restores in all twelve
blocks. Missing external native counters are unavailable, not zero.

For B09, serving seconds average 43.643 candidate, 42.484 ARC, 29.840 Continuum
and 41.008 upstream. Actual latency figures follow in milliseconds; each is an
arithmetic mean of the named process statistic. A mean of process p95s is not a
pooled percentile or population-tail guarantee.

| B09 arm | Mean TTFT ms | Mean process p95 TTFT ms | Mean completion ms | Mean process p95 completion ms |
| --- | ---: | ---: | ---: | ---: |
| Candidate | 5881.409 | 13087.611 | 8820.174 | 15664.495 |
| ARC | 5492.068 | 12631.278 | 8321.630 | 15723.632 |
| Continuum | 2708.248 | 15290.478 | 4770.513 | 18386.012 |
| Upstream | 5085.238 | 13605.931 | 7797.725 | 16533.650 |

All four B09 arms complete 768 requests. Output tokens are 84,360 candidate,
84,384 ARC, 80,055 Continuum and 83,828 upstream; Continuum has 25 length caps,
the other arms zero. These are completed workload throughput and observed
latency comparisons, not correctness-adjusted useful-answer throughput. Live
outputs, runtimes and interfaces differ; source findings and caps remain visible.
No quality equivalence, universal superiority or GPU-only attribution follows.

The [terminal Ministral inventory](confirmation-ministral-attempts-complete.json)
now verifies all **60 planned configurations exactly once / 7,812 sealed files**,
including inner artifact/aggregate and full graph/tool/unit/cleanup checks.
**58 measurements and 28/30 complete pairs** are valid; B06 repeats0 and3 remain
excluded without activity retries. The immutable schedule assigns 60 Ministral
and 100 Qwen3-4B configurations, plus364 Qwen3-8B, totaling524. Qwen3-4B continues;
this is model execution completion, not campaign or final analytical completion.

The latest [source-review index](confirm-review-index-17.json) is complete for
its static 161-block snapshot: **6,153 selected outputs / 784 contexts**, including
**all 201 new full contexts explicitly read/judged** (54 QA, 147 investigation finals).
[Full components](confirm-review-components-17.json) check exact fields and actual
preceding dependencies. All 424 archived instruction sources/SYSTEM verify; all 83
supplied source strings bind earlier explicit readings. [Independent checks](confirm-review-validation-17.json)
verify every current ID/signature, duplicate judgments and complete manual coverage.
The isolated [frozen validator](confirm-quality-checkpoint-17.json) accepts all 6,153
ratings. Exact prior reuse includes 21 cap07-only contexts / 169 selected occurrences
with matching archived instructions. No novel judgment is automatic.

Novel findings are **116 source failures, 150 instruction failures and 17 not
useful**; selected duplicate-inclusive findings are **1,301, 3,786 and 153**.
Source fidelity, instruction following and usefulness remain independent: one
answer supplies correct literal PASS, a supported fact and genuine limitation
while adding an unsupported send restriction. Other answers have literal PASS but
only a false behavior/limitation. The earlier [81-context](confirm-review-progress-17-01.json)
and [132-context](confirm-review-progress-17-02.json) checkpoints remain history;
the 132 partial component records match the completed bindings. These are selected
findings, not population rates or quality equivalence. Objective results, timings,
frozen exclusions and live quality files remain unchanged. Later acquisition,
fresh terminal review and final all-phase audits remain pending.

See [continuation instructions](CONTINUE.md) for the active owner, exact paths,
remaining phases and enabled health monitoring.

The latest [partial confirmation source review18](confirm-review-index-18.json)
is complete for a static **184-block** snapshot: **9,464 objective finals,
5,746 automatic failures, 291 capped intermediates and 6,667 selected outputs /
834 full contexts**. All 55 new finals (four QA/51 investigations) were explicitly
read and judged in 14 parts. All 36 distinct supplied sources in the new contexts bind previous explicit readings;
all 470 archived instructions/SYSTEM verify. [Independent validation](confirm-review-validation-18.json)
checks every current ID/signature, duplicate judgment, manual index and exact prior
reuse, with no dependency-only judgment drift. The isolated frozen validator
accepts all 6,667 ratings; live quality and serving data remain unchanged.

The 55 new-context findings are **31 source failures, 24 instruction failures and
zero not useful**. Correct behavior fragments remain useful alongside false test
reports or invented API limitations; all four new QA answers meet source and JSON
requirements. Selected duplicate-inclusive counts are **1,411 source failures,
3,911 instruction failures and 165 not useful**, not population rates or quality
equivalence. This remains a partial snapshot. Fresh terminal confirmation quality,
all-phase cap inventory, supplementary measurements and final audits remain required.
