# Quality scope and initial review

The Stage 1 main corpus is the existing authored repository-investigation workflow.
Its status checks compare the final reported PASS/FAIL to the actual pytest exit
code. That narrow check does not grade the repository analysis itself.

Exploratory review after Stage 1 measurement began found shared answer issues:

- `transport-proof` answers claimed that mixed outcomes, empty lookup keys and
  repeated scheduler steps were not covered. Actual tests include
  `test_mixed_request_outcomes_are_explicitly_unavailable_and_totals_are_safe` and
  `test_live_shaped_empty_lookup_keys_and_repeated_scheduler_steps_are_valid`.
- `summary-overlap` answers claimed absent multi-worker coverage while citing
  `test_multi_worker_transfer_aggregation_keeps_time_bounds_and_status`.
- Many answers exceeded the shared 70-word instruction, despite natural EOS.

These issues appear in both ARC and active-wait initial repetitions. They are
retained as quality failures, not silently regraded as correct. Their shared
presence does not prove absence of other quality differences or establish useful
code-task success. No general answer-quality claim follows from Stage 1.

`benchmarks/research_checks/qualification_quality.py` retains transparent narrow
checks and explicit contradiction flags. It is exploratory for Stage 1, frozen
before Stage 2 and independent-family evaluation. Public PEP tasks add independent
Status/Type/PEP references validated from the frozen documents by real tools.
External repository tasks name an exact function/test class and execute actual
unit tests. Those checks also remain narrower than general code reasoning.

Fresh-computation references and fixed-token replay separately test cache-path
correctness; they do not improve the underlying model's reasoning ability.

Stage 2 first ARC final-answer review (before viewing its wait counterpart) found
additional concrete shared-workload limitations: `async-scoring` claims stale
results/asynchronous publication are not explicitly tested, despite
`test_worker_async_publication_rejects_stale_generation` and
`test_completed_later_observation_waits_for_ordered_publication`.
`score-search` claims no test compares indexed/full dot scores, despite
`test_all_probes_recover_all_rows_and_exact_dot_product_ranking`, which uses exact
`torch.testing.assert_close(..., rtol=0, atol=0)`. These are manually audited
unsupported coverage claims, separate from the previously frozen two-flag
scorer; no post hoc scoring change enters performance eligibility. Final responses
also sometimes repeat bash commands when no further application tool will run.
Generated shell is never executed by this workload.

The first upstream `hint-validation` final is scored unparseable by frozen
normalized-status-v2 (15/16 normalized-status checks), although human review sees
an explicit PASS statement and JSON `test_status` equal to PASS. This is a scorer
format miss, not an observed wrong test outcome. Preserve the original grade and
record the manual adjudication separately in
`output/wait-qualification-20261105/manual-quality-adjudications.json`; do not
silently change the scoring rule after evaluation.

Independent repository evaluation: the complete first repetition of each system
was source-audited for both full-cache and cohosted cells. The fresh neutral
confirmation's first repetition reused only exact text/source matches with identical
frozen inputs. At this point 128 independent-family finals are accounted for; the
first-repetition facets and source checks are in `independent-manual-review.json`.
Every system contains invented missing-coverage or incorrect API claims under the
original instructions. Their timing remains all-offered-work evidence, with no
claim of equal-quality successful code analysis.

Two bounded instruction calibrations used the same four already-seen tasks and
never selected serving settings or timing. The adopted literal-template version
gave four supported behaviors and correct test statuses, two concrete supported
limitations and two honest missing limitations. Missing boundaries remain missing
under the original rubric. The untouched sixteen-task reserve, its identical
instruction across systems and the unchanged half-byte deployment geometry were
then frozen before fresh validation. Original evaluations are development for
this application-instruction amendment. Both calibration attempts and all original
quality failures are preserved in `instruction-calibration-audit.json` and the
raw campaigns; no timing/quality pooling or silent regrading.


The single seen-document literal-template calibration preserves four correct
metadata/test outcomes and short six-line finals, but retains unsupported or
missing limitations (notably PEP492 implementation information). It was adopted
unchanged without further calibration. Fresh full/half reserve specs were both
frozen before any reserve outcomes. The first fresh half-budget ARC case has
all 16 subject/metadata/status reports correct, supported primary facts, and
many unsupported/missing limitations. The actual metadata-only check boundary
is acceptable under the frozen rubric; saying merely that no limitation is
established is still missing. PEP593 correctly delegates annotation semantics
to consumers; vague care with PEP654 exception fields supplies no concrete rule.
The remaining first-repetition conditions will be audited under the same rubric.
Author audits remain distinct from independent human evaluation; no timing
exclusions or post hoc eligibility changes follow from these annotations.

Both fresh document byte geometries now have their complete first triplets
source-audited (96 finals total), with exact reuse only on identical source ID,
text hash and frozen corpus. Cumulative coverage 448 finals/28 cases, including
the earlier development/diagnostic first repetitions; this is not448 distinct
tasks or independent timing repetitions. Reuse identity checks pass. Correct
subject/metadata/status is narrower than useful factual completion: unsupported
implementation-absence and missing-boundary claims remain in every system.
An upstream full-budget PEP701 answer correctly states that grammar permits
nesting but implementations need not provide arbitrary depth, and is credited
under the same unchanged rubric. Conversely, no-backwards-incompatible-changes
is true for PEP655 but does not itself supply a concrete restriction. No general
equal-quality task-success or production-agent claim follows from these timings.

Stage 4 graph-workflow first ARC review, before viewing its wait counterpart,
read all 16 finals and again found an unsupported `async-scoring` absent-coverage
claim despite the stale-generation and ordered-publication tests. Generic
all-edge-cases limitations and extra shell blocks remain common. The narrow
normalized test-status checks pass 16/16 while strict-format failures are13/16;
this is not general task success. The frozen scorer and timing eligibility stay
unchanged, all offered work remains included, and the exact response/source hashes
are retained in `stage4-graph-arc-quality-note.json`. Runtime telemetry additions
are visible in source-tool answers and do not silently become a new quality rubric.

The 1.5B full-budget first ARC case's16 finals were read before its wait
counterpart. Several report no concrete code symbol (including literal None or
SUCCESS), file paths only, generic future-investigation plans and ambiguous
claims that outcomes cannot be determined despite a final PASS declaration.
Expected-error tests are sometimes called Fail while the final report says PASS.
This is a substantive model/task-quality limitation, separate from cache-path
correctness. The frozen scorer, inputs, serving settings and fixed repetitions
remain unchanged; no timing result can establish useful repository-analysis
quality from these outputs. Exact response hashes and review scope are retained
in `stage4-qwen15-arc-quality-note.json`.

In the 1.5B first pair, both cases score14/16 under the frozen normalized-status
scorer. Three of the four misses are human-readable PASS headings that match
actual successful tests. The wait `hint-retention` miss is different: it lists
`test_cpu_prefix_stops_at_first_missing_or_pending_block` in both passed and
not-passed groups despite the successful pytest outcome. This is a retained
incorrect per-test claim, not merely a format miss. Separate manual adjudications
retain exact text hashes without changing original scores or excluding work.
No equal-useful-quality or successful-task speed claim follows for this model.
