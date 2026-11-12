---
name: close-results
description: Close a meaningful investigation or feature experiment by verifying evidence, writing a concise project result, updating findings and pruning authorized obsolete code. Use for investigation closeout, not routine fixes or merely finishing a run.
---

# Close results

Read the project instructions and result specification, normally `docs/results.md`.
It owns presentation, figure conventions, evidence locations, validation and cleanup
boundaries. Use the project's pinned procedure revision when present. If a result
specification is missing, propose the smallest useful format before converting
records; do not impose another project's scientific or chart requirements.

1. Establish the question, comparison, executed implementation and allowed claim.
   Read supporting measurements and reports, including failures and adverse cases.
   Verify substantive claims against execution evidence rather than a producer's
   verdict alone. Distinguish current qualification from imported research. State
   missing provenance and its limits; do not manufacture precision, change
   denominators or silently pool experiments. An inactive mechanism leaves its
   value unresolved; inconclusive evidence is not rejection or equivalence.
2. Produce a conclusion-led result in the project's format. Use a chart or table
   when it helps, followed by interpretation, tested conditions, limitations and
   the decision: retain, revise, discard or unresolved. Preserve compact numerical
   inputs and a regeneration recipe for figures. Identify exact executed source,
   including relevant dirty changes, and dependencies needed to support the claim.
3. Update the short findings index with the scoped conclusion and result link.
   Keep raw execution records immutable. Make corrections to summaries explicit;
   materially new questions or evidence get new results under the project rules.
   Group related runs into a coherent investigation, not one result per session.
4. Review implementation and current documents against the decision. Prune only
   authorized obsolete code after preserving recoverable source and required
   inputs. Keep useful shared tools and tests. Do not delete raw attempts, active
   dependencies or valuable data just to finish a task. Missing reproduction
   inputs prevent unsupported reproducibility claims and destructive cleanup.
5. Run project result validation and appropriate code checks after cleanup. Exercise
   source recovery and rendering when introducing or changing those procedures.
   Close the assigned task only after the result, findings, affected current docs
   and authorized pruning are complete. Delete the completed task; Git preserves
   its history. If blocked, rewrite current state and the concrete next action.
   A bounded investigation can close with an unresolved scientific conclusion.

Current guidance belongs in `docs/`, unfinished work in `tasks/`, durable lessons
in `results/`, and their short index in `FINDINGS.md`. Routine maintenance belongs
in code, current docs and Git history; it does not need a result page. Report what
actually ran, what was reused and what remains untested.

This procedure grants no new publication, compute or deletion authority. It does
not create another runner, benchmark campaign, maintenance schedule or agent review.
