# Results maintenance design

## Goal
Agree on a reusable closeout procedure that turns investigations into concise,
project-specific results and leaves the active codebase maintainable.

## Authorized
Design only. Philip requested a proposal for a shared skill with project-specific
result formats. Historical evidence migration, storage/backup changes and skill
implementation wait for this decision. Agent operations and benchmarks come last.

## Done when
Philip has chosen the result format, closeout procedure and pruning boundaries.

## State
Proposed structure:
- FINDINGS.md becomes a short index of questions, current conclusions and result links.
- One result per research question or feature, with a descriptive title, verdict,
  chart when useful, short explanation, tested scope, limitations and evidence links.
- Compact numerical data and chart-generation inputs accompany figures; a chart
  alone is insufficient evidence. Failures and inconclusive results use the same format.
- Raw attempts and frozen execution inputs remain separate and immutable. Result
  summaries can be corrected with a clear correction note and source reference;
  materially new experiments get a new result and update the index.
- A shared closeout skill reads the project's tracked result profile: required
  fields, visual conventions, evidence location and allowed cleanup. It invokes
  existing project tools rather than embedding another runner.
- Closeout verifies claims, renders the result, updates the index, identifies code
  to retain/remove, applies authorized cleanup, checks the remaining project and
  closes the task. Completion is not conditional on a positive result.
- Rejected feature implementations may leave active source when removal is within
  approved scope. Preserve the exact source identity and required reproducibility
  dependencies before removal. Raw artifacts and valuable shared data are not
  automatically deleted. Useful reusable tests/tools survive when still relevant.

The current immutable-evidence convention and blanket binary-file ban need explicit
reconciliation with the chosen result/figure format before implementation. A shared
skill supplies the procedure; the project supplies its presentation and retention
choices. No new evidence hierarchy has been installed yet.

## Next
Review the proposal with Philip, then implement the agreed format and apply it to
one existing result before migrating the remaining relevant historical evidence.
