# Results and closeout

Use [close-results](../skills/close-results/SKILL.md) when an investigation or
feature experiment reaches a conclusion, including a negative or unresolved one.
This tracked copy is the portable source of the shared skill. Install it for local
Codex discovery by copying its directory to `~/.codex/skills/close-results/`; compare
existing contents before replacing an installed version. Project-specific choices
live here, not in the shared skill.

## Shared organization

Use these roles across projects; each project's profile chooses its presentation:

| Location | Owns |
| --- | --- |
| `README.md` | Introduction, setup and navigation. |
| `AGENTS.md` | Rules needed on every task and links to detailed procedures. |
| `docs/` | How the project works now and why; replace obsolete explanations. |
| `docs/results.md` | Result format, closeout procedure and storage boundaries. |
| `tasks/NAME.md` | Unfinished multi-session work; rewrite state, delete at completion. |
| `FINDINGS.md` | Short conclusions linked to results, including failures and uncertainty. |
| `results/NAME/result.md` | A durable answer to one coherent question. |
| Beside the result | Useful figures, small figure inputs and supporting assets. |
| `results/sources/` | Archived supporting reports needed to preserve traceable evidence. |
| `skills/close-results/SKILL.md` | Portable closeout procedure; project choices live here. |
| Git history | Implementation changes and completed task history. |

A result is warranted when a future developer needs the lesson, evidence or decision
without reconstructing the investigation. Routine fixes and document edits need no
result page. Do not create one per run, commit or session. Keep negative and unresolved
conclusions as readily discoverable as successful ones.

Across projects, charts, JSON and source archives are conditional on usefulness;
a concise Markdown result can suffice. This project's existing research reports use
the structured format below to check source recovery and reproduce figures. That is
a local format choice, not a requirement to build the same tooling in every project.

## Result format

`FINDINGS.md` is a generated short index. Each `results/NAME/result.json` owns one
question or coherent study. It contains schema, title, date, origin (`current` or
`historical`), verdict, summary, scope, limitations, decision, sources and optional
chart data. `result.md` and `chart.svg` are generated views, not separate facts.
Do not edit generated pages or the index directly. Titles should state the conclusion
so FINDINGS remains useful without duplicating the full report.
Keep the explanation lean enough to read in a minute; preserve distinctions that
change the conclusion rather than recounting the investigation's chronology.

A source records repository, full revision, relative path and SHA-256. Its exact
report content is archived in `results/sources/SHA256.json` and checked locally. Historical
sources use the earlier repository's exact inspected revision. Current results
also identify a committed report; the report links executed snapshots. These are
report identities, not an assertion that the report commit executed the experiment.

A chart uses a single metric/unit and explicit baseline. Each point has a label,
estimate and optional lower/upper bounds. Its caption explains repetitions,
intervals and direction. Keep independent studies separate; do not turn no interval
into a zero-width confidence interval. SVG keeps figures reviewable and avoids a
blanket exception for binary artifacts. Functional or unresolved findings may use
prose without a figure. Original detailed reports remain recoverable through their
source identity, even when only compact measurements are carried into this project.

## Closing work

1. Read FINDINGS and relevant results before investigating; explain why closed work
   needs reopening. Keep unfinished work in a task file when it spans sessions.
2. At closeout, invoke [close-results](../skills/close-results/SKILL.md) with the task,
   question and evidence locations. Produce a result only when it retains a useful lesson.
3. Present the conclusion, useful chart or table, short interpretation, scope,
   limitations, decision and sources. Retain compact inputs with the result; link
   large raw runs precisely instead of importing them into Git.
4. Update current topic docs and remove discarded source only within authorized
   scope. Keep the result even when the implementation is removed.
5. Regenerate and validate the result and FINDINGS, then delete the completed task
   in the same checkpoint. If blocked, rewrite its State and Next instead.

## Commands

- `uv run --locked python tools/check.py --render-results`: regenerate result pages,
  charts and FINDINGS from the records.
- `uv run --locked python tools/check.py --component results`: check schemas,
  source identities, generated-file hashes and index membership; also runs in the
  normal local check. This checks structure and freshness, not scientific truth.
- `uv run --locked python tools/check.py --verify-result-sources PATH`: verify every
  referenced report against a Git repository or bare archive at PATH. Sources from
  other repositories are reported separately, not silently declared verified.

Run full acceptance before committing. Inspect rendered figures and confirm that
numbers and qualifications match the cited report. Keep failed/incomplete results.

## Corrections and pruning

Raw execution records stay immutable. Correct a result's explanation or extracted
numbers in place with a nonempty correction note describing the reason and source;
Git retains the previous version. Materially new experiments get a new result ID.
An old result remains useful when it establishes a failure or a boundary. Remove
redundant summaries only after their useful conclusions and sources are retained.

Remove discarded implementations from active source only within approved scope,
after ensuring their exact revision and reproduction dependencies remain available.
This does not authorize deleting the previous repository, raw experiments or paused
work. Historical candidates were not imported as active features here, so their
negative results do not require copying and then deleting their implementations.

## Coverage and recovery

[Coverage](../results/coverage.json) maps the earlier evidence collections to these
results or an explicit exclusion. It records the inspected snapshot, not a claim
that every raw attempt was reanalyzed. Storage and restoration are described in
[recovery](recovery.md). Agent operations and new benchmark design remain separate.
