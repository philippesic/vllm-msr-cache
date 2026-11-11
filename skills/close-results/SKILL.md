---
name: close-results
description: Close an investigation or feature experiment by turning its evidence into a concise project result, updating the findings index, and pruning authorized obsolete code. Use the project's result profile; do not invent a new experiment or treat inconclusive evidence as a rejection.
---

# Close results

Read the repository instructions and its result profile, normally `docs/results.md`.
The profile owns format, chart conventions, validation commands, artifact locations
and cleanup boundaries. If no profile exists, propose the smallest useful format
before migrating records. Do not impose another project's presentation choices.

Use this for durable lessons or decisions that future work needs. Routine fixes
belong in Git history and current documentation; they do not need result pages.
The shared convention is current explanations in `docs/`, unfinished work in
`tasks/`, conclusions in `results/NAME/result.md`, and short conclusion/link entries
in `FINDINGS.md`. JSON, figures and archived reports are project-specific choices,
not universal requirements. Follow the profile's authoring and generation rules.

1. Identify the question, executed implementation, comparison and scope. Read the
   underlying report and measurements, including failures and adverse cases. Mark
   imported research separately from results for current code. Do not silently
   pool experiments, change denominators or manufacture uncertainty intervals.
2. Write the result in the project's format: conclusion first, useful figure or
   table, short interpretation, limitations, decision and traceable sources. Keep
   scope explicit; retain numeric inputs and the rendering recipe with figures.
   A successful run does not imply a beneficial feature. An inactive mechanism leaves its value unresolved.
3. Update the findings index to the current conclusion. Correct explanatory text
   explicitly; preserve immutable execution records. New experiments receive new
   result identities. Follow the profile for replacement and correction lineage.
4. Review active code and documents against the decision. Remove discarded feature
   code only within authorized scope, after preserving its exact source identity
   and needed reproduction inputs. Keep useful shared tools and tests. Do not
   delete raw attempts, active dependencies or valuable data merely to make a task
   look finished. Record unresolved research as a bounded conclusion, not an active
   instruction to resume it.
5. Run project result validation and appropriate code checks. Exercise source
   recovery and rendering when introducing a format. Close the task only after the
   result, current topic docs and any authorized pruning are complete. Delete the
   completed task file; Git retains its history. If blocked, rewrite its current
   state and next step instead. State missing evidence plainly.

Do not create a parallel runner, a new benchmark campaign or a maintenance schedule.
The skill organizes evidence; it does not grant authority to publish or spend GPU
resources. Evaluate its usefulness through ordinary work and repair observed faults.
