# Tasks

One file per open task that spans more than one session or agent. The agent doing
the task owns its file; other agents read it. Delete the file in the commit that
completes the task. If the work establishes a durable lesson, use
[close-results](../skills/close-results/SKILL.md) and [the result profile](../docs/results.md)
to retain it and update [FINDINGS](../FINDINGS.md). Routine fixes need no result page.
Update current topic docs before closing; do not move completed tasks into an archive.
Git history keeps the record of finished work. Work on a task file only when asked
to work on that task; an open task file is context, not an instruction to resume it.

Name the file `tasks/NAME.md`. The check requires exactly this shape:

```markdown
# Short task title

## Goal
What the task is for, in one or two sentences.

## Authorized
What Philip approved: scope, resources (CPU only, or GPU with host and window), limits.

## Done when
Observable completion criteria.

## State
Current facts only: branch or worktree, what is finished, what failed, blockers.
Rewrite this section; do not append a log.

## Next
The next concrete step.
```

Update the owned task after a material decision, scope change, blocker or handoff.
Keep settled constraints and their rationale concise within Authorized or State;
distinguish user decisions from proposals. Preserve valid authorization across sessions
and do not treat an unanswered proposal as approval. State should describe the current
situation, not accumulate a transcript. Continue authorized work through routine
repairs and maintenance; a blocked handoff is a fallback when further progress needs
an unavailable resource, consequential decision or additional authorization.
