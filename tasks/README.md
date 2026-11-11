# Tasks

One file per open task that spans more than one session or agent. The agent doing
the task owns its file; other agents read it. Delete the file in the commit that
completes the task, and record any established result in [FINDINGS](../FINDINGS.md).
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
