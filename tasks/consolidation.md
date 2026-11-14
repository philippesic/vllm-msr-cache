# Consolidate the project as wait-hint-kv

## Goal
Keep the maintained project, useful historical evidence and both Git histories in
one repository named wait-hint-kv, with obsolete external copies removed.

## Authorized
Philip on 2026-11-12: consolidate evidence and artifacts, graft onto the old project
keeping both histories, publish main and the next-pre-graft-main tag to the public
remote, and rename the repository and current references to wait-hint-kv. On
2026-11-13: delete the frozen checkouts and stale copies, move all work to the 2080 Ti
checkout with everything contained in it, delete the old 2080 and Mac checkouts once
ported, leave a .git-only Mac folder, add a favicon and reconcile T3 projects.
Philip renamed the GitHub repository to wait-hint-kv. No force-push or visibility
change.

## Done when
The checkout, package, environments and legacy data live in ~/wait-hint-kv on the
2080 Ti and the documented checks pass there; old checkouts and retired hosted
repositories are gone.

## State
On 2026-11-13 the old checkout ~/vllm-semantic-cache-ext, its T3 worktrees and old
venvs were deleted after every ref was confirmed present here (30 newer T3 checkpoint
refs were archived first). Environments moved to artifacts/venvs and legacy data to
artifacts/legacy (see docs/recovery.md). The vLLM fork's own commits are archived in
refs/archive/vllm-fork/patches. The remote is philippesic/wait-hint-kv.

Two retired hosted repositories remain: philippesic/vllm-semantic-cache-ext-next and
philippesic/vllm-semantic-cache. Deletion needs the delete_repo scope, which the
token lacks.

## Next
Philip deletes both hosted repositories (gh auth refresh -s delete_repo, then gh repo
delete for each, or the repository settings pages). Then delete this task.
