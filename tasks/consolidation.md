# Consolidate the project as wait-hint-kv

## Goal
Keep the maintained project, useful historical evidence and both Git histories in
one repository named wait-hint-kv, with obsolete external copies removed.

## Authorized
Philip on 2026-11-12: consolidate evidence and artifacts, graft onto the old project
keeping both histories, publish main and the next-pre-graft-main tag to the public
philippesic/vllm-msr-cache remote, and rename the repository and current references
to wait-hint-kv. Relayed again on 2026-11-13 ("execute all 3 of its plan"): the
rename, deleting the frozen checkouts and the stale private hosted copy. No
force-push, remote or visibility change.

## Done when
The checkout, package and data directory are wait-hint-kv and the documented checks
pass from there; obsolete checkouts and the stale hosted copy are gone.

## State
History graft, evidence curation and publication are done (see docs/recovery.md).
On 2026-11-13 the ten frozen benchmark checkouts were deleted after every HEAD was
confirmed reachable here; the only dirty change, in benchmark-coding-05, already
exists in main. Attempts, archives, backups and benchmark assets were kept as
inert history. The checkout moved to ~/wait-hint-kv, the data directory to
~/.local/share/wait-hint-kv, and the package is named wait-hint-kv.

The stale private hosted copy philippesic/vllm-semantic-cache-ext-next still
exists: deletion returned 403 because the token lacks the delete_repo scope.

## Next
Philip deletes the hosted copy (gh auth refresh -s delete_repo, then gh repo delete
philippesic/vllm-semantic-cache-ext-next, or the repository settings page). Then
delete this task.
