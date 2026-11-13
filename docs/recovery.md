# Storage and recovery

The repository owns project source, workflow definitions, compact results and their
supporting evidence. `results/sources/` stores cited reports as inert, hash-checked
JSON; larger texts use ordered parts. Reading historical conclusions and regenerating
charts must work from this repository without an old checkout or external archive.

Generate temporary runs, logs, benchmark assets, runtime environments and temporary
checkouts under ignored `artifacts/`. Keep reusable definitions in tracked source.
Follow [results](results.md) to curate useful evidence, including failures and
inconclusive outcomes, then remove obsolete raw material. This deliberately gives up
exact raw reanalysis when it is no longer worth retaining. Shared package/model/Docker
caches are dependencies, not the sole home of authored project work or results.

## External data

Raw benchmark attempts, archives, backups and benchmark assets from before the
in-repository artifact layout live under `/home/philip/.local/share/wait-hint-kv/`.
No maintained command reads them. release-01 stopped after panel 0/7 and its
follow-up is cancelled; do not launch it. The ten frozen benchmark checkouts were
deleted on 2026-11-13 after every HEAD was confirmed reachable here and their only
local change was found in main. The old output tree and inactive agent pilot, early
qualification, overhaul and maintenance directories were pruned after useful evidence
was retained; all 31 old output collections have curated result homes and
[coverage](../results/coverage.json) records source verification and removed roots.
Useful ignored experiment code and early qualification receipts remain in Git archive
refs. Exact raw experiment replay is deliberately unavailable. Runtime environments
live under `~/.local/share/venvs/wait-hint-*`; the runtime and offload environments
install this checkout as an editable package.

Both original Git histories are being retained inside this repository under
`refs/archive/original/` and `refs/archive/retained/`, including the separate 5080
snapshot. Ignored historical scripts and patches are preserved under
`refs/archive/original/ignored-experiment-source`; the exact commit and file-count
verification live in coverage.json. They are inert historical code, not maintained
entry points or a promise of exact environment replay. History integration is verified:
original main 1319f09 is an ancestor and the graft retained the maintained tree exactly.
The checkout, package and data directory are named `wait-hint-kv`. Historical hashes,
source labels and immutable report bytes keep their original names; changing them
would break provenance.

## Verify recovery

Run `uv run --locked python tools/check.py --full`, then
`uv run --locked python tools/check.py --render-results` and confirm
`git diff --exit-code -- FINDINGS.md results` succeeds. To verify cited Git source
bytes, use `uv run --locked python tools/check.py --verify-result-sources .`.
Archived local-only reports are verified directly by result checks. Structural
validation does not establish scientific truth or missing hardware qualification.

The established public remote is `git@github.com:philippesic/vllm-msr-cache.git`.
Philip explicitly authorized publishing the grafted main, benchmark tooling and result
pages there. Push only main as a normal fast-forward and the single lightweight tag
`next-pre-graft-main`, sourced from local `refs/archive/next/pre-graft-main`. That tag
retains original NEXT hashes cited in receipts. Do not push other archive refs: ignored
experiment source, old checkpoint refs and early qualification receipts remain local
unless separately authorized. No force-push or visibility change is authorized.

The prior verified local bundle is a temporary migration recovery copy, not an
off-machine backup; remove redundant external copies after integrated Git is verified.
The stale private hosted copy `philippesic/vllm-semantic-cache-ext-next` still exists.
Deleting it needs the `delete_repo` scope on Philip's account
(`gh auth refresh -s delete_repo`, then `gh repo delete`). Its removal does not
affect local data.
