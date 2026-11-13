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

## Current transition

Existing benchmark acquisition still uses
`/home/philip/.local/share/vllm-semantic-cache-ext-next/` and isolated runtime
venvs. release-01 stopped after panel 0/7 and its follow-up is cancelled.
Preserve the attempts, frozen checkouts and required environments for verification
and possible explicitly authorized resume; do not launch the cancelled queue.
The old output tree and inactive agent pilot, early qualification, overhaul and
maintenance directories have been pruned after useful evidence was retained and
active ownership checked. All 31 old output collections have curated result homes;
[coverage](../results/coverage.json) records source verification and removed roots.
Useful ignored experiment code and early success/failure qualification receipts
remain in Git archive refs. Exact raw experiment replay is deliberately unavailable.
The active benchmark attempts, assets, checkouts and environments still need their
final closeout and migration; they are not safe to delete yet.
The remaining consolidation is tracked in [the open task](../tasks/consolidation.md).

Both original Git histories are being retained inside this repository under
`refs/archive/original/` and `refs/archive/retained/`, including the separate 5080
snapshot. Ignored historical scripts and patches are preserved under
`refs/archive/original/ignored-experiment-source`; the exact commit and file-count
verification live in coverage.json. They are inert historical code, not maintained
entry points or a promise of exact environment replay. History integration is verified:
original main 1319f09 is an ancestor and the graft retained the maintained tree exactly.
The `wait-hint-kv` directory/package rename follows evidence closeout. Historical hashes, source labels and
immutable report bytes retain their original names; changing them would break provenance.

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
The earlier hosted `philippesic/vllm-semantic-cache-ext-next` copy was requested for
deletion; prior credentials lacked deletion scope. Its removal is separate from
preserving local data.
