# Storage and recovery

Git tracks the code, skill, result profile, summaries, compact chart data and
archived source reports. `results/sources/` stores report content as inert JSON,
keyed by its SHA-256; result validation checks content and source metadata. These
copies let a fresh checkout inspect every curated source without the old checkout.
They do not turn old reports into active project instructions.

Raw attempts, model files, large datasets and frozen execution trees stay outside
Git. Existing local roots are:

- `/home/philip/.local/share/vllm-semantic-cache-ext-next/attempts/`: current
  functional attempts, with source snapshots and admission/terminal records.
- `/home/philip/.local/share/vllm-semantic-cache-ext-next/archives/legacy.git`:
  bare historical repository, including the pinned 5080 snapshot under
  `refs/archive/5080`.
- `/home/philip/vllm-semantic-cache-ext/output/`: earlier local experiments.
  Remote 5080 raw artifacts remain on their original host.

Do not delete these roots as part of result closeout. Report-level recovery and
chart regeneration are supported by the tracked files. Full experiment reproduction
still requires each report's original runtime, inputs and raw artifacts; a fresh
clone alone does not provide those dependencies. Raw artifacts are not claimed to
have an independent off-machine backup.

## Recover the project

A verified local Git bundle is retained at
`/home/philip/.local/share/vllm-semantic-cache-ext-next/backups/before-remote-removal.bundle`.
It contains complete history through the results-maintenance checkpoint. Clone the
bundle into a new directory, install the locked development environment, and enable
the local hook with `git config core.hooksPath .githooks`. Later commits remain in
this working repository; refresh a bundle deliberately when a new checkpoint is needed.
There is no configured remote. The previously created hosted copy awaits deletion
because the available GitHub credential lacks deletion permission; see the open
results-review task.
Run `uv run --locked python tools/check.py --full` to validate the restored project.
Run `uv run --locked python tools/check.py --render-results` and confirm
`git diff --exit-code -- FINDINGS.md results` succeeds. This checks reproducible
presentation, not the truth of each scientific claim.

The current functional verifier can independently read retained attempts with
`python -m tools.runtime_verify --verify /external/attempt` using the runtime
Python. A failed or missing raw attempt remains a limitation even if its compact
report is recoverable. Never replace missing measurements with reconstructed data.

When moving valuable raw artifacts, copy first, verify the existing receipt and
required dependencies at the destination, and only then consider deleting an owned
source copy. Active jobs and paused/shared work are outside routine pruning.
