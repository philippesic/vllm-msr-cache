# Checks

`uv run --locked python tools/check.py` is the fast lane: every static group and the
CPU tests in a few seconds, labeled partial. `--full` adds the rule proofs (about 30
seconds) and is acceptance; the `.githooks/pre-commit` hook runs it. Enable
the hook once per clone with `git config core.hooksPath .githooks`; no check command
changes git configuration. Neither mode runs GPU tests or builds. Validation is local; there is no Actions CI.

| Command | Purpose |
| --- | --- |
| `tools/check.py` | Fast lane while working; partial |
| `tools/check.py --full` | Acceptance: fast lane plus rule proofs; run before committing |
| `tools/check.py --self-test` | Rule proofs only: each rule rejects a bad specimen and accepts a good one |
| `tools/check.py --test tests/PATH.py` | Fast partial feedback; never acceptance |
| `tools/check.py --component NAME` | One group, e.g. `policy`, `types`, `lint` |
| `tools/check.py --update-tests` | Refresh both test inventories and print the diff |
| `tools/check.py --runtime-python PY` | Runtime lane: strict runtime types plus pinned CPU integration, one-line verdict |
| `tools/check.py --runtime-python PY --test tests/runtime/PATH.py` | Fast partial runtime feedback; never acceptance |

All commands are prefixed with `uv run --locked python`. Supported entry points are
listed in the [system map](system-map.md). The pinned check image is
[Dockerfile.checks](../Dockerfile.checks); keep `.git` in its build context because
policy reads the index.

## Common failures and fixes

| Diagnostic | Fix |
| --- | --- |
| `cpu.collection: ... extra=[...]` | You added or renamed tests. Run `--update-tests` and review the diff. |
| `cpu.collection: missing=[...]` | A required test disappeared. Restore it, or justify the removal to Philip. |
| `cpu.skipped` | Skip and xfail are not allowed. Make the test pass or remove it with justification. |
| `comments.banned`, `comments.docstring` | Delete the comment. Move intent into names, constants or tests, and rationale into the topic doc. |
| `footgun.environment` | Read environment variables only in `tools/config.py` or `wait_hint_offload/config.py`. |
| `footgun.print` | Library code returns values or raises; it does not print. |
| `layout.script` | Use an existing entry point instead of a new script or main guard. |
| `unused.code` | Delete the unused code. |
| `boundary.*`, Import Linter | Only `vllm_adapter` imports vLLM; the pure core imports only the standard library. |
| `types.suppression`, `exceptions.*` | Fix the type error. A directive needs an exact, reasoned `tools/exceptions.toml` entry and Philip's review. |
| `docs.link`, `docs.map` | Fix the link, or keep `docs/system-map.md` rows in sync with `tools/system_map.py`. |
| `docs.task` | Task files follow the template in [tasks/README.md](../tasks/README.md). |
| `repo.artifacts` | Raw outputs, logs, plans, scratch and status diaries stay out of Git. Use `tasks/` or the result format in `docs/results.md`. |

## Enforced groups

| Group | Enforcement |
| --- | --- |
| Format, lint | Ruff format (check only) and the selected Ruff diagnostics |
| Types | mypy `--strict` on the package, tools and tests; no broad exemptions |
| Imports | Only the adapter imports vLLM; contracts never import the adapter; acyclic siblings |
| Boundaries | Pure core uses the standard library only, has no import-time calls and no dynamic imports |
| Repository policy | One root AGENTS.md and a CLAUDE.md pointer; no raw outputs, copied trees, status diaries, GPU reservation code, symlinks, binaries or large files; selected secret patterns; doc links resolve; task file shape |
| Agent policy | No comments, bare excepts, library prints, stray environment access, test sleeps or standalone scripts |
| Unused | Vulture at 60% confidence; deptry dependency checks |
| Native | Fails closed if C++/CUDA/CMake material appears without qualified recipes; skipped otherwise |
| Runtime | Confirms the runtime verifier refuses to run without explicit inputs; reported as skipped |
| CPU pytest | Fixed test inventory; no missing, extra, skipped or xfailed tests |
| Rule proofs (`--self-test`) | Each project rule rejects a bad specimen and accepts a good one, in a throwaway repository. Third-party tool behavior is trusted and only its configuration is checked. |

The full rule inventory is [rules.json](../tools/rules.json), with specimens in
[check-cases.json](../tools/check-cases.json). Tool directive exceptions live in
[exceptions.toml](../tools/exceptions.toml); each needs an exact path, a reason and
an actual use. Stale entries fail.

## Scope

The portable check covers repository tooling and the pure core. It does not exercise
KV transfers, device fences or live serving; the runtime and GPU lanes in
[runtime](runtime.md) do. Native and runtime groups are reported as skipped, never as
passed. Secret scanning recognizes selected patterns only.

The runtime verifier snapshots tracked, dirty and untracked inputs into a new
attempt under ignored `artifacts/`, with separate admission and terminal records. It records the
actual Python, loaded native libraries, package versions and CUDA state.
`--verify DIR` checks a retained attempt without rerunning it. Its process helper
enforces a wall-time limit, makes no automatic retries, and only signals the
session it created.

## Choosing validation

Run the fast lane during edits and full acceptance before commits. Changes to
the adapter, hint protocol or runtime tests also need the pinned runtime CPU lane.
Changes affecting real transfers or serving need the affected hardware checks,
with GPU authorization; report any pending hardware coverage explicitly.

Portable and runtime type checks use separate caches under `.mypy_cache/`; runtime
caches are separated by interpreter path. They can run alongside each other.
Rule proofs use minimal product specimens rather than copying live product tests,
so adding a product test cannot change a proof fixture's imports or dependencies.

Result schemas, archived-source hashes, rendering freshness and index membership
are checked in the `results` group. Use `--render-results` after editing a result;
see [results](results.md) for closeout and source recovery.

## Maintaining checks and tests

For a behavior change, inspect the relevant tests and update or add cases where the
existing coverage cannot demonstrate the intended outcome. Prefer observable behavior
and meaningful failure cases over tests that repeat the implementation. Existing
coverage may suffice; documentation-only edits need no new product tests.

When tests are added, renamed or intentionally removed, refresh the affected inventory
with `--update-tests` and review the additions and removals. Do not accept an inventory
change merely to hide a missing test. Preserve the runtime/GPU lane boundaries.

When a maintained check, option or diagnostic changes, update its command examples and
repair guidance here. Project-rule changes also need matching good/bad rule specimens;
changes to supported entry points follow the system-map maintenance guide. Keep full
acceptance at coherent checkpoints; the normal commit hook runs it. Passing checks
does not replace inspecting whether tests cover the changed behavior.

The frozen-main `tools.benchmark pr-feasibility` and `verify-pr-feasibility`
commands are a separate CPU interface diagnostic, documented in
[benchmarks](benchmarks.md). They do not replace the supported-runtime CPU lane or
qualify the new runtime for serving.
