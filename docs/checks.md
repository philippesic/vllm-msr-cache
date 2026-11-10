# Checks

`uv run --locked python tools/check.py` is full acceptance, locally and in CI. The
`.githooks/pre-commit` hook runs exactly that command; enable it with
`git config core.hooksPath .githooks`. It never runs GPU tests or builds.

| Command | Purpose |
| --- | --- |
| `tools/check.py` | Full portable acceptance |
| `tools/check.py --test tests/PATH.py` | Fast partial feedback; never acceptance |
| `tools/check.py --component NAME` | One group, e.g. `policy`, `types`, `lint` |
| `tools/check.py --self-test` | Only the rule proofs |
| `tools/check.py --update-tests` | Refresh the test inventory and print the diff |
| `tools/check.py --runtime-python PY --out DIR` | Strict runtime types plus pinned CPU integration |

All commands are prefixed with `uv run --locked python`. The pinned check image is
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

## Enforced groups

| Group | Enforcement |
| --- | --- |
| Format, lint | Ruff format (check only) and the selected Ruff diagnostics |
| Types | mypy `--strict` on the package, tools and tests; no broad exemptions |
| Imports | Only the adapter imports vLLM; production never imports benchmarks; contracts never import the adapter; acyclic siblings |
| Boundaries | Pure core uses the standard library only, has no import-time calls and no dynamic imports |
| Repository policy | One root AGENTS.md and a CLAUDE.md pointer; size cap; no raw outputs, copied trees, symlinks, binaries or large files; selected secret patterns; doc links resolve |
| Agent policy | No comments, bare excepts, library prints, stray environment access, test sleeps, standalone scripts or static registries |
| Unused | Vulture at 60% confidence; deptry dependency checks |
| Native | Fails closed if C++/CUDA/CMake material appears without qualified recipes; skipped otherwise |
| Runtime | Confirms the runtime verifier refuses to run without explicit inputs; reported as skipped |
| CPU pytest | Fixed test inventory; no missing, extra, skipped or xfailed tests |
| Rule proofs | Each rule rejects a bad specimen and accepts a good one, in a throwaway repository |

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
external attempt, with separate admission and terminal records. It records the
actual Python, loaded native libraries, package versions and CUDA state.
`--verify DIR` checks a retained attempt without rerunning it. Its process helper
enforces a wall-time limit, makes no automatic retries, and only signals the
session it created.
