# vllm-semantic-cache-ext-next

Exact CPU KV offload for vLLM with causal wait-hint advice. A client hint marks a
session's ready CPU prefix as recently used in vLLM's native ARC cache. Advice only
reorders the cache; exact native block keys alone decide KV reuse. A hint never pins
KV, makes a block ready, starts a transfer or permits approximate reuse.

# Where things are

- [README](README.md): setup and everyday commands.
- [Architecture](docs/architecture.md): components, ownership and the hint protocol.
- [Runtime](docs/runtime.md): pinned vLLM runtime, serving and the GPU gate.
- [Checks](docs/checks.md): what the check enforces and how to fix failures.
- [System map](docs/system-map.md): supported entry points.
- [FINDINGS](FINDINGS.md): what is established, with evidence. Read it before
  planning experiments or claiming results, and do not redo closed work without
  a stated reason.

# Code

- `wait_hint_offload/` is the pure core: standard library only, no vLLM.
  `wait_hint_offload/vllm_adapter/` is the only code that imports vLLM.
- No code comments or docstrings. Carry intent in names, types, named constants
  and tests. Rationale that cannot live in code belongs in the topic doc.
- Make invalid states hard to represent: closed schemas, refuse unknown fields
  and versions, typed outcomes, no silent fallbacks or hidden defaults.
- Each piece of state has one owner that mutates it; everything else reads.
- Tests assert observable behavior with small fakes and fake clocks.
- Read environment variables only in the config modules. Add no standalone
  scripts; extend the existing entry points.

# Validation

- `uv run --locked python tools/check.py` is full acceptance. The pre-commit hook
  runs it; enable it in a new clone with `git config core.hooksPath .githooks`.
- `--test tests/PATH.py` gives fast partial feedback. Never report it as acceptance.
- After adding or renaming tests, run `--update-tests` and review the printed diff.
- The runtime lane (`--runtime-python`, `--out`) is separate; see [Runtime](docs/runtime.md).
- When a check fails, fix the cause. Weakening a check or adding an exception in
  `tools/exceptions.toml` needs Philip's review.

# Working autonomously

- Within the task's scope, proceed through reversible work without asking:
  repairs, tests, refactors, reruns and commits. Ask only for a missing
  consequential decision, a scope change, a weaker guarantee or an unauthorized
  resource. Do not wait for permission you already have.
- GPU use needs Philip's explicit authorization naming the host and window.
  Without it, stay on CPU.
- Parallel agents each use their own git worktree. Integrate one at a time and
  rerun the full check on the combined tree.
- Commit coherent checkpoints with the check passing.

# Evidence and reporting

- Raw runs, logs and generated data live outside the repository. Git holds code,
  tests, compact records in `evidence/` and conclusions in FINDINGS.
- Keep failed, unfavorable and incomplete results. Never rerun selectively until a
  result looks good.
- Done means the claimed behavior was exercised and acceptance passed, not that
  the steps were followed.
- Report what changed, what actually ran versus was reused, what passed and what
  was not tested. Skipped is not passed. A CPU pass says nothing about GPU
  behavior, performance or answer quality.
- Write plainly. State each fact and its scope once.
- Never put credentials in tracked files.
