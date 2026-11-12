# vllm-semantic-cache-ext-next

Exact CPU KV offload for vLLM with causal wait-hint advice. A client hint marks a
session's ready CPU prefix as recently used in vLLM's native ARC cache. Advice only
reorders the cache; exact native block keys alone decide KV reuse. A hint never pins
KV, makes a block ready, starts a transfer or permits approximate reuse.

# Where things are

- [README](README.md): setup and everyday commands.
- [Architecture](docs/architecture.md): components, ownership and the hint protocol.
- [Runtime](docs/runtime.md): pinned vLLM runtime, runtime test lane, serving, GPU gate.
- [Checks](docs/checks.md): check modes, entry points and how to fix failures.

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

- `uv run --locked python tools/check.py` is the fast lane (seconds, partial); run
  it while working. `--full` adds the rule proofs and is acceptance; the pre-commit
  hook runs it. Enable the hook once per clone: `git config core.hooksPath .githooks`.
- `--test tests/PATH.py` gives fast partial feedback. Never report it as acceptance.
- After adding or renaming tests, run `--update-tests` and review the printed diff.
- When a check fails, fix the cause. Weakening a check or adding an exception in
  `tools/exceptions.toml` needs Philip's review.

# Working autonomously

- Within the task's scope, proceed through reversible work without asking:
  repairs, tests, refactors, reruns and commits. Ask only for a missing
  consequential decision, a scope change, a weaker guarantee or an unauthorized
  resource. Do not wait for permission you already have.
- GPU use needs Philip's explicit authorization; otherwise stay on CPU. There is no
  GPU reservation system and none should be built; check the GPU is idle first.
- Parallel agents each use their own git worktree. Integrate one at a time and
  rerun the check on the combined tree.
- Commit coherent checkpoints with `--full` passing.

# Documents

Maintaining affected tests, supported entry points, current documentation and task
state is part of completing a change. Carry authorized work through that upkeep,
verification and cleanup; do not hand back routine unfinished maintenance. Update
supporting material in the same coherent change when behavior, commands or ownership
change. Do not add unrelated edits, new tests or result pages just to satisfy a
checklist. Follow the topic guides below; passing structural checks does not prove
that explanations or test coverage are correct.

Each fact has one home, chosen by how long it stays true:

- `AGENTS.md`: rules for every task. Philip owns it; propose changes.
- `docs/*.md`: how each part works now, and why. Update in the same commit as the
  code. Delete outdated text instead of marking it superseded; git keeps history.
- [FINDINGS](FINDINGS.md): short generated conclusion/link index, including failures.
- Close investigations with [close-results](skills/close-results/SKILL.md) and
  [the result profile](docs/results.md); results retain lessons, raw records stay immutable.
  Routine fixes need no result. Update current docs, then delete completed task files.
- [`tasks/`](tasks/README.md): one file per open multi-session task, owned by the
  agent doing it and deleted when the task completes.
- Git history: what changed and when. There is no status or history document.

At the start of a task, scan FINDINGS and the names of open task files, then read
the task and result details relevant to the request. Do not load every plan or
historical report by default. Read other task state when it may affect ownership
or shared resources. Do not redo closed work without a stated reason. An open task
file is context, not an instruction to resume unrelated work.

# Evidence and reporting

- Raw runs, logs and generated data live outside the repository.
- Keep failed, unfavorable and incomplete results. Never rerun selectively until a
  result looks good.
- Done means the claimed behavior was exercised and acceptance passed, not that
  the steps were followed.
- Report what changed, what actually ran versus was reused, what passed and what
  was not tested. Skipped is not passed. A CPU pass says nothing about GPU
  behavior, performance or answer quality.
- Write plainly. State each fact and its scope once.
- Never put credentials in tracked files.
