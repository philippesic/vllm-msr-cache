# Local development notes

- The existing WSL environment is `/home/philip/.local/share/venvs/vllm-semantic-cache-ext`.
  Use `make test PY=/home/philip/.local/share/venvs/vllm-semantic-cache-ext/bin/python`
  for the focused CPU checks through the patched runtime; direct wheel-only tests
  can miss the required native-publication API. See [runtime setup](vendor/VLLM.md).
- RTX 2080 Ti serving requires FP16 and `VLLM_WSL2_ENABLE_PIN_MEMORY=1` (the launcher
  sets it); BF16 and FlashAttention 2 are unsupported here.
- For ongoing work and shared GPU ownership, use [docs/STATUS.md](docs/STATUS.md).
  Update the affected entry at handoff or a material change; move completed outcomes
  to the brief history. Preserve other tasks' entries. Keep logs and proofs in
  linked run/evidence files rather than copying them into status.
- For tests or experiments, use [docs/BENCHMARKING.md](docs/BENCHMARKING.md).
  The maintained commands are `make bench-smoke`, `bench-screen`, `bench-confirm`,
  and `bench-operational` with explicit `PY`, local `MODEL`, and new `OUT`.
  Use `bench-tests` for runner changes and `bench-verify` for retained results.
  A screen is advisory: missing activity, noisy timing, or an unrepresentative
  workload cannot justify rejecting a potentially useful change. A passing run
  means complete measurements, not a policy win or production qualification.
  Match the corpus, cache pressure, candidate configuration, and required activity
  to the claim; confirmation and operational runs require declared corpus provenance.
  `output/` contains active and frozen scripts as well as generated data. Do not
  move those dependencies during a run. New experiment freezes should cover executed
  inputs; unrelated documentation changes are provenance, not serving failures.
- Keep reusable experiment code in `benchmarks/` or `harness/`, compact findings in
  `evidence/`, and raw runs in ignored `output/`. Search project code separately
  from the full `vendor/vllm` tree unless investigating upstream behavior.

- New GPU supervisors must use `harness.gpu_lock.gpu_reserved` for the entire
  invocation, including idle gaps; direct GPU tasks use `GPULock()`. Do not unlink
  lock files or give the supervisor descriptor to children. Preserve historical
  child locks and check device state before launch.
- `make check-light` is the portable CI check; it does not qualify vLLM or GPU
  serving. Reuse `benchmarks/research_checks/` for retained research validation.
  Frozen reproduction coordinators require historical local artifacts; use the
  maintained four-tier suite for new work.

## Git checkpoints and project files

- Commit coherent checkpoints and push the working branch to the established
  user-owned origin at meaningful milestones and before handoff. This is standing
  authorization; respect a later explicit no-push instruction. Never force-push,
  push to third-party upstreams, or stage unrelated active work. Report failed
  pushes and local-only work instead of calling them synchronized.
- Track source, tests, benchmark definitions, project AGENTS.md/skills, setup
  templates, current state, history and compact findings here. p-stack handles
  shared policy/tooling; it does not synchronize this project's files.
- Ignore only secrets, host-private settings, downloaded dependencies, large raw
  artifacts and reproducible caches. Keep reusable scripts and written findings
  out of ignored run trees; preserve frozen inputs in place while jobs use them.
  At handoff, check `git status --short`, `git status --ignored --short`, and
  upstream divergence. A clean tree alone does not prove work was pushed.
