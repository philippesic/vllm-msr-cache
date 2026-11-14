# Evidence for proposing the wait hint to vLLM

## Goal
Decide whether the reachable-hold wait hint still helps on current vLLM and, if it
does, collect evidence a vLLM pull request can cite across models, workloads,
load levels, machines and competitors.

## Authorized
Philip on 2026-11-13: planning only. Two proposals if pursued: hint-free cache policy
changes separately from the wait-hint. Hardware path: the 2080 Ti for a basic
stop-or-continue screen, then the RTX 5080, then rented compute only if confident.
Not yet authorized: Stage 1 (port to vLLM main as a plugin, new runtime
environment), any GPU use, benchmark implementation, raising the WSL memory limit.
Ask again before each.

## Done when
Either a stop decision with its evidence is recorded, or a result on current vLLM
meets the plan's Stage 3 protocol, and this task is replaced by the upstream work.

## State
Active, planning only. The [evidence plan](../docs/pr-evidence-plan.md) received the
earlier multi-model reviews and a final single-agent review requested by Philip on
2026-11-13. The final review reconciled conflicting rules, separated the hint-free
and wait-hint proposal tracks, added release/readiness gates and named execution
details that must be frozen before dependent runs. The six-minute confirmation-run
estimate was incorrect: the shared schedule requires at least 22 minutes plus
measured arrivals per arm, before preparation. A costed manifest is still required.

Philip prefers competitor superiority if the evidence supports it and leaves scope
to judgment. Plan: establish native-vLLM value first, then prioritize qualified
competitor comparisons; a scoped PR/plugin remains possible without superiority.
Pending: the release harm limit for traffic without hints. Philip asked why opt-in
needs that guardrail: hinted and unhinted sessions share cache capacity and execution
queues, so protection can displace other sessions and increase latency. A fixed 5%
upper-confidence harm limit is a recommendation, not approval. Stage 1 must settle calibration order,
hold metadata limits, workload admission and instrumentation; platform pilots must
freeze sample sizing and statistical edge cases before confirmation. Stage 0's
campaign-record corrections are already committed in d8cf1d63.

Facts behind the plan: vLLM main e9281bd93 (2026-11-13) is 6,090 commits past the
pinned dc1be7903; stock CPU offload defaults to LRU and prompt-only storage, whereas
every earlier result used the project's connector forcing ARC and
offload_prompt_only=False; main stores only semantic sliding-window and Mamba
checkpoints by default (#52216), carries a KvHintsEnvelope on requests (#53423) and
loads out-of-tree cache policies (#49114). Earlier measured gain on the pinned
version (results/reachable-hold-decide): server busy time x1.03 Qwen (noise level)
and x1.15 Gemma versus that ARC baseline. The WSL instance on the 2080 Ti machine
sees about 15 GB RAM, capping CPU pools near 4 GiB there.

Both hint delivery routes are now planned (A4t request TTL as a pure plugin, A4p
post-response with a hashed core patch and its patched control A3p), with
independent route verdicts. GPT-6 Astra and GPT-6.1 Sol debated the plan on
2026-11-13 and both accepted it at 9a8e0acf, adding a route contract, an automatic
hold comparator, claim scoping, harm-cohort sizing and a cost drop order.

## Next
Resolve the pending harm decision; prepare a bounded Stage 1 implementation
estimate. Then, only if authorized, begin the CPU-only feasibility work on frozen
vLLM main. GPU admission and each later platform need separate authorization.
