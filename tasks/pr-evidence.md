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
Active. The full plan is [the evidence plan](../docs/pr-evidence-plan.md): stages
and decision rules, accepted caveats, and the concrete workloads, models, comparators
and machine tiers, each accepted by the reviewing models (GPT-6 Astra, Claude Fable
5.2, GPT-6.1 Sol) on 2026-11-13. Philip is sanity-checking the models, workloads and
machine progression before any stage is authorized.

Facts behind the plan: vLLM main e9281bd93 (2026-11-13) is 6,090 commits past the
pinned dc1be7903; stock CPU offload defaults to LRU and prompt-only storage, whereas
every earlier result used the project's connector forcing ARC and
offload_prompt_only=False; main stores only semantic sliding-window and Mamba
checkpoints by default (#52216), carries a KvHintsEnvelope on requests (#53423) and
loads out-of-tree cache policies (#49114). Earlier measured gain on the pinned
version (results/reachable-hold-decide): server busy time x1.03 Qwen (noise level)
and x1.15 Gemma versus that ARC baseline. The WSL instance on the 2080 Ti machine
sees about 15 GB RAM, capping CPU pools near 4 GiB there.

## Next
Philip's review of the plan. Then, if authorized, Stage 1 (CPU-only feasibility on
vLLM main) as the plan specifies.
