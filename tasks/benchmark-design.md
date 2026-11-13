# Repair benchmark evidence and investigate advice policy

## Goal
Find whether causal wait advice saves recomputation under realistic multi-session
cache competition, and qualify correctness before making a benefit claim.

## Authorized
Philip's audit handoff on 2026-11-12 supersedes the release-01 continuation. Cancel
the waiting follow-up and stop at the next panel boundary. Preserve every attempt
and resumable frozen path. Do not start block 1, compare-02 or LMCache supplements.
Use the existing RTX 2080 Ti, no paid compute, 5080, new agents or expanded matrix.
Implement a bounded Gemma divergence diagnostic, a CPU replay of captured application
traffic comparing native/current/promote/bounded-protect advice, then one real
multi-session pressure benchmark. Promotion/protection are experimental simulator
variants; adopting either in the product requires Philip's contract decision.
Compare remains one repetition within 45 minutes; release remains two within 30 hours,
but the stopped campaign is not authorized to resume. Philip authorized publishing
grafted main and one original-NEXT tag to public philippesic/vllm-msr-cache; other
archive refs remain local. See tasks/consolidation.md.

## Done when
Retained acquisition has an independently verified partial disposition. Gemma's
restored-continuation divergence is diagnosed or has a concrete unresolved dependency.
CPU replay runs through the maintained command with audited native-policy semantics
and explicit modeling limits. One bounded live multi-session pressure experiment
measures return recomputation and TTFT, cache competition and an unhinted cohort.
Current instructions and results describe what is measured and what remains unproven.

## State
Ownership transferred by Philip to a separate delegated Claude Opus 5.5 agent.
This thread is stopped after its bounded-probe handoff; it must not start simulator,
hint, benchmark or GPU work. Its recurring continuation is disabled (nextRunAt null):
`scheduled-task:command:mcp:acd6cba2-620f-415a-8977-34d1eb52ad1f:schedule-task:semantic-cache-next-benchmark-completion-20261112`.

release-01 stopped after panel sessions/0/7, retaining a terminal-000.json receipt.
Owner PID 322014 and queued watcher PID 335195 exited. No next panel was admitted,
no block 1 or queued job launched, and the temporary empty boundary barrier was
removed. Preserve the external attempt, frozen 50e53cf checkout and environments;
do not resume without new authorization. The coordinator was briefly paused while
installing the barrier; retain that intervention when interpreting panel 0/7 timing.
Stop receipts are in artifacts/maintenance/audit-*.json. Independent verify-campaign
completed: 236 planned, 118 attempted, 87 complete, 31 failed, 118 not run; measurement
incomplete, release not eligible, benefit not established, within the original budget.
The report is artifacts/maintenance/stopped-release-independent-report.json.

Two bounded I1 Gemma diagnostics completed from source 91d8121a and were independently
verified with the maintained verify-probe command. The native attempt took 123.77 s;
the GPU-only control took 97.27 s. Their exact executed command is:

```sh
uv run --locked python -m tools.benchmark probe-model \
  --runtime-python /home/philip/.local/share/venvs/wait-hint-benchmark-runtime/bin/python \
  --model /home/philip/.cache/huggingface/hub/models--axteam--gemma-4-12B-it-AWQ-4bit/snapshots/7c781be42ce8d29d0975b857f6d0c0b2558d9b63 \
  --system native-arc --trace-case I1 --trace-seconds 180 --timeout 600 \
  --out artifacts/runs/gemma-invariance-01
```

The second command changes only `--system gpu-only` and
`--out artifacts/runs/gemma-invariance-gpu-01`. These output directories already
exist; preserve them. Each contains admission/source identities, terminal.json and
panel/0 request/reset/logprob evidence. Verification commands are
`uv run --locked python -m tools.benchmark verify-probe artifacts/runs/gemma-invariance-01`
and the same command with `artifacts/runs/gemma-invariance-gpu-01`. Retained reader
outputs are artifacts/maintenance/gemma-invariance-01-independent.json and
artifacts/maintenance/gemma-invariance-gpu-01-independent.json; compact token comparison
is artifacts/maintenance/gemma-invariance-handoff.json.

Both diagnostics used token IDs 5000 through 8071, 32 generated tokens, temperature 0,
seed 0, and top-five logprobs. In each arm all three fresh outputs matched exactly.
The GPU-resident return used 3008 cached tokens and diverged first at output token 2;
its full output was identical between the native and GPU-only arms. Thus output
changes with GPU prefix caching even without CPU offload. Native CPU restoration
used 3071 cached tokens and differed only at token 32 (236771 versus 236764), matching
the failed soak. The GPU-only displaced return used zero cached tokens and matched
fresh. The native final-token logprobs tie comma and zero; fresh slightly favors zero.
This separates cache-path output sensitivity from fresh-run repeat variability;
it does not prove CPU byte correctness or harmless rounding. Keep both original
Gemma soaks failed and the exact-continuation/operating claim unresolved. Both Qwen
two-hour operating checks independently passed sustained token comparisons.

I1 is implemented and tested through the maintained probe-model command. A completed
diagnostic may report unequal tokens; completion does not mean invariance passed.
Portable acceptance passed 71 CPU tests and 105 rule proofs before publication.
Runtime preparation previously passed 18 CPU tests and four GPU cases. No simulator,
new hint variant or multi-session benchmark implementation was begun by this thread.
The old detailed plan remains in Git under refs/archive/next/pre-graft-main; the
maintained contract is docs/benchmark-contract.md. External audit scripts remain in
artifacts/external-review-20261112. A read-only tokenizer feasibility check matched
Qwen prompt counts; Gemma's direct tokenize=True template output was one token short,
so replay must validate exact tokenizer preprocessing rather than assume identities.

## Next
The new delegated owner takes over Gemma root-cause work, CPU trace replay, candidate
exploration and the single real multi-session pressure benchmark. Preserve failed
and censored evidence, distinguish simulated policy direction from live benefit, and
confirm a contract change before adopting promotion/protection. This thread must
remain idle; its continuation is off. Do not resume the stopped campaign or queue.
