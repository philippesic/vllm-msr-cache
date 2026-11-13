# Repair benchmark evidence and investigate advice policy

## Goal
Find whether causal wait advice saves recomputation under realistic multi-session
cache competition, adopt a better hint mechanism if evidence supports it, and
qualify correctness before making a benefit claim.

## Authorized
Philip's audit handoff on 2026-11-12 stopped release-01; preserve every attempt,
frozen checkout and archive ref, and do not resume it. On 2026-11-12 Philip also
approved changing the product contract for audit finding 2: if a native-touch
(promotion) or bounded-protection hint consistently beats the current recency hint,
make it the wait-hint mechanism and retire the recency path. The delegated owner
(Claude Opus 5.5) decides routine matters and records them here.
Limits: RTX 2080 Ti only, idle check before each use, no run over 30 minutes, at most
4 GPU hours for this task, no full campaign, no vLLM pin or runtime package change,
no new agents. Fast-forward pushes to origin main only. A full pressure run needs
Philip's decision after the smoke profile and projected cost are delivered.

## Done when
CPU trace replay runs through `python -m tools.benchmark simulate` against the real
pinned ARC policy and manager. The candidate is adopted or rejected by the rule below,
with retained evidence. A multi-session pressure benchmark proves it can discriminate
through controls, activity contracts, a pre-GPU simulator check, a smoke profile,
CPU fixtures and mutation checks, and `plan` projects the full-run cost. Gemma
restored-KV exactness is settled or reported as provisional within 45 GPU minutes.

## State
Decision rule, fixed 2026-11-12 after simulator results and before any engine result.
Candidate: promote-hold (native touch of the ready prefix, then exclude it from
eviction until the session's next request, the hint deadline, retirement or reset;
held keys are capped at half the CPU pool, and an eviction that cannot succeed
without held keys ignores the holds). Reference: current recency hint. Control:
advice disabled. The candidate is adopted only if all hold:
1. Hinted returning sessions: candidate recomputed prefix tokens are no more than the
   reference in every discriminating simulator scenario and in every engine
   repetition, and the aggregate is at least 10% lower in the simulator and lower in
   the engine total for each model.
2. Unhinted returning sessions: recomputed share of reusable tokens is at most 3
   percentage points above the control in each simulator scenario and in each
   engine model's pooled result (tolerance chosen after the simulator, before engines).
3. Store failures: candidate failures are no more than the control's in each scenario
   and engine model total; abandoned hints cause no store failure.
4. Engine evidence: at least three independent repetitions per arm per exercised
   model, from cells whose controls and activity contracts pass; correctness checks
   (runtime CPU lane, GPU functional cases) pass with the candidate.
Promote alone was eliminated in the simulator: it was worse than recency in two
Gemma scenarios. Plain hold was dominated by promote-hold in 30 s waits and in
unhinted harm under one-shot competition.

Simulator (artifacts/runs/simulate-01, 10 scenarios, 6 or 4 seeds): rule met in all 9
discriminating scenarios; aggregate hinted recomputed tokens 1.34M candidate versus
3.19M reference; worst unhinted change +2.6 points. The release-01 configuration is
correctly non-discriminating (no recomputation at all: GPU cache 2.5x the working set).

release-01 stayed stopped after panel sessions/0/7: 236 planned, 118 attempted, 87
complete, 31 failed; not eligible, benefit not established. Independent report and
stop receipts: artifacts/maintenance/stopped-release-independent-report.json and
artifacts/maintenance/audit-*.json. The previous owner's schedule is disabled.

Gemma I1 diagnostics (preserve artifacts/runs/gemma-invariance-01 and -gpu-01 and
their maintenance reports): fresh outputs match; GPU-resident reuse alters output
even without CPU offload; native restore differs only at token 32. Keep both
original Gemma soaks failed; exactness remains unresolved.

Pressure benchmark (`python -m tools.benchmark pressure`, benchmarks/pressure.json) is
implemented with controls, activity contracts, cohort-covering seeds, a pre-GPU
simulator check and CPU fixture/mutation tests. Its pre-check predicted, before any
engine comparison, that on the Qwen engine workload the candidate cuts hinted
recomputation but raises unhinted recomputation about 9 points (rule 2 fails in
simulation there); Gemma is predicted to pass. The first smoke attempt
(pressure-smoke-01) failed at Qwen startup (20 GPU blocks cannot hold one hybrid
request); pressure cells now use 36 blocks and max length 6144.

## Next
Finish smoke pressure-smoke-03, then run the `advice` profile (3 seeds, native-arc,
wait-hint, candidate, both models), apply the rule, then the Gemma KV byte case.
