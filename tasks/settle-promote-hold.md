# Settle promote-hold versus recency advice

## Goal
Decide with eight fresh seeds per model whether promote-hold replaces the recency
wait hint, using a pooled rule fixed before any run of this round.

## Authorized
Philip chose "More runs, then decide" on 2026-11-13 after the first round
(results/promote-hold-advice). Up to 2.5 GPU hours on the RTX 2080 Ti, hard stop;
no run over 30 minutes; idle check (0% utilization, no compute processes; about
300 MiB is held by a virtual display) before each use, waiting for any process
that is not ours. No tuning: candidate, hold cap, workload definitions and pool
sizes stay as in pressure-advice-qwen-01 and pressure-advice-gemma-01 (model
definition hashes verified equal). A defect in the benchmark stops the round and
is reported. No vLLM pin or package change, no new agents, no evidence deletion.

## Done when
The rule below is applied, the outcome is executed (adopt, split or keep), a result
keeping the first round visible is recorded, FINDINGS and docs are updated, and this
file is removed.

## State
Profile `settle`, arms native-arc (advice off), wait-hint (recency), candidate
(promote-hold); one invocation per model and arm, all seeds in one server.
Seeds are the first eight from 301 that the existing cohort check accepts.
Qwen: 301 303 305 308 312 313 314 315; reserve order 316 317 319 323 324 325 327 328.
Gemma: 302 304 305 307 308 309 310 311; reserve order 312 313 317 318 321 330 331 332.
The pre-GPU simulator check simulates contiguous seeds from the first seed, as in
the first round; this round does not change it.

Rule (new seeds only; the first round's four seeds are reported beside it but do
not count). A seed counts for a comparison when both cells are valid under the
existing activity contracts; seeds are never dropped for their outcome. If fewer
than six seeds per model are usable for candidate versus recency, reserve seeds are
added in order while budget lasts.
- R1 benefit, per model: candidate pooled hinted recomputed tokens are at most 0.85
  times recency's over counted seeds, and the candidate is not worse than recency
  in at least three quarters of counted seeds (ties count as not worse).
- R2 no weak spot, per model: in no counted seed is the candidate worse than
  recency by more than 10 percentage points of that seed's reusable hinted tokens.
- R3 unhinted sessions, per model: candidate pooled unhinted recomputed share is at
  most 3 points above advice-off and at most 3 points above recency.
- R4 store failures: none added relative to advice-off; abandoned hints cause none.
- R5 correctness: runtime CPU lane and dense GPU functional lane pass on final code.
Outcome: R1-R5 on both models adopts (promote-hold becomes wait-hint, recency path
deleted, checks rerun). R2-R5 on both with R1 on one model is a split verdict:
retire nothing. Anything else keeps recency. Advice-off cells invalid because the
native cache restored nothing are listed in the result; they do not block that
seed's candidate versus recency comparison.

## Next
Run Qwen arms, then Gemma arms (confirm Gemma GPU KV 9,676 tokens), optional A/A
repeats, Gemma seed 201 analysis, apply the rule, then the Gemma KV attribution.
