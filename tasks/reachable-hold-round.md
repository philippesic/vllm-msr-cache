# Decide reachable-hold versus recency, then run the full pressure campaign

## Goal
Decide with eight fresh seeds per model whether reachable-hold replaces the recency
wait hint under a rule fixed before any run, then run the full seven-arm pressure
campaign on whatever is the default afterwards.

## Authorized
Philip, relayed by the ShaderLOD agent on 2026-11-13 11:06 PST ("execute all 3 of
its plan"): a new candidate round (hold yields to running requests; explain Qwen's
~20% first), the full pressure campaign, and the consolidation tidy-up. RTX 2080 Ti,
at most 5 GPU hours in total for the diagnostics, this round and the campaign; no
run over 30 minutes; idle check before each use (0% utilization, no compute
processes; about 300 MiB is held by a virtual display). ShaderLOD holds its GPU
work until this finishes. No vLLM pin or package change, no new agents, no paid
resources, no evidence deletion.

## Done when
The rule below is applied and its outcome executed, the campaign has run, results
for both are recorded with failed and invalid cells kept, FINDINGS and docs are
updated, and this file is removed.

## State
Qwen's ~20% (settle round, every arm) is explained without GPU-wide tuning: one
debug-logged Qwen cell (artifacts/runs/diag-qwen-lookup-03, seed 301, candidate)
showed every full loss with a complete CPU copy was a pinned-vLLM hybrid lookup
effect, not eviction. When a quick return still has full-attention blocks on GPU
but no matching Mamba state, the scheduler asks the connector only for tokens
beyond the GPU hit, gets zero, and then falls back to the boundary all groups agree
on (often zero), discarding the CPU copy. The rest were compaction returns whose
528-token Mamba checkpoint was never stored. The Qwen pool was also oversized: the
engine stores about 2.5 keys per 528-token chunk (Mamba only at scheduling-step
ends), not 4, so 100 keys held the whole working set. This round sets the Qwen pool
to 60 keys, 0.6 of the working set as documented, and the simulator now stores
recurrent keys only at step ends and simulates the run's exact seeds.

Candidate `reachable-hold`: promote-hold plus (a) the hint marks the session's
unreachable sliding-window and recurrent keys, which evictions take first, and
(b) when an eviction fails with holds in place, holds are released earliest
deadline first instead of all at once. Cap stays half the pool; deadline is the
client's. Simulated on these seeds, hinted recomputed share: Qwen recency 20.9%,
candidate 4.1%; Gemma recency 38.0%, candidate 5.9%; unhinted below advice-off on
both (artifacts/runs/sim-decide). The simulator does not model the vLLM hybrid
lookup effect, so real Qwen shares will be higher for every arm.

Profile `decide`, arms native-arc (advice off), wait-hint (recency), candidate
(reachable-hold); one invocation per model and arm, all seeds in one server; A/A
repeats of native-arc on two seeds per model. Seeds are the first eight from 401
that the cohort check accepts.
Qwen: 401 402 403 404 405 407 408 412; reserve order 413 417 418 420 422 425 428 430.
Gemma: 401 402 404 409 411 412 413 416; reserve order 418 420 421 422 425 427 430 431.

Rule (unchanged from the settle round; new seeds only). A seed counts for a
comparison when both cells are valid under the existing activity contracts; seeds
are never dropped for their outcome. If fewer than six seeds per model are usable
for candidate versus recency, reserve seeds are added in order while budget lasts.
- R1 benefit, per model: candidate pooled hinted recomputed tokens are at most 0.85
  times recency's over counted seeds, and the candidate is not worse than recency
  in at least three quarters of counted seeds (ties count as not worse).
- R2 no weak spot, per model: in no counted seed is the candidate worse than
  recency by more than 10 percentage points of that seed's reusable hinted tokens.
- R3 unhinted sessions, per model: candidate pooled unhinted recomputed share is at
  most 3 points above advice-off and at most 3 points above recency.
- R4 store failures: none added relative to advice-off; abandoned hints cause none.
- R5 correctness: runtime CPU lane and dense GPU functional lane pass on final code.
Outcome: R1-R5 on both models adopts (reachable-hold becomes the default wait hint;
recency and promote-hold paths are deleted; checks rerun). R2-R5 on both with R1 on
one model is a split verdict: retire nothing and bring it to Philip. Anything else
keeps recency. Advice-off cells invalid because the native cache restored nothing
are listed in the result; they do not block that seed's comparison.

Campaign: profile `full` (seven arms, seeds 305 308 312) on both models after the
decision. If reachable-hold is adopted, the wait-hint arm uses it and the candidate
arm is dropped; otherwise the candidate arm stays reachable-hold.

## Next
Run Qwen arms, then Gemma arms, then A/A repeats; apply the rule; then the campaign.
