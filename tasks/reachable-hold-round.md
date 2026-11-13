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
Round decided (commit 20b56941 rule, results/reachable-hold-decide): R1-R5 held on
both models and reachable-hold was adopted; recency, promote-hold, the
advice_mechanism setting and the benchmark's candidate arm were removed. Pooled
hinted recomputed share: Qwen 44.3% advice-off, 31.1% recency, 17.4% reachable-hold;
Gemma 53.7%, 41.2%, 6.6%. All 52 cells valid, no store failures. Runtime CPU lane
and GPU functional lane (functional-gpu-05) passed on the adopted code. GPU used so
far about 2 h 25 min of 5 h.

Campaign: profile `full` on both models with the adopted code, six arms (gpu-only,
native-arc, native-arc-repeat, wait-hint, lmcache, oracle), seeds 305 308 312, one
invocation per model and arm, no selective reruns. LMCache restore and eviction
counters are unobservable; its cells are judged by the contracts it can satisfy.

## Next
Run the campaign, verify it, record a result with invalid cells kept, then delete
this task.
