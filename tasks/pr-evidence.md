# PR evidence on frozen vLLM main

## Goal
Execute Stage 1 and the bounded 2080 Ti Stage 2 screen in the accepted evidence
plan, then retain per-route findings for Philip's 5080 decision.

## Authorized
Philip authorized Stage 1 in full and Stage 2 admission/screen on 2026-11-13.
Frozen vLLM: e9281bd939d4ade90d6aacf4b2319aa89ea46c39. GPU ceiling: 9 hours total,
at most 1 hour admission and 8 hours screen. Check idle before every GPU job;
wait for other users. Complete paired blocks in plan order, re-quote after the
first. Freeze contracts, mechanism, limits, W1 manifest and patch/control identity
before dependent GPU runs. Retain all attempts and adverse results. No Stage 2b,
Stage 3/pilot, 5080, rental, WSL memory increase, changed decision rules or check
exceptions. Commit and push checkpoints. Send Philip's exact authorization message
to thread 4dd4323d-196a-4973-a4cb-3e985a47c982 only after completion and GPU release.

## Done when
Stage 1 and Stage 2 finish or stop under a plan rule/budget, with close-results
recording route/model/workload verdicts, harms, opportunity, exclusions and not-run
work. Findings and current docs are updated; coherent commits are pushed.

## State
Owned by the active agent on main. Initial checkout clean at 3a43529d. Plan and
required context read. No GPU time used. Existing runtime and runner target the old
pin and do not qualify the new plan. Feasibility inspection comes before building
runtime and acquisition machinery. Release harm limit remains Philip's Stage 3
decision; it does not block the authorized Stage 2 screening rule.

## Next
Record the staged estimate, inspect the frozen CachePolicy interface and test the
route feasibility gates. Continue implementation for feasible routes, then freeze
Stage 1 inputs before hardware admission.
