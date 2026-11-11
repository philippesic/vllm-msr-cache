# Benchmark design

## Goal
Design measurements that reveal where the evolving product helps or harms, using
curated historical findings without fixing the product to a predetermined workload.

## Authorized
Philip authorized collaborative benchmark design and review of historical evidence.
Benchmark implementation and GPU acquisition are not authorized. Agent operations
remain a separate discussion; their completion is not a prerequisite for design.

## Done when
Philip has agreed on the questions, controls, coverage, metrics and interpretation
needed for the next product-development experiments.

## State
The current design discussion is owned by the primary agent in the next repository
on main. Existing functional checks establish narrow correctness coverage, not
performance or application value. FINDINGS links relevant historical gains, losses
and unresolved cases.

The proposed measurement families remain provisional:

- Serving smoke: validate the measurement setup and actual offload/hint activity.
- Overhead controls: measure costs when reuse or advice cannot help.
- Controlled reuse and pressure: locate beneficial and harmful operating conditions.
- Fixed-work replay: compare the same serving work under controlled arrivals.
- Live workflows: compare application completion time and valid task outcomes.
- Interference and sustained operation: expose harm to other traffic and degradation.
- Alternatives and coverage: compare native controls and relevant external systems
  across deliberately qualified model and runtime configurations.

Workload and compatibility coverage should follow the mechanism and observed
opportunity. Existing exactness and failure invariants remain requirements.
The design must account for dataset ancestry and prior tuning exposure, fresh
confirmation, hint-production costs, output validity, noise, and incomplete runs.

## Next
Refine the proposal with Philip into an initial scope, comparisons, workloads,
metrics, cost and interpretation. During benchmark design, choose the maintained
benchmark entry point and shared execution/evidence helpers; do not implement it
as part of repository preparation. Coordinate interruption, resumption and completion
notification with the separate agent-operations discussion.
