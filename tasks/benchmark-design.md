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

The product is the research subject: useful applications are to be discovered from
its behavior, not selected as permanent optimization targets. Step 1 is controlled
characterization, separate from application confirmation. The primary agent and Sol
agree on the following scope and controls; Philip has not yet approved the design.

Five experiment families share inputs and instrumentation:
1. Measurement/overhead controls: GPU-only native prefix caching, connector advice
   disabled, and advice enabled; distinguish no registration, registration without
   hints and matched hint traffic. Include identical-system comparisons and a
   sensitivity control with an independently established added cost.
2. Reuse structure/distance: independent returning prefixes, shared session prefixes,
   growing transcripts and partial overlap; uniform and skewed popularity. Measure
   intervening distinct native KV-block bytes separately from elapsed wait time,
   counting shared ancestors once.
3. Capacity/work boundaries: independent GPU and CPU capacities, active decode
   allocation, reusable prefix/new suffix/output lengths. Cover no pressure, GPU
   pressure with adequate CPU capacity and CPU competition; selected interactions.
4. Hint timing/interference: none, timely valid, late valid, refused stale, and valid
   hints for abandoned continuations; separate hinted/unhinted outcomes. Include
   shared-prefix interference because abandoned-session advice may still help others.
5. Arrivals/continuations: fixed offered arrival schedules with bounded overload,
   plus a small completion-dependent session check with declared waits.

The advice comparison preserves correctness guards, registration, capacities and
intended hint traffic. GPU-only caching is a system alternative, not isolated advice
attribution. Match explicit GPU capacity and disclose CPU allocation initially.
Fix one qualified dense FP16 TP1 configuration on the 2080 Ti; broader model and
hardware claims need later coverage. Do not silently repin or repair the known ARC
fallback limitation: measure store/eviction refusals, since its effects need not
cancel between advice modes.

Record scheduled/submitted arrivals, first token, token gaps, completion, backlog,
failures and cancellations, with declared queue/termination/drain rules. Report
latency and completed work alongside CPU/GPU memory, CPU cost, transfers,
recomputation, preemptions, registration/hint costs and actual token counts. Trace
registration/readiness/refresh through later retention and reuse rather than
inferring value from acknowledgements alone. Separate explanatory tracing from
primary timing, or measure its overhead.

Use equivalent clean cache states and separately deliberate warm states; runtime
warmup must not accidentally seed measured prefixes. Alternate/randomize paired
orders and choose repetition from pilot variability. Whole-run uncertainty must
not treat correlated requests as independent; small samples cannot establish p99.
Select contrasting corners and interactions, then refine transition boundaries,
including adverse ones. No exhaustive Cartesian sweep or universal repeat count.

The old workload catalog and repository-pressure task examples were inspected as
historically exposed authored assets, not fresh confirmation inputs. Their task
labels and configured cache regimes cannot establish measured pressure or reuse.
Fixed external prompts/schedules control nominal demand, not output identity.
Recorded-token replay requires separately qualified forced-token execution and
cannot establish free-generation quality. Live continuations use each arm's actual
outputs and may diverge; a recorded transcript must not be labelled live.
Initial acquisition must choose and label these protocols explicitly.

Application discovery/confirmation, realistic hint availability and prediction
cost, representative mixtures, sustained reliability, external alternatives and
additional models/hardware remain later work. This scope is sufficient to map
mechanism behavior on the tested configuration, not to establish general utility.

Research references:
- [vLLM serving benchmark controls](https://docs.vllm.ai/en/stable/cli/bench/serve/)
- [vLLM cache carryover warning](https://github.com/vllm-project/vllm/blob/main/docs/benchmarking/cli.md)
- [vLLM prefix caching scope](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)
- [MLCommons inference scenarios](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)

## Next
Refine the proposal with Philip into an initial scope, comparisons, workloads,
metrics, cost and interpretation. During benchmark design, choose the maintained
benchmark entry point and shared execution/evidence helpers; do not implement it
as part of repository preparation. Coordinate interruption, resumption and completion
notification with the separate agent-operations discussion.
