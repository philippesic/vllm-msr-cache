# MSR hybrid qualification

Authorized on 2026-11-03: implement and measure the proposed hybrid, advance to
comparative performance tests if qualifications pass, investigate plausible
in-scope alternatives on failures. No deployment or default-policy change.

## Questions and decision rules (before new measurements)

1. Does actual queued-demand restoration leave delay worth hiding? Trace native
   arrival, async-load issue/receipt, admission and per-step reclaimable blocks.
   Report transfer-residence time separately from queue time. The former is an
   optimistic ceiling on direct request savings, not a throughput bound. A small
   screen or zero restore activity cannot reject the mechanism. Follow inactive
   screens with greater working-set pressure and/or longer histories.
2. Do request-local MSR features improve retained cache value over cheap features?
   Evaluate held-out conversations, never random turns across the same session.
   Retain original scalar mean-QK, richer per-block score distributions, and
   query/key representations as separately named hypotheses. Extra representation
   success does not establish original MSR success. Freeze features/selection
   before evaluating; report negative results and uncertainty.
3. Is the combined opportunity positive after capture, scoring, displacement and
   interference? Four eventual live arms: ARC, wait-hint ARC, wait-hint ARC plus
   nonsemantic prefetch, and semantic hybrid; add semantic retention only to
   identify interaction. A nonsemantic-prefetch loss alone does not reject hybrid.

Advancing to a live hybrid requires measurable relevant activity and positive
held-out cache-value evidence against cheap features, with a paired 95% interval
above zero and capture/scoring costs below the attainable gain. Evidence on a
proxy label cannot satisfy the real reuse qualification regardless of significance.
No invented SLO: report mean and tails and seek repeatable net gain, without
claiming a default or cross-hardware generality from one configuration.

## Data boundary

Retained WildChat-1M normalized source has 800 public conversations. Its user
timestamps are missing; assistant timestamps record response completion (dataset
card). Recorded continuation and completion gaps are useful *proxy* labels but
not observed user-return times. Terminal records are not proof of abandonment.
Any proxy pilot must say so. Existing fixed-turn/random-wait serving workloads
can test system opportunity and overhead, not natural semantic reuse prediction.
Do not train on future text, final session length, next-answer length, or future
timing; seal conversation splits and retain causal feature construction.

## Execution and accounting

Use maintained serving suite, explicit local models and equal CPU/GPU pools,
native opt-in observations identical in compared arms, shared GPU reservation,
source manifests, retained failures, verified owned-process cleanup. Keep large
data ignored. No runtime policy change until its information/opportunity gates
are supported. Instrumented baseline timings are diagnostics, not a new qualified
performance result. Fixed GPU allocation means no VRAM-reduction claim without
an explicit smaller-pool comparison at a maintained service level.
