> Historical pre-port proposal. The active scope and superseding decisions are in
> [the port plan](porting-overhaul.md). Benchmark methodology and high-level agent
> operations remain deferred; no commands below imply implemented capability.

# Remaining port stages

Historical planning inventory. The newer [wait-hint port plan](porting-overhaul.md)
proposes the replacement scope and sequence after Philip's clarification. Existing
implementation policies remain in force until its M0 changes are authorized.

This is the consolidated plan for the work that remains after stage 0, the non-source foundation already in this repository. The remaining stages are source port, benchmark, GPU qualification, and skills. Nothing advances a stage without Philip's explicit go. This document is planning only; it does not authorize starting any of those stages.

The original porting plan used twelve implementation steps. Steps 1 and 2 were the provenance capture and the foundation check layer, which are done. The mapping below uses Philip's stage names and pulls the later steps into them. Where a later plan or foundation note narrowed a step, that narrowing is recorded. Where a stage was left undefined, this document says so and does not invent detail.

## Sources

- `/workspace/research/PORTING_PLAN.md` on the shared box, copied from the read-only Sol plan in `/home/philip/.t3/scratch/2026-11-09-read-only-analysis-and-planning-f5d203c3/PORTING_PLAN.md` on the 2080 Ti (DESKTOP-T4JKGJB). Prepared 9 November 2026. Cited below as the porting plan.
- `/workspace/research/vllm-porting-plan.md`, the short completion note for that same thread.
- `/workspace/research/vllm-skeleton-report.md` and `/workspace/research/vllm-dune-additions-report.md`, the foundation and guardrail completion notes.
- This repository's `README.md`, `docs/benchmarking.md`, `docs/runtime.md`, `docs/system-map.md`, `docs/checks.md`, `docs/decisions/`, and `FINDINGS.md`.
- `/workspace/research/migration/BRIEF.md`, Philip's approved stage order, including that skills come last and are out of scope until a later request.
- `/workspace/research/migration/DUNE-ADDITIONS.md` and `/workspace/research/vllm-dune-issue-rule-table.md`, for deferred runtime and ownership work.

## Current state

Stage 0 is complete in this local repository, with no remote. `./check` is the one CPU command. `uv run --locked python -m tools.runtime_verify` exists only as a stub that exits 2 and must not be treated as a pass. The benchmark entry listed in the system map is still absent. Native compiler recipes remain unqualified. The Python package namespace is still `semantic_offload`.

The foundation completion note recorded Philip's then-open decisions: permanent name and license, external artifact backup location, initial runtime feature breadth, future quality and fairness margins, and native CI budget. Historical webhook revocation was, and remains, unverified in the plans.

## Source port

### Scope

Port exact CPU offload and wait-hint behavior first, with a matched native ARC control. Semantic ranking and causal prefetch stay behind separately qualified experimental profiles unless Philip chooses a broader first milestone. SVD, MTP, the custom affinity scheduler, and unqualified ARC or AWQ experiments stay historical unless separately requested.

What is ported, in the plan's order after foundation:

- Exact runtime, patch, and ABI manifests, and one supported constructor path, preserving the distinct root-development and Stage 6 dependency closures (plan step 3).
- Reproducible native vLLM and LMCache build recipes for sm_75 and sm_120, with inspection of every required binary (plan step 4).
- Exact transfer and job completion, and native ARC control, including computed-token and store-layout guards (plan step 5).
- Observed prefix and transcript registry, validated hint endpoint, CPU recency advice, and bounded continuation priority (plan step 6).
- Optional semantic summaries, query capture, and asynchronous scoring, then optional prefetch ownership, only after the exact-advice baseline is reviewable, and not enabled by mere presence of the code (plan step 10).

What is not ported in this stage: the benchmark runner, baseline adapters as a performance study, GPU jobs, the connector or vendored vLLM tree wholesale, campaign machinery, the GPU lock system, skills, or a claim that runtime verify has passed.

The plan splits the 10,177-line connector, the 2,383-line manager, and the 2,259-line worker along named responsibilities: typed contracts, retention and admission, prefetch state, scoring, transfer, vLLM integration, and observation. Import boundaries already enforced in the foundation stay in force.

### Entry criteria

Philip's explicit go for the source stage. Stage 0 remains green. The permanent name, license, artifact root, and initial feature breadth should be settled first if they still affect what is extracted, as listed in the plan's open decisions. No source copy starts from a dirty foundation tree.

### Exit evidence and gates

Each rewritten responsibility is accepted only after its parity gate passes in this repo. Old tests are characterization material; they were not re-run for the plan. Historical clean-install evidence cited 1,381 passed and 26 skipped for missing artifacts; those skips are not acceptable evidence in the new checkout.

- P0, runtime and config: the pinned installed runtime loads the actual connector, spec, and worker; wrong Python, wheel, patched API, unknown key, or unsupported layout is rejected (plan section 4).
- P1, exact storage and transfer: exact key, slot, group, and order mapping; complete writes before readiness; failed submission, cancel, or reset never publishes invalid data. Later GPU byte-level round trips are a qualification gate, not a CPU exit.
- P2, admission and eviction: native ARC decisions on the control path; non-mutating victim preview; referenced or pending blocks ineligible; equal-cost and chain rejection atomic.
- P3, hints, lifecycle, and priority: same TTL, namespace, rounding, idempotence, and cancellation; CPU-only hints do not pin, create blocks, score, or start GPU work.
- P4, speculative prefetch: 137 functions in `test_step_1_5_prefetch.py` are the old characterization set. The feature stays disabled until the complete gate passes.
- P5, summaries and scoring: numeric equality within declared tolerance; old score generations never update a reinserted key. Required only if that optional work is in the approved breadth.

The plan's effort for these source steps is 1.5–2.5 days (runtime manifests), 2–4 days (native recipes), 2–3 days (transfer and ARC), 2–3 days (hints and priority), and 3–5 days (optional semantic and prefetch). Those are engineering estimates, not a schedule.

### Risks

Private vLLM callback and block-pool seams are more fragile than ordinary imports; start from one exact pin and fail closed. Torch working does not certify LMCache or a transfer component. Scattered job IDs, refcounts, and summary generations can race if algorithm and transport are rewritten together. Old tests and scripts sometimes require absent output; missing evidence is a documented gap, not a skip or an invented golden result. Concurrent 5080 Stage 6 work can move past the plan's snapshot; re-read that branch before a later handoff.

## Benchmark

### Scope

Rebuild one public entry, `uv run --locked python -m benchmarks`, with `plan`, `run`, `verify`, and `report` modes over one versioned schema (plan section 8 and steps 7–9 and 11). Then integrate the Continuum and vLLM PR #59714 adapters with pinned environments, and port observation, attribution, and quality validation.

What is in: smoke, screen, confirmation, and operational tiers, plus repository or tool-graph workloads used inside those tiers; typed plans and results; owned process cleanup; raw output only under an explicit external `--out`; matched comparison refusal with no `--force`; workloads that verify their own premise; correctness checks kept separate from timing.

What is not in: the old GPU lock or any successor reservation protocol; a second ladder of ad-hoc campaign launchers; automatic GPU work on untrusted PRs; treating three confirmation repeats as universal statistical sufficiency; calling quality equivalence from token counts or failure-selected reviews.

Default tier sizes in the plan are migration references, not a complete Stage 6 study: smoke is 1 repetition, 4 sessions times 2 requests, concurrency 1, 8-token cap; screen is 1 repetition, 6 sessions times 2 requests, concurrency 2, 32-token cap; confirmation is at least 3 repetitions, 12 disjoint sessions times 3 requests, concurrency 4, 128-token default cap; operational is 1 process per arm with 12 sessions times 3 requests times 4 waves.

### Entry criteria

Philip's go after source parity for the approved first candidate is reviewable. The plan says the exact CPU-offload and advice baseline and the benchmark spine should be reviewable before optional semantic or prefetch extraction.

### Exit evidence and gates

- P6, benchmark validity, and P8, lifecycle and reproduction: frozen offered work, true process repetitions, balanced order, token limits and finish reasons retained, unknown counters reported as unavailable. A pressure fixture with zero restores fails its premise. Native versus candidate mismatch fails comparison.
- P7, causality, observation, and quality: no aggregate hit count used as exact saved work; missing counters stay unavailable; quality checks keep all failures, caps, and denominators.
- Every comparison mismatch and bad premise has a deliberate failing test, plus a positive control. Stub-server plan, run, verify, report, offline relocation, and interrupted-run fixtures pass.
- Wrong baseline revision, capacity, or activation is rejected for the published adapters.

The plan's effort is 2–3 days for the CLI, 1.5–2.5 days for adapters, 1.5–2.5 days for observation and quality, and 1–1.5 days to reconcile docs and inventory rows. Foundation already drafted the protocol in `docs/benchmarking.md`; the runner itself is still future work.

### Risks

Old repository tasks read files the new repo drops. Freeze old sources for reproduction; a new-repo task set has a new identity and needs its own calibration. Pressure depends on machine, model, and output evolution; screen success does not guarantee timed confirmation activity, as in the later 5080 B06 example. Quality and performance can move in opposite directions. False-hint interference can hide inside aggregate throughput; the Stage 6 fair-hint bound that was exceeded was +14.56% misleading-hint TTFT, interval [1.1066, 1.1860], against a 5% bound.

## GPU qualification

### Scope

Separately authorized hardware work, first on the 2080 Ti, then on the 5080: transfer, then fresh-reference smoke, then purpose-valid screens, then a frozen new study, then confirmation, then operational and quality audit (plan step 12 and section 7.4). Runtime verify becomes a real store, evict, and restore round trip on a tiny model, with restored data matching exactly. That full body was deferred in the Dune pass; only the failing stub exists now.

What is not in this stage: calling binary inspection a completed device proof; treating a transfer round trip as a completed serving study; pooling cross-host results into one speed ratio; automatic qualification on untrusted pull requests.

### Entry criteria

Philip's go after the benchmark spine can refuse bad comparisons and after each adapter has a pinned, inspected build for the target machine. Disk, fixture, and backup locations should be decided before large acquisition. The plan recorded that today's ignored raw artifacts are not in Git.

### Exit evidence and gates

- Each adapter and host passes its own hardware gates. Full process blocks and fixed budgets.
- No pressure claim without actual returning CPU restores, and, for retention, eviction-affected returns.
- Quality equivalence only if a preregistered analysis establishes it. Stage 6 objective coverage was 10,616 primary and 3,504 addendum finals; equivalence remains unestablished in FINDINGS.
- Real FP16 GPU-to-CPU-to-GPU byte equality, both selected transfer backends where applicable, layout and slot-replacement cases, transfer failure and fence ordering, and, where supported, Compute Sanitizer checks on small kernels. The plan did not run these.
- Historical references, not reruns: Stage 5's unforced functional probe passed 48 assertions across nine waves; the 5080 rebuild showed a 393,216-byte bidirectional LMCache round trip; the November 9 5080 study was still incomplete at the plan's checkpoint.

The plan's engineering estimate is 2–4 days, plus declared acquisition and review time. The 5080 November 8 failure was a missing sm_120 and PTX in the original LMCache wheel; the later rebuild of the same source is not a finished performance study.

### Risks

A later 5080 campaign may have advanced past the plan snapshot. Build cost for dual-architecture upstream wheels may dominate CI; the native CI budget is still an open decision. Workload inactivity can appear after a successful screen. The leaked historical webhook credential was identified in old evidence and still needs owner-side rotation; that work is independent of qualification and is not done by this plan.

## Skills

Out of scope until Philip makes a later request. The migration brief and the porting plan both say agent skills come last and are not part of this port. The Dune issue table deferred blind-eval skills. This stage has no inventory, no entry criteria, no exit gates, and no effort estimate in the existing plans.

## Open questions awaiting Philip

These are from the porting plan's section 14 and the foundation completion note. None of them is settled in the later foundation work except where a default was recorded as still open.

1. Permanent repository name, license, and remote. The directory has been renamed to follow the old checkout, `vllm-semantic-cache-ext-next`. The public GitHub name of the old repo is `vllm-msr-cache`. Publication visibility is still Philip's choice.
2. Initial supported feature breadth. Recommended default: exact CPU offload plus wait-hint ARC first; semantic ranking and causal prefetch later, behind experimental profiles.
3. Support promise. Recommended default: the three historically tested dense GQA models, FP16, TP1. Broader layouts and engines need separate qualification.
4. Task-level quality margin, latency and fairness bounds, and useful throughput gain before a new confirmation study. The old 3% overhead and 5% fairness bounds are historical references, not a finding that they were met.
5. Native build artifact distribution and CI budget, including whether images and wheels stay private.
6. External artifact and backup location for raw evidence and caches, before the old repo is treated as permanently frozen.
7. Rotation or revocation of the exposed historical webhook credential. The plans record the fact and the file location only.
8. Whether to capture a final immutable checkpoint of the newer 5080 branch after that study completes, including inactive and failed blocks.

## What this document does not add

It does not change effort numbers, invent a skills plan, rerun old tests, or start a stage. Historical measurements stay attributed to their original receipts.
