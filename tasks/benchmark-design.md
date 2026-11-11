# Release evidence benchmark proposal

## Goal
Design one release-evidence benchmark that can defend explicit correctness,
performance and operating claims. Derive a functional check and quick development
comparison from it. Discover useful applications from product behavior; do not
optimize to a permanent target workload or conceal losses in an aggregate score.

## Authorized
Philip requested a comprehensive release-first proposal for review. Design, research,
repository inspection and recording the proposal are authorized. Implementation,
GPU acquisition, downloads, runtime repins, external publication and agent-operations
implementation remain unauthorized. Numeric gates below are proposed policy, not
approved performance requirements or measured properties.

## Done when
Philip reviews the release claim, coverage, evidence requirements, gates, cost and
repository integration. The executable manifest must subsequently resolve exact
models, corpora, limits and repetitions before acquisition. Approval of this design
alone does not qualify the product or authorize an unlimited GPU campaign.

## State
Owner: primary agent on main in /home/philip/vllm-semantic-cache-ext-next.
Status: revised release-first proposal; not implemented or exercised. The earlier
12-benchmark organization has been replaced. Useful cases are retained inside one
release benchmark. The only normal user-facing categories will be functional,
compare and release. Existing correctness tests remain tests, not extra benchmark
categories. Sol reviewed earlier characterization scope; this revised complete
release design has not received a separate independent review.

### 1. What a release verdict means

A release result evaluates a frozen implementation, configuration, runtime, model
set, workload population and operating envelope. It supports a statement such as:

"On the listed FP16 single-GPU configurations, the feature preserves the tested
exact-cache and protocol invariants, meets the declared quality and interference
bounds, operates through the stated load/lifecycle tests, and delivers a repeatable
useful benefit in the listed operating regions relative to the named controls."

It does not establish universal improvement, indefinite reliability, support for
untested models/hardware, or general model intelligence. The report must contain
ready-to-use claim text with these boundaries, not just a green badge.

Separate three verdicts:

- Measurement valid/invalid/incomplete: was the comparison actually executed and
  measured correctly? A valid negative result is successful measurement.
- Release eligible/not eligible/unresolved: do correctness, quality, operating and
  resource requirements hold throughout the declared envelope?
- Benefit supported/not supported/unresolved: does the stated performance claim
  meet its useful-effect and uncertainty requirements?

An optimization release requires eligibility AND its benefit claim. A correctness
fix can justify a release without a performance gain, but cannot advertise one.
Missing mandatory evidence blocks the corresponding verdict. Coverage cannot be
removed after seeing a loss to preserve an old claim; narrowing creates a new,
explicitly limited release claim and requires fresh confirmation where selection
could have biased the result.

### 2. Work completed to construct this proposal

Reviewed the current architecture and runtime constraints, actual CPU/GPU test
inventory, entry points, process/receipt tools and result convention. Revisited
historical interference/quality, workload/model dependence, measurement-reliability,
replay and external-comparison results. Inspected the old model/system catalogs and
sampled its workload catalog and repository-pressure tasks. These assets are known
historically exposed examples, not a complete re-audit of every raw experiment.

Researched primary vLLM benchmark guidance and MLCommons inference/agentic evaluation
methodology. The useful lessons are independent offered-load accounting, explicit
cache state, quality measured on the timed configuration, and application/session
outcomes. This proposal is not an MLPerf implementation or certification. Current
upstream documentation does not establish compatibility with our pinned runtime.

Mapped the historical failure modes to mandatory evidence below. No new serving
measurements, model-fit tests, corpus qualification or precision pilot were run.
Consequently, the proposal can establish coverage intent, not empirical sufficiency.

### 3. Frozen release contract and reference systems

Before confirmation, freeze the commit/executed-source snapshot, runtime and native
libraries, model/tokenizer revisions, templates, decoding and stopping rules,
resource budgets, workload manifests, split ancestry, scheduling policy, arm order,
metrics, useful effects, regression margins and analysis/stopping plan. Hash dirty
inputs when exploring; release evidence must identify a reproducible release tree.
No post-result tuning of an arm inside a confirmation comparison.

References:

- Native GPU prefix caching without CPU offload: practical simple-system reference.
- Current guarded CPU offload with advice disabled: causal wait-hint control.
- Release candidate with advice enabled: same guards/runtime/resource configuration.
- Frozen previous release when one exists: regression control, run on affected and
  core cases; never silently substitute the moving development branch.
- One qualified relevant external cache/offload alternative for a competitive claim.
  A narrow claim of improvement over native offload does not prove superiority to
  external systems. The report distinguishes those claims explicitly.

For advice attribution, match registration, offered hint events, API processing,
GPU/CPU capacities, scheduler options and observation overhead. Separately measure
end-to-end no-feature versus feature cost so matching integrations does not hide
registration/hint overhead. Namespace/timestamps legitimately differ per run; match
causal event semantics rather than replaying invalid clock identities.

Whole-system comparisons match hardware and explicit resource constraints; disclose
extra CPU memory and total observed usage. Include a labelled practical configuration
comparison where each system has documented sensible tuning. Do not confuse matched
KV capacity with equal total memory. Pin decode-KV publication, chunking, paging,
priority, templates and output settings; expose unavoidable differences.

The pinned ARC eviction limitation is a release-relevant confound. Count failed
stores/evictions and test its characterized path. If claimed benefit depends on the
bug, describe the claim as specific to that runtime; a broad policy claim needs a
separately authorized corrected-runtime sensitivity study. Do not assume identical
bugs cancel, or silently repair the runtime during the campaign.

### 4. Model and deployment coverage

Proposed minimum for a multi-model release claim:

| Role | Required evidence |
| --- | --- |
| Existing small qualified model | Harness controls, deterministic functional cases and fast regression feedback; not enough for practical-value claims. |
| Practical model A | Full release workload panel, actual cache competition, output-validity assessment and operational checks. |
| Practical model B from another family | Same panel; meaningfully different tokenizer/template and KV/compute geometry, not just another seed or alias. |
| Every additional advertised model/configuration | Functional checks plus representative load/pressure/quality cases; full panel for any model-specific benefit claim. |

A and B must fit the intended runtime/host and produce useful answers to the selected
tasks. Select exact models from inspected assets and current model documentation,
then verify fit, attention layout and generation before freezing the manifest. Two
families challenge obvious model-specific behavior; they do not prove arbitrary-model
support. Model names, byte identities and actual context lengths cannot remain TODOs
in an executable release plan.

Initial deployment claim: RTX 2080 Ti/WSL2, dense FP16 full attention, TP1, HND, eager,
TRITON_ATTN, same-host causal hints and one trusted operator. Host CPU/RAM/driver,
power settings and runtime are part of the identity. Native Linux, other GPUs,
quantization, BF16, graphs, LoRA, hybrid/sliding attention, speculative decoding and
parallelism are excluded until specifically qualified. Construction/refusal behavior
for unsupported settings is tested. We will not quietly replace the larger models
with the tiny model or quantization if fit is difficult: resolve the resource/scope
choice with Philip.

The product's scope/session names are not security tenants. Test endpoint auth and
identity binding, but make no multi-tenant confidentiality or remote-clock claim.
Broader claims require corresponding security/clock design and evaluation, not just
more latency trials.

### 5. The release workload panel

These are six sections of ONE benchmark, run with common measurements and controls.
They challenge distinct reuse/compute/session behaviors; none is a permanent product
objective. Concrete source selection follows characterization and asset inspection,
but preserves no-benefit and harmful-opportunity cases. Initial examples below are
proposed rather than already qualified corpora.

| Section | Actual execution | What it challenges | Validity/quality check |
| --- | --- | --- | --- |
| Independent requests | Fresh grounded QA/extraction/short generation, including little or no shared prefix | General overhead; decode-heavy work; accidental shared templates | Reference facts/fields, correct formatting, completion/length/stop checks |
| Shared-source requests | Different questions over common documents or fixed context, with hot and rare sources | Cross-session sharing, popularity skew and partial prefix overlap | Answers against retained source evidence; unanswerable/contradictory controls |
| Growing conversations | Real multi-turn outputs appended to subsequent requests, varying turn count and waits | Growing KV, return distance and identity lifecycle | Context-dependent facts/instructions and per-turn/final success |
| Tool-wait workflows | Real read/search/test or bounded lookup tools; subsequent prompts consume their results | Actual opportunity for causal hints; tool and integration cost | Claims checked against actual tool results; tool failure/timeout/truncation retained |
| Branching sessions | Shared initial context followed by divergent branches and a dependent final step | Shared ancestors, partial hits, abandoned branches and overlapping returns | Branch outputs and final synthesis checked against retained inputs/results |
| Mixed traffic | Fixed mixtures of the preceding sections, including hinted/unhinted and short/long requests | Interference, queueing and hiding harm in aggregate | Every cohort separately; constituent checks retained |

For each section include short/medium/long actual input lengths, short/long outputs,
and relevant context-near-limit cases within supported bounds. Controlled fixtures
vary reusable prefix, new suffix and generated length separately. Natural workloads
are not padded merely to manufacture reuse. Preserve actual token IDs after template
application and verify intended prefix identity. Different seeds are not evidence
of independent source material.

Use both fixed external request traces and actual output-dependent sessions. Fixed
prompts/schedules control nominal demand, not generated output identity. Exact
forced-token replay is optional only with a qualified mechanism and clearly limited
claims. Actual generation must appear in release evidence; matching token counts
or replaying yesterday's answers cannot establish application quality.

### 6. Required operating conditions and tractable matrix

Each practical model runs the six sections under three declared environments:

1. Roomy/light: reusable state fits, little queueing; exposes overhead without a
   retention advantage. Preserve genuinely no-reuse requests too.
2. Competitive/steady: GPU overflow with CPU working-set competition, at a shared
   offered rate below the control's sustained saturation boundary.
3. Busy/changing: higher offered load, bursts or shifts in the hot set, with bounded
   overload and recovery; mixed traffic includes misleading/abandoned advice.

This is a core floor of 6 x 3 x 2 = 36 matched workload/model/environment cells,
not 36 requests. Each includes independent source groups and repetitions determined
by the precision plan. These cells alone are insufficient for pressure/load
interactions: add a small interaction panel on each practical model crossing three
capacity regimes (GPU-resident, GPU spill/CPU fits, CPU competition) with three load
regimes (light, near sustainable, bounded overload). Reuse matching core cases when
valid; otherwise at most nine additional cells per model. This prevents pressure
and demand from always changing together.

Across the matrix explicitly cover uniform and skewed reuse, increasing intervening
distinct KV bytes separately from elapsed wait, cold and deliberately warm state,
active decode allocations and selected block-boundary lengths. Track which concrete
cell covers each condition. Every advertised model gets no-benefit, genuine restore,
CPU-competition and live-quality coverage. Missing intersections remain listed;
important newly observed interactions trigger additional cells, not a completeness
claim based on pairwise coverage alone.

Open-loop schedules record intended arrival before any client concurrency gate,
actual submission, first token, completion and backlog. All arms receive the same
absolute offered rates. Completion-dependent sessions additionally use predecessor
completion plus declared waits and are analysed as causal applications. Overload
uses declared queue/time caps and drain rules; don't run unlimited queues or count
only survivors. Report sustainable capacity and latency together. Configure SLOs only
for a stated use; where none exists, publish latency/load curves rather than invent
an arbitrary universal SLO score.

### 7. Correctness is a mandatory release gate

Integrate existing CPU/runtime/GPU evidence; expand actual hardware coverage where
current tests are narrow. No performance gain can compensate for a correctness error.

| Obligation | Required execution/evidence |
| --- | --- |
| Exact KV transport and safe publication | Direct FP16 byte checks over store/evict/restore, varied block counts/boundaries, overlapping loads/stores, and pressure; readiness only after successful transfer. |
| Correct reuse identity | Shared and nonmatching prefixes, namespace/session/generation/incarnation changes, partial chunks and computed-token limits; no sampled-but-uncomputed token published. |
| Advice is recency only | No pinning, prefetch, queue promotion, approximate reuse or scheduler-priority change; queue membership and exact native keys retain ownership. |
| Causal protocol | Real HTTP-to-engine delivery; timestamp/domain/deadline checks, exact replay, changed replay refusal, reset/retirement invalidation, capacity limits and explicit refusal reasons. |
| Failure handling | Failed stores never publish; failed loads invalidate destinations and safely recompute; unknown completion fence stops safely rather than continuing with corrupt state. Use safe deterministic injection in isolated qualification cases. |
| Concurrency/lifecycle | Requests finishing while hints arrive, cancellations/disconnects, retirement/reset, duplicate/delayed events and shutdown; no double release or retained stale references. |
| Endpoint/auth/input limits | Missing/wrong credentials, malformed/oversized/unknown-schema inputs, wrong scope identity and bounded refusal; valid traffic remains serviceable. |
| Resource boundaries | Session/event/metadata capacity reached, replaced and reclaimed; supported maximum boundaries and unsupported configuration refusal. |

Run deterministic fresh-versus-restored output comparisons and, where feasible,
logit comparisons with declared tolerances calibrated against fresh-versus-fresh
numerical variability. Exact KV bytes are required; exact output identity across
arbitrary concurrent schedules is not assumed. Any unexplained divergence is
investigated, not silently dismissed as nondeterminism. Ordinary realistic performance
runs also check output validity; passing a separate correctness fixture is not enough.

All advertised combinations must have applicable tests; document which checks use
real GPU/native paths versus fakes. Fault injection is separate from timed benefit
estimates and demonstrates expected safe behavior, not a normal-serving failure.

### 8. Output quality and useful work

Check the outputs of the actual performance configuration, not only a separate
low-load accuracy run. Retain task success, factual/field accuracy, format errors,
truncation, refusal, tool failure, EOS/context-limit termination and response lengths
separately. Performance is reported both for offered/completed work and correctly
completed tasks; no dropping failed tasks from time or denominators.

Use objective task checks where possible. Source-grounded or open-ended tasks need
predeclared rubrics, blinded paired review and a random sample plus separately labelled
failure-focused review. A failure-selected audit cannot estimate population accuracy.
If an automated judge is used, pin it, measure disagreement against human checks and
account for cost; no external paid judge is assumed authorized. A weak judge cannot
establish quality equivalence.

Require baseline absolute usefulness as well as candidate non-inferiority. Two
systems that both fail the task do not qualify a useful-work claim. Task-specific
minimum success criteria must be frozen before confirmation. The practical models
must support the tasks; impossible or ceiling-saturated tasks are replaced during
pilot with reasons, not silently removed after comparative results.

### 9. Stability, recovery and installability

Include sustained mixed traffic, session churn, changing hot sets, quiet/restart
periods and repeated capacity exhaustion/reclamation. Observe windowed throughput,
latency, memory, metadata, queues, errors and completion. Distinguish allocator/cache
warmup from continued unreclaimed growth. Verify logical counts directly and retain
RSS/GPU allocation as complementary measurements.

Proposed first-release exposure: an eight-hour mixed-load run plus two independently
started two-hour runs for candidate and matched native control on the primary practical
model, and at least a two-hour mixed/churn run on the second. Also require three full
configured session/event-capacity churn/reclamation cycles; shorten configured limits
in a separate stress case when normal limits make turnover impractical. These are
engineering exposure targets for review, not proof of a universal failure rate or
unlimited stability. Pilot may justify a different budget before confirmation.
Report observed error counts and exposure; do not infer request independence from a
single long process. Known operational risks may demand longer or targeted exposure.

Exercise real request abort, client disappearance, failed startup, timeout, server
termination and restart, disk exhaustion through bounded fault simulation, and
interrupted report/receipt writes. Verify recovery generation and cleanup of only
owned processes. Partial artifacts remain verifiable as incomplete. Restarted caches
are cold unless deliberately rebuilt; resume never claims to restore lost cache state.

Reconstruct the release from its retained Git source and locked inputs in a clean
checkout/environment. Exercise installation, documented command, one live serve path,
offline verification and report regeneration. Report which dependencies/models came
from caches; cached setup is not a fresh-download claim. Offline report recovery must
not require the original checkout. Exact reruns still require declared external assets.

### 10. Demonstrate that the benchmark can detect problems

Before using the benchmark to decide a release:

- A/A trials measure repeated identical-system noise and order/drift effects.
- A known delay/cost inserted inside the measured path must appear in independent
  timing. The control is labelled and excluded from candidate benefit estimates.
- An intentionally inactive advice path must fail activation validation in a case
  that requires advice, while no-benefit controls remain valid with zero activity.
- A deliberately corrupted/incorrect result must fail the independent correctness/
  task checker; a missing counter or event must not become a zero.
- A client-side backlog must appear in scheduled-to-result timing, not disappear
  behind a concurrency semaphore. Overload failures remain in the denominator.
- Controlled cold/warm cases verify actual cache state and detect carryover.
- Mismatched source/model/config, truncated streams, duplicate records, changed
  outputs and interrupted attempts must be detected by offline verification.
- A minimal frozen evidence bundle must regenerate its displayed metrics/figures
  from raw events, with hand-calculated fixtures checking the aggregation logic.

These establish sensitivity of specific checks; they do not prove universal detector
power. Measure diagnostic overhead. Expensive tracing runs separately or has symmetric,
measured cost. Prefer existing supported runtime counters and narrow adapter hooks;
never mistake a successful receipt hash for evidence that the scientific claim holds.

### 11. Statistics, data exposure and stopping

Inventory existing corpora/source trees, licenses and historical exposure before
acquisition. Related sessions, turns, variants and source projects stay in the same
split where relevant. Maintain development, exploratory discovery and reserved
confirmation groups. Historical research corpora are exposed. Reserve inputs before
candidate selection; new seeds over memorized templates are not enough. Release
confirmation data can become regression data later, but fresh confirmation is then
needed for new selected claims. Keep holdout identities/provenance without using
outcomes to tune the candidate.

Choose primary outcomes and claimed regions before confirmation. For each relevant
cell, report paired whole-run effects and uncertainty, with independent run blocks
and grouped source resampling where appropriate. Requests within a session/process
are correlated. Predeclare multiplicity handling for multiple confirmatory claims
(e.g. simultaneous intervals or Holm adjustment); exploratory heatmaps remain
exploratory. Separate discovery-selected regions from fresh confirmation estimates.

Pilot estimates noise, baseline variability and task-success frequency; calculate
sample sizes against useful-effect/non-inferiority margins. Require independent
process starts and sessions, not merely many requests in one run. Freeze repetitions,
analysis and stopping before confirmation, or predeclare a valid sequential procedure.
If the resource cap is reached with an interval crossing a gate, verdict is unresolved.
No rerunning only unfavorable arms/pairs. Report incomplete pairs and cause-specific
failures; paired estimates use complete pairs without erasing failed acquisition.

Tail claims require adequate independent exposure and quantile uncertainty, not a
p99 from sixteen requests. With insufficient evidence, report descriptive quantiles
and explicitly withhold the tail guarantee. Averages and weighted scores cannot
cancel failed mandatory cohorts. Any headline aggregation includes its weighting
and sensitivity, alongside the full model/workload panel.

### 12. Proposed release gates

These numbers are concrete starting recommendations for Philip to review. They are
engineering tolerance choices, not laws or conclusions from old measurements. Pilot
checks whether they are measurable at affordable cost; noisy results do not justify
relaxing a gate after the fact. Require the stated confidence bound, not just a point
estimate; confirmation uses the declared multiplicity policy.

| Gate | Proposed requirement | Reason |
| --- | --- | --- |
| Correctness | Zero observed cache corruption, identity violation, unsafe reuse or protocol invariant failure in required tests/runs; all mandatory functional cases pass | Exact cache behavior is the product contract; finite tests do not prove zero future failures. |
| Normal operation | No unexplained crash, hang or failed normal request in admitted below-overload cases; intentional fault cases behave as specified | Throughput cannot compensate for instability. Unrelated infra faults invalidate affected evidence rather than count as a product pass. |
| Coverage | Every mandatory model/workload/condition cell complete and valid, with required activity and quality evidence; no silent fallback | Prevents scope shrinking or inactive-feature wins. |
| No-op overhead | Lower confidence bound of candidate/control throughput ratio >= 0.97 and upper bound of p95 latency ratio <= 1.05 in declared no-benefit cases | Limits the cost imposed on traffic unable to benefit. |
| Protected traffic | Upper confidence bound of unhinted cohort p95 TTFT and end-to-end latency ratios <= 1.05, and throughput lower bound >= 0.97 at matched admitted load | Prevents aggregate wins from hiding displacement harm. Report absolute delays too. |
| Quality | Lower bound of candidate minus control task-success difference >= -0.02 (two percentage points), plus frozen absolute task-usefulness floors; stricter zero-error checks for deterministic assertions | Performance must not be purchased through worse answers. Margins may need tightening for the intended task; reference quality is not presumed. |
| Claimed benefit | Lower bound of valid-task-throughput ratio >= 1.05 OR upper bound of chosen latency ratio <= 0.95 in predeclared useful regions, with no failure of protected secondary metrics | Requires a useful, repeatable effect rather than statistical detection of a negligible change. The chosen primary metric is frozen in advance. |
| Resources | No unbounded logical state growth; no breach of declared CPU/GPU/disk budgets; after equivalent churn/drain, retained logical state returns to expected bounded level | CPU memory spent to save GPU work must remain explicit. RSS alone cannot establish a leak or its absence. |
| Repeatability | Confirmatory result survives independent starts and held-out source groups; all artifacts verify and summary regenerates | One favorable run or a non-reproducible report is insufficient. |

No universal all-workload win is required. For broad multi-model benefit language,
confirm benefit on both practical model families and more than one independent
workload/source family, with all advertised-envelope safety/regression gates met.
If benefit appears only in one region, release text must say so. Stronger words such
as "generally faster" require an explicit representative population and weighting;
this benchmark does not automatically authorize them. If adversarial advice causes
unacceptable harm, restrict the supported advice policy or repair it; do not average
the harm away or claim robust operation under that condition.

Exact absolute latency/task-success floors, overload acceptance behavior and any
exceptions require an explicit claim-specific contract before execution. The defaults
above make the proposal reviewable but do not substitute for that contract.

### 13. Competitive evidence

Include native references in the entire core panel. Before claiming competitiveness,
select at least one relevant maintained external exact cache/offload system, inspect
its supported configuration, and qualify a fair adapter. Start with the same core
workloads/claimed regions and include no-benefit and interference controls; cannot
select only workloads on which our candidate won. If practical budget prevents full
coverage, limit the competitive claim to the exercised common subset.

Report separately: isolated advice effect, whole-system comparison at matched
resources, and practical tuned-system comparison. Use actual output/task quality;
token rate does not rank differently sized or incorrect answers. An unavailable or
unsupported competitor is listed as such, never treated as slower. Existing historical
Continuum/upstream experiments provide fairness lessons, not current qualification.
Selection is an implementation-stage research item; no package is silently chosen
or downloaded by this plan.

### 14. Release evidence package and claim audit

One report contains:

- Release claim and named exclusions; frozen manifest and full coverage table.
- Correctness and operational gate ledger with direct evidence links.
- Model/workload performance panel, uncertainty and absolute timing/resource values.
- Quality, failure and hinted/unhinted cohort panels alongside performance.
- Cold/warm and load/capacity context, actual activation and saved-work evidence.
- Regressions, adverse/missing/incomplete cases and their disposition.
- External-system fairness statement where competitive claims are made.
- Repeatability, setup/restore evidence and exact reproduction commands.
- A gate-by-gate eligible/not-eligible/unresolved verdict and permitted release text.

Each material claim maps to one or more valid measurements plus its acceptance rule.
An independent reviewer, when authorized, should challenge this mapping and inspect
selected raw traces, not just agree with a summary. No missing obligation is converted
to a pass through review. Unknowns remain visible. A release-significant fix after
confirmation invalidates affected evidence; rerun affected cells and common sentinels,
and repeat broad validation when scheduler/runtime/geometry changes have broad reach.

### 15. Three-category workflow and minimal repository integration

Only three normal commands/profiles are proposed:

- Functional: existing tests plus a small real-server subset of release correctness
  cases. Fast setup confidence; no performance release verdict.
- Compare: a stable, stratified subset of release workloads/conditions/references,
  fewer repetitions and shorter lifecycle exposure. Covers every major workload
  shape, no-benefit and interference, not only synthetic favorable cases. Advisory.
- Release: the frozen full contract above, qualified instrumentation, fresh confirmation
  inputs, required repetitions, lifecycle and recovery coverage. Gate ledger mandatory.

The same case IDs, generators/adapters, schema, metrics and verification serve all
three. Diagnostic investigations are case filters or separate traced runs, not extra
normal categories. Compare does not secretly redefine denominators or stopping rules.

Proposed single CLI: `uv run --locked python -m tools.benchmark`, with plan/run/verify/
report/resume subcommands and `--profile functional|compare|release`. CLI does not
exist yet. Reuse tools/process.py ownership/timeout/cleanup and existing snapshot
helpers; preserve functional receipt semantics when extracting shared helpers.
Keep runtime_verify as the existing correctness tool, not a second campaign runner.

Place compact definitions/profiles and fixtures in benchmarks/; typed orchestration,
accounting and validation in tools/benchmarking/; vLLM imports/telemetry only in
wait_hint_offload/vllm_adapter/. Use explicit workload/system adapters and model
profiles, no dynamic plugin framework. All environment reads remain in config modules.
New candidates change arm identity/config, not copied scripts. New models qualify
existing cases; new mechanisms declare their actual activity evidence.

Add one reviewed script exception, system-map entry, README link and docs/benchmarking.md.
Propose a short AGENTS link telling agents which profile to use, that valid measurement
is not a win, and to retain negative/incomplete attempts. Full portable checks cover
schemas, references, accounting fixtures, adapters and offline verification without
GPU. Pinned CPU/GPU lanes qualify actual integrations. No hosted CI or GPU reservation
system is introduced. A project execution skill can later wrap the exercised command;
agent operations must not own another benchmark implementation.

Raw plans/attempts/events/output/frozen execution trees stay outside Git. Preserve
source, schemas, definitions, compact chart inputs and coherent findings in Git.
Resume creates linked attempts at independent block boundaries; it never overwrites
failures or fakes cache continuity. Long acquisition needs an explicitly authorized
budget, durable owner and tested notification/cleanup; until available, run bounded
supervised blocks. close-results produces a concise durable report/chart and FINDINGS
entry with the larger release evidence linked. Local bundles preserve project records;
raw artifact backup remains separate and no new remote is authorized.

### 16. Cost and implementation sequence

The core matrix is 36 cells x 3 arms x R independent blocks = 108R arm-runs before
previous-release/external references, interaction cells and sustained tests. At a
hypothetical three minutes per arm and R=6 that alone is 32.4 GPU-hours, excluding
startup, warmup, drain and failures. This is a sizing illustration, not a repetition
recommendation or measured runtime. Broad release confidence costs more than a short
screen; expose that cost before committing resources.

Plan reports measured/estimated startup and per-case time, event/record volume,
model/runtime storage, peak memory and acquisition ceilings. Pilot a representative
short, long, mixed and failure case to estimate those values and statistical
precision. Select valid repetition/queue limits and approved budgets before the full
run. If the necessary evidence is unaffordable, report the unresolved claim or negotiate
scope explicitly; do not silently rename the quick comparison "release validation".

Build in this order:

1. Freeze this release contract and choose actual models/source assets and quality
   rubrics; inventory coverage and known unknowns before runner implementation.
2. Implement minimal shared plan/runner/events/verifier around one real path and
   the benchmark-sensitivity controls. Exercise relevant real failures.
3. Implement the release workload panel and correctness gates; derive functional
   and compare profiles from the same definitions immediately.
4. Pilot, resolve telemetry/cost/precision, and freeze confirmation protocol/data.
5. Execute authorized release evidence, preserve all outcomes, validate independently
   where authorized, and publish the gate ledger and bounded claims.

No universal framework, automatic policy optimizer, bespoke forced-token engine or
model zoo is prerequisite. Missing telemetry or incompatible models are engineering
work to resolve, not assumed available features.

### 17. Why this is sufficient for the stated claim, and remaining uncertainty

| Concern that could invalidate a release | Required defense |
| --- | --- |
| Fast because it reuses wrong KV | Direct byte/identity/publication tests, actual pressure/concurrency and timed-run output checks |
| Fast because it returns shorter/worse answers | Actual output lengths/termination, task validity, absolute usefulness and quality non-inferiority |
| Wins only with privileged hints | Real tool/application signals and costs; ideal hints labelled diagnostics only |
| Hides harm to unrelated work | Mandatory per-cohort latency/throughput gates and mixed/adverse-advice cases |
| Cache state/load generator manufactures gains | Cold/warm canaries, scheduled-arrival accounting, A/A and sensitivity controls |
| Small model or one corpus determines everything | Two practical families, six workload structures, source-group holdouts and explicit coverage matrix |
| Pressure/load interactions missed | Independent capacity/load panel, decode competition and targeted follow-up for observed interactions |
| Tuning/search finds a lucky winner | Frozen confirmation, independent runs, multiplicity control, exposed-data tracking and preserved failures |
| Only works briefly or in the developer's checkout | Lifecycle/soak/failure recovery and clean reconstruction/report regeneration |
| Another system/configuration explains the ranking | Guarded causal control plus resource-matched/system comparisons and fairness disclosure |
| Known upstream bug drives results | Store/eviction accounting and explicit pinned-runtime scope; corrected-runtime study for broader claims |

This is a reasoned obligation map, not proof that nothing has been missed. No finite
suite proves arbitrary model/hardware compatibility, multi-tenant security, indefinite
uptime or universal workload benefit. Those are excluded from the initial claim,
not silently treated as passed. New mechanisms or observed failure modes require
review of this map and focused cases. Confidence becomes empirical only after the
harness sensitivity checks, coverage and repeated release runs actually succeed.

Still unresolved before executable release acquisition: exact practical model IDs
and fit; fresh corpus/source groups and rights; absolute task-quality floors and
latency objectives where relevant; telemetry availability/overhead; statistical power;
external-system compatibility; raw-storage budget; durable ownership/notification;
review of proposed numeric tolerance and exposure targets. These are explicit release
readiness blockers, not items an agent may fill with favorable defaults after results.

### 18. Sources and evidence used

- [vLLM benchmark CLI](https://docs.vllm.ai/en/latest/benchmarking/cli/): cache carryover and serving benchmark guidance.
- [vLLM serving controls](https://docs.vllm.ai/en/stable/cli/bench/serve/): offered load versus concurrency, timing and goodput.
- [MLCommons inference rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc): distinct accuracy, performance and scenario requirements.
- [MLCommons agentic evaluation](https://mlcommons.org/2026/07/agentic-inference-for-mlperf-inference/): session outcomes and checking actual performance-run outputs.
- [Architecture](../docs/architecture.md) and [runtime](../docs/runtime.md): actual contract, support boundaries and known ARC limitation.
- [Historical interference/quality](../results/interference-quality/result.md): aggregate gains can conceal harm and quality uncertainty.
- [Historical workload/model dependence](../results/workload-dependence/result.md): heterogeneous outcomes and missing actual pressure.
- [Historical measurement safeguards](../results/measurement-reliability/result.md): failure/cleanup controls are not policy value.
- [Historical external comparison](../results/external-comparisons/result.md): runtime/output differences and weak replication.
- [Historical replay](../results/wait-recency-replay/result.md): constrained replay cannot establish general live benefit.

## Next
Philip reviews the release-first contract, especially the coverage floor, proposed
numeric gates, sustained exposure and explicit deployment boundary. Then resolve
actual model/data selection and pilot resources before implementation/acquisition.
