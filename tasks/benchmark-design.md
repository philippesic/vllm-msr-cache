# Benchmark optimization and release evidence plan

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
categories. Sol adversarially reviewed the complete release-first plan at 6f4f09f.
The primary agent incorporated its findings and independent web research below.
A second Sol review found the revised plan ready for user design review with two
wording corrections, now applied. Gates remain proposals for Philip, not accepted
product guarantees. No reviewer qualified a runtime or performance result. Subsequent user-directed
comparison, workload and model refinements have not received another Sol review.
Philip has fixed performance repetitions at one validation and two release blocks;
model pair and concrete workload counts below remain proposals for review. Philip
requires an 11 GiB 2080 Ti primary target, optional 16 GiB 5080 runs, current model
architectures and coding-first coverage. Previous small full-attention pair rejected.

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

Gate identities must be explicit in each manifest: population, candidate, named
comparator, traffic model, primary outcome, denominator, activity requirement and
acceptance rule. Separate advice overhead (enabled versus disabled with matched
registration/events), full integration overhead (feature-enabled versus no-registration/
no-hint guarded offload), and whole-system offload versus GPU-only cost. No-reuse,
GPU-resident reuse, registration-only and useless-hint controls are distinct cases.
Only the first pair isolates advice; no blanket "control" field may change meaning
between a chart and a gate. Valid-task-rate ratios require a sufficiently useful,
nonzero baseline; otherwise the benefit comparison is unresolved.

### 3a. Revised comparison roster for review

Research on 2026-11-11 and the old published-system adapter establish that the old
four names are not a sufficient or accurately labelled permanent matrix. The old
"upstream" arm is karen-sy's draft vLLM PR #59714 (open, draft, unmerged when checked),
not released upstream vLLM. It applies GPU cached-block priorities at completion;
this differs from our CPU ARC refresh during a wait. Continuum's public README calls
its implementation a preview without the paper's estimation component. Preserve these
identities and do not generalize preview results to the complete paper system.

Proposed comparison roster, subject to runtime qualification:

| Target | Role and proposed coverage |
| --- | --- |
| Pinned vLLM GPU prefix cache, no CPU offload | Mandatory simple deployment reference across core compare/release cases |
| Guarded native CPU ARC, advice disabled | Mandatory causal control, same runtime/guards as wait-hint |
| Guarded native CPU ARC plus wait-hint | Candidate, all core cases |
| vLLM plus LMCache, plain exact CPU caching | First external practical offload target; qualify on the common model/workload subset, include in compare and release |
| Continuum public preview plus its required LMCache integration | Direct session-aware research competitor; include live-agent, low-opportunity and interference cases in compare and release, not only a favorable tool-wait case |
| GPU retention PR #59714, or a verified merged successor | Relevant GPU-tier mechanism; use the four-way experiment below after explicit runtime-integration approval |
| SGLang HiCache / TensorRT-LLM retention | Relevant alternatives to reassess if the deployment expands; not mandatory extra engine ports for the initial 2080 Ti scope. Hardware/runtime feasibility remains unverified, not claimed unsupported or inferior. |

Plain LMCache distinguishes a practical caching alternative from Continuum's combined
scheduler/cache system. Where possible run Continuum on/off on its own runtime with
identical LMCache settings to attribute its scheduling effect. Main-runtime LMCache
and preview-runtime LMCache are not interchangeable controls. Pin publication of
decode KV, chunk geometry, memory budgets and default versus tuned configurations.
LMCache is a replacement connector here, not currently a drop-in wait-hint extension:
our advice mutates native ARC state. Combining them requires a separate adapter/design.
Do not create that implementation merely to populate a benchmark table.

GPU retention and CPU advice may complement each other. On ONE qualified common
runtime and identical guarded CPU ARC, run: neither, GPU retention only, CPU advice
only, both. This measures whether CPU advice still helps after GPU retention and
whether combining them harms other traffic. Include GPU-resident, GPU-spill and CPU-
competition conditions. Count GPU survival, CPU restores and recomputation separately.
This is a compatibility hypothesis, not an available supported configuration. Current
runtime pins/guards must be requalified; a cross-version comparison cannot substitute
for this four-way attribution. Use a common causal signal policy, with no future-return
knowledge; declare that completion-time retention and later wait updates have different
interfaces, and separate native best-use comparisons from matched-information tests.

Quick compare (also called validation in discussion) includes the qualified external
shortlist on a compact common subset, not only ARC versus wait-hint. Core three arms
cover every selected workload family; LMCache and Continuum cover common no-benefit,
pressure, real-agent and interference cases. The four-way combination is included on
its relevant subset once qualified. Explicit case filters support faster diagnostics,
but a filtered run is not the whole compare profile. Expanded arms invalidate the
previous informal 30–60 minute target until pilots measure startup and case costs.
Unsupported configurations remain visible. Competitive claims cover only jointly
exercised models/cases; external arms need not multiply every correctness/fault test.

### 4. Coding-first model and deployment proposal

Philip requires the 11 GiB RTX 2080 Ti as the primary target; selected runs may use
an RTX 5080 with 16 GiB. Larger GPUs are not assumed. The previous two small 2025
full-attention models were rejected: selection must represent current local coding
and materially different cache architectures, not merely convenient existing support.

Recommend Qwen/Qwen3.5-9B and google/gemma-4-12B-it, text-only, with qualified 4-bit
weight-only quantization and FP16 activations/cache on the 2080 Ti. These are 2026
models from widely used families. Exact quantized artifacts remain admission candidates,
not approved runtime inputs. Qwen's hybrid DeltaNet state and Gemma's local/global
attention require real product support work; neither currently qualifies through this
connector. Model size alone must not substitute for architecture coverage.

| Model | Architectural distinction | Benchmark implication |
| --- | --- | --- |
| Qwen3.5-9B, W4A16 | 24 linear-attention layers and eight full-attention layers | Resume requires compatible recurrent state/checkpoints plus full-attention KV; preserving ordinary KV alone may be insufficient |
| Gemma 4 12B IT, W4A16 | 40 sliding-window layers and eight full-attention layers, 1,024-token window; different local/global head geometry and shared K/V in global attention | Restore must respect window lifetime and cache-group alignment; global-prefix retention can matter even when local state is bounded |

These test two genuinely distinct cache-lifecycle problems. Keep the existing tiny
full-attention fixture for functional regression only, not a third performance model.
Two architectures are a useful first supported scope, not proof of universal support;
MoE, MLA, distributed and multimodal claims would require corresponding evidence.

Artifact shortlist: cyankiwi/Qwen3.5-9B-AWQ-4bit and
axteam/gemma-4-12B-it-AWQ-4bit. Inspect actual quantization config, ignored tensors,
calibration provenance, tensor bytes and loader/kernel support before selecting pins.
Hugging Face file metadata inspected 2026-11-11 gives 9,068,588,904 tensor bytes
(8.45 GiB) for Qwen revision e4d2d39cede8c5d1716d824cd843e2889c648083 and
8,421,320,792 bytes (7.84 GiB) for Gemma revision
7c781be42ce8d29d0975b857f6d0c0b2558d9b63. These are file payloads, not measured
loaded peaks; both leave tight headroom on 11 GiB. The Gemma publisher reports only B200 execution,
so its Turing compatibility statement is not a measured 2080 Ti result. FP16 casting
must include the execution path, not just changing a metadata field. Neither an AWQ
label nor total parameter count establishes peak memory. Some other Gemma INT4 builds
are over 11 GB on disk and are unsuitable for the intended memory budget.

Weight quantization and cache precision are separate. Current connector code requires
FP16 KV, not an explicit ban on quantized weights. Weight-only INT4 can retain exact
FP16 KV semantics after qualification. Hybrid/group support is the architectural
blocker; it must not be bypassed by removing guards. Required review covers group
identity, recurrent checkpoint readiness, window-aware retention, exact restore,
publication/failure handling and end-to-end reuse eligibility. Start with a CPU/runtime
feasibility audit and explicit implementation proposal; this planning task does not
authorize the architecture expansion or a runtime repin.

For 5080, propose a small same-model W4A16 versus W8A16 Qwen sensitivity panel (eight
coding tasks, core three systems, two repetitions per precision). First qualify the
INT8 format/kernel and 16 GiB fit. Run BOTH precisions on the same 5080; never attribute
2080-vs-5080 differences to quantization. FP8/NVFP4 are alternative hardware-specific
studies, not automatic substitutes on Turing. Preserve FP16 KV initially so weight
precision is isolated; cache quantization is separate product work. Avoid expanding
this into every model x every precision x every workload.

Admit at 8,192 total context tokens and probe 16,384; long coding sessions need an
explicit common context-management policy, not arbitrary truncation. Full memory
accounting includes retained high-precision weights/embeddings, all cache groups,
recurrent states, workspace/activations and concurrent decoding. If a model cannot
fit a useful workload on 11 GiB, return the limitation; do not silently make the 5080
mandatory or substitute an older model. This is a recommended pair pending feasibility,
not a promise that both already run on this hardware.

### 5. Exact coding-first workload proposal

Approximately 80–85% of task identities are coding/coding-adjacent. This is Philip's
product-priority weighting, not a measured claim about all local-model users. Keep
non-coding breadth and harmful/no-opportunity cases, but drop mandatory retail/airline
and generic chat campaigns from the first release. Six workload types share a runner;
real harnesses execute tools and feed actual outputs into later turns.

| ID | Workload, source and check | Validation tasks | Release tasks per repetition |
| --- | --- | ---: | ---: |
| C1 | Bug repair: mini-swe-agent, SWE-bench Verified, real read/edit/test loop, official hidden grading; split by repository and task ancestry | 2 | 16 |
| C2 | Regression-test authoring: real issue/fixed-version pairs from separate SWE-bench source groups; agent writes tests, evaluator checks original fails and fixed version passes; reference patch hidden | 2 | 8 |
| C3 | Repository navigation/change-impact: locate implementation, trace callers, identify affected tests in pinned more-itertools, Click, Requests and pytest repositories; retained symbol/file evidence and runnable targeted tests | 2 | 8 |
| C4 | Terminal development/debugging: pinned Terminal-Bench 2.0 via Harbor; CPU-only programming, build and environment-repair tasks with original validators; no network-dependent or extra-GPU tasks | 1 | 8 |
| C5 | Multi-turn coding revision and branches: four-step investigate/edit/test/revise tasks over real repositories, actual generated patches/test output, two competing branches on half the tasks; executable acceptance checks | 1 | 4 |
| B1 | Non-coding grounded document work: half independent QA/summaries, half shared-source search/read/compare over frozen PEPs/RFCs and admitted LongBench sources; source-supported answers | 2 | 8 |
| Total | Coding/coding-adjacent identities: 8/10 validation, 44/52 release | 10 | 52 |

Counts are per included model/system/repetition, not per arbitrary environment grid.
C2/C3/C5 are explicitly project-authored tasks on real repositories, not official
published benchmark scores. C1/C4 retain actual application harnesses. Natural tools
have natural waits, not inserted sleeps. Select by source, task type, context size and
baseline usability before confirmation; never choose based on wait-hint speedup.
Validation and release groups remain disjoint; exact task IDs, revisions, references,
context policies and grading fixtures must be frozen before acquisition. Counts are
proposed coverage budgets, not proof of the 2pp quality margin.

Validation uses the complete ten-task panel on the current primary model (default
Qwen) for GPU-only/native ARC/wait-hint: 30 task executions. LMCache and Continuum each
run four sentinels: one C1, C3, C5 and B1 task, adding eight executions. The second
model runs one C1 and one C3 sentinel on the core three systems, adding six executions.
Total: 44 task executions, each ONCE, plus three short mechanism cases on the core
three systems (nine case-runs). Those are (a) no-return/unhinted traffic, (b) actual
CPU competition with returning tool sessions and (c) returning sessions competing with
abandoned hints. Case sizes are set to reach real pressure, not merely a fixed tiny
request count. A model-specific change makes that model primary; changing both models'
common cache machinery requires the full validation panel on both, explicitly a larger
run. Missing/incompatible external arms remain listed, not silently dropped as passes.

Release: both models run the 52-task full panel on the core three systems twice:
624 task executions. Each external system runs a frozen 20-task subset per model per
repetition: six C1, three C2, three C3, three C4, two C5 and three B1, adding 160 tasks.
Total: 784 main-panel task executions. External claims cover that subset only. Repeated
source identities are not independent quality examples. Native/guarded ARC identities
must be re-established for hybrid support; existing ARC assumptions cannot be carried
over unexamined. Continuum may not support these architectures on its preview runtime:
if it does not, label it incompatible and reassess the closest functioning competitor,
rather than porting an entire engine just to preserve its historical name.

### 6. Conditions and runtime budget

Run the main live panel at one frozen realistic deployment configuration per model,
with interleaved sessions and admitted concurrency. Do not repeat every long agent
task under every possible load/pressure setting. A targeted six-case mechanism panel
covers: no reuse; GPU-resident reuse; GPU spill/CPU fits; CPU competition; long decode
plus returning tools; and misleading/abandoned hints harming unhinted traffic. These
cases include mixed coding and document traffic. Fixed-input controlled wait sweeps
are diagnostic and separate from actual application timings. Baseline-only fit/load
pilots set working sets, arrivals and exposure to activate each case.

Core release mechanism panel: six cases x three systems x two models x two repetitions
= 72 case-runs. External pressure/no-benefit subset: two cases x two systems x two
models x two repetitions = 16 case-runs. GPU-retention combination study, if separately
qualified: three cases (GPU-resident, spill, CPU competition) x four settings x two
models x two repetitions = 48 case-runs. Capacity/load curves use the mixed coding
case; counts/costs for boundary-search probes must be explicit in the acquisition
manifest, with a planning reserve of four hours and no unlimited search.

One validation block cannot estimate between-run variance. Two release blocks give
repeatability evidence, not precise run-variance or tail guarantees. Report both runs,
within-run task distributions and inconsistencies. Underpowered gates remain unresolved;
no automatic extra repetitions. Repeat the same frozen release task set with independent
starts and changed arm order. Keep cold/warm state, actual arrivals/backlog, complete
failure accounting, output quality, thermal state, tool filesystem resets and driver
headroom visible. Streaming chunks are not individual token timestamps.

Planning ETAs BELOW ARE NOT MEASURED and are conditional on successful architecture,
kernel and model-fit qualification. They assume one GPU used serially, cached weights/
containers, two to four interleaved live sessions where fit allows, aggregate generated
throughput roughly 15–40 tokens/s on the 2080 Ti, ordinary bounded CPU tools and no
repeated model downloads. Reasoning tokens count. No 5080 speedup is assumed.

| Acquisition component | Planning arithmetic / assumption | Wall-clock estimate |
| --- | --- | --- |
| Validation main core | Three ten-task panels, 12–25 min each for deliberately short development tasks | 36–75 min |
| Validation external sentinels | Two four-task panels, 4–10 min each | 8–20 min |
| Validation second-model sentinels | Three two-task panels, 2–5 min each | 6–15 min |
| Validation mechanism + startup/report | Nine short cases plus model loads and checking | 25–50 min |
| Validation total | 44 task executions + nine cases, one repetition | Approximately 1.5–3 hours |
| Release main core | Twelve 52-task panels (3 systems x 2 models x 2 reps), 2–4 hours each | 24–48 hours |
| Release external subset | Eight 20-task panels, 45–90 min each | 6–12 hours |
| Release mechanism cases | 88 cases at an assumed 3–6 min each | 4.4–8.8 hours |
| Release combination study | 48 cases at an assumed 3–6 min each | 2.4–4.8 hours |
| Release sustained exposure | Existing primary-model candidate/control 8+2+2h each, secondary candidate/control 2h each | 28 hours |
| Release fault checks, startup/report, capacity-search reserve | 3–6h fault work + 3–6h loading/report + up to 4h search | 10–16 hours |
| Release total including combination | All rows above, two performance repetitions | Approximately 75–120 hours, or 3–5 days |
| Optional 5080 weight-precision panel | Eight coding tasks x three core systems x two precisions x two repetitions = 96 task executions | Reserve another 4–8 hours, replace with measured estimate |

Panel assumptions correspond to roughly 8–15k generated tokens for the short validation
panel and 70–120k for the full release panel, plus prefill and tool time. They are
forecast inputs, not caps to force favorable timing; actual reasoning/tool loops can
exceed them. Before a campaign, pilot representative short/edit/terminal/long-context
cases and replace estimates with measured token volume, prefill/decode and tool costs.
If estimates exceed budget, return a concrete scope/cost choice; do not truncate tasks
and present the shortened run as complete. Suggested per-session emergency ceilings
are 30 model calls/20min for coding and terminal work, 12 calls/10min for navigation
and document work; reaching a ceiling is an incomplete outcome, not success. Ceilings
must accommodate official harness validity requirements or the claim is a subset/adaptation.

The 5080 is a separately qualified lane. Compare quantization only within that lane,
with the same model and cache settings; compare hardware only with matched precision.
The primary 2080 Ti release must not depend on optional 5080 availability. No paid
services, machine changes, downloads or GPU execution are authorized by this proposal.

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

Reproducibility controls cannot assume a supported batch-invariant mode on this
host. Current vLLM documentation lists NVIDIA compute capability 8.0+ for that mode;
the target 2080 Ti is sm_75. Do not enable an unsupported feature or switch runtime
silently to make output checks pass. Calibrate fresh-versus-fresh variation at the
same batch/schedule shapes; use direct KV-byte checks and bounded deterministic
fixtures to distinguish transport defects from ordinary numerical variation. Live
output divergence still requires measured task-quality assessment. Known corruption
or identity violations never receive a statistical quality tolerance.

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

Check consequential intermediate actions as well as the final response: tool choice,
arguments, claimed observations and dependencies between steps. Qualify checkers with
plausible wrong-but-well-formatted answers, incorrect copied source spans, false test
claims and invalid intermediate actions that happen to lead to a plausible final.
Ground truth must come from retained source/tool evidence independently of the model
output being scored. Live arms receive the same task assignment, not necessarily the
same subsequent work; report divergence and cost rather than claiming bit-identical
computation. Unsampled or unjudgeable output quality remains uncertain.

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

Every injected fault needs a concrete acceptance record, including expected affected
request outcomes, stale-state invalidation, allowed unaffected traffic, owned cleanup,
a predeclared recovery deadline and a post-recovery correctness/load probe:

| Fault | Required postcondition beyond "server restarted" |
| --- | --- |
| Disconnect/cancel | Abort observed, no continued owned work or duplicate completion, unrelated requests progress. |
| Transfer failure | Failed KV is never published/reused; actual native recompute or safe engine stop observed as specified. A fake CPU handler does not qualify native GPU failure recovery. |
| Engine stop/restart | Old clock-domain observations refuse; no stale request/state reuse, cold-cache identity explicit, fresh generation verified. |
| Timeout/failed launch | Owned descendants removed, port/resources released or refusal explained, unrelated processes untouched. |
| Disk/write interruption | Admission/events/terminal boundaries leave either valid committed records or explicit incomplete records; no fabricated terminal success; replayed reporting never double-counts. |

Zero observed failures during finite exposure is not an uptime or MTBF guarantee.
Any numerical reliability bound must state its stochastic assumptions and independence
unit; a constant-rate model is not automatic for correlated resource/leak failures.
Exposure targets remain proposed test coverage, not a statistically proved lifetime.

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

Source diversity is an admission requirement: for every claimed workload family,
list source projects/collections, independent scenarios, generator/template ancestors,
related instances and overlap across all three profiles. A broad multi-workload claim
requires more than one independent source/scenario-construction group per claimed
family; a single-source family remains explicitly scoped. This is a minimum audit
check, not proof that two sources suffice. Shared context within a case is intentional
cache structure; it does not create independent evaluation units. Review distributions
and concrete samples before freezing the matrix. No new workload name, document copy
or seed makes reused templates fresh confirmation.

For the quality margin, calculate precision from paired task discordance, source/session
clustering and independent starts. Declare which quality claims are per cell/cohort
and which are grouped, with a rationale and separate checks for heterogeneous harm.
Allocate judgement effort and GPU cost jointly. Report required versus affordable
independent task counts and gate power in the planning receipt; do not approve a
numeric margin based only on a count of generated answers. Multiple invocations of
the same task add runtime exposure but not independent task diversity.

Choose primary outcomes and claimed regions before confirmation. For each relevant
cell, report paired whole-run effects and uncertainty, with independent run blocks
and grouped source resampling where appropriate. Requests within a session/process
are correlated. Predeclare multiplicity handling for multiple confirmatory claims
(e.g. simultaneous intervals or Holm adjustment); exploratory heatmaps remain
exploratory. Separate discovery-selected regions from fresh confirmation estimates.

User-set repetition policy: functional runs once per check; compare/validation runs
ONE matched block per selected case; release runs TWO independently started matched
blocks per confirmation case. Each block runs every assigned arm on matched task IDs
and arrival seeds. Release repeats the task set with reset application/cache state,
reverses arm order for the second block and separates the starts in time. Record
residual order effects; two blocks cannot balance every multi-arm ordering.

Validation reports within-run variation and descriptive comparisons only; one block
cannot estimate between-run noise. Release reports both effects and their disagreement;
two blocks do not establish a reliable run-variance distribution or narrow 95% bounds.
Do not substitute correlated requests for independent starts. Enough distinct source
sessions can inform task-population uncertainty conditional on these runs, but cannot
remove uncertainty about process/host variation. A small or inconsistent effect remains
unresolved. These repetition limits are Philip's decision, not defaults to expand
automatically. No optional stopping or added runs until a preferred sign appears.
A targeted follow-up requires a separately agreed budget and analysis. Sustained/fault
exposure is separate and is not multiplied by performance repetition count. Quality
precision requirements remain explicit; small proposed task samples cannot establish
two-percentage-point equivalence merely because both arms have similar scores.

Pilot estimates noise, baseline variability and task-success frequency; calculate
sample sizes against useful-effect/non-inferiority margins. Require independent
process starts and sessions, not merely many requests in one run. Freeze repetitions,
analysis and stopping before confirmation, or predeclare a valid sequential procedure.
If the resource cap is reached with an interval crossing a gate, verdict is unresolved.
No rerunning only unfavorable arms/pairs. Every scheduled trial stays in a trial ledger.
A candidate-caused crash, timeout or invalid output remains in its assigned arm's
failure/quality denominators; it cannot vanish because the pair is incomplete.
Invariant violations and prohibited normal-operation failures block their gates;
ordinary task errors enter quality estimates and are judged against frozen quality
requirements rather than automatically failing every release. Complete-pair latency estimates are explicitly
conditional summaries, never the sole release evidence. Report sensitivity bounds
for censored outcomes or withhold the claim when missingness could change it. Replacing
an infrastructure-invalid block requires independent evidence and predeclared rules,
retains the original attempt and replaces the entire paired block; unknown cause is
unresolved, not automatically infrastructure.

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
| No-op overhead | At fixed admitted load, upper bound of p95 latency ratio <= 1.05 with no added normal-request failures; where capacity is actually measured, capacity ratio lower bound >= 0.97 | Limits the cost imposed on traffic unable to benefit. |
| Protected traffic | Upper bound of unhinted p95 TTFT and end-to-end ratios <= 1.05 at fixed admitted load, no reduced completion fraction; capacity ratio lower bound >= 0.97 only in capacity trials | Prevents aggregate wins from hiding displacement harm. Report absolute delays too. |
| Quality | Lower bound of candidate minus control task-success difference >= -0.02 (two percentage points), plus frozen absolute task-usefulness floors; stricter zero-error checks for deterministic assertions | Performance must not be purchased through worse answers. Margins may need tightening for the intended task; reference quality is not presumed. |
| Claimed benefit | For the preselected primary outcome: capacity/valid-work-rate ratio lower bound >= 1.05 in a capacity or fixed-work experiment, OR latency ratio upper bound <= 0.95 at fixed load; protected secondary gates also pass | Requires a useful, repeatable effect rather than statistical detection of a negligible change. The chosen primary metric is frozen in advance. |
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

Each latency gate also declares an absolute-time tolerance and reporting resolution
before confirmation. Near-zero or unresolvably small baselines use a predeclared
absolute rule instead of an unstable ratio; agents may not choose whichever passes.
At overloaded points, finite-latency non-inferiority is not assumed: apply the declared
bounded-refusal, failure accounting and recovery contract. Quality margins apply at
predeclared task/cohort units; group averages cannot excuse systematic failures in a
protected task class. Power the quality and tail gates separately from mean throughput.
If a two-percentage-point bound needs more independent cases than affordable, the
quality conclusion remains unresolved rather than treating no detected difference
as parity. Specify which multiple-testing procedure applies to which family; a joint
all-gates-must-pass non-inferiority claim differs from selecting any winning cell.

Different advice conditions have different obligations. Plausible admitted advice
must satisfy declared protected-traffic gates. Accepted but deliberately misleading
out-of-envelope advice is a mandatory stress characterization with bounded operation
and explicit harm reporting; no blanket promise of unchanged latency is made under
arbitrary cache manipulation. Define the deployable advice envelope prospectively
from observable input/cadence conditions, not from knowing which hints later succeed.
If those conditions cannot bound observed harm, repair/disable/restrict the feature
or withhold the protective claim. The user must approve this scope/value trade-off;
we cannot make the system appear robust by relabelling adverse results after the run.

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
The proposed shortlist and combination study are specified in section 3a; exact
versions, hardware fit and adapters remain admission work. No package is downloaded
or runtime changed by this plan.

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

Profiles must preserve required behaviors, not just labels:

| Profile | Mandatory evidence / forbidden inference |
| --- | --- |
| Functional | Real positive store/evict/restore and causal advice delivery, plus negative identity/refusal checks. An unexercised required path is not a pass. |
| Compare | Each selected case retains sufficient working set, return order and exposure to trigger its intended pressure/queue/interaction; minimum evidence is case-specific. Report uncovered families, unreliable tails and unexercised paths. No release badge. |
| Release | Full frozen gate/coverage ledger, power and qualified fault/recovery evidence. Incomplete mandatory cells cannot inherit a pass from compare. |

Before relying on compare, compare its conclusions with representative full cases and
known regression controls. If shortening loses the mechanism or reverses the ranking,
repair the subset/intensity or label the diagnostic limit. Maintain sentinel cases for
no-benefit overhead, actual CPU competition and interference, with rotating source
instances so development exposure remains visible. Rare failures, long-term growth
and fresh quality claims remain release obligations; fast checks cannot be immune.

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

Section 6 owns acquisition counts, arithmetic and conditional ETA ranges. The earlier
full workload x three environments Cartesian product is removed. Keep the broad live
panel and targeted mechanism coverage distinct; new obligations require explicit cost
rather than silently multiplying all cells. One-time hybrid/window support, quantized
kernel qualification and runner implementation are prerequisites, not included in the
GPU acquisition ETAs and not yet estimated. This design remains under user review.

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
| Small model or one corpus determines everything | Two materially different architectures, coding-first live workflows, source-group holdouts and explicit coverage matrix |
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

Still unresolved before executable release acquisition: hybrid/window architecture support, quantized kernel compatibility and model fit; fresh corpus/source groups and rights; absolute task-quality floors and
latency objectives where relevant; telemetry availability/overhead; statistical power;
external-system compatibility; raw-storage budget; durable ownership/notification;
review of proposed numeric tolerance and exposure targets. These are explicit release
readiness blockers, not items an agent may fill with favorable defaults after results.

### 18. Adversarial review and disposition

Sol's complete-plan review identified eleven material issues; the primary agent's
research also checked open/closed load models, batch-dependent reproducibility and
finite-exposure reliability. The review did not bless the original plan. Revisions
below repair the specification; no immunity is claimed until their execution controls
are implemented and exercised.

| Risk | Disposition in this plan | What remains |
| --- | --- | --- |
| Arrival-limited throughput hides slowdown | Separate fixed-load latency, capacity search and complete-session outcomes | Capacity objective and pilot precision |
| Failed pairs vanish from results | All-assigned trial ledger, failure-inclusive gates, conditional survivor estimates only | Exercise timeout/missingness cases |
| Arbitrary bad advice makes protection impossible | Separate safety obligations, prospective advice envelope and stress harm | Philip's permitted trade-off; measured behavior |
| Comparator/denominator changes meaning | Explicit gate identity and separate overhead controls; absolute rules when ratios unreliable | Philip's margins and pilot resolution |
| Quality gate unaffordable or underpowered | Grouping/discordance/cluster/power worksheet; unknown means unresolved | Real task/judge variability and budget |
| Six labels hide one corpus/template | Source/scenario ancestry admission and holdout audit across profiles | Actual asset inventory/diversity |
| Models fit weights but not workload | Whole-memory worksheet plus context/concurrency/quality pilot | Hardware evidence or scope decision |
| Short profile never reaches eviction/restore | Activity contracts and compare/full sensitivity checks | Real profile qualification |
| Checker rewards plausible wrong output | Independent evidence and adversarial checker fixtures, intermediate checks | Judge reliability/unsampled quality |
| Restart masks stale state/orphaned work | Fault-specific outcome/state/ownership/deadline/postcheck records | Actual native-path recovery trials |
| Grid count is mistaken for complete coverage | Mechanism-selected intersections and explicit remaining holes | New interactions exposed by pilots |
| Determinism unavailable on target hardware | Direct exact KV tests plus calibrated numerical variation; no unsupported batch-invariant assumption | Live quality evidence on pinned runtime |
| Healthy finite soak implies indefinite reliability | Exposure report with no unjustified failure-rate inference | Longer/targeted evidence for stronger claims |
| Driver, thermal or tool state biases arms | Driver headroom, host-state records, tool-state controls and arm-blind exclusions | Measured overhead and residual environmental noise |

The second Sol review found no remaining material blocker to user design review.
It requested clarifying quality-error treatment and the initial interaction-panel
size; both corrections are incorporated. This is not an approved executable release
contract. User choices are limited to the
supported advice/traffic envelope, meaningful absolute/relative tolerances, deployment
scope and acquisition budget. Engineering resolves instrumentation, model fit, corpus
ancestry, sample-size feasibility and recovery tests, then reports their evidence.
No agent may silently choose favorable tolerances or waive an unresolved requirement.

### 19. Sources and evidence used

Current-model/coding-first research (2026-11-11):

- [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) and [configuration](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json): recurrent/full-attention architecture.
- [Gemma 4 12B IT](https://huggingface.co/google/gemma-4-12B-it) and [configuration](https://huggingface.co/google/gemma-4-12B-it/blob/main/config.json): sliding/global attention and distinct group geometry.
- [Gemma releases](https://ai.google.dev/gemma/docs/releases): June 2026 12B release.
- [Qwen quantized candidate](https://huggingface.co/cyankiwi/Qwen3.5-9B-AWQ-4bit) and [Gemma quantized candidate](https://huggingface.co/axteam/gemma-4-12B-it-AWQ-4bit): inspectable artifact candidates, not hardware qualification.
- [vLLM quantization](https://docs.vllm.ai/en/stable/features/quantization/) and [hybrid caching](https://docs.vllm.ai/en/v0.30.0/features/automatic_prefix_caching/): kernel and state compatibility need explicit verification against the selected runtime.
- [Terminal-Bench 2.0 and Harbor](https://www.tbench.ai/news/announcement-2-0): real terminal harness; use a pinned CPU-only development-task subset, not the moving latest suite.
- [SWE-bench Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified): real repair tasks with test-based grading; selected subsets are not official full-suite scores.

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

Additional adversarial-review sources:

- [Open Versus Closed: A Cautionary Tale, NSDI](https://static.usenix.org/events/nsdi06/tech/full_papers/schroeder/schroeder_html/index.html): completion-dependent and independent-arrival models can produce materially different outcomes.
- [vLLM reproducibility](https://docs.vllm.ai/en/stable/usage/reproducibility/) and [batch invariance requirements](https://docs.vllm.ai/en/stable/features/batch_invariance/): scheduling/numerical effects and limits of a determinism remedy on the target GPU.
- [NIST finite-exposure reliability bounds](https://www.itl.nist.gov/div898/handbook/apr/section4/apr451.htm): zero failures still require assumptions and confidence bounds for quantitative lifetime claims.

Additional comparison and workload research (2026-11-11):

- [GPU retention draft PR #59714](https://github.com/vllm-project/vllm/pull/59714): actual old "upstream" target; completion-time GPU priority, open draft when checked through GitHub API.
- [Continuum preview](https://github.com/Hanchenli/vllm-continuum): public implementation omits paper estimation; documented LMCache integration and mini-swe-agent path.
- [LMCache paper and project](https://arxiv.org/abs/2510.09665): practical exact KV storage/reuse alternative, separate from a session scheduler.
- [SGLang HiCache](https://docs.sglang.io/docs/advanced_features/hicache) and [TensorRT-LLM retention](https://developer.nvidia.com/blog/introducing-new-kv-cache-reuse-optimizations-in-nvidia-tensorrt-llm/): broader alternative shortlist, not target-hardware qualification.
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent): actual coding-agent harness candidate.
- [Sierra text harness](https://github.com/sierra-research/tau2-bench) and [simulation API](https://github.com/sierra-research/tau2-bench/blob/main/docs/running_simulations.md): live dependent agent/user/tool interaction; simulator cost and versioning matter.
- [LongBench original task runner](https://github.com/THUDM/LongBench/blob/main/LongBench/pred.py): source/evaluator candidates; source data is not by itself live agent execution.
- [WildChat data card](https://huggingface.co/datasets/allenai/WildChat-1M): authentic conversation shapes; timestamp and ancestry limitations require inspection.

## Next
Philip reviews the current-model architecture scope, coding-first roster and explicit
acquisition budget. Audit hybrid/window offload feasibility and Turing quantization
before implementing the revised suite. No GPU campaign can be scheduled from the
conditional ETAs until supported paths and actual model fit are established.
