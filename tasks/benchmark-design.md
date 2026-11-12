# Benchmark optimization and release evidence plan

## Goal
Design one release-evidence benchmark that can defend explicit correctness,
performance and operating claims. Derive a functional check and quick development
comparison from it. Discover useful applications from product behavior; do not
optimize to a permanent target workload or conceal losses in an aggregate score.

## Authorized
Philip has now requested implementation and end-to-end verification of this plan,
including the necessary GPU qualification, downloads and compatibility work. Use the
11 GiB 2080 Ti and the agreed one/two repetition policy. Keep validation within
45 minutes and recurring release acquisition within 30 hours; preparation and
qualification are recorded separately. No paid resources, publication, automatic
delegation or optional 5080/combination expansion is authorized by this task.

## Done when
The functional, compare and release workflows are implemented through maintained
entry points, exercised end to end with real admitted inputs, independently verified
by their evidence readers, and documented with actual results and limitations.
Unsupported paths and incomplete measurements cannot count as completion or passes.

## State

Implementation is active; the suite is not complete or qualified. Latest authorization
covers implementation, necessary local setup, and end-to-end GPU verification. No
paid resources or delegation. Compare remains one repetition / 45 minutes; recurring
release remains two repetitions / 30 hours, with setup qualification accounted separately.

Implemented schedule accounting preserves 15 compare applications, 204 release
applications, 52 traces and 6 hours of soak. These are allocations, not executed coverage.
A bounded model probe and receipt reader are being qualified. Qwen3.5 W4A16 generated
on the 2080 Ti and an initial wait-hint probe demonstrated CPU restoration after GPU
reset with matched fresh text. Token-ID checks and complete frozen receipt verification
are being added; this is not broad hybrid correctness qualification.

Gemma's pinned artifact loads with Transformers 5.14.0 in the separate
`wait-hint-benchmark-runtime` environment; the established runtime remains unchanged.
Transformers 5.22.1 is incompatible with this vLLM's legacy Gemma configuration access.
At batch 512 / GPU utilization 0.89, Gemma gets 1.14 GiB KV capacity and nominal 8K fit,
but TRITON_ATTN startup fails: 98,304 bytes shared memory required versus Turing's
65,536-byte limit. FLEX_ATTENTION is under bounded investigation. No model substitution,
smaller context, weight offload or 5080 requirement has been silently introduced.

Native group-aware recency is under implementation and tests. Observations align
full, sliding-window and recurrent checkpoint groups; no partial group may refresh.
Portable acceptance passed 38 tests and 105 rule proofs before latest probe edits.
Initial runtime CPU extension passed 18 tests; new compatibility environment testing
is in progress. Full dense GPU regression remains pending.

Docker and isolated mini-swe-agent 2.4.6 / Harbor 0.25.0 / SWE-bench 5.0.2 dependencies
are installed. Terminal-Bench 2.0 source tasks and SWE-bench Verified are cached outside
Git. Real-task admission, grading, workload definitions, runner lifecycle, reports,
external compatibility and the campaigns remain unfinished. The Continuum public
preview at 316a58794a6ff86b216e579b74fd56ed0c5a911f has neither Qwen3.5 nor Gemma4
native model registration and depends on torch 2.8; it is not an admitted comparator.
LMCache's current documented hybrid path requires separate MP integration qualification.

Attempt records remain under the local external attempts root, including failures;
old receipts are immutable and use the schema captured when executed. New probes
use v2 receipts with model/source hashes and runtime package versions.

The following design remains the intended contract; its historical planning-only
authorization statements are superseded by the implementation authorization above.

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
  regression sentinels allocated in section 6; never silently substitute the moving
  development branch. Broader affected-path work substitutes allocated cases or needs
  a separately reviewed budget; it is not an uncounted second full matrix.
- One qualified relevant external cache/offload alternative for a competitive claim.
  A narrow claim of improvement over native offload does not prove superiority to
  external systems. The report distinguishes those claims explicitly.

For advice attribution, match registration, offered hint events, API processing,
GPU/CPU capacities, scheduler options and observation overhead. Separately measure
end-to-end no-feature versus feature cost so matching integrations does not hide
registration/hint overhead. Namespace/timestamps legitimately differ per run; match
causal event semantics rather than replaying invalid clock identities.

Whole-system comparisons match hardware and explicit resource constraints; disclose
extra CPU memory and total observed usage. Use one frozen, documented sensible configuration per system within the matched
resource envelope. A separately tuned configuration is a separate study, not a
second uncounted default comparison. Do not confuse matched
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
scheduler/cache system. The default reports Continuum as a whole system. A causal scheduling claim requires
a separately allocated Continuum on/off comparison on its own runtime with identical
LMCache settings; it is not included in the default count. Main-runtime LMCache
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
pressure, real-agent and interference cases. The four-way combination is an optional replacement study requiring an explicit
coverage/budget revision, not a mandatory default run. Explicit case filters support faster diagnostics,
but a filtered run is not the whole compare profile. Section 6 owns the current conditional time budget; pilots must measure startup
and case costs before acquisition.
Unsupported configurations remain visible. Competitive claims cover only jointly
exercised models/cases; external arms need not multiply every correctness/fault test.

### 4. Coding-first model and deployment proposal

Philip requires the 11 GiB RTX 2080 Ti as the primary target; selected runs may use
an RTX 5080 with 16 GiB. Larger GPUs are not assumed. The previous two small 2025
full-attention models were rejected: selection must represent current local coding
and materially different cache architectures, not merely convenient existing support.

Recommend Qwen/Qwen3.5-4B and google/gemma-4-12B-it, text-only, with qualified 4-bit
weight-only quantization and FP16 activations/cache on the 2080 Ti. These are 2026
models from widely used families. Exact quantized artifacts remain admission candidates,
not approved runtime inputs. Qwen's hybrid DeltaNet state and Gemma's local/global
attention require real product support work; neither currently qualifies through this
connector. Model size alone must not substitute for architecture coverage.

| Model | Architectural distinction | Benchmark implication |
| --- | --- | --- |
| Qwen3.5-4B, W4A16 | 24 linear-attention layers and eight full-attention layers | Resume requires compatible recurrent state/checkpoints plus full-attention KV; preserving ordinary KV alone may be insufficient |
| Gemma 4 12B IT, W4A16 | 40 sliding-window layers and eight full-attention layers, 1,024-token window; different local/global head geometry and shared K/V in global attention | Restore must respect window lifetime and cache-group alignment; global-prefix retention can matter even when local state is bounded |

These test two genuinely distinct cache-lifecycle problems. Keep the existing tiny
full-attention fixture for functional regression only, not a third performance model.
Two architectures are a useful first supported scope, not proof of universal support;
MoE, MLA, distributed and multimodal claims would require corresponding evidence.

Artifact shortlist: cyankiwi/Qwen3.5-4B-AWQ-4bit and
axteam/gemma-4-12B-it-AWQ-4bit. Inspect actual quantization config, ignored tensors,
calibration provenance, tensor bytes and loader/kernel support before selecting pins.
Hugging Face file metadata inspected 2026-11-11 gives 4,040,461,440 tensor bytes
(3.76 GiB) for Qwen revision d3e78b50c727b1f34802f440ca84b25781625352 and
8,421,320,792 bytes (7.84 GiB) for Gemma revision
7c781be42ce8d29d0975b857f6d0c0b2558d9b63. These are file payloads, not measured
loaded peaks. Qwen now deliberately exercises greater cache/concurrency headroom;
Gemma remains the tighter-memory contrast on 11 GiB. The Gemma publisher reports only B200 execution,
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

For 5080, propose a small same-model W4A16 versus W8A16 Qwen sensitivity panel (two
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

### 5. Bounded coding-first workloads

Philip requests approximately a fourfold reduction in campaign size and switches the
primary model to Qwen3.5-4B for memory headroom. Keep one validation repetition and
two release repetitions. Scope reduction must remove actual execution, not merely
rename time estimates or silently stop long tasks early.

Validation runs THREE short complete workflows on Qwen, each once on all five admitted
systems (GPU-only, native ARC, wait-hint, LMCache, Continuum or a reviewed compatible
replacement): 15 task executions. A model-specific change uses Gemma instead, not both
by default. The workflows are (1) a small real bug investigation/edit/regression-test/
revision loop, (2) repository navigation with actual search/read/test tools, and (3)
grounded document questions. Select bounded development tasks that can complete naturally;
no tiny slice of an unfinished SWE-bench issue is called a successful workflow.

Interleave these same workflows in a fixed schedule with unhinted independent requests,
returning tool sessions and an abandoned-hint cohort. The small request fixtures add
load/activation evidence, not independent task-quality samples. One run per system has
explicit no-benefit, competition and interference phases, rather than launching nine
separate diagnostic cases. Natural workflows and controlled diagnostic phases have separate results. Natural
low opportunity is valid evidence and must not be tuned away. Controlled phases must
show a causal tool-wait hint, GPU displacement, multiple eligible competing CPU
prefixes, a return after competition, and restore/recomputation and unhinted-cohort
accounting. A refresh acknowledgement alone is insufficient: record survival and
saved/repeated work against the matched advice-disabled control, even when the
observed effect is zero or negative. Missing required diagnostic activity is
unexercised, never a pass. Freeze hint cadence, return order and working sets.
Rotate one of the three workflow slots to test authoring, terminal debugging or
branching when the changed mechanism requires it, preserving no-benefit and
interference phases. A second-architecture change requires its targeted functional
and performance cases; a Qwen-only default cannot qualify Gemma.
Validation is advisory, not a complete per-family/model qualification.

Release retains six workload types but only THIRTEEN independent tasks per model and
repetition. Both models get the same source tasks using their own native templates.

| ID | Actual work and evaluation | Release tasks per repetition |
| --- | --- | ---: |
| C1 | mini-swe-agent on SWE-bench Verified issues; actual read/edit/test loops and official hidden grading, spread across source projects | 4 |
| C2 | Regression-test authoring on real issue/fixed-version pairs; generated test must fail before fix and pass after, reference patch hidden | 2 |
| C3 | Repository navigation/change-impact over pinned real projects; trace implementation/callers and verify affected tests | 2 |
| C4 | Harbor with a pinned CPU-only Terminal-Bench 2.0 programming/build/debugging subset and original validators | 2 |
| C5 | Multi-turn coding revision with competing branches, real patches and test results, executable final acceptance | 1 |
| B1 | One independent grounded QA/summary task and one shared-source search/read/compare task | 2 |
| Total | Eleven coding/coding-adjacent tasks and two breadth tasks | 13 |

C5 must fork a shared transcript into distinct continuations with branch-specific
tool results, revisit the earlier branch after competition, and exercise the frozen
context-compaction policy. Check branch isolation and final executable acceptance;
one scenario establishes a case result, not broad branching coverage.

C2/C3/C5 and the short validation workflows are project-authored definitions on real
repositories, not official benchmark scores. Natural tools have natural waits. Choose
source, task type and baseline usability before confirmation; do not select on candidate
wins. Development and confirmation source groups remain disjoint. Freeze actual task
IDs, source revisions, context policy, validators and limits before acquisition.

Core release: 13 tasks x three systems x two models x two repetitions = 156 task
executions. Each external system gets six tasks (one from each row) per model per
repetition: 6 x two external systems x two models x two repetitions = 48 executions.
Total: 204 application executions. Competitive claims cover those six shared tasks,
not all 13. Freeze the same six task IDs for every compared system. Assign at least
one low-opportunity task, one actual CPU-pressure/return task and one mixed
hinted/unhinted interference schedule within that subset, identically across arms.
Report natural and controlled conditions separately; no external claim extends to
an unexercised condition. Missing external/model compatibility is visible; preserving an old preview's
name does not justify porting its entire engine. Native/guarded ARC semantics must be
re-established for hybrid support. The four-task C1 subset is workload coverage, not
an accurate estimate of general SWE-bench success rate.

### 6. Conditions and time budgets

The main live panel uses one frozen realistic deployment per model. Three compact
controlled traces cover six conditions through separately accounted phases:

1. No reuse and GPU-resident reuse: overhead without a CPU retention advantage.
2. GPU spill with CPU fit, followed by CPU competition: actual restoration and eviction.
3. Long decode with returning tools, then abandoned hints plus unhinted traffic: interference.

Controlled traces reset or explicitly account for phase carryover. They supplement
natural application runs and do not establish natural wait-time prevalence. Baseline-only
fit/load pilots set working sets and arrivals; count actual activation rather than
assuming a short trace reached pressure. Core release: three traces x three systems x
two models x two repetitions = 36 trace-runs. External activation/pressure evidence is
collected during its six-task subset rather than another independent matrix.

The remaining 16 short run slots are mandatory controls, not an unavailable GPU
retention study. Execution IDs and their allocations are:

| IDs | Configuration and purpose | Runs |
| --- | --- | ---: |
| T1–T3 | Three phased traces, core three systems, both models, two blocks | 36 |
| O1–O2 | Qwen guarded ARC with no registration/hints, and registration-only; no-reuse/resident/pressure phases, two blocks each; matched enabled and advice-disabled references come from T1–T3 | 4 |
| R1 | Frozen previous release on Qwen and Gemma, one mixed pressure/interference trace per model, two blocks; current references come from T1–T3 | 4 |
| D1 | Small dense full-attention sentinel, candidate and frozen previous release (or guarded ARC on the initial release), two blocks | 4 |
| A1 | Extra identical guarded-ARC Qwen start per block paired with T1–T3 for A/A drift/noise diagnostics | 2 |
| S1 | Qwen guarded-ARC trace with a known measured-path delay, one per block, compared with T1–T3 | 2 |
| Total | Short traces/control runs; no additional full model matrix | 52 |

If no architecture-compatible previous release exists, R1 uses the frozen admitted
base implementation and is labelled causal baseline evidence, not previous-release
regression evidence. D1 preserves the existing dense path: pin a supported small
dense model, context and cache budgets before acquisition. It must exercise both
no-benefit overhead and CPU competition/restore; it does not establish broad dense
model performance. Compare identical runtime/configuration wherever attributing a
source change. Runtime migration comparisons are labelled whole-stack comparisons.

Within T1–T3, declare working-set bytes relative to usable GPU and CPU cache capacity,
not just prompt lengths. Include below-capacity GPU reuse, GPU spill/CPU fit, and
CPU oversubscription. Independently declare low-load and near-limit arrival phases,
plus bounded overload/drain/recovery in T3. Freeze arrivals from baseline-only probes;
account for scheduled arrivals, queued/finished/failed work and hinted/unhinted cohorts.
Near-limit means the baseline meets the frozen completion/queue criterion; overload
means offered work exceeds its sustainable completion rate and tests bounded recovery.
The at-most-30-minute baseline probe allowance sets these points, not a capacity curve.
Initial claims concern fixed-load latency or fixed-work completion; maximum-capacity
claims are excluded unless a separately specified capacity study replaces other work.
If phase activation or drain cannot fit the trace allowance, the pilot must reallocate
within the overall cap or return incomplete coverage, never silently shorten the test.

Initial harness qualification (checker corruption, inactive path, missing/duplicate
records, interruption and accounting fixtures in section 10) is a prerequisite, not
recurring application acquisition. Record its own cost and receipts; it is not free
or included in the 18–28-hour recurring forecast. Requalify changed instruments.
Recurring A/A and delay sensitivity are explicitly counted above. Native correctness,
architecture/fault probes and model startup are charged to the 2–3.5-hour allowance;
its itemized execution manifest must fit before a release run can be admitted.

User-set counts remain ONE validation and TWO release repetitions. No automatic
additional runs, exhaustive model/precision grid, or full-suite runs on both GPUs.
Validation cannot estimate between-run variance. Release reports both effects and
disagreement; two starts cannot support narrow variance/tail guarantees. Task and run
uncertainty stay distinct. All failures and incomplete attempts remain visible.

Time targets are planning envelopes, not measured forecasts: validation 20–40 minutes,
release 18–28 hours on one 2080 Ti after compatibility qualification. A proposed hard
acquisition ceiling is 45 minutes for validation and 30 hours for release; freeze the
actual limit with the run manifest. Hitting it means incomplete evidence, never a pass.
Do not hide inability to fit required work by changing quality gates after results.

| Component | Assumed execution cost, to replace with pilot measurements | Planning time |
| --- | --- | --- |
| Validation | Five system runs, 3–6 minutes of short interleaved workflows each, plus 5–10 minutes total startup/report | 20–40 minutes |
| Release core applications | Twelve 13-task panels, 30–60 minutes each | 6–12 hours |
| Release external subset | Eight six-task panels, 15–30 minutes each | 2–4 hours |
| Release controlled traces | 36 runs at 2–3 minutes each | 1.2–1.8 hours |
| Release controls and dense sentinel | 16 allocated runs at 2–3 minutes each; model switches charged to startup allowance | 0.5–0.8 hours |
| Release sustained exposure | Primary candidate/control two hours each; secondary candidate/control one hour each | 6 hours |
| Fault checks, startup/report, bounded load probes | Combined planning allowance including at most 30 minutes of load search | 2–3.5 hours |
| Release total | 204 application executions, 52 trace-runs, sustained/fault checks | Approximately 18–28 hours |
| Optional 5080 precision check | Two coding tasks x three systems x two precisions x two repetitions = 24 task executions | Separately budgeted 1–2 hours |

These forecasts assume cached assets, serial use of one GPU, short validation tasks,
and bounded actual tool execution. The application panels should generate approximately
2–4k tokens per validation system run and 20–35k per core release panel; at an assumed
15–40 aggregate tokens/s, decoding alone can take roughly 1–4.5 minutes and 8–39 minutes,
respectively. Prefill, tool waits and startup add time; model reasoning may exceed the
forecast. No performance claim or measured speedup is implied. Qualification pilots
must establish those costs and actual cache activity before scheduling the full suite.
If whole valid workflows do not fit, return a scope/cost decision rather than secretly
multiplying runs or declaring timed-out tasks completed. The 5080 is optional and
has no assumed speed multiplier. Compare precisions on the same GPU and model.

The reduced budget deliberately buys less statistical precision and shorter stability
exposure. It does not support a 2pp quality-equivalence claim from 13 tasks, reliable
p99 guarantees, general coding-benchmark scores or indefinite reliability. Keep the
correctness/failure gates; qualify performance and quality statements to the actual
observations and samples. Large quality/soak studies are targeted follow-ups only when
needed for a stronger claim, not hidden prerequisites to every default release run.
No GPU use, runtime repin, architecture implementation or paid resources are authorized
by this design revision.

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

Architecture-specific functional/fault cases must establish recurrent checkpoint/token
alignment for Qwen, local-window boundary crossing for Gemma, all-required-group
readiness, partial-group transfer failure and fresh-versus-restored continuations.
The manifest specifies which tensors/checkpoints are transported byte-exactly and
which states are intentionally recomputed before publication. Generic FP16 KV checks
cannot substitute for these invariants. Retain dense functional qualification alongside
D1. Architecture expansion/runtime repins remain separately authorized product work.

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

Require baseline and candidate absolute usefulness plus paired regression review. Two
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

Proposed first-release exposure: one two-hour mixed-load run for each of candidate
and matched control on the primary model, plus one hour for each on the second model
(six hours total). These are sustained checks, not extra performance repetitions.
Also require three full
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

At initial harness qualification and after relevant instrument changes, exercise the
following controls. Recurring A/A and delay checks are allocated in section 6; other
qualification receipts are reused only when their executed paths remain unchanged:

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

Report paired task discordance, source/session grouping and both independent starts.
Declare which observed quality conclusions are per task/cohort; report heterogeneous
harm separately. Multiple invocations add runtime exposure, not independent task
diversity. A stronger population quality-margin claim requires its own precision,
clustering and cost analysis and is excluded from the default release contract.

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

Pilot estimates measurement resolution, baseline variability and task usability.
Check whether the proposed case-scoped useful effects can be resolved within budget. Require independent
process starts and sessions, not merely many requests in one run. Freeze repetitions,
analysis and stopping before confirmation.
If the resource cap is reached with mandatory evidence missing or ambiguous, verdict
is unresolved; do not add repetitions or imply significance from two starts.
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

Default release claims are case-scoped observed outcomes, not population non-inferiority
or tail guarantees. Freeze useful effects, absolute task-usefulness floors and tolerances
before results. The initial 5% performance tolerances below are proposed engineering
choices subject to pilot resolution and user review, not statistical guarantees.
Report both independent blocks and source-level outcomes; an effect smaller than
instrument resolution, inconsistent across blocks or confounded remains unresolved.
Stronger confidence-bound claims require a separately powered protocol and budget;
no default gate secretly requires a larger campaign.

| Gate | Proposed requirement | Reason |
| --- | --- | --- |
| Correctness | Zero observed cache corruption, identity violation, unsafe reuse or protocol invariant failure in required tests/runs; all mandatory functional cases pass | Exact cache behavior is the product contract; finite tests do not prove zero future failures. |
| Normal operation | No unexplained crash, hang or failed normal request in admitted below-overload cases; intentional fault cases behave as specified | Throughput cannot compensate for instability. Unrelated infra faults invalidate affected evidence rather than count as a product pass. |
| Coverage | Every mandatory model/workload/condition cell complete and valid, with required activity and quality evidence; no silent fallback | Prevents scope shrinking or inactive-feature wins. |
| No-op overhead | At fixed admitted load, observed per-block task/session latency ratio <= 1.05 under the frozen absolute tolerance, with no added normal-request failures; descriptive tails also reported, no p95 guarantee | Limits the cost imposed on traffic unable to benefit. |
| Protected traffic | Observed per-block unhinted cohort TTFT and end-to-end ratios <= 1.05 under the frozen absolute tolerance at fixed admitted load, no reduced completion fraction; investigate individual outliers, no tail or capacity guarantee | Prevents aggregate wins from hiding displacement harm. Report absolute delays too. |
| Quality | All deterministic correctness assertions pass; meet frozen absolute task-usefulness floors; report paired task outcomes and investigate every candidate/control regression. Unexplained candidate-associated regressions block the affected claim. Population quality parity is unestablished | Performance must not be purchased through worse answers. Margins may need tightening for the intended task; reference quality is not presumed. |
| Claimed benefit | For the preselected primary outcome, observed valid-work-rate ratio >= 1.05 in a fixed-work experiment OR latency ratio <= 0.95 at fixed load in both blocks, exceeding measured instrument resolution; protected secondary gates pass. Claim only the measured cases, not a population lower bound | Requires a useful, repeatable effect rather than statistical detection of a negligible change. The chosen primary metric is frozen in advance. |
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
bounded-refusal, failure accounting and recovery contract. Task-usefulness floors apply at predeclared task/cohort units; group averages cannot
excuse systematic failures in a protected class. Use paired outcomes to identify
regressions, not a small-sample test to declare parity. A model failing the absolute
usefulness floor does not qualify useful application acceleration, even if it is fast.
Task failures arising in both arms remain visible; cache invariants are strict, whereas
model capability is evaluated against the frozen task rubric. Attribute or resolve
candidate-associated regressions before publishing the affected performance claim.

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

Report separately: isolated advice effect and whole-system comparison at matched
resources with documented sensible configurations. Separately tuned-system comparisons
are optional studies, not additional mandatory default arms. Use actual output/task quality;
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
  fewer repetitions and shorter lifecycle exposure. Covers the default development shapes, no-benefit and interference; change-directed
  slot substitution covers affected additional shapes. It is not every release family. Advisory.
- Release: the frozen full contract above, qualified instrumentation, fresh confirmation
  inputs, required repetitions, lifecycle and recovery coverage. Gate ledger mandatory.

Profiles must preserve required behaviors, not just labels:

| Profile | Mandatory evidence / forbidden inference |
| --- | --- |
| Functional | Real positive store/evict/restore and causal advice delivery, plus negative identity/refusal checks. An unexercised required path is not a pass. |
| Compare | Each selected case retains sufficient working set, return order and exposure to trigger its intended pressure/queue/interaction; minimum evidence is case-specific. Report uncovered families, unreliable tails and unexercised paths. No release badge. |
| Release | Full frozen gate/coverage ledger, case-scoped claims and qualified fault/recovery evidence. Incomplete mandatory cells cannot inherit a pass from compare. |

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
| Fast because it returns shorter/worse answers | Actual output lengths/termination, task validity, absolute usefulness and paired regression review |
| Wins only with privileged hints | Real tool/application signals and costs; ideal hints labelled diagnostics only |
| Hides harm to unrelated work | Mandatory per-cohort latency/throughput gates and mixed/adverse-advice cases |
| Cache state/load generator manufactures gains | Cold/warm canaries, scheduled-arrival accounting, A/A and sensitivity controls |
| Small model or one corpus determines everything | Two materially different architectures, coding-first live workflows, source-group holdouts and explicit coverage matrix |
| Pressure/load interactions missed | Explicit capacity-ratio and low/near-limit/overload phases, decode competition; no default maximum-capacity claim |
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
latency objectives where relevant; telemetry availability/overhead and measurement resolution;
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
| Quality gate unaffordable or underpowered | Case-scoped task acceptance and paired regressions; no default population parity claim | Frozen usefulness floors and judge reliability |
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

The third Sol review challenged the reduced campaign, finding quality-gate contradiction,
uncounted controls, unspecified load coverage, validation activation gaps, architecture/
branching omissions and lost dense performance coverage. This revision removes the
unaffordable statistical parity/tail guarantees, assigns the 16 control slots, specifies
load/activity contracts, adds branch/group-state checks and retains D1. No extra full
matrix or repetitions are added. The review supports conditional, narrowly scoped
release evidence; it does not certify this revised manifest or any runtime result.
The remaining product decisions are architecture authorization and prospective advice/
harm tolerances. Exact task IDs, dense artifact, startup/fault costs and fit are bounded
admission work, not permission to expand the campaign or waive coverage after results.

### 19. Sources and evidence used

Current-model/coding-first research (2026-11-11):

- [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) and [configuration](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json): recurrent/full-attention architecture.
- [Gemma 4 12B IT](https://huggingface.co/google/gemma-4-12B-it) and [configuration](https://huggingface.co/google/gemma-4-12B-it/blob/main/config.json): sliding/global attention and distinct group geometry.
- [Gemma releases](https://ai.google.dev/gemma/docs/releases): June 2026 12B release.
- [Qwen quantized candidate](https://huggingface.co/cyankiwi/Qwen3.5-4B-AWQ-4bit) and [Gemma quantized candidate](https://huggingface.co/axteam/gemma-4-12B-it-AWQ-4bit): inspectable artifact candidates, not hardware qualification.
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
Philip reviews the consolidated bounded contract, architecture implementation scope,
prospective advice/harm tolerances and explicit acquisition budget. Audit hybrid/window offload feasibility and Turing quantization
before implementing the revised suite. No GPU campaign can be scheduled from the
conditional ETAs until supported paths and actual model fit are established.
