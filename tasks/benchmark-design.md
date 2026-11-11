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
model pair and concrete workload counts below remain proposals for review.

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

Before model admission, prepare a memory worksheet from exact configuration and
weight bytes: weights, KV per token/block, maximum simultaneous active/cached tokens,
workspace/activations, allocator headroom and CPU/pinned-memory commitments. For the
supported dense full-attention layout, raw KV payload per token is twice the sum over
layers of KV-head count x head dimension x bytes per element; verify actual allocation
rather than treating that formula as peak memory. Historical AWQ fit is not FP16 fit.
An authorized fit/quality pilot must exercise intended context/concurrency, not just
startup. Infeasible cells remain blocked; preserve the broad design and return scope/
hardware choices to Philip rather than silently shrinking the workload.

The product's scope/session names are not security tenants. Test endpoint auth and
identity binding, but make no multi-tenant confidentiality or remote-clock claim.
Broader claims require corresponding security/clock design and evaluation, not just
more latency trials.

### 4a. Concrete two-model recommendation

Recommend Qwen/Qwen3-4B-Instruct-2507 and HuggingFaceTB/SmolLM3-3B, both served in
FP16, non-thinking mode, TP1/eager/full attention. The source weights are BF16;
explicit FP16 conversion at load is required and must be checked for usable generation.
No weight quantization or silent model replacement. Both are publicly available
Apache-2.0 instruction models from 2025, not the newest 2026 architecture generation.
Hugging Face API checked 2026-11-11 reports approximately 2.58M and 297k downloads,
respectively; these are adoption indicators, not unique users or quality evidence.

| Property | Qwen3-4B-Instruct-2507 | SmolLM3-3B |
| --- | --- | --- |
| Proposed source revision | cdbee75f17c01a7cc42f958dc650907174af0554 | a07cc9a04f16550a088caea529712d1d335b0ac1 |
| Dense full-attention layers | 36 | 36 |
| Query / KV heads, head dimension | 32 / 8, 128 | 16 / 4, 128 |
| Derived raw FP16 KV per token | 144 KiB | 72 KiB |
| Derived raw KV for 8,192 tokens | 1.125 GiB | 0.5625 GiB |
| Source tensor payload, approximate | 7.49 GiB | 5.73 GiB |
| Positional encoding | RoPE | Every fourth layer omits RoPE; still full attention |
| Local pinned vLLM route inspected | Qwen native architecture | Registry maps SmolLM3ForCausalLM to TransformersForCausalLM |

The 2x KV difference changes bytes transferred, number of retained tokens at equal
cache bytes and eviction opportunity; smaller SmolLM weights also leave more potential
GPU-cache capacity. Different widths, training, templates, tokenizers and positional
handling challenge model dependence. Do not claim NoPE alone causes a policy ranking:
this is a two-model robustness comparison, not an isolated architecture ablation.
Compare systems WITHIN each model; task tokens and answer lengths differ across models.
Use both practical equal-hardware deployment results and controlled cache-pressure
settings, clearly separated. Do not force equal token counts by padding real inputs.

Start model admission at an 8,192-token total context cap, then attempt 16,384 for the
release long-context subset if the same serving constraints fit. Neither advertised
64K/256K context implies those lengths fit an 11 GiB GPU. Weight/KV figures exclude
workspace, activations, allocator, model-loading peak and CPU transfers. Registry
presence is not offload qualification: exercise the Transformers route, cache geometry,
FP16 numerics, tool formatting and real tasks. No model/GPU download or launch occurred.

Why these rather than other inspected candidates: Llama-3.2-3B is more established
but older and has eight KV heads; Phi-4-mini also has eight. Ministral-3-3B is newer
and attractive for agents, but its 26 layers/eight KV heads give less KV contrast and
its multimodal packaging adds qualification work (its BF16 card advertises 16GB fit,
not proof of 2080 Ti fit). Qwen3.5 mixes linear and full attention, outside the current
single-full-attention-group product contract. Hybrid, MLA, sliding-window, MoE and
multimodal support require separate architectural qualification, not a token third
model. Two models suffice for a bounded two-family full-attention claim while varying
pressure/load directly; they cannot establish model-universal performance. Add another
model only for an advertised new architecture/size/deployment or unexplained dependence.

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

### 5a. Authentic workloads and coverage admission

The six families describe cache behavior, not six synthetic generators. Retain three
kinds of evidence inside the existing profiles; these are not new benchmark categories:

- Controlled fixtures isolate no reuse, partial hits, eviction, bad hints and worst-case
  interference. Useful for explanation and correctness, insufficient for application claims.
- Trace-derived traffic preserves authentic prompt/session structures and available
  timing. Proposed sources include WildChat conversations and retained agent trajectories.
  Replay is labelled replay. Recorded assistant answers are not live application success;
  future user turns may depend on those recorded answers. Audit timestamps before deriving
  think time: WildChat assistant timestamps denote full-response receipt, not pure tool wait.
- Live application harnesses execute fresh model outputs, real tools and dependent turns.
  Require a coding/repository workflow and a non-coding tool/document workflow for broad
  agentic release claims, plus ordinary independent request traffic. No synthetic padding
  or inserted sleeps in the natural-condition results. Pressure sweeps remain separately
  labelled controlled deployments of these workloads.

Proposed actual assets/harnesses for admission pilots:

| Asset/harness | Use | Constraint |
| --- | --- | --- |
| mini-swe-agent with a frozen SWE-bench task subset | Actual read/edit/test loops in isolated repositories; objective tests and complete trajectories | Task/model/context feasibility is unproved on FP16 2080 Ti. Do not select tasks by wait-hint wins; select using independent task properties and baseline usability. A simpler real repository task set is explicitly narrower, not SWE-bench qualification. |
| Sierra tau-bench text harness, pinned revision/domain | Multi-turn customer-service agent, state-changing tools, final-state checks; initially retail or airline | Simulated users, not production human traffic. Freeze simulator/config and account for its latency/cost separately. Separate simulator resources from the serving GPU; API spend needs authorization. Current tau2-bench repository has evolved, so freeze actual task/evaluator versions rather than a name alone. |
| Grounded document workflow using retained real documents and qualified public QA sources (e.g. LongBench constituent datasets) | Search/read/answer sessions plus independent QA; reference-supported correctness, shared and unrelated documents | Public QA data alone is not an application harness. Any new search loop is our harness, labelled as such; retain original evaluator and expose prompt/context adaptations. |
| WildChat and archived agent trajectories | Distribution checks and replay of natural conversation shapes | Audit ancestry, timing, license and truncation; not a substitute for live quality. Previously examined local trajectories are exposed development data. |

At least two distinct live application domains must support a broad application claim;
exact source counts/task counts follow the quality precision plan. First admission
choices are coding plus customer-service; grounded document workflows are useful
ordinary-application coverage and an explicitly scoped fallback if agent/model fit
fails. A model that fails almost all authentic tasks cannot establish useful agent
throughput. Hardware/model scope may need review rather than replacing live work with
synthetic success fixtures. Natural short tool waits and absence of hint opportunity
are valid outcomes and remain in the evidence.

Choose spread before candidate timing results. For each source/task group record
actual input/output lengths, turns, reusable-prefix fraction, distinct intervening
KV bytes, wait distributions, concurrency, source popularity, branching/abandonment,
and recompute cost relative to transfer. Include negative/misleading opportunity,
no/low opportunity, intermediate opportunity, and high opportunity. Opportunity labels
come from workload and baseline cache behavior, not observed candidate wins. Maintain
a common-application panel and a separately reported niche/stress panel; equal family
counts are not a claim about real-world prevalence. No headline population weighting
without a specified target deployment or defensible source distribution.

Audit coverage against held-out real sources and inspect concrete trajectories, not
just histograms. Pilot ranking disagreements between fixtures, replay and live runs
trigger investigation. Freeze coverage gaps and why each case exists; remove redundant
cases only by predeclared coverage reasoning, never because they show losses. We can
justify a balanced, mechanism-relevant spread, not prove a universally optimal spread
without a target workload distribution and cost objective.

### 5b. Concrete validation and release workload proposal

The following are exact workload definitions and proposed acquisition counts for
review, not already assembled or qualified manifests. Sample IDs, source revisions,
licenses and tool/evaluator pins must be frozen before execution using deterministic,
source-group-stratified selection. Counts are per model, assigned arm and operating
condition, per repetition. Run both selected models in normal validation and release.
Validation uses exposed development groups; release uses disjoint confirmation groups.
The same release tasks repeat in the second block; that does not double source diversity.

| ID / workload | What actually runs | Validation, one block | Release, two blocks |
| --- | --- | --- | --- |
| W1 Independent QA and summaries | Intact SQuAD v1.1 paragraphs with original questions; LongBench multi_news/gov_report documents with source-grounded summaries. No hints, independent sources. | 16 QA + 4 summaries | 128 QA + 32 summaries per block |
| W2 Shared-document questions | SQuAD articles with four original questions over a shared context, interleaved across articles. Same-source reuse without waiting advice. | 4 articles x 4 questions = 16 requests | 32 articles x 4 questions = 128 requests per block |
| W3 Natural conversation replay | WildChat English text-only 4–8-turn conversations; replay actual historical prefixes, generating fresh responses but retaining recorded histories for later prompts. No authored generic follow-ups or hints. | 8 conversations | 64 conversations per block |
| W4 Live coding agent | mini-swe-agent on SWE-bench Verified issues in isolated repository environments; real read/edit/test actions, model-dependent continuation, official test-based grading. | 2 issues | 24 issues per block, stratified over at least four source projects |
| W5 Live customer-service agent | Sierra text harness in English, retail and airline; real simulated-domain API calls and model/user-dependent dialogue, original final-state/policy evaluator. | 2 tasks/domain = 4 tasks | 12 tasks/domain = 24 tasks per block |
| W6 Live document research | Local search/read tools over frozen Python PEPs and IETF RFCs: find relevant documents, extract supported facts, compare requirements, answer with citations. Actual outputs feed later turns. | 4 tasks, 2 per source collection | 24 tasks, 12 per source collection, per block |
| W7 Branch-and-merge document research | W6 harness; shared initial evidence, two parallel model/tool branches, final answer consumes actual branch outputs. One additional controlled variant cancels one branch and records it. | 4 parent tasks | 16 parent tasks per block |
| W8 Mixed applications | Concurrent W1/W6/W4/W5 sessions in fixed 60/20/10/10 start-count proportions; short unhinted traffic competes with real returning agents. Report each cohort separately. | 20 starts: 12/4/2/2 | 100 starts: 60/20/10/10 per block |

These counts prioritize workload breadth and affordable acquisition; they do not
power the earlier 2pp quality gate or precise p99 claims automatically. W8 may reuse
profile tasks from standalone rows but is an additional execution; reused identities
are clustered together in quality analysis. Mixture weights are an authored coverage
choice, not measured market prevalence. Agent tasks run with real outputs and failures
remain in denominators. Frozen initial task difficulty/context selection cannot use
candidate speedup or confirmation successes. Keep oracle patches/reference answers out
of agent-visible input. Confirmation source projects/documents/scenario ancestors must
be distinct from development where the split contract requires it; if counts cannot
be met, flag the gap rather than duplicate rows.

Validation conditions: W1/W3 at light load; W2 once GPU-resident and once with GPU
spill; W4–W7 under moderate controlled competition; W8 under CPU competition with a
hot-set shift. These are different cases, each with ONE execution per arm, not hidden
repetitions. Small live samples are integration/performance diagnostics, not quality
certification. Core GPU-only/ARC/wait arms run all rows. Plain LMCache runs all common
rows; Continuum runs W1/W2/W4/W5/W8; qualified GPU-retention four-way runs W2/W6/W8.
Both models use the same admitted source tasks, with model-native templates.

Release runs all eight rows under the three environments in section 6 for core arms;
LMCache runs the common full panel, Continuum W1/W2/W4/W5/W8, and the four-way study
W2/W6/W8 across GPU-resident, GPU-spill and CPU-competition regimes. Missing model/
external compatibility is an explicit claim limit. The interaction panel adds long
summary decode competing with W6 returns, and W7 cancellation competing with W1.
Capacity search uses W1 and W8, with fixed-load latency retained for other cases.

Natural tasks use no inserted sleeps. Issuing a hint follows a real tool-start event
using only then-available information; no future return timestamps. W3 is replay and
cannot validate fresh conversational quality. Controlled W6/W7 diagnostics separately
sweep 0/1/10/60-second declared waits and abandoned hints to expose the opportunity
range; these never support claims about natural tool durations. Source waits longer
than the hint protocol permits do not receive secretly extended deadlines.

Initial per-request generation caps: QA 128, summaries 512, document/branch responses
512, coding/support actions 1,024 tokens. Live coding has a 30-model-call/20-minute
session ceiling; support 30 agent calls/10 minutes; document/branch workflows 12 calls
per branch or session/10 minutes. These are proposed operational ceilings, not expected
durations or permission to truncate silently. All ceilings apply equally across arms;
hits/timeouts are recorded as incomplete, including harness user-simulator time. Task
inputs must fit both tokenizers' admitted context limits, with full-source eligibility
for QA/summary; no answer-aware cropping as used historically. Live context overflow
fails visibly unless a single explicit common context-management policy was qualified
and frozen beforehand. Actual peak concurrency, cache budgets and arrivals are frozen
from baseline-only fit/capacity pilots; no invented requests/sec before measurement.

Release still includes the separately specified correctness, recovery and sustained
runs. Their exposure is not doubled automatically. Agent environment/tool startup is
reported separately from timed sessions; repeated-arm filesystem and simulator state
are reset. The simulator's exact model/provider remains an execution dependency and
cost decision, not implicit paid-service authorization. No aggregate wall-clock
promise is possible before real tool/agent pilots. A 20-minute coding cap multiplied
by issues, configurations, conditions and models is a real budget constraint.

### 6. Required operating conditions and tractable matrix

Each practical model runs the eight concrete workloads in section 5b, covering the
six behavioral families, under three declared environments:

1. Roomy/light: reusable state fits, little queueing; exposes overhead without a
   retention advantage. Preserve genuinely no-reuse requests too.
2. Competitive/steady: GPU overflow with CPU working-set competition, at a shared
   offered rate below the control's sustained saturation boundary.
3. Busy/changing: higher offered load, bursts or shifts in the hot set, with bounded
   overload and recovery; mixed traffic includes misleading/abandoned advice.

The concrete eight-row roster in section 5b gives 8 x 3 x 2 = 48 matched
workload/model/environment cells,
not 36 requests. Each includes independent source groups and repetitions determined
by the precision plan. These cells alone are insufficient for pressure/load
interactions: add a small interaction panel on each practical model crossing three
capacity regimes (GPU-resident, GPU spill/CPU fits, CPU competition) with three load
regimes (light, near sustainable, bounded overload). Reuse matching core cases when
valid; otherwise initially up to nine additional cells per model. This prevents pressure
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

Measure outcomes according to the traffic model. At a fixed below-capacity arrival
rate, completed requests/second mostly measures offered load, not service capacity.
Use latency, queueing, error/completion fraction and resource cost there. Measure
capacity with a separate bracketed load search under declared latency/quality/error
constraints and a bounded stable-backlog criterion, then confirm selected boundary
points with fresh runs. Report the search bracket and uncertainty. If no application
latency objective exists, publish a capacity/latency curve without calling one point
universally acceptable. Fixed-work makespan is another useful outcome; declare
startup, warmup, tool waits and final drain rather than changing the denominator to
obtain a gain. Do not claim a 5% throughput gain in an arrival-limited case.

Streaming receipts distinguish transport chunks from tokens. If a chunk contains
multiple tokens, do not fabricate their individual delivery timestamps or label
chunk-gap measurements as true token-gap measurements. Record unavailable timing
precision and use only metrics the instrumentation actually observes.

Record host CPU/RAM pressure, background activity, GPU clocks/temperature/power and
client scheduling lag. Warm operating state and randomized order control drift.
Candidate-induced contention or throttling is a result, not an exclusion. External
interference is excluded only by predeclared, arm-blind evidence rules. The driver
must demonstrate headroom at peak offered rate; otherwise the result is load-generator
limited. Reset or identically seed tool filesystem state, tool caches and external
responses; repeated tools must not make the second arm easier. Charge genuine
application contention to both complete workflows.

Each interaction case must name its workload and causal hypothesis, not just occupy
a slot in a grid. Mandatory examples include long decode plus CPU competition and
returning sessions; abandoned advice competing with unhinted returns; shared ancestors
under skewed popularity; and a load burst followed by a change in the hot set. Record
expected opportunity separately from realized activity. Primary outcomes include all
assigned sessions, never only sessions that happened to receive an accepted hint or
restore. A useful treatment may reduce evictions, so do not require identical eviction
counts in both arms. Require evidence of opportunity in the control and actual path
activation where claimed; unexpected inactivity is investigated and labelled.

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

The core matrix is 48 cells x 3 arms x R independent blocks = 144R arm-runs before
previous-release/external references, interaction cells and sustained tests. With the user-set R=2 release count and a
hypothetical three minutes per arm that alone is 14.4 GPU-hours, excluding startup,
warmup, drain and failures. External and combination subsets add their own runs; live
application sessions may take much longer than three minutes. This is a sizing
illustration, not a measured runtime or evidence that two blocks meet every gate. Broad release confidence costs more than a short
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

Still unresolved before executable release acquisition: qualification of the recommended model IDs and fit; fresh corpus/source groups and rights; absolute task-quality floors and
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

Model selection research (2026-11-11):

- [Qwen3-4B-Instruct-2507 card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), [configuration](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/main/config.json) and [tensor index](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/main/model.safetensors.index.json).
- [SmolLM3-3B card](https://huggingface.co/HuggingFaceTB/SmolLM3-3B), [configuration](https://huggingface.co/HuggingFaceTB/SmolLM3-3B/blob/main/config.json) and [tensor index](https://huggingface.co/HuggingFaceTB/SmolLM3-3B/blob/main/model.safetensors.index.json). Native cache shape computed from configurations, not measured allocation.
- [Qwen model metadata](https://huggingface.co/api/models/Qwen/Qwen3-4B-Instruct-2507) and [SmolLM metadata](https://huggingface.co/api/models/HuggingFaceTB/SmolLM3-3B): snapshot adoption indicators and source revisions.
- [Ministral BF16](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16), [Phi-4-mini config](https://huggingface.co/microsoft/Phi-4-mini-instruct/blob/main/config.json), [Qwen3.5 config](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json): alternative-selection constraints.
- [SWE-bench Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified): concrete coding source and test-based grading, proposed subset not an official full benchmark score.


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
Philip reviews the release-first contract, especially the coverage floor, proposed
numeric gates, sustained exposure and explicit deployment boundary. Then resolve
actual model/data selection and pilot resources before implementation/acquisition.
