# Wait-hint validation and release plan

Recorded 2026-11-04. This is the qualification plan, not a declaration that its
gates have passed. The user subsequently authorized staged execution on 2026-11-05, including
routine diagnosis while unavailable. The [qualification protocol](../evidence/wait-qualification-20261105/PROTOCOL.md)
fixes concrete scope, owner-selected margins and finite run budgets. Earlier quick
comparisons remain exploratory and cannot substitute for qualification.

Selected local execution is now complete across all four stages. The
[outcomes and limits](../evidence/wait-qualification-20261105/README.md) support
CPU wait retention on specified cache-pressure workloads, while broader adoption,
useful-answer equality and production reliability remain unqualified. Execution
completion does not replace the original gates below. The protocol filled this
outline's decision slots before each selected evaluation under the user's
autonomous-execution authorization; its numeric margins are owner-selected.

On 2026-11-05, the user agreed to record three next release stages: **5 optimize,
6 run more and wider testing, 7 compile the release**. These stages are pending;
the completed Stage 1–4 evidence stays frozen. The intended release consists of
a GitHub README, posts, deeper documentation and a focused vLLM discussion, not
a formal paper submission. Stronger experiments should make those materials
more useful without requiring universal superiority or production qualification.

Workflow batch time / throughput remains primary. First-token latency, including
returning-request medians and tails, is secondary. GPU/CPU memory is a resource
constraint and comparison control; VRAM savings are not a target benefit claim.


The objective is to determine whether our workflow-aware CPU ARC retention is
worth adopting beyond the local benchmark. Competitive performance and broader
applicability are necessary, but causal attribution and operational usefulness
also need evidence.

The intended outcome is reproducible evidence suitable for public performance
claims and serious serving-system review. Claims must follow results, including
negative results; the purpose is to test the candidate, not produce a favorable
marketing number. Stability and performance are separate acceptance decisions.

## Starting evidence and limits

The [repeated local ARC comparison](../evidence/wait-opportunity-20261104/README.md)
supports a benefit on the tested repository-agent workload. The
[five-system comparison](../evidence/five-way-20261104/README.md) is exploratory:
one pass per system, different runtimes, and different generated responses.
It does not establish general superiority over Continuum or upstream vLLM.

The development path explored SVD and semantic representations, offloading,
prefetch, and multi-request tool waits. The useful result so far is a simpler
mechanism: ongoing wait advice refreshes reusable CPU prefixes within their ARC
queues, including while GPU copies remain present. Hints do not count as real
cache hits or alter ARC's learned queue balance. The measured benefit cannot be
credited to semantic representations, and unsuccessful predecessors do not
become successful retrospectively. Novelty of the specific combination remains
plausible but unestablished; product value and research originality are separate
questions.

## Stage 1 Establish what works and why

Selected local execution is complete; see [Stage1 outcomes](../evidence/wait-qualification-20261105/STAGE1.md)
for live controls, replay and traced reuse, including attribution limits. Establish the narrow
claim that CPU wait retention saves enough recomputation to make the tested
workflow finish faster after paying for hint production and processing.

### Freeze the setup

Use the existing repository workload, model, machine, runtime, cache budgets,
and shared scheduling configuration. Freeze the exact inputs and settings before
running; do not tune between measured cases. Disable semantic scoring, SVD, and
prefetch. The resulting claim is conditional on this shared scheduling setup,
not a comparison against every default vLLM configuration.

### Compare three live conditions

| Condition | Behavior | Purpose |
| --- | --- | --- |
| A ARC without hints | Ordinary ARC with no hint production or processing | Baseline without hint machinery |
| B ARC with inactive hints | Produce and process hints without changing cache ordering | Measure hint overhead |
| C ARC with active hints | Same machinery with CPU recency refresh enabled | Measure the retention change |

C versus A measures the complete addition's net benefit. C versus B isolates
the retention behavior. B versus A measures the machinery's overhead. Existing
local ARC controls already send hints, so they do not by themselves establish
the full C versus A comparison.

Start with three runs per condition, rotating order. Use a fresh serving process
and identical warmup for each case. Evaluate whole-run paired differences, not
individual requests as independent repetitions. This is an initial checkpoint:
if variation obscures the result, report it as inconclusive rather than running
until a favorable result appears. Agree on any extension with the user.

### Diagnose the mechanism with controlled replay

Build a replay adapter for one frozen workload trace, selected before comparing
variants. Enforce identical prompt and continuation token sequences and replay
recorded tool delays; a fixed random seed alone is insufficient. Each next turn
still depends on the previous turn's completion. Keep future tool completion
information unavailable to the hint policy. Label replay and its artificial
token constraints explicitly; replay timing is diagnostic evidence rather than
a replacement for live application performance.

Compare ARC with the full mechanism, then compare the full mechanism with:

- Protection delayed until the GPU copy is gone.
- A single initial recency refresh without repeated refreshes.

These targeted removals test whether early CPU protection and continued refresh
contribute. Record their precise implementation so they do not silently change
unrelated behavior. If a simpler version performs equally well, retain that
finding rather than claiming every component is necessary. These checks do not
establish every possible component's independent effect or interaction.

### Measurements and decision

The primary outcome is time to complete the whole workflow batch. Also retain
returning-request latency including tails, prompt tokens recomputed and restored,
CPU-cache evictions, transfer bytes, hint-processing cost, failures, and answer
checks. Report output differences in live cases. Existing test-status checks
cover a narrow aspect of answer quality, not general task success.

In a small diagnostic run, trace specific blocks to connect advice, changed
eviction order, survival, and subsequent reuse. Keep extra diagnostic tracing
separate from primary timing runs. Hint-update counts alone establish activation,
not useful work saved.

Gate: live C versus A shows a net improvement distinguishable from run variation,
and controlled replay plus cache evidence supports the proposed cause. Set the
minimum useful improvement and acceptable regression criteria before execution.
If only replay improves, application value remains unestablished. If only live
timing improves, causal attribution remains unresolved. The selected local evidence satisfies this narrow gate; it does not establish
competitive superiority or broad applicability.

## Stage 2 Establish why someone would choose it

Selected local execution is complete; see [Stage2 outcomes](../evidence/wait-qualification-20261105/STAGE2.md)
for configured published systems and resource controls. For the workload and
resource budget validated in Stage 1, establish a defensible reason to choose
our implementation over available substitutes. Broader applicability belongs
to Stage 3.

### Fix the workload and choose credible alternatives

The main contenders are plain ARC without hints, our validated wait-hint
implementation, Continuum's published implementation, and a pinned upstream
vLLM retention implementation. Semantic scoring is outside this comparison.
Identify the simplest credible existing retention option as an additional
substitute: an available setting delivering the same value may remove the need
for our implementation.

Retain published implementations as requested. Any future same-runtime ports
are separate experiments and must be labeled adaptations.

### Audit fairness before measuring

- Confirm each retention mechanism activates and receives the workflow
  information its interface needs without access to future outcomes.
- Match model, precision, context limits, task instructions, and CPU/GPU cache
  budgets wherever supported. Record remaining differences, including scheduling,
  CPU eviction policies, transfer paths, and observation overhead.
- Check supported cache-publication settings, including whether generated-token
  KV is saved, and actual chunk sizes/kernel paths. A default is not necessarily
  a fair configuration or an inseparable part of a published algorithm. Resolve
  supported mismatches before timing; distinguish unavoidable version limitations.
- Use a small, separately identified calibration step to select reasonable
  settings, then freeze them. Do not tune alternatives less carefully than ours.
- Resolve material runtime disadvantages or narrow the resulting claim. The
  existing five-way run does not complete this audit: upstream required a
  different host-memory setting and Continuum's public code omits its paper's
  automatic estimator.

The comparison-design debate resolved two avoidable mismatches: use upstream's
built-in CPU ARC and forward the same application continuation priority used by
our local arms, through its native priority scheduler. Preserve Continuum's own
scheduler. Describe these as configured published systems, not untouched defaults
or proven optimal settings. The verified host-pointer compatibility patch must
be explicitly labeled and frozen with its upstream base; it is not published
upstream code. Match working pinned-memory and model-runner paths where verified,
and disclose any remaining mismatch. A supported setting is not a policy port.

### Separate system performance from retention benefit

First compare complete published systems in their supported runtimes to measure
which finishes the workload best. Second, where a clean feature-off control
exists, compare each retention mechanism against its own runtime with that
feature disabled. If a clean control is unavailable, record that limitation.

A faster underlying runtime can win overall despite a weaker retention policy.
Improvement percentages against different baselines do not by themselves rank
the policies. Do not describe a cross-runtime system win as isolated algorithmic
superiority.

### Measure the user-visible trade-off

Start with three rotated measured runs per main contender. Use whole-workflow
completion time as the primary outcome, alongside returning-request latency,
memory use, failures, output length, and task-answer checks. Faster but less
complete answers are not a clean win. Select appropriate answer checks before
measuring; test-status labels alone do not establish general answer quality.

Separately test whether modestly more CPU cache lets plain ARC eliminate our
advantage. Report this as a resource-cost comparison, not an equal-memory result.
Retain the same discipline on inconclusive results as Stage 1; additional runs
require an explicit reason and agreement rather than chasing favorable timings.

Gate: a repeatable, practically useful advantage over credible alternatives under
the stated conditions, or comparable performance with demonstrably lower resource
or integration cost. Universal superiority is unnecessary. Beating plain ARC
while tying an existing, equally easy retention option supports the usefulness
of hints but does not establish a compelling reason to adopt our implementation.

## Stage 3 Repeatability across realistic conditions

Establish whether benefits survive independent workloads, changing resource
availability, and realistic operating conditions. Keep the runtime and model
fixed initially so this stage primarily tests workload and operating range.

### Independent live workloads

Freeze at least two new workload families before evaluating the candidate: a
code-and-test workflow in a different repository, and a document-analysis workflow
with real search, parsing, or data-processing tools. These are starting coverage,
not automatically sufficient evidence of broad applicability. Define the intended
deployment population and explain how chosen workloads represent it. Include
publicly reproducible tasks or distributable traces where practical. Authored
fixed tool pipelines are not production traffic or autonomous agents; label them
accurately and include external workload evidence before making those claims.

Separate development/calibration tasks from held-out evaluation tasks. Freeze
candidate and competitor settings before evaluation. If held-out results drive a
change, retain those results as development evidence and obtain fresh validation.
Record context growth, tool durations, return probability, and concurrency rather
than only workload names. Check task outcomes using predefined references or
executable checks, not merely successful HTTP responses or test-status labels.

Compare against plain ARC and the strongest practical Stage 2 alternative on the
declared use case. Record how that alternative was chosen and retain the other
Stage 2 results; do not quietly remove a contender that becomes strong elsewhere.
Before each new regime, check that the selected alternatives remain credible
using documented capabilities and separate calibration. Include another contender
when its relevant strengths are not represented; do not select rivals from
favorable evaluation outcomes or assume the original workload's winner remains
the strongest everywhere.

### Operating conditions and adverse cases

Choose a bounded matrix before running rather than a full Cartesian product:

- Ample cache and low contention, where advice should provide little benefit.
- Constrained CPU/GPU memory, growing histories, and stronger concurrency.
- Mixed competing traffic, including requests receiving no hints; measure their
  latency and progress so a favored group cannot hide harm to others.
- Short, long, and uneven tool waits; vary actual resource contention where useful.
- Abandoned/cancelled workflows, delayed or duplicate hints, and changed return
  rates. Verify bounded stale-advice effects and correct cache ownership.

Vary one factor at a time for explanation, then include a small number of
predeclared realistic combined-stress cases. Use trace replay or injected delays
only as labeled diagnostics, not substitutes for independent live workloads.
Resource budgets must have a plausible deployment rationale and include points
away from the narrow cache geometry that originally produced a win.

Run repeated batches and a bounded sustained-load test with session churn. Track
memory/bookkeeping growth, hint-processing cost, transfer traffic, latency tails,
errors, and useful task completion. Preserve failures and no-benefit cases.
Closed-loop workflow completion and independently offered load answer different
questions: keep genuine within-workflow dependencies, but use controlled session
arrivals for saturation/latency claims so slow clients do not hide overload.

Gate: repeatable gains on independent workloads within a defined operating range,
with acceptable overhead and regressions outside favorable conditions. Report
where it helps, does nothing, or hurts. If only the original tuned regime wins,
classify the result as narrow. Define pass/fail margins before qualification;
absence of a statistically resolved regression is not proof of harmlessness.

## Stage 4 Compatibility across models and environments

Establish correct integration and retained usefulness across an explicitly
supported matrix. Loading successfully is insufficient. Model compatibility and
performance must both be tested because architectures change cache footprints,
transfer cost, and the opportunity for reuse.

### Select a representative supported matrix

Start with an additional model from a different family/cache configuration, an
additional GPU environment preferably on native Linux, and another supported
vLLM revision. Select cases to cover meaningful differences such as model size,
KV layout, quantization, GPU memory/transfer characteristics, and execution mode.
Additional architecture types require explicit support and tests; do not imply
all models supported by vLLM are supported by this extension.

Keep other variables fixed where feasible; label combined environment changes
when isolation is unavailable. Compare both baseline and candidate within each
environment. Match resource budgets in bytes, report actual cache pressure, and
do not assume equal block counts mean equal memory or opportunity across models.
Use a documented conversion rule rather than tuning each new model until it wins.
Any port or runtime patch is separately scoped, reviewed, and frozen before its
validation. Keep pristine published baselines and label compatibility fixes.
Qualification attaches to explicitly tested model/runtime/hardware/configuration
combinations. Testing each axis separately does not qualify their cross-product.
Add selected combined cells for broader support claims; an exhaustive matrix is
not required if the supported scope stays correspondingly narrow.

### Compatibility and reliability acceptance

For each supported configuration, verify installation, effective settings, real
CPU stores/restores under eviction pressure, and correct cache identities and
prefix boundaries. Use controlled token/byte checks where deterministic, alongside
task-quality checks for live execution; do not require live cross-runtime output
identity as a substitute for correctness. Exercise cancellations, restart,
invalid/stale hints, namespace isolation, and bounded metadata/resource cleanup.
Include a trusted fresh-computation or cache-bypassed reference on controlled
workloads using the same runtime, model and precision: agreement between
candidate and ARC can miss a bug they share in
cache publication or lookup. Declare numerical tolerances where exact equality
is inappropriate rather than accepting unexplained output divergence.

Verify relevant execution/transfer paths explicitly: pinned host memory, runner
version, eager or graph mode, and any claimed quantization. Functional probes
cannot establish performance; retain paired measurements of useful completed
work, latency, memory and overhead on the configurations used in public claims.
Mark each matrix cell supported and qualified, functional only, untested, or
unsupported. No silent fallback that changes the tested product.

Provide a minimal integration example, supported-version list, sane defaults,
diagnostics, and a tested off switch. Distinguish hint activity from work saved.
Have an independent person or isolated reproduction exercise follow the published
instructions without undocumented tuning; an author-run clean install alone is
not independent reproduction. Record and fix integration failures.

Gate: specified models/runtimes/environments satisfy correctness, reliability,
resource-bound, and performance criteria. Stability applies to those supported
configurations, not all environments. A model that runs correctly but gains no
speed may be compatible without supporting a performance claim.
Specify tested load, churn, duration, permitted failure rates, resource-growth
limits, and recovery behavior before reliability qualification. Zero failures in
a short soak does not establish a low long-run failure rate; report exposure and
the uncertainty it actually supports.

## Stage 5 Optimize — pending

Investigate whether a small, bounded panel improves on the current queue-refresh
rule. Earlier mechanism ablations are evidence for the existing implementation,
not a systematic search establishing an optimal rule.

Keep plain no-hint ARC and the frozen current wait-hint implementation as
controls. Select roughly five or six variants before screening, rather than a
large Cartesian search. Candidate dimensions include:

- Which blocks receive advice: the full reusable prefix versus a bounded leading
  portion, with explicit prefix-ordering rules.
- Which ARC queues receive advice: both recent/frequent queues versus one queue.
- Strength of the recency movement: full refresh versus a bounded boost.
- Refresh cadence and coalescing of repeated or shared-prefix updates.
- Pressure-aware activation when retention has an opportunity to avoid eviction.
- Per-session advice budgets to limit displacement by long-waiting sessions.

Preserve cache ownership and correctness. The initial panel should preserve
ARC queue membership, ghost history and adaptive learning; a variant that changes
those semantics must be identified as a different policy, not ordinary advice.
Hints must use information available at the decision, never actual future returns.

Include favorable cache pressure and troublesome tight-document/mixed-traffic
conditions in development. Evaluate throughput, returning and offered first-token
latency, tails, useful work, recomputation, transfer traffic, CPU overhead and
effects on unhinted traffic. Keep request-admission changes separate from queue
rule changes so a scheduling change cannot masquerade as a retention improvement.

Define the panel, finite screening budget and selection criteria before running.
Choose on repeatable trade-offs and simplicity, not the best isolated speedup.
Retain all candidates and losses. The existing rule may remain the winner:
finding no improvement still completes this stage. Freeze the selected candidate
and its configuration for Stage 6; preserve the original evidence and controls.

Deliverable: a documented panel, per-variant results, selection rationale and
frozen candidate. A larger headline gain is not an exit requirement.

## Stage 6 Run more and wider testing — pending

Confirm the frozen Stage 5 choice on fresh cases and broaden the supported
evidence. Declare the matrix, corpus provenance, margins, repetitions, exposure,
precision/stopping rules and finite budget before each campaign. The earlier
Stage 1–4 protocol and results remain separate; do not pool them into a new
confirmation cohort.

Prioritize:

- Additional meaningful model families and cache layouts under actual CPU
  eviction/restoration pressure. Loading or tiny-model correctness checks alone
  do not establish a speed benefit.
- Another GPU and native Linux if accessible. Record unavailable environments
  as untested and continue useful local work; do not silently broaden support.
- Cache capacities, concurrency and short/long/uneven tool waits, including
  neutral all-fit cases and preselected combined stress conditions.
- Inaccurate, delayed, duplicate, expired and abandoned advice; changed return
  rates; mixed hinted/unhinted traffic and fairness.
- Sustained session churn, bounded bookkeeping, cancellation, drain and recovery
  under explicitly reported exposure.
- Stronger task-success and useful-answer checks alongside retained output work.
  Keep fixed-token causal diagnostics distinct from live application performance.

Compare against plain ARC, the frozen pre-optimization wait-hint implementation
and relevant published alternatives with verified activation and matched resource
budgets. Preserve runtime/backend differences and make comparisons within the
same runtime where needed to support a specific mechanism claim. Measure latency
tails and overhead as well as throughput; complete measurements are not automatic
performance, quality or reliability passes.

ARC remains primary. A loss triggers an alarm, diagnosis and repair where useful;
it must not disappear into an aggregate. Retain adverse and inconclusive cases,
narrow claims where appropriate, and continue independent useful work. Only stop
the overall effort if evidence shows the mechanism is genuinely unworthy and
further work would be wasteful, not because of a routine failure or unavailable
environment. This preserves the user's stated stopping criterion.

If confirmation results cause a policy/settings change, retain those cases as
development evidence, return to Stage 5 and obtain fresh confirmation. Do not
tune and validate on the same cases or extend measurements until they look good.

Deliverable: an expanded evidence and support matrix, verified per-case results,
regression/diagnosis report, and defensible claim boundaries for the release.

## Stage 7 Compile the release — pending

Turn the final implementation and evidence into a concrete, reviewable research
release. The scope is a public repository and technical communication, rather
than a formal paper or an assertion of production readiness.

- Prepare reviewed commits with clear scope, preserving unrelated and paused
  work. Keep model weights, environments and large raw logs outside Git.
- Verify the documented installation, minimal integration, runtime requirements,
  diagnostics and off switch against the exact release candidate.
- Write the README, deeper mechanism/evaluation/reproduction documentation and
  post drafts. Explain how the selected rule differs from the original rule and
  relevant existing retention approaches.
- Compile throughput and first-token latency results with uncertainty, output
  work, quality outcomes, adverse cases and explicit support limits. Distinguish
  batch-time reduction from reciprocal throughput gain; claim no universal
  upstream lead, p95 benefit or VRAM saving without supporting evidence.
- Provide durable evidence artifacts or documented retrieval usable from a fresh
  clone, with source/dependency/model pins, commands and verification scripts.
  A link into ignored local output alone is not a public reproduction path.
- Prepare a concise vLLM proposal connecting the CPU-retention mechanism and
  measurements to existing KV-hint work, outlining a small maintainable
  integration and disclosing the remaining runtime compatibility work.

Deliverable: the reviewed release candidate, evidence-backed README and deeper
docs, accessible reproducibility package, post drafts and focused vLLM proposal.
Further optimization is not an indefinite prerequisite for compiling this release.

## Qualification protocol and public claims

Before qualification, freeze a claim-and-experiment manifest with:

- Intended use case, candidate revision, baseline versions/settings, and exact
  wording/scope of each proposed claim.
- Workload provenance, development/held-out split, task-quality scoring, hint
  information available at each point, and all selected matrix cells.
- Primary metric, minimum useful gain, acceptable quality/latency/overhead and
  per-group regression margins, plus functional and reliability requirements.
- Warmup, cache reset, measurement windows, run ordering, seeds, repetitions,
  sample-size rationale, precision targets, stopping rules and invalid-run rules.
- Treatment of timeouts, errors, partial work and answer-quality failures in the
  primary decision. Never claim speedup by timing successful survivors alone;
  include all offered work, success/quality rates and any declared deadline.
- Coverage and weighting rule for the intended workload population: how favorable,
  neutral and adverse regimes determine a broad benefit claim. A positive mean
  across convenient tasks alone cannot establish that claim.

A justified curated suite supports claims about its tasks and declared regimes.
Probabilistic claims about a broader workload population require a defensible
sampling basis, not only chosen aggregate weights.

This plan is not execution-ready for qualification until the user and experiment
owner agree on the exact matrix, metrics, margins, exposure and confirmation
budget. These are required decision slots, not permission to choose favorable
criteria after results. No numerical threshold is implied by the outline.

Three repetitions and the quick four-way pass are development checkpoints, not
automatic public-claim qualification. Use pilot variance and the practical effect
size to choose a bounded confirmation budget in advance. If precision is
insufficient, conclude inconclusive; any extension needs an explicit statistical
plan rather than optional stopping when a favorable result appears. Whole-run
paired units, workload clusters and independent sessions must be treated correctly;
do not count correlated requests as independent experimental repetitions. Tail
latency needs enough observations for the claimed percentile and uncertainty.

Report effect sizes and intervals, every predeclared case, failures, output work,
task quality and resource use. Predeclare aggregate weights and comparisons;
report per-workload/per-environment results as well as aggregates so wins cannot
hide regressions. Treat exploratory comparisons as exploratory and account for
multiple comparisons when making simultaneous superiority claims. Equal live
inputs can produce different outputs; distinguish end-to-end application value
from fixed-work mechanism evidence. A complete-system win is not automatically
retention-policy superiority or research novelty.

Release a compact reproducibility package: machine/driver/OS inventory, exact
model/tokenizer identifiers and sampling parameters, code and dependency pins,
patches, configuration manifests, corpus provenance, commands, raw per-run results,
analysis/verification scripts, and known limitations. Keep large logs outside Git;
provide stable downloadable artifacts or documented retrieval rather than local
output paths alone. Preserve all measurement attempts and document exclusions;
never silently drop failed runs. Keep diagnostic tracing out of performance runs
unless its cost is included consistently and disclosed.

Final deliverables are an evidence-backed claim table, supported-configuration
matrix, regression/limitation report, and reproducible benchmark release. A broad
claim such as "generally beneficial" requires defined population coverage and
acceptable adverse-case behavior; no finite small matrix proves universal benefit.
Prefer specific claims with baseline, conditions, effect and uncertainty. Call the
candidate stable only after the reliability gates pass independently of speed.

## Decision discipline

Independent Astra review accepted this framework after tightening tested-combination
coverage, trusted correctness references, competitor relevance, population scope,
failure-aware performance decisions, and reliability exposure. This is a design
review, not execution evidence or an endorsement by vLLM maintainers. Exact
cases, thresholds, confirmation budget and reliability exposure remain open
decisions before qualification.

Review the evidence with the user at each gate. Narrow the claim, revise the
mechanism, or stop if the evidence fails; do not treat work invested as proof of
value. Avoid declaring product success from a local speedup, novelty from code
differences, or policy superiority from unmatched runtimes. The current five-way
result does not mean Stage 2 has already passed.
