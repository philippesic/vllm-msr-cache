# Benchmark design and implementation proposal

## Goal
Build a maintained benchmark suite that characterizes evolving candidates, explains
observed effects, and tests whether discovered opportunities produce useful outcomes
against relevant alternatives. Useful applications emerge from the product; no
application or aggregate score is a permanent optimization target.

## Authorized
Philip requested this comprehensive plan for review, following the agreed direction
for controlled characterization. Design and repository inspection are authorized.
The specific implementation contract below is proposed, not implemented or approved
for execution. No GPU runs, downloads, runtime repins, external publication or agent
operations implementation are authorized by this document.

## Done when
Philip has reviewed the benchmark inventory, comparison rules, implementation wiring,
agent guidance, acquisition scope and remaining choices. Approved implementation work
will have its own explicit authorization and completion criteria. A plan is not
qualification; a runner is not qualified until its real execution path is exercised.

## State
Owner: primary agent, main branch, /home/philip/vllm-semantic-cache-ext-next.
Status: complete proposal for review. No benchmark implementation or acquisition.
Existing portable tests, pinned runtime CPU tests and narrow GPU functional evidence
remain the correctness foundation. There is currently no maintained performance
benchmark command in this repository. Sol reviewed the five characterization
families; the rest of this document consolidates the broader proposed suite and
repository integration, and has not received a separate Sol review.

### 1. Decisions and scope

The suite supports three decisions: where a candidate helps or harms; whether to
replace an incumbent for a stated operating region; and whether the resulting system
is useful compared with alternatives. Characterization, explanation and practical
confirmation share measurement infrastructure but permit different claims.

Definitions:

- Candidate: the configuration or implementation being investigated.
- Incumbent: a previously selected implementation for an explicit scope, frozen by
  source/runtime/configuration identity. It is not automatically the current branch.
- Baseline: a fixed reference system; multiple baselines answer different questions.
- Qualified: the specified behavior was exercised on the named configuration.
- Supported: the maintained implementation accepts the configuration; this alone
  does not establish performance, answer quality or hardware qualification.
- Inconclusive: valid evidence does not resolve the question. It is not equivalence,
  rejection, inactivity or execution failure.

Start characterization on the qualified RTX 2080 Ti, dense FP16, TP1, eager,
TRITON_ATTN configuration with the small qualified model. This establishes the
instrument and a scoped behavior map. It is insufficient for broad product value.
A larger model and another model family enter through B11 before broader claims;
quantization, graphs, different layouts or hardware require explicit qualification.

### 2. Common comparison contract

Three initial systems:

- G: native GPU prefix caching, no CPU-offload connector.
- C: current guarded CPU-offload connector, advice disabled.
- H: the same connector and guards, advice enabled.

C versus H isolates advice when capacities, registration, intended hint traffic,
scheduler options, runtime, input schedule and instrumentation match. G versus C/H
compares whole systems; it cannot isolate advice. Initially match explicit GPU KV
capacity and disclose CPU capacity and total measured resource use. Later B10 also
compares systems under equal total resource constraints. GPU capacity configuration
is not total device memory. Historical unguarded ARC is not a substitute for C.

An incumbent adds another frozen arm only when needed for the development decision.
Changing any arm's tuning after viewing results creates a new exploratory comparison.
Preserve stable baseline definitions and their original results across upgrades.

Record exact model/tokenizer bytes, template, precision, layout, block geometry,
runtime and executed source, including dirty inputs. Keep seeds, request order,
decoding settings and intended arrivals fixed within pairs. Alternate or randomize
arm order across independent blocks. Never mutate executed inputs during a block.

Separate runtime/kernel warmup from workload-cache warmup. Clean-cache cases verify
that measured prefixes are absent; warm cases use the same declared prefix-seeding
sequence. Do not let a previous run accidentally warm the next one. Startup, model
loading and warmup are reported separately from the measurement window.

The pinned ARC fallback limitation may affect C and H differently. Record eviction
and store refusals and describe this limitation with affected results. Fixing it is a
separate runtime change, never an unrecorded benchmark adjustment.

### 3. Benchmark inventory

Every benchmark below specifies a question, execution, comparisons, outputs and
claim boundary. These are experiment definitions sharing one runner, not independent
scripts. Small screening subsets and deeper confirmation use the same definitions.

#### B00: Serving and measurement smoke

- Question: does this exact setup execute the intended serving and measurement path?
- Execute a few short requests, register completed prefixes, deliberately displace
  selected prefixes from GPU memory, restore them from CPU and send a causal hint.
  Include startup, streaming, normal drain and owned-process cleanup.
- Arms: C and H; G startup/request compatibility when introduced.
- Measure request completion, actual lengths, registration/hint outcomes, store and
  restore evidence, protocol validity, timeouts and cleanup. Include deterministic
  restored/fresh output checks where the existing functional protocol supports them.
- Validity: prove readiness and real restoration independently of hint acceptance.
  Reuse the existing GPU byte-integrity test as a prerequisite where relevant;
  HTTP success and matching text do not replace KV-byte checks.
- Use before campaigns and after runner/runtime changes. Exit success means the
  measurement setup worked, not that a policy improved performance.

#### B01: Overhead and measurement sensitivity

- Question: what does each layer cost when it has no useful cache opportunity?
- Execute independent prompts, GPU-resident repeats and roomy-cache requests. Within
  C/H distinguish no registration, registration without hints, and matched hint
  traffic. Measure at light load and one shared busy operating point.
- Arms: G/C/H as applicable; unchanged-versus-unchanged pairs establish noise.
- Measure latency, completed requests, CPU time, RSS, GPU allocation and registration/
  hint delivery costs. Report server and client costs separately and together.
- Qualify timing sensitivity with a separately identified, intentionally added delay
  inside the measured path. Verify it against an independent clock/event receipt.
  This diagnostic never enters candidate performance estimates. Misleading hints
  are not assumed to be a known slowdown; B04 measures whether they actually harm.
- Output: overhead by layer and a noise/sensitivity report. Required for new mechanisms
  and integration changes; no universal percentage rejection rule.

#### B02: Reuse structure and distance

- Question: which patterns of exact reuse make retention valuable?
- Execute independent returning prefixes, prefixes shared across sessions, and
  growing/partially overlapping transcripts. Include no reuse, uniform reuse and
  skewed popularity. Vary elapsed wait independently from intervening distinct
  native KV-block bytes where feasible; count shared ancestors only once.
- Arms: C/H, with selected G anchors. Start with fixed external prompt traces;
  completion-dependent cases are identified separately through B05/B06.
- Measure return TTFT, completion latency, reuse/recompute tokens, CPU retention,
  eviction and restoration. Record offered prefix structure and observed exact-key
  opportunity, not just a configured reuse percentage.
- Output: behavior by reuse structure and distance. Negative-control early-prefix
  changes verify that similar text is not being treated as exact reuse.
- Claim: controlled cache behavior, not prevalence of that behavior in applications.

#### B03: Capacity and computation boundaries

- Question: when do saved prefill costs outweigh movement, admission and contention?
- Independently vary GPU and CPU KV capacity: GPU-resident working set, GPU spill
  with adequate CPU space, and competing CPU working set. Include active generation
  competing for GPU blocks. Vary reusable prefix, new suffix and output lengths
  separately, with selected mixed-length cases and block-boundary cases.
- Arms: G/C/H, matching explicit GPU capacity and disclosing extra CPU resources.
- Measure TTFT, inter-token gaps, end-to-end latency, completed work, transfer bytes,
  recomputation, preemptions, memory and refused stores/evictions. Record actual
  unique working-set bytes and active versus reusable cached occupancy.
- Output: capacity/latency/throughput curves and observed transition regions.
- Claim: resource trade-offs for this geometry; an artificial cache cap is labelled
  as a controlled pressure experiment, not a normal deployment configuration.

#### B04: Hint usefulness, timing and interference

- Question: when does advice help, waste work, or displace useful state for others?
- Execute no hints, timely valid hints, late valid hints, refused stale hints and
  accepted hints for abandoned continuations. Vary the fraction of hinted sessions
  and include shared-prefix and independent-prefix cases. Repeated event replay
  and genuinely new periodic events are separate cases; replays must not be counted
  as new cache refreshes.
- Arms: matched C/H; no-registration/no-hint overhead references from B01.
- Measure hinted and unhinted cohorts separately, hint production and RPC cost,
  refusals, new refreshes, later reuse and latency. Trace the observation/readiness/
  refresh/retention/restore chain. A refreshed key is not automatically saved work.
- Use the actual returned server observation and same-host monotonic clock. Never
  fabricate readiness, future text or exact future completion knowledge. Ideal
  timing is an explicitly synthetic opportunity bound; real availability is B07.
- Output: benefit/harm by advice quality and timing. An abandoned-session hint may
  benefit another session sharing its keys; do not automatically label it wasted.

#### B05: Demand, queueing and causal returns

- Question: how do load and feedback change the observed advantage?
- Primary mode: predeclared offered arrival schedules, including spaced, random and
  bursty arrivals, from light load through a bounded overload episode. Compare arms
  at the same absolute offered rates; relative-to-own-capacity plots are secondary.
- Companion mode: each session returns after its predecessor completes plus a
  declared wait. New-session arrivals remain explicit. This reveals feedback among
  service time, growing contexts and cache competition.
- Record scheduled arrival, client submission, server admission when available,
  first token, subsequent token events, completion, outstanding backlog and failures.
  Client concurrency limits must not silently turn offered load into completion-
  limited load. Declare queue caps, timeout, overload termination and drain policy.
- Output: latency/load curves, attained throughput and backlog growth; separate
  client waiting from server time. Report failures and unfinished requests in all
  denominators. Do not invent an application SLO; later report goodput against
  declared latency limits only when those limits have a stated purpose.
- Claim: controlled load behavior and causal feedback, not realistic traffic mix.

#### B06: Repeatable replay and change confirmation

- Question: does a proposed change improve a repeatable, stated serving problem?
- Assemble a versioned development panel from B01-B05 with positive-opportunity,
  no-benefit, adverse and combined-condition cases. Retain unfavorable regions.
  Compare a new candidate with frozen incumbent and native controls as relevant.
- Fixed external replay supplies identical tokenized prompts/schedules/decoding
  settings. Requested and actual output lengths and fingerprints are retained.
  Different outputs mean nominal-demand comparison, not identical execution.
- Exact recorded-token replay is optional and requires an explicitly qualified
  execution path that performs genuine decode/cache operations. Do not implement
  a custom forced-token engine solely to claim identical work. If unavailable,
  retain the limitation and use fixed prompts plus live confirmation.
- Completion-dependent replay is labelled separately from externally scheduled
  replay. A recorded transcript is never described as the current arm's live output.
- Output: paired changes and uncertainty per case, plus a compact regression panel.
  No single weighted score determines selection. A short screen is advisory.
- Confirmation uses fresh independent trace instances/source groups and a declared
  analysis plan after tuning freezes. Previously examined variants remain exposed.

#### B07: Product-derived application discovery and confirmation

- Question: do opportunities found in characterization occur naturally and improve
  useful completed work when actual outputs and integrations are included?
- First inspect candidate real traces/tasks for reuse, return gaps, pressure and
  available hint signals. Select applications from that observed fit, including
  weak-opportunity cases. Initial possibilities are document follow-ups, repository
  investigation with real tools and other session-based tasks; these are proposals,
  not mandatory targets or measured production proportions.
- Execute actual tools and use actual generated outputs in subsequent requests.
  Record tool duration, failures and truncation. Hints use only information the
  application possessed then; include prediction, integration and RPC costs.
- Arms: candidate/incumbent/native control, plus selected B10 alternatives later.
- Measure full workflow time, return-ready-to-first-token delay, completed and valid
  tasks per time, errors and resources. Define task-specific checks before comparing
  results: e.g. reported test outcome versus real exit status, source-grounded answer
  checks, and strict format/truncation checks. Retain exact outputs for review.
- Human or model-assisted audits must disclose selection and uncertainty; no paid
  judging service is assumed. Separate objective checks from uncertain judgments.
- Output: application result with success/quality and performance jointly presented.
  Different live outputs are allowed, but prevent identical-work speedup claims.
  Scripted waits are labelled; they are not measured human behavior.

#### B08: Mixed-traffic impact

- Question: does a benefit survive sharing the server with unrelated work?
- Mix returning hinted sessions, returning unhinted sessions and independent traffic;
  include short and long requests and a change in which cohort is popular. Derive
  initial cases from B02-B05; later substitute B07 tasks without changing accounting.
- Arms: C/H or incumbent/candidate; keep request classes and resource allocation
  comparable. Include misleading advice under real competition.
- Measure each cohort's latency, completion/failure rate, queueing and cached-work
  savings as well as aggregate outcomes. Report slowdown distributions and sustained
  backlog by cohort; sample counts accompany tail metrics.
- Output: explicit benefit distribution and interference. No pooled average can
  establish fairness, and no fairness bound is assumed without Philip choosing it.
- This expands B04's mechanism probe into system-level coexistence evidence.

#### B09: Sustained operation and recovery

- Question: does behavior remain stable across time, churn and interruption?
- Keep the server alive across waves, bursts, quiet periods, changing hot sets,
  session creation/retirement and abandoned sessions. Include owned request
  cancellation/disconnection and bounded server failure/restart controls.
- Exercise configured metadata/session/event capacities and reclamation; counters
  and RSS trends distinguish bounded refusal from continuing memory growth.
- Measure windowed throughput/latency, memory, retained state, errors, queue drain,
  cancellation acknowledgement and successful recovery generation.
- Arms: relevant candidate and control; runtime failure injection is isolated from
  performance intervals. Preserve failed attempts and verify owned-process cleanup.
- Output: duration- and workload-bounded operational evidence. A passing short wave
  test is not an indefinite leak test or production certification. Choose longer
  exposure from observed lifecycle/capacity times, not an arbitrary overnight ritual.

#### B10: External alternatives and resource trade-offs

- Question: would another maintained system be a better practical choice?
- Select a currently relevant exact prefix-cache/offload alternative after checking
  compatibility and maintenance. Historical Continuum comparisons are research leads,
  not a selection or a current ranking. Selection and installation are later work.
- Run a small B06 panel, selected B07 applications and B08 interference cases. Report
  both explicitly matched resource constraints and separately labelled documented
  system-specific tuning. Freeze tuning before fresh comparison.
- Record differences in runtime, chunk size, decode-KV publication, scheduling,
  paging, templates, stopping and instrumentation. Unsupported features do not
  silently fall back. Correctness or output-quality differences limit the comparison.
- Measure latency, valid completed work, CPU/GPU resources and setup/integration
  costs relevant to adoption. Token rate alone cannot rank different live outputs.
- Output: scoped system comparison, not causal attribution to a single policy.

#### B11: Model/runtime/hardware coverage admission

- Question: which conclusions transfer beyond the initial configuration?
- This is a qualification matrix reusing B00-B10, not another workload generator.
  Add a larger feasible model, then another model family; select cases that challenge
  memory geometry and compute/transfer balance. Do not run every cell everywhere.
- Each model declares tokenizer/template, immutable identity, attention layout,
  precision, context bounds and measured KV bytes. Regenerate tokenized inputs when
  needed while preserving source ancestry and documenting changed actual lengths.
- New precision, architecture, runtime or hardware first passes relevant functional
  CPU/GPU checks, B00, then B01 and selected B03/B06/B07 cases. Additional hardware
  requires permission; no 5080 availability or qualification is assumed.
- Output: tested/unsupported/blocked/untested coverage with evidence links. Capability
  admission never changes the meaning or claimed scope of old results.

### 4. Measurement and interpretation shared by the suite

Primary measures are request/session latency, first-token delay, token delivery
stalls, completed work, task validity where applicable, failures and resource cost.
For fixed-length panels requests/second is interpretable; for mixed/live workloads
also expose input/output lengths and valid task completions. Denominators include
submitted, offered, rejected, incomplete and failed counts as appropriate, with an
explicit measurement window and drain treatment. Startup is separate.

Diagnostics include exact reused/recomputed tokens, CPU loads/stores and transfer
bytes, eviction/store refusals, preemptions, hint outcomes and retained state.
Separate logical per-request token accounting from physical repeated work or
transfers under preemption. Missing data is unavailable, never silently zero.
Only demand restore or reduced recomputation can support saved-work attribution;
a ready allocation or accepted hint cannot.

Capture low-cost client events in timed runs. Establish the overhead of required
server counters; use symmetric instrumentation. Expensive per-block tracing is a
separate diagnostic replay unless its timed cost is measured and disclosed. Reuse
supported runtime counters where possible; any new adapter telemetry must stay in
the adapter boundary and be validated against actual behavior. If critical evidence
is unavailable, narrow the claim or qualify telemetry; do not invent counters.

Use paired independent execution blocks as the main uncertainty unit, preserving
within-run correlation. Report estimates and intervals per operating condition,
not only pooled wins. A/A trials characterize noise and drift. Pilot variability
and the decision's useful effect size determine repetition. Freeze confirmation
counts/precision and stopping rules before acquisition; adaptive exploration is
labelled and separate. If budgets expire with broad uncertainty, report inconclusive.
Tail latency needs adequate observations; small-sample quantiles remain descriptive.
Multiple exploratory cells do not justify cherry-picked winner significance.

### 5. Workload assets, ancestry and exposure

Inspect existing assets before downloading or generating replacements. The old
serving workload catalog and repository task/pressure task examples were sampled:
they are authored and historically exposed. Their labels do not establish real
traffic or actual pressure. Inventory any selected raw corpus by source, version,
license/retention constraints, transforms, parent/session/project grouping, exact
prompt structure, prior experiments and known exposure. Unknown history is unknown.
Do not copy the whole old runner or restore retired feature implementations.

Keep three roles distinct: repeatedly used development data; exploratory discovery
inputs; fresh confirmation groups reserved after a decision is fixed. Related turns,
variants and projects stay together when that dependency matters. Synthetic seeds
alone do not make a workload family independent. Store source-level ancestry before
tokenization and record tokenizer/template transforms and actual realized lengths.

Characterization can begin with small authored exact-prefix fixtures and controlled
schedules. B07 uses useful source material and real tasks. No large dataset acquisition
is required to start. Capture all planned/actual mixture proportions; authored mixes
are coverage choices, not estimates of deployed traffic.

### 6. Proposed commands and repository wiring

Add one maintained entry point, `uv run --locked python -m tools.benchmark`, with
subcommands below. These commands do not exist yet. Keep `tools/check.py` the validator
and `tools/runtime_verify.py` the functional verifier; neither becomes a performance
campaign script. Reuse their helpers rather than create competing process managers.

| Proposed command | Responsibility |
| --- | --- |
| `tools.benchmark plan --definition FILE --model PATH --runtime-python PY --out NEW_DIR` | Validate/resolve inputs without serving; print arm/case counts, bounds, estimated cost and unverified runtime facts; write immutable plan. |
| `tools.benchmark run --plan PLAN --out NEW_DIR --gpu` | Execute explicitly authorized acquisition with frozen resolved inputs, finite limits and owned cleanup. |
| `tools.benchmark verify RUN_DIR` | Offline artifact/semantic accounting verification; no server or GPU. |
| `tools.benchmark report RUN_DIR --out NEW_DIR` | Recompute summaries/plots from retained events; never rewrite raw records. |
| `tools.benchmark resume --run RUN_DIR --out NEW_DIR --gpu` | Create a linked continuation for eligible missing cases; preserve original failed/interrupted attempts. |

All commands use the portable interpreter as controller. Run invokes the explicit
pinned runtime interpreter only in owned subprocesses. Plan can inspect local model
metadata on CPU but cannot claim model fit, successful import or hardware behavior.
Paths are structured arguments, not interpolated shell commands. No downloads,
credential changes or implicit network services. `--gpu` is an execution opt-in,
not proof of user authorization; agents must retain the task's resource permission.

Proposed file ownership and integration:

| Location | Owns / required change |
| --- | --- |
| `benchmarks/definitions/` | Small versioned benchmark/campaign JSON definitions, system and workload choices; no executable run-local scripts. |
| `benchmarks/fixtures/` | Compact authored portable examples, source descriptors and expected accounting cases; large corpora remain external. |
| `tools/benchmark.py` | CLI parsing and verdict presentation only; one explicit entry-point exception. |
| `tools/benchmarking/` | Typed definitions, case expansion, workload adapters, client scheduling, execution, event accounting, verification and reporting; start with actual needed modules. |
| `tools/config.py` | All new environment reads and external artifact-root resolution. |
| `tools/process.py` | Reuse owned-process identity, timeout and cleanup; extend lifecycle operations only when required and test actual processes. |
| Existing runtime snapshot/receipt helpers | Extract narrowly reusable code if necessary; preserve existing functional receipt readers and schema meanings. Benchmark records have their own version. |
| `wait_hint_offload/vllm_adapter/` | Only location for new vLLM-specific telemetry/integration; pure product core never imports benchmark code. |
| `tests/` | Portable benchmark contracts and accounting tests; normal discovery/inventory/full acceptance. |
| `tests/runtime/` | Pinned integration and explicit GPU qualification tests, keeping CPU and GPU discovery separate. |
| `tools/system_map.py`, `docs/system-map.md` | Add the maintained benchmark command when implemented; entries must reflect actual availability. |
| `tools/check.py`, `tools/rules.json`, proof specimens | Add useful mechanical definition/reference checks and their proofs; no GPU in portable acceptance. |
| `tools/exceptions.toml` | Narrow reviewed CLI exception; no general exemption for benchmark scripts, suppressions or imports. |
| `pyproject.toml`, `uv.lock` | Keep typing/lint/dependency coverage; add dependencies only for implemented needs, pinned through the existing mechanism. |
| `README.md`, `docs/benchmarking.md`, `docs/checks.md` | Short navigation, maintained procedure, targeted/full validation and common repairs. |
| `AGENTS.md` | Proposed brief benchmark navigation and interpretation rule; Philip owns approval. |

`tools/benchmarking/` is already under the typed development root, while benchmark
JSON has explicit schema/reference checks. Check import and dependency tooling when
adding modules so no authored path escapes lint/types/tests. Test inventories are
updated through existing commands; do not create a parallel testing framework.
Keep authored files tracked and raw runs external; add no broad ignore for benchmarks.

### 7. Extensibility with explicit boundaries

A benchmark definition selects workload, system arms, model profile, traffic,
resource geometry, metrics, required evidence and budgets. Version closed schemas;
reject unknown fields/IDs and unsupported combinations with a repair instruction.
Keep host-private paths/API keys outside tracked definitions; store redacted resolved
identity and secret-free commands in receipts.

Three concrete variation points suffice initially:

- Workload adapter: prepares requests or causal sessions and declares input ancestry,
  timing dependencies, hint availability and output-validity rules.
- System adapter: validates capabilities, launches the owned server, exposes supported
  endpoints/telemetry and stops it safely. Initial adapters are G and guarded C/H.
- Model profile: immutable model/tokenizer identity, template and supported runtime
  geometry. It supplies facts and admission constraints, not arbitrary callbacks.

Use a small explicit registry of implemented adapters, not dynamic code loading,
entry-point discovery or a user plugin framework. Each adapter supplies only evidence
it actually supports. The verifier selects required evidence from the benchmark's
claim. Lack of optional diagnostics is explicit; lack of claim-critical evidence
makes the measurement invalid for that claim.

Adding a workload means one adapter only if the existing schema cannot express it,
plus fixtures and documented validity rules. Adding a model normally changes a
profile and tokenized inputs, then B11 qualification. Adding a candidate changes an
arm/source/config identity, not copied runners. A new mechanism may require new
activity evidence and diagnostics; wait-hint counters are not universal requirements.
Adding an external system means an explicit adapter and fairness statement.

### 8. Records, recovery and result closeout

Every run retains a resolved immutable plan, corpus/trace provenance, source/runtime/
model identity, arm commands, per-request and session events, resource samples,
required server observations, exact outputs, failures, terminal status and cleanup
receipt. Raw locations are outside Git under the existing project data root or an
explicit new external directory. Never overwrite an attempt.

Keep four states separate: planned, implemented, exercised and qualified. Execution
status (complete/failed/interrupted), measurement validity and performance conclusion
are separate fields. A complete valid loss is successful measurement. Offline verify
can validate the integrity of an honestly failed attempt without calling it a pass.
Report prints a short verdict, scope, limitations and artifact path.

Resume occurs at declared independent case/pair boundaries. Do not pretend to restore
an in-memory cache after a crash. An interrupted warmed sequence restarts from its
specified initial state in a new attempt. Preserve the failed prefix and exclude it
from a complete-pair estimate only with explicit reason and counts; never erase it.
A changed runtime/source/model/config creates a new plan with lineage. Missing source
or receipts fail verification; report-only recovery is not exact rerun capability.

Use existing `results/NAME/result.json`, generated Markdown/chart and FINDINGS for
coherent completed investigations. Store compact figure inputs and supporting reports,
not raw streams or one page per run. Prefer existing figure support initially;
extend the result schema/rendering only when a useful curve or panel requires it,
with old readers/results preserved. Do not force unrelated metrics into one chart.
Invoke close-results at investigation closeout; retain losses and uncertainty.
The local Git bundle preserves source/compact results only. External raw/model
preservation remains separately documented; this plan creates no hosted remote.

### 9. Agent guidance and long-running work

Proposed AGENTS guidance, to approve with implementation: use docs/benchmarking.md;
start with plan and the smallest relevant benchmark; checks are not performance
qualification; a valid run is not necessarily a win; keep adverse/incomplete evidence;
obtain explicit GPU scope and close results through the existing procedure.

The topic guide should answer: which benchmark fits the question, how to inspect
cost, how to supply model/runtime/assets, what authorization is needed, how to run
and verify, what a refusal means, and how to interpret/resume/close the result. Include
one worked retained example and one failed attempt. Do not duplicate full commands
and thresholds in AGENTS or personal skills.

A project benchmark-run skill is a candidate for the separate agent-operations
phase, after the command has been exercised. It should wrap the maintained plan/run/
verify/close workflow, not own another runner. Shared benchmark-design remains design
guidance. GPU permissions, completion notifications and sustained-work ownership are
recorded in the task; reading a task does not assign it to another agent.

Long runs need a durable owning process, explicit time/resource limits, append-only
progress, owned cleanup and a tested completion/failure notification integration.
Coordinate the mechanism with agent operations; do not invent a GPU reservation
system or assume a session ending cancels ownership. Until that integration is
qualified, keep acquisitions bounded and supervised. Handle healthy jobs without
moving their frozen dependencies or reclaiming their artifacts.

### 10. Cost, cadence and acquisition limits

These are proposed engineering targets, not measured turnaround or authorized GPU
budgets. Pilot observations must replace estimates before a larger campaign.

| Use | Proposed initial intensity / cost control |
| --- | --- |
| Portable development | Relevant targeted tests while editing; full acceptance before commits. Keep planning/verification CPU-only. Measure added fast-lane cost. |
| B00 | One tiny case per introduced arm; aim for minutes after model availability. Report startup separately. |
| Calibration pilot | One paired case per selected family, plus A/A and sensitivity controls; enough to estimate activity, record volume and timing noise, not a winner claim. |
| Routine screen | A fixed small panel, initially at most 6-8 cases with short bounded windows; aim for tens of minutes including startup, revisable from pilot. |
| Exploration | At most 24-32 distinct conditions in the first proposed batch, selected for corners/interactions rather than all combinations; approve actual wall-time/storage bounds after pilot. |
| Confirmation | Freeze selected cases, fresh inputs and useful precision; size repeats from pilot variability and cap total acquisition explicitly. No universal pair count. |
| B09 | Short lifecycle/capacity checks first; longer exposure sized from observed turnover and resource trends with a separately approved duration. |
| B10/B11 | Small representative panels first; expand only for a stated comparison or transfer claim. |

Plan reports cases x arms x repeats, startup/warmup/load/drain bounds, measured or
estimated wall time, expected/max output volume, model/runtime storage needs and
required memory headroom. Estimate traces from pilot events/second and bytes/event,
including frozen inputs and report intermediates. Enforce declared ceilings; a stopped
case is retained as incomplete. Do not selectively extend only favorable cells.

### 11. Implementation sequence and acceptance evidence

1. Implement portable definitions, plan validation and offline accounting/verification.
   Exercise bad IDs, incompatible geometry, missing assets, unknown versions,
   malformed/duplicate/out-of-order events and incomplete records. Hand-computed
   fixtures verify denominators and timing; hashes alone are insufficient.
2. Implement one authentic G/C/H serving path and B00/B01. Reuse existing functional
   qualification. With separate GPU permission, demonstrate real activity, clean/warm
   distinction, A/A noise, measured sensitivity and owned cleanup. Preserve failures.
3. Implement B02-B05 as definitions/adapters on that path; qualify representative
   conditions including no benefit, interference and overload. Verify actual pressure,
   client scheduling lag, causal hint timing and server counter agreement.
4. Freeze a small B06 development panel and exercise one source/config change through
   plan/run/verify/report. Demonstrate interrupted-pair resume and stale-source refusal.
5. Select and implement one B07 application from characterized opportunity, with actual
   tools/outputs and predeclared validity checks. Challenge apparent serving gains
   early, before optimizing or acquiring a large campaign. Then expand B08 mixtures.
6. Qualify B09 lifecycle/recovery and the agreed operations handoff. Real timeout,
   failed startup, disconnect and interruption controls must exercise owned processes;
   fake fixtures alone do not establish operational readiness.
7. Add B10 alternatives and B11 larger/different-model coverage for explicit claims.
   Resolve incompatible runtime/precision facts through qualification, never fallback.
8. Publish the first coherent result with chart/data/limitations using close-results;
   verify report regeneration and artifact recovery. Update current docs, remove this
   completed design task only when its decisions are incorporated into maintained
   definitions/docs, and retain unfinished implementation in the task convention.

Each milestone reports planned versus implemented versus actually exercised coverage,
acceptance results, GPU scope, unresolved facts, local preservation and next work.
No source changes beyond the approved milestone, CI adoption, new remote or runtime
repin are implied. Project acceptance remains local pre-commit and full checks.

### 12. Review choices and unresolved facts

Recommended choices for Philip's review:

1. Adopt B00-B11 as the inventory, with B11 explicitly reusing prior benchmarks;
   implement incrementally rather than making every family a prerequisite to start.
2. Approve the single `tools.benchmark` entry point, closed definitions and narrow
   adapters, with the corresponding reviewed script exception and AGENTS navigation.
3. Keep fixed external requests plus a small causal continuation check initially;
   defer forced-token replay unless its additional precision justifies integration.
4. Treat the proposed screen/exploration sizes as starting ceilings for cost planning;
   choose acquisition budgets only after pilot estimates and explicit GPU permission.
5. Choose real application and external-system examples from inspected evidence
   later, preserving the product-led discovery approach.

Technical facts to resolve during implementation/pilots, not questions delegated to
Philip: exact telemetry availability and cost; causal registration behavior with
replayed inputs; affordable model sizes on this runtime; observed pressure boundaries;
measurement noise and sample size; compatibility of a selected external system;
actual long-run notification integration. Missing facts must remain labelled until
exercised. Meaningful changes to scope, guarantees or resource budgets return to Philip.

### 13. Research and retained evidence

- [vLLM serving benchmark controls](https://docs.vllm.ai/en/stable/cli/bench/serve/): arrival/concurrency distinction, token timing and goodput controls.
- [vLLM cache carryover warning](https://github.com/vllm-project/vllm/blob/main/docs/benchmarking/cli.md): explicit initial cache state.
- [vLLM prefix caching scope](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/): prefill savings versus decode work.
- [MLCommons inference scenarios](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc): controlled offered-load methodology, not a requirement to implement MLPerf.
- [Measurement reliability](../results/measurement-reliability/result.md): activity, failure controls and immutable attempts.
- [Interference and output validity](../results/interference-quality/result.md): cohort harm and limits of throughput.
- [Replay findings](../results/wait-recency-replay/result.md): constrained replay scope and latency trade-offs.
- [Workload dependence](../results/workload-dependence/result.md): conditional gains/losses and misleading cache-regime labels.
- [External comparisons](../results/external-comparisons/result.md): configuration and output differences prevent simple rankings.

Current upstream documentation informs design, not compatibility with the pinned
runtime. Historical results are hypothesis sources, not qualification of this code.

## Next
Philip reviews this proposal. Revise agreed choices here, then authorize a bounded
implementation milestone and its resource scope. Keep application selection, numeric
performance thresholds and acquisition budgets open until their evidence is available.
