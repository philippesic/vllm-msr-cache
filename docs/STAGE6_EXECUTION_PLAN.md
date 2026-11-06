# Stage 6 execution plan for repeatable vLLM evaluation

Recorded 2026-11-06. Status: execution design ready; pipeline implementation and
Stage 6 acquisition have not started. This document is the handoff for the agent
implementing and executing the campaign, not evidence that its checks passed.

Build a maintained, resumable benchmark pipeline, complete its three-model
RTX 2080 Ti campaign, and leave a verified reproduction package for the RTX 5080.
The remaining work at a successful handoff must be the 5080 acquisition and the
release write-up. Produce technical findings now; public release writing is Stage 7.

## 1 Scope and execution mandate

The user selected vLLM only and these three models:

1. `Qwen/Qwen3-8B-AWQ`, the primary model.
2. `mistralai/Ministral-3-3B-Instruct-2512-BF16`.
3. `Qwen/Qwen3-4B-Instruct-2507`.

Defer SGLang, llama.cpp, Qwen3.5, Gemma, hybrid-state support, distributed serving,
other devices, and further Stage 5 policy optimization. Do not substitute a model
or revive paused AWQ-loading research to make the matrix easier.

The execution agent should implement the pipeline, acquire pinned public inputs,
assemble isolated environments, repair ordinary compatibility/runner failures,
run and verify the local campaign, preserve all attempts, and clean up owned
resources. Continue independently through routine failures. Do not stop at the
first benchmark, an inconclusive comparison, or an unfavorable result. A question
or status request does not cancel execution. This document-writing turn does not
itself start that campaign.

Read [AGENTS.md](../AGENTS.md), [STATUS.md](STATUS.md),
[BENCHMARKING.md](BENCHMARKING.md), [runtime setup](../vendor/VLLM.md), and
[supported wait-hint behavior](WAIT_HINT_SUPPORT.md) first. User instructions take
precedence over older plans. Keep other tasks and the paused AWQ experiment intact.
Commit coherent implementation/evidence milestones and push the established
user-owned remote; do not force-push or push to a third-party upstream.

Success means complete, reproducible measurements and honest decisions. It does
not require winning every comparison. Missing required model measurements or
broken correctness cannot be called complete. If an external system cannot run a
selected cell after a documented compatibility investigation, retain its explicit
capability gap and finish independent work; that is a partial external comparison,
not an observed performance loss. Do not claim that only 5080 work remains while
a required local cell is still blocked.

## 2 Starting point and evidence boundaries

Stage 5 is complete. Keep the selected CPU wait-retention rule; no experimental
optimization qualified. See [Stage 5 findings](../evidence/stage5-optimization-20261105/README.md)
and [selected candidate](../evidence/stage5-optimization-20261105/selected-candidate.json).
The behavior refreshes already-ready exact-prefix CPU entries inside ARC's queues
while a workflow waits for tools and its returning request's first token. It does
not count advice as demand hits, change ARC's learned balance, reserve GPU pages,
or dispatch speculative transfers.

The selected JSON contains the old model namespace and resource budget. Treat it
as a behavioral template, not a configuration to copy unchanged. Resolve the real
model/tokenizer revisions, layout identity, and per-cell byte budgets for every
new model. Preserve its other policy settings and its reference client behavior:
before-tool advice, refresh during actual waits and return queues, stop at first
token, one-second refresh cadence, and the existing 750 ms declared estimate.
Never expose actual future completion times to a policy.

The active profile has a positive enable budget but is CPU-retention-only with
diagnostic no-dispatch enabled. Setting its budget to zero is part of disabling
the addition. Disabling only `wait_hint_cpu_retention_only` can enable a different
path and is not an adequate off switch. Assert no speculative dispatch in every
candidate case. Linear ARC and all other Stage 5 experiments remain disabled.

Historical results establish workload-specific ARC gains, neutral regimes, and
upstream/Continuum losses. Reuse designs and code, not historical timings as new
confirmation observations. The [operating-range report](../evidence/wait-qualification-20261105/STAGE3.md)
and [support matrix](../evidence/wait-qualification-20261105/SUPPORT_MATRIX.md)
give the relevant limits. Existing source/test tasks are authored tool workflows,
not autonomous-agent trajectories or SWE-bench.

**Upstream configuration precedence:** the later
[Stage 2 qualified comparison](../evidence/wait-qualification-20261105/STAGE2.md)
uses the disclosed host-pointer compatibility fix, V2 runner, and actual pinned
backing. Earlier five-way instructions with pinning disabled and V2 disabled are
historical probes. Inspect retained manifests and reconstruct the later working
configuration; do not silently adopt the obsolete workaround.

## 3 Build on the maintained suite

Keep the implementation in `benchmarks/serving_suite/`, with reusable helpers in
`harness/` where appropriate. Add a campaign coordinator, not a parallel benchmark
framework or a supervisor importing scripts from `output/`. Preserve existing
four-tier entry points and their historical artifact verification.

| Existing component | Reuse or extend |
| --- | --- |
| `contracts.py` | Versioned validation, canonical digests, atomic JSON writes |
| `runner.py` | Case launch, effective configuration, capture, source freezing; extract small reusable boundaries where needed |
| `resources.py` | Owned server/process cleanup, resource sampling, readiness |
| `workload.py`, `repo_workload.py` | Streaming transport, real tool execution, observed-prefix publication |
| `published.py`, `published_server.py` | Pinned Continuum/upstream launch, bridge, activation and geometry checks |
| `report.py`, `agent_report.py`, `verify.py` | Raw-derived metrics, activity, artifact seals, offline recomputation |
| `harness/gpu_lock.py` | Supervisor and direct-process GPU ownership |
| `benchmarks/research_checks/qualification_inputs.py` | Source/test and PEP corpus construction patterns |
| `qualification_prompt_repair.py`, `qualification_quality.py` | Frozen instruction/scoring lessons; audit before reuse |
| `qualification_shape.py`, `qualification_allocation_receipts.py` | Functional restoration and actual allocation checks |
| `wait_hint_qualification.py`, `stage5_optimization.py` | Historical order/freeze/attempt conventions; do not depend on their hard-coded run paths |

Known implementation gaps to address explicitly:

- The current `agent` runner assumes four model requests and three tools. It is
  not a general workflow graph executor.
- Current corpus validation describes linear turns. Add a versioned graph/task
  contract for dependent workers and tools; preserve old corpus readability.
- Current published launch accepts selected agent/eager configurations, and
  runtime pins are embedded in code. Expose validated catalog identities without
  weakening source checks or relabeling patched runtimes as published originals.
- Generic tier defaults are not the selected wait profile. Every campaign arm
  must receive an explicit resolved configuration.
- Existing confirmation session selection requires disjoint sessions per repeat.
  A campaign may explicitly use a fixed evaluation cohort across independent
  process repetitions; do not silently bypass or mislabel that distinction.
- Extend the source freezer to include new modules, manifests, templates, tool
  definitions, and scoring inputs actually executed by the campaign.

Recommended additions, with names adjustable only if the same responsibilities
remain clear:

```text
benchmarks/serving_suite/
  campaign.py                 # CLI and phase orchestration
  campaign_contracts.py       # catalogs, resolved plan, ledger, result schemas
  assets.py                   # explicit preparation, pinning, path bindings
  workflow.py                 # typed request/tool/dependency execution
  campaign_report.py          # cross-cell/model/platform reporting
  campaign_verify.py          # verify/export/import without GPU or model loads
  catalogs/
    models.json
    workloads.json
    systems.json
  campaigns/stage6.json
  templates/host.example.json
  tools/                      # reusable source/document tools if extracted
tests/
  test_campaign_contracts.py
  test_campaign_resume.py
  test_campaign_workflows.py
  test_campaign_reporting.py
```

Model and workload additions should normally be manifest entries plus an existing
adapter. A new workload type may need one small adapter. Do not scatter model-name
conditionals through orchestration, statistics, and reporting. Keep inference
execution vLLM-specific for now; no speculative multi-engine abstraction project.

## 4 Manifest and artifact contracts

Use versioned JSON contracts with strict validation and explicit migrations.
Logical identities are portable; host filesystem bindings are separate. Reject
unknown or contradictory critical settings before launching a server.

| Record | Required information |
| --- | --- |
| Model catalog | Logical ID, publisher/repository, immutable revision, file hashes, tokenizer/template hashes, architecture/group requirements, weight format, execution/KV dtype, thinking mode, generation defaults, license/source |
| Workload catalog | Stable cell ID, adapter/version, source revision and license, task IDs/split, dependency graph, arrival/wait rules, prompt/output budgets, tool definitions, cache regime, hint eligibility, activity expectations, quality rubric |
| System catalog | ARC/wait/Continuum/draft identity, checkout and build pins, dependency/native identity, explicit patch chain, launch/bridge adapter, connector, retention settings, activation evidence |
| Host profile | GPU identity/VRAM, CPU/RAM/OS/WSL, driver/CUDA, local Python/source/model/cache paths, ports, disk/resource limits; private bindings kept out of portable manifests |
| Resolved campaign | All selected model/cell/system combinations, actual byte budgets/block geometry, seeds/order, cohorts, repetitions, execution mode, policy fingerprint, timeouts, stopping rules and estimated cost |
| Attempt ledger | Stable logical case ID, fresh attempt ID, phase/repeat/system, identity digest, start/end, status, artifact location, failure class, supersession/repair links |
| Result | Measurement validity, feature activation, correctness, quality, resource envelope, statistical interpretation, and cleanup as separate fields |

The logical case identity includes model artifact, workload/cohort, system,
hardware profile, configuration, and repetition. Do not key solely by a friendly
model name or reuse a result because its directory exists.

Keep execution state separate from result interpretation. Ledger states should
include planned, running, interrupted, failed, verified, superseded, unsupported,
and blocked. Statistical outcomes include benefit, bounded regression/neutral,
regression, and inconclusive. An inactive expected feature is not a negative
performance result. An intentionally inactive feature in ordinary traffic is valid.

Use an append-only event ledger plus an atomically replaced index. A completed
attempt is immutable after sealing; diagnostics and revised reports go in new
artifacts linked to it. Resume verifies identity and seals before skipping work.
An interrupted measured process is rerun as a new whole-case attempt, not resumed
midstream and concatenated into an independent repetition. A broken comparison
block is retained and replaced as a block when its pairing/order assumptions fail.

Preparation is the only phase that fetches assets. Measurement runs from resolved
local inputs with no model or corpus downloads. Resolve movable publisher refs
once, freeze immutable identities, then use those identities on both machines.
Do not save authentication tokens, full environment dumps, or private paths in
portable public evidence. Export allowlisted environment settings only.

Provide a checked-in host template with this shape (illustrative field names for
the new contract, not an existing accepted schema):

```json
{
  "schema": "serving-campaign-host-v1",
  "id": "local-2080ti",
  "device": "cuda:0",
  "runtime_python": "/absolute/path/to/pinned/python",
  "asset_cache": "/absolute/path/to/reusable/assets",
  "models": {},
  "runtime_bindings": {},
  "limits": {
    "startup_timeout_s": 900,
    "request_timeout_s": 180,
    "tool_timeout_s": 180,
    "case_timeout_s": 1800
  }
}
```

Preparation fills logical-model-to-snapshot and runtime bindings in a private
resolved file; it must not overwrite a user's existing host profile. Compilation
records measured hardware facts separately from these user-supplied bindings.
Development probes can revise timeouts with a reason before freeze. Confirmation
uses the locked values, with case timeout at least three times the longest valid
pilot exposure if 1,800 seconds is insufficient. Timeout failures stay in the ledger.

## 5 Model preparation and compatibility

| ID | Source | Intended configuration and gate |
| --- | --- | --- |
| `qwen3_8b_awq` | [Official checkpoint](https://huggingface.co/Qwen/Qwen3-8B-AWQ) | Publisher AWQ weights, FP16 execution and KV, TP1, conventional attention; explicitly disable thinking for this campaign |
| `ministral3_3b` | [Official checkpoint](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16) | Load the BF16 source with supported FP16 execution; verify finite outputs and text-only handling of its multimodal artifact |
| `qwen3_4b_2507` | [Official checkpoint](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | Non-thinking instruction model, FP16 if practical; predeclared quantization fallback below |

The RTX 2080 Ti requires FP16 and the configured pinned-memory path. Do not select
BF16, FP8 execution, FlashAttention 2, or an SM80-only quantization kernel merely
because a model card example uses it. Validate the actual selected kernel and
native library on this device. Do not upgrade the active environment in place.

The Ministral artifact contains a language model and vision encoder. Determine
whether the pinned runtime has a correct text-only loader/configuration and
whether it unnecessarily allocates encoder memory. Avoid ad hoc architecture-name
rewrites or deleting model weights. A documented runtime loader fix belongs in a
versioned patch and must pass the same correctness checks as any other repair.

For the two unquantized targets, if FP16 cannot fit the predeclared useful context
and cache envelope, prepare a reproducible 4-bit weight variant of the same model
before performance screening. Prefer publisher artifacts; otherwise record the
quantizer version, compatible SM75 kernel, exact source revision, calibration
inputs/settings, and resulting file hashes. Calibration cannot use confirmation
tasks. Check baseline answer quality and fresh/restored output behavior before
freezing the variant. Keep weight offloading disabled: its PCIe traffic would
confound the CPU KV mechanism. No model substitution or opportunistic precision
change between arms, cells in the same model profile, or machines.

For capacity planning, FP16 full-attention KV is
`layers * 2 * KV_heads * head_dim * 2 bytes` per token. The published configurations
give 144 KiB/token for both Qwen models and 104 KiB/token for Ministral's text
backbone. This is an estimate, not measured allocation. Weight quantization does
not quantize KV. Measure actual block bytes, engine workspace/startup peak, pinned
host allocation, CPU RSS/PSS where available, and swap activity. Use these receipts
to choose concurrency; published maximum context is not the target serving length.

For each model first prove, on tiny development inputs:

1. Correct loader, tokenizer/template, dtype, group geometry, and streaming output.
2. Store to actual CPU backing, GPU eviction, CPU restore, and subsequent decode.
3. Positive ready-key refresh in the wait arm when expected, and no speculative
   transfer or scoring path accidentally enabled.
4. A fresh reference with zero external hits recomputes the entire prompt; compare
   restored/fresh greedy token IDs on a deterministic functional fixture. Diagnose
   numerical nondeterminism rather than weakening the check silently.
5. Publication identifies actual observed token IDs and boundaries. Re-tokenizing
   assistant text or changing chat-template thinking handling must not fabricate
   prefix identity.
6. Invalid/expired advice, cancellation, drain, recovery, and owned cleanup.

Use greedy output equality for the functional fixture only. Pin sensible identical
generation settings within each performance comparison; ordinary workloads use
non-thinking mode and natural EOS with declared finite safety caps. Record every
length-capped response and its quality outcome.

## 6 Workload assembly

Freeze twelve settings. This table supplies starting development parameters;
Phase D resolves them using capacity/activity evidence before confirmation.
Do not automatically sweep their Cartesian product.

| ID | Workload and starting shape | Reuse, tools and evidence role |
| --- | --- | --- |
| B01 | Single conversation, six turns, one active request, initial history about 1–2k tokens; declared 1/5/15-second user pauses | Public text-chat inputs; no hints; measure request-active latency separately from user-paused wall time |
| B02 | 64 independent short QA/extraction tasks, roughly 256–1,024 input tokens and 32–128 output budget, four offered requests | No deliberate shared evidence prefix or hints; generic serving control, schema and factual checks |
| B03 | 16 one-pass document summaries, roughly 4–8k input tokens, output cap 512, concurrency two | No returning workflow or hints; expensive prefill and generation control |
| B04 | 48 document-QA requests over a fixed hot set plus distinct cold documents, three declared phases: hot / cold scan / hot return | No tool hints; preserve exact stable document prefixes where intended; report phase-specific reuse and latency |
| B05 | 16 repository investigations sequentially; four model requests and three real tools per workflow | Reuse source/search/test/evidence design; only one workflow active, valid neutral opportunity |
| B06 | 16 independent repository investigations, eight server sequences and four tool workers initially | Existing more-itertools family; long growing histories, real waits, target CPU/GPU cache competition |
| B07 | Exact B06 tasks, arrivals and generation settings with roomy cache | Change cache capacity only; retain no-eviction/low-opportunity result rather than force a benefit |
| B08 | 16 four-request PEP/document investigations, eight sequences/four tool workers initially, full CPU cache budget | Existing document search/parse/evidence tools; preserve previously useful regime |
| B09 | Exact B08 tasks/traffic with half its CPU cache bytes initially | Constrained counterpart; keep GPU budget unchanged to isolate CPU capacity unless explicitly labeled historical multi-resource reproduction |
| B10 | Four team workflows; coordinator dispatches three specialist roles then consumes their actual generated results and synthesizes | Frozen role graph and allowlisted real tools; eight server sequences initially; genuine fan-out/fan-in dependencies |
| B11 | 16 agent workflows plus 16 ordinary unhinted requests/workflows interleaved on the same endpoint | Fixed 50/50 assignment by offered unit, record actual token/resource share; no retention metadata on unhinted traffic |
| B12 | Exact B06 investigations and arrivals with one tool worker | Tool queue plus actual subprocess wait retained; known adverse contender comparison |

Counts are offered workloads, not always HTTP requests. Store the exact graph and
expected request/tool identities. B10's completed unit is a whole team; do not
count specialists as independent successful applications. B11 reports each class
and the common exposure, not a misleading aggregate of unequal workflow units.

Corpus construction requirements:

- Reuse versioned repository task definitions and `qualification_inputs.py`
  patterns. Acquire pinned public more-itertools and python/peps checkouts and
  retain their source identities/licenses. Do not require historical local clones
  or ignored `output/` artifacts to rebuild a new campaign.
- Use a small deterministic text-only sample from
  [WildChat](https://huggingface.co/datasets/allenai/WildChat-1M) for chat breadth.
  Preserve sample IDs/filtering, omit identity/location metadata, and choose
  follow-ups that remain coherent with generated rather than historical assistant
  answers. Validate coherence before speed measurement. Declare pauses as authored,
  not inferred production timing. An inaccessible source gets an explicitly
  recorded public replacement before freeze, not a synthetic substitute mislabeled
  as real traffic.
- Use selected [LongBench](https://github.com/THUDM/LongBench) QA/summarization
  inputs for B02–B04 where appropriate; pin the dataset/config/revision and original
  task references. Adapted serving tasks are not published LongBench scores.
  Author extraction checks against actual documents with independently parsed
  references. Include a reference for every objectively graded answer.
- Keep historical tasks as labeled reproduction/development inputs. Assemble a
  distinct confirmation task reserve not used for prompt or capacity tuning.
  Freshness is relative to this project's development history, not model training.
- A manageable frozen reserve may repeat across independent process repetitions.
  Declare `cohort_mode=fixed` and make the statistical claim conditional on that
  cohort. If using disjoint tasks per repetition, validate sufficient source tasks
  in advance. Reused tasks are never counted as new independent populations.
- Separate preparation/development tasks, confirmation tasks, quality references,
  and repair reserves. A performance-informed prompt or policy edit invalidates
  the affected confirmation cohort for fresh confirmation claims.
- Count tokens using each actual tokenizer. Choose intact source excerpts or
  explicit sections; do not silently truncate away answers, tool schemas, or
  dependency results. Keep task content matched across systems within a model.

Tool and graph execution requirements:

- Run live read-only `rg`, Python parsing, and bounded pytest commands. Each tool
  declares argv, working directory, valid exit codes, timeout, output limits, and
  a result checker. Use typed allowlisted arguments; never execute model-written
  shell. Record queue time and subprocess time separately.
- Stabilize irrelevant elapsed-time text before inclusion in model prompts while
  retaining exact original stdout/stderr and transformation identity in evidence.
- B10 uses real dependency results. The coordinator cannot receive cached canned
  specialist answers. Fixed role orchestration is legitimate; label it as such,
  not an autonomous agent choosing its own graph. A small model-directed routing
  diagnostic is optional and cannot expand the mandatory matrix.
- Application hints reflect currently known waits and observed prefixes. No hints
  for B01–B04 or B11's unhinted class. A coordinator may issue advice while awaiting
  known children, but cannot know their future completion times.
- For B04, a hot set means actual identical token prefixes; similar topics or
  reordered retrieval passages are not exact-prefix reuse. Distinct cold records
  create scan pressure without synthetic padding.
- Use workflow-dependent closed-loop submission for agents and declared external
  arrivals for independent traffic. Record offered-to-admission and server queues;
  do not omit queued or timed-out requests from latency/throughput denominators.

## 7 Systems and exact comparison matrix

Primary systems are `arc` and `wait`. Both use the same local runtime, model,
execution mode, GPU/CPU budgets, scheduling configuration, prompts, and arrival
rules. ARC must omit hint production/RPCs and plugin activation. Audit whether its
connector still loads extension machinery; disclose that and implement a real
native connector control where required by the overhead comparison. Shared
necessary runtime patches are allowed and recorded. Never call this a comparison
with untouched stock vLLM if both arms use patched runtime support.

Continuum is the published preview, with its real scheduler/LMCache backend and
`LMCACHE_SAVE_DECODE_CACHE=true`. Upstream means the pinned retention proposal
snapshot plus a disclosed compatibility patch, not the latest released vLLM.
Start from the existing `published.py` pins. Reconstruct environments independently
with their native dependencies. Keep clean published sources and a separate
reviewable execution patch chain. Feature/algorithm changes are not compatibility
repairs and require a new system identity.

Continuum's preview recognizes generated tool declarations and lacks the paper's
automatic timeout estimator. Give all four arms in a comparison the same frozen
tool-declaration instruction. Execute only the declared allowlisted real tool.
Verify actual Continuum pins and upstream retention updates; do not infer feature
activation from successful completions. For unhinted traffic, omit external
retention metadata as well as local advice. Missing unavailable native counters
remain null with an explanation, never fabricated zeros.

Use eager execution for this campaign's common comparison mode, preserving the
historical four-system constraint. Graph-enabled serving can be a later declared
study; do not add it as an unplanned matrix dimension. This does not qualify
default graph-mode performance. Keep common application continuation priority
matched and explain that its presence narrows the deployment claim.

| Model | ARC and wait | Continuum and upstream |
| --- | --- | --- |
| Qwen3 8B AWQ | B01–B12 | B06, B08, B09, B11, B12 |
| Ministral 3 3B | B02, B06, B08, B09, B11 | Not selected |
| Qwen3 4B Instruct | B02, B06, B08, B09, B11 | Not selected |

There are 54 system/cell configurations per repetition on each machine: 24 local
anchor configurations, 10 additional external configurations, and 20 secondary
model configurations. Four-system cells should be one balanced four-arm block,
sharing their ARC/wait observations with the primary comparison only when prompts,
source, cohorts, configuration, and measurement exposure match exactly. Do not
run or pool a second incompatible primary pair just to fill the report.

Use fresh server processes and equivalent warmup per measured arm. Balance two-arm
order across repetitions and four-arm position/predecessor order across blocks.
Generate the complete order before outcomes. Warmup should use separate inputs;
reset and prove an empty application cache or explicitly declare a matched warm
start. Never warm the next arm with another system's measured cache state.

## 8 Cache and host resource controls

Resolve byte budgets from actual geometry, not a model-independent number of
blocks. Match allocated GPU KV and CPU KV bytes across arms within a cell, allowing
only documented chunk rounding. Continuum's chunk size may differ; record unused
capacity and effective usable tokens. A declared config value is not an allocation
receipt. Verify pinned/shared backing on the actual transfer path.

Use development inputs and ARC capacity/activity measurements to select the base
pressure envelope. Start with approximately 2.25 GiB GPU KV and 2.625 GiB CPU KV
only if measured memory fit permits; these are historical starting points, not
universal requirements. Resolve common supported byte multiples for the contenders.
If necessary lower context/concurrency jointly before freezing, retaining enough
real context for useful tasks. Do not adjust capacities to maximize candidate wins.

B06 should exhibit evicted returning prefixes and actual CPU restore opportunities.
B07 changes capacity to remove or materially reduce competition. If the requested
roomy working set cannot fit, report that fact and choose a smaller matched B06/B07
development pair before freeze; do not silently change only the roomy workload.
B08/B09 retain identical tasks and a declared full/half CPU-byte contrast. Their
labels are relative resource regimes, not promises of a particular hit rate.

Watch host paging, tool memory, available RAM, GPU clocks/temperature, utilization,
and other device users. Do not modify swap or global host settings to improve a
result. Severe resource exhaustion is a development configuration defect; ordinary
repeatable paging within a frozen practical configuration is a measured covariate,
not a post-hoc exclusion reason. Preserve all slow cases and their telemetry.

On the 5080, first replay the same portable specification, model artifacts,
precision, context, tool limits, and explicit cache bytes. The new native build
must support its GPU and pass correctness. Hardware-specific kernel differences
are recorded, not concealed. A separately declared practical-capacity run may use
the larger device, but must not replace the matched replication or be silently
pooled with it. Differences between complete machines are not pure GPU scaling.

## 9 Measurement, quality and precision decisions

Primary metrics are completed workflow throughput and whole-workflow duration.
For independent tasks use completed task/request rate. Retain successful and
failed counts and time to termination; never calculate throughput over a surviving
subset with dropped failures. Record useful-task completion rate separately.

Secondary metrics include request-ready-to-first-token, tool-ready-to-first-token,
end-to-end request/workflow latency, inter-token latency, output token rate,
generation lengths/finish reasons, CPU/GPU memory, actual transfers, restored and
recomputed prompt work, evictions/preemptions, hint RPC/processing cost, and tool
queue/process duration. Startup and preparation are separate from serving. Report
B11 groups separately and B04 phases separately. A throughput win does not imply
a returning-latency or fairness win.

Quality checks are frozen before confirmation:

- B02/B04: reference answer/entity correctness and extraction schema validity.
- B03: required facts, unsupported assertions, and source-grounded summary rubric;
  an overlap metric alone is insufficient.
- B05–B09/B12: correct subject, actual test/parse outcome, at least one supported
  behavior/fact, and a concrete evidence-based limitation. Echoing PASS alone is
  not a useful investigation.
- B10: completion of child dependencies and a synthesis grounded in actual child
  evidence; identify dropped or contradicted results.
- B01/B11: instruction adherence, coherence, and applicable task-specific checks.

Run objective checks on every applicable output. Retain a deterministically
selected blinded review sample of up to 32 final answers per cell/system, covering
repetitions and traffic classes; review all automatic failures separately. Labels
must not expose the system or speed result. A judge model, if used, needs a fixed
identity/rubric and human/agent source audit of a sample; do not download another
large judge onto the serving GPU or silently treat judge scores as ground truth.
Do not discard poor-quality responses from the raw performance result. Report
raw speed plus valid/useful task outcomes and any quality regression. Fix objective
scorer bugs uniformly across all arms with versioned recomputation; subjective
rubric changes create a new evaluation version, preserving the original.

Use whole independent process runs as timing replicates, not requests as independent
samples. Pair runs by model/cell/cohort and planned block. Report per-run points,
geometric throughput ratios and paired 95% intervals on log ratios, with exact
denominators and interval method. Report time reduction separately; it is not the
same percentage as throughput gain. Do not pool workloads, machines, old campaigns,
or pilot and confirmation runs into a single win number. Mark intervals conditional
and not corrected for every comparison in the matrix.

Planning defaults below are agent-selected decision rules, not user-specified
service guarantees. Encode them in the locked protocol before confirmation:

- A practical throughput benefit has an estimate of at least 2% and a paired
  95% interval excluding zero. A stronger claim that the benefit exceeds 2%
  requires its lower bound to exceed 2%.
- Generic no-regression target: throughput ratio lower bound at least 0.97.
  Plugin-disabled/loading overhead target uses the same 3% margin. Failure to
  resolve that bound means inconclusive, not a proven regression.
- B11 unhinted mean-latency regression target: upper ratio bound at most 1.05;
  inspect and report tails regardless. A 10% p95 increase is a diagnostic alarm,
  not an automatic significance claim.
- All completed slower pairs are retained and surfaced. Large isolated changes,
  failures, quality regressions, and any source/activation mismatch trigger review.
- Tail intervals use whole-run resampling and show sample counts. With fewer than
  200 returning observations per reported group, treat p95 as descriptive and do
  not promise a tail guarantee. Do not add repetitions solely to rescue a tail win.

After one development screen per selected configuration, collect three independent
pilot blocks on representative classes: B02, B06, B09, B11/B12 for the anchor and
B06 for each secondary model. Use development tasks. Measure startup/case duration
and log-ratio standard deviation. Select confirmation repetitions from 6, 8, 10,
or 12 using a two-sided t interval planning half-width of approximately 2%, taking
the conservative applicable pilot variance for unpiloted cells. Four-arm blocks
must retain balanced order; prefer 8 or 12 there. Freeze the calculation, choices,
run order, expected wall time and disk budget in `protocol.lock.json` before the
first confirmation outcome. If 12 cannot achieve desired precision, state that
limitation in advance and run the frozen count. At completion report inconclusive
cells honestly; do not extend until a desired result appears.

## 10 Overhead and operational sequences

Keep these outside the 54-configuration throughput matrix and show their added
cost in the compiled plan. Run the overhead study on the anchor model using B02
and B01; use a short loaded/no-hints smoke on the secondary models.

| Control | Required behavior |
| --- | --- |
| O0 | Native ARC, extension/plugin unloaded, no hint production |
| O1 | Extension loaded, retention disabled, no hints; preserve native ARC behavior |
| O2 | Extension enabled, no hints; exercise idle endpoint/registry overhead |
| O3 | Extension enabled with valid hints on B06; active-feature reference |

O3 may reference an identical B06 confirmation result; it is not directly paired
against O0 on a different workload. Compare O0/O1/O2 on identical B01/B02 requests.
If needed add a B06 hints-produced-but-inactive diagnostic to isolate processing
from retention. Record actual imported modules, connector paths and background
activity; package installation alone is not evidence of plugin loading. Use a
fixed six-run balanced overhead confirmation after the pilot; report unresolved
3% bounds without run-until-pass extensions.

Operational coverage: one anchor ARC/wait sequence of at least six session waves,
keeping each server alive across waves, with new identities and a frozen bounded
mix of delayed, duplicate, expired, and abandoned advice. Add an isolated native
cancellation, prove its scheduler abort/drain receipt, then recovery generation.
Repeat a smaller cancellation/restore/recovery check on each secondary model.
Ensure unknown/adversarial hints cannot protect unrelated sessions. Exercise
configured registry limits in focused tests and a bounded development probe; do
not launch thousands of expensive workflows merely to reach a metadata limit.

Default completed-sequence acceptance is zero unexplained request/tool failures,
correct rejection/no-op behavior, restored fresh-reference correctness, no owned
processes/sockets after cleanup, and no persistent outstanding native transfers.
Record first/final and per-wave memory, registry sizes and peak resources. Use the
historical 512 MiB endpoint-growth threshold only as a diagnostic alarm; it is not
a proof of leak freedom. A reproducible operational bug must be repaired and the
affected sequence rerun before local completion.

## 11 Durable command interface to implement

The following campaign commands are the required new interface. They do not exist
yet. Implement and document them before executing the examples. Existing
`make bench-smoke`, `bench-screen`, `bench-confirm`, `bench-operational`,
`bench-tests`, and `bench-verify` remain supported.

```bash
PY=/home/philip/.local/share/venvs/vllm-semantic-cache-ext/bin/python
CAMPAIGN=benchmarks/serving_suite/campaigns/stage6.json
HOST_PROFILE=/absolute/private/path/host-2080ti.json
RUN_ROOT=output/stage6-2080ti-YYYYMMDD

# Resolve assets explicitly; no server/GPU work in prepare or compile.
"$PY" -m benchmarks.serving_suite.campaign prepare --campaign "$CAMPAIGN" --host "$HOST_PROFILE"
"$PY" -m benchmarks.serving_suite.campaign compile --campaign "$CAMPAIGN" --host "$HOST_PROFILE" --out "$RUN_ROOT"
"$PY" -m benchmarks.serving_suite.campaign run --out "$RUN_ROOT" --phase preflight
"$PY" -m benchmarks.serving_suite.campaign run --out "$RUN_ROOT" --phase screen
"$PY" -m benchmarks.serving_suite.campaign run --out "$RUN_ROOT" --phase pilot
"$PY" -m benchmarks.serving_suite.campaign freeze --out "$RUN_ROOT"
"$PY" -m benchmarks.serving_suite.campaign run --out "$RUN_ROOT" --phase confirm
"$PY" -m benchmarks.serving_suite.campaign run --out "$RUN_ROOT" --phase overhead
"$PY" -m benchmarks.serving_suite.campaign run --out "$RUN_ROOT" --phase operational
"$PY" -m benchmarks.serving_suite.campaign verify --out "$RUN_ROOT"
"$PY" -m benchmarks.serving_suite.campaign report --out "$RUN_ROOT"
"$PY" -m benchmarks.serving_suite.campaign export --out "$RUN_ROOT" --destination /absolute/path/stage6-transfer
```

`compile` prints the exact matrix, dependencies, unresolved capabilities, resource
budgets, and estimated cost; it must not say runnable while required bindings are
missing. Add `status --out`, `resume --out --phase`, and `--dry-run` for actions.
`run` refuses unresolved or unverified prerequisites. `freeze` validates corpus,
generation settings, source, repetitions and order; configuration changes after
freeze require a new lock/cohort with an explicit reason. Report/verify/export must
work offline without CUDA or loaded models. `--only model:cell` is useful for
development/repair but cannot silently mark omitted required cells complete.

Provide the corresponding import and recovery interface:

```bash
"$PY" -m benchmarks.serving_suite.campaign status --out "$RUN_ROOT"
"$PY" -m benchmarks.serving_suite.campaign resume --out "$RUN_ROOT" --phase confirm

# Later, on the 5080 host, using that host's local interpreter and bindings:
"$PY" -m benchmarks.serving_suite.campaign import \
  --bundle /absolute/path/stage6-transfer \
  --host /absolute/private/path/host-5080.json \
  --out output/stage6-5080-YYYYMMDD
```

Import validates the bundle before materialization, creates a new hardware-bound
campaign ID, preserves the portable protocol/task/order/repetition identities, and
links the parent 2080 Ti campaign. It never mutates original sealed artifacts.
Host path rebinding is not a workload change. Actual hardware/build differences
are explicit provenance; changes to model, precision, cache budget, prompts, or
policy create a separate comparison profile. Run the imported hardware preflight
before confirmation. Do not reselect workloads or repetitions based on 5080 wins.

Separate setup, compilation, preflight, screening, and confirmation. Screens
establish instrumentation, opportunity and runtime estimates; unfavorable or
noisy screens do not remove a selected cell. If current low-level APIs cannot
support a phase cleanly, refactor them and test the shared boundary rather than
generate bespoke shell scripts.

## 12 Ordered implementation and execution steps

### A Inspect and preserve

Read current status and inspect working tree, upstream divergence, active processes,
ports, locks, model caches, installed runtime environments, disk and host memory.
Check GPU state before claiming custody; stale status is not ownership evidence.
Reuse verified model/dependency caches. Do not delete or relocate active/frozen
`output/` dependencies. Capture the selected Stage 5 behavior and exact source
baseline. Update only the Stage 6 status entry with ownership and next step.

### B Implement the campaign contracts and coordinator

Add manifests, compilation, phase prerequisites, immutable attempts, resume,
offline verification and export/import. Integrate with `run_case` and existing
resource/transport helpers. Extract coupled functions only as needed, preserving
legacy suite commands and verification schemas. Add tests for rejected mismatched
identities, interrupted attempts, atomic writes, source mutation, missing artifacts,
wrong paired cohorts, duplicate/missing requests, unknown metrics, and path traversal.

GPU supervisors must use `harness.gpu_lock.gpu_reserved` for their entire invocation,
including idle gaps. A supervisor holds the primary lock while each child holds
the legacy lock, following the existing qualification coordinator pattern; a
standalone direct GPU task uses `GPULock()`. Do not invoke the standalone runner's
dual-lock path underneath an already-held primary reservation and deadlock.
Never unlink lock files or pass reservation descriptors to children. Add focused
lock-contention and cleanup tests without touching another process's ownership.

### C Assemble the assets and workload adapters

Resolve the three models, runtime catalogs, public corpora, real tool environments,
and frozen task definitions. Port historical corpus construction into maintained
adapters without historical filesystem dependencies. Implement graph execution
for B10 and per-class arrivals/metrics for B11. Test dependency ordering, streaming
errors, truncation detection, tool timeouts, cancel propagation, and unexpected
tool output. Use a fake HTTP server for deterministic runner tests, then real
smoke probes for claims that require native execution.

### D Prove compatibility and freeze resource regimes

Run the model functional gates, then anchor external-system probes. Audit the
matched native ARC baseline and explicit off switches. Obtain actual byte/pinned
allocation receipts. Resolve context, concurrency and cache envelopes using
development data. Generate a capability matrix before long jobs. Complete the
screen and variance/duration pilot. Fix routine loader/adapter/counter problems;
retain failed attempts. Freeze configurations, fresh confirmation corpus,
statistical rules and scheduled order only after these checks pass.

### E Acquire primary and supplementary confirmation

Execute the frozen model/cell schedule with incremental sealed attempts. Run all
selected cases even when some lose. Verify each completed attempt promptly to
catch missing telemetry before hundreds of jobs accumulate. Surface failures
with links to their logs and preserve progress. Never mix results across differing
source/configuration identities to fill a cell. Monitor resource-envelope
violations; diagnose without modifying policy based on confirmation outcomes.
Hash the executed source/input closure before each block. Unrelated documentation
edits are provenance updates, not serving-input changes; changed imported code,
native libraries, workload definitions or templates require a new source identity.

### F Complete overhead, quality and operational evidence

Execute the specified controls and persistent-server sequences. Finish objective
scoring and blinded source-grounded review. Investigate every material regression:
first check input/output work, activation, memory/paging, tool timing, queueing,
and restore/recompute evidence. Use separate short traced/replay diagnostics if
needed; never include their instrumentation overhead in primary timings. An
unexplained loss can remain a result with a bounded claim; corrupted measurement
or failed native correctness cannot.

### G Repair and rerun affected scope

Classify each failure as setup/compatibility, runner/instrumentation, workload,
correctness, resource envelope, or valid performance/quality outcome. Fix routine
defects, add a meaningful regression test, and create a new attempt/repair record.
If only a derived scorer/report was wrong, recompute every affected arm from sealed
raw evidence under a new analysis version. If serving behavior, prompts, timing,
resource settings, or executed inputs changed, rerun complete affected comparison
blocks; use fresh tasks where the edit was informed by confirmation outcomes.

A reproducible policy loss is not a runner bug. Do not restart Stage 5 optimization
or tune hidden budgets until it disappears. After three substantively similar
failed repair attempts, stop repeating the same action, write the concrete blocker
and alternatives, and continue independent cells. Escalate only when resolving it
requires a scope/model/policy change or unavailable user information. Do not
silently replace a required model/system or claim a fallback was the original.

### H Verify portability and build the 5080 handoff

On the 2080 Ti host, export the portable manifest, source/dependency/patch identities,
corpus/task/reference assets, checksums, host template and documented setup commands.
Include the pipeline source commit and any non-public artifact dependency inventory.
Large weights may be referenced by immutable downloadable artifacts; any locally
created quantization must be exported or reproducibly rebuildable from retained
calibration/tool inputs. Never depend solely on an ignored local directory.

Test import and dry compilation in a fresh temporary directory with different
path bindings; run offline verification/reporting with no original absolute paths
and no access to historical `output/` inputs. A read-only relocated copy of an
existing result should verify, or the format needs repair. Exercise a deliberately
interrupted development campaign and resume before declaring the pipeline durable.

The 5080 operator should only need to install the recorded runtime/build for that
GPU, bind local asset paths, import the frozen campaign, pass hardware preflight,
and execute the same phase commands. Clock-dependent hints currently require the
load generator and server on the same host; launch the harness locally on each
machine, rather than sending 2080 Ti monotonic timestamps across SSH. A wrapper
can invoke that local command through the configured Tailscale alias later.

Do not access or start jobs on the 5080 in this local milestone. Its hardware/build
compatibility remains unverified until its own preflight. Do not claim portability
testing on the local host is actual second-machine replication.

### I Publish technical evidence and clean up

Run offline verification and the appropriate source checks. Preserve useful caches,
raw attempts, repaired/failed evidence and all frozen dependencies. Reap owned
servers, bridges and tool process groups; close their ports, prove queues/transfers
drained, and release locks normally. Delete only recoverable disposable artifacts
owned by this campaign. Keep raw/large data ignored; commit reusable code, manifests,
tests, instructions and compact findings. Push coherent checkpoints and confirm
remote synchronization. Update STATUS with measured outcome, boundaries, resource
release and the exact 5080 next command. Move completed history into the existing
history document, preserving unrelated tasks.

## 13 Required tests and evidence layout

Run these existing checks with the patched runtime, extending the runner test
target to include new campaign tests:

```bash
PY=/home/philip/.local/share/venvs/vllm-semantic-cache-ext/bin/python
make runtime-check PY="$PY"
make test PY="$PY"
make bench-tests PY="$PY"
make check-light PY="$PY"
```

Do not substitute wheel-only tests for native-publication checks. Run affected
tests while developing and the required final checks once source settles. New
tests must cover meaningful failure modes, not merely mirror serialization code.
Required integration evidence includes a real GPU smoke on all three models,
positive retention and restore activity under pressure, all external selected
system probes, the four control states, operational receipts, an interrupt/resume
exercise, and offline verification after relocation.

Suggested raw layout:

```text
output/stage6-2080ti-<date>/
  campaign.json              # portable logical inputs
  resolved.json              # effective local plan; private bindings separated
  protocol.lock.json
  assets.json
  capability.json
  ledger.jsonl
  index.json
  attempts/<case>/<attempt>/ # inputs, request/tool events, logs, metrics, seal
  diagnostics/
  analysis/<version>/
  export/
```

Each attempt retains source/build/model/template identity, plan/order/cohort,
timestamped request and tool events, token counts and outputs, terminal reasons,
quality references/results, allocation and activation receipts, resource samples,
logs, cleanup evidence, and hashes. Use relative artifact references inside exports.
Do not compress/delete files still used by a running process or sealed verifier.

Commit compact evidence under `evidence/stage6-2080ti-<date>/`:

- `README.md`: outcome, supported configurations, limitations, reproduction link.
- `results.json` and `results.csv`: per-model/cell/system estimates, intervals,
  sample counts, quality and measurement classifications, raw evidence pointers.
- `compatibility.json`: exact observed model/runtime/device capabilities and gaps.
- `validation.json`: required tests, acquisition completeness, all-attempt and
  relocation verification, cleanup and synchronization proof references.
- `regressions.md`: adverse/inconclusive cells, diagnoses and unresolved causes.
- `quality.md`: rubric, complete automatic checks and reviewed sample findings.
- `handoff-5080.md`: setup, asset bindings, import/preflight/run commands, expected
  matrix, remaining hardware uncertainties and report-merge procedure.

Publish the export inventory and integrity hashes in tracked evidence. Ensure its
large payload has an explicit retained location and transfer/reacquisition recipe;
a Git push alone does not back up ignored raw artifacts. Do not upload results or
contact other people without separate authorization.

## 14 Completion checklist and final agent response

- [ ] Maintained catalog/campaign pipeline implemented with documented commands.
- [ ] Three selected models prepared, pinned, and proven correct on the 2080 Ti.
- [ ] Twelve anchor settings and five settings per secondary model assembled.
- [ ] Five supplementary anchor settings run with verified external activation.
- [ ] Confirmation completed under the frozen precision/order rules; losses and
      inconclusive estimates retained with no optional stopping for wins.
- [ ] Overhead, quality, fairness and operational evidence completed.
- [ ] Source/runner/native checks pass; all attempts and final analyses verify.
- [ ] Interrupt/resume and relocated offline reproduction demonstrated.
- [ ] Portable assets/manifests and exact 5080 execution instructions retained.
- [ ] Owned processes/ports/locks cleaned; useful evidence and caches preserved.
- [ ] Project status/history updated; coherent changes committed and pushed.

At handoff inspect `git status --short`, `git status --ignored --short`, and actual
upstream divergence. Report failed pushes or local-only work. The final response
should lead with the 2080 Ti outcome, link the maintained pipeline instructions
and compact findings, state resolved benefits/losses and remaining uncertainty,
give the precise 5080 entry point, and disclose any incomplete checklist item.
Do not call an incomplete local campaign finished merely because its scripts exist.

The release write-up remains pending until the 5080 evidence is integrated. The
technical report must already be sufficient to understand and reproduce every
local conclusion without reading the original chat.
