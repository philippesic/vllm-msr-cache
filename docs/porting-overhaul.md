# Agent first wait hint port plan

Proposed implementation plan, 9 November 2026. Reviewed in two rounds with
Cursor Opus 5.5 at high effort; amended 10 November after cross-project coverage
agreement with ShaderLOD and Binary Eagle. This is the proposed port plan for
`vllm-semantic-cache-ext-next`. It replaces the scope and sequence in
[remaining stages](remaining-stages.md); that document remains a historical
inventory. Implementation requires a subsequent task authorization. Approval of
a task authorizes its internal steps through its stated stopping conditions;
it does not require another permission request at every milestone.

## Conclusion and boundaries

Keep the foundation's reproducible tooling, import boundaries, strict types,
explicit unavailable states, and tested enforcement. Rebalance it toward a small
real runtime, behavioral tests, actionable diagnostics, and accurate capability
reporting. The current system can reject many undesirable spellings and layouts,
but it cannot yet establish correct wait-hint behavior or safe runtime operation.
It is a useful foundation, not yet an effective complete agent development system.

Port exact CPU offload with causal wait-hint CPU ARC recency advice. Semantic
ranking, summaries, GPU prefetch/staging and their experimental variants stay in
the old repository. Do not build optional placeholders for them. Use the same
constructed port with advice disabled as the behavioral control,
with identical correctness guards in both modes. Keep native ARC policy and
delegate native transfer and cache ownership to vLLM wherever the selected runtime actually satisfies the required contracts.

The deliverable is a minimal functional port with reproducible installation,
CPU integration tests, and an explicit later hardware qualification gate. This
plan specifies a future benchmark interface only. It chooses no workloads,
tiers, sample sizes, metrics, statistical methods, performance margins or winners.
The historical benchmark prescriptions are not requirements for the new runner.

User constraints are no code comments or commentary docstrings, one short root
AGENTS.md, no skills for now, explicit GPU authorization, and autonomous execution
within a defined task. Parallel implementation uses separate worktrees. Other
inherited preferences are defaults, subject to the concrete changes below.

## Audit evidence and its limits

The target was inspected at `99636b80a5b2ef2b8e4267dcc95e2c7f9e9809f3`.
Historical local source was inspected at
`1319f09bafac46bab89df465dd433c67441f4b26`; it was not changed.
The target's canonical CPU check passed during this audit: 151 tests, 9 passed
groups, 2 explicitly skipped groups, 0 failed, 43.13 seconds reported by the
validator. Native and runtime were the skipped groups. No serving test, GPU job,
container build or hosted CI run was performed. There is no target remote.

The source-local original plan was recovered from
`/home/philip/.t3/scratch/2026-11-09-read-only-analysis-and-planning-f5d203c3/PORTING_PLAN.md`.
Other `/workspace/research/` interview and migration sources cited by the old
plan were not recovered in this audit. The private Garden Repair canvas could
not be fetched. Its contents are therefore represented only by the user's
explanation and the existing [issue analysis](issue-analysis.md), not independently
verified as an original source.

### What works and what is missing

| Finding | Evidence in the target or historical source | Consequence |
| --- | --- | --- |
| Production boundaries are enforced | [Import contracts](../pyproject.toml), [boundary policy](../tools/repo_policy.py), positive and negative specimens | Preserve production versus benchmark separation and the vLLM adapter boundary. These address real coupling. |
| The CPU check is honest about absent runtime | [Check dispatcher](../tools/check.py), [runtime stub](../tools/runtime_verify.py) | Preserve explicit qualification levels; replace the stub with real checks as source arrives. |
| A lock prohibition rejects legitimate block names | `LOCK_PATTERN` in [repo policy](../tools/repo_policy.py) matches `gpu_block_pool` and `num_gpu_blocks_override` | Narrow policy to prohibited APIs/imports; ordinary vLLM vocabulary must pass positive controls. |
| Extension policy also bans unrelated stores named registry | [Agent policy](../tools/agent_policy.py) rejects `registry` modules and names ending `_REGISTRY` | Distinguish extension registration from ordinary session/prefix maps. Do not rename correct concepts merely to evade checks. |
| Strict adapter typing lacks an actual runtime environment | [Development dependencies](../pyproject.toml), [runtime assumptions](runtime.md) | Add a separately locked runtime CPU lane before extracting the adapter; do not compensate with broad missing-import ignores. |
| Proof execution dominates the recorded check time | [Prior validation](guardrail-validation.md) records 44.66 seconds full and 43.71 seconds proof-only; this audit's full check was 43.13 seconds | Add an honest targeted loop. Preserve full acceptance; do not infer a speed gain from these unrelated invocations. |
| No automatic acceptance is established | Optional hook; no remote; hosted CI unverified | A passing local command is not enforced branch protection. Establish local acceptance; hosted CI and PRs are not prerequisites while local-only. |
| Fixed collection catches deletion and skipping | [Pytest gate](../tools/pytest_gate.py) and manifests | Keep this guarantee; add explicit manifest-update and targeted-test workflows without silently rebaselining acceptance. |
| Identity examples are not the runtime identity | [Contracts](../semantic_offload/contracts.py) | Replace illustrative token/string equality and retention scores with actual namespace, observed-prefix, generation and native block-key contracts. |
| Planning prose already disagrees with code | `remaining-stages.md` advertises absent `./check` | Link commands to the checked system map and keep one active plan, not duplicated command inventories. |
| Ownership and secrets have narrow tripwires | Result filename checks; selected token/webhook patterns in repo policy | Treat these as partial detection, not proof of writer ownership or absence of credentials. |

### Historical code that changes the extraction strategy

The selected profile includes semantic and prefetch configuration fields, but its
active advice path refreshes ready CPU blocks within their existing ARC queues.
It does not run semantic scoring or stage GPU KV. See the immutable
[selected configuration](https://github.com/philippesic/vllm-msr-cache/blob/1319f09bafac46bab89df465dd433c67441f4b26/evidence/stage5-optimization-20261105/selected-candidate.json),
[manager](https://github.com/philippesic/vllm-msr-cache/blob/1319f09bafac46bab89df465dd433c67441f4b26/semantic_offload/manager.py)
and [connector](https://github.com/philippesic/vllm-msr-cache/blob/1319f09bafac46bab89df465dd433c67441f4b26/semantic_offload/connector.py).
The names and mixed configuration make misattribution easy. Some apparently
prefetch-specific switches are active advice gates: `prefetch_trigger`, nonzero
`prefetch_budget_fraction` and the global disable flag control advice execution;
`wait_hint_native_adoption` gates completed-transcript registration.
`instrumentation_only` with ARC selects the stock policy, and
`preservation_events` enables historical transfer-failure handling.
`wait_hint_shared_local_prefix` and `wait_hint_publish_at_lookup` have no reader
in the historical production package. Map active switches to an explicit advice
mode and required safety behavior; reject dead keys. Do not equate a misleading
name with an unused dependency. Test the translation and emit its resolved identity.

The recency operation is small; the full port is not just that operation.
Registration, computed-token boundaries, namespace and generation identity,
causal clocks, failure propagation and native integration determine correctness.
In particular, the historical connector's `_calc_num_offloadable_tokens` excludes
sampled tokens without computed KV and subtracts in-flight tokens on completion.
Do not delete that guard merely because the new worker delegates transfers to vLLM.
Every removed override needs an explicit disposition: irrelevant feature,
equivalent native behavior demonstrated, or required behavior reimplemented.

The native finished-request store path uses `req.num_tokens`; the computed-token
guard therefore protects both advice-enabled and advice-disabled modes. An
unloaded stock connector is a separate reference with its own defects, not the
matched control. Include this guard's upstream status in the pre-hardware runtime
decision.

The pinned offloading worker asserts successful submission/completion and does
not support transfer failure. Assertions disappearing under optimized Python can
turn failure into apparent completion. The default extraction retains a narrow
store/load subset of historical failure reporting, including failed-load block
IDs, scheduler cleanup and fencing, without the preservation-audit machinery.
Make failure outcomes explicit rather than assertions; reject optimized runtime
execution while required upstream safety still relies on assertions. Prove the
store/load subset with fake-handler fault cases and later device tests. If safe
recovery cannot be established within the authorized task, stop the affected
engine without publishing readiness, preserve evidence, and report recovery as
unavailable. A fail-stop-only product is a capability reduction requiring an
explicit scope decision, not a silently accepted minimal port.

The old CPU-recency tests import fixtures from the large prefetch suite. Use their
behavioral cases as characterization material with fresh small fixtures; do not
copy that scaffolding. The old hint schema also contains benchmark run/cell IDs
and a fixed declared-wait set. Separate product identity from experiment identity.
Replace run/cell IDs with an opaque product scope shared by registration and hint
lookup. Scope session names and generations by authenticated caller and that
product scope, so independent clients cannot supersede one another. Version the
per-request `kv_transfer_params` registration fields together with the hint API.
Specify a client-reproducible observed-prefix digest over exact token IDs and
namespace, with shared golden vectors. Keep namespace, generation, bounded inputs,
causal validation and replay behavior. A versioned contract change must list every
intentional semantic difference. Name the supported readers and schema versions
for hints, registration and verification receipts; refuse unknown versions and
producer/consumer disagreement. An intentional artifact migration creates a new
artifact with explicit source lineage and preserves the original. No automatic
legacy compatibility or schema-registry service is required.

Continuation request priority is a separate client/scheduler mechanism, not an
intrinsic consequence of wait hints. Historical clients invoke
[request priority](https://github.com/philippesic/vllm-msr-cache/blob/1319f09bafac46bab89df465dd433c67441f4b26/semantic_offload/request_priority.py).
Exclude custom continuation priority from the minimal runtime unless extraction
establishes a required functional dependency. Record it as absent in new receipts;
do not transfer historical gains to a system with different mechanisms.

## Research principles applied here

Cursor's first-party [Projects gardening section](https://cursor.com/blog/projects#gardening)
describes ongoing cleanup and converting repeated mistakes into lint rules.
The useful application here is to keep examples worth copying and remove obsolete
paths before they become templates. It is not evidence that more rules always
improve development or that this repository needs a fleet of agents.

Anthropic's [long-running harness account](https://www.anthropic.com/research/effective-harnesses-for-long-running-agents)
reports premature completion, incomplete handoffs, and the value of incremental
work with executable acceptance criteria. Apply those lessons with small commits,
real runtime checks and a compact task record. Do not copy its particular web-app
harness or introduce another project coordinator.

Its [effective agents guidance](https://www.anthropic.com/research/building-effective-agents)
argues for simple composable systems before frameworks. Here that means direct
ownership APIs and one supported launch path before a rules engine, plugin
framework, or generalized evidence service.

The requested Dune inspiration is used as a design hierarchy: eliminate an invalid
state or path in architecture first; enforce remaining boundaries with types and
checks; prove behavior in tests; use short instructions for irreducible judgment.
Public talk summaries were found, but no primary Dune implementation was inspected.
No claim of reproducing Dune or its reported productivity follows from this plan.

These are engineering recommendations, not measured productivity improvements in
this repo. Validate them through actual port tasks: can an agent install, locate a
change, run relevant tests, understand a failure, recover, and accurately state what
is qualified without asking Philip routine plumbing questions?

## Add remove and change

### Add

1. A locked runtime CPU environment and attested launch path, separate from the
   portable tooling environment. Record interpreter, dependency/wheel hashes,
   exact Python patch set and loaded files. Hide CUDA for CPU invocations and
   verify import/construction does not initialize it. This is accidental-use
   prevention, not a security sandbox.
2. Closed typed runtime configuration. Reject unknown keys, incompatible modes
   and unsupported layouts. Explicitly allowlist required native spec settings such
   as `cpu_bytes_to_use`, `eviction_policy` and `store_threshold`. Emit the resolved active mechanism identity from the
   same constructed components used for execution. Test that disabling advice
   changes both behavior and the emitted identity; configured does not mean used.
3. Behavioral contracts for observed prefixes, advice, real pinned ARC, inherited
   transfer boundaries, endpoint/RPC routing and cancellation/reset. Include
   invalid-state and failure-injection cases with bounded deterministic clocks.
4. A targeted CPU test command, useful failure diagnostics and explicit test
   inventory maintenance. Full acceptance remains the authoritative command.
5. A compact capability/verification record. Each finding names the system,
   mechanism, observation scope and limits. Create or refresh the dated alternatives
   record when a comparison decision is made; it is not a source-port dependency.
6. A documented task/resume contract using existing harness task state and Git,
   plus owned-process cleanup and atomic diagnostic output for runtime verification.

### Remove from the active port

1. Semantic scoring, summaries, query capture, prefetch, staging leases, GPU-aware
   bridging, admission experiments, SVD/MTP, custom affinity scheduling and their
   unused configuration, hooks and placeholder tests. Preserve historical evidence.
2. Mandatory LMCache/Continuum builds and all alternative adapter ports. Those are
   future research decisions, not prerequisites for this wait-hint runtime.
3. A requirement to create project-owned native sources or dual-architecture build
   machinery before a supported existing wheel has been assessed. Native binaries
   still need runtime identity and device qualification; no local kernel is implied.
4. Benchmark recipes and margins from the active plan, including inherited tier
   sizes. Keep historical results and their unfavorable or inconclusive outcomes.
5. Redundant live status narratives, generic plugin infrastructure, and placeholder
   semantic types that advertise capabilities the new product will not have.

### Change with concrete reasons

1. Use `wait_hint_offload` as the proposed internal package name, with wait-hint
   connector/schema names. Public repository naming can wait. Record old-to-new
   provenance once; avoid compatibility aliases unless an actual consumer needs one.
2. Narrow GPU-reservation and registry bans to the actual prohibited facilities.
   Keep reservation machinery outside the repo and suffix discovery as the default
   for a future real extension point. No extension loader is needed for one policy.
3. Keep all current proof coverage initially. After the fast lane exists, consolidate
   redundant third-party flag specimens only if wiring and regression coverage remain
   explicit. Do not cut a percentage of tests or introduce proof-cache invalidation
   machinery before a measured need. Project-authored rules retain good/bad specimens.
4. Keep fixed full-suite manifests. Provide an explicit update operation that prints
   additions, removals and renames; never update as part of validation. Ordinary test
   additions are autonomous. Deleting required behavior needs justification and review.
5. Separate portable full checks, runtime CPU checks and explicit hardware checks
   within one documented check interface. A portable pass cannot label the source
   port accepted. No selected required test can silently skip or xfail.
6. Human review applies to weakening guarantees or adding substantive exceptions,
   not every new test or mechanical manifest edit. Propose the exact patch and
   coverage impact; continue independent authorized work while review is pending.
7. Replace raw result-path string checks as the primary ownership guarantee with an
   injected result sink owned by the verification/benchmark caller. Library code
   emits typed events and returns outcomes; it does not choose durable result paths.
8. Broaden credential checks around the actual custom-webhook leak class using
   sanitized specimens and exact exceptions for legitimate links. Prefer known
   credential-bearing URL patterns and redacted diagnostics over a blanket ban on
   long URLs. Do not claim exhaustive secret detection or historical revocation.

## Runtime architecture and ownership

Create modules as the corresponding slice is extracted, not as empty scaffolding.
The proposed dependency direction is adapter to application logic to contracts.
Pure modules cannot import vLLM, benchmarks, tooling, filesystem output writers,
or host environment configuration. Avoid shared-state mixins and a replacement
mega-connector. Ownership is per authoritative state or result, not all IO:
read-only diagnostics may inspect state and temporary writers use owned roots.
Each adapter override lists its required native seam in the runtime
compatibility manifest and has a focused integration test.

| Responsibility | Owns | Allowed dependencies and boundaries |
| --- | --- | --- |
| `config` | Validated settings and resolved mechanism description | One environment translation at entry; no dynamic hidden defaults scattered through execution |
| `hints` | Versioned namespace, session, generation, prefix and event identities; outcomes | Pure parsing and validation; no experiment run IDs or future-demand labels |
| `observed` | Bounded observed-prefix/session state, generation supersession, replay records | One scheduler-side writer; immutable views; explicit reset, retirement and capacity outcomes |
| `advice` | Eligibility decisions from hint, observation, time and resident view | Returns an ordered eligible prefix; no allocation, transfer, pinning or scheduler priority |
| `vllm_adapter` | Construction, endpoint/RPC mapping, exact native keys, ARC queue refresh and native callback adaptation | Only vLLM import site; serialized with scheduler ownership; no second copy of native refcounts or transfer-job state |
| Observation | Typed bounded outcomes and counters produced at actual transitions | No raw prompts by default, no synchronous file output in the hot path; unavailable differs from zero |
| Verification entry | Environment attestation, subprocess lifetime and result output | Explicit output root and timeout; never hidden inside imports or ordinary CPU tests |

Native vLLM remains the owner of CPU/GPU storage, refcounts, transfer jobs and
completion fences unless a specific defect requires a narrow correction. Advice
reorders eligible ready residents within their existing ARC queue, with the root
block most recent. It must not
promote T1 to T2, alter ghost history or adaptive target, add references, make a
pending block ready, allocate GPU blocks, or submit transfers. Ready blocks already
referenced by a demand retain exactly those references. Missing/pending prefix
blocks stop eligibility; later keys cannot be treated as a reachable prefix.

Observation of completed transcripts registers only computed complete block
boundaries. Native key geometry and digest construction are checked at the adapter;
string equality in a frozen dataclass cannot replace those checks. Namespace,
adapter, model/tokenizer revision, layout and generation are validated before
advice. Cross-process monotonic timestamps require a declared shared clock domain;
reject unsupported clock domains rather than comparing arbitrary client clocks.

The product endpoint delegates to scheduler-owned logic. It cannot mutate ARC
from the frontend thread/process. The historical plugin route sits outside the
pinned server authentication
middleware's protected path prefixes. Integrate the new route with server
authentication and enforce caller scope for both registration and hint lookup.
A server-wide shared API key establishes only a shared trust domain, not separate
tenants. Do not claim tenant isolation without distinct authenticated identities.
Permit unauthenticated local development only under an explicit loopback-only
mode; refuse unauthenticated non-loopback binding. Test rejected unauthenticated
calls and scope mismatch. Residency acknowledgements must not expose another
caller's scoped observations.

Limit payload size, prefix lengths, session capacity and replay history. Retire
an inactive session's observations, generation and replay state together with a
counted `session_retired` outcome. A hint for retired state finds no observation
and does nothing. Use a new session incarnation on re-registration so delayed
old hints cannot become valid when a session name is reused. Test full-table
retirement, delayed hint/registration delivery, re-registration and restart.
If safe retirement is unavailable, bounded no-action is preferable to partial
state deletion, but capacity cannot permanently disable all new sessions without
an explicit reset/recovery path. These guards protect advice identity and
priority; exact native key matching remains the KV correctness boundary.

## Runtime selection and upstream drift

Start characterization against the historical exact native vLLM pin, while treating
the new minimal port as a new system identity. Use one supported runtime closure
for the new port; preserve root-development and Stage 6 closures only as historical
provenance. Prefer an exact wheel plus a minimal attested Python patch/overlay.
Keep a hash-verified copy of the nightly wheel outside Git so installation does
not depend indefinitely on nightly hosting. Do not claim that this removes native
dependencies or certifies either GPU.

Initially extract only required hint RPC plumbing from `scheduler.py`, `core.py`,
`core_client.py` and `async_llm.py`. Characterize and exclude unused block-pool
publication, allocation epochs, lookup hook and idle-drain changes; do not carry
benchmark abort-receipt instrumentation by default. The private-seam inventory
includes `manager._policy`, `t1`/`t2`, `is_ready` and `move_to_end`, and is rechecked
on every repin. Preserve upstream license notices for extracted material.

The audit found an upstream [ARC fallback fix](https://github.com/vllm-project/vllm/pull/55461),
merged 6 September 2026, after the historical pin. The local pinned `evict()` can
return failure when T2 has no eligible candidate without retrying eligible T1
entries below the adaptive target. Characterize this case on real pinned ARC.
Its occurrence or effect in historical measurements is unknown; shared code in
both arms does not establish that its interaction with advice was neutral.

Do not silently fix or repin during extraction. Establish behavior on the old pin,
then make an explicit runtime decision before hardware qualification: retain the
bounded known limitation, backport the specific fix to both modes with a new
identity, or qualify a newer pin. This decision need not block pure source work.

Dated alternatives evidence must distinguish code existence, open/merged state,
tested compatibility and measured results. On 9 November 2026, public pages show
[session-aware CPU eviction](https://github.com/vllm-project/vllm/pull/50422),
[priority retention](https://github.com/vllm-project/vllm/pull/38514), and
[programmatic retention](https://github.com/vllm-project/vllm/pull/59714).
Their existence is relevant; none is thereby qualified in this repo. Record exact
revisions and check dates when a comparison decision is made, and refresh then.
Do not gate deterministic CPU checks on the network or calendar. Do not call an
alternative undeveloped based on old notes or infer effectiveness from a PR title.

## Functional acceptance and feedback

The following are correctness checks, not a performance study. Each assertion
must fail for the relevant defect, not just verify that a method was called.

| Slice | Required observable behavior |
| --- | --- |
| Configuration | Reject unknown/dead historical keys, bad types, unsupported layout/group/parallelism; one selected constructor; no hidden scoring or prefetch |
| Identity and observation | Namespace mismatch, stale generation, unobserved prefix, partial block, future or incompatible clock, replay, bounded capacity, reset and restart handled explicitly |
| Real ARC | Same queue membership, ghost lists, adaptive target, references and eviction eligibility; root most recent in its own queue; stops at a prefix hole; advice-disabled mode performs no advice mutation |
| Native integration | Actual pinned imports and construction; missing/wrong RPC method rejected; hint reaches scheduler owner; endpoint errors preserve structured reasons |
| Store and load lifecycle | CPU fake-handler cases: computed full-block extent, exact key/slot/order mapping, no readiness before completion, failed submission/load/store and cancel/reset cannot publish invalid state. Device fences and real worker construction remain M5 gates |
| Observation | Receipt names active mechanism and actual transitions; absent counters unavailable; telemetry disable does not change policy outcomes |
| Operational faults | Owned child timeout, failed startup, interruption, write failure and restart preserve incomplete evidence and clean only owned resources |
| Later GPU qualification | Real FP16 store/evict/restore with bytes checked, correct failure fences and restore-versus-fresh functional behavior on each explicitly authorized supported host |

Use fake clocks and small deterministic fakes for pure logic and failure injection.
With CUDA hidden, exercise scheduler-side connector/manager/ARC construction and
endpoint/RPC routing with CPU fixtures. The real transfer worker uses CUDA streams
and host registration, so CPU failure-path tests use an injected fake handler;
they cannot qualify real worker construction. Use real pinned ARC and real adapter
imports for integration; mocked imports alone cannot qualify a private API seam. Fresh fixtures must not import old prefetch
modules. Exhaustively enumerate small relevant state transitions before adopting
a general property-testing framework. Each removed historical safety override
needs a test showing its invariant is still enforced.

Derive oracle expectations independently of production implementation: manually
work small ARC order/refcount cases, retain historical failure counterexamples,
and use independently computed digest/block-boundary vectors across registration,
endpoint and scheduler consumers. A second agent or two consumers generated from
the same schema is not independent derivation. Include a deliberately disagreeing
producer/consumer pair and unknown schema version; both must refuse. Keep required
safety checks effective under optimized execution or reject that execution mode.

The supported check interface should offer a documented targeted mode for a test
path or selector, a portable full mode, and a runtime CPU mode. Targeted results
must say partial and name omitted groups; their receipt cannot satisfy full
acceptance. Run the full applicable lanes before committing an accepted source
milestone. Ordinary code changes should not rebuild environments or rerun every
policy proof after every edit. Keep the optional full CPU hook; provide one setup
instruction and report whether it is actually enabled.

Diagnostics should name the rule or failed invariant, path, observed condition,
and the supported repair command. Bound subprocesses and print timing by group.
Optimize only observed slow steps; do not set an arbitrary tool-count or line-count
target. Native and hardware absence remain visible outside passed counts.
If incremental builds or CPU checks reuse cached work, report executed and reused
checks separately with the identities that justify reuse. A cached CPU result is
not a fresh hardware observation. Never memoize physical performance measurements
or substitute historical device qualification for a requested new device run.
This adds reporting obligations, not a requirement to build a proof cache.

## Accurate communication and recoverable operation

Record the actual server scheduling policy alongside the resolved advice mode.
One capability record should separate implemented, CPU-tested, GPU-qualified and
performance-evaluated states, bound to source/runtime identity. Derive executable
capability fields from construction and test receipts; do not maintain a second
manually edited success list. The system map remains navigation, not evidence of
behavior. A durable finding references the relevant receipt and states scope.
A schema can prevent missing attribution; it cannot establish the truth of prose.
Source review remains necessary for consequential interpretations.

Keep authority, readiness, evidence validity, selection and scientific usefulness
separate. Authorization does not make stale or corrupt evidence valid; valid
checks do not grant execution permission. An editable approval field or content
hash authenticates neither authority nor policy approval. A selected result is
not necessarily the latest attempt, and a healthy process may yield no useful
scientific observation. Report causes as confirmed, hypothesis or unknown.

A useful completion message says what changed, which mechanism was exercised,
which checks passed, what remains unavailable, and where evidence lives. Examples:
“CPU ARC recency-advice cases passed; no GPU transfer was tested” is acceptable.
“Semantic caching improved” is not supported by that result. Historical performance
is never relabeled as a result for the minimal port. Do not use test counts as a
proxy for functional coverage or output token counts as a proxy for answer quality.

An authorized task records objective, scope, acceptance conditions, resource
permission, and stop conditions in the existing harness. Admission and terminal
reports, both human-readable and machine-readable, identify the goal, actual
mechanism activation (or not-yet-observed state), starting code/runtime identity,
limiting risk, resource budget, stop action and evidence scope. Do not imply a
functional verification run establishes performance or scientific progress.

Git holds code, tests, compact decisions and accepted evidence. At a handoff keep one bounded resume
record in the task with current branch/commit, pending checks and known blockers.
If the harness cannot retain that state, retain one compact tracked task record,
with outcomes linked rather than copied. Do not create competing STATUS/history
ledgers or ban all resumable state in the name of banning diaries.

Proceed autonomously through reversible repairs, tests and authorized milestones.
Ask only for a missing consequential decision, a scope expansion, guardrail
weakening, or resources not authorized. Continue independent work while waiting.
A failed attempt is evidence to diagnose, not a stop condition or a reason to
retry selectively until results look favorable. The execution owner enforces
wall-time/resource bounds, stop action, retry count and cumulative failed-attempt
cost even when the agent exits or notifications fail. Use supplied task limits;
choose explicit bounded operational defaults within authorized scope when needed,
not undeclared scientific stopping rules. Persist budget accounting across a
restart. Test lost notifications, exhausted retries and timeout cleanup in M4.
Recovery within these bounds remains autonomous; changed semantics or a reduced
capability must be reported and resolved through the stated scope process.

Stop/cancel/retry controls bind to the current run, attempt and owner through the
existing harness or local access boundary. Reject foreign, stale and replayed
control requests; test delayed cancellation after restart. Authenticate access
without inventing a new credential service, and do not confuse access with Philip's
authorization or tenant isolation. CPU fault injection proves process-crash handling,
not power-loss durability; claim the latter only after separate relevant evidence.

For concurrent implementation, one owner per responsibility and one worktree per
agent. Integrate serially, regenerate the full test manifest with the explicit update
command, review its diff, and run acceptance on the combined tree. Worktrees
separate working files and indexes, but do not isolate GPUs, ports, environments,
shared caches or artifacts; see [Git worktree documentation](https://git-scm.com/docs/git-worktree).
Use per-invocation output roots and ownership records. Do not mutate an environment
or delete an artifact another process is using. Remove only clean disposable
worktrees and owned processes after preserving results. Bind running attempts to
immutable snapshots including dirty tracked content, untracked/generated executable
inputs, and actual loaded libraries/binaries, not HEAD alone. Refuse or invalidate
relevant qualification on identity drift; changed inputs start a new attempt.
Identify owned processes by host/boot, process birth identity and attempt, not PID
alone; test PID reuse and refusal to kill foreign processes.

Cleanup first classifies the root as disposable task scratch, reusable cache,
authored input or retained evidence. Recheck ownership, active users, dependency
references and recoverability immediately before reclaim; record a deletion
receipt outside the deleted root. Unknown or shared ownership refuses deletion.
Keep useful failed/negative evidence and dependencies of retained results. Do not
add an autonomous deletion service, arbitrary shared-cache cleanup or automatic
scientific evidence expiry. This is a narrow owned-resource cleanup contract.

GPU permission is necessary but not an exclusion mechanism. A CLI flag does not
prove authorization; an idle-device observation does not prevent a race. Keep
reservation protocols out of this repo as requested, but require an externally
established exclusive execution window/owner for GPU qualification. Authorization
names host, device and window, plus either an existing external reservation wrapper
(not imported into this repo) or Philip's confirmation that no other work uses that
device in the window. The idle-device check remains a sanity precondition. If no such
arrangement exists, the GPU step remains blocked; CPU work continues. Any later
shared autonomous GPU service requires a separately approved external ownership
contract, not a home-grown lock disguised as a preflight. Participating external
owners must agree on a host/device namespace across projects. An advisory mechanism
excludes only cooperating users; ambient or uncooperative occupancy remains a
reason to refuse or report uncertainty. Recheck disk headroom before execution,
but do not describe that observation as OS-reserved capacity. No scheduler, queue
or lease service is mandated.

## Benchmark interface only

Keep `python -m benchmarks` and plan/run/verify/report as reserved interface names,
not implementation requirements for the source port. No runner or extension
loader is needed merely to claim the interface exists. Mark absent commands as
unavailable until implemented and exercised.

The future interface accepts a versioned immutable execution description containing
source/runtime/config identities, resolved mechanisms, explicit resource limits,
seed/config inputs and external artifact location. Product hints do not depend on
benchmark plan IDs. The benchmark caller may attach experiment identity outside
the runtime contract.

A run has a unique identity and explicit created/running/failed/interrupted/complete
state. One caller-owned writer records events and atomically commits its terminal
record. Reporting derives views without rewriting the attempt. Verification checks
schema, input/artifact identities, required observations and lifecycle completeness;
a complete run is not automatically a valid comparison or a research win. Preserve
failed attempts and unavailable measurements. Offline relocation must preserve
verification through relative artifact references and content identities.

Only lifecycle, provenance, structured outcomes and ownership contracts are fixed
here. Workloads, tiers, performance counters, statistical units, comparison rules,
quality evaluation and acceptance margins await the later benchmark-design task.
Functional runtime verification can produce a small independent correctness receipt;
it need not depend on a research runner that does not yet exist.

## Implementation sequence and stopping conditions

Each milestone is a coherent commit with acceptance evidence, not an automatic
human approval pause. The executing task's scope determines how far to proceed.

| Milestone | Work | Exit evidence |
| --- | --- | --- |
| M0 Reconcile foundation | Adopt active scope and reconcile the decision clauses listed below; repair overbroad bans with controls; define targeted/full lanes and review boundary; remove misleading future requirements from active docs | Existing full CPU checks pass; legitimate vLLM identifiers pass; prohibited APIs still fail; documented commands exist or explicitly remain unavailable |
| M1 Bind runtime | Lock one runtime closure and exact minimal patch; prove portable/runtime environment separation; characterize ARC caveat and native seams | Clean CPU install/import and scheduler-side construction with CUDA hidden; wrong interpreter/wheel/patch rejected; loaded and dirty/generated identities retained; drift rejection exercised |
| M2 Extract pure behavior | Closed config, identities, observed store, advice; rename internal package and schema; record intentional historical differences | Causality, replay, capacity, generation and prefix cases pass without vLLM; no scoring/prefetch configuration accepted |
| M3 Integrate native behavior | Thin adapter/RPC/authenticated endpoint; advice-disabled ARC control; computed-token and explicit failure-path guards; actual mechanism receipts | Real pinned ARC/adapter CPU cases and per-override disposition pass; independent small oracles and schema/conformance refusal pass; no copied giant fixtures or semantic dependency |
| M4 Qualify minimal source port | Clean reproducible setup, all required CPU lanes, operational fault cases, readable evidence, source/interface docs | Complete CPU functional acceptance including lost notification, retry budget, stale control, PID reuse and safe-cleanup refusal cases; capabilities explicitly say GPU and performance unqualified; benchmark interface reserved only |
| M5 Qualify hardware when authorized | External ownership, supported model/layout/host, actual store/evict/restore and cleanup | External ownership and occupancy/headroom sanity recorded; per-host functional receipt; failure preserved; no implied throughput or quality result |
| M6 Later research and preferences | Separate benchmark redesign; optional skills/interface work only on request | Acceptance criteria defined in those later tasks, not invented here |

M0 must reconcile authoritative documents in the same implementation change,
not leave this plan as an exception agents must discover. Mark superseded clauses
in decisions 001 (namespace and candidate scope), 003 (active versus historical
runtime closures), 004 and G005-OWNERSHIP (benchmark versus functional-verifier
writers), G005-IDENTITY (illustrative contracts), and G005 review policy. Rewrite
AGENTS.md within its existing size cap around the wait-hint question, exact reuse,
canonical check navigation, substantive-review boundary, GPU authorization and
worktrees. Move detailed benchmark guidance into its domain document. Keep
historical decisions/evidence clearly labeled; do not erase provenance. Assign one
current owner/location for each design or workflow rule, with supersession links
from retired guidance. Keep immutable scoped findings and adverse outcomes in the
existing FINDINGS index; any additional view derives from those records rather
than duplicating conclusions. A historical record must not remain an active
instruction merely because it is retained.

Seed the per-override disposition during M1: reimplement computed-token bounds,
completed-transcript registration, hint publication and reset state clearing;
extract narrow transfer-failure reporting; exclude scoring, prefetch and detailed
preservation instrumentation; demonstrate admission-disabled manager mixins are
irrelevant before removing them. This is a starting inventory, not a claim that
all callback dependencies have already been enumerated.

M1 and M2 can overlap only after shared contracts are settled; do not multiply
agents for its own sake. M3 waits for real runtime binding. M4 must not call the
project operationally complete because CPU checks pass. GPU evidence belongs to
M5, and performance evidence belongs to later research.

Before source implementation, approve the proposed policy changes as one concrete
scope. Public naming, license/publication, remote hosting, artifact backup and CI
runtime budget need decisions only when they affect the next action. Use existing
local resources for CPU work; no public release or remote creation is implicit.
Runtime pin/fix selection is due before M5. Benchmark choices do not block M0–M4.

## Debate disposition

Two Cursor Opus 5.5 high-effort reviews reached broad agreement on the additions,
removals and changes. The second review accepted the parent's rejection of line
count promises, percentage-based test cuts, premature proof caching, blanket URL
heuristics and idle-device checks as ownership guarantees. It identified active
historical switches, stock transfer failures, matched-control safety, product
session scoping, endpoint authentication, decision conflicts, CPU construction
limits and bounded session retirement; these corrections are incorporated.

One explicit judgment remains: the reviewer preferred fail-stop transfer handling
as the minimal default. This plan instead targets narrow recovery because the
user prioritizes recoverable operation and the historical selected path supplied
it. Both sides accepted either choice if its capability consequences are stated.
Fail-stop is a safe emergency behavior, not advertised recovery. No further debate
is needed to execute the plan; implementation evidence may justify revisiting the
choice through the stated scope process.

The 10 November cross-project agreement adds focused identity, authority, budget,
oracle, schema-evolution, control-channel and cleanup acceptance cases to M1–M5.
It does not adopt a common framework, change local full-check/manifest or discovery
preferences, add skills, import GPU reservation machinery, require hosted CI, or
specify benchmark methodology. Architecture and typed APIs remain the first line
of enforcement, followed by behavior checks and concise guidance.
