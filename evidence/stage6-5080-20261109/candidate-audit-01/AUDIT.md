# Wait-hint candidate: structural and operational audit

Current overall performance assessment across the 2080 Ti/5080 campaigns:
**inconclusive**, at Philip's request. [Assessment](assessment.json). See the [visual summary](chart.html).
Original cell estimates, adverse results, capability outcomes and the three 5080
activity exclusions remain unchanged. This audit does not reclassify them.

The candidate is not simply failing to receive hints. The audit finds real
operational/design weaknesses, two reproducible API robustness gaps, and limits
in what the existing benchmark checks can establish. It does not find evidence
of wrong-prefix KV reuse, GPU ownership leaks or an accidentally disabled hint
path in the retained supported cases.

## Scope and evidence

- Read-only audit of 269 retained 5080 candidate/control measurements across
  preflight, screen, confirmation, overhead and operational phases. Every result
  consumed here matches its retained attempt seal. The 207 selected candidate
  confirmation cases remain the original analysis set.
- Existing 2080 Ti primary, addendum, qualification and operational records are read
  locally. Its full raw tool/request histories are not copied to this machine;
  no 2080 Ti machine was contacted.
- 125 existing CPU checks passed for hint contracts, recency updates, endpoint
  behavior, audit/registry mode and GPU-state gating. Additional CPU-only probes
  exercise lifecycle exhaustion, prefix eligibility, conflicting retry identity,
  expiry semantics, unsupported wait estimates and malformed JSON.
- No model inference, GPU workload, real repository-tool benchmark rerun, candidate
  implementation edit, tuning or new serving-performance experiment was performed.
  GPU visibility was disabled for CPU probes. Campaign reports/ledgers remain unchanged.
- [Hint-flow evidence](hint-flow.json), [prefix-extent evidence](prefix-coverage.json),
  [CPU structural probes](cpu-probes.json), [API input probe](cpu-api-input-probe-02.json)
  and [machine-readable findings](findings.json) retain exact records and scope.

## 1. Valid but wrong advice can hurt unrelated traffic

**Observed design risk; highest operational priority.** The 2080 Ti addendum's
misleading-advice arm increases unhinted requests' mean time to first token by
14.6%, with 95% interval 10.7–18.6%, versus ordinary ARC. All four prespecified
paired blocks have actual CPU pressure and positive recency touches for abandoned
sessions that never return. Aggregate throughput is approximately unchanged:
ratio 1.0032 [0.9953, 1.0113]. Aggregate completion rate conceals interference.

These are valid observations of real cached prefixes, not malformed/future-leaking
input. The experiment changes the application's predictions while preserving the
selected candidate. Incorrect predictions and abandoned/cancelled workflows are
therefore relevant operational cases, even though the exact stress-cohort penalty
is not a general production latency estimate.

The selected CPU-only branch refreshes every reachable ready prefix key on each
fresh hint. It has no selected per-session/tenant recency budget, utility test,
cooldown or feedback disabling unproductive advice. Optional ARC-advice controls
exist, but were not enabled in these campaigns. The nonzero 0.6 prefetch-budget
setting enables this branch; it does not limit CPU-recency touches to 60%.
GPU-state gating and suffix bridging were also disabled in the measured candidate.

The measured harm is a system-level advice effect. The saved records do not
isolate cache displacement from scheduler/API processing costs. Either way, there
is no selected isolation/cost bound that prevents valid wrong advice from hurting
other traffic. Validation of namespace, generation and causality does not validate
the usefulness of an application's forecast.

Expiry is not rollback: a CPU-only hint creates no retention lease. It changes
ordinary ARC recency. Later expiry rejects a new stale submission but does not
undo previous queue reordering. The CPU probe confirms this; it is an intended
semantic distinction, not a stranded GPU reservation.

**Direction to investigate before relying on it:** bound shared-cache advice cost
and interference, add utility/consumption feedback and evaluate degradation under
wrong/abandoned hints. Do not mistake input validation for prediction reliability.

## 2. New-session advice eventually shuts off in bounded operating mode

**Confirmed lifecycle limitation, already intentional/documented—not a newly
observed Stage 6 crash.** With `preservation_audit=false`, the default registry
retains 4096 distinct session identities with generation tombstones for the server's lifetime.
Finishing a request/session does not retire the tombstone. Once full, new sessions
receive `session_registry_capacity` until `reset_cache`/restart. Ordinary LLM
serving continues; the optimization silently degrades unless the caller monitors
the acknowledgement reason.

A native CPU-metadata probe sets the limit to two, finishes both requests, then
submits a third session. There are zero active requests, but the third hint is
refused because the two lifetime identities remain. Existing tests intentionally
verify this behavior to prevent delayed old generations from being resurrected.
Deleting tombstones without a replay/epoch design would remove a safety property.

With `preservation_audit=true`, there is no session limit. That research profile
retains history instead. All three 5080 candidate operational cases use this audit
profile: their actual session limits are null and their maxima are 96/32/32. The
separate preflight registry probe tests bounded fail-closed behavior with limit 2;
it does not prove session recycling or indefinite feature availability.

**Direction:** define a complete bounded session/epoch/replay lifecycle and expose
advice-degradation counters. Do not claim a production soak from the finite audit
profile, or remove stale-generation protection as a quick fix.

## 3. The hint path runs, but positive touches do not prove useful retention

**Observed activation; selected-policy blind spot demonstrated separately.**
Across the 207 selected 5080 candidate confirmation cases, there are 18,929 recorded
hint submissions. 18,914 report positive ready-key refresh counts; 15 report
`observed_prefix_unavailable`. There are zero recorded delivery errors, mismatched
event acknowledgements or positive refresh decisions after their nomination deadlines.
Every hint-producing session in the selected set has at least one positive
refresh. The no-hint chat/QA/control cells are intentionally unhinted.

The returned status is `no_action` because the CPU-only mode does no speculative
GPU transfer. A positive `cpu_prefix_recency_refreshed` reason means that the CPU
manager's refresh method ran; `hint_accepted=0` is not evidence that this mode was
disabled. Counts repeat keys and do not establish unique bytes kept, a queue-order
change on every repeated touch, useful later consumption or avoided computation.

There is a large gap between activation and extent on document workloads:

| 5080 selected workload | Hint submissions | Positive ready-key refreshes | Full CPU-prefix-ready acknowledgements | Median ready extent at acknowledgement |
| --- | ---: | ---: | ---: | ---: |
| Qwen 8B full CPU cache documents | 2775 | 2775 | 190 | 9.5% |
| Qwen 8B half CPU cache documents | 3267 | 3267 | 0 | 4.4% |
| Ministral 3B full CPU cache documents | 993 | 993 | 2 | 4.2% |
| Qwen 4B full CPU cache documents | 3106 | 3091 | 397 | 52.3% |

In an illustrative sealed Qwen 8B half-cache case, 216 of 274 refreshes touch only
eight 16-token blocks. The first 127 token positions are identical across all 16
initial prompts—seven complete blocks. Much repeated work is therefore at a small
leading fragment, largely common prompt material, rather than necessarily the
private long prefix whose reuse could save substantial recomputation.

The implementation stops at the first missing or pending CPU block. A CPU probe
constructs a GPU-resident first block and a ready CPU second block. The selected
path refreshes zero keys; the existing optional GPU-prefix bridge refreshes the
one reachable CPU suffix block. This is a real eligibility limitation of the
selected configuration. The bridge is not enabled/qualified by this audit.

**Important limit:** 100% CPU-prefix coverage is not necessary for useful reuse:
GPU roots can bridge into a CPU suffix, and ordinary capacity pressure can make
full CPU residency impossible. The percentages describe state at hint delivery,
not future cache survival. They are not a proposed new pass/fail threshold and
do not prove the missing suffix existed in every live trace.

The useful-work concern has direct work counters too. On 5080 Qwen 8B half-cache
documents, candidate loading rises to 0.904 GB/run versus 0.060 GB, while returning
prompt recomputation is 131,920 versus 130,345.5 tokens. There is no average measured
work-avoidance payoff there. The precise copy/CPU/scheduling cost is not profiled.

**Direction:** align eligibility with native demand-reachable GPU+CPU prefixes;
measure whether advice preserves data actually consumed on return and avoids
recomputation. Do not qualify usefulness from acknowledgement count alone, and
do not fix the benchmark by demanding universal CPU residency.

## 4. Conflicting retries can mask a namespace error

**Reproduced API correctness gap; not observed in campaign IDs.** The result
cache is keyed by event ID. A retry reusing a successful event ID but changing the
model namespace receives the old positive acknowledgement, with
`idempotent_replay=true`, before the current namespace is checked. The same bad
namespace with a new event ID is correctly rejected. Queue order does not change
on replay, so this does not demonstrate wrong-model KV reuse.

The benchmark client checks response schema/event ID, not a full request-payload
fingerprint. A conflicting ID can therefore conceal an integration error. The
ordinary recorded client generated unique IDs and showed no such conflict.

**Direction:** bind idempotency to the complete validated payload and reject
conflicting reuse of an ID. Preserve true same-payload idempotency.

## 5. Non-object JSON gets a server error

**Reproduced input-validation bug; low severity.** In-process ASGI requests with
JSON list, null, integer or string bodies returnHTTP 500 because `.get()` is called
before validating that the body is a mapping. The error is `AttributeError`,
outside the handled `TypeError`/`ValueError` cases. An invalid object returns 400.
The scheduler is never called; this is not a model crash or cache mutation.
The existing operational invalid-field probes cover objects and miss this shape.

**Direction:** validate JSON shape at the endpoint/contract boundary and return
a client error. Add coverage for this boundary when implementing a fix.

## 6. The external contract is still benchmark-specific

**Integration constraint, not failure of the supported local client.** Declared
waits are restricted to 0/50/250/750 ms; otherwise valid 1000/2000 ms estimates are
rejected with an error explicitly referring to the frozen screen. A missing
estimate is parsed but produces `wait_estimate_unavailable` and no advice. The
normal client works around longer waits by publishing fresh 750 ms estimates once
per second, through tools and until the next request's first token.

Timestamps also require a common host monotonic clock. Cross-host clock
translation is not implemented. Correctly rejecting unsupported input is safe,
but these restrictions prevent treating the interface as a general remote-agent
tool-wait protocol without an explicit integration contract.

## What the benchmark rules prove—and what they do not

- Correct-prefix/fresh-reference token checks, namespace/generation/future/stale
  guards, duplicate replay of identical input, cancel/drain/recovery and no
  speculative GPU jobs pass within the tested scope. There is no saved evidence
  here of KV corruption or GPU-reference leakage.
- All 22,805 hint submissions audited across all 269 retained candidate/control
  cases report zero GPU reservations/speculative-transfer bytes. That is expected
  for the CPU-only candidate, not missing prefetch activation. Useful-on-time
  speculative-copy selection is a different path and was not tested as this policy.
- Capability and activity checks establish that the feature can execute, with
  required work and cleanup. They do not tie a particular hint to later useful
  demand. The `cpu_recency` activity value sums returned ready-key counts.
- The three 5080 B06 exclusions had positive ready-key refreshes:24/84/84 events,
  but zero CPU restores. The anchor case had scheduler preemptions; the two
  Ministral cases had 15 eviction-affected returns. They are purpose/activity
  failures, not proof that hint delivery was broken, and not by themselves a
  unique causal diagnosis of a candidate bug. Their original exclusions remain.
- Screen-cohort pressure does not guarantee that a different held-out confirmation
  cohort creates the same pressure. GPU-resident controls cannot prove CPU-cache
  retention benefit. This is a coverage limit, not a reason to force restores.
- The operational expiry check rejects a delayed new hint. It does not prove
  automatic rollback of prior recency or safety from prolonged valid wrong hints.
  The registry checks prove bounded refusal, not ongoing capacity for new sessions.
- The 2080 Ti enabled/no-hint24-pair control showed 3.33% throughput loss. This warns
  of a cost in the measured research profile, but does not isolate the irreducible
  overhead of the minimal unaudited operating profile. The 5080 short-QA bound
  remains inconclusive. Do not label this alone an inherent 3.33% production tax.
- Strict source/JSON/status/word-limit output checks concern answer compliance.
  Failing them is not proof of cache-state corruption. Source review does not
  establish statistical answer-quality equivalence between arms.

## Assessment and next priorities

The wait-hint concept is not shown to be fundamentally unsound. The selected
candidate does have real shared-cache interference risk, a lifecycle availability
cliff, limited prefix eligibility/usefulness evidence and two API hardening gaps.
These remain meaningful even with overall throughput marked inconclusive.

Prioritize useful/cost-bounded retention and interference, then session lifecycle,
then replay/input correctness and a general integration contract. Existing
optional GPU-aware/budget controls are candidates for a separate design/evaluation,
not fixes silently enabled here. Original serving code and campaign decisions
were not changed. This investigation authorizes no new GPU campaign or training.

Reproduction: the three `stage6_candidate_*` audit helpers in
`benchmarks/research_checks/` consume the retained inputs and run the CPU probes.
The probes use `tools/vllm_runtime.py` with `CUDA_VISIBLE_DEVICES=''`; the
125-test command and bound input/source hashes are in [validation](validation.json).
Source locations for each finding are in [findings](findings.json).
