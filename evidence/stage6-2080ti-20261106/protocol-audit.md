# PR-evidence audit before the bounded addendum

This is a source/receipt audit, not new benchmark acquisition or final
qualification. The original policy and primary acquisition remain frozen. The
[authorized addendum](../../docs/STAGE6_ADDENDUM.md) addresses the narrow new gaps;
original quality, integrity and portability work remains pending.

| Pinned model | Architecture/config | FP16 KV bytes/token | GPU tokens / CPU tokens |
| --- | --- | ---: | ---: |
| Qwen/Qwen3-8B-AWQ | Qwen3ForCausalLM; 36 layers, 32 Q/8 KV heads, head dimension 128 | 147,456 | 16,384 / 18,944 |
| mistralai/Ministral-3-3B-Instruct-2512-BF16 | Mistral3ForConditionalGeneration, text config ministral3; 26 layers, 32 Q/8 KV heads, dimension 128 | 106,496 | 22,528 / 18,944 |
| Qwen/Qwen3-4B-Instruct-2507 | Qwen3ForCausalLM; 36 layers, 32 Q/8 KV heads, dimension 128 | 147,456 | 9,216 / 18,944 |

All three execute dense full attention GQA, TP1, FP16/eager on the 2080 Ti/WSL2
host. Ministral's source weights are BF16 but execution is FP16 and text-only.
This is two model families, not three independent attention architectures.
Base CPU allocations are respectively 2,793,406,464 / 2,017,460,224 /
2,793,406,464 bytes. B09 halves them. The anchor roomy control is 4,831,838,208
bytes at the unchanged GPU capacity. No replacement model is needed for an
honestly scoped dense-GQA preliminary PR; hybrid/sliding/MHA/MQA, TP>1, native
Linux and other GPUs remain unqualified. Native restore/fresh correctness
fixtures pass all three models, but do not establish broad architecture support.
See [model fixtures](model-functional-checkpoint.json), the model catalog and
effective plans in `../../output/stage6-2080ti-20261106/inputs-15/`.

The primary is 22 model/workload cells, 54 distinct configurations and 524 process
repetitions in 210 matched blocks. The anchor covers B01–B12; the secondary models
cover B02/B06/B08/B09/B11, with external systems only on five anchor cells. It is
not the full three-model/twelve-workload/four-system product. Repetitions reuse
fixed source cohorts and measure process noise, not independent task populations.

The effective anchor cohorts are: B01 one six-request conversation (27 seconds
of declared user pauses); B02 64 QA requests; B03 sixteen summaries; B04 48
hot/scan/return QA requests from twenty distinct records; B05/B06/B07/B12 eight
repository workflows (32 requests/24 tools); B08/B09 sixteen document workflows
(64/48); B10 four teams (20/12); B11 sixteen complete repository workflows plus
sixteen QA units (80 requests/48 tools). B05 is sequential, B06 tool pool four,
B12 tool pool one, B07 changes only CPU capacity, B09 halves document CPU capacity.
Secondary B06 retains sixteen complete repository workflows. The tasks execute
fixed real tools, not model-generated commands; tool prompt excerpts are 2,000
characters with original raw outputs retained. Workflow completion is not
autonomous-agent success or answer correctness.

Anchor B06/B07/B10/B12 are GPU-resident controls in the inspected confirmation
receipts, not CPU-pressure proof. The B06 first-eight cohort fits its GPU envelope;
zero CPU eviction/restore is legitimate behavior, not a policy failure. The
full-sixteen operational runs do establish pressure: ARC 141,726 loaded tokens
and 265/288 eviction-affected returns; candidate 273,296 and 270/288. Those are
one persistent process per arm across six waves. They cannot replace independent
matched four-system pressure repetitions. Secondary development pressure screens
had actual eviction/restore activity; their final confirmation receipts still
need the explicit per-model/cell audit implemented in `addendum_report`.

Hints bind to actually observed native token prefixes, not future answers. The
750ms declaration is not a 750ms inserted tool wait; advice is refreshed every
second over real tool and server-queue waits until the next first token. The
selected .6 enable fraction is not a 60% CPU-retention quota. Its CPU-only path
touches ready reachable prefix keys within ARC recency, without speculative
transfers, hard pinning or treating advice as demand hits. Repeated ready-key
touch sums are not unique saved/consumed tokens. Native per-request loaded and
computed counters establish activity; exact counterfactual avoided work is not
identified when outputs/scheduling differ.

All three models' validation probes reject future/invalid observations and check
duplicates, delayed expiry and native stale/wrong-namespace guards. Operational
abandonment is covered, with zero sampled speculative jobs/reservations. These
checks do not measure the cost of valid but false predictions under subsequent
pressure. The new advice cohort adds that distinct experiment, with no-return
and at-least-45-second-late returns and identical unhinted traffic across arms.

B11 already records per-class mean/p95 latency, queue/tool handoffs, workflow
durations, counts and tokens. Paired class comparisons are implemented in the
primary report and remain to be finalized; it lacks a p99 report field. Sixteen
unhinted requests/process (128 total in the eight anchor repetitions) cannot
support a credible p99 SLA. Equal offered units do not mean equal requests,
tokens or resource share. The addendum reports descriptive p95/p99 and paired
unhinted mean end-user latency, explicitly including client queue time.

Original short-QA O2/O0 enabled-no-hints throughput is 0.9821 with 95% CI
[0.9554, 1.0096], six pairs: inconclusive at a 3% loss bound, not proof of a >3%
regression. No extra precision batch was previously planned; the new fixed
24-pair repair-cohort extension is now authorized. Keep the cohorts separate;
O1 remains outside the extension. The original two-repetition repository
supplement could only give extremely broad df=1 CIs; four new matched repetitions
give df=3 intervals, still potentially broad. Neither design guarantees a win.

Candidate/ARC share the instrumented patched native runtime; ARC is not the
unloaded-plugin O0 control. Continuum and pinned upstream use matching byte
budgets and first inputs but different runtime/backend/chunk and hint interfaces,
and generated continuations can differ. External cache counters remain unavailable
where not exposed. Compare configured systems, retain output caps and source
quality defects, and report useful workflow/time denominators alongside token
counts. Full source/blinded review, every capped intermediate and final response,
archive integrity and actual relocation/import remain prerequisites to readiness.
See [input/activation audit](../../output/stage6-2080ti-20261106/development-15/analysis/four-system-input-activation-15.json)
and [overhead checkpoint](overhead-checkpoint.json).

SAE now specifically means [CPU-offload PR #50422](https://github.com/vllm-project/vllm/pull/50422),
not the unrelated historical SAECache paper. It is a direct relevant competitor;
the current factory-registered policy requires additional session lifecycle hooks
absent locally. The bounded feasibility finding and inspected head are in the
addendum protocol. No SAE performance comparison or dominance claim exists.
