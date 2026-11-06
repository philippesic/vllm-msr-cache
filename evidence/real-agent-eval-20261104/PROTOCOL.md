# Real repository-agent benchmark acceptance

The objective is an inexpensive, reusable serving benchmark that exercises the
application's real tool-wait/cache-return cycle and retains enough evidence for
later policy optimization. It is benchmark development, not deployment or a new
performance qualification.

Each arm starts the maintained HTTP vLLM server. Eight independent repository
maintenance investigations make four live, autoregressive model requests with
actual generated history. The application executes a source search, the relevant
pytest file, and an evidence search between those requests. Commands and outputs
are real. No arbitrary model-generated shell is executed, and no sleep emulates
a tool. Initial source/test excerpts and tool output limits are explicit; full
raw tool output remains in the run.

The default inputs are authored, read-only maintenance questions about this
repository, with versioned task definitions and source hashes. They are not
production issues, historical user arrival observations, a patch-writing agent,
or a general task-success benchmark. They instantiate the intended serving
mechanism: concurrent long histories, application-observed waits, interleaved
requests, and exact-prefix returns under constrained GPU/CPU cache capacity.

The final acceptance plan uses Qwen2.5-7B-Instruct-AWQ, FP16 KV, TRITON_ATTN,
FULL_DECODE_ONLY graphs, 1,024 GPU blocks (16 tokens each), 896 MiB CPU KV,
8,192-token context, four active client/server requests and eight sessions.
Both ARC and GPU-gated CPU-retention wait-hint ARC receive identical causal hint
notifications and instrumentation. Hints contain only the completed exact prefix
and a constant 750 ms estimate, refresh every second during the real subprocess,
and never contain its future output or eventual completion time. The CPU-only
branch needs a positive runtime enable budget; native staging must remain zero.
No historical continuation priority credit is added to either arm.

Two repetitions reverse arm order (ARC/wait, wait/ARC). Workloads remain live:
different generated outputs and actual subprocess timings may cause subsequent
prompts to differ. Fingerprints make that visible. No result should be promoted
to an identical-work causal speedup when this happens. Two pairs exercise order
and repeatability of measurement; they do not establish a population confidence
interval or a generally useful policy gain.

Acceptance requires complete request/tool/session coverage, drained and reaped
owned processes, measured resources, native receipts joined to canonical request
IDs, actual store/restore activity, positive candidate CPU-key refreshes, and
observed evictions of returning prefixes under cache competition. Native computed
prefill, loaded tokens, prompt tokens and request counts must agree with endpoint
counters. Missing or conflicting required evidence fails measurement; inactive
techniques or missing contention are inconclusive rather than negative verdicts.

Reports include session throughput and duration, initial/return/end-user TTFT,
latency and queueing distributions, actual tool lifetime/exit/output provenance,
hint attempts/cancellations/RTT/lead time/reasons, CPU refreshes, native per-request
reuse/recomputation and transfers, speculative staging receipts, native scheduler
pressure, sampled cache/device/host gauges, CPU cost and device energy. Tracing
and HTTP scrape costs are charged equally to both arms; there is no claim of
zero observation overhead. Final claimed test outcome is checked against the
actual pytest result, separately from strict formatting. Broader reasoning and
task resolution remain ungraded.

The maintained offline verifier checks artifact hashes and recomputes request
aggregates and agent diagnostics. Executed Python/JSON source is archived even
from a dirty workspace. Model and wheel identities, commands, raw streams,
requests, tool stdout, native logs and resource samples remain local under
`output/real-agent-eval-20261104/`. Temporary runtime staging is excluded from
sealing. Failed preparation and inactive runs remain identified as such.
