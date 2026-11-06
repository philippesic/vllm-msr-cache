# Real repository-agent serving evaluation

The maintained benchmark now runs an actual repository investigation workflow:
four live model requests per session, with executed search, pytest, and evidence
tools between requests. Waits are actual subprocess lifetimes. Concurrent long
histories compete for bounded GPU and CPU KV capacity. No recorded tool transcript,
fake tool sleep, future output, or future completion time drives the hints.

The final acceptance run passed **four serving cases, 128 responses, 96 executed
tools and 32 final test-status checks**, in **7.56 minutes including four server
startups and provenance capture**. Each serving phase took 65–67 seconds.
The usual one-pair command is roughly a four-minute development cycle on this
host; the earlier complete one-pair diagnostic took 230 seconds. GPU-free plan
validation and the runner CPU checks are much faster.

## Run it

```bash
PY=/home/philip/.local/share/venvs/vllm-semantic-cache-ext/bin/python
MODEL=/home/philip/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct-AWQ/snapshots/b25037543e9394b818fdfca67ab2a00ecc7dd641
make bench-agent PY="$PY" MODEL="$MODEL" OUT=output/my-agent-eval \
  BENCH_ARGS='--execution graphs'
make bench-verify PY="$PY" OUT=output/my-agent-eval
```

Use a new output directory. The default is one ARC/wait-hint ARC pair. Add
`--repeats 2` for reversed-order pairs, as in the retained acceptance run.
See [usage and metric definitions](../../docs/BENCHMARKING.md#real-repository-agent-workflow),
the [protocol](PROTOCOL.md), [exact plan](plan.json), and
[versioned task definitions](../../benchmarks/serving_suite/repo_tasks.json).

## Measured comparison

Qwen2.5-7B-Instruct-AWQ, RTX 2080 Ti, FP16 KV, TRITON_ATTN, decode CUDA graphs,
eight sessions, four active requests, 8,192-token context, 1,024 GPU KV blocks
(896 MiB) and 896 MiB CPU KV. This is real contention within an explicitly bounded
KV allocation; it does not claim all 11 GiB of device VRAM was consumed. The
application runs a declared read-only tool pipeline, rather than executing
model-generated shell. Questions are authored maintenance investigations of this
repository, not recorded production requests or an autonomous patch-writing task.

| Pair | Arm | Workload time (s) | Mean return TTFT (s) | Computed prefill tokens | CPU-restored tokens | Ready-key refresh observations | GPU peak (MiB) | Server RSS peak (MiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | ARC | 66.261 | 1.449 | 63,049 | 1,008 | 0 | 7390 | 4109 |
| 1 | Wait-hint ARC | 65.110 | 1.406 | 59,036 | 3,136 | 2,039 | 7390 | 4262 |
| 2 | Wait-hint ARC | 65.245 | 1.405 | 61,350 | 3,552 | 2,949 | 7390 | 4255 |
| 2 | ARC | 67.421 | 1.678 | 63,148 | 1,072 | 0 | 7388 | 4011 |

Across the pairs, mean workload time was **66.841 s for ARC versus 65.177 s for
wait hints** (2.49% less elapsed time). Mean return first-token latency was
**1.563 s versus 1.405 s** (158 ms lower). CPU-restored tokens totaled 2,080 versus
6,688; computed prefill totaled 126,197 versus 120,386. Native receipts prove
ready-key refreshes in five distinct sessions in each wait arm. There were
17–18 eviction-affected returning requests in every case. Speculative GPU staging
remained exactly zero, as expected for CPU-retention-only advice.

These are useful diagnostic observations, **not a confirmed causal policy win**.
Both pairs have different live prompt/output tokens, including actual tool timing
text. Token-throughput ratios were +5.09% and -0.23%; quoting one as the speedup
would be misleading. There are only two independent pairs and no performance
confidence interval. Client queueing, tool waits and native/server timing are
separate in the reports. Mean return TTFT excludes client queueing; the
user-facing tool-completion-to-first-token metric includes it.

All final answers correctly reported the real test outcome. Half violated the
requested strict status formatting, reported separately. Broader code reasoning,
citation accuracy and task resolution are ungraded. This is not a quality
qualification. Sampled server RSS was higher in wait arms; GPU peak was nearly
equal. RSS can double-count shared pages and these are load-window samples, not
allocator or startup peaks. Scraping/tracing overhead is included equally; there
is no uninstrumented-overhead estimate.

## Debugging and reliability

Each case keeps `agent.json`, `request-diagnostics.csv`, raw `requests.jsonl` and
SSE, `workflow-*.jsonl`, complete tool stdout, native preservation and scheduler
records, before/after Prometheus snapshots and sampled resource/cache gauges.
The CSV joins each canonical request ID to computed/reused/restored tokens,
prior CPU eviction, queue/first-token/end-to-end/transfer timing and availability.
Hints report attempts, cancelled in-flight RPCs, acknowledgement reasons/RTT,
actual advance windows and CPU refreshes. Missing transfer critical-path
observations stay unavailable. Refresh counts can repeat the same key and do not
mean that many unique keys were saved.

Native and endpoint totals agree in all 16 checks: request count, prompt tokens,
computed prefill and restored tokens for each of the four cases. Offline
verification rehashes artifacts and recomputes request aggregates and diagnostics.
Executed Python/JSON source is archived, including dirty/untracked work, alongside
model/wheel identities and exact server commands. The detailed traces and HTTP
scrapes are enabled for both arms and their cost is included.

Two development failures are retained honestly. The first run was intentionally
interrupted after canonical request-identity joins were found missing. The next
pair completed but correctly reported inconclusive: zero enable budget suppressed
CPU advice in this runtime. The corrected profile uses the established positive
budget and CPU-only/no-dispatch settings. A native regression test proves actual
CPU refresh with zero GPU allocation; serving confirms no staging. Temporary
runtime import stages are excluded from sealing because launcher cleanup removes
them. Quality scoring separates an incorrect test claim from duplicated or
Markdown-formatted correct status text.

## Recommendation and remaining limits

Use `bench-agent` for further wait-hint and semantic-cache serving development;
keep smoke/screen fixtures for quick mechanics checks. This setup exposes the
retention opportunity and reports useful work, rather than treating activity or
allocated VRAM as a speedup. The observed extra reuse makes further bounded
optimization credible. The earlier small scripted-workload results cannot justify
a broad negative verdict about these policies.

This acceptance compared ARC and CPU wait-hint ARC. It does not reproduce the
historical frozen priority profile, test the semantic/SVD factorial in this new
workflow, or establish general answer quality. `--candidate-config` supports
follow-up arms; declare their required activity and inspect scoring, staging,
consumption and actual KV-compression receipts as applicable. No deployment or
change to prior runtime qualification is implied.

Raw runs, failures, validation logs and recovery backup remain under
`output/real-agent-eval-20261104/`; the final run is
[acceptance-v3](../../output/real-agent-eval-20261104/acceptance-v3/REPORT.md).
[Compact measurements](results.json),
[interactive comparison](../../output/real-agent-eval-20261104/acceptance-v3/comparison.html),
and [offline verification](verification.json) retain the supporting evidence.

Validation completed with 1,423 tests passed and 26 skipped, plus 104 runner and
31 portable checks passed. The final verifier checked 282 retained files and all
2,204 source-archive file hashes. Both GPU locks are available, the GPU has no
compute clients, and all four serving process trees have verified cleanup.
About 331 MiB of useful runs/source freezes/failures/recovery evidence is retained.
Seven empty owned runtime-stage directories were removed; WSL free space increased
by 28,672 bytes in that cleanup measurement. The
[bounded storage inventory](../../output/real-agent-eval-20261104/storage-inventory.json)
records exact paths and before/after free bytes. Windows backing-volume space was
not measured. Prior work and shared caches were preserved; nothing was deployed.
