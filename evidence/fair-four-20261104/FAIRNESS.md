# Frozen four-system fairness preflight

Prepared before timing on 2026-11-04. One pass per system; no policy tuning or
statistical ranking. The claim is complete configured systems on this machine.
Predeclared order seed 17012 yields: active wait ARC, ARC without hints,
Continuum public preview, upstream draft plus host-pointer compatibility fix.
This order is frozen before timing; no alternative shuffle was selected. Semantic scoring/SVD/speculative
prefetch are disabled in both local arms.

Common settings: RTX 2080 Ti (compute capability 7.5, 11 GiB), same local
Qwen2.5-7B-Instruct-AWQ snapshot, AWQ weights/FP16 compute and KV, single GPU,
eager execution, Triton attention, prefix caching, 8,192 context, 1,024 batched
prefill tokens, 8 server sequences, 2,560 GPU blocks of 16 tokens (2,240 MiB KV),
2,818,572,288 CPU backing bytes (2,688 MiB). Sixteen identical authored repository
investigation workflows, four model requests and three real tools each, four tool
workers, unbounded client admission. Same tokenizer/chat template and initial
prompt construction. Shared fixed-tool bash instruction is required by Continuum;
the application executes declared argv only. One identical tiny warmup per fresh
server is excluded. Temperature zero and same seeds; EOS/context stopping, no
small output cap, no retries. Actual live outputs and histories may differ.
Whole-batch time includes hint production/transport and tool time, excludes startup.

| Arm | Retention / backing | Scheduling / runner | Memory path |
| --- | --- | --- | --- |
| Plain ARC | Local native CPU ARC, no hint production, plugin, RPC or recency processing | Native priority, same application continuation credit of 8 s; V2 runner | Pinned native CPU backing |
| Active wait ARC | Same local config except CPU hint retention enabled and its enable budget; no speculative dispatch | Same priority/credit/V2 | Same pinned backing |
| Continuum public preview | Published pinning/scheduling; LMCache 0.3.6 LRU, 256-token chunks | Its integral Continuum scheduler; published V1 runner | LMCache's native pinned allocator; vLLM input buffers remain WSL unpinned in this version |
| Upstream draft + host-pointer compatibility fix | Published native CPU ARC explicitly selected; priority 10, 60 s expiry following published example; clear claim on final turn | Native priority, identical application priority function/8 s credit forwarded to AsyncLLM; V2 | Pinned mmap with CUDA device-alias compatibility correction |

Local controls share all connector values except `wait_hint_cpu_retention_only`
and `prefetch_budget_fraction` (the latter enables CPU advice; diagnostic mode
prevents speculative GPU work). ARC has `publish_hints=false` and empty plugin
list; zero publications and absence of hint processing must be verified after
measurement. Both retain identical native accounting instrumentation. Wait advice
is published before real tools, refreshed each second during tools and queueing,
and uses only the already completed prefix, never future tool outcomes.

The upstream compatibility copy retains the exact pinned published revision;
only `vllm/v1/kv_offload/cpu/gpu_worker.py` differs. The manifest requires the
patch hash, reconstructs that file from original plus patch, compares every other
Python file including submodule support, and freezes actually executed Python and
native binary content hashes. The original Git checkout remains clean. This is
explicitly a modified upstream system, not an unmodified published result. A
separate V2 probe already passed three identical 32-token outputs and two real
pinned restores of 170,655,744 bytes each. Logs are in the linked output directory.

Remaining differences are material and retained: vLLM 0.26 development / torch
2.13 local versus vLLM 0.30 development / torch 2.13 upstream versus vLLM 0.10.2 /
torch 2.8 / LMCache 0.3.6 Continuum; native scheduler internals and quantization
kernels can differ across versions. Continuum's public code omits its paper's
automatic timeout estimator; its own scheduler and 256-token LMCache backend
remain integral. Local/upstream native CPU chunks default to one GPU block (16 tokens),
verified from their offloading config source; actual worker geometry and bytes
are checked before the measured window. LMCache uses 256-token chunks. The older
Continuum vLLM input buffers use its supported WSL unpinned path even though
LMCache CPU KV backing uses its own pinned allocator. We do not silently patch
Continuum to change this runtime. Native instrumentation is different; external
systems use the same thin HTTP bridge, upstream additionally has read-only policy
return observation. External native receipts/recomputation/transfer metrics remain
unavailable wherever incomparable. No isolated retention-policy superiority or
production-quality conclusion follows from cross-runtime timings.

Post-run required checks: all 256 responses and EOS/length/error counts, all 192
real tools, matching initial prompt token IDs and selected inputs, actual output
lengths, tool-return-to-first-token mean/p95, final test-status labels versus actual
test result, no contradictory status. These narrow checks do not grade general
code reasoning. Local native ARC pressure/wait retention and Continuum recorded
pins/upstream priority applications and selections must be observed. Record actual
effective GPU blocks, CPU backing, chunk granularity, runner/attention/quantization
paths from source and retained startup logs. Verify sealed results offline and
owned process cleanup/GPU locks after the final case.

Runtime manifest: [runtimes.json](../../output/fair-four-20261104/runtimes.json).
V2 compatibility probe: [log](../../output/fair-four-20261104/v2-probe.log),
[completed transfers](../../output/fair-four-20261104/v2-probe-transfers.jsonl).

Operating limits: model files and OS page cache are warm from earlier work; no
OS cache flushing, swap changes, clock locking or system settings edits. Startup
and a common tiny warmup precede every measured window. Sequential one-pass order
can still confound thermal/background/cache state. GPU temperatures/clocks/power
and host memory/swap state are retained before and after; tool subprocess caches
remain naturally shared across cases. This is one predeclared exploratory pass.


## Post-run supported-setting correction (2026-11-05)

This note was added after the initial four-arm measurements; it is not a claim
that the following control was covered by the original preflight. The initial
Continuum row used supported default `save_decode_cache=false`, while native arms
allow decode KV publication. Preserving the default was not required by published
code. Source/config and a live boundary-crossing probe confirmed that true is
supported. The [separately predeclared correction](REPLACEMENT.md) selects one
replacement pass with `LMCACHE_SAVE_DECODE_CACHE=true`, regardless of time, while
retaining the false row as diagnostic. No other settings or arms were tuned/rerun.
The replacement completed with 35 decode-save decisions matched to completed
native stores; primary results use this corrected row. Source freezes, different
runtime/kernel/chunk behavior, live outputs, host paging and orchestration gaps
remain explicit limits; these results do not establish a general winner.
