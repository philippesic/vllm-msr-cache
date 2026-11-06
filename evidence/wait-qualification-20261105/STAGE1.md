# Stage 1: live overhead controls and exact-token replay

The live checkpoint and replacement replay both favor wait-hint over no-hint ARC.
Individual-block tracing also passed; the narrow Stage 1 gate is met. Three paired repetitions provide conditional evidence for these inputs,
not a population-wide performance guarantee.

| Comparison | Batch-time reduction | Paired 95% interval |
| --- | ---: | ---: |
| Live full wait vs no-hint ARC | 8.07% | 7.00–9.12% |
| Live full wait vs inactive hints | 7.66% | 5.28–9.97% |
| Exact-token replay full wait vs no-hint ARC | 4.72% | 3.12–6.30% |
| Replay single initial refresh vs no-hint ARC | 0.23% | −0.11–0.57% |
| Replay GPU-gated protection vs no-hint ARC | 4.14% | 1.76–6.46% |

The full replay mechanism was about 0.60% faster than GPU-gated protection; its
paired direction is unresolved. The removal does not establish that early CPU
protection is necessary. Full wait outperformed a single initial refresh by about
4.50% in batch time. Repeated updates during recorded waits and the returning queue
are supported in this setting. GPU-gated protection also adds native prefix-index
lookups, so a small timing difference cannot be attributed solely to early protection.

All twelve replacement replay cases completed 64 requests with identical frozen
prompt and continuation IDs (768 requests total). ARC recomputed 63,386–63,498
prompt tokens per case; full wait recomputed 55,290–56,282. Single refresh recomputed
62,458–62,522. These are native first-admission source counts, not estimates from
hint activity. Replay executes ordinary model forwards with an artificial logits
processor, and replays recorded queue/tool delays without executing tools. It is
causal diagnostic evidence, not a second live application benchmark. The trace was
selected from the latest fair-system run before the variants were measured.

The live controls completed 576 responses and 432 actual subprocess tools. They
use the original repository instructions; replay's frozen fair-system prompts
also include declared fixed bash tools. Their percentages must not be pooled or
interpreted as improvement from one protocol to the other. Live continuations and
resulting prompts differ. Final observed-test-status reporting passed in all live
conditions; broader shared coverage hallucinations and word-limit failures remain
in [the quality audit](QUALITY.md). Equal general task quality is not established.

Inactive-hint overhead remains unresolved, not proven zero. First live ARC had
about 412 MiB host swap-out; later ARC cases had about 2.7 and 41.8 MiB while wait
cases had less than 1 MiB. Later pairs still favor wait, but the live host-state
mismatch must remain disclosed. Replacement replay had little paging (at most
about 8.4 MiB swap-out), with ARC times tightly grouped around 110.7 seconds.

The first repeated replay stopped after six completed exact-token cases because
one hint publication disconnected in the GPU-gated arm. It is retained as failed,
not pooled into the replacement intervals. The replay adapter had used persistent
HTTP sockets; matching the maintained live adapter's fresh-socket transport
removed stale connection reuse. The complete replacement passed without retries.
See [repair record](REPLAY_REPAIR.md).

Raw immutable runs: `output/wait-qualification-20261105/stage1-live/`,
`stage1-replay/` (failed), and `stage1-replay-fresh-sockets/`. Offline verification
is recorded at the end of each campaign log. Compact recomputed reports are
`stage1-live-summary.json`, `stage1-replay-summary.json`, and
`stage1-quality-final.json` in the same parent directory. All failed preflights
and the original interrupted campaign are preserved.

The separate full-lifecycle diagnostic completed both cases and verified 162 files
/ 128 request records. Active advice moved 45 of 46 sampled distinct keys toward
MRU; 39 distinct keys link an earlier CPU commitment, queue movement, no recorded
intervening eviction, and a later successful nonprefetch demand load. Inactive
advice moved zero of 46 keys across 635 hint events (active: 608 events). These
observed chains support the proposed behavior; they do not prove each individual
key would have been evicted without advice. Exact-token paired replay supplies
the comparative work-saved evidence. Trace timings are excluded from primary
intervals. Raw run: `stage1-trace/`; lifecycle links: `stage1-trace-wait-links.json`
and `stage1-trace-inactive-links.json` in the qualification output parent.

A stricter v2 lifecycle audit additionally rejects intervening CPU recommits,
which could otherwise represent re-creation instead of survival. It retains the
same 39 distinct links. The original v1 report is preserved; the strengthened
report is `stage1-trace-wait-links-v2.json`, with its analyzer snapshot alongside.
Absence of recorded eviction/recommit is still an observation-based chain, not an
independent proof of continuous residency or a per-key counterfactual victim.
