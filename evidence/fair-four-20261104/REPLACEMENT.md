# Predeclared Continuum decode-publication correction

Recorded before replacement timing on 2026-11-05. The earlier four-case seal
remains unchanged. Its Continuum row used LMCache's supported default
`save_decode_cache=false`, while native arms permit decode KV publication. This
was a missed supported configuration control, not an invariant required to run
published Continuum. The primary comparison therefore replaces that row with
**one** pass setting `LMCACHE_SAVE_DECODE_CACHE=true`, irrespective of its time.
The earlier false row remains diagnostic; we will not choose the faster row.

The pinned LMCache 0.3.6 config accepts/validates the setting and forwards it to
its vLLM adapter. A separate live proof passed: 2,808 real-context prompt tokens
and 32 forced probe output tokens crossed a 256-token boundary. Read-only adapter
observation records a 2,816-token decode-phase publication containing eight
computed decode tokens; native LMCache logs show its corresponding completed
256-token store. Backing remains 2,818,572,288 pinned bytes. This bounded forced
probe tests publication, not answer quality or representative performance.

Replacement keeps the identical selected corpus, prompt/tokenizer construction,
model/FP16/eager execution, Continuum scheduler, real tools/workers, GPU/CPU byte
budgets, 256-token chunks, EOS stopping and common tiny warmup. No chunk/kernel
search or other arm rerun. A fresh output freezes current source including the
one-line coordinator import fix, an added read-only decode-publication observer,
actual Continuum/LMCache Python and native content identities. Positive stores,
retrievals, scheduler pins and decode publication must be verified in the new
measurement. Native ARC's 16-token versus LMCache's 256-token chunks remain a
backend difference, along with supported runtime/kernel/input-buffer differences.

Continuum's pinned `AWQMarlinConfig.get_min_capability()` requires compute
capability 8.0; this card is 7.5. Its older AWQ kernel path therefore remains a
runtime support difference, rather than a supported flag we silently left off.

- [Config-only validation](../../output/fair-four-20261104/continuum-decode-config-probe.log).
- [Live probe log](../../output/fair-four-20261104/continuum-decode-probe/server.log),
  [read-only publication evidence](../../output/fair-four-20261104/continuum-decode-probe/runtime-observation.jsonl).
- [Predeclared replacement coordinator](../../output/fair-four-20261104/run_matched_continuum.py).
