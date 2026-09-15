# Future work

This is the retained backlog at the September 2026 pause. Items are ranked by
expected value relative to effort; none is authorized or currently running.

## Highest-priority optimizations

1. **Reject uneconomic prefetches.** Use causal confidence, remaining lead time,
   estimated movement cost, and GPU headroom to fall back to demand restoration.
   This mainly protects against waste and displacement.
2. **Reduce prefetch delivery overhead.** Shorten scheduling, binding,
   installation, synchronization, and request-usable handoff. A transferred block
   only helps when demand can consume it early enough.
3. **Improve causal reuse prediction.** Calibrate semantic ranking on independent
   development/evaluation inputs and score the transferable exact prefix rather
   than an unrelated or unavailable representation.
4. **Retain contiguous exact prefixes efficiently.** Make admission and eviction
   partition-independent, preserve useful chains, and measure actual physical
   displacement rather than inferring it from occupancy.
5. **Adapt budget and timing to load.** Vary speculative capacity and launch
   timing with concurrency, burst pressure, and available GPU headroom.

## Measurement gaps

- A multi-point useful-capacity frontier at equal total memory and a fixed service
  target.
- Exact physical displacement identity and its downstream reuse consequence.
- Process CPU, RSS byte-time, and semantic-index memory.
- A declared resource-price basis for cost per request; transfer bytes alone are
  not provider cache-write cost.
- Repeated independent traces where semantic selection is both useful and faster
  than matched ARC/LRU and FIFO-prefetch controls.
- Longer controlled idle-gap and intervening-traffic sweeps before making any TTL
  claim.

## Ideas not yet fully tried

- Mean-only summaries and incremental lookup telemetry to lower scoring overhead.
- Cheaper splice installation or a graph-compatible device-resident handoff.
- Confidence calibration that combines semantic score separation with measured
  transfer/install cost.
- Explicit caller hints for workloads that can provide them, measured and
  documented as caller-assisted rather than automatic prediction.
- Broader model/layout support: tensor parallelism, context/pipeline parallelism,
  MLA or hybrid KV layouts, larger models, and longer contexts.
- Packaging as a pinned out-of-tree extension with explicit LRU/ARC/MSR selection,
  bounded defaults, telemetry, rollback, and compatibility checks.

## If work resumes

Freeze one provider claim, comparator, metric denominator, serving tradeoff, and
activation condition before implementation. Preserve `store_threshold=2` in all
matched DRAM arms unless deliberately testing admission semantics. Run a small
exact A/L/R/M smoke first; advance only a materially promising configuration to
independent inputs and wider coverage. Keep raw negative attempts and never infer
missing measurements as zero.

Stop retuning a losing new feature, but still complete separately requested
baseline characterization and decision-critical metric coverage even when the
baseline loses. A negative result is an outcome, not a reason to leave required
measurements unavailable.
