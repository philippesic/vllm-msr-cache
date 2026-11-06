# Admission experiment

Explicit user request: run the proposed admission comparison with threshold 0 or 1 and report results. Choice: threshold 0, subject to native source verification that it disables the frequency gate. Existing small-trial/minimal shared-GPU permission applies. No paused campaign resumed.

## Frozen comparison to prepare on WSL

Four fresh short serving arms: ARC demand with threshold 2; reduced-scoring prefetch with threshold 2; ARC demand with threshold 0; reduced-scoring prefetch with threshold 0. Same measured v3 production source, cached dense Qwen3-4B BF16 TP1, exact 45-request six-agent plus background trace, fixed native root seed, 640 GPU blocks / 1440 MiB, 448 MiB DRAM, maximum three running sequences. Flat scoring search, no capture optimization, no whole-chain admission helper. Native adoption setting remains unchanged; partial prefixes may still use splice. Counter instrumentation must be equivalent across arms and must not affect decisions.

Freeze execution order and input/config/source/hooks before timing. One run per arm, no statistical significance claim. End-to-end results include actual hint/scoring/prefetch cost. Closed-loop arrivals and cache histories may change; distinguish product differences from direct component costs.

## Questions and measurements

1. At what stage do blocks disappear? Count complete eligible blocks, store candidates, threshold qualification, accepted/submitted/completed copies, retained blocks and exact contiguous prefix available at resume. Distinguish unique block identities, repeated store events, concurrent occupancy and bytes. Assign actual rejection reasons; do not infer every missing store was threshold rejection.
2. Does admission produce useful reuse? Track demand loads, recomputed prompt tokens, prefetch on-time consumption, wasted transfers and CPU eviction. Count simultaneous eligible sessions and constrained decisions; pool size alone does not imply useful ranking opportunity.
3. Is there a memory benefit? Track true GPU block occupancy/free/referenced/reserved/staging and CPU resident bytes. A D2H copy does not necessarily free the GPU source. Distinguish fixed allocated VRAM from logical occupancy, peaks from averages, and sampled estimates from transition-integrated byte time. Keep duplicate staging/destination occupancy in the same fixed pool.
4. What is the cost? Report D2H and total/demand/speculative H2D, transfer/fence envelopes, scoring/capture cost where available, release/submit first-token mean/median, token latency, throughput, background effects, SLO misses and failures.

Comparisons: threshold0 vs2 within ARC isolates the admission-policy package at product level; prefetch vsARC within each threshold measures the complete prefetch package. A larger DRAM pool without useful contiguous prefixes is not success. Lower recomputation with much greater transfer traffic is a tradeoff. Equal throughput with more GPU room is only a potential benefit until actual occupancy or a fixed-latency capacity comparison supports it.

## Boundaries and verification

Engineering planning boundaries: at most four model lifecycles, at most180seconds each and fresh finite whole-session admission up to30minutes from acceptance. These are assistant planning choices, not user-imposed time limits. No extra model download/build, synthetic qualification, tiled summaries, capacity expansion, parameter sweep, production admission fix or search/capture optimization. Native CPU checks must verify threshold semantics and counter reconciliation without initializing CUDA. Check exact outputs, native ordered KV identities, actual transfers/fences and source/config pins.

Preserve all useful evidence and negative results outside Git. Reconcile owned processes/project lock, restore original baseline, release native custody and retain cleanup receipts. Shared jobs/models/runtime/environment/caches and explicit pauses remain protected. Report any unmeasured quantities explicitly.
