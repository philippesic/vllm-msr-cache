# Public hint-event timing screen

All three fresh runs and all three pairwise output/native-key comparisons pass; comparison with the pending-start semantic baseline also passes. The candidate misses 99% throughput against both fresh ARC controls. This is a single screen and a distinct causal producer/scoring-timing alternative, not confirmation or identical-policy evidence.

| Arm | Tokens/s | Requests/s | CPU s | Overall TTFT mean/p95 ms | Return TTFT mean/p95 ms | Hint RPCs |
|---|---:|---:|---:|---|---|---:|
| ARC400 | 31.1327 | 0.7297 | 25.99 | 424.92/840.11 | 147.96/184.62 | 0 |
| last-peer candidate400 | 30.6024 | 0.7172 | 26.57 | 413.19/837.99 | 106.72/108.20 | 42 |
| ARC640 | 31.1699 | 0.7305 | 25.88 | 408.10/836.09 | 99.08/115.07 | 0 |
| preceding pending baseline400 | 30.7851 | 0.7215 | 26.66 | 412.50/835.73 | 104.18/110.34 | 102 |

| Candidate versus | Throughput delta | CPU delta | Return TTFT delta | Hint count delta |
|---|---:|---:|---:|---|
| pending_start_baseline | -0.593% | -0.338% | +2.434% | -58.824% |
| arc_full | -1.821% | +2.666% | +7.714% | ARC sends no hint RPCs |
| arc_small | -1.703% | +2.232% | -27.873% | ARC sends no hint RPCs |

Hint RPCs decline 102→42 (-58.824%): full_prefix_unavailable 66→15, active_prefill_in_progress 9→0, queued 12→12, coalesced 15→15. This does not yield a throughput win: candidate is 0.593% slower than the preceding semantic baseline and CPU 0.338% lower. Single sequential observations do not establish causation or equivalence.

Source remains frozen prefix-mean. Server configuration/cache geometry matches the pending baseline exactly; the existing public last-peer-submission event changes producer activation. Zero post-peer idle and independent return launch without acknowledgment gating remain unchanged. Query/lease timing may change scalar scores and staging decisions; no scoring-quality claim is made.

Candidate captures 33 calls/36 query rows, performs 12 numerical prefix evaluations, applies 6 scored candidates and natively adopts 6 jobs. No constrained competing selection occurs in this screen. All physical proofs pass; transfer/allocation failures, preemptions, completed-prefetch waste, return demand H2D and splice bytes are 0.

Candidate actual CUDA allocated/reserved/NVML peak is 3308.542/3362/4046.688MiB, pending baseline identical; ARC400 3306.691/3360/4044.688MiB, ARC640 3411.691/3446/4130.688MiB. Candidate/ARC400 use 175MiB GPU KV and 896MiB CPU KV; ARC640 280MiB GPU KV with the same CPU budget.

The failed ARC640 output from the separate gather screen remains unresolved, is not retried or waived, and is not pooled here. All current fresh exactness checks are independent.

[Standard results](hint-event-screen-results.json) · [Extended assessment](hint-event-screen-assessment.json) · [Plan](hint-event-screen-plan.json). The standard analyzer uses confirmation-shaped labels but this explicit screen supplies no confirmation claim. No additional GPU jobs are recommended from this result.
