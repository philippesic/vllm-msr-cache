# Five-system real repository comparison

One completed run per system on the RTX 2080 Ti with Qwen2.5-7B-Instruct-AWQ,
FP16, 16 sessions / 64 requests per system, 8 server sequences, 4 real tool
workers, 2,560 GPU blocks and 2,688 MiB CPU backing. All five cases passed their
measurement checks: 320 EOS responses, 240 real tools, matching initial prompts,
and verified process cleanup. This is exploratory system evidence, not a
repeatable performance ranking or production qualification.

| System | Batch time (s) ↓ | Workflows/min ↑ | Output tokens/s ↑ | Offered first-token mean (s) ↓ | Test-status labels |
| --- | ---: | ---: | ---: | ---: | ---: |
| ARC | 107.20 | 8.955 | 69.65 | 8.384 | 16/16 |
| Wait-hint | 101.77 | 9.433 | 73.51 | 7.524 | 16/16 |
| Semantic | 129.30 | 7.425 | 57.86 | 10.002 | 16/16 |
| Continuum public preview | 108.36 | 8.859 | 69.66 | 9.148 | 16/16 |
| Upstream PR #59714 | 108.49 | 8.849 | 74.26 | 8.852 | 15/16 |

Wait-hint completed the batch fastest. Semantic took 27.1% longer than wait-hint
in this run. Upstream produced the highest token rate but also more output tokens
(8,056 versus wait-hint's 7,481), so that is not an identical-work speedup.
Upstream omitted the required test-status label on `return-priority`; it did not
report a contradictory test outcome. All actual test subprocesses succeeded.
Code-reasoning quality remains ungraded.

## What is being compared

- Local ARC, early CPU wait-hint ARC, and semantic-plus-wait retain the existing
  policies and common continuation scheduling credit. Their earlier results are
  preserved unchanged.
- [Continuum public preview](https://github.com/Hanchenli/vllm-continuum) uses its
  published pinning and scheduling code, plus LMCache 0.3.6. It omits the paper's
  automatic estimator. The native scheduler recorder observed 22 pins.
- [Upstream draft PR #59714](https://github.com/vllm-project/vllm/pull/59714) uses
  the published priority-retention implementation and native CPU offloading.
  Its documented example supplies priority 10 and a 60-second expiry; the final
  request clears the session claim. A read-only Python return observer recorded
  48 positive-priority applications and 574 priority-selection calls. These prove
  activation, not the counterfactual value of individual eviction decisions.

Both published source checkouts stayed clean and pinned. They are separate
installed runtimes, not policies recreated in our engine. A small HTTP bridge
forwards requests to their AsyncLLM API because the upstream draft's hint argument
is not wired through its standard completions endpoint. Continuum's parser
requires generated bash blocks, so **all five arms** receive the same instruction
to echo the next fixed tool command. Only the original application-selected argv
is executed; generated shell is never executed.

All arms use eager execution, whereas the preceding three-way confirmation used
CUDA graphs. The extra shared instruction and current source snapshots also
change the prompts. **Compare these five rows together, not directly against the
historical three-way figures.** Subsequent live responses differ in every pair.
The published runtimes use different vLLM/PyTorch versions and CPU-cache backends,
and native observation costs differ. Cache-source accounting unavailable in the
published runtimes stays unavailable; it is not invented or equated with our
native receipts.

## Compatibility and retained setup attempts

Upstream initially failed with illegal CUDA memory access under pinned host
memory, both with its V2 and older model runners. Its supported WSL unpinned-memory
setting and older model runner passed the real-tool probe and the full run.
The exact manifest retains `VLLM_WSL2_ENABLE_PIN_MEMORY=0` and
`VLLM_USE_V2_MODEL_RUNNER=0`. This hardware-specific setting limits generalization;
it does not establish an upstream policy defect.

Native binaries came from matching published wheels: Continuum's C++ source and
CMake tree match vLLM 0.10.2; upstream uses the wheel for the PR's parent revision.
No CUDA compiler was installed. Dependency sets, pins, commands and native-base
provenance are retained. Both external systems passed one-session real-tool
checks before the timed comparison. One initial ARC run was deliberately stopped
to correct the planned upstream one-shot TTL before any upstream measurement;
that interrupted run is retained and excluded.

## Reproduction and evidence

```bash
make bench-agent-five PY=/home/philip/.local/share/venvs/vllm-semantic-cache-ext/bin/python \
  MODEL=/absolute/path/to/local/model OUT=output/new-five-way \
  RUNTIMES=output/five-way-20261104/runtimes.json
make bench-verify PY=/home/philip/.local/share/venvs/vllm-semantic-cache-ext/bin/python \
  OUT=output/new-five-way
```

The target defaults to one pass. The [benchmark guide](../../docs/BENCHMARKING.md#five-system-comparison)
and [example runtime manifest](../../benchmarks/serving_suite/published-runtimes.example.json)
describe preparing another installation.

- [Compact numbers](comparison.json).
- [Full sealed report](../../output/five-way-20261104/comparison-final/REPORT.md)
  and [result](../../output/five-way-20261104/comparison-final/result.json).
- [Runtime manifest](../../output/five-way-20261104/runtimes.json) and
  [native binary provenance](../../output/five-way-20261104/source-audit/native-bases.json).
- [Offline verification](../../output/five-way-20261104/verification.log): 628 files,
  five cases and 320 request records verified; primary comparisons recomputed.
- [Benchmark checks](../../output/five-way-20261104/bench-tests.log): 120 passed;
  [portable checks](../../output/five-way-20261104/check-light.log): 31 passed plus compilation.
  The final TTL change also passed all six focused adapter checks.
- [Cleanup](../../output/five-way-20261104/cleanup.json): all serving processes
  reaped, GPU free of compute clients, both locks available.

Reusable installed runtimes, source checkouts, build caches, failed setup evidence
and prior-work backups are retained in ignored output; large artifacts are not in
Git. No deployment, push, or resumed AWQ work occurred.
