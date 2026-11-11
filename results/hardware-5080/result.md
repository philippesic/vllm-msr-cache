# An architecture-specific rebuild repaired 5080 transfers; the retained study is incomplete

2026-11-10 · historical · unresolved

The November 8 Continuum warmups failed because its binary lacked sm_120 or PTX. Rebuilding the same LMCache source repaired exact transfers and all required screens. The November 9 report remains a running-campaign snapshot.

## Scope

Same-source LMCache 0.3.6 rebuild; 393,216-byte bidirectional equality; 54 screening configurations and 22 matched blocks. B06 had 31 valid measurements out of 32 attempts and seven of eight complete blocks.

## Limits

One candidate attempt had zero restores and preemptions and remains excluded. Confirmation, overhead, operational and final audits were not complete at this revision. No final campaign result or GPU-only cross-host attribution is claimed.

## Decision

Retain the binary capability finding and incomplete-study boundary. New-code 5080 qualification remains deferred, not silently inherited.

## Sources

- [evidence/stage6-5080-20261108/RESULTS.md](https://github.com/philippesic/vllm-msr-cache/blob/6bbdbdee7ab8ac8012dc27a2938175477d58fb0f/evidence/stage6-5080-20261108/RESULTS.md)
- [evidence/stage6-5080-20261109/RESULTS.md](https://github.com/philippesic/vllm-msr-cache/blob/6bbdbdee7ab8ac8012dc27a2938175477d58fb0f/evidence/stage6-5080-20261109/RESULTS.md)
- [evidence/stage6-5080-20261109/lmcache-build.json](https://github.com/philippesic/vllm-msr-cache/blob/6bbdbdee7ab8ac8012dc27a2938175477d58fb0f/evidence/stage6-5080-20261109/lmcache-build.json)

Exact source hashes and figure inputs: [result.json](result.json).
