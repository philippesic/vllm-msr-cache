# Natural agent traffic exposed little competitive semantic opportunity

2026-11-10 · historical · negative

Live semantic prefetch was 5.6% slower than ARC in the early simulated-agent trial. Transfers and real scores were observed, but each nomination had only one eligible source: useful semantic ranking was not tested.

## Scope

RTX 5080, dense Qwen3-4B BF16, six simulated agents plus twelve background requests, 45 requests per arm. ARC, request-order prefetch and semantic prefetch used the same 1,440 MiB GPU/448 MiB CPU cache.

## Limits

One fixed-order run per arm; all 135 output token lists matched but random native root salts prevented cross-server native digest equality. All consumed prefixes used GPU splice, not native adoption. On-time speculative coverage was below 0.2% of returning-context bytes.

## Decision

Use this as evidence about opportunity and full-path overhead, not a rejection of unexercised competitive ranking. A separate empty-nomination/digest-copy optimization passed CPU integration but had no measured serving benefit.

## Sources

- [output/agent-load-trial/results/REPORT.md](../sources/a711fecd06011d5de628a0c466b6982f52ec5718f4c97d0e159a845530152dfa.json)
- [output/optimization-astra/REPORT.md](../sources/0645ebf698346e3ff116533cc70de9cda5ad0fb0f7e04e624884fd9b5ed8c5c6.json)

Exact source hashes and figure inputs: [result.json](result.json).
