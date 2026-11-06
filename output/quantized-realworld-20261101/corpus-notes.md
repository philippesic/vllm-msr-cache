# Public content corpus for quantized serving replay

This is a serving workload corpus, not a live user study or coding-agent task-success evaluation. It contains **144 distinct sessions**, 72 public human conversations and 72 recorded software-agent content sequences. Every session has three or four requests. Source IDs are deterministically partitioned across seeds 1103/2207/3301, 24 sessions per cohort per seed. The runtime owns selection, arrivals, synthetic wait profiles and actual model generation.

## Sources and attribution

- **Allen Institute for AI, WildChat-1M**, ODC-BY: <https://huggingface.co/datasets/allenai/WildChat-1M>. Dataset revision `7d6490e462285cf85d91eabea0f9a954fbddcd1f`. This dataset contains real public human/chatbot conversations. Eight nonrandom 100-row windows at offsets 0, 100, 1000, 10000, 100000, 200000, 400000 and 600000 were retrieved with the official dataset-viewer API; every response's `x-revision` matched the pinned revision. This is a small sample of English multi-turn users, not a probability-weighted sample of production traffic. Keep the attribution and retained license in `data/wildchat-license.md` when reusing it.
- **tarsur385, SWE-bench Verified mini-SWE-agent trajectories**, MIT: <https://huggingface.co/datasets/tarsur385/swebench-verified-trajectories>. Revision `773748a7c1222e8a642a7059821498e14293562a`. The creator publishes native recorded trajectories extracted from Docent. The source model is `gpt-5-mini`; 96 file paths were sampled in SHA256(path) order independently of resolution outcomes. Seventy-two eligible instance IDs were selected in SHA256(ID) order. `data/swe-card.md` retains provenance/license metadata. The tasks are real repository issues, and observations are recorded tool outputs from those historical runs.

The gated LMSYS-Chat-1M dataset was considered but not accessed. No credentials or agreement acceptance was needed for the selected exports. The WildChat filter endpoint timed out; its replacement was bounded row retrieval, not unbounded corpus download.

## Causality and input construction

The prepared JSON is `prepared/session-corpus.json`, schema `quantized-realworld-sessions-v1`. It includes `sessions[{id,cohort,source_id,seed_partition,turns,metadata}]`. Each turn has `user_text`, matching tokenizer IDs, original/retained token counts, content kind and original message index. The canonical input helper `messages_for_turn(session, index, generated)` uses only user fragments through that index and **actual supplied generated outputs** from previous completed requests. Historical assistant responses are kept only in the normalized public source and are absent from prepared sessions and generated histories. The driver must obey this consumer boundary; a helper-only test does not prove its use.

Recorded tool outputs cannot honestly be called results of this benchmark model's newly generated commands. Agent-cohort later turns therefore explicitly identify them as **external observations from a fixed historical trace**, not execution of the previous response. No commands are executed. This is agent-content replay with causal release of fixed observations, not interactive SWE-agent performance or task success. No future observation is included in an earlier request.

Source IP hashes, location and browser/request headers are discarded. Directly recognizable email and international-phone patterns are redacted. This is not a guarantee that all PII has been detected in public text. Raw WildChat API responses containing unused metadata were removed after creating normalized sources and recording their hashes/revisions; exact paths and logical byte counts are in `data/wildchat-source-manifest.json`.

## Timing limits

WildChat user-message timestamps are null in these exports. Assistant-message timestamps permit an **inter-response interval**, which includes original model generation plus human delay. It is not an observed think time. These intervals are retained only as source metadata; `observed_wait_ms` is null for every prepared turn. SWE trajectory tool durations are also unavailable. The runtime must explicitly label all arrival/think/tool waits as simulated; the corpus makes no real-time-replay claim. Zero-wait and synthetic short/long/bursty profiles should be distinguished. Source timestamps and future schedule decisions must not become hint-policy deadline or selected-winner inputs.

## Tokenization and size

Tokenizer: official `Qwen/Qwen2.5-7B-Instruct-AWQ`, revision `b25037543e9394b818fdfca67ab2a00ecc7dd641`, loaded locally without model inference. `session-corpus.json` records tokenizer file hashes. User content is never padded or duplicated. Windows retain at most 3072 tokens in the first input, 2048 in later inputs, with a 6000-token cumulative source-input cap. Truncation is a prefix-token window and is explicitly recorded; actual generated outputs and chat-template overhead are additional and must be included in runtime bounds.

| Cohort | Sessions | Requests/session | First-input tokens median / p95 / max | Cumulative source-input tokens median / p95 / max | Truncated fragments |
|---|---:|---|---|---|---:|
| Human chat | 72 | 3–4 | 34 / 748 / 2666 | 102 / 1473 / 3132 | 0 |
| Recorded agent content | 72 | 4 | 1310 / 2194 / 3072 | 4042 / 6000 / 6000 | 15 |

The short chat distribution is authentic to this filtered sample. It was not replaced with long pasted material to induce eviction. Agent inputs provide naturally longer contexts. If a runtime chooses length-balanced subsets instead of hash-order subsets, that additional selection must be reported.

## Reproduction and qualification

```
python -B corpus.py fetch
python -B corpus.py prepare --tokenizer /path/to/pinned/Qwen2.5-7B-Instruct-AWQ/snapshot
python -B corpus.py check
```

The fetch mode uses fixed revision URLs for SWE/license files, validates the WildChat viewer's revision header and retains source checksums. On a future viewer revision change it fails rather than silently relabeling new data as the old revision; the normalized pinned source remains usable. The CPU checker validates source uniqueness, disjoint seed partitions, chronological request construction and rejection of future-turn and historical-assistant-answer controls. It writes `prepared/corpus-validation.json` with the prepared-input/source/script manifest. Runtime actual generation, exact prompts/outputs/native-key proof and performance remain separate qualifications owned by the serving runner.
