# 2080 Ti WSL notes

- The project environment is `/home/philip/.local/share/venvs/vllm-semantic-cache-ext`; activate it or use its `bin/python`. It contains the vLLM prerelease wheel used by the existing opportunity screen.
- On this WSL host, set `VLLM_WSL2_ENABLE_PIN_MEMORY=1` for vLLM: the semantic-cache connector requires the V2 runner and mapped pinned memory. Use FP16 for the RTX 2080 Ti; BF16 and FlashAttention 2 are unsupported on this GPU.
- The checkout contains a copy of uncommitted source and existing evidence from the Mac; check its current state before comparing or changing experiments. Historical GPU results from other machines are not measurements of this host.
