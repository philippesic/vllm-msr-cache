#!/usr/bin/env bash
# Continue the frozen schedule after the 2026-11-10 tool-host interruption.
# Earlier cells include an intentionally retained activity-invalid B06 block:
# never run a whole-phase resume that would retry that block for activity.
set -euo pipefail

cd /home/philip/vllm-semantic-cache-ext
python=/home/philip/.cache/semantic-offload/stage6-5080-runtimes-01/local-env/bin/python
base=/home/philip/vllm-semantic-cache-ext/output/stage6-5080-20261109
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/mnt/c/vllm-semantic-cache-ext-stage6-5080/huggingface
export PATH=/usr/lib/wsl/lib:$PATH
unset PYTHONPATH

# Exact remaining cell suffix of protocol.lock.json, preserving each cell's
# complete frozen repetition/system order through the original acquisition CLI.
for selection in \
  qwen3_8b_awq:B11 \
  qwen3_8b_awq:B12 \
  ministral3_3b:B02 \
  ministral3_3b:B06 \
  ministral3_3b:B08 \
  ministral3_3b:B09 \
  ministral3_3b:B11 \
  qwen3_4b_2507:B02 \
  qwen3_4b_2507:B06 \
  qwen3_4b_2507:B08 \
  qwen3_4b_2507:B09 \
  qwen3_4b_2507:B11
do
  suffix=${selection//:/-}
  "$python" -m benchmarks.research_checks.stage6_5080_launch \
    --phase confirm --resume --only "$selection" \
    --log "$base/confirm-02-$suffix.log" \
    --receipt "$base/confirm-owner-02-$suffix.private.json"
done

# Zero exit means every selected acquisition command returned. Individual
# attempt validity, the full identity inventory and all later audits still apply.
