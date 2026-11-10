#!/usr/bin/env bash
# Repair the independent service's tool PATH; retain all earlier attempts.
set -euo pipefail

cd /home/philip/vllm-semantic-cache-ext
python=/home/philip/.cache/semantic-offload/stage6-5080-runtimes-01/local-env/bin/python
base=/home/philip/vllm-semantic-cache-ext/output/stage6-5080-20261109
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/mnt/c/vllm-semantic-cache-ext-stage6-5080/huggingface
# systemd's default PATH omits the existing host-provided ripgrep executable.
# Pass the verified interactive PATH explicitly when creating this service.
export PATH=/usr/lib/wsl/lib:${STAGE6_CAMPAIGN_PATH:?Pass the verified campaign tool PATH}
unset PYTHONPATH

# Fail before GPU acquisition if a required external tool is unavailable.
"$python" -c 'import shutil; assert shutil.which("rg"), "Missing required rg in service PATH"'
rg --version
"$python" -m pytest --version

# Exact remaining cell suffix. The unchanged frozen CLI reuses complete blocks
# and replaces incomplete setup-failed B11 blocks as whole comparisons.
# Earlier activity-invalid Qwen3-8B B06 is deliberately never retried.
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
    --log "$base/confirm-03-$suffix.log" \
    --receipt "$base/confirm-owner-03-$suffix.private.json"
done

# Command exit is not measurement validity; final inventory/audits are required.
