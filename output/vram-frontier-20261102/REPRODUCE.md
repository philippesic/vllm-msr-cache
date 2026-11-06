# Reproduction

Run on the recorded WSL RTX 2080 Ti host using `/home/philip/.local/share/venvs/vllm-semantic-cache-ext/bin/python`. This directory is a private, uncommitted experiment overlay. It does not install the candidate into the canonical checkout. Retain the existing AWQ model cache and the pinned public corpus; their identities are in the plans and `REUSE_MANIFEST.json`.

The exact per-point argument vectors and dependency hashes are stored in each plan's `runs` and `pins`. `frontier.py` verifies all pins before every point and rejects completed responses with missing or failed strict proofs. It never converts incomplete requests into proof passes. Existing output names must not be overwritten: to rerun, copy the experiment into a new directory, preserve the source/model/data identities, and regenerate only the output paths and associated plan hashes.

Execution history, in order:

1. `SWEEP_PLAN.json`: original sixteen-point plan; stopped after the valid 2048 pair and failed 1024 candidate attempt.
2. `SWEEP_REMAINDER_PLAN.json`: qualified common pending-transfer observer, fourteen fresh points; stopped at the invalid 1280 candidate point.
3. `CONTINUATION_PLAN.json`: five independent missing points, unchanged qualified source and observer. The invalid 1280 point remains invalid.

The command form is:

```sh
/home/philip/.local/share/venvs/vllm-semantic-cache-ext/bin/python -B frontier.py --plan PLAN.json
```

`ANALYSIS_PLAN.json` identifies the sixteen intended screen points, including the invalid hole and the older-observer 2048 pair. It is an analysis inventory, not a replacement execution history. `REJECTED_ATTEMPTS.json` retains failed attempts separately. Qualification and import-plumbing failures are excluded from performance.

After all GPU work is idle:

```sh
/home/philip/.local/share/venvs/vllm-semantic-cache-ext/bin/python -B -m pytest -q test_analysis.py
/home/philip/.local/share/venvs/vllm-semantic-cache-ext/bin/python -B analyze.py
```

Plotting uses task-local disposable Python packages with versions recorded in `PLOT_REQUIREMENTS.txt`; it does not change the serving environment. The initial task-local venv attempt lacked ensurepip and was discarded; uv target installation was used instead. `plot_results.py` reads the derived results and writes standalone PNG/SVG artifacts. Raw measurements are compressed in each run directory; summaries, comparisons and proof dispositions remain readable JSON. The strict proof validator is pinned by the plans; do not edit it to make an invalid point pass.

The curves describe a finite, actual-output closed-loop public-agent workflow with simulated waits. They are not fixed-work throughput or a production capacity guarantee. Compare `comparisons/` before treating two arms as equal work. CPU offload is fixed at 3.5 GiB and model/runtime memory remains part of the total GPU footprint.

The exact plot commands, after installing the pinned small plotting dependencies, are:

```sh
/home/philip/.local/bin/uv pip install --python /home/philip/.local/share/venvs/vllm-semantic-cache-ext/bin/python --target .plot-packages --cache-dir .plot-download-cache --no-compile-bytecode -r PLOT_REQUIREMENTS.txt
PYTHONPATH="$PWD/.plot-packages" MPLCONFIGDIR="$PWD/.plot-config" /home/philip/.local/share/venvs/vllm-semantic-cache-ext/bin/python -B plot_results.py --input SCREEN_RESULTS.json
PYTHONPATH="$PWD/.plot-packages" MPLCONFIGDIR="$PWD/.plot-config" /home/philip/.local/share/venvs/vllm-semantic-cache-ext/bin/python -B plot_results.py --input SCREEN_RESULTS.json --reference highest-observed
```

`FINAL_DISPOSITION.json` freezes the no-new-confirmation decision before plots. `THRESHOLDS.json/CSV` keep the predeclared high-budget ARC normalization and a separately labelled posthoc highest-observed-ARC sensitivity. `EXTERNAL_REPLICATION.json` references the previous three paired 2,048-block partitions without pooling them. `DURATION.json` separates actual measured traffic/startup, qualification and time outside run lifecycles.

Final cleanup removes the owned `.plot-packages`, download cache, plotting configuration and failed temporary venv after all six standalone artifacts and exact package versions are saved. Large extracted diagnostic JSONs are retained as losslessly verified `.json.gz`; the compact cause/trace receipts remain plain JSON. The serving/model environment and reusable model cache are untouched. See `STORAGE.json` and `FINAL_VALIDATION.json`.
