# Reproducing this host's campaign

Run only on the qualified RTX2080Ti WSL host and serialize GPU ownership. Preserve the existing output directories; drivers intentionally refuse to overwrite a run. The model cache is reusable and pinned in model-selection.json. The public corpus is already prepared and pinned, with attribution in corpus-notes.md. No corpus fetch is needed for repetition.

Python: `/home/philip/.local/share/venvs/vllm-semantic-cache-ext/bin/python`.
Evidence root: `/home/philip/vllm-semantic-cache-ext/output/quantized-realworld-20261101`.

The exact server commands, package/model settings, request parameters, prompt/output token IDs, hints, timing/resource samples and identities are in each `runs/NAME/raw.json.gz`. The exact driver commands and order are in:

- `CONFIRMATION_PLAN.json`: old24; its agent candidate seed1103 has one invalid proof and is never repaired retroactively. Old valid chat results remain useful.
- `AGENT_READMISSION_PLAN.json`: new12 agent runs, common observer-only V2 repair, strict fail-closed orchestration.
- `DIAGNOSTIC_PLAN.json`: seven separate bounded quality, abandonment and finite offered-load diagnostics. The shadow quality run is excluded from performance comparisons.

Each plan contains file hashes checked before each run. The new observer is common to all four arms. Candidate sources, original reference sources and runtime overlays are frozen under `sources/` and `runtime*/`; canonical checkout changes made by others were preserved. `PRIVATE_PATCH.diff` and `PRIVATE_PATCH_MANIFEST.json` identify the private connector and two vLLM overlay changes. These are a diagnostic private branch; applying the diff alone to an arbitrary vLLM version is not a production installation recipe.

To repeat the new agent matrix, create a new plan changing only output run names and progress filename. This keeps the model, source, traffic, budgets, order and checked pins fixed:

```bash
cd /home/philip/vllm-semantic-cache-ext
/home/philip/.local/share/venvs/vllm-semantic-cache-ext/bin/python -B - <<'PY'
import json
from pathlib import Path
h=Path('output/quantized-realworld-20261101')
p=json.loads((h/'AGENT_READMISSION_PLAN.json').read_text())
for run in p['runs']:
    old=run['name']; run['name']='repeat-'+old
    run['command']=[run['name'] if x==old else x for x in run['command']]
p['progress_file']='repeat-agent-progress.json'
(h/'REPEAT_AGENT_PLAN.json').write_text(json.dumps(p,indent=2)+'\n')
PY
/home/philip/.local/share/venvs/vllm-semantic-cache-ext/bin/python -B \
  output/quantized-realworld-20261101/confirm_strict.py \
  --plan output/quantized-realworld-20261101/REPEAT_AGENT_PLAN.json
```

The exact 640-block preemption correctness stress command is recorded in `runs/qualify-v2-readmission-public-agent640-v2/raw.json.gz` and uses run_readmission.py with candidate/shared-lookup, CPU4096, active8, clients8, agent3turn/EOS96, bursty waits and `--audit`. It exercises native preemption but is not a performance workload. The dedicated local-prefix audit is separately recorded in `runs/qualify-frozen-hooks-native-bytes/`; absence of configured local snapshots is explicitly not an audit pass.

Per-run summary and pair comparison (read only, no GPU):

```bash
/home/philip/.local/share/venvs/vllm-semantic-cache-ext/bin/python -B \
  output/quantized-realworld-20261101/campaign_metrics.py summarize RUN_DIRECTORY \
  --output RUN_DIRECTORY/summary.json
/home/philip/.local/share/venvs/vllm-semantic-cache-ext/bin/python -B \
  output/quantized-realworld-20261101/campaign_metrics.py compare ARC_RUN_DIRECTORY CANDIDATE_RUN_DIRECTORY \
  --output COMPARISON.json
```

A valid schema is insufficient: require `proofs.all_passed`, proof count equal request count, no request failures and no owned descendants. Cross-arm `performance_claim_eligible` additionally requires exact prompts, outputs and native keys; agent generated histories do not meet that criterion in this campaign. Repeated finite closed-loop runs are not a sustainable capacity test. Explicit offered-load probes remain finite first-turn controls.

Source/allocator/helper tests and CPU invalid controls are documented in qualification/*.log; actual-model qualification receipts, rather than helper tests alone, establish the observed runtime boundaries. All timing claims exclude intrusive audit/shadow runs. No commit or push was made.
