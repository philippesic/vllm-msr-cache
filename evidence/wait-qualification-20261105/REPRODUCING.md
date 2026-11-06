# Reproducing and checking this campaign

Install the exact Python 3.12 runtime as described in
[vendor/VLLM.md](../../vendor/VLLM.md), including the pinned wheel, dependency
closure, native patches and editable extension. The generic package extra or
`uv sync` does not reproduce this runtime. On the tested 2080 Ti, use FP16,
the maintained launcher and its WSL pinned-memory setting. Use only an idle GPU;
the coordinator reserves both maintained locks for its entire invocation.

Frozen inputs and raw results live in `output/wait-qualification-20261105/`.
Every campaign includes `spec.json`, its exact `corpus.json`, base and per-case
effective plans, executed-source archive, package/model identities, process and
runtime/allocation evidence where collected, every raw response/tool output, sampling and a hash seal. Stage 2–3 published-variant campaigns directly observe CPU pinning and tensor byte geometry; Stage 1 lacks that observer receipt. Stage 4 requires its separate startup tensor observation.
External experiments additionally freeze published sources and executed inputs.
Earlier failed attempts remain separate; do not pool them into replacement runs.

With a retained spec and a **new** output directory:

```bash
PY=/home/philip/.local/share/venvs/vllm-semantic-cache-ext/bin/python
$PY -m benchmarks.research_checks.wait_hint_qualification \
  --spec output/wait-qualification-20261105/stage1-live-spec.json \
  --output output/wait-reproduction-new
$PY -m benchmarks.research_checks.wait_hint_qualification \
  --verify --output output/wait-reproduction-new
$PY -m benchmarks.research_checks.qualification_summary \
  output/wait-reproduction-new --output output/wait-reproduction-summary.json
$PY -m benchmarks.research_checks.qualification_quality \
  output/wait-reproduction-new --output output/wait-reproduction-quality.json
```

Use the corresponding retained spec for each later cell. These are specialized
source-frozen research coordinators built on the maintained suite. Ordinary new
experiments should use [the four-tier commands](../../docs/BENCHMARKING.md).
Artifact verification establishes integrity and aggregate consistency; it does
not convert a measurement pass into a policy win or certify answer quality.

The specs contain this host's absolute paths. On a different checkout, relocate
the structured model, project, Python, public-repository and published-runtime
paths before execution, retaining the original spec as provenance. Preserve the
embedded frozen context text, sampled token data, conditions, ordering, namespaces
and byte budgets. Do not silently change input files to satisfy a hash mismatch.
Record relocated paths and any resulting prompt differences. Published runtime
ports or compatibility changes are new, separately scoped experiments.

Model weights are not in the source/evidence package. Retrieve the model revisions
named in `model-files.json` and compare their hashes with the retained identities.
Independent public source checkouts are more-itertools revision
`f42daddf205f335fea6e200dad46049f33e71221` and Python PEPs revision
`28581cca58bda3ad476e8a210be6df4733e88c97`; source/input hashes are retained
alongside the frozen evaluation and reserve corpora. The public repositories are
[more-itertools](https://github.com/more-itertools/more-itertools) and
[python/peps](https://github.com/python/peps).

Recompute paired whole-run estimates and inspect all offered work, output lengths,
paging, cache activity and quality. Do not treat requests as independent timing
replicates, average arbitrary workload families into a population claim, remove
slow/failed tasks, or pool the original and fresh confirmation sets. The trace
and forced-token replay are labeled diagnostics; extra tracing is not primary
performance evidence. General production stability and research novelty remain
separate questions.

## Isolated installation exercise

Use a separate source copy and a new virtual environment, preserving the primary
checkout and environment. The author-run exercise uses the same commands as the
runtime instructions, with cached exact packages preferred:

```bash
uv venv --python 3.12 /path/to/isolated/.venv
uv pip install --offline --python /path/to/isolated/.venv/bin/python \
  -r /path/to/isolated/source/requirements-runtime.txt
cd /path/to/isolated/source
uv pip install --offline --python /path/to/isolated/.venv/bin/python --no-deps -e .
make runtime-check PY=/path/to/isolated/.venv/bin/python
make test PY=/path/to/isolated/.venv/bin/python
```

Offline installation requires the exact wheel/dependency closure already cached;
otherwise omit `--offline` and retain the actual retrieval/installation log. Do
not substitute a different vLLM release. The maintained launcher overlays the
pinned native sources and preserves the wheel's compiled components. Module-path
proof must identify this copy and environment, rather than importing the primary
editable checkout accidentally. This exercise is isolated author reproduction,
not an independent person's reproduction or a different machine.

The supervised minimal client invocation owns a server, runs the public standalone
client with a real SHA-256 subprocess, requires positive ready-CPU-key refresh,
and verifies drain and cleanup:

```bash
$PY -m benchmarks.research_checks.qualification_example \
  --spec output/wait-qualification-20261105/example-client-spec.json \
  --output output/wait-example-new
```

For isolation, relocate the structured project, Python and task-file paths in a
copy of that spec, use the isolated interpreter and source working directory,
retain shared model weights read-only, and choose a fresh result directory. Keep
the original spec and the exact relocation receipt. Successful tiny integration
checks establish the documented flow, not application speed or production uptime.

The retained isolation includes `installed-requirements.txt`, a noneditable
`dependency-closure.txt`, package-version comparison, module-path proof and all
installation/test logs under `isolated-reproduction/`. The declared requirements
pin major packages but leave additional transitive dependencies to resolution:
the actual offline install selected 201 dependencies plus the extension, with
seven transitive versions different from the primary environment. No versions
were tuned after outcomes and the primary environment was untouched. Use the
observed closure for the isolated example's environment; do not claim a bitwise
environment clone or reproduction of benchmark timing from this exercise. The
measurement identities and primary package reference remain separate provenance.
