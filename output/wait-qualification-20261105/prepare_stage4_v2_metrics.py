"""Prepare opt-in V2 graph-stat propagation; apply only after Stage3 is quiet."""
import argparse, ast, difflib, hashlib, json, time
from pathlib import Path

b = Path(__file__).resolve().parent
root = b.parents[1]
native = 'vendor/vllm/vllm/v1/worker/gpu/model_runner.py'
tests = 'tests/test_v2_cudagraph_metrics.py'
patch_name = 'patches/vllm-v2-cudagraph-metrics.patch'

METHOD = '''    def _cudagraph_statistics(
        self, input_batch: InputBatch, batch_desc: BatchExecutionDescriptor,
        *, dummy_run: bool, is_profile: bool,
    ) -> CUDAGraphStat | None:
        # Report the mode actually dispatched, excluding capture/profiling.
        if not self.observability_config.cudagraph_metrics or dummy_run or is_profile:
            return None
        return CUDAGraphStat(
            num_unpadded_tokens=input_batch.num_tokens,
            num_padded_tokens=input_batch.num_tokens_after_padding,
            num_paddings=input_batch.num_tokens_after_padding - input_batch.num_tokens,
            runtime_mode=str(batch_desc.cg_mode),
        )

'''

TESTS = '''"""V2 telemetry must describe real dispatch and exclude capture/profile work."""
from types import SimpleNamespace

import pytest
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.model_runner import GPUModelRunner


@pytest.mark.parametrize('enabled,dummy,profile', [
    (False, False, False), (True, True, False), (True, False, True),
])
def test_unobserved_paths_never_read_batch_or_emit_statistics(enabled, dummy, profile):
    runner = SimpleNamespace(observability_config=SimpleNamespace(cudagraph_metrics=enabled))
    # These sentinels have no batch metadata: disabled/capture paths must not read it.
    assert GPUModelRunner._cudagraph_statistics(runner, object(), object(),
        dummy_run=dummy, is_profile=profile) is None


@pytest.mark.parametrize('mode', [CUDAGraphMode.FULL, CUDAGraphMode.NONE])
def test_observed_mode_and_padding_come_from_actual_execution_descriptor(mode):
    runner = SimpleNamespace(observability_config=SimpleNamespace(cudagraph_metrics=True))
    batch = SimpleNamespace(num_tokens=3, num_tokens_after_padding=4)
    observed = GPUModelRunner._cudagraph_statistics(runner, batch,
        SimpleNamespace(cg_mode=mode), dummy_run=False, is_profile=False)
    assert observed.num_unpadded_tokens == 3 and observed.num_padded_tokens == 4
    assert observed.num_paddings == 1 and observed.runtime_mode == str(mode)
'''


def prepare():
    before = (root/native).read_text()
    after = before
    changes = [
        ('from vllm.compilation.counter import compilation_counter',
         'from vllm.compilation.counter import compilation_counter\nfrom vllm.compilation.cuda_graph import CUDAGraphStat'),
        ('    @torch.inference_mode()\n    def execute_model(', METHOD+'    @torch.inference_mode()\n    def execute_model('),
        ('            finished_req_ids=finished_req_ids,\n        )',
         '''            finished_req_ids=finished_req_ids,
            cudagraph_stats=self._cudagraph_statistics(
                input_batch, batch_desc, dummy_run=dummy_run, is_profile=is_profile
            ),
        )'''),
        ('        aux_hidden_states = self.execute_model_state.aux_hidden_states\n        finished_req_ids = self.execute_model_state.finished_req_ids\n        self.execute_model_state = None',
         '        aux_hidden_states = self.execute_model_state.aux_hidden_states\n        finished_req_ids = self.execute_model_state.finished_req_ids\n        cudagraph_stats = self.execute_model_state.cudagraph_stats\n        self.execute_model_state = None'),
        ('            prompt_logprobs_dict=prompt_logprobs_dict,  # type: ignore[arg-type]\n        )',
         '            prompt_logprobs_dict=prompt_logprobs_dict,  # type: ignore[arg-type]\n            cudagraph_stats=cudagraph_stats,\n        )'),
        ('    finished_req_ids: set[str]\n\n\ndef sort_batch_req_ids(',
         '    finished_req_ids: set[str]\n    cudagraph_stats: CUDAGraphStat | None = None\n\n\ndef sort_batch_req_ids('),
    ]
    for old,new in changes:
        if after.count(old) != 1:
            raise RuntimeError('Unexpected native source context: '+old[:80])
        after = after.replace(old,new,1)
    ast.parse(after,filename=native)
    ast.parse(TESTS,filename=tests)
    native_patch = ''.join(difflib.unified_diff(before.splitlines(keepends=True),
        after.splitlines(keepends=True), fromfile='a/vllm/v1/worker/gpu/model_runner.py',
        tofile='b/vllm/v1/worker/gpu/model_runner.py'))
    return before, after, native_patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    before, after, native_patch = prepare()
    if (root/tests).exists() or (root/patch_name).exists():
        raise FileExistsError('Preserve/review existing V2 repair files')
    files = {native: dict(before_sha256=hashlib.sha256(before.encode()).hexdigest(),
                         after_sha256=hashlib.sha256(after.encode()).hexdigest()),
             tests: dict(before_sha256=None, after_sha256=hashlib.sha256(TESTS.encode()).hexdigest()),
             patch_name: dict(before_sha256=None, after_sha256=hashlib.sha256(native_patch.encode()).hexdigest())}
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    prepared = b/'stage4-v2-graph-prepared.json'
    if args.apply:
        expected = json.loads(prepared.read_text())
        if expected['files'] != files or expected['script_sha256'] != script_sha:
            raise RuntimeError('Native source/prepared script changed; review first')
        applied = b/'stage4-v2-graph-applied.json'
        if applied.exists():
            raise FileExistsError('Preserve prior V2 repair attempt')
        backup = b/'stage4-repair-before'/native
        backup.parent.mkdir(parents=True,exist_ok=True); backup.write_text(before)
        (root/native).write_text(after)
        (root/tests).write_text(TESTS)
        (root/patch_name).write_text(native_patch)
        applied.write_text(json.dumps(dict(applied_unix=time.time(),files=files,
            scope='Opt-in CUDAGraphStat propagation through V2 ExecuteModelState and ModelRunnerOutput to existing scheduler/frontend logger. Excludes dummy/profile forwards. No graph dispatch, kernel, allocation or policy change. Tests and GPU proof still required.'),indent=2)+'\n')
        print('Applied V2 native telemetry + retained patch + five CPU test cases; validation pending')
    else:
        if prepared.exists():
            raise FileExistsError('Preserve prior V2 preparation')
        (b/'stage4-v2-cudagraph-metrics.patch').write_text(native_patch)
        prepared.write_text(json.dumps(dict(prepared_unix=time.time(),files=files,
            script_sha256=script_sha,ast_parsed=True,applied=False,
            reason='Actual selected V2 runner lacked CUDAGraphStat construction/propagation; native CLI flag alone would not produce execution receipts. Prepare before any Stage4 measurement.'),indent=2)+'\n')
        print('Prepared V2 telemetry patch and five tests; watched source unchanged')


if __name__ == '__main__':
    main()
