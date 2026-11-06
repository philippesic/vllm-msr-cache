"""V2 telemetry must describe real dispatch and exclude capture/profile work."""
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
