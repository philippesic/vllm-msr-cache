"""Bit-level controls for the opt-in loading-only AWQ memory reduction."""

import copy
from types import SimpleNamespace

import pytest
import torch

from vllm import envs
from vllm.model_executor import parameter
from vllm.model_executor.layers.quantization.auto_awq import (
    AutoAWQMarlinLinearMethod,
    _convert_awq_to_standard_format,
    _repack_awq_weight_rows,
)


def scalar_reference(weights):
    order = [0, 4, 1, 5, 2, 6, 3, 7]
    rows, packed_columns = weights.shape
    output = torch.empty((rows // 8, packed_columns * 8), dtype=torch.int32)
    for row in range(rows // 8):
        for column in range(packed_columns * 8):
            word = 0
            for slot in range(8):
                value = int(weights[row * 8 + slot, column // 8])
                word |= ((value >> (4 * order[column % 8])) & 15) << (4 * slot)
            output[row, column] = word if word < 2**31 else word - 2**32
    return output


@pytest.mark.parametrize("shape", [(8, 1), (24, 5), (520, 3)])
@pytest.mark.parametrize("chunk_rows", [8, 16, 512, 1024])
def test_chunking_preserves_every_packed_word(shape, chunk_rows):
    generator = torch.Generator().manual_seed(901)
    weights = torch.randint(-(2**31), 2**31, shape, dtype=torch.int32, generator=generator)
    # Exercise signed high bits and unusual strides as well as partial chunks.
    weights[0, 0] = -(2**31)
    weights[-1, -1] = 2**31 - 1
    weights = weights.t().contiguous().t()
    before = weights.clone()
    result = _repack_awq_weight_rows(weights, 4, chunk_rows)
    assert torch.equal(result, scalar_reference(weights))
    assert torch.equal(weights, before)
    assert result.is_contiguous() and result.dtype == torch.int32


@pytest.mark.parametrize("value", [0, -1, -(2**31), 2**31 - 1])
def test_extreme_bit_patterns(value):
    weights = torch.full((24, 5), value, dtype=torch.int32)
    assert torch.equal(_repack_awq_weight_rows(weights, 4, 16), scalar_reference(weights))


@pytest.mark.parametrize("chunk_rows", [-8, -1, 1, 7, 9])
def test_invalid_chunks_rejected(chunk_rows):
    with pytest.raises(ValueError, match="chunk rows"):
        _repack_awq_weight_rows(torch.zeros((24, 5), dtype=torch.int32), 4, chunk_rows)


@pytest.mark.parametrize("weights,bits", [
    (torch.zeros((24, 5), dtype=torch.int64), 4),
    (torch.zeros((24, 5), dtype=torch.int32), 8),
    (torch.zeros((25, 5), dtype=torch.int32), 4),
    (torch.zeros((24,), dtype=torch.int32), 4),
])
def test_unsupported_layout_rejected(weights, bits):
    with pytest.raises(ValueError):
        _repack_awq_weight_rows(weights, bits, 8)


def test_whole_layer_preserves_zero_points_scales_and_parameter_layout(monkeypatch):
    # Packed parameters require TP identity even for this CPU-only TP=1 case.
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 1)
    layer = torch.nn.Module()
    generator = torch.Generator().manual_seed(911)
    for name, shape in (("qweight", (520, 7)), ("qzeros", (5, 7))):
        tensor = torch.randint(-(2**31), 2**31, shape, dtype=torch.int32, generator=generator)
        layer.register_parameter(name, torch.nn.Parameter(tensor, requires_grad=False))
    layer.register_parameter("scales", torch.nn.Parameter(torch.ones((5, 56), dtype=torch.float16), requires_grad=False))
    reference = copy.deepcopy(layer)
    monkeypatch.setattr(envs, "VLLM_AWQ_REPACK_CHUNK_ROWS", 0)
    _convert_awq_to_standard_format(reference, "qweight", "qzeros", 4)
    monkeypatch.setattr(envs, "VLLM_AWQ_REPACK_CHUNK_ROWS", 512)
    _convert_awq_to_standard_format(layer, "qweight", "qzeros", 4)
    for name in ("qweight", "qzeros", "scales"):
        actual, expected = getattr(layer, name), getattr(reference, name)
        assert torch.equal(actual, expected)
        assert actual.shape == expected.shape and actual.stride() == expected.stride()
        assert type(actual) is type(expected)
        for attribute in ("input_dim", "output_dim", "packed_dim", "packed_factor"):
            assert getattr(actual, attribute, None) == getattr(expected, attribute, None)


def test_loading_cleanup_does_not_initialize_cuda_for_cpu_parameters(monkeypatch):
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(envs, "VLLM_AWQ_REPACK_CHUNK_ROWS", 16)
    def unexpected(*args, **kwargs):
        raise AssertionError("CPU weight processing entered CUDA cleanup")
    for name in ("synchronize", "empty_cache", "device"):
        monkeypatch.setattr(torch.cuda, name, unexpected)
    layer = torch.nn.Module()
    for name, shape in (("qweight", (24, 7)), ("qzeros", (3, 7))):
        layer.register_parameter(name, torch.nn.Parameter(torch.zeros(shape, dtype=torch.int32), requires_grad=False))
    method = object.__new__(AutoAWQMarlinLinearMethod)
    method.quant_config = SimpleNamespace(quant_type=SimpleNamespace(size_bits=4))
    method.kernel = SimpleNamespace(process_weights_after_loading=lambda value: None)
    method.process_weights_after_loading(layer)
    assert layer.qweight.shape == (3, 56)
