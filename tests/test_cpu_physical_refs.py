"""CPU slot zero is real data; GPU block zero is the native null sentinel."""
import pytest
from vllm.v1.kv_offload.base import GPULoadStoreSpec
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from semantic_offload.connector import _physical_block_refs


@pytest.mark.parametrize('ids', [[0], [0, 5, 2], [7, 0, 3]])
def test_cpu_slot_zero_keeps_its_ordered_physical_identity(ids):
    assert _physical_block_refs(CPULoadStoreSpec(ids)) == [(0, i) for i in ids]


def test_gpu_null_block_remains_excluded():
    assert _physical_block_refs(GPULoadStoreSpec([0, 5], [2], [0])) == [(0, 5)]


def test_equal_gpu_ids_in_different_groups_remain_distinct():
    spec = GPULoadStoreSpec([0, 5, 5, 0], [2, 2], [0, 0])
    assert _physical_block_refs(spec) == [(0, 5), (1, 5)]
