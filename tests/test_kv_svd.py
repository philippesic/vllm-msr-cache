import pytest
import torch
from semantic_offload.kv_svd import encode, decode


def test_actual_kv_low_rank_roundtrip():
    torch.manual_seed(19)
    x = (torch.randn(3, 16, 4) @ torch.randn(3, 4, 256)).half()
    u, p = encode(x, 4)
    restored = decode(u, p)
    assert restored.shape == x.shape
    assert (restored.float()-x.float()).norm()/x.float().norm() < .001
    assert u.numel()+p.numel() < x.numel()


def test_rank_error_tradeoff():
    torch.manual_seed(20)
    x = torch.randn(2, 16, 256).half()
    errors = [float((decode(*encode(x, r)).float()-x.float()).norm()) for r in (4, 8, 12)]
    assert errors[0] > errors[1] > errors[2]


def test_slot_replacement_and_unpublished_load(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import numpy as np
    import semantic_offload.kv_svd as codec
    monkeypatch.setattr(codec, 'PIN_MEMORY', False)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda: SimpleNamespace(synchronize=lambda: None))
    class Event:
        def __init__(self, **kwargs): pass
        def record(self, stream): pass
        def synchronize(self): pass
        def elapsed_time(self, end): return 1.0
    monkeypatch.setattr(torch.cuda, 'Event', Event)
    source = torch.randn(3, 16*256).half()
    cache = SimpleNamespace(tensors=[SimpleNamespace(tensor=source, page_size_bytes=16*256*2)],group_data_refs=[[None]])
    transport=codec.SVDTransport(cache, 2, 8, tmp_path/'codec.jsonl')
    def spec(ids): return SimpleNamespace(block_ids=np.array(ids))
    with pytest.raises(RuntimeError, match='unpublished'):
        transport.transfer(0,spec([0]),spec([2]),False)
    transport.transfer(1,spec([0]),spec([0]),True)
    expected=decode(*encode(source[0].reshape(1,16,256),8)).flatten()
    transport.transfer(2,spec([0]),spec([2]),False)
    torch.testing.assert_close(source[2], expected)
    transport.transfer(3,spec([1]),spec([0]),True)
    replacement=decode(*encode(source[1].reshape(1,16,256),8)).flatten()
    transport.transfer(4,spec([0]),spec([2]),False)
    torch.testing.assert_close(source[2], replacement)
    assert transport.allocated_bytes < 2*16*256*2
    directions=[codec.SVDDirection(transport,True),codec.SVDDirection(transport,False)]
    assert len(directions[0].get_finished())==4
    assert directions[1].get_finished()==[]
    transport.shutdown()
    assert transport.factors==[]
