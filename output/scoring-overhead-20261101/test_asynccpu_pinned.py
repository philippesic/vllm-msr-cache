import weakref
from unittest.mock import patch
import pytest, torch
from semantic_offload.worker import SemanticOffloadingWorker


class Source:
    is_cuda=True
    shape=(3,2,128)
    dtype=torch.float32
    device=torch.device('cuda:0')


def test_owned_pinned_copy_waits_cooperatively_before_return():
    seen=[];source=Source();polls=iter([False,True])
    class Destination:
        def copy_(self,value,non_blocking=False):
            assert value is source
            seen.append(('copy',non_blocking));return self
    owned=Destination()
    class Stream:
        def synchronize(self):seen.append('unexpected_stream_sync')
    stream=Stream()
    class Event:
        def record(self,value):assert value is stream;seen.append('record')
        def query(self):seen.append('query');return next(polls)
        def synchronize(self):seen.append('event_sync')
    def allocate(shape,**kw):
        assert shape==source.shape and kw=={'dtype':source.dtype,'device':'cpu','pin_memory':True}
        seen.append('allocate');return owned
    worker=SemanticOffloadingWorker.__new__(SemanticOffloadingWorker)
    with patch('torch.empty',side_effect=allocate), \
         patch('torch.cuda.current_stream',return_value=stream), \
         patch('torch.cuda.Event',side_effect=Event), \
         patch('semantic_offload.async_scoring.time.sleep',side_effect=lambda delay:seen.append(('sleep',delay))):
        result=worker._copy_to_owned_cpu(source)
    assert result is owned
    assert seen==['allocate',('copy',True),'record','query',('sleep',.0005),'query','event_sync']


class CopyInterrupt(BaseException):pass


@pytest.mark.parametrize('failure',['copy','event_create','event_record','event_query','event_sync','interrupt'])
def test_copy_errors_fence_inside_helper_while_cpu_destination_and_source_are_alive(failure):
    seen=[];source=Source();source_ref=weakref.ref(source);destinations=[]
    class Destination:
        def copy_(self,value,non_blocking=False):
            seen.append('copy_issued')
            if failure=='copy':raise RuntimeError('copy')
            if failure=='interrupt':raise CopyInterrupt('interrupt')
            return self
    class Stream:
        def synchronize(self):
            assert source_ref() is not None
            assert destinations[0]() is not None
            seen.append('inside_frame_stream_fence')
    stream=Stream()
    class Event:
        def __init__(self):
            if failure=='event_create':raise RuntimeError('event_create')
        def record(self,_):
            if failure=='event_record':raise RuntimeError('event_record')
        def query(self):
            if failure=='event_query':raise RuntimeError('event_query')
            return True
        def synchronize(self):
            seen.append('event_sync')
            if failure=='event_sync':raise RuntimeError('event_sync')
    def allocate(*a,**kw):
        owned=Destination();destinations.append(weakref.ref(owned));return owned
    worker=SemanticOffloadingWorker.__new__(SemanticOffloadingWorker)
    error=CopyInterrupt if failure=='interrupt' else RuntimeError
    with patch('torch.empty',side_effect=allocate), \
         patch('torch.cuda.current_stream',return_value=stream), \
         patch('torch.cuda.Event',side_effect=Event), \
         pytest.raises(error,match=failure):
        worker._copy_to_owned_cpu(source)
    assert seen[0]=='copy_issued'
    assert seen[-1]=='inside_frame_stream_fence'
    assert seen.count('inside_frame_stream_fence')==1


def test_cpu_source_still_returns_owned_detached_values_without_cuda():
    source=torch.arange(48,dtype=torch.float32).reshape(3,2,8)[:,:,::2].requires_grad_()
    worker=SemanticOffloadingWorker.__new__(SemanticOffloadingWorker)
    with patch('torch.cuda.current_stream',side_effect=AssertionError('CPU source touched CUDA')):
        owned=worker._copy_to_owned_cpu(source)
    assert torch.equal(owned,source)
    assert owned.data_ptr()!=source.data_ptr()
    assert owned.device.type=='cpu' and not owned.requires_grad and owned.grad_fn is None
    with torch.no_grad():source.zero_()
    assert owned.count_nonzero()>0
