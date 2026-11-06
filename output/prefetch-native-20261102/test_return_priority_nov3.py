import pytest
from return_priority_nov3 import gateway_priority
from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request
from vllm.v1.core.sched.request_queue import PriorityRequestQueue

def request(name,at_ms,turn,credit):
    return Request(request_id=name,prompt_token_ids=[1],sampling_params=SamplingParams(max_tokens=1),pooling_params=None,arrival_time=at_ms/1000,priority=gateway_priority(1_000_000_000+at_ms*1_000_000,1_000_000_000,turn,credit))

def ordered(*requests):
    q=PriorityRequestQueue()
    for r in requests:q.add_request(r)
    return [q.pop_request().request_id for _ in requests]

def test_zero_credit_uses_exact_native_FIFO_arrival_order():
    a=request('a',0,0,0);b=request('b',1,1,0)
    assert a.priority==b.priority==0 and ordered(b,a)==['a','b']

@pytest.mark.parametrize('credit',[2000,8000])
def test_return_credit_can_overtake_younger_than_credit_initial(credit):
    assert ordered(request('initial',0,0,credit),request('return',credit-1,1,credit))==['return','initial']

@pytest.mark.parametrize('credit',[2000,8000])
def test_initial_older_than_credit_cannot_be_overtaken(credit):
    assert ordered(request('return',credit+1,1,credit),request('initial',0,0,credit))==['initial','return']

@pytest.mark.parametrize('credit',[2000,8000])
def test_new_initial_cannot_overtake_old_return(credit):
    assert ordered(request('initial',credit+5000,0,credit),request('return',0,1,credit))==['return','initial']

@pytest.mark.parametrize('turn',[1,2,12])
def test_credit_is_fixed_not_cumulative_by_turn(turn):
    assert gateway_priority(11_000_000_000,1_000_000_000,turn,2000)==8000

def test_relative_clock_shift_invariance():
    assert gateway_priority(11_000_000_000,1_000_000_000,1,2000)==gateway_priority(21_000_000_000,11_000_000_000,1,2000)

@pytest.mark.parametrize('args',[(True,0,0,0),(1,False,0,0),(1,0,True,0),(1,0,0,False),(1.,0,0,0)])
def test_types_rejected(args):
    with pytest.raises(TypeError):gateway_priority(*args)

@pytest.mark.parametrize('args',[(0,1,0,0),(1,-1,0,0),(1,0,-1,0),(1,0,0,1000)])
def test_invalid_clock_turn_credit_rejected(args):
    with pytest.raises(ValueError):gateway_priority(*args)

def test_fifo_queue_tie_keeps_native_request_order_rule():
    a=request('a',2000,0,2000);b=request('b',4000,1,2000)
    assert a.priority==b.priority and ordered(b,a)==['a','b']
