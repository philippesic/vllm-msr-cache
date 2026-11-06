import pytest
from analyze_cpu_simple_confirmation import inference

def test_noisy_positive_mean_fails():
 r=inference([.05,-.04,.05,-.04,.05,-.03]);assert not r['passed_statistics']

def test_consistent_gain_qualifies():
 r=inference([.02,.025,.021,.019,.022,.024]);assert r['passed_statistics'] and r['one_sided_exact_sign_flip_p']==1/64

def test_negative_result_fails():
 assert not inference([-.01]*6)['passed_statistics']

def test_requires_whole_predeclared_sample():
 with pytest.raises(AssertionError):inference([.02]*5)
