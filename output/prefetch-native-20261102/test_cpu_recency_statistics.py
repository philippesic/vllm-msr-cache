import pytest
from analyze_cpu_recency_confirmation import inference

def test_twelve_complete_pairs_are_required():
 with pytest.raises(AssertionError):inference([.02]*11)

def test_positive_but_noisy_mean_is_not_proof():
 assert not inference([.05,-.04]*6)['passed_statistics']

def test_consistent_gain_passes_both_tests():
 r=inference([.02,.021,.018,.022]*3)
 assert r['passed_statistics'] and r['one_sided_exact_sign_flip_p']==1/4096

def test_heterogeneous_positive_trace_effects_can_be_confirmed_with_more_pairs():
 r=inference([.037,.004,.014]*4)
 assert r['passed_statistics'] and r['lower95_gain_fraction']>0
