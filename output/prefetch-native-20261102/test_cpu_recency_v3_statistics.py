import pytest
from analyze_cpu_recency_v3_confirmation import inference

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


def test_efficiency_ratio_uses_actual_memory_and_rate():
 import math
 from analyze_cpu_recency_v3_confirmation import normalized_log_ratio
 c={'throughput_tokens_s':102,'memory':8};a={'throughput_tokens_s':100,'memory':9}
 assert math.isclose(math.exp(normalized_log_ratio(c,a,'memory')),1.02*9/8)


def test_memory_savings_cannot_substitute_for_throughput_primary():
 from analyze_cpu_recency_v3_confirmation import normalized_log_ratio
 c={'throughput_tokens_s':99,'memory':8};a={'throughput_tokens_s':100,'memory':9}
 assert inference([normalized_log_ratio(c,a,'memory')]*12)['passed_statistics']
 assert not inference([-.01]*12)['passed_statistics']
