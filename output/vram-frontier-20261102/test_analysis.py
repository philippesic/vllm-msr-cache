from analyze import thresholds,selection,nondominated,highest_observed_sensitivity

def point(name,arm,budget,tps,ttft,activity=True):
 return {'name':name,'phase':'screen','seed':1103,'arm':arm,'gpu_blocks':budget,'complete_load_eligible':True,'TPS':tps,'TTFT_p95_ms':ttft,'absolute15s_TTFT_pass':ttft<=15000,'absolute_SLO_pass':ttft<=15000,'numerical_calls':1 if activity else 0,'native_adoptions':1 if activity else 0,'CUDA_allocated_peak_bytes':budget*1000}

def test_predeclared_activity_guard_skips_disabled_false_win():
 rows=[point('anchor','arc',4096,100,10000),point('a1024','arc',1024,98,11000),point('c1024','candidate',1024,99.5,10400),point('a768','arc',768,98,11000),point('c768','candidate',768,99.5,10200,False)];targets=thresholds(rows,4096);r=selection(rows);assert r['proposed_budget_blocks']==1024
 assert next(t for t in targets if t['arm']=='candidate' and t['target_TPS_fraction']==.99 and t['TTFT_tolerance']==1.05)['actual_scoring_and_adoption'] is False

def test_relative_target_does_not_invent_absolute_SLO_pass():
 rows=[point('anchor','arc',4096,100,20000),point('c','candidate',1024,100,20500)];targets=thresholds(rows,4096);t=next(t for t in targets if t['arm']=='candidate' and t['TTFT_tolerance']==1.05);assert t['smallest_measured_budget_blocks']==1024 and t['absolute15s_TTFT_pass'] is False

def test_censored_point_does_not_win_or_define_complete_curve():
 anchor=point('anchor','arc',4096,100,10000);c=point('c','candidate',768,999,1);c['complete_load_eligible']=False
 targets=thresholds([anchor,c],4096);assert all(t['smallest_measured_budget_blocks'] is None for t in targets if t['arm']=='candidate');assert 'c' not in nondominated([anchor,c])

def test_nondominated_requires_three_objectives():
 a=point('a','arc',2048,100,10000);b=point('b','candidate',3072,99,11000);c=point('c','candidate',1024,90,12000);assert set(nondominated([a,b,c]))=={'a','c'}

def test_posthoc_anchor_sensitivity_keeps_predeclared_reference():
 rows=[point('large','arc',4096,100,20000),point('fast','arc',3072,103,20000),point('c','candidate',2048,99,20000)]
 primary=thresholds(rows,4096);secondary=highest_observed_sensitivity(rows)
 assert rows[2]['TPS_fraction_highARC']==.99
 assert rows[2]['TPS_fraction_highest_observedARC']==99/103
 assert secondary[0]['anchor']['gpu_blocks']==3072 and secondary[0]['posthoc']
 assert next(x for x in primary if x['arm']=='candidate' and x['target_TPS_fraction']==.99)['smallest_measured_budget_blocks']==2048
 assert next(x for x in secondary[0]['thresholds'] if x['arm']=='candidate' and x['target_TPS_fraction']==.99)['smallest_measured_budget_blocks'] is None
