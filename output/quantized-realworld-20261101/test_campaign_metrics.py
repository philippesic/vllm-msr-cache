"""CPU-only checks including adversarial mutations of real model audit events."""
import copy
import gzip
import importlib.util
import json
from pathlib import Path
import unittest
import tempfile

H=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('campaign_metrics',H/'campaign_metrics.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
AUDIT=H/'runs/qualify-actual-bytes-v2/raw.json.gz'


def real_fixture():
    raw=json.loads(gzip.decompress(AUDIT.read_bytes()))
    turns={}
    for row in raw['requests']:
        sid=row['session_id'];row['turn']=turns.get(sid,0);turns[sid]=row['turn']+1;row['cohort']='agent'
    raw.update(corpus_identity={'fixture':'actual audit'},model_identity={'fixture':'actual Qwen AWQ audit'},performance_eligible=True,planned_requests=len(raw['requests']))
    return raw


def matched_event(raw):
    rid=raw['requests'][0]['payload']['request_id']
    return next(e for e in raw['terminals'] if e.get('external_request_id')==rid)


class MetricsTests(unittest.TestCase):
    def test_actual_complete_native_audit(self):
        summary=m.summarize(real_fixture())
        self.assertTrue(summary['valid'],summary['errors'])
        self.assertTrue(summary['proofs']['all_passed'],summary['proofs'])
        self.assertEqual(summary['proofs']['passed'],6)
        self.assertGreater(summary['counters']['scored_candidates'],0)
        self.assertGreater(summary['counters']['prefetch_consumed_bytes'],0)

    def test_independent_identity_not_order_or_index(self):
        a=real_fixture();b=copy.deepcopy(a);b['requests'].reverse()
        for row in b['requests']:row['index']+=10000
        compared=m.compare(a,b)
        self.assertTrue(compared['identity_exact'])
        self.assertTrue(compared['matched_correctness_passed'])

    def test_actual_wrong_prompt_rejected(self):
        a=real_fixture();b=copy.deepcopy(a);b['requests'][0]['payload']['prompt'][0]+=1
        compared=m.compare(a,b)
        self.assertEqual(len(compared['inventory']['prompt_mismatches']),1)
        self.assertFalse(compared['right_summary']['proofs']['all_passed'])
        self.assertFalse(compared['performance_claim_eligible'])

    def test_actual_wrong_output_rejected_without_waiver(self):
        a=real_fixture();b=copy.deepcopy(a);b['requests'][0]['output_token_ids'][0]+=1
        compared=m.compare(a,b)
        self.assertEqual(len(compared['inventory']['output_mismatches']),1)
        self.assertFalse(compared['matched_correctness_passed'])

    def test_actual_wrong_native_key_rejected(self):
        a=real_fixture();b=copy.deepcopy(a)
        matched_event(b)['ordered_key_positions'][0]['key']='definitely-wrong'
        compared=m.compare(a,b)
        self.assertEqual(len(compared['inventory']['key_mismatches']),1)
        self.assertFalse(compared['matched_correctness_passed'])

    def test_actual_wrong_physical_map_rejected(self):
        a=real_fixture();b=copy.deepcopy(a)
        matched_event(b)['physical_block_table_provenance']['group_block_ids'][0][0]+=100000
        compared=m.compare(a,b)
        self.assertFalse(compared['right_summary']['proofs']['all_passed'])
        self.assertFalse(compared['matched_correctness_passed'])

    def test_empty_and_partial_proofs_fail_closed(self):
        for change in ('empty','partial','key','physical'):
            raw=real_fixture()
            if change=='empty':raw['terminals']=[]
            if change=='partial':raw['terminals']=raw['terminals'][:-1]
            if change=='key':matched_event(raw)['ordered_key_positions']=[]
            if change=='physical':matched_event(raw).pop('physical_block_table_provenance')
            self.assertFalse(m.summarize(raw)['proofs']['all_passed'],change)

    def test_bad_timing_controls(self):
        for change in ('first','receipt','end','release','count'):
            raw=real_fixture();row=raw['requests'][0]
            if change=='first':row['first_token_ns']+=1
            if change=='receipt':row['token_receipt_ns'][1]=row['token_receipt_ns'][0]-1
            if change=='end':row['request_end_ns']=raw['load_end_ns']+1
            if change=='release':row['intended_release_ns']=row['request_start_ns']+1
            if change=='count':row['token_receipt_ns'].pop()
            summary=m.summarize(raw)
            self.assertFalse(summary['valid'],change)
            self.assertFalse(m.compare(real_fixture(),raw)['performance_claim_eligible'],change)

    def test_identity_missing_duplicate_and_request_loss(self):
        a=real_fixture();b=copy.deepcopy(a);b['requests'][0].pop('turn')
        with self.assertRaises(ValueError):m.compare(a,b)
        b=copy.deepcopy(a);b['requests'].append(copy.deepcopy(b['requests'][0]))
        with self.assertRaises(ValueError):m.compare(a,b)
        b=copy.deepcopy(a);b['requests'].pop()
        compared=m.compare(a,b)
        self.assertEqual(len(compared['inventory']['left_only']),1)
        self.assertFalse(compared['performance_claim_eligible'])

    def test_unknown_resources_are_null(self):
        r=m.resources({},10,20)
        for key in ('cpu_s_sampled','rss_peak_bytes_sampled','nvml_device_memory_peak_bytes_sampled','load_cuda_allocated_peak_bytes_sum_workers'):
            self.assertIsNone(r[key],key)

    def test_stable_cpu_pid_and_start_end_resource_filter(self):
        def sample(at,cpu,rss,ident='10:100'):
            return {'at_ns':at,'processes':[{'identity':ident,'cpu_s':cpu,'rss_bytes':rss}], 'gpu':{'memory_used_bytes':rss,'utilization_pct':20}}
        raw={'resource_samples':[sample(0,0,1000000),sample(10,1,100),sample(15,3,200),sample(20,5,300),sample(30,10,2000000)]}
        r=m.resources(raw,10,20)
        self.assertEqual(r['cpu_s_sampled'],4)
        self.assertEqual(r['rss_peak_bytes_sampled'],300)
        self.assertEqual(r['nvml_device_memory_peak_bytes_sampled'],300)
        self.assertEqual(r['sample_count'],3)
        raw['resource_samples'][2]['processes'][0]['identity']='10:101'
        r=m.resources(raw,10,20);self.assertIsNone(r['cpu_s_sampled']);self.assertEqual(r['cpu_unstable_pids'],['10'])

    def test_missing_process_or_cpu_not_fabricated(self):
        raw={'resource_samples':[{'at_ns':10,'processes':[{'identity':'1:2','cpu_s':2,'rss_bytes':5}]}, {'at_ns':20,'processes':[]}]}
        self.assertIsNone(m.resources(raw,10,20)['cpu_s_sampled'])
        raw['resource_samples'][1]['processes']=[{'identity':'1:2','rss_bytes':6}]
        self.assertIsNone(m.resources(raw,10,20)['cpu_s_sampled'])

    def test_quantiles_and_full_interval_denominator(self):
        rows=[{'success':True,'timing_valid':True,'slo_pass':True,'slo_defined':True,'input_tokens':3,'output_tokens':2,'itl_ms':[100], 'finish_reasons':['stop'],**{k:100 for k in ('release_ttft_ms','submit_ttft_ms','release_e2e_ms','submit_e2e_ms','release_lateness_ms','server_queue_ms')}}]
        a=m.aggregate(rows,10)
        self.assertEqual(a['output_tokens_s'],.2);self.assertEqual(a['requests_s'],.1)
        self.assertEqual(a['goodput_requests_s'],.1)
        self.assertAlmostEqual(m.dist([0,10,20,30])['p95'],28.5)

    def test_scheduler_config_not_achieved(self):
        raw=real_fixture();raw['geometry']['max_num_seqs']=999
        c=m.concurrency(raw,raw['load_start_ns'],raw['load_end_ns'])
        self.assertIsNone(c['running_peak'])
        self.assertGreater(c['client_inflight_peak'],0)

    def test_new_scheduler_samples_filtered(self):
        raw={'requests':[],'postload_export':{'scheduler':{'samples':[
            {'at_ns':5,'running_requests':999,'waiting_requests':999,'scheduled_requests':999},
            {'at_ns':15,'running_requests':4,'waiting_requests':3,'scheduled_requests':2},
            {'at_ns':25,'running_requests':999,'waiting_requests':999,'scheduled_requests':999}]}}}
        c=m.concurrency(raw,10,20)
        self.assertEqual(c['running_peak'],4);self.assertEqual(c['queued_peak'],3);self.assertEqual(c['scheduled_peak'],2)

    def test_all_step_aggregates_need_complete_coverage(self):
        counts={'native_scheduler_calls':10,'native_running_max':4,'native_running_sum':30,'native_running_hist_2':5,'native_running_hist_4':5}
        raw={'postload_export':{'scheduler':{'counts':counts}}}
        c=m.concurrency(raw,10,20)
        self.assertEqual(c['running_peak'],4);self.assertEqual(c['running_mean'],3)
        self.assertEqual(c['running_distribution']['n'],10)
        self.assertEqual(c['running_distribution']['p50'],3)
        counts['native_running_hist_4']=4
        self.assertIsNone(m.concurrency(raw,10,20)['running_peak'])

    def test_both_arms_missing_same_request_cannot_claim_success(self):
        a=real_fixture();a['requests'].pop();b=copy.deepcopy(a)
        c=m.compare(a,b)
        self.assertTrue(c['identity_exact']);self.assertFalse(c['performance_claim_eligible'])

    def test_slo_failure_does_not_hide_completed_work(self):
        raw=real_fixture();row=raw['requests'][0];events=m.terminal_index(raw);module,_=m.validator()
        # Set an authentic row's client receipt times to a declared slow response.
        row.update(intended_release_ns=0,request_start_ns=0,first_token_ns=16_000_000_000,request_end_ns=20_000_000_000)
        row['token_receipt_ns']=[16_000_000_000+i*1_000_000 for i in range(len(row['output_token_ids']))]
        r=m.request_metrics(row,0,21_000_000_000,events,module)
        self.assertTrue(r['timing_valid']);self.assertFalse(r['slo_pass'])
        a=m.aggregate([r],21)
        self.assertEqual(a['completed_valid'],1);self.assertEqual(a['slo_good_requests'],0)

    def test_missing_counter_zero_requires_exact_reset_evidence(self):
        raw=real_fixture();start=raw['load_start_ns'];end=raw['load_end_ns']
        self.assertIsNone(m.counters(raw,start,end)['worker_prefix_mean_scoring_delta']['numerical_calls'])
        raw['reset_scope']={'prefix_mean_scoring_stats':'explicit clear in worker and scheduler at pre-load reset; absent Counter keys therefore zero'}
        raw['before_load_export']['workers'][0]['metric_reset_scope']={'missing_counter_after_explicit_reset_is_zero':True}
        self.assertEqual(m.counters(raw,start,end)['worker_prefix_mean_scoring_delta']['numerical_calls'],4)
        raw['before_load_export']['workers'][0]['metric_reset_scope']['missing_counter_after_explicit_reset_is_zero']=False
        self.assertIsNone(m.counters(raw,start,end)['worker_prefix_mean_scoring_delta']['numerical_calls'])

    def test_request_p95_itl_allows_two_outliers_but_preserves_max_violations(self):
        raw=real_fixture();module,_=m.validator();events=m.terminal_index(raw)
        for outliers,expected in [(2,True),(6,False)]:
            row=copy.deepcopy(raw['requests'][0]);row['cohort']='chat'
            gaps=[100]*(100-outliers)+[1000]*outliers
            receipts=[1_000_000_000]
            for gap in gaps:receipts.append(receipts[-1]+gap*1_000_000)
            row.update(intended_release_ns=0,request_start_ns=0,first_token_ns=receipts[0],request_end_ns=receipts[-1]+1,token_receipt_ns=receipts,output_token_ids=[1]*101)
            r=m.request_metrics(row,0,30_000_000_000,events,module)
            self.assertTrue(r['timing_valid']);self.assertEqual(r['slo_pass'],expected)
            self.assertEqual(r['itl_gaps_above_slo'],outliers)
            self.assertFalse(r['every_itl_gap_within_slo'])
            self.assertEqual(r['request_itl_max_ms'],1000)
            self.assertEqual(r['request_itl_p95_ms'],100 if outliers==2 else 1000)
            aggregate=m.aggregate([r],30)
            self.assertEqual(aggregate['slo_good_requests'],int(expected))
            self.assertEqual(aggregate['itl_gaps_above_slo'],outliers)
            self.assertEqual(aggregate['aggregate_latency_slo']['passed'],expected)

    def test_aggregate_p95_latency_slo_distinct_from_request_goodput(self):
        raw=real_fixture();row=copy.deepcopy(raw['requests'][0]);module,_=m.validator();events=m.terminal_index(raw)
        receipts=[1_000_000_000+i*100_000_000 for i in range(len(row['output_token_ids']))]
        row.update(cohort='chat',intended_release_ns=0,request_start_ns=0,first_token_ns=receipts[0],request_end_ns=receipts[-1]+1,token_receipt_ns=receipts)
        good=m.request_metrics(row,0,30_000_000_000,events,module)
        bad=copy.deepcopy(good);bad['release_ttft_ms']=6000;bad['slo_pass']=False
        a=m.aggregate([good]*98+[bad]*2,30)
        self.assertEqual(a['slo_good_requests'],98)
        self.assertTrue(a['aggregate_latency_slo']['passed'])

    def test_host_swap_and_per_process_faults(self):
        def sample(at,available,used,sin,sout,minor,major,identity='10:100'):
            return {'at_ns':at,'processes':[{'identity':identity,'cpu_s':at,'rss_bytes':10,'minor_faults':minor,'major_faults':major}],
                    'host_memory':{'available_bytes':available,'swap_used_bytes':used,'swap_in_bytes':sin,'swap_out_bytes':sout}}
        raw={'resource_samples':[sample(5,1,1,1,1,1,1),sample(10,1000,100,200,300,10,20),sample(15,500,250,220,320,12,22),sample(20,800,80,230,350,15,26),sample(25,1,1,99999,99999,99999,99999)]}
        r=m.resources(raw,10,20);h=r['host_memory']
        self.assertEqual(h['swap_in_bytes_delta'],30);self.assertEqual(h['swap_out_bytes_delta'],50)
        self.assertEqual(h['available_bytes_min_sampled'],500);self.assertEqual(h['swap_used_bytes_delta'],-20)
        self.assertEqual(r['process_fault_deltas_sampled']['major_faults'],{'10:100':6})
        self.assertEqual(r['process_fault_totals_sampled']['minor_faults'],5)
        raw['resource_samples'][2]['processes'][0]['identity']='10:101'
        r=m.resources(raw,10,20)
        self.assertIsNone(r['process_fault_totals_sampled']['major_faults'])
        self.assertIsNone(m.resources({},10,20)['host_memory']['swap_in_bytes_delta'])

    def test_source_or_config_identity_mismatch(self):
        a=real_fixture();b=copy.deepcopy(a);b['model_identity']['different']='yes'
        c=m.compare(a,b)
        self.assertTrue(c['matched_correctness_passed']);self.assertFalse(c['performance_claim_eligible'])

    def staging_fixture(self):
        events=[]
        def event(name,eid,at,**fields):
            e={'event':name,'event_id':eid,m.EVENT_TIMESTAMPS[name]:at,**fields};events.append(e);return e
        for eid,job,start,ids in [('a',1,10,[1,2]),('b',2,40,[3,4]),('c',3,90,[1,2])]:
            event('nomination_decision',eid,start-1,chosen=[{'event_id':eid,'selected_keys':['x','y']}])
            event('staging_submitted',eid,start,job_id=job,reserved_block_ids=ids,submitted_bytes=20,bytes_per_key=10)
            event('staging_ready',eid,start+10,job_id=job,transfer_bytes=20)
        event('resume_bound','a',30);event('staging_bound_request_owned','a',35,reserved_blocks=2)
        event('native_prefix_published','a',60,block_ids=[7,8,9,1,2],local_prefix_count=3,prefetched_suffix_block_ids=[1,2])
        event('native_prefix_adopted','a',61,published_block_ids=[7,8,9,1,2],actual_block_ids=[7,8,9,1,2])
        event('resume_bound','b',55);event('staging_bound_request_owned','b',56,reserved_blocks=2)
        event('staging_retired_pending_fence','b',57,job_id=2,reason='cancelled')
        event('staging_retired','b',80,released_blocks=2,released_bytes=20,reason='cancelled')
        event('staging_retired','c',110,released_blocks=2,released_bytes=20,reason='expired')
        terminal={'external_request_id':'r','prefetch_attribution':{'wait_hint':{'event_id':'a'}},'first_admission':{'admitted_at_ns':65},'speculative_transfer_accounting':{'consumed_bytes':20}}
        return {'wait_hint_events':events,'terminals':[terminal]}

    def test_native_misses_and_owned_retirement_timestamps_not_hidden(self):
        events=[{'event':name,'event_id':name,field:15} for name,field in m.EVENT_TIMESTAMPS.items()]
        raw={'wait_hint_events':events}
        c=m.counters(raw,10,20)
        self.assertEqual(c['wait_event_counts']['native_prefix_missed'],1)
        self.assertEqual(c['wait_event_counts']['staging_bound_request_owned'],1)
        self.assertEqual(c['wait_event_counts']['staging_retired_pending_fence'],1)
        self.assertEqual(c['wait_event_timestamp_coverage']['missing_timestamp'],0)
        events.append({'event':'native_prefix_missed','event_id':'unknown-time'})
        c=m.counters(raw,10,20)
        self.assertEqual(c['wait_event_counts']['native_prefix_missed'],1)
        self.assertEqual(c['wait_event_counts_all_recorded']['native_prefix_missed'],2)
        self.assertEqual(c['wait_event_timestamp_coverage']['missing_timestamp_event_counts']['native_prefix_missed'],1)

    def test_staging_lifetime_highwater_and_suffix_accounting(self):
        raw=self.staging_fixture();u=m.staging_utility(raw,0,120)
        self.assertEqual(u['unique_chosen_event_ids'],3)
        self.assertEqual(u['global_ready_completed_bytes_observed'],60)
        self.assertEqual(u['global_named_demand_consumed_bytes'],20)
        self.assertEqual(u['global_native_adopted_transferred_suffix_bytes'],20) # not five published blocks
        self.assertEqual(u['global_completed_unconsumed_by_named_demand_bytes'],40)
        self.assertEqual(u['reservation_highwater']['known_closed_interval_peak_unique_blocks'],4) # recycled [1,2] is not cumulative capacity
        self.assertEqual(u['request_owned_highwater']['known_closed_interval_peak_unique_blocks'],4)
        rows={r['event_id']:r for r in u['event_rows']}
        self.assertEqual(rows['a']['reservation_lifetime_ms'],50/1e6)
        self.assertEqual(rows['a']['request_owned_reservation_lifetime_ms'],25/1e6)
        self.assertEqual(rows['b']['reservation_lifetime_ms'],40/1e6) # pending fence at57 is not release
        self.assertTrue(rows['a']['ready_before_enqueue']);self.assertTrue(rows['a']['ready_before_first_admission'])
        self.assertEqual(u['completed_unconsumed_retired_bytes_by_category']['cancelled'],20)
        self.assertIsNone(u['global_explicit_successful_completion_bytes_observed'])

    def test_duplicate_ready_records_do_not_double_count_traffic(self):
        raw=self.staging_fixture();ready=next(e for e in raw['wait_hint_events'] if e['event']=='staging_ready')
        raw['wait_hint_events'].append(copy.deepcopy(ready))
        u=m.staging_utility(raw,0,120)
        self.assertEqual(u['global_ready_completed_bytes_observed'],60)
        self.assertEqual(u['exact_duplicate_event_records_removed'],1)
        altered=copy.deepcopy(ready);altered['transfer_bytes']+=1;raw['wait_hint_events'].append(altered)
        u=m.staging_utility(raw,0,120)
        self.assertIsNone(u['global_ready_completed_bytes_observed']);self.assertTrue(u['ready_job_conflicts'])

    def test_missing_release_is_unknown_and_physical_overlap_visible(self):
        raw=self.staging_fixture();raw['wait_hint_events']=[e for e in raw['wait_hint_events'] if not(e['event_id']=='c' and e['event']=='staging_retired')]
        u=m.staging_utility(raw,0,120)
        self.assertIn('c',u['unknown_reservation_intervals']);self.assertFalse(u['highwater_complete_for_submitted_attempts'])
        row=next(r for r in u['event_rows'] if r['event_id']=='c')
        self.assertIsNone(row['reservation_lifetime_ms']);self.assertIsNone(row['completed_unconsumed_by_named_demand_bytes'])
        raw=self.staging_fixture();sub=next(e for e in raw['wait_hint_events'] if e['event_id']=='b' and e['event']=='staging_submitted');sub['reserved_block_ids']=[1,2]
        u=m.staging_utility(raw,0,120)
        self.assertTrue(u['reservation_highwater']['physical_overlap_errors'])
        self.assertEqual(u['reservation_highwater']['known_closed_interval_peak_unique_blocks'],2)

    def test_expired_success_without_ready_stays_unknown_until_explicit_completion(self):
        raw=self.staging_fixture();raw['wait_hint_events']=[e for e in raw['wait_hint_events'] if not(e['event_id']=='c' and e['event']=='staging_ready')]
        u=m.staging_utility(raw,0,120);self.assertIn('c',u['unknown_completed_attempts'])
        raw['speculative_transfer_completions']=[{'event_id':eid,'job_id':job,'success':True,'cancelled':False,'completed_at_ns':at,'bytes':20} for eid,job,at in [('a',1,20),('b',2,50),('c',3,100)]]
        raw['speculative_transfer_completions'].append(copy.deepcopy(raw['speculative_transfer_completions'][0]))
        u=m.staging_utility(raw,0,120)
        self.assertEqual(u['global_ready_completed_bytes_observed'],40)
        self.assertEqual(u['global_explicit_successful_completion_bytes_observed'],60)
        self.assertFalse(u['unknown_completed_attempts'])
        self.assertEqual(u['global_completed_unconsumed_by_named_demand_bytes'],40)

    def test_worker_generic_counter_deltas_preserve_missing_baselines(self):
        raw={'postload_export':{'workers':[{'pid':1,'counts':{'legacy_mean_numerical_calls':5}}]},'before_load_export':{'workers':[{'pid':1,'counts':{}}]}}
        self.assertIsNone(m.counters(raw,0,10)['worker_counts_delta']['legacy_mean_numerical_calls'])
        raw['reset_scope']={'passive_counts_times_samples':'explicit clear'}
        self.assertEqual(m.counters(raw,0,10)['worker_counts_delta']['legacy_mean_numerical_calls'],5)

    def test_reused_transfer_job_identity_is_unknown_not_double_credit(self):
        raw=self.staging_fixture()
        next(e for e in raw['wait_hint_events'] if e['event_id']=='b' and e['event']=='staging_submitted')['job_id']=1
        u=m.staging_utility(raw,0,120)
        self.assertTrue(u['submitted_job_identity_conflicts'])
        self.assertIsNone(u['global_named_demand_consumed_bytes'])
        self.assertIsNone(u['global_submitted_transfer_bytes'])

    def test_missing_and_conflicting_completion_records_stay_visible(self):
        raw=self.staging_fixture();raw['speculative_transfer_completions']=[{'event_id':'a','job_id':1,'success':True,'cancelled':False,'completed_at_ns':20,'bytes':20}, {'event_id':'a','job_id':1,'success':True,'cancelled':False,'completed_at_ns':20,'bytes':21}]
        u=m.staging_utility(raw,0,120)
        self.assertEqual(u['completion_collection_conflicting_jobs'],['1'])
        self.assertFalse(u['explicit_completion_coverage_for_submitted_jobs'])
        self.assertIsNone(u['global_explicit_successful_completion_bytes_observed'])

    def test_archiving_v2_summary_preserves_exact_bytes_and_rejects_conflict(self):
        with tempfile.TemporaryDirectory(prefix='campaign-metrics-test-') as directory:
            path=Path(directory)/'summary.json';body=b'{"metric_definition_version":"v2-per-request-p95-itl","goodput":1}\n';path.write_bytes(body)
            archive=m.archive_previous_output(path)
            self.assertEqual(Path(archive['archive']).name,'summary-v2-p95-itl.json')
            self.assertEqual(Path(archive['archive']).read_bytes(),body)
            path.write_text('{"metric_definition_version":"v2-per-request-p95-itl","goodput":2}')
            with self.assertRaises(ValueError):m.archive_previous_output(path)


if __name__=='__main__':unittest.main(verbosity=2)
