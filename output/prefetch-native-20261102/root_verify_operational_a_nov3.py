"""Independent ROOT checks of original persistent A cases, retaining probe failures."""
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics

R=Path('/home/philip/vllm-semantic-cache-ext')
H=R/'output/prefetch-native-20261102'
E=R/'evidence/live-optimization-2080ti-20261103'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def p95(xs):
    xs=sorted(xs);pos=.95*(len(xs)-1);i=int(pos)
    return xs[i]+(xs[min(i+1,len(xs)-1)]-xs[i])*(pos-i)
def latencies(qs):
    t=[(q['first_token_ns']-q['request_start_ns'])/1e6 for q in qs]
    e=[(q['request_end_ns']-q['request_start_ns'])/1e6 for q in qs]
    return dict(count=len(qs),mean_ttft_ms=statistics.mean(t),p95_ttft_ms=p95(t),
                max_ttft_ms=max(t),mean_e2e_ms=statistics.mean(e),p95_e2e_ms=p95(e),max_e2e_ms=max(e))

def verify():
    manifest=json.loads((E/'operational-frozen.json').read_text())
    oldpins={str((R/n).resolve()):v for g in ('existing_frozen_inputs','new_review_file_pins','file_pins') for n,v in manifest[g].items()}
    mismatches=[p for p,d in oldpins.items() if sha(Path(p))!=d]
    assert mismatches==[str(R/'README.md')]
    rows=[];arms={};first_prompts={}
    for arm in ('native8','candidate'):
        name='live-nov3-operational-a-'+arm
        path=H/'runs'/name/'raw.json.gz';raw=json.load(gzip.open(path,'rt'))
        assert raw['status']=='failed' and raw['error']=="RuntimeError('Actual backend abort / natural-EOS health qualification failed or was not exercised')"
        assert not raw['performance_eligible'] and raw['operational_qualification']
        assert len(raw['waves'])==24 and len(raw['requests'])==1728 and not raw['owned_descendants_remaining']
        origin=raw['load_start_ns'];assert raw['operational_contract']['gateway_origin_ns']==origin
        all_ids=set();all_sids=set();boundaries=[];arm_rows=[];returns=contiguous=0
        for wave in raw['waves']:
            wi=wave['wave_index'];qs=wave['requests'];assert len(qs)==72 and wave['status']=='complete'
            byindex={q['index']:q for q in qs}
            sids=set()
            for q in qs:
                b=q['payload'];oid=b['request_id'];sid=q['session_id'];out=q['output_token_ids']
                assert oid not in all_ids;all_ids.add(oid);sids.add(sid)
                assert q['http_status']==200 and q['success'] and q['natural_eos'] and q['done_seen'] and q['terminal_complete'] and q['finish_reasons']==['stop']
                assert len(b['prompt'])+b['max_tokens']==7680 and 0<len(out)<b['max_tokens']
                assert not b.get('ignore_eos',False) and not b.get('stop') and not b.get('stop_token_ids')
                assert q['priority_origin_ns']==origin and q['priority_clock_ns']==q['request_start_ns']
                assert b['priority']==(q['request_start_ns']-origin)//1000000-(8000 if q['turn'] else 0)
                assert len(q['token_receipt_ns'])==len(out) and q['first_token_ns']==q['token_receipt_ns'][0]
                assert hashlib.sha256(json.dumps(out,sort_keys=True,separators=(',',':')).encode()).hexdigest()==q['output_token_digest']
                if q['turn']:
                    prior=byindex[q['index']-1];assert sid==prior['session_id']
                    hist=prior['payload']['prompt']+prior['output_token_ids'][:-1];boundary=len(hist)//16*16
                    returns+=1;contiguous+=b['prompt'][:boundary]==hist[:boundary]
            assert len(sids)==24 and not sids&all_sids;all_sids.update(sids)
            seed=wave['seed'];initial={q['index']:q['payload']['prompt'] for q in qs if q['turn']==0}
            if seed in first_prompts:assert initial==first_prompts[seed]
            first_prompts[seed]=initial
            hc=wave['http_client_configuration']
            assert hc['force_close'] and hc['automatic_POST_retry'] is False and hc['application_retries']==0
            cc=collections.Counter(e['event'] for e in wave['http_connection_events']);attempts=72+len(wave['hints'])
            assert cc==collections.Counter(dict.fromkeys(['request_start','connection_create_start','connection_create_end','request_headers_received'],attempts)),cc
            assert all(e['status']==200 for e in wave['http_connection_events'] if e['event']=='request_headers_received')
            exported=wave['settled_export'];sc=exported['scheduler']['operational_scalar_state']
            assert sc['native_request_count']==sc['native_running_count']==sc['native_waiting_count']==0
            if arm=='candidate':
                sl=sc['scheduler_lengths'];ml=sc['manager_lengths'];ws=exported['workers'][0]['operational_worker_state']['worker_lengths']
                assert sl['_observed_prefix_generations']==24*(wi+1) and sl['_observed_prefixes']<=1024 and sl['_wait_hint_results']<=4096
                assert all(v==0 for k,v in sl.items() if k not in ('_observed_prefix_generations','_observed_prefixes','_wait_hint_results'))
                assert ml['_score_residencies']<=2048 and all(v==0 for k,v in ml.items() if k!='_score_residencies')
                assert all(v==0 for v in ws.values())
            else:sl={}
            near=sorted(raw['resource_samples'],key=lambda s:abs(s['at_ns']-wave['load_end_ns']))[:5]
            engine=str(sc['pid'])+':'
            engine_rss=[p['rss_bytes'] for s in near for p in s['processes'] if p['identity'].startswith(engine)]
            assert engine_rss
            seconds=(wave['load_end_ns']-wave['load_start_ns'])/1e9;tokens=sum(len(q['output_token_ids']) for q in qs)
            summary={'arm':arm,'wave':wi,'seed':seed,'load_s':seconds,'tokens':tokens,'tokens_s':tokens/seconds,
                     'engine_boundary_rss_mib':statistics.median(engine_rss)/2**20,
                     'session_identities':sl.get('_observed_prefix_generations'),'prefix_entries':sl.get('_observed_prefixes'),
                     'ACK_entries':sl.get('_wait_hint_results'),**latencies(qs)}
            rows.append(summary);arm_rows.append(summary);boundaries.append(exported)
        probe=raw['cancellation_probe'];assert probe['qualified'] is False
        assert len(probe['health_requests'])==4 and all(q['natural_eos'] and q['finish_reasons']==['stop'] for q in probe['health_requests'])
        abort=probe['abort_request'];assert abort['disconnect_ns'] and not abort['terminal_before_close']
        observed=probe['abort_observer'];assert observed['count']==1 and len(observed['records'])==1 and observed['records'][0]['status']=='FINISHED_ABORTED'
        assert probe['actual_abort_transition_matches']==[]
        for relative,pin in raw['source_pins'].items():assert sha(Path(raw['args']['source_dir'])/relative)==pin
        for relative,pin in raw['runtime_pins'].items():assert sha(H/relative)==pin
        gpu=[s['gpu']['memory_used_bytes']/2**20 for s in raw['resource_samples'] if 'gpu' in s]
        arms[arm]={'raw_sha256':sha(path),'natural_eos':1728,'health_natural_eos':4,'distinct_sessions':len(all_sids),
                   'return_requests':returns,'actual_contiguous_returns':contiguous,
                   'case_wall_s':(raw['load_end_ns']-raw['load_start_ns'])/1e9,
                   'generation_only_24wave_load_s_sum':sum(q['load_s'] for q in arm_rows),
                   'output_tokens':sum(q['tokens'] for q in arm_rows),
                   'all_native_idle_boundaries':True,'candidate_all_pending_and_audit_metadata_zero':True if arm=='candidate' else None,
                   'NVML_mib_range':[min(gpu),max(gpu)],
                   'engine_boundary_rss_mib_first_last':[arm_rows[0]['engine_boundary_rss_mib'],arm_rows[-1]['engine_boundary_rss_mib']],
                   'engine_boundary_rss_mib_last6_range':[min(q['engine_boundary_rss_mib'] for q in arm_rows[-6:]),max(q['engine_boundary_rss_mib'] for q in arm_rows[-6:])],
                   'prefix_cap_first_wave':next((q['wave'] for q in arm_rows if q['prefix_entries']==1024),None),
                   'ACK_cap_first_wave':next((q['wave'] for q in arm_rows if q['ACK_entries']==4096),None),
                   'formal_probe':'failed_preserved','observed_actual_abort_transitions':1,
                   'final_prometheus_metrics':'unavailable_after_probe_exception_not_zero'}
    result={'schema':'root-independent-operational-a-v1','arms':arms,'rows':rows,
            'all_3456_normal_and8health_naturalEOS_verified':True,'source_runtime_priority_HTTP_actual_history_verified':True,
            'original783_pins_only_README_changed':True,'formal_cancellation_qualification':'Both failed; original matcher did not include randomized native suffix. Real abort observed, direct external mapping unrecorded.',
            'limitations':['Two single persistent cases; no new confidence intervals or fresh trace generalization.',
               'Repeated public first prompts and private waits; actual outputs regenerated.',
               '576 observed session identities is below default4096 cap.',
               'RSS includes shared mappings and swapping; aggregate RSS does not establish private-memory retention.',
               'Native metadata unavailable remains None, not zero.',
               '24 waves is a finite observed horizon, not indefinite leak or fairness proof.',
               'Documentation changed concurrently; source/runtime/input pins held.'],
            'remaining':'Separate B–E coverage and direct actual-abort mapping qualification', 'production_endorsement':False}
    (E/'operational-a-root-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    with (E/'operational-a-waves.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({'verified':True,'arms':arms},indent=2))

if __name__=='__main__':verify()
