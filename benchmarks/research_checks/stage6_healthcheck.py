"""One bounded, GPU-free agent health check; never supervises or changes workers."""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import psutil

from benchmarks.serving_suite.campaign_contracts import Ledger, atomic_json, load
from benchmarks.serving_suite.campaign_validity import measurement_validity
from harness.gpu_lock import GPU_LOCK


def coordinators(out):
    found=[]
    for p in psutil.process_iter(['pid','cmdline','create_time']):
        try:
            args=p.info['cmdline'] or []
            ordinary='benchmarks.serving_suite.campaign' in args and any(x in args for x in ('run','resume'))
            recovery='benchmarks.research_checks.stage6_resume' in args and '--check' not in args
            if str(out) in args and (ordinary or recovery) and p.uids().real==os.getuid():
                found.append(p)
        except (psutil.NoSuchProcess,psutil.AccessDenied):
            pass
    return found


def lock_owners(path):
    if not path.is_file():
        return []
    inode=path.stat().st_ino
    owners=[]
    for line in Path('/proc/locks').read_text().splitlines():
        fields=line.split()
        if len(fields)>5 and fields[1]=='FLOCK' and fields[5].endswith(':'+str(inode)):
            owners.append(int(fields[4]))
    return owners


def gpu_snapshot():
    import pynvml as nv
    nv.nvmlInit()
    try:
        h=nv.nvmlDeviceGetHandleByIndex(0)
        m=nv.nvmlDeviceGetMemoryInfo(h)
        return dict(memory_used=m.used,memory_total=m.total,utilization=nv.nvmlDeviceGetUtilizationRates(h).gpu,
            temperature_c=nv.nvmlDeviceGetTemperature(h,nv.NVML_TEMPERATURE_GPU),
            processes=[p.pid for p in nv.nvmlDeviceGetComputeRunningProcesses(h)])
    finally:
        nv.nvmlShutdown()


def assess_progress(previous,current,now,threshold):
    signature=(current['ledger_sequence'],current['request_bytes'],current['stream_bytes'])
    changed=now if previous.get('progress_signature')!=list(signature) else previous.get('progress_changed_unix',now)
    return dict(progress_signature=list(signature),progress_changed_unix=changed,
        stalled=now-changed>threshold)


def inspect(out,state_dir):
    out,state_dir=Path(out).resolve(),Path(state_dir).resolve()
    # In particular, do not recreate a path hidden by the relocation proof.
    if not (out/'ledger.jsonl').is_file():
        return dict(status='attention',issues=['campaign_path_missing_do_not_recreate'],out=str(out))
    previous=load(state_dir/'latest.json') if (state_dir/'latest.json').is_file() else {}
    now=time.time(); events=Ledger(out).index(); latest=max((e['sequence'] for e in events.values()),default=0)
    lock=load(out/'protocol.lock.json'); expected=sum(len(b['systems']) for b in lock['schedule'])
    valid=[]; running=[]; issues=[]; coverage=[]; quality_signals=[]
    previous_sequence=previous.get('ledger_sequence',0)
    required={(b['model'],b['cell'],b['repeat']):set(b['systems']) for b in lock['schedule']}
    for e in events.values():
        i=e['details'].get('identity',{})
        if i.get('phase')!='confirm':continue
        if i.get('source')!=lock['source']['digest']:
            issues.append('confirmation_source_differs_from_frozen_protocol');continue
        if e['state']=='running':running.append(e)
        if e['state'] not in ('verified','failed','interrupted'):continue
        d=out/e['details']['path']
        validity=measurement_validity(d,e)
        if not validity['valid']:
            # A later valid replacement of this exact logical case resolves it.
            replacement=any(x['case_id']==e['case_id'] and x['sequence']>e['sequence'] and x['state']=='verified'
                for x in events.values())
            if not replacement:issues.append(dict(kind='invalid_attempt',path=e['details']['path'],classification=validity['classification']))
            continue
        valid.append(e)
        if e['sequence']<=previous_sequence:continue
        c=load(d/'measurement/result.json')['cases'][0]
        observation=c.get('campaign',{})
        quality_signals.append(dict(model=i['model'],cell=i['cell'],system=i['system'],
            completed=c['summary']['completed'],output_tokens=c['summary'].get('output_tokens'),
            caps=observation.get('output_caps'),scope='Signals only; retained source-based answer review remains required.'))
        if c['summary'].get('output_tokens')==0:
            issues.append(dict(kind='no_model_output_for_completed_workload',path=e['details']['path']))
        if c['summary'].get('finish_reasons',{}).get('length')==c['summary']['completed'] and c['summary']['completed']:
            issues.append(dict(kind='all_requests_capped_review_workload_purpose',path=e['details']['path']))
        if (c['summary']['failed'] or c['summary']['completed']!=c['expected_requests'] or not c['cleanup']['verified'] or
                observation.get('measurement_errors') or observation.get('completed_units')!=observation.get('offered_units')):
            issues.append(dict(kind='incomplete_or_unclean_case',path=e['details']['path']))
        corpus=load(d/'measurement/corpus.json')
        expected_tools=sum(n['kind']=='tool' for s in corpus['sessions'] for n in s['graph']['nodes'])
        if observation.get('tools')!=expected_tools:
            issues.append(dict(kind='incomplete_tool_coverage',path=e['details']['path']))
        native=c.get('campaign',{}).get('native_cache')
        if i['system'] in ('arc','wait') and (not native or not native.get('complete') or native.get('receipts')!=c['expected_requests']):
            issues.append(dict(kind='missing_native_receipts',path=e['details']['path']))
        if i['cell']=='B06' and i['system']=='arc' and native and (
                not native.get('returning_loaded_tokens') or not native.get('returning_requests_with_prior_cpu_eviction')):
            coverage.append(i['model']+':B06_repository_pressure')
    groups={}
    for e in valid:
        i=e['details']['identity'];key=(i['model'],i['cell'],i['repeat'])
        if key in required:groups.setdefault((*key,e['details']['block_id']),set()).add(i['system'])
    complete_blocks={k[:3] for k,v in groups.items() if v==required[k[:3]]}
    unique_cases={e['case_id'] for e in valid}
    acquisition_complete=len(complete_blocks)==len(required)
    procs=coordinators(out)
    if len(procs)>1:issues.append('duplicate_coordinators')
    owned=[]
    if procs:
        for p in [procs[0]]+procs[0].children(recursive=True):
            try:owned.append(dict(pid=p.pid,create_time=p.create_time()))
            except psutil.NoSuchProcess:pass
        if procs[0].pid not in lock_owners(GPU_LOCK):issues.append('coordinator_does_not_own_gpu_reservation')
    elif not acquisition_complete:issues.append('coordinator_stopped_before_completion')
    try:
        gpu=gpu_snapshot()
        if procs and set(gpu['processes'])-{p['pid'] for p in owned}:
            issues.append('gpu_client_outside_owned_process_tree')
    except Exception as error:
        gpu=dict(error=type(error).__name__+': '+str(error));issues.append('gpu_telemetry_unavailable')
    memory=psutil.virtual_memory(); swap=psutil.swap_memory()
    if memory.available<128*1024**2 and swap.free<128*1024**2:issues.append('severe_host_memory_exhaustion')
    request_bytes=stream_bytes=0; active=[]; threshold=1200
    resolved=load(out/'resolved.json')
    for e in running:
        i=e['details']['identity']; d=out/e['details']['path']/'measurement'
        cell=next(c for c in resolved['cells'] if c['model']==i['model'] and c['cell']==i['cell'])
        plan=load(cell['confirmation_binding']['plan'])
        bound=plan['startup_timeout_s']+plan['case_timeout_s']+180
        age=now-e['timestamp_unix']
        if age>bound+90:issues.append('worker_exceeded_supervised_deadline')
        for p in d.glob('*/requests.jsonl'):request_bytes+=p.stat().st_size
        for p in d.glob('*/streams/*.sse'):stream_bytes+=p.stat().st_size
        threshold=min(threshold,max(300,plan['request_timeout_s']+120))
        active.append(dict(model=i['model'],cell=i['cell'],system=i['system'],repeat=i['repeat'],age_s=round(age),deadline_s=bound))
    progress=assess_progress(previous,dict(ledger_sequence=latest,request_bytes=request_bytes,stream_bytes=stream_bytes),now,threshold)
    if procs and running and progress['stalled']:issues.append('no_ledger_request_or_stream_progress')
    previous_gaps=set(previous.get('known_coverage_gaps',[])); gaps=previous_gaps|set(coverage)
    result=dict(schema='stage6-unattended-health-v1',checked_unix=now,
        status='attention' if issues else 'acquisition_complete' if not procs and acquisition_complete else 'healthy',
        issues=issues,valid_configurations=len(unique_cases),expected_configurations=expected,
        complete_matched_blocks=len(complete_blocks),expected_matched_blocks=len(required),
        raw_states=dict(Counter(e['state'] for e in events.values())),ledger_sequence=latest,
        request_bytes=request_bytes,stream_bytes=stream_bytes,active=active,owned_processes=owned,
        gpu=gpu,host=dict(available_memory_bytes=memory.available,swap_used_bytes=swap.used),
        new_coverage_gaps=sorted(set(coverage)-previous_gaps),known_coverage_gaps=sorted(gaps),**progress)
    result['new_quality_signals']=quality_signals
    atomic_json(state_dir/'latest.json',result)
    with (state_dir/'history.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--state-dir',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(inspect(args.out,args.state_dir),indent=2))
