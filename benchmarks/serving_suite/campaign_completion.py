"""Completion evidence for controls, persistent operation and portable testing."""
from pathlib import Path

from .campaign_contracts import Ledger,digest,load,local_path,read_protocol,verify_tree
from .campaign_report import completed_blocks,paired_estimate


def supplementary_blocks(out,phase):
    out = Path(out)
    locked = read_protocol(out)
    expected = {(b['model'],b['cell'],b['repeat']):set(b['systems'])
        for b in locked['supplementary_schedules'][phase]}
    groups = {}
    for event in Ledger(out).index().values():
        identity = event['details'].get('identity',{})
        if event['state']!='verified' or identity.get('phase')!=phase or identity.get('source')!=locked['source']['digest']:
            continue
        key = (identity['model'],identity['cell'],identity['repeat'])
        if identity['system'] not in expected.get(key,set()):
            continue
        directory = local_path(out,event['details']['path'])
        verify_tree(directory,event['details']['seal'])
        case = load(directory/'measurement/result.json')['cases'][0]
        if case['evaluation']['status']!='passed' or not case['cleanup']['verified']:
            continue
        group = groups.setdefault((*key,event['details']['block_id']),{})
        group[identity['system']] = dict(event=event,case=case,
            rate=case['campaign']['completed_units_per_s'],units=case['campaign']['completed_units'])
    selected = {}
    for key,systems in groups.items():
        if set(systems)!=expected[key[:3]]:
            continue
        if len({s['units'] for s in systems.values()})!=1:
            raise ValueError('Supplementary block has mismatched offered work')
        previous = selected.get(key[:3])
        sequence = max(s['event']['sequence'] for s in systems.values())
        if previous is None or sequence>previous['sequence']:
            selected[key[:3]] = dict(systems=systems,block_id=key[3],sequence=sequence)
    missing = [dict(model=m,cell=c,repeat=r) for m,c,r in expected if (m,c,r) not in selected]
    return selected,missing


def reference(data):
    return dict(attempt=data['event']['attempt_id'],path=data['event']['details']['path'],
        seal=data['event']['details']['seal'])


def overhead(out):
    selected,missing = supplementary_blocks(out,'overhead')
    comparisons,controls = [],[]
    for key,block in sorted(selected.items()):
        for system,data in block['systems'].items():
            receipt = data['case'].get('control_receipt',{})
            if not receipt.get('verified') or receipt.get('hint_production') or data['case']['hints']:
                missing.append(dict(model=key[0],cell=key[1],repeat=key[2],system=system,reason='control_not_established'))
            controls.append(dict(model=key[0],cell=key[1],repeat=key[2],control=system,
                completed_units_per_s=data['rate'],duration_s=data['case']['load_duration_s'],
                startup_s=data['case']['startup_s'],resources=data['case']['telemetry'],artifacts=reference(data)))
    for cell in ('B01','B02'):
        blocks = [v for k,v in sorted(selected.items()) if k[:2]==('qwen3_8b_awq',cell)]
        for candidate,baseline in (('O1','O0'),('O2','O0'),('O2','O1')):
            if not blocks:
                continue
            estimate = paired_estimate([b['systems'][candidate]['rate']/b['systems'][baseline]['rate'] for b in blocks])
            interval = estimate['paired_95_interval']
            bound = ('bounded' if interval and interval[0]>=.97 else
                'regression' if interval and interval[1]<.97 else 'inconclusive')
            comparisons.append(dict(cell=cell,candidate=candidate,baseline=baseline,estimate=estimate,
                three_percent_throughput_loss_bound=bound,scope='Fixed six process repetitions; tool-free identical requests within cell'))
    active = completed_blocks(out,'confirm')
    active = [dict(model=m,cell=c,repeat=r,artifacts=reference(b['systems']['wait']),
        completed_units_per_s=b['systems']['wait']['rate']) for (m,c,r),b in sorted(active.items())
        if m=='qwen3_8b_awq' and c=='B06']
    return dict(complete=not missing,missing=missing,controls=controls,comparisons=comparisons,
        O3_active_reference=active,O3_scope='B06 confirmation reference; never paired with B01/B02 O0')


def operational(out):
    selected,missing = supplementary_blocks(out,'operational')
    cases = []
    required = {'duplicate_is_idempotent','unknown_session_cannot_protect_prefix',
        'future_information_rejected','oversize_identifier_rejected','delayed_advice_expires'}
    for (model,cell,repeat),block in sorted(selected.items()):
        for system,data in block['systems'].items():
            case = data['case']
            observation = case['campaign']
            operation = case.get('operational',{})
            expected_waves = 6 if model=='qwen3_8b_awq' else 2
            problems = []
            if len(case['waves'])!=expected_waves or not operation.get('passed') or not operation.get('native_transfer_drain_verified'):
                problems.append('persistent_cancel_drain_recovery_incomplete')
            if len(case['source_ids'])!=16 or case['summary']['completed']!=expected_waves*64:
                problems.append('persistent_full_workload_coverage_incomplete')
            if system=='wait':
                checks = observation.get('operational_advice_checks',[])
                if not required<={c['check'] for c in checks if c['passed']} or any(not c['passed'] for c in checks):
                    problems.append('advice_perturbation_checks_incomplete')
                if len(observation.get('abandoned_advice',[]))!=expected_waves:
                    problems.append('abandoned_advice_coverage_incomplete')
                if not case['metrics'].get('vllm:kv_offload_load_bytes_total',0):
                    problems.append('persistent_restore_unobserved')
            snapshots = observation.get('registry_snapshots',[])
            if not snapshots or any(s['speculative_jobs'] or s['reserved_gpu_blocks'] for s in snapshots):
                problems.append('native_registry_or_dispatch_proof_incomplete')
            if problems:
                missing.append(dict(model=model,system=system,reasons=problems))
            cases.append(dict(model=model,system=system,waves=len(case['waves']),
                completed_requests=case['summary']['completed'],completed_units=observation['completed_units'],
                operational=operation,registry_first_final=[snapshots[0],snapshots[-1]] if snapshots else [],
                registry_peaks={k:max(s[k] for s in snapshots) for k in ('sessions','prefixes','results','leases','native_jobs')} if snapshots else {},
                checks=observation.get('operational_advice_checks',[]),artifacts=reference(data)))
    return dict(complete=not missing,missing=missing,cases=cases,
        scope='Bounded persistent sequences and native receipts; no production or leak-freedom claim')


def recovery(out):
    out = Path(out)
    entries = list(Ledger(out).index().values())
    proofs = []
    for event in entries:
        identity = event['details'].get('identity',{})
        if event['state']!='interrupted' or identity.get('phase')!='screen':
            continue
        latest = completed_blocks(out,'screen',source=identity['source'])
        block = latest.get((identity['model'],identity['cell'],identity['repeat']))
        if not block or block['block_id']==event['details']['block_id']:
            continue
        prior = [e for e in entries if e['state']=='verified' and
            e['details'].get('block_id')==event['details']['block_id'] and e['sequence']<event['sequence']]
        directory = local_path(out,event['details']['path'])
        verify_tree(directory,event['details']['seal'])
        path = directory/'measurement/result.json'
        clean = path.exists() and load(path)['cases'][0]['cleanup']['verified']
        if prior and clean:
            proofs.append(dict(interrupted_attempt=event['attempt_id'],interrupted_artifacts=event['details']['path'],
                prior_partial_block_attempts=[e['attempt_id'] for e in prior],
                replacement_attempts=[reference(d) for d in block['systems'].values()],
                owned_cleanup_verified=True,whole_block_replaced=True))
    return dict(complete=bool(proofs),proofs=proofs,
        scope='Real interrupted measured process and sealed whole-block replacement; no partial-process continuation')


def readiness(out,throughput_complete,quality):
    out = Path(out)
    controls,operations,interruptions = overhead(out),operational(out),recovery(out)
    portability_path = out/'analysis/portability-proof.json'
    portability = load(portability_path) if portability_path.exists() else dict(complete=False)
    quality_ok = quality.get('status')=='audited' and quality.get('phase')=='confirm'
    if quality_ok:
        ratings = load(out/'analysis/quality/source-audit.json')
        packet = load(out/'analysis/quality/blind-packet.json')
        quality_ok = (digest(ratings)==quality['audit_digest'] and
            digest(packet['items'])==quality['packet_digest'] and
            {r['review_id'] for r in ratings['ratings']}=={r['review_id'] for r in packet['items']})
        objective = load(out/'analysis/quality/objective.json')['outputs']
        selected = completed_blocks(out,'confirm')
        attempts = {d['event']['attempt_id'] for b in selected.values() for d in b['systems'].values()}
        expected = sum(len(load(local_path(out,d['event']['details']['path'])/'measurement/corpus.json')['sessions'])
            for b in selected.values() for d in b['systems'].values())
        quality_ok = (quality_ok and len(objective)==expected and len({x['review_id'] for x in objective})==expected and
            {x['attempt'] for x in objective}==attempts and quality['objective_outputs']==expected)
    complete = bool(throughput_complete and quality_ok and controls['complete'] and operations['complete'] and
        interruptions['complete'] and portability.get('complete'))
    return dict(complete=complete,throughput=throughput_complete,quality_audit_complete=quality_ok,
        overhead=controls,operational=operations,recovery=interruptions,portability=portability,
        scope='Ready for the separately gated 5080 replication; no 5080 performance or production claim')
