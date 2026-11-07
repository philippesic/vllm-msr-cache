"""Blind source-review inventory of every ordinary capped Stage 6 response.

This retained analysis includes earlier development attempts and all acquisition
phases. It supplements the frozen confirmation quality sample. Intentional native
fixtures, warmups, cancellation probes and recovery requests are not task answers.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from benchmarks.serving_suite.campaign_contracts import Ledger,atomic_json,digest,load,local_path,verify_tree
from benchmarks.serving_suite.campaign_quality import RUBRIC,objective

VERSION = 'stage6-all-attempt-capped-review-v1'
PHASES = {'screen','pilot','confirm','overhead','operational'}


def ledger_inputs(root):
    return {p.name:digest(Ledger(p).events()) for p in sorted(root.iterdir()) if (p/'ledger.jsonl').is_file()}


def prepare(root,out):
    root,out = Path(root).resolve(),Path(out).resolve()
    if out.exists():
        raise ValueError('Use a new audit destination; preserve earlier packets and ratings')
    inputs = ledger_inputs(root)
    items,labels,pending,attempts,partial = [],{},[],[],[]
    for campaign in sorted(p for p in root.iterdir() if (p/'ledger.jsonl').is_file()):
        for event in Ledger(campaign).index().values():
            details = event['details']
            identity = details.get('identity',{})
            if identity.get('phase') not in PHASES:
                continue
            if event['state'] in ('running','planned'):
                pending.append(dict(campaign=campaign.name,attempt=event['attempt_id']))
                continue
            directory = local_path(campaign,details['path'])
            if not details.get('seal'):
                raise ValueError('Terminal ordinary attempt lacks seal: '+event['attempt_id'])
            verify_tree(directory,details['seal'])
            measurement = directory/'measurement'
            result = measurement/'result.json'
            attempts.append(dict(campaign=campaign.name,attempt=event['attempt_id'],seal=details['seal']))
            corpus_path = measurement/'corpus.json'
            sessions = load(corpus_path)['sessions'] if corpus_path.exists() else []
            cases = load(result)['cases'] if result.is_file() else [dict(name=p.parent.name)
                for p in measurement.glob('*/requests.jsonl')]
            for case in cases:
                request_path = measurement/case['name']/'requests.jsonl'
                if not request_path.exists():
                    continue
                raw = []
                text = request_path.read_text()
                lines = text.splitlines()
                for index,line in enumerate(lines):
                    if not line.strip():
                        continue
                    try:
                        raw.append(json.loads(line))
                    except json.JSONDecodeError:
                        if index!=len(lines)-1 or text.endswith('\n'):
                            raise
                        partial.append(dict(campaign=campaign.name,attempt=event['attempt_id'],case=case['name'],
                            scope='Incomplete final JSONL fragment after interrupted capture; not a terminal capped response'))
                by_id = {session['id']:(slot,session) for slot,session in enumerate(sessions)}
                for row_index,row in enumerate(raw):
                    if row.get('finish_reason')!='length' or row.get('cancel_probe') or row.get('wave',0)<0:
                        continue
                    if not row.get('source_id'):
                        continue  # Isolated warmup/recovery probes have no corpus task identity.
                    if row['source_id'] not in by_id:
                        raise ValueError('Capped task has no retained source reference')
                    slot,session = by_id[row['source_id']]
                    graph = session['graph']
                    nodes = {n['id']:n for n in graph['nodes']}
                    node = nodes[row['node_id']]
                    ancestors,pending_nodes = set(),list(node.get('depends_on',[]))
                    while pending_nodes:
                        ancestor = pending_nodes.pop()
                        if ancestor not in ancestors:
                            ancestors.add(ancestor)
                            pending_nodes.extend(nodes[ancestor].get('depends_on',[]))
                    final = row['node_id'] in graph['outputs']
                    reference = dict(session['quality_reference'])
                    if identity['cell']=='B10':
                        reference['kind'] = 'team'
                    if not final:
                        reference['kind'] = 'intermediate'
                    item = dict(source_id=session['id'],node_id=row['node_id'],
                        response_role='final' if final else 'capped_intermediate',
                        current_instruction=node['prompt'],instructions=session['turns'],
                        source_context=session.get('context',''),reference=reference,
                        answer=row.get('generated_text',''),objective=objective(row.get('generated_text',''),reference,capped=True),
                        actual_dependencies=[r['generated_text'] for r in raw[:row_index] if
                            r.get('source_id')==session['id'] and r.get('wave',0)==row.get('wave',0) and r.get('node_id') in ancestors],
                        tools=[dict(output=re.sub(r'\bin \d+(?:\.\d+)?s\b','in [elapsed time omitted]',p.read_text(errors='replace')))
                            for p in sorted((measurement/case['name']/'tools').glob(f'{row.get("wave",0)}-{slot}-*.txt'))
                            if p.stem.split('-',2)[2] in ancestors],
                        instruction_context_digest=digest(dict(prompt=row['payload']['prompt'],
                            workload_source=load(measurement/'source.json')['files'].get('benchmarks/serving_suite/campaign_workload.py'))))
                    label = dict(campaign=campaign.name,phase=identity['phase'],model=identity['model'],
                        cell=identity['cell'],system=identity['system'],attempt=event['attempt_id'],
                        request_id=row['payload']['request_id'],wave=row.get('wave',0),node_id=row['node_id'])
                    item['review_id'] = digest(label)[:24]
                    if item['review_id'] in labels:
                        raise ValueError('Duplicate capped-response identity')
                    labels[item['review_id']] = label
                    items.append(item)
    random.Random(17011).shuffle(items)
    canonical = {}
    for item in items:
        signature = digest({k:v for k,v in item.items() if k!='review_id'})
        if signature in canonical:
            item['duplicate_of'] = canonical[signature]
        else:
            canonical[signature] = item['review_id']
    atomic_json(out/'blind-packet.json',dict(schema=VERSION,rubric=RUBRIC,items=items))
    atomic_json(out/'private-labels.json',labels)
    (out/'private-labels.json').chmod(0o600)
    atomic_json(out/'private-bindings.json',dict(root=str(root)))
    (out/'private-bindings.json').chmod(0o600)
    summary = dict(schema=VERSION,complete_inventory=not pending and ledger_inputs(root)==inputs,
        input_ledger_digests=inputs,pending=pending,partial_stream_fragments=partial,capped_responses=len(items),
        unique_contexts=len(canonical),phases=dict(Counter(v['phase'] for v in labels.values())),
        roles=dict(Counter(i['response_role'] for i in items)),verified_attempts=attempts,
        packet_digest=digest(items),rubric_digest=digest(RUBRIC),status='pending_source_audit',
        scope='All sealed ordinary capped responses in immediate campaign children; no timing exclusion. Native fixtures and isolated warmup/cancel/recovery probes excluded.')
    atomic_json(out/'summary.json',summary)
    return summary


def finish(out,ratings_path):
    out = Path(out)
    packet,summary = load(out/'blind-packet.json'),load(out/'summary.json')
    if not summary['complete_inventory']:
        raise ValueError('Complete acquisition before finalizing the all-attempt cap audit')
    root = Path(load(out/'private-bindings.json')['root'])
    if ledger_inputs(root)!=summary['input_ledger_digests']:
        raise ValueError('Acquisition changed after inventory; regenerate the cap packet')
    ratings = load(ratings_path)
    if ratings.get('schema')!='stage6-source-audit-v1' or ratings.get('rubric_digest')!=digest(RUBRIC) or not ratings.get('reviewer','').strip():
        raise ValueError('Source audit identity/rubric mismatch')
    expected = {i['review_id'] for i in packet['items']}
    observed = set()
    for row in ratings['ratings']:
        if row['review_id'] in observed or row['review_id'] not in expected:
            raise ValueError('Duplicate or unknown capped audit judgment')
        observed.add(row['review_id'])
        if any(type(row.get(k)) is not bool for k in ('source_supported','instruction_followed','useful')) or not row.get('notes','').strip():
            raise ValueError('Explicit source judgments and notes required')
        if not isinstance(row.get('unsupported_claims'),list) or any(not isinstance(c,str) for c in row['unsupported_claims']):
            raise ValueError('Unsupported claims must be an explicit string list')
    if observed!=expected or digest(packet['items'])!=summary['packet_digest']:
        raise ValueError('Cap audit does not match the complete sealed packet')
    atomic_json(out/'source-audit.json',ratings)
    summary.update(status='audited',reviewed=len(observed),reviewer=ratings['reviewer'],audit_digest=digest(ratings))
    atomic_json(out/'summary.json',summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--ratings',type=Path)
    args = parser.parse_args()
    if args.ratings:
        result = finish(args.out,args.ratings)
    elif args.root:
        result = prepare(args.root,args.out)
    else:
        parser.error('--root or --ratings is required')
    print(json.dumps({k:v for k,v in result.items() if k!='verified_attempts'},indent=2))


if __name__=='__main__':
    main()
