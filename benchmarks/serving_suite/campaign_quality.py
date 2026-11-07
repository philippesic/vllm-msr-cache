"""Versioned objective checks and deterministic blinded source-review packets."""
from __future__ import annotations

import json
import random
import re
import string
from pathlib import Path

from .campaign_contracts import atomic_json, digest, load, local_path
from .campaign_report import completed_blocks

RUBRIC = dict(version='stage6-quality-v1',
    answer_json='Exactly one JSON string field answer; normalized reference exact match, with span presence separately reported.',
    investigation='Correct subject and actual validation status, with requested status format separately checked; at least one source-supported fact and a concrete evidence-based limitation.',
    team='Synthesis uses actual generated specialist findings and validation, without dropped or contradicted material results.',
    summary='Key source facts, unsupported claims, missing material facts and useful concise summary. Overlap alone is not a pass.',
    chat='Coherent response to the actual current instruction and source context, with no invented tool outcomes.',
    audit='Blind system/speed labels; inspect source, actual tools/dependencies and answer. Review all automatic failures and every length-capped final separately.')


def normalize(text):
    text = ''.join(c for c in text.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',text).split())


def objective(text, reference, *, capped=False):
    result = dict(nonempty=bool(text.strip()),length_capped=capped,kind=reference.get('kind','investigation'))
    if reference.get('kind')=='answer_json':
        def unique_fields(pairs):
            value = {}
            for key,item in pairs:
                if key in value:
                    raise ValueError('Duplicate answer field')
                value[key] = item
            return value
        try:
            parsed = json.loads(text,object_pairs_hook=unique_fields)
        except (ValueError,TypeError):
            parsed = None
        valid = isinstance(parsed,dict) and set(parsed)=={'answer'} and isinstance(parsed['answer'],str)
        answer = parsed['answer'] if valid else text
        actual = normalize(answer)
        answers = [normalize(a) for a in reference['answers']]
        result.update(schema_valid=valid,reference_exact=actual in answers,
                      reference_span_present=any(a and re.search(r'(?<!\w)'+re.escape(a)+r'(?!\w)',actual) for a in answers))
        result['automatic_failure'] = not(valid and result['reference_exact'])
    elif result['kind']=='investigation':
        subject = reference.get('symbol') or reference.get('PEP')
        exact_status = bool(re.search(r'\btest_status\s*=\s*PASS\b',text,re.I))
        normalized_text = re.sub(r'[*`]', '', text)
        status = bool(re.search(r'\btest[ _]status\s*[:=]\s*PASS\b',normalized_text,re.I))
        # Full test/parse tools must themselves pass acquisition. This checks
        # whether the final answer states that observed result, not whether
        # merely echoing PASS constitutes a useful investigation.
        result.update(subject_named=str(subject).lower() in text.lower() if subject else None,
                      actual_validation_status_stated=status,
                      requested_status_format=exact_status,
                      limitation_signal=bool(re.search(r'\blimit\w*|\bonly\b|\bdoes not\b|\bnot (?:prove|test|cover|guarantee)\b',text,re.I)))
        result['automatic_failure'] = not result['nonempty'] or not status or not exact_status or result['subject_named'] is False
    else:
        result['automatic_failure'] = not result['nonempty']
    result['source_review_required'] = True
    return result


def prepare_review(out, phase='confirm'):
    out = Path(out)
    if phase not in ('screen','confirm'):
        raise ValueError('Quality review requires screen or confirmation measurements')
    blocks = completed_blocks(out,phase)
    entries, groups, durations = [], {}, {}
    for (model,cell,repeat), block in sorted(blocks.items()):
        for system,data in sorted(block['systems'].items()):
            event = data['event']
            directory = local_path(out,event['details']['path'])/'measurement'
            corpus = load(directory/'corpus.json')
            case = data['case']
            durations[event['attempt_id']] = case['load_duration_s']
            raw = [json.loads(line) for line in (directory/case['name']/'requests.jsonl').read_text().splitlines()]
            for session in corpus['sessions']:
                outputs = set(session['graph']['outputs'])
                for row in raw:
                    if row.get('source_id')!=session['id'] or row.get('node_id') not in outputs:
                        continue
                    text = row.get('generated_text','')
                    reference = session['quality_reference']
                    if cell=='B10':
                        reference = {**reference,'kind':'team'}
                    check = objective(text,reference,capped=row.get('finish_reason')=='length')
                    item = dict(model=model,cell=cell,repeat=repeat,system=system,attempt=event['attempt_id'],
                        request_id=row['payload']['request_id'],source_id=session['id'],traffic_class=row['hint_group'],
                        answer=text,objective=check,source_context=session.get('context',''),reference=reference,
                        instructions=session['turns'],
                        actual_dependencies=[r['generated_text'] for r in raw if r.get('source_id')==session['id'] and
                                             r.get('node_id')!=row['node_id']],
                        tools=[dict(output=re.sub(r'\bin \d+(?:\.\d+)?s\b','in [elapsed time omitted]',p.read_text(errors='replace')))
                               for p in sorted((directory/case['name']/'tools').glob('*.txt'))
                               if p.name.startswith('0-'+str(corpus['sessions'].index(session))+'-')])
                    item['review_id'] = digest({k:item[k] for k in ('attempt','request_id','source_id')})[:24]
                    entries.append(item)
                    groups.setdefault((model,cell,system),[]).append(item)
    chosen = {}
    for key,items in sorted(groups.items()):
        # Round-robin traffic class/repetition strata; then deterministic order.
        strata = {}
        for item in items:
            strata.setdefault((item['repeat'],item['traffic_class']),[]).append(item)
        rng = random.Random(digest(key))
        for values in strata.values():
            rng.shuffle(values)
        sample = []
        while len(sample)<min(32,len(items)):
            for stratum in sorted(strata):
                if strata[stratum] and len(sample)<min(32,len(items)):
                    sample.append(strata[stratum].pop())
        for item in sample:
            chosen[item['review_id']] = 'sample'
    for item in entries:
        if item['objective']['automatic_failure']:
            chosen[item['review_id']] = 'automatic_failure'
        elif item['objective']['length_capped']:
            chosen[item['review_id']] = 'length_capped'
    blind, mapping = [], {}
    for item in entries:
        if item['review_id'] not in chosen:
            continue
        mapping[item['review_id']] = {k:item[k] for k in ('model','cell','repeat','system','attempt','request_id','source_id')}
        blind.append({k:v for k,v in item.items() if k not in ('model','cell','repeat','system','attempt','request_id')})
    random.Random(17011).shuffle(blind)
    canonical = {}
    for item in blind:
        signature = digest({k:item[k] for k in ('answer','source_context','reference','instructions','actual_dependencies','tools')})
        if signature in canonical:
            item['duplicate_of'] = canonical[signature]
        else:
            canonical[signature] = item['review_id']
    destination = out/'analysis'/('quality' if phase=='confirm' else 'quality-screen')
    atomic_json(destination/'rubric.json',RUBRIC)
    atomic_json(destination/'objective.json',dict(schema='stage6-quality-objective-v1',rubric=RUBRIC,outputs=entries))
    atomic_json(destination/'blind-packet.json',dict(schema='stage6-blind-quality-packet-v1',rubric=RUBRIC,items=blind))
    atomic_json(destination/'private-labels.json',mapping)
    (destination/'private-labels.json').chmod(0o600)
    result = dict(status='pending_source_audit',phase=phase,objective_outputs=len(entries),blind_review_items=len(blind),
                  automatic_failures=sum(i['objective']['automatic_failure'] for i in entries),
                  objective_groups=aggregate_objective(entries,durations),
                  rubric_digest=digest(RUBRIC),packet_digest=digest(blind),
                  scope='Objective checks cover every final output; every capped final and automatic failure requires the retained source audit')
    atomic_json(destination/'summary.json',result)
    return result


def aggregate_objective(entries,durations):
    groups = {}
    for item in entries:
        key = ':'.join(item[k] for k in ('model','cell','system'))
        group = groups.setdefault(key,dict(final_outputs=0,automatic_failures=0,length_capped=0,
            observed_checks={},processes={}))
        group['final_outputs'] += 1
        check = item['objective']
        group['automatic_failures'] += check['automatic_failure']
        group['length_capped'] += check['length_capped']
        for name in ('nonempty','schema_valid','reference_exact','reference_span_present',
                     'subject_named','actual_validation_status_stated','requested_status_format','limitation_signal'):
            if check.get(name) is not None:
                observed = group['observed_checks'].setdefault(name,dict(passed=0,observed=0))
                observed['passed'] += bool(check[name])
                observed['observed'] += 1
        process = group['processes'].setdefault(item['attempt'],dict(final_outputs=0,mechanical_passes=0,
            duration_s=durations[item['attempt']]))
        process['final_outputs'] += 1
        process['mechanical_passes'] += not check['automatic_failure']
    for group in groups.values():
        for process in group['processes'].values():
            process['mechanical_passes_per_s'] = process['mechanical_passes']/process['duration_s']
        group['scope'] = ('Every final output, including repeats of the fixed cohort. Mechanical checks '
            'are not source-audited usefulness; slower or poor-quality cases remain in raw throughput.')
    return groups


def finish_review(out, ratings_path):
    from .campaign_contracts import fields
    out = Path(out)
    destination = out/'analysis/quality'
    packet = load(destination/'blind-packet.json')
    ratings = load(ratings_path)
    fields(ratings,('schema','reviewer','rubric_digest','ratings'))
    if ratings['schema']!='stage6-source-audit-v1' or ratings['rubric_digest']!=digest(RUBRIC) or not ratings['reviewer'].strip():
        raise ValueError('Audit identity/rubric mismatch')
    values = {}
    expected = {item['review_id']:item for item in packet['items']}
    for row in ratings['ratings']:
        fields(row,('review_id','source_supported','instruction_followed','useful','unsupported_claims','notes'))
        if row['review_id'] in values or row['review_id'] not in expected:
            raise ValueError('Duplicate or unknown audit item')
        if any(type(row[k]) is not bool for k in ('source_supported','instruction_followed','useful')) or not isinstance(row['notes'],str) or not row['notes'].strip():
            raise ValueError('Every audit item requires explicit judgments and source notes')
        if not isinstance(row['unsupported_claims'],list) or any(not isinstance(c,str) for c in row['unsupported_claims']):
            raise ValueError('Unsupported claims must be an explicit list')
        values[row['review_id']] = row
    if values.keys()!=expected.keys():
        raise ValueError('Source audit does not cover the complete sample and all automatic failures')
    labels = load(destination/'private-labels.json')
    groups = {}
    for key,row in values.items():
        label = labels[key]
        group = groups.setdefault(label['model']+':'+label['cell']+':'+label['system'],dict(reviewed=0,useful=0,unsupported=0))
        group['reviewed'] += 1
        group['useful'] += row['useful']
        group['unsupported'] += bool(row['unsupported_claims']) or not row['source_supported']
    for group in groups.values():
        group['useful_fraction_in_reviewed_sample'] = group['useful']/group['reviewed']
    atomic_json(destination/'source-audit.json',ratings)
    previous = load(destination/'summary.json')
    previous.update(status='audited',reviewer=ratings['reviewer'],reviewed=len(values),groups=groups,
                    audit_digest=digest(ratings),sample_scope='Stratified deterministic sample plus all automatic failures and capped finals; not an unbiased population useful-rate estimate')
    atomic_json(destination/'summary.json',previous)
    return previous
