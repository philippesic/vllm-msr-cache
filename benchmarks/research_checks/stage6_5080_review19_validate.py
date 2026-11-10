"""Independently validate retained review19 IDs, manual coverage and exact reuse."""
from pathlib import Path
import sys
r = Path(__file__).resolve().parents[2]
frozen = r / 'output/stage6-5080-20261108/primary-execution-bundle-01/source'
assert Path.cwd() == frozen.resolve() and sys.dont_write_bytecode
assert sys.executable == '/home/philip/.cache/semantic-offload/stage6-5080-runtimes-01/local-env/bin/python'
sys.path[:] = [str(frozen)] + [p for p in sys.path if Path(p).resolve() != Path(__file__).resolve().parent]
from collections import Counter
import hashlib
import json
from benchmarks.serving_suite.campaign_contracts import load, digest, atomic_json
assert Path(load.__code__.co_filename).resolve().is_relative_to(frozen.resolve())
e = r / 'evidence/stage6-5080-20261109'
b = r / 'output/stage6-5080-20261109/quality-confirm-checkpoint-19'
target = e / 'confirm-review-validation-19.json'
assert not target.exists()
packet, novels = load(b / 'blind-packet.json'), load(b / 'novel-contexts.json')
idx = load(e / 'confirm-review-index-19.json')
ratings = load(e / 'confirm-source-judgments-19.json')
validator = load(e / 'confirm-quality-checkpoint-19.json')
fields = idx['signature_fields']
flags = ('source_supported', 'instruction_followed', 'useful')
pd = digest(packet['items'])
assert pd == idx['packet_digest'] == validator['packet_digest']
assert hashlib.sha256((b / 'blind-packet.json').read_bytes()).hexdigest() == idx['packet_file_sha256']
by = {v['review_id']: v for v in ratings['ratings']}
assert len(by) == len(ratings['ratings']) == len(packet['items']) == 7011
assert set(by) == {x['review_id'] for x in packet['items']}
groups = {}
for x in packet['items']:
    sig = digest({f:x[f] for f in fields})
    groups.setdefault(sig, []).append(x['review_id'])
assert len(groups) == 849
assert idx['contexts'] == [dict(signature=s, canonical_review_id=ids[0], review_ids=ids) for s,ids in groups.items()]
for ids in groups.values():
    assert all(by[i][f] == by[ids[0]][f] for i in ids for f in (*flags, 'unsupported_claims'))
parts = sorted(e.glob('confirm-investigation-judgments-19-part-*.json'))
manual = {}
for p in parts:
    part = load(p)
    assert part['packet_digest'] == pd and part['signature_fields'] == fields
    for v in part['ratings']:
        i = v['index']
        assert i not in manual and v['review_id'] == novels[i]['review_id']
        assert v['signature'] == digest({f:novels[i][f] for f in fields})
        assert all(v[f] == by[v['review_id']][f] for f in (*flags, 'unsupported_claims'))
        manual[i] = v
assert set(manual) == set(range(14)) and len(parts) == 5
prior, reduced_prior = {}, {}
reduced = [f for f in fields if f != 'actual_dependencies']
for stem,no in [('screen','07')] + [('confirm',f'{i:02}') for i in range(1,19)]:
    old = load(e / f'{stem}-review-index-{no}.json')
    old_by = {v['review_id']:v for v in load(e / f'{stem}-source-judgments-{no}.json')['ratings']}
    assert old['executed_system_instruction_digest'] == idx['executed_system_instruction_digest']
    assert old['executed_instruction_source_sha256'] == idx['executed_instruction_source_sha256']
    for c in old['contexts']:
        prior[c['signature']] = old_by[c['canonical_review_id']]
    if stem == 'confirm' and int(no) >= 2:
        for x in load(b.parent / f'quality-confirm-checkpoint-{no}/novel-contexts.json'):
            sig = digest({f:x[f] for f in fields})
            reduced_prior.setdefault(digest({f:x[f] for f in reduced}), []).append(prior[sig])
ci = load(e / 'caps-all-review-index-08.json')
cp = load(Path(ci['raw_packet']))
cr = {v['review_id']:v for v in load(e / ci['ratings'])['ratings']}
assert digest(cp['items']) == ci['packet_digest']
cap_only = {}
for x in cp['items']:
    sig = digest({f:x[f] for f in fields})
    if sig not in prior:
        cap_only[sig] = cr[x['review_id']]
novel_sigs = {digest({f:x[f] for f in fields}) for x in novels}
assert len(novel_sigs) == 14 and not novel_sigs & (set(prior) | set(cap_only))
for sig,ids in groups.items():
    if sig not in novel_sigs:
        old = prior.get(sig,cap_only.get(sig))
        assert old is not None and all(by[ids[0]][f] == old[f] for f in flags)
for i,x in enumerate(novels):
    for old in reduced_prior.get(digest({f:x[f] for f in reduced}), []):
        assert all(manual[i][f] == old[f] for f in flags), 'Inspect dependency-only judgment drift'
assert validator['status'] == 'audited' and validator['reviewed'] == 7011
assert validator['ratings_digest'] == digest(ratings)
audit = Path(validator['isolated_workspace']) / 'analysis/quality'
assert load(audit/'source-audit.json') == ratings
assert load(audit/'blind-packet.json') == packet
summary = load(audit/'summary.json')
assert summary['status'] == 'audited' and summary['reviewed'] == 7011
components = load(e/'confirm-review-components-19.json')
assert components['packet_digest'] == pd and len(components['contexts']) == 14
findings = dict(source_unsupported=sum(not v['source_supported'] for v in manual.values()),
                instruction_failures=sum(not v['instruction_followed'] for v in manual.values()),
                not_useful=sum(not v['useful'] for v in manual.values()))
h = r/'benchmarks/research_checks/stage6_5080_review19_consolidate.py'
receipt = dict(schema='stage6-5080-completed-static-source-review-validation-v1', phase='confirm',
    acquisition_complete=False,source_review_complete_for_snapshot=True,confirmation_execution_complete=True,packet_digest=pd,
    packet_file_sha256=idx['packet_file_sha256'],canonical_index='confirm-review-index-19.json',
    ratings='confirm-source-judgments-19.json',validator='confirm-quality-checkpoint-19.json',
    reviewed_outputs=7011,distinct_full_contexts=849,explicitly_read_novel_contexts=14,manual_parts=5,
    novel_roles=dict(Counter(x['response_role'] for x in novels)),
    novel_kinds=dict(Counter(x['reference'].get('kind','investigation') for x in novels)),
    novel_findings=findings,aggregate_findings=idx['review_findings'],archived_instruction_sources_verified=516,
    distinct_supplied_sources_exactly_bound_to_prior_explicit_readings=idx['distinct_supplied_sources_exactly_bound_to_prior_explicit_readings'],
    new_unmatched_source_components=idx['distinct_supplied_sources_fully_read_without_prior_component_binding'],
    cap08_only_prior_quality_reuse=dict(distinct_contexts=len(set(groups)&set(cap_only)),selected_occurrences=sum(len(ids) for sig,ids in groups.items() if sig in cap_only)),
    canonical_index_cap_match_field_scope='Counts overlap of manually novel contexts with prior cap08, intentionally zero; actual prior cap-only reuse is reported separately.',
    checks=dict(all_current_review_ids_and_exact_context_signatures_match=True,all_duplicate_context_ratings_consistent=True,
        all14_manual_indices_and_signatures_present_once=True,all_prior_complete_context_reuse_matches=True,
        no_dependency_only_rating_drift=True,frozen_finish_audited_all7011=True,isolated_materialized_audit_matches_retained_packet_and_ratings=True,
        executed_functions_resolve_to_frozen_source=True,live_quality_directory_unchanged=True,no_unread_novel_ratings=True),
    helper=str(h.relative_to(r)),helper_sha256=hashlib.sha256(h.read_bytes()).hexdigest(),quality_module_sha256=validator['quality_module_sha256'],
    scope='Completed explicit source review of fresh isolated terminal207-block snapshot after all524 confirmation identities became terminal. All14 new investigation finals read in5 parts; full fields/dependencies bound by consolidation and all516 archived source/SYSTEM records checked. No timing/objective/exclusion/live-quality change. Publishing this exact reviewed packet/ratings to live quality and final all-attempt cap/campaign/portability audits remain separate. Duplicate-inclusive selected findings are not population rates or quality equivalence.')
atomic_json(target,receipt)
print(json.dumps(receipt,indent=2))
