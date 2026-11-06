import json,hashlib,copy
from pathlib import Path
b=Path(__file__).parent;p=b/'independent-manual-review.json';a=json.loads(p.read_text());known={}
for c in a['cases']:
 if c['campaign']=='stage3-repository-reserve':
  for d in c['details']:known[(d['source_id'],d['text_sha256'])]=(d,c['case'])
notes={
'split_before':('supported','missing','Splitting before a matching item is documented and implemented. No concrete limitation supplied.'),
'substrings':('supported','missing','Loop over n from1 through input length yields shortest substrings first. No concrete limitation supplied.'),
'split_after':('supported','missing','Matching item ends its group. No concrete limitation supplied.'),
'derangements':('supported','missing','Position-based derangements supported. No concrete limitation supplied.'),
'distribute':('supported','supported','tee/islice distribution and n<1 ValueError source/test match.'),
'interleave_evenly':('supported','supported_with_qualification','Unequal-length interleaving tested. Known lengths required; manual lengths need be specified only when automatic __len__ lookup is unavailable, not for every input.'),
'zip_offset':('supported','missing','shortest/longest options supported. No concrete limitation supplied.'),
'run_length':('supported','missing','encode uses groupby/ilen to encode runs. No concrete limitation supplied.'),
'split_into':('supported','missing','Docstring and test_iterable_too_small/extra show oversized groups give shorter/empty lists. No concrete limitation supplied.'),
'distinct_combinations':('supported','supported','Distinct outputs and negative-r ValueError source/test match.'),
'stagger':('supported','missing','Relative offsets implemented through zip_offset. No concrete limitation supplied.'),
'consecutive_groups':('supported','missing','Grouping by adjacent values/ordering supported. No concrete limitation supplied.'),
'islice_extended':('supported','missing','Negative slice values/steps supported with documented caching. No concrete limitation supplied.'),
'divide':('supported','supported','Source divides ordered contiguous ranges; n<1 ValueError tested.'),
'zip_broadcast':('supported','contradicted','Broadcasting supported, but (str,bytes) does not restrict ALL scalars: get_iterator also treats any noniterable as scalar, and docstring broadcasts integer0. scalar_types selects additional iterable types treated as scalar.')}
assert not any(c['campaign']=='stage3-repository-reserve' and c['case']=='00-upstream' for c in a['cases']);details=[]
for line in (b/'stage3-repository-reserve/00-upstream/requests.jsonl').read_text().splitlines():
 r=json.loads(line)
 if r.get('turn')!=3:continue
 sid=r['source_id'];sha=hashlib.sha256(r['generated_text'].encode()).hexdigest();key=sid,sha
 if key in known:
  prior,origin=known[key];d=copy.deepcopy(prior);d['same_answer_as']='stage3-repository-reserve/'+origin;d['review_method']='Exact source_id/text SHA reuse on identical reserve inputs and actual passing tool outcomes.'
 else:
  f,l,n=notes[sid.removeprefix('itertools-')];d=dict(source_id=sid,requested_subject_present=True,observed_test_outcome_correct=True,substantiated_behavior=f,limitation=l,source_audit=n,word_count=len(r['generated_text'].split()),text_sha256=sha)
 details.append(d)
assert len(details)==16
a['cases'].append(dict(campaign='stage3-repository-reserve',case='00-upstream',details=details));p.write_text(json.dumps(a,indent=2)+'\n');print('Recorded16 upstream reserve finals:1 exact reuse/15 new audits. One incorrect behavior, eleven missing limitations, one contradicted limitation.')
