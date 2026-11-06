"""Persist explicit author audits after reading the first wait/upstream finals."""
import copy,hashlib,json
from pathlib import Path
base=Path(__file__).parent;p=base/'independent-manual-review.json';report=json.loads(p.read_text())
root=base/'stage3-repository';baseline={x['source_id']:x for x in report['cases'][0]['details']}
wait={
'itertools-replace':('incorrect','contradicted','Same false negative-count ValueError and absent-large-window claims as ARC; test_window_size_large exists. Status formatting changes do not repair content.'),
'itertools-map_reduce':('supported','supported_as_input_constraint','Custom reduction is supported; defaultdict grouping requires hashable keyfunc results. No unhashable-key test was identified; an invalid key is an input constraint, not an observed regression.'),
'itertools-collapse':('supported','contradicted','Nesting behavior supported. Empty inputs are valid, and test_collapse explicitly includes atomic integer leaves; implementation catches noniterable leaves.'),
'itertools-sort_together':('supported','contradicted','Trimming unequal inputs is supported; the limitation reverses strict semantics. strict=True rejects unequal lengths, default permits them.'),
'itertools-split_when':('supported','weak','Empty-input behavior supported; finite predicate coverage is a generic limitation. Same substantive content as ARC.')}
up={
'distinct_permutations':('supported','partly_unsupported','Unsortable unique permutations are tested; docstring warns about equal/nonidentical elements, but preserving input order is not a general permutation contract.'),
'split_when':('supported','contradicted','Empty input supported. test_max_split explicitly tests positive maxsplit=1,2 with remaining possible splits.'),
'split_at':('incorrect','contradicted','test_basic explicitly includes empty/single-separator strings and compares str.split; empty input does not raise ValueError as asserted.'),
'windowed':('supported','weak','Padding fact supported; class contains no performance benchmark, but the broad huge-input absence statement is not comprehensive evidence.'),
'replace':('supported','unsupported','test_count proves count limit. No specific noninteger failure was demonstrated; implementation compares its replacement counter with count and may accept numeric nonintegers.'),
'chunked':('supported','supported_narrowly','test_odd proves short final chunk. No direct empty or performance case identified in selected class; no defect thereby established.'),
'partitions':('supported','contradicted','test_empty supported; test_duplicates directly contradicts absent duplicate-element coverage.'),
'collapse':('supported','contradicted','test_collapse_to_level supported. test_collapse includes atomic integer leaves, contrary to absent noniterable-element coverage.'),
'sort_together':('supported','contradicted','Unequal trimming supported. test_invalid_key_list contradicts absent invalid-key-list coverage.'),
'map_reduce':('supported','supported','test_valuefunc proves transformation; implementation materializes grouped value lists before reduction, matching the storage limitation.'),
'seekable':('supported','contradicted','test_past_end supported. test_elements uses map(str,count()) and seekable(iterable) without maxlen, contradicting absent infinite-input coverage.'),
'bucket':('supported','contradicted','Empty/invalid-key behavior is supported by implementation. test_falsey_validator_is_honoured explicitly uses an all-rejecting validator, contradicting absent all-invalid coverage.'),
'set_partitions':('supported','partly_unsupported','Negative k raises ValueError. k is an integer API argument, but min_size/max_size use numeric comparisons and no blanket noninteger failure was established.'),
'ichunked':('supported','contradicted','Out-of-order and bounded-memory cases are tested. test_odd explicitly tests nondivisible input.'),
'numeric_range':('supported','contradicted','test_zero_step supported; test_contains_items_with_inexact_step and test_contains_non_items_with_inexact_step contradict absent inexact-step coverage.'),
'peekable':('supported','contradicted','Indexing supported. Both allegedly unimplemented tests have actual bodies in frozen source and executed passing suite; excerpt truncation cannot establish absent implementation.')}
for case,overrides in [('00-wait',wait),('00-upstream',{'itertools-'+k:v for k,v in up.items()})]:
 details=[]
 for line in (root/case/'requests.jsonl').read_text().splitlines():
  row=json.loads(line)
  if row.get('turn')!=3:continue
  d=copy.deepcopy(baseline[row['source_id']]);old_sha=d['text_sha256'];text=row['generated_text']
  d.update(word_count=len(text.split()),text_sha256=hashlib.sha256(text.encode()).hexdigest())
  if row['source_id'] in overrides:
   fact,limit,note=overrides[row['source_id']];d.update(substantiated_behavior=fact,limitation=limit,source_audit=note)
  elif d['text_sha256']==old_sha:d['same_answer_as']='stage3-repository/00-arc'
  else:raise RuntimeError('Changed final requires explicit source audit: '+row['source_id'])
  details.append(d)
 report['cases'].append(dict(campaign='stage3-repository',case=case,details=details))
p.write_text(json.dumps(report,indent=2)+'\n');print('Recorded',sum(len(c['details']) for c in report['cases']),'total final-answer audits.')
