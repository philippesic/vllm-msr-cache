"""Record the explicit first-ARC source audit; no automatic quality eligibility changes."""
import json,hashlib
from pathlib import Path
base=Path(__file__).parent;root=base/'stage3-repository';case='00-arc'
notes={
'windowed':('supported','contradicted','WindowedTests.test_step explicitly includes step 8 with iterable length 7. Padding behavior is supported.'),
'distinct_permutations':('supported','supported_with_qualification','Docstring explicitly warns about equal but non-identical elements; test_unsorted_equivalent covers 1/True. Calling them distinct is imprecise, but the warning is documented.'),
'bucket':('supported','weak','Validator behavior is supported. Tests are functional rather than performance benchmarks; the broad complex-edge-case absence claim is not established.'),
'collapse':('supported','contradicted','Implementation yields atomic noniterable nodes after catching TypeError; empty inputs with large levels do not exhibit the asserted TypeError. Nesting behavior is supported.'),
'split_when':('supported','weak','test_empty_iterable supports the fact. A finite predicate sample does not cover every predicate, but this is a generic limitation.'),
'replace':('incorrect','contradicted','Implementation validates window_size, not negative count; a negative count prevents replacements rather than raising ValueError. test_window_size_large explicitly covers oversized windows.'),
'split_at':('supported','unsupported','Empty cases occur in test_basic. The limitation conflates non-list/string with noniterable, and invents an inadequate exception-message failure; generic iterables are supported.'),
'partitions':('supported','contradicted','test_empty supports the fact; test_duplicates explicitly covers repeated elements, contrary to the limitation.'),
'ichunked':('partly_incorrect','supported_narrowly','Negative n raises ValueError, but n=0 yields no chunks (test_zero and test_zero_nonempty), so the asserted zero-size error is false. No concurrent source-mutation test was identified.'),
'peekable':('supported','contradicted','Indexing does not advance. test_slicing_negative_step_omitted_stop_read_bound uses count() to verify bounded reads; an absence-of-infinite-input coverage claim is false.'),
'map_reduce':('supported','supported_narrowly','test_reducefunc supports the fact. test_valuefunc converts to int; the default string-valued case is covered without a supplied valuefunc. Hypothetical reducer type incompatibility is not an observed function bug.'),
'chunked':('supported','supported_narrowly','test_odd supports the exact final short chunk. The selected ChunkedTests class has no direct empty/n=0 case; this is a class-specific coverage observation, not proof of a defect.'),
'numeric_range':('supported','partly_unsupported','Floating values are supported and the docstring warns about floating precision. The stronger lack-of-explicit-handling claim is not established; test_contains_items_with_inexact_step and its non-items counterpart exist.'),
'sort_together':('supported','unsupported','test_uneven_iterables supports trimming. Implementation uses zip over generic iterables; mixed list/tuple acceptance is intended, not an established failure.'),
'set_partitions':('supported','contradicted','k<1 raises ValueError. Implementation converts any iterable to a list and operates on arbitrary elements; the alleged noninteger/non-list restriction is false.'),
'seekable':('supported','contradicted','test_exhaustion_reset supports the fact. test_maxlen_zero and explicit zero-cache peek/bool tests contradict the limitation.')}
rows=[json.loads(x) for x in (root/case/'requests.jsonl').read_text().splitlines() if x]
corpus=json.loads((root/'corpus.json').read_text());sources={s['id']:s for s in corpus['sessions']};details=[]
for row in rows:
 if row.get('turn')!=3:continue
 symbol=sources[row['source_id']]['quality_reference']['symbol'];fact,limit,note=notes[symbol]
 details.append(dict(source_id=row['source_id'],requested_subject_present=True,observed_test_outcome_correct=True,
  substantiated_behavior=fact,limitation=limit,source_audit=note,word_count=len(row['generated_text'].split()),
  text_sha256=hashlib.sha256(row['generated_text'].encode()).hexdigest()))
report=dict(schema='qualification-independent-source-audit-v1',reviewer='Codex source audit, not an independent person or independent human evaluation',
 scope='Predeclared rubric facets and explicit unsupported/contradicted claims. No post-hoc exclusion from timing; no new scalar quality grade.',
 cases=[dict(campaign='stage3-repository',case=case,details=details)])
p=base/'independent-manual-review.json'
if p.exists():raise FileExistsError(p)
p.write_text(json.dumps(report,indent=2)+'\n');print('Recorded',len(details),'first-ARC final-answer reviews.')
