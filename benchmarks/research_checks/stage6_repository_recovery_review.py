"""Reproduce the explicit source judgments for repository recovery 03 only."""
from pathlib import Path
import copy
from benchmarks.serving_suite.campaign_contracts import load,atomic_json,digest
from benchmarks.serving_suite.campaign_quality import RUBRIC,finish_review
base=Path('output/stage6-2080ti-20261106'); out=base/'addendum-03'; q=out/'analysis/quality-screen'
novel=load(q/'novel-contexts.json'); packet=load(q/'blind-packet.json')['items']
assert digest(packet) == 'e2fff24142215ab609ed5d677d74713de304b0ec45356c7e29bcc62c6fa794ff', 'Review packet changed'
# Explicit judgments after reading source, dependencies, actual tool results and answers.
judgments=[
(True,False,True,[], 'first returns the first element; four tests pass. TypeError on non-iterables follows the for loop. Missing literal test_status=PASS.'),
(True,False,True,[], 'doublestarmap calls func(**item); signature mismatch TypeError is documented and covered in eight passing tests. Missing literal requested status format.'),
(False,False,False,['Invented test_negative name; claimed None confusion persists despite regression coverage.'], 'extract rejects negative indices and seven tests/7776 subtests pass. The referenced regression documents the former None-buffer bug as fixed, not a current limitation. Test is test_negative_one_bug, not test_negative.'),
(False,False,False,['Infinite inputs may fail to terminate; no source evidence that they raise errors merely for infinity.'], 'exactly_n stops after n+1 truthy values or exhaustion; three tests pass. Supported predicate behavior but incorrect infinite-input limitation and missing literal status format.'),
(True,False,True,[], 'filter_map yields non-None mapped values; four tests pass. Non-iterable TypeError follows for x in iterable. Missing literal status format.'),
(True,True,True,[], 'gray_product changes one coordinate at a time; test_basic agrees, all five tests pass, and len<2 explicitly raises ValueError. Literal requested status present.'),
(False,False,False,['Custom key functions and grouped/non-adjacent data are not gray_product behavior.','Answer claims four passed tests despite actual five.'], 'Capped intermediate repeats unrelated groupby claims and ends mid-sentence. gray_product takes iterables/repeat, no key function; actual tool says five passed. Does not distinguish evidence from assumptions.'),
(True,False,True,[], 'iequals abc/acb false follows order-sensitive equality; ten tests pass. Non-iterable TypeError follows zip; missing literal status format.'),
(True,False,False,[], 'iequals accepts equal sequences across types, ten tests pass. False for unequal lengths is intended semantics, not a limitation. Missing requested literal status format.'),
(True,False,True,[], 'ilen consumes the iterable and counts elements; one test passes. Full consumption prevents terminating on infinite input. Missing literal status format.'),
(False,False,False,['Non-hashable-key inconsistency is unsupported; groupby uses equality, not hash lookup.'], 'groupby_transform wraps groupby and optionally maps/reduces; four tests pass. Claimed hashing limitation is unsupported. Final unnecessarily repeats tool command and misses literal status format.'),
(True,False,True,[], 'first defaults/ValueError behavior matches code and four passing tests. Missing literal status format; extra search command is unnecessary but does not negate factual summary.'),
(True,True,True,[], 'gray_product test_basic checks coordinate-changing order, five tests pass, and the minimum-two-elements restriction is explicit. Literal status present.'),
(True,False,False,[], 'iequals compares element/order/length; ten tests pass. Unequal lengths yielding False is correct behavior, not failure or concrete limitation; missing literal status.'),
(False,False,False,['gray_product generates Cartesian-product tuples, not iterable comparisons.'], 'Five gray_product tests pass and non-iterable TypeError is plausible from tuple conversion. The claimed main comparison behavior misidentifies purpose; literal status alone is insufficient.'),
(False,False,False,['Final answers groupby_transform instead of gray_product.'], 'Final propagates unrelated grouping statements from capped dependency, omits gray_product and observed PASS. Actual five tests concern Cartesian-product ordering and minimum input size. Not useful for requested subject.'),
(False,False,False,['Cannot handle infinite or unbounded iterables contradicts mixed finite/infinite test.'], 'interleave stops at shortest via zip; three tests pass including count() with finite companions. Stopping at shortest is intended, and infinite input is supported. Missing literal status.'),
(False,False,False,['Answer names duplicates_everseen and first-element behavior instead of iequals.'], 'Actual task/source compare iterables and ten tests pass. Answer is on the wrong function and gives wrong behavior; Test status=PASS also lacks requested underscore.')]
keys=('answer','source_context','reference','instructions','actual_dependencies','tools','response_role','current_instruction')
def sig(i):return digest({k:i[k] for k in keys})
canon={sig(i):i for i in reversed(packet)}
reused=load(q/'reused-canonical.json'); canonical={}
for s,i in canon.items():
 if i['review_id'] in reused:
  r=copy.deepcopy(reused[i['review_id']]);r['notes']+=' Identical full-context reuse from addendum-02 review '+r['review_id']+'.';r['review_id']=i['review_id'];canonical[s]=r
for i,(supported,followed,useful,claims,notes) in zip(novel,judgments,strict=True):
 canonical[sig(i)]=dict(review_id=i['review_id'],source_supported=supported,instruction_followed=followed,useful=useful,unsupported_claims=claims,notes=notes)
ratings=[]
for i in packet:
 r=copy.deepcopy(canonical[sig(i)])
 if r['review_id']!=i['review_id']:r['notes']+=' Exact complete-context duplicate of '+r['review_id']+'.'
 r['review_id']=i['review_id'];ratings.append(r)
record=dict(schema='stage6-source-audit-v1',reviewer='Codex source review; system and speed labels hidden; 24 exact historical full-context judgments reused, 18 novel contexts explicitly read',rubric_digest=digest(RUBRIC),ratings=ratings)
atomic_json(q/'source-ratings.json',record)
# Existing finish_review consumes analysis/quality; retain the screen packet there as a derived audit mirror.
import shutil
shutil.copytree(q,out/'analysis/quality',dirs_exist_ok=True)
summary=finish_review(out,q/'source-ratings.json')
evidence=Path('evidence/stage6-2080ti-20261106/repository-recovery-03')
atomic_json(evidence/'source-judgments.json',record);atomic_json(evidence/'quality-final.json',summary)
atomic_json(evidence/'review-context-index.json',dict(signature_fields=keys,contexts=[dict(signature=s,canonical_id=i['review_id'],historical_reuse=i['review_id'] in reused) for s,i in canon.items()],packet_digest=digest(packet)))
print(summary['reviewed'], summary['groups'])
