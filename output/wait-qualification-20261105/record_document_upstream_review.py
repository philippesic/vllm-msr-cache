import json,hashlib
from pathlib import Path
b=Path(__file__).parent;p=b/'independent-manual-review.json';a=json.loads(p.read_text());root=b/'stage3-documents'
notes={
'pep-0020':('supported','weak','Obvious-way aphorism is in PEP20; claimed implementation-complexity limitation is an interpretation, not a concrete established restriction.'),
'pep-0008':('supported','supported_with_qualification','Four-space block indentation and continuation flexibility supported. Preserve backward compatibility is a general style exception, not a special old-Python indentation rule.'),
'pep-0380':('supported','supported','yield-from refactoring and GeneratorExit nonreraising limitation explicitly documented224ff.'),
'pep-0498':('supported','contradicted','Format mini-language supported; != exception concerns ! syntax, not backslashes. Same conflation as ARC/wait in differently formatted text.'),
'pep-0343':('supported','supported','with/try-finally cleanup and truthy __exit__ suppression supported. Additional assertion that implementation support is consistent is not established by metadata checks.'),
'pep-0420':('supported','supported_with_speculation','Implicit namespaces omit __init__.py; parent-path change detection/recomputation documented. No measured performance-overhead magnitude or observed failure is established.'),
'pep-0257':('supported_with_qualification','supported_metadata_boundary','First-statement docstring definition and triple-double-quote recommendation supported; one-line forms apply where suitable. Active/Informational does not certify every tool/environment.'),
'pep-0492':('supported','supported_with_speculation','async def and CO_COROUTINE flag are explicit; unmeasured complexity is interpretation rather than demonstrated overhead.'),
'pep-0451':('supported','contradicted','ModuleSpec purpose supported. No significant finder/loader responsibility change conflicts with explicit post-PEP assignment369ff of import boilerplate to ModuleSpec.'),
'pep-0440':('partly_incorrect','supported_with_speculation','Epoch is OPTIONAL ([N!]); quoted N! grammar omits optionality. Version uniqueness is supported. Semantic versioning not mandatory; ambiguity claim not demonstrated.'),
'pep-0517':('supported','partly_incorrect','Missing build-backend legacy fallback supported148ff. Read-only source support is SHOULD and backends MAY opt out237ff; answer overstates it as MUST.'),
'pep-0484':('supported','supported','Unannotated self/cls implicit types explicit109ff; runtime checking outside proposal/third-party implementation explicit75-89.'),
'pep-0526':('supported','supported','Local/global annotations supported; same-scope global/nonlocal annotated target SyntaxError documented320ff.'),
'pep-0518':('supported','unsupported_scope_claim','build-system.requires supported. Actual universal adoption/enforcement is not measured by document metadata, so assertion of nonuniversal implementation needs external evidence or scope qualification.'),
'pep-0567':('supported','supported_with_qualification','Context mapping cannot be directly mutated, but changing values specifically requires ContextVar.set after making a context current; Context.run alone does not set a value.'),
'pep-0508':('supported','partly_contradicted','Name/extras rule supported. Document contains implementation-related marker behavior, complete grammar and battle-tested implementation motivation; missing-implementation assertion overbroad, and widespread support is not established by metadata.')}
assert not any(c['campaign']==root.name and c['case']=='00-upstream' for c in a['cases']);details=[]
for line in (root/'00-upstream/requests.jsonl').read_text().splitlines():
 r=json.loads(line)
 if r.get('turn')!=3:continue
 f,l,n=notes[r['source_id']];details.append(dict(source_id=r['source_id'],requested_subject_present=True,requested_metadata_correct=True,observed_test_outcome_correct=True,substantiated_behavior=f,limitation=l,source_audit=n,word_count=len(r['generated_text'].split()),text_sha256=hashlib.sha256(r['generated_text'].encode()).hexdigest()))
assert len(details)==16
a['cases'].append(dict(campaign=root.name,case='00-upstream',details=details));p.write_text(json.dumps(a,indent=2)+'\n');print('Recorded16 new upstream document audits; all requested metadata/test outcomes correct, grammar optionality error and multiple incorrect/unsupported limitations retained.')
