import json,hashlib
from pathlib import Path
b=Path(__file__).parent;p=b/'independent-manual-review.json';a=json.loads(p.read_text());root=b/'stage3-documents'
notes={
'pep-0008':('supported','supported','PEP8 line118 explicitly makes four-space rule optional for continuation lines; aligned/hanging indentation rule matches source.'),
'pep-0020':('supported','supported_metadata_boundary','Quoted obvious-way aphorism matches text. Actual metadata has no Python-Version; this is a document metadata boundary, not evidence of unsupported Python versions.'),
'pep-0257':('supported','supported_with_qualification','Module/public-function/class docstrings are recommended. Informational/Active metadata does not certify actual implementation or maintenance activity; do not infer unmaintained status.'),
'pep-0343':('supported','supported','with cleanup on exceptions in an entered block and truthy __exit__ suppression are explicit semantics. Suppression is a designed behavior, not an observed failure.'),
'pep-0380':('supported','supported','Refactoring rationale explicitly says catching GeneratorExit without reraising prevents exact behavioral preservation (lines224ff).'),
'pep-0420':('supported','weak','Split portions and dynamic __path__ recomputation supported. Predicted dependency-management complexity is speculative rather than a concrete demonstrated limitation.'),
'pep-0440':('supported','supported_with_speculation','Public-version grammar matches PEP. It explicitly allows semantic versioning without requiring it; claimed ambiguity is not established.'),
'pep-0451':('supported','contradicted','ModuleSpec purpose supported. Post-PEP responsibility list369ff transfers sys.modules/import boilerplate to ModuleSpec; it does not leave all creation/registration/cleanup boilerplate with loaders.'),
'pep-0484':('supported','supported','Sequence[int] appears explicitly; runtime checking is outside core proposal and requires third-party implementation (lines75-89).'),
'pep-0492':('supported','contradicted','async def coroutine rule supported. PEP contains explicit Backwards Compatibility (859ff) and Reference Implementation (1300ff), contrary to claimed absent/extremely limited discussion.'),
'pep-0498':('supported','contradicted','Shared format mini-language supported. != exception applies to ! expression syntax; it is not an exception to the backslash restriction in this historical PEP text (235ff/240ff).'),
'pep-0508':('supported','supported_historical_document_scope','Name/extras/version/URL/marker summary supported. The frozen PEP explicitly says wheel extra special handling has no current specification241ff; this audit describes that document, not all later packaging standards.'),
'pep-0517':('supported','supported_with_qualification','build-backend selects a Python backend object; setup.py may remain for compatibility/backend needs, but is not universally required by PEP517.'),
'pep-0518':('supported','supported_with_qualification','Build-dependency declaration supported. Rejected semantics-version237ff discusses renaming keys/tables for incompatible semantic changes; not only table renaming.'),
'pep-0526':('supported','contradicted','Variable annotation syntax supported. Conditional readability problem83ff concerns OLD type comments; new syntax explicitly improves this with declaration before branch168ff.'),
'pep-0567':('supported','supported_with_qualification','Context/contextvars semantics supported. No direct reference/current Context mapping mutation is explicit, but ContextVar.set still changes current values; the mapping restriction does not make values immutable.')}
assert not any(c['campaign']==root.name and c['case']=='00-arc' for c in a['cases']);details=[]
for line in (root/'00-arc/requests.jsonl').read_text().splitlines():
 r=json.loads(line)
 if r.get('turn')!=3:continue
 fact,limit,note=notes[r['source_id']];details.append(dict(source_id=r['source_id'],requested_subject_present=True,requested_metadata_correct=True,observed_test_outcome_correct=True,substantiated_behavior=fact,limitation=limit,source_audit=note,word_count=len(r['generated_text'].split()),text_sha256=hashlib.sha256(r['generated_text'].encode()).hexdigest()))
assert len(details)==16
a['cases'].append(dict(campaign=root.name,case='00-arc',details=details));p.write_text(json.dumps(a,indent=2)+'\n');print('Recorded16 document ARC finals; all metadata/status/core rules supported, four contradicted limitations and one speculative limitation; other boundaries retain qualifications.')
r=json.loads((root/'result.json').read_text())
for c in r['cases']:
 print(c['name'],c['agent'].get('quality',{}).get('passed'),[d for d in c['agent'].get('quality',{}).get('details',[]) if not d.get('passed')])
