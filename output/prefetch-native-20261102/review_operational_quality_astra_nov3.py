"""Grounded independent notes after reading every selected distinct text."""
import hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent
samples=json.loads((H/'live-nov3-operational-quality-samples.json').read_text())
unique={}
for row in samples['selected']:unique.setdefault((row['source_id'],row['turn'],row['generated_text_sha256']),row)
notes=[
 'The reported remainder-transformer propagation problem is supported, but the output incorrectly says the final result is missing. The issue includes an actual result with wrong boolean dtype/values.',
 'Faithful concise summary of the reported generated-fixture visibility change and CI impact. It restates the issue rather than adding new evidence, as this is turn0.',
 'Correctly distinguishes current ValueError behavior from the proposed empty Min/Max identities. Does not claim an implementation.',
 'Faithful summary of the quoted rational-precision reproducer. Evidence is the reported example, not new service execution.',
 'Correctly limits the latest class-location grep: it does not show set_output behavior. The original issue does supply actual behavioral evidence, so the missing-evidence wording should be restricted to the latest observation.',
 'The latest grep shows MinMaxBase and Min, not Max. The claim that both class definitions were observed is unsupported; uncertainty about zero-argument behavior is appropriate.',
 'Appropriately says a bare recorded exit0 is not evidence about fixture visibility. Earlier issue evidence supports the rest of the summary.',
 'Reported precision problem is supported by the issue. Latest truncated lambdify/module excerpt does not locate the rational printer; requesting precise implementation evidence is appropriate.',
 'SQL templates are accurately quoted from the latest grep, but the answer misses the issue-specific column-suffix whitespace blocker. Multiline backend list and code-like templates violate the concise one-paragraph/no-code contract.',
 'The latest grep confirms function definition/call locations, not its body or runtime behavior. The indentation mechanism comes from the original issue; zero indentation itself is real, the bug is including the first line in the minimum. Multiline list violates the paragraph contract.',
 'Latest grep does not establish that the problematic membership check is currently being executed. The output overstates this and mirrors the issue informal wording that membership returns an array rather than raising while truth-testing it. Fenced snippet/multiple paragraphs violate the format contract.',
 'Faithful issue-grounded account of BETWEEN using calendar-year bounds for ISO-year filtering.',
 'Invents a documentation basis and treats referenced source line numbers as inspected implementation. Only a short issue statement and requested line change are supplied.',
 'Faithfully reports path_info dropping the script prefix. The referenced patch is a link in the issue, not an inspected patch or applied change.',
 'Reported locale problem is supported, but it misses the en-lat-us versus en-latn-us typo in the reproduced URL. Expected200 behavior is explicitly stated; a successful implementation result is what is absent.',
 'Correctly limits URL-registration grep evidence: it locates catch_all_view but does not establish FORCE_SCRIPT_NAME handling.',
 'Correctly says class/subclass locations do not resolve the ISO-year bug. Original query examples remain behavioral evidence despite the latest grep being nondiagnostic.',
 'Correctly identifies the broad class-name grep as nondiagnostic for deletion/PK behavior and names the missing implementation/runtime evidence.',
 'A function-signature default is not an observed application configuration or a tested attempted fix. The output overstates both configuration and its observed inability to resolve the locale issue.',
 'Same unsupported claim that both Min and Max definitions were observed; only MinMaxBase/Min appear. Correctly does not claim the requested change was implemented.',
 'Restates the initial fixture-visibility report and largely ignores the latest bare exit0. No evidence supports a resolution; no service command-execution claim.',
 'Reported rational-precision issue is supported by the issue, not established afresh by the truncated module excerpt. Correctly notes no evidence of a fix, but does not identify the missing printer-specific code.',
]
assert len(unique)==len(notes)==22
rows=[]
for index,(key,row) in enumerate(unique.items()):
 rows.append({'source_id':row['source_id'],'turn':row['turn'],'generated_text_sha256':row['generated_text_sha256'],'example_case':row['case'],'example_wave':row['wave'],'note':notes[index]})
out={'schema':'astra-operational-grounded-quality-review-v1','sample_sha256':hashlib.sha256((H/'live-nov3-operational-quality-samples.json').read_bytes()).hexdigest(),
 'selected_observations':len(samples['selected']),'distinct_texts_reviewed':len(rows),'unique_source_issues':len({r['source_id'] for r in rows}),
 'review_method':'Read all22distinct generated texts against retained initial issue/latest recorded observation. Exact duplicate text/source/turn observations share notes; not an accuracy-equivalence test.',
 'all_normal_outputs':{'native8':{'responses':1728,'Status_prefix':1728,'code_fences':1,'multiple_blank_blocks':8,'degeneracy_flags':0},'candidate':{'responses':1728,'Status_prefix':1728,'code_fences':1,'multiple_blank_blocks':6,'degeneracy_flags':0}},
 'notes':rows,'assessment':'No conspicuous runaway generation or claim that this service edited files/ran commands in the selected texts. Shared factual/source-reading defects persist, including unsupported current-execution inference and treating a signature default as runtime configuration. Human review is necessary; draft usefulness/correction cost and autonomous task success remain unmeasured.',
 'quality_equivalence_proven':False,'production_endorsement':False}
(H/'live-nov3-operational-quality-review.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'selected':len(samples['selected']),'distinct_reviewed':22,'issues':out['unique_source_issues']}))
