"""Persist Astra's manual source-grounded review; not an accuracy benchmark."""
import hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent
p=H/'live-nov3-operational-v2-quality-samples.json';x=json.loads(p.read_text())
notes={
0:'Faithful to reported fixture visibility regression; report rather than independent reproduction.',
1:'Defect: says final result is missing, whereas quoted actual results contain both columns with wrong Boolean dtype.',
2:'Faithful issue summary; appropriately distinguishes proposal from implementation.',
3:'Defect: confuses combining factors of equal multiplicity with higher multiplicity, and misdescribes multiplicities1/3 for the selected factors. Shared exact output in both B arms.',
4:'Supported rational precision issue summary.',
5:'Latest grep facts supported; absence of direct runtime evidence is fair, but initial issue already supplies the reported link mismatch.',
6:'Accurately limits latest class-definition observation; line853 claim comes from initial issue.',
7:'Reasonable separation of initial report from empty successful observation; requests corroboration, does not claim service execution.',
8:'Overstatement: latest grep lists MinMaxBase and Min, not Max. Correctly says zero-argument behavior remains unverified.',
9:'Source function/module description supported; no claim of a fix.',
10:'Defect: grep definition/call lines do not prove execution of problematic code. Fenced code/multiple paragraphs violate draft format; no claim that service itself ran commands.',
11:'Faithful initial reported SQL behavior.',
12:'Defect: adds nonexistent documentation basis and treats suggested line280 change as observed implementation. Initial issue supplies only report and pointer.',
13:'Faithful reported redirect distinction; referenced patch is a source claim, not independently verified.',
14:'Generally faithful reported locale problem, but fails to flag en-lat-us versus en-latn-us inconsistency in reproduction.',
15:'Latest definition/registration supported; appropriately cannot determine behavior from grep.',
16:'Faithful initial report and correctly limited grep evidence.',
17:'Faithful limitation of unrelated Model grep results; no reproduction claimed.',
18:'Defect: signature default is called application configuration. Latest observation does not establish actual runtime argument value.',
19:'Faithful fixture report with reasonable unresolved configuration question.',
23:'Supported rational precision summary.',
24:'Accurately limits class-definition evidence.',
25:'Exit0 inference is supported as recorded command outcome; no service execution claim.',
26:'Defect: no Max definition or Min body shown; cannot infer absence of zero-argument logic from class-header grep.',
27:'Reported precision issue supported but weak responsiveness: latest observation already identifies lambdify file/function/module mappings.',
28:'Mechanism is in initial issue, not latest grep. Multi-line list violates concise-paragraph contract; no patch execution claimed.',
29:'Defect: grep call presence still does not prove code executed. Code fence/multiple paragraphs violate contract; caveat properly attributes error to original traceback.',
30:'Reported SQL issue supported; fix status is attributed too strongly to linked PR description without inspecting or executing patch.',
34:'Class names/get_bound_params string supported. Calls error-message string a comment; loses issue-specific blocker.',
35:'Supported registration observation and explicit lack of behavior evidence.',
36:'Correctly distinguishes broad Model grep from actual deletion behavior.',
37:'Same default-argument-as-configuration overstatement as18.',
42:'Supported recorded exit0; correctly limits verification.',
46:'Issue and latest event-connection grep support substance. Says event is being emitted rather than emit call present, so static evidence is overstated slightly; no service command claim.',
47:'Reported issue/fix description supported, but linked PR is not independent fix verification.',
52:'Query attributes supported; missing reproduction/environment statement is correct only for latest excerpt. Initial issue explicitly includes versions and reproducer, so whole-context summary is incomplete.',
53:'Definition/registration supported; catch-all handler is conditional/fallback, so wording handles all URLs overbroad.',
54:'Issue/grep supported; implementation details/fix evidence phrasing too strong for class headers plus linked-PR description.',
55:'Signature default described as parameter setTrue, ambiguous runtime interpretation; correctly limits behavior evidence.',
62:'Latest grep does not show Max definition. Min inheritance and lack of observed change supported.',
65:'Reasonable issue traceback summary with explicitly limited latest grep; no execution claim.',
72:'Supported registration and explicit limitation.',
73:'Same linked-fix verification caveat as54; latest class definitions do not prove a working fix.'}
seen=set();reviews=[]
for i,q in enumerate(x['selected']):
 key=(q['source_id'],q['turn'],q['generated_text_sha256'])
 if key in seen:continue
 seen.add(key);assert i in notes,i
 reviews.append({'selection_index':i,'source_id':q['source_id'],'turn':q['turn'],'text_sha256':q['generated_text_sha256'],'assessment':notes[i]})
assert len(reviews)==43
analysis=json.loads((H/'live-nov3-operational-v2-independent-analysis.json').read_text())
out={'schema':'astra-operational-v2-quality-review-v1','selection_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'selected_observations':len(x['selected']),'distinct_texts_read_against_initial_issue_and_latest_observation':len(reviews),'distinct_source_issues':len({q['source_id'] for q in x['selected']}),'reviews':reviews,
 'mechanical_all_normal_output_counts':{n:{k:r['format'][k] for k in ('responses','starts_status','code_fences','multiple_blank_blocks','degeneracy_flags')} for n,r in analysis['cases'].items() if r['attempted']},
 'conclusion':'No conspicuous output loop or service claim of executing edits/commands in selected texts. Real source-reading defects and paragraph/code-format violations remain, including shared baseline errors. Human-reviewed drafts only; not autonomous diagnosis or quality equivalence. Cache performance does not establish correctness, usefulness, or reviewer productivity. No output was excluded or used to tune this block.'}
(H/'live-nov3-operational-v2-quality-review.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'selected':len(x['selected']),'distinct_reviewed':len(reviews),'issues':out['distinct_source_issues']}))
