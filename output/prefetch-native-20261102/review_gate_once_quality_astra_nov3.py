"""Source-grounded inspection of145 observations; exact prior review reuse is explicit."""
import collections,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent
NOTES={
('django__django-10914',2):'Latest grep establishes defaultNone and setting accessor only. Responses overstate umask as the sole determinant and infer storage permission-application logic not shown; initial issue specifically describes tempfile/rename0600 behavior.',
('django__django-14089',0):'Accurately restates requested reversed support and proposed __reversed__ method; absence of implementation is a requested feature rather than missing reproduction evidence.',
('django__django-14089',2):'Grep fails on missing conda paths outside the repository with exit−1. Output describes those errors, but extrapolates to an incomplete project environment or inability to execute commands without evidence. No OrderedSet implementation was obtained.',
('sphinx-doc__sphinx-9281',0):'Reported Enum repr versus qualified-name rendering is supported. Calling it just the underlying integer is imprecise: actual repr includes both Enum name and integer.',
('sphinx-doc__sphinx-9281',2):'Material unsupported execution-state claim: latest observation is Exit code0 plus None with no command identity, not proof a documentation build succeeded. Repeats initial Enum report.',
('sympy__sympy-19346',0):'Issue demonstrates Symbol representations for list/tuple versus bare symbols for set/dict. Response reverses the dict/set output order when saying respectively; uncertainty about intended contract is reasonable.',
('sympy__sympy-19346',2):'Grep only locates srepr at repr.py:343; no implementation shown. Saying failure evidence is missing ignores explicit retained initial examples.',
('django__django-13128',0):'Reported temporal subtraction FieldError is supported. Root cause is not demonstrated; output does not distinguish output_field on Value from the combined expression and mostly restates exception guidance.',
('django__django-13128',2):'Latest grep locates error string; it does not supply conversion implementation. Reported mixed-type failure is grounded in initial issue, not a new executed reproduction.',
('sphinx-doc__sphinx-8269',0):'HTTP-error-masked-by-anchor report is supported. Claim that expected output is missing is directly contradicted by explicit expected404 output in issue.',
('sphinx-doc__sphinx-8269',2):'Repeats the false missing-expected-output claim despite retained explicit404 example. Latest observation is None and supplies no new execution evidence.',
('django__django-11138',0):'Core hardcodedUTC conversion problem is supported by SQL/report. Response conflates database and application destination timezones; example sets both toParis, whereas reported fix is database source timezone to application destination.',
('django__django-16454',0):'Issue explicitly documents missing CommandParser subparser formatting arguments and contrasting usage/traceback. Suggested argument propagation is reported, not tested by this service.',
('django__django-16454',2):'Latest grep only locates CommandParser; correct about absent new implementation evidence. Broad statement that subparser issue evidence is missing overlooks initial full repro and traceback.',
('django__django-13346',0):'Reported key-transform __in returning0 versus direct-value312 is accurately summarized. Cross-backend scope comes from issue title, not independent tests.',
('django__django-13346',2):'Latest observation only lists JSONField class definitions; output appropriately distinguishes it from behavioral tests but mostly repeats issue.',
('sympy__sympy-17655',0):'Reports internal TypeError and coordinate mismatch but misses scalar-left versus scalar-right distinction and final GeometryError. No actual new operation was executed.',
('sympy__sympy-17655',2):'Multiple Point classes in grep do not establish a cause; speculation about class confusion is unsupported. Important operand-order distinction remains unaddressed.',
('django__django-15161',0):'Initial issue says F simplification already changed in PR14047 and asks same technique for other expressions. Response broadly captures this, although phrasing need/change can imply unfinishedF work.',
('django__django-15161',2):'Latest grep shows other deconstruct methods, but output frames already-completedF change as merely proposed/unverified and overlooks issue request to simplify other expressions. Oncemode observation of no expressions.py entry is true but not a diagnosis.',
('pydata__xarray-3305',0):'Explicit MCVE and OrderedDict actual/expected output support failure to retain units attribute. Versions are supplied; no current test executed.',
('pydata__xarray-3305',2):'Grep only lists quantile definitions; correct limitation. Claims versions and MCVE are needed even though both are retained in initial issue.',
('django__django-15127',0):'Faithful account of reported stale LEVEL_TAGS under override_settings and empty level_tag.',
('django__django-15127',2):'Latest grep shows module-level get_level_tags assignment and later dictionary lookup, consistent with reported mechanism. No update or runtime test observed; output correctly stops short of a validated fix.',
('sphinx-doc__sphinx-7462',0):'Issue explicitly shows emptyTuple IndexError atresult.pop. Broader statement that other types never fail is not established by this single report.',
('sympy__sympy-16597',0):'Reports actualNone and author expectation that even implies finite, while keeping the intended symbolic-system rule uncertain.',
('sympy__sympy-16597',2):'Latest None output has no command identity. Initial report supports m.is_finite None, but latest observation alone does not establish a new test; output fails to separate the sources.',
('django__django-15104',0):'KeyError and customFK explanation are supported. Proposed fix mislocalizes handling to deconstruction method rather than autodetector deletion shown in traceback/proposed pop change. It also proposes a fix despite status-only no-patch direction.',
('django__django-15104',2):'Latest observation is None, so directory content comes from prior observation, not latest. Continues incorrect focus on custom-field deconstruction changes as fix evidence; no fix tested.',
('pytest-dev__pytest-7205',0):'Initial trace supports implicit str conversion of bytes under -bb --setup-show and suggests saferepr. Output omits important -bb trigger and moves toward proposed remedy rather than only status.',
('pytest-dev__pytest-7205',2):'Latest grep locates _show_fixture_action calls/definition; actual BytesWarning is from initial retained trace. No current handling/fix result shown.',
('django__django-15503',0):'SQLite numeric-key failure and test are supported. Specific MySQL/Oracle reproductions are absent although title asserts wider impact; asking for those details is reasonable.',
('django__django-15503',2):'Grep locates JSONField lookup names alongside unrelated cache/session has_key methods; it does not independently show SQLite behavior. Output combines report with locations without claiming an executed fix.'}
def key(x):return (x['source_id'],x['turn'],x['generated_text'],json.dumps(x['source_evidence_turns'],sort_keys=True))
def main():
 p=H/'live-nov3-gate-once-quality-samples.json';q=json.loads(p.read_text());old={}
 for prefix in ['gpu-gate-confirmation','triage-confirmation']:
  sp=H/f'live-nov3-{prefix}-quality-samples.json';rp=H/f'live-nov3-{prefix}-quality-review.json';samples=json.loads(sp.read_text());review=json.loads(rp.read_text());lookup={(r['run'],r['index']):r for r in review['rows']}
  for run in samples['runs']:
   for x in run['samples']:
    prior=lookup.get((run['name'],x['index']))
    if prior:old[key(x)]={'review_receipt':str(rp),'review_receipt_sha256':hashlib.sha256(rp.read_bytes()).hexdigest(),'run':run['name'],'index':x['index'],'note':prior.get('note',prior.get('finding'))}
 rows=[];unique=set()
 for run in q['runs']:
  for x in run.get('samples',[]):
   unique.add(key(x));prior=old.get(key(x));note=prior['note'] if prior else NOTES[x['source_id'],x['turn']]
   row={'run':run['name'],'partition':run['partition'],'arm':run['arm'],'index':x['index'],'source_id':x['source_id'],'turn':x['turn'],'selection':x['selection'],'format_flags':x['format_flags'],'text_sha256':hashlib.sha256(x['generated_text'].encode()).hexdigest(),'source_evidence_sha256':hashlib.sha256(json.dumps(x['source_evidence_turns'],sort_keys=True).encode()).hexdigest(),'finding':note,'exact_text_and_evidence_prior_review_reuse':prior,'self_claimed_service_execution_or_edit_observed':False,'runaway_repetition_observed':False}
   if 'code_fence' in x['format_flags']:row['format_finding']='Fenced function definition and multiple paragraphs violate no-code/one-paragraph contract; quoting existing source is not a claimed applied patch.'
   rows.append(row)
 a=json.loads((H/'live-nov3-gate-once-independent-analysis.json').read_text());mechanical={}
 for arm in ['native','periodic','once_positive']:
  rs=[r for r in a['rows'].values() if r['arm']==arm];mechanical[arm]={k:sum(r['service_format'][k] for r in rs) for k in ['responses','starts_status','single_blank_line_block','with_code_fence']};mechanical[arm]['completed_action_phrase_flags']=sum(len(r['service_format']['completed_action_phrase_indices']) for r in rs);mechanical[arm]['degeneration_screen_flags']=sum(r['output_screen']['flagged'] for r in rs)
 out={'schema':'astra-gated-once-source-grounded-quality-review-v1','sample_receipt_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'reviewed_observations':len(rows),'distinct_source_turn_text_evidence_cases':len(unique),'exact_prior_review_reuses':sum(x['exact_text_and_evidence_prior_review_reuse'] is not None for x in rows),'rows':rows,'mechanical_all_outputs':mechanical,'assessment':'No runaway response or self-claimed service action in selected texts, but repeated material source-reading defects across all arms; once mode adds no demonstrated quality benefit. One once-arm fenced-code output violates the service contract. No accuracy/equivalence/task-success or human-productivity claim.','limits':['All selected generated texts were read;46 exact text+retained-evidence matches explicitly reuse prior grounded notes, others reviewed against source issue/latest observations.','Directory listing boilerplate may be omitted from reviewer display; full evidence and actual earlier generated histories remain preserved.','Missing-evidence claims often ignore earlier retained issue text; report-level assertions are not fresh tests.','Mechanical blank-line count is not proof of one paragraph; all once-arm format flags additionally inspected.','Quality errors are not evidence of KV corruption and are not expected to be fixed by caching. No output-based exclusions or tuning.']}
 (H/'live-nov3-gate-once-quality-review.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ['reviewed_observations','distinct_source_turn_text_evidence_cases','exact_prior_review_reuses','mechanical_all_outputs']},indent=2))
if __name__=='__main__':main()
