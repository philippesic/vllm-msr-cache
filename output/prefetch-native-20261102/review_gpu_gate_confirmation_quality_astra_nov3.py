"""Record independent human source-grounding review; no model-based quality score."""
import collections,gzip,hashlib,json,re
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
NOTES={
('django__django-15380',0):('caveat','Correct KeyError/report summary, but asks for model names already supplied (MyModel to MyModel2); field names/reproducer details remain missing.'),
('django__django-15380',2):('defect','Misreads grep locations as traceback evidence:182 is the call,821 the definition,992 a comment. Repeats missing-model-names claim despite issue naming MyModel/MyModel2.'),
('sympy__sympy-22456',0):('grounded','Faithfully contrasts positional argument invariance with kwargs reconstruction. No independent validation of intended Basic contract.'),
('sympy__sympy-22456',2):('grounded','Latest observation only locates class String(Token) at ast.py:872; output correctly limits its claim and asks for implementation analysis.'),
('django__django-11333',0):('grounded','Issue explicitly describes None versus ROOT_URLCONF cache duplication and proposed pre-memoization normalization; no measured impact supplied. It repeats the issue/proposal despite status-only contract.'),
('django__django-11333',1):('caveat','Directory inventory is supported; well-organized is subjective and adds little issue-specific status. No get_resolver implementation has been observed.'),
('django__django-15973',0):('grounded','The issue includes the separate-app through-model failure, traceback, and same-app workaround. Root-cause localization remains report-level rather than executed validation.'),
('django__django-15973',2):('caveat','Latest observation is only Exit code:0 with no command identity. Candidate safely notes no new error details, though unresolved status should mean no demonstrated fix. Native incorrectly calls this a completed migration process.'),
('django__django-15629',2):('grounded','Issue SQL explicitly shows missing COLLATE on FK columns; latest grep shows field db_collation handling. Output combines those sources plausibly but must not imply new SQL/tests were executed.'),
('django__django-14122',0):('caveat','Accurately summarizes reported incomplete fix and linked PR; current unresolved status is not independently established by an old report.'),
('django__django-14122',2):('caveat','Latest grep locates GROUP BY methods, not a demonstrated fix/test. Native claim that grep includes Meta.ordering checks is unsupported; candidate also overstates that listed mechanisms fail to prevent it.'),
('sympy__sympy-17655',2):('defect','Observed duplicate Point class names do not establish a cause. It misses the important operand-order distinction and speculates unrelated classes cause confusion; traceback ends with GeometryError after internal TypeError.'),
('sympy__sympy-15599',0):('caveat','Initial evidence contains PR-style claims of modified mod.py and an added test, so it is not fabricated from nothing. Output presents reported success as current fact while admitting no executed confirmation.'),
('sympy__sympy-15599',2):('defect','Latest complete mod.py still has the old if mod_l branch and lacks the proposed was/changed changes. Output repeats successful-modification claim and misses this contrary repository evidence.'),
('sphinx-doc__sphinx-7462',2):('grounded','Issue and shown ast.Tuple unparse branch support empty-tuple pop failure. General claim other types do not fail is broader than tested evidence; no fresh build occurred.'),
('astropy__astropy-14539',0):('caveat','Self-diff failure and VLA reproducer are reported; output promotes suspected VLA handling cause into certainty. The report says author could not reproduce without VLA, not universal exclusivity.'),
('astropy__astropy-14539',2):('grounded','Grep shows FITSDiff tests/usages but no diagnostic VLA implementation. Correctly requests further analysis; FITSDDiff spelling is a minor error.'),
('pytest-dev__pytest-10081',0):('grounded','Issue directly supplies class-level skip, --pdb, tearDown NameError, versions and minimal repro. No mechanism beyond evidence asserted.'),
('pytest-dev__pytest-10081',2):('grounded','Latest grep locates unittest integration and SkipTest handling; no full teardown implementation. Output correctly describes limited evidence.'),
('sympy__sympy-19637',2):('defect','Confuses undefined local kern inside kernS with function kernS being undefined in tests. Latest grep explicitly shows kernS imported at test_sympify.py:5, refuting the proposed diagnosis.'),
('astropy__astropy-13977',2):('caveat','Latest grep mostly lists locations, not implementation. time/core line is a TODO rather than method definition. Incompatible units misstates compatible m/mm plus incompatible duck input type; initial traceback already establishes ValueError behavior.'),
('pydata__xarray-3677',0):('grounded','Issue traceback explicitly reaches obj.items in coerce_pandas_values on DataArray; contrast with top-level merge is supported.'),
('pydata__xarray-3677',2):('defect','Latest grep is locations, not a call stack. Says exact items failure location is missing even though the retained initial issue provides full traceback through merge.py:394.'),
('scikit-learn__scikit-learn-9288',0):('grounded','Reported inertia values show serial slightly lower than parallel with fixed random_state. Output correctly leaves cause unverified.'),
('scikit-learn__scikit-learn-9288',2):('grounded','Latest observation only locates KMeans class at line763; both outputs correctly limit implementation claims.'),
('sphinx-doc__sphinx-9229',0):('caveat','Correctly summarizes one-versus-two alias behavior and possible #4422 relation from issue; much of the requested configuration is already supplied, and this mostly restates issue.'),
('sphinx-doc__sphinx-9229',2):('caveat','Translations and alias-producing autodoc lines are present, but no evidence ties translations to missing docstrings. Speculative translation cause distracts from supported locations. Latest observation is explicitly truncated.'),
('django__django-11141',0):('grounded','Accurate issue-level account of obsolete __file__ guard versus pkgutil.iter_modules using __path__; not independently executed.'),
('django__django-11141',2):('caveat','Misses directly relevant migration loader __file__ guard at loader.py:89 shown in latest grep, focusing on unrelated paths and asking for __init__.py evidence. Weak responsiveness rather than a proven fix.'),
('django__django-13344',0):('caveat','Correct main coroutine/HttpResponse report, but claims evidence for correctly functioning later middleware is missing despite issue explicitly showing both outputs.'),
('django__django-13344',2):('grounded','ASGIRequest definition/handler/client references are directly shown; correctly leaves middleware coroutine mechanism unresolved.'),
('sympy__sympy-18763',0):('caveat','Textual actual/desired LaTeX formulas are supported. Image links were supplied but images were not loaded, so visual confirmation is not established. No patch proposed.'),
('sphinx-doc__sphinx-7748',2):('caveat','Grep shows True default and conditional checks, not actual runtime configuration or overloaded-signature behavior. Saying configuration is enabled overstates default evidence.'),
('django__django-12262',2):('defect','Conflates reported TemplateSyntaxError for valid keyword-only calls with a grep hit raising ValueError for decorator setup. Blaming admin_list/admin_urls usage and missing argument conditions is unsupported; issue supplies those conditions.')}
def main():
 p=H/'live-nov3-gpu-gate-confirmation-quality-samples.json';q=json.loads(p.read_text());a=json.loads((H/'live-nov3-gpu-gate-confirmation-independent-analysis.json').read_text());rows=[]
 for run in q['runs']:
  for sample in run['samples']:
   kind,note=NOTES[sample['source_id'],sample['turn']]
   if sample['source_id']=='django__django-15973' and sample['turn']==2 and run['arm']=='native':kind='defect'
   rows.append({'run':run['name'],'partition':run['partition'],'arm':run['arm'],'index':sample['index'],'source_id':sample['source_id'],'turn':sample['turn'],'selection':sample['selection'],'review':kind,'note':note,'text_sha256':hashlib.sha256(sample['generated_text'].encode()).hexdigest(),'single_paragraph_without_code_fence':bool(sample['generated_text']) and '\n' not in sample['generated_text'] and '```' not in sample['generated_text'],'self_claimed_execution_or_edit':False,'conspicuous_runaway_repetition':False})
 mechanical={};extras=[]
 for arm in ['native','candidate']:
  rs=[r for r in a['rows'].values() if r['arm']==arm]
  mechanical[arm]={k:sum(r['service_format'][k] for r in rs) for k in ['responses','starts_status','single_blank_line_block','with_code_fence']}
  mechanical[arm]['completed_action_phrase_flags']=sum(len(r['service_format']['completed_action_phrase_indices']) for r in rs)
  mechanical[arm]['degeneration_screen_flags']=sum(r['output_screen']['flagged'] for r in rs)
  for r in rs:
   if r['service_format']['with_code_fence']:
    raw=json.load(gzip.open(ROOT/r['raw'],'rt'))
    for x in raw['requests']:
     if '```' in x.get('generated_text',''):extras.append({'run':r['name'],'index':x['index'],'session_id':x['session_id'],'generated_text':x['generated_text'],'finding':'Emits a fenced function definition, contrary to no-code/one-paragraph service contract. It quotes existing code, not a claimed applied patch. Supplementary format check, outside predeclared source-grounded sample.'})
 by={(x['partition'],x['index'],x['arm']):x for x in rows};shared=[(x,by[x['partition'],x['index'],'candidate']) for x in rows if x['arm']=='native' and (x['partition'],x['index'],'candidate') in by]
 out={'schema':'astra-integrated-GPU-gate-source-grounded-quality-review-v1','sample_receipt_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'predeclared_samples_reviewed':len(rows),'shared_slot_turn_pairs':len(shared),'exact_generated_text_matches_within_shared_pairs':sum(x['text_sha256']==y['text_sha256'] for x,y in shared),'rows':rows,'mechanical_all_outputs':mechanical,'additional_code_fence_findings':extras,'summary':'All64 selected texts read against retained issue/latest observations and available actual earlier history. Many summaries are plausible but serious source-reading defects occur in both arms; no demonstrated quality equivalence. Human review required. No selected text claims the service executed commands or edited files; claims of current source state can still be unsupported.','limits':['Samples deliberately include longest responses; categories are inspection notes, not an accuracy estimate.','Mechanical single-blank-line-block does not prove one paragraph: two fenced candidate outputs violate the format contract.','Human editing cost/usefulness and autonomous task success unmeasured. Recorded observations were not executed by this service.','No source or prompt tuned on these quality outcomes.','Caching is not expected to repair baseline model reasoning; these findings limit the service claim independently of the measured cache effect.']}
 (H/'live-nov3-gpu-gate-confirmation-quality-review.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ['predeclared_samples_reviewed','shared_slot_turn_pairs','exact_generated_text_matches_within_shared_pairs','mechanical_all_outputs']},indent=2))
if __name__=='__main__':main()
