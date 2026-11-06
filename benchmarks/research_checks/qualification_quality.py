"""Transparent narrow task checks; never relabel these as general answer quality.

Frozen before Stage 2 and independent-family evaluations. The Stage 1 audit is
exploratory because primary measurements had already started when it was added.
"""
from __future__ import annotations
import argparse
import ast
import json
import re
from pathlib import Path
from benchmarks.serving_suite.contracts import write_json


def score(directory):
    directory=Path(directory)
    corpus=json.loads((directory/'corpus.json').read_text())
    sources={s['id']:s for s in corpus['sessions']}
    result=dict(schema='wait-qualification-quality-v1',
        scope='Observed test status, known source symbol or document metadata, word limit, and explicit coverage-contradiction flags only. Not a general reasoning/task-success grade.',cases={})
    for case_dir in sorted(directory.iterdir()):
        if not (case_dir/'requests.jsonl').is_file():continue
        details=[]
        for line in (case_dir/'requests.jsonl').read_text().splitlines():
            row=json.loads(line)
            if row.get('turn')!=3:continue
            source=sources[row['source_id']]
            text=row.get('generated_text',''); normalized=text.replace('*','').casefold()
            reference=source.get('quality_reference',{})
            checks={}
            if 'symbol' in reference:
                checks['requested_symbol_present']=bool(re.search(r'\b'+re.escape(reference['symbol'])+r'\b',text))
            if 'PEP' in reference:
                checks['pep_number_present']=bool(re.search(r'\b'+re.escape(reference['PEP'])+r'\b',text))
                for key in ('Status','Type'):
                    checks[key.casefold()+'_present']=bool(re.search(r'\b'+re.escape(reference[key])+r'\b',text,re.I))
            # These two concrete contradictions were identified against actual
            # test names; retain them as flags, not a comprehensive rubric.
            contradictions=[]
            if row['source_id']=='transport-proof' and 'do not' in normalized and any(s in normalized for s in ('mixed outcomes','empty lookup keys','repeated scheduler steps')):
                contradictions.append('Claims absent coverage explicitly present in test_mixed_request_outcomes_are_explicitly_unavailable_and_totals_are_safe / test_live_shaped_empty_lookup_keys_and_repeated_scheduler_steps_are_valid')
            if row['source_id']=='summary-overlap' and re.search(r'do not cover.{0,100}multi.worker',normalized):
                contradictions.append('Claims absent multi-worker coverage despite test_multi_worker_transfer_aggregation_keeps_time_bounds_and_status')
            claims=re.findall(r'test[_ ]status\s*[:=]\s*(pass|fail)',normalized)
            details.append(dict(source_id=row['source_id'],request_id=row['request_id'],
                requested_reference_checks=checks,all_reference_checks_passed=all(checks.values()) if checks else None,
                reported_test_status=claims,word_count=len(text.split()),within_70_words=len(text.split())<=70,
                explicit_coverage_contradiction_flags=contradictions,
                text_sha256=__import__('hashlib').sha256(text.encode()).hexdigest()))
        result['cases'][case_dir.name]=dict(finals=len(details),details=details,
            reference_checks_failed=sum(d['all_reference_checks_passed'] is False for d in details),
            word_limit_failures=sum(not d['within_70_words'] for d in details),
            explicit_contradiction_flags=sum(bool(d['explicit_coverage_contradiction_flags']) for d in details))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    write_json(a.output,score(a.directory))

if __name__=='__main__':main()
