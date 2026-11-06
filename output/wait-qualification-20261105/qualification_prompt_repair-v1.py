"""Prepare an explicit final-answer quality calibration on already-seen tasks.

This changes application instructions for every condition, never serving policy.
Calibration timing is not qualification. Fresh reserve validation is required if
the revised instruction is adopted after examining evaluation answers.
"""
from __future__ import annotations
import argparse
import copy
import json
import hashlib
import time
from pathlib import Path
from benchmarks.serving_suite.contracts import digest, write_json


def final_instruction(symbol):
    return (
        f'FINAL ANSWER for {symbol}. No more tools will run. In at most 70 words, '
        'write exactly four labeled lines: component; behavior; test_status; limitation. '
        'Name the requested component. Give one concrete behavior directly supported '
        'by the supplied source or actual test results. Write test_status=PASS or '
        'test_status=FAIL, selecting the actual observed test outcome. For limitation, '
        'give one documented input restriction, resource cost, or precision boundary '
        'visible in the source; do not invent a defect or assert missing test coverage. '
        'If no limitation is established, state that explicitly. Include no commands '
        'or code fences.'
    )


def prepare_calibration(base):
    original=json.loads((base/'stage3-repository-spec.json').read_text())
    spec=copy.deepcopy(original)
    chosen={'itertools-replace','itertools-ichunked','itertools-seekable','itertools-bucket'}
    spec['corpus']['sessions']=[s for s in spec['corpus']['sessions'] if s['id'] in chosen]
    if len(spec['corpus']['sessions'])!=4:
        raise ValueError('Missing declared development calibration task')
    for session in spec['corpus']['sessions']:
        session['turns'][-1]=final_instruction(session['quality_reference']['symbol'])
    spec['corpus']['qualification_amendment']='Final-answer instruction calibration after source audits found unsupported limitations. Original evaluation becomes development evidence for this amendment. Public sources/tools unchanged; reserve tasks remain unseen.'
    corpus_path=base/'repository-quality-calibration-corpus.json'
    spec['plan'].update(sessions=4,repeats=1,waves=1,corpus_sha256=digest(spec['corpus']),
        task_file=str(corpus_path.resolve()),
        scope_note='Four already-seen repository tasks, one ARC case, quality calibration only; never count its timing as qualification or select serving settings from it.')
    spec['conditions']={'arc':spec['conditions']['arc']}
    spec['order']=[[0,'arc']]
    destination=base/'stage3-quality-calibration-spec.json'
    if destination.exists() or corpus_path.exists():
        raise FileExistsError('Retain existing calibration inputs; do not overwrite')
    write_json(corpus_path,spec['corpus']);write_json(destination,spec)
    return destination


def prepare_reserve(base):
    """Adopt the calibrated application instruction, before reserve execution."""
    original=json.loads((base/'stage3-repository-cohost-spec.json').read_text())
    spec=copy.deepcopy(original)
    spec['corpus']=json.loads((base/'independent-inputs/repository-reserve.json').read_text())
    for session in spec['corpus']['sessions']:
        session['turns'][-1]=final_instruction(session['quality_reference']['symbol'])
    spec['corpus']['qualification_amendment']='Explicit final-answer instruction calibrated only on four original evaluation tasks. Original evaluation is development for this instruction. Sixteen previously unused reserve tasks; serving policy and predeclared half-byte deployment geometry unchanged.'
    corpus_path=base/'repository-reserve-revised-corpus.json'
    spec['plan'].update(task_file=str(corpus_path.resolve()),corpus_sha256=digest(spec['corpus']),
        scope_note='Fresh reserve validation of revised final-answer instruction; same instruction for every condition, fixed three rotating repetitions, same half-byte deployment as original cohost cell. No pooling with development or calibration.')
    destination=base/'stage3-repository-reserve-spec.json'
    receipt=base/'stage3-reserve-premeasurement-manifest.json'
    if any(p.exists() for p in (destination,corpus_path,receipt)):
        raise FileExistsError('Retain existing reserve inputs; do not overwrite')
    if len(spec['corpus']['sessions'])!=16 or len(spec['order'])!=9:
        raise ValueError('Unexpected reserve validation size')
    write_json(corpus_path,spec['corpus']);write_json(destination,spec)
    helper=Path(__file__)
    snapshot=base/'qualification_prompt_repair-frozen.py'
    snapshot.write_bytes(helper.read_bytes())
    write_json(receipt,dict(prepared_before_reserve_evaluation_unix=time.time(),
        application_instruction_only=True,calibration='stage3-quality-calibration',
        source_evaluation_role='development for revised instruction; all original outcomes retained',
        sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (destination,corpus_path,snapshot)}))
    return destination


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base',type=Path)
    parser.add_argument('--reserve',action='store_true',help='Adopt only after source-auditing the seen-task calibration')
    args=parser.parse_args()
    print((prepare_reserve if args.reserve else prepare_calibration)(args.base))


if __name__=='__main__':main()
