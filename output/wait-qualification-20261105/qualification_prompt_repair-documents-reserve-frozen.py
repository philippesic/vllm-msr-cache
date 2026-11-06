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
        'fill in this exact four-line template, replacing the bracketed fields:\n'
        f'component: {symbol}\n'
        'behavior: [one concrete behavior directly supported by the source or tests]\n'
        'test_status: [PASS or FAIL, selecting the actual observed outcome]\n'
        'limitation: [one documented input restriction, resource cost, or precision boundary]\n'
        'Use the supplied implementation/docstring for the limitation. Do not invent '
        'a defect or assert missing test coverage. A statement that there is no '
        'explicit input limit is not a concrete limitation. If none is established, '
        'write "not established from supplied evidence". Include no additional '
        'headings, tables, commands, or code fences.'
    )


def prepare_calibration(base,label='quality-calibration'):
    original=json.loads((base/'stage3-repository-spec.json').read_text())
    spec=copy.deepcopy(original)
    chosen={'itertools-replace','itertools-ichunked','itertools-seekable','itertools-bucket'}
    spec['corpus']['sessions']=[s for s in spec['corpus']['sessions'] if s['id'] in chosen]
    if len(spec['corpus']['sessions'])!=4:
        raise ValueError('Missing declared development calibration task')
    for session in spec['corpus']['sessions']:
        session['turns'][-1]=final_instruction(session['quality_reference']['symbol'])
    spec['corpus']['qualification_amendment']='Final-answer instruction calibration after source audits found unsupported limitations. Original evaluation becomes development evidence for this amendment. Public sources/tools unchanged; reserve tasks remain unseen.'
    corpus_path=base/f'repository-{label}-corpus.json'
    spec['plan'].update(sessions=4,repeats=1,waves=1,corpus_sha256=digest(spec['corpus']),
        task_file=str(corpus_path.resolve()),
        scope_note='Four already-seen repository tasks, one ARC case, quality calibration only; never count its timing as qualification or select serving settings from it.')
    spec['conditions']={'arc':spec['conditions']['arc']}
    spec['order']=[[0,'arc']]
    destination=base/f'stage3-{label}-spec.json'
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
        application_instruction_only=True,calibration=['stage3-quality-calibration','stage3-quality-calibration-v2'],
        source_evaluation_role='development for revised instruction; all original outcomes retained',
        sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (destination,corpus_path,snapshot)}))
    return destination


def final_document_instruction(number):
    return (
        f'FINAL ANSWER for PEP {number}. No more tools will run. In at most 70 words, '
        'fill in exactly these six lines, replacing the bracketed fields:\n'
        f'PEP: {number}\n'
        'Status: [actual document Status]\n'
        'Type: [actual document Type]\n'
        'rule: [one concrete rule directly supported by the supplied document]\n'
        'test_status: [PASS or FAIL from the actual metadata validation]\n'
        'limitation: [one restriction established by the source or actual checks]\n'
        'Do not infer implementation support from document status or invent missing '
        'compatibility information. If no document restriction is established, state '
        'that the actual checks validate metadata, not implementation correctness. '
        'Include no additional headings, tables, commands or code fences.'
    )


def prepare_documents(base,*,reserve=False):
    """One seen-task calibration, then untouched document reserve validation."""
    label='documents-reserve' if reserve else 'documents-quality-calibration'
    source=base/('stage3-documents-cohost-spec.json' if reserve else 'stage3-documents-spec.json')
    spec=copy.deepcopy(json.loads(source.read_text()))
    if reserve:
        spec['corpus']=json.loads((base/'independent-inputs/documents-reserve.json').read_text())
    else:
        chosen={'pep-0451','pep-0492','pep-0498','pep-0526'}
        spec['corpus']['sessions']=[s for s in spec['corpus']['sessions'] if s['id'] in chosen]
        if len(spec['corpus']['sessions'])!=4:raise ValueError('Missing declared seen PEP calibration task')
        spec['conditions']={'arc':spec['conditions']['arc']};spec['order']=[[0,'arc']]
        spec['plan'].update(sessions=4,repeats=1,waves=1)
    for session in spec['corpus']['sessions']:
        session['turns'][-1]=final_document_instruction(session['quality_reference']['PEP'])
    spec['corpus']['qualification_amendment']='Literal final-report template after original first-ARC source audit found incorrect/unsupported limitations and excessive command repetition. Serving policy unchanged. Original document evaluation is development for this application instruction; reserve outcomes have not selected settings.'
    corpus_path=base/f'{label}-corpus.json';destination=base/f'stage3-{label}-spec.json'
    receipt=base/f'stage3-{label}-premeasurement-manifest.json'
    if any(p.exists() for p in (corpus_path,destination,receipt)):
        raise FileExistsError('Preserve existing document instruction attempt')
    spec['plan'].update(task_file=str(corpus_path.resolve()),corpus_sha256=digest(spec['corpus']),
        scope_note='Fresh sixteen-task PEP reserve, three rotating matched half-byte cases per system; no pooling with original evaluation or calibration.' if reserve else 'Four already-seen PEP tasks, one ARC quality calibration; no timing qualification or further calibration extension.')
    write_json(corpus_path,spec['corpus']);write_json(destination,spec)
    snapshot=base/f'qualification_prompt_repair-{label}-frozen.py';snapshot.write_bytes(Path(__file__).read_bytes())
    write_json(receipt,dict(prepared_before_evaluation_unix=time.time(),application_instruction_only=True,
        sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (destination,corpus_path,snapshot)}))
    return destination


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base',type=Path)
    parser.add_argument('--reserve',action='store_true',help='Adopt only after source-auditing the seen-task calibration')
    parser.add_argument('--label',default='quality-calibration')
    parser.add_argument('--documents',action='store_true')
    args=parser.parse_args()
    print(prepare_documents(args.base,reserve=args.reserve) if args.documents else
          prepare_reserve(args.base) if args.reserve else prepare_calibration(args.base,args.label))


if __name__=='__main__':main()
