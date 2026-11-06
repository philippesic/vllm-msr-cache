"""Frozen independent task families and real, read-only document-analysis tools.

Corpus preparation is independent of serving. All benchmark tools execute from
this versioned module, and all public source inputs are retained with identities.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from email.parser import Parser
from pathlib import Path

from benchmarks.serving_suite.contracts import write_json

REPOSITORY_TASKS = [
 ('chunked','ChunkedTests'), ('peekable','PeekableTests'),
 ('distinct_permutations','DistinctPermutationsTests'), ('windowed','WindowedTests'),
 ('bucket','BucketTests'), ('collapse','TestCollapse'),
 ('split_at','SplitAtTests'), ('split_when','SplitWhenTests'),
 ('sort_together','SortTogetherTest'), ('numeric_range','NumericRangeTests'),
 ('seekable','SeekableTest'), ('map_reduce','MapReduceTests'),
 ('replace','ReplaceTests'), ('partitions','PartitionsTest'),
 ('set_partitions','SetPartitionsTests'), ('ichunked','IchunkedTests'),
]
REPOSITORY_RESERVE = [
 ('derangements','DerangementsTests'), ('substrings','SubstringsTests'),
 ('spy','SpyTests'), ('interleave_evenly','InterleaveEvenlyTests'),
 ('split_before','SplitBeforeTest'), ('split_after','SplitAfterTest'),
 ('split_into','SplitIntoTests'), ('distribute','DistributeTest'),
 ('stagger','StaggerTest'), ('zip_offset','ZipOffsetTest'),
 ('divide','DivideTest'), ('islice_extended','IsliceExtendedTests'),
 ('consecutive_groups','ConsecutiveGroupsTest'), ('run_length','RunLengthTest'),
 ('distinct_combinations','DistinctCombinationsTests'), ('zip_broadcast','ZipBroadcastTests'),
]
DOCUMENT_TASKS = [8,20,257,343,380,420,440,451,484,492,498,508,517,518,526,567]
DOCUMENT_RESERVE = [570,572,585,593,604,612,615,634,635,636,654,655,668,680,695,701]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_revision(root):
    return subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()


def ast_excerpt(path,name,limit):
    text=Path(path).read_text(); lines=text.splitlines(True)
    node=next(n for n in ast.parse(text).body if getattr(n,'name',None)==name)
    excerpt=''.join(lines[node.lineno-1:node.end_lineno])
    return f'FILE {Path(path).name}, symbol {name}, lines {node.lineno}-{node.end_lineno}; excerpt {min(limit,len(excerpt))}/{len(excerpt)} chars\n'+excerpt[:limit]


def repository_corpus(root,*,reserve=False):
    root=Path(root).resolve()
    tasks=REPOSITORY_RESERVE if reserve else REPOSITORY_TASKS
    source=root/'more_itertools/more.py'; tests=root/'tests/test_more.py'
    sessions=[]
    for symbol,test in tasks:
        sessions.append(dict(id='itertools-'+symbol,repository=str(root),
            context=ast_excerpt(source,symbol,6000)+'\n\n'+ast_excerpt(tests,test,6000),
            input_files={str(source):sha(source),str(tests):sha(tests)},
            evidence=symbol,
            quality_reference=dict(symbol=symbol,test_class=test),
            turns=[f'Investigate the behavior and boundary cases of more_itertools.{symbol}. Give a brief plan based on the supplied implementation and tests.',
                   f'Identify one concrete behavior of {symbol} supported by the actual source search.',
                   f'Interpret the actual tests for {symbol}; distinguish what passed from behavior not covered.',
                   f'Give a final report naming {symbol}, one supported boundary behavior, test_status=PASS or FAIL from the observed tests, and a concrete limitation.'],
            tools=[dict(kind='search',argv=['rg','-n','--max-count','30',symbol,str(source)],valid_exit_codes=[0]),
                   dict(kind='test',argv=[sys.executable,'-m','pytest','-q','-p','no:cacheprovider',f'tests/test_more.py::{test}'],valid_exit_codes=[0]),
                   dict(kind='evidence',argv=['rg','-n','--max-count','35','assert|raise|def test_',str(tests)],valid_exit_codes=[0])]))
    return dict(schema='semantic-serving-corpus-v1',synthetic=False,
        workload_kind='live_external_repository_investigation',sessions=sessions,
        provenance=f'Authored read-only source/test investigations of public more-itertools, revision {git_revision(root)}. Distinct function/test tasks; no patch-writing or production-traffic claim. '+('Reserved fresh validation tasks.' if reserve else 'Held-out evaluation tasks; not used for candidate tuning.'))


def parse_document(path):
    text=Path(path).read_text()
    headers=Parser().parsestr(text.split('\n\n',1)[0])
    result={k:headers.get(k) for k in ('PEP','Title','Status','Type','Created','Python-Version')}
    result.update(characters=len(text),words=len(text.split()),
                  headings=re.findall(r'^([^\n]+)\n[=~-]{3,}\s*$',text,re.M))
    return result


def document_corpus(root,references,*,reserve=False):
    root=Path(root).resolve();references=Path(references).resolve()
    sessions=[];manifest={}
    for number in DOCUMENT_RESERVE if reserve else DOCUMENT_TASKS:
        path=root/'peps'/f'pep-{number:04d}.rst';text=path.read_text()
        # Independent simple header extraction forms the reference; the executed
        # tool uses email.Parser. Both derive from actual frozen public documents.
        expected={k:re.search(r'^'+re.escape(k)+r':\s*(.+)$',text,re.M).group(1)
                  for k in ('PEP','Title','Status','Type','Created')}
        manifest[str(number)]=expected
        common=[sys.executable,'-m','benchmarks.research_checks.qualification_inputs','document',
                '--file',str(path),'--reference',str(references)]
        sessions.append(dict(id=f'pep-{number:04d}',repository=str(Path.cwd()),
            context=f'ACTUAL PUBLIC DOCUMENT {path.name}; first {min(len(text),12000)}/{len(text)} characters\n'+text[:12000],
            input_files={str(path):sha(path)},evidence=f'PEP {number}',
            quality_reference=expected,
            turns=[f'Analyze PEP {number}: its status, type, purpose and one restriction. Start with an evidence-gathering plan.',
                   'Use the actual search output to identify the proposal and relevant specification sections.',
                   'Interpret the metadata validation and actual parser output. Distinguish document status from implementation support.',
                   f'Give the final report for PEP {number}: explicitly state its Status and Type, summarize one source-supported rule and limitation, and report test_status=PASS or FAIL from the actual validation.'],
            tools=[dict(kind='search',argv=['rg','-n','--max-count','35','^Status:|^Type:|^Title:|^Abstract|^Specification|^Rationale|^Backwards|^Security|^Reference',str(path)],valid_exit_codes=[0]),
                   dict(kind='test',argv=common+['--action','validate'],valid_exit_codes=[0]),
                   dict(kind='evidence',argv=common+['--action','summarize'],valid_exit_codes=[0])]))
    write_json(references,manifest)
    return dict(schema='semantic-serving-corpus-v1',synthetic=False,
        workload_kind='live_document_analysis',sessions=sessions,
        provenance=f'Authored metadata/specification analysis of actual public Python PEPs, revision {git_revision(root)}. Real search, parser and reference checks; no artificial delay or production-traffic claim. '+('Reserved fresh validation tasks.' if reserve else 'Held-out evaluation tasks; not used for candidate tuning.'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    tool=sub.add_parser('document');tool.add_argument('--file',type=Path,required=True)
    tool.add_argument('--reference',type=Path,required=True)
    tool.add_argument('--action',choices=['validate','summarize'],required=True)
    build=sub.add_parser('build');build.add_argument('--repository',type=Path,required=True)
    build.add_argument('--documents',type=Path,required=True)
    build.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='document':
        actual=parse_document(args.file)
        expected=json.loads(args.reference.read_text())[actual['PEP']]
        if args.action=='validate':
            if any(actual[k]!=v for k,v in expected.items()):
                raise ValueError('Actual document differs from frozen reference')
            print(json.dumps(dict(test_status='PASS',validated_fields=expected,source=args.file.name)))
        else:
            print(json.dumps(actual))
    else:
        args.output.mkdir(parents=True,exist_ok=False)
        for reserve in (False,True):
            suffix='reserve' if reserve else 'evaluation'
            write_json(args.output/f'repository-{suffix}.json',repository_corpus(args.repository,reserve=reserve))
            write_json(args.output/f'documents-{suffix}.json',document_corpus(args.documents,args.output/f'document-references-{suffix}.json',reserve=reserve))

if __name__=='__main__':
    main()
