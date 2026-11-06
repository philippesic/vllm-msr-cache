"""Pinned public source preparation and fresh, auditable task construction.

No performance selection occurs here. Historical task lists define exclusions,
not timing data. Corpus partitions and all source IDs are retained explicitly.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

from .campaign_contracts import atomic_json, file_hash, load
from .contracts import validate_corpus

SOURCE_PINS = {
    'more-itertools': ('https://github.com/more-itertools/more-itertools.git', 'f42daddf205f335fea6e200dad46049f33e71221'),
    'peps': ('https://github.com/python/peps.git', 'd728748f4a99ec08e3f4c101c1cb07ba16b140f1'),
}
DOCUMENT_FRESH = [241, 279, 282, 289, 305, 307, 310, 311, 314, 315, 318, 322, 327, 328, 342, 352]
DOCUMENT_REPAIR = [353, 357, 358, 366, 370, 371, 372, 376, 386, 389, 391, 393, 405, 409, 411, 414]
SQUAD_URL = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json'
SQUAD_SHA256 = '3527663986b8295af4f7fcdff1ba1ff3f72d07d61a20f487cb238a6ef92fd955'


def prepare_datasets(cache):
    from huggingface_hub import hf_hub_download, HfApi
    import os
    import urllib.request
    cache = Path(cache)
    receipts = []
    for repository,revision,filename in (
        ('THUDM/LongBench','5e628be450b7e67fb7ae6e201bd6d8f7056f7672','data.zip'),
        ('allenai/WildChat-1M','7d6490e462285cf85d91eabea0f9a954fbddcd1f','data/train-00000-of-00014.parquet')):
        path = hf_hub_download(repository,filename,repo_type='dataset',revision=revision)
        info = HfApi().dataset_info(repository,revision=revision)
        receipts.append(dict(repository=repository,revision=revision,file=filename,path=path,
            sha256=file_hash(path),card_metadata=info.card_data.to_dict() if info.card_data else {},
            license_scope='Dataset-card declaration; upstream component terms may differ'))
    target = cache/'datasets/squad-train-v1.1.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        temporary = target.with_suffix('.download')
        with urllib.request.urlopen(SQUAD_URL,timeout=60) as response,temporary.open('xb') as output:
            while chunk:=response.read(2**20):
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if file_hash(temporary)!=SQUAD_SHA256:
            raise ValueError('Downloaded SQuAD source does not match the recorded pin')
        temporary.replace(target)
    if file_hash(target)!=SQUAD_SHA256:
        raise ValueError('Prepared SQuAD source changed')
    receipts.append(dict(url=SQUAD_URL,revision='v1.1',path=str(target.resolve()),sha256=SQUAD_SHA256,
                         license='CC-BY-SA-4.0',license_source='https://rajpurkar.github.io/SQuAD-explorer/'))
    atomic_json(cache/'datasets.json',receipts)
    return receipts


def prepare_sources(cache):
    cache = Path(cache)
    result = {}
    for name, (repository, revision) in SOURCE_PINS.items():
        target = cache / 'sources' / name
        if not target.exists():
            target.mkdir(parents=True)
            subprocess.run(['git', 'init', '-q', str(target)], check=True)
            subprocess.run(['git', '-C', str(target), 'remote', 'add', 'origin', repository], check=True)
            subprocess.run(['git', '-C', str(target), 'fetch', '--depth', '1', 'origin', revision], check=True)
            subprocess.run(['git', '-C', str(target), 'checkout', '--detach', revision], check=True)
        actual = subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != revision or subprocess.check_output(['git', '-C', str(target), 'status', '--porcelain'], text=True).strip():
            raise ValueError(f'Prepared public source changed: {name}')
        licenses = {str(p.relative_to(target)): file_hash(p) for p in target.glob('*LICENSE*') if p.is_file()}
        if not licenses:
            licenses = {str(p.relative_to(target)): file_hash(p) for p in target.glob('*COPYING*') if p.is_file()}
        result[name] = dict(repository=repository, revision=revision, path=str(target.resolve()), licenses=licenses)
    atomic_json(cache / 'sources.json', result)
    return result


def excerpt_symbol(path, node):
    lines = Path(path).read_text().splitlines(True)
    first = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])])
    return ''.join(lines[first - 1:node.end_lineno])


def repository_tasks(root, partition, python):
    from benchmarks.research_checks.qualification_inputs import REPOSITORY_TASKS, REPOSITORY_RESERVE
    root = Path(root).resolve()
    source, tests = root / 'more_itertools/more.py', root / 'tests/test_more.py'
    tracked = subprocess.check_output(['git','-C',str(root),'ls-files','-z']).decode().split('\0')
    executed_inputs = {str(root/name):file_hash(root/name) for name in filter(None,tracked)}
    functions = {n.name: n for n in ast.parse(source.read_text()).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and not n.name.startswith('_')}
    classes = {n.name: n for n in ast.parse(tests.read_text()).body if isinstance(n, ast.ClassDef)}
    old = {name for name, _ in REPOSITORY_TASKS + REPOSITORY_RESERVE}
    normalized = lambda value: re.sub('[^a-z0-9]', '', value.lower()).removeprefix('test').removesuffix('tests').removesuffix('test')
    def context_for(name, test):
        implementation = excerpt_symbol(source, functions[name])
        whole = excerpt_symbol(tests, classes[test])
        if len(implementation) + len(whole) <= 12000:
            return implementation + '\n\n' + whole
        # Large test classes stay intact on disk and are executed in full. Show
        # complete individual methods, explicitly identified as a source excerpt.
        selected = []
        for method in classes[test].body:
            if isinstance(method, ast.FunctionDef) and method.name.startswith('test'):
                text = excerpt_symbol(tests, method)
                if len(implementation) + sum(map(len, selected)) + len(text) > 11500:
                    break
                selected.append(text)
                if len(selected) == 3:
                    break
        if not selected:
            raise ValueError(f'No intact test method fits the source envelope: {name}')
        return implementation + '\n\nEXPLICIT TEST SOURCE EXCERPT; whole class executed by the test tool:\n' + '\n'.join(selected)
    candidates = []
    for name, function in sorted(functions.items()):
        matches = [test for test in classes if normalized(test) == normalized(name)]
        if name not in old and len(matches) == 1:
            # Whole symbol excerpts, never arbitrary character truncation.
            try:
                text = context_for(name, matches[0])
            except ValueError:
                continue
            if len(text) <= 12000:
                candidates.append((name, matches[0], text))
    if partition == 'development':
        selected = [(name, test, context_for(name, test)) for name, test in REPOSITORY_TASKS]
    elif partition in ('confirmation', 'repair'):
        offset = 0 if partition == 'confirmation' else 16
        selected = candidates[offset:offset + 16]
    else:
        raise ValueError('Unknown task partition')
    if len(selected) != 16:
        raise ValueError(f'Insufficient fresh repository tasks for {partition}: {len(selected)}')
    sessions = []
    for name, test, context in selected:
        sessions.append(dict(id='more-' + name, repository=str(root), context=f'PUBLIC SOURCE more_itertools.{name} AND TEST CLASS {test}\n' + context,
            input_files=executed_inputs, evidence=name,
            quality_reference=dict(symbol=name, test_class=test, source_symbol=excerpt_symbol(source, functions[name]), test_symbol=excerpt_symbol(tests, classes[test])),
            turns=[f'Investigate {name}: describe its purpose and plan a boundary-case check.',
                   f'Explain one concrete {name} behavior supported by the actual search results.',
                   f'Interpret the observed tests for {name}. Distinguish evidence from assumptions.',
                   f'Report {name}: one supported behavior, test_status=PASS or FAIL from the executed tests, and a concrete limitation.'],
            tools=[dict(kind='search', argv=['rg', '-n', '--max-count', '30', name, str(source)], valid_exit_codes=[0]),
                   dict(kind='test', argv=[python, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', f'tests/test_more.py::{test}'], valid_exit_codes=[0]),
                   dict(kind='evidence', argv=['rg', '-n', '--max-count', '35', 'assert|raise|def test_', str(tests)], valid_exit_codes=[0])]))
    return corpus(sessions, 'repository', partition, SOURCE_PINS['more-itertools'])


def document_tasks(root, partition, python, references):
    from benchmarks.research_checks.qualification_inputs import DOCUMENT_TASKS
    from benchmarks.research_checks.qualification_inputs import parse_document
    root = Path(root).resolve()
    numbers = {'development': DOCUMENT_TASKS, 'confirmation': DOCUMENT_FRESH, 'repair': DOCUMENT_REPAIR}[partition]
    refs, sessions = {}, []
    for number in numbers:
        path = root / 'peps' / f'pep-{number:04d}.rst'
        text = path.read_text()
        expected = {}
        for key in ('PEP', 'Title', 'Status', 'Type', 'Created'):
            match = re.search(r'^' + re.escape(key) + r':\s*(.+)$', text, re.M)
            if match is None:
                raise ValueError(f'Public PEP missing {key}: {number}')
            expected[key] = match.group(1)
        refs[str(number)] = expected
        # Keep the header and whole opening sections. The full original remains
        # available to the bounded real tools and the quality source audit.
        chunks = re.split(r'(?m)(?=^[^\n]+\n[=~-]{3,}\s*$)', text)
        shown = ''
        for chunk in chunks:
            if shown and len(shown) + len(chunk) > 14000:
                break
            shown += chunk
        common = [python, '-m', 'benchmarks.research_checks.qualification_inputs', 'document', '--file', str(path), '--reference', str(Path(references).resolve())]
        sessions.append(dict(id=f'pep-{number:04d}', repository=str(Path(__file__).resolve().parents[2]),
            context=f'PUBLIC PEP {number}; complete opening sections, {len(shown)}/{len(text)} characters\n' + shown,
            input_files={str(path): file_hash(path)}, evidence=f'PEP {number}', quality_reference=expected,
            turns=[f'Investigate PEP {number}: identify its purpose and plan how to verify its status and a rule.',
                   'Use the actual document search results to identify its specification and restrictions.',
                   'Interpret the actual metadata validation; distinguish proposal status from runtime support.',
                   f'Report PEP {number}: its Status and Type, one supported rule, test_status=PASS or FAIL from validation, and a concrete limitation.'],
            tools=[dict(kind='search', argv=['rg', '-n', '--max-count', '35', '^Status:|^Type:|^Title:|^Abstract|^Specification|^Rationale|^Backwards|^Security|^Reference', str(path)], valid_exit_codes=[0]),
                   dict(kind='test', argv=common + ['--action', 'validate'], valid_exit_codes=[0]),
                   dict(kind='evidence', argv=common + ['--action', 'summarize'], valid_exit_codes=[0])]))
    atomic_json(references, refs)
    for session in sessions:
        session['input_files'][str(Path(references).resolve())] = file_hash(references)
    return corpus(sessions, 'documents', partition, SOURCE_PINS['peps'])


def corpus(sessions, adapter, partition, source):
    return validate_corpus(dict(schema='semantic-serving-corpus-v1', synthetic=False,
        provenance=f'Authored {adapter} workflows over pinned public source {source[0]} at {source[1]}; {partition} partition. Real read-only tools, authored arrivals, no production/autonomous-agent claim.',
        workload_kind=adapter, partition=partition, cohort_mode='fixed', sessions=sessions))


CHAT_EXCLUDED = (r'developer mode|jailbreak|\bDAN\b|ignore (all |any )?(the |previous |prior )?instructions|'
                 r'no restrictions|uncensored|nsfw|sexually explicit|pretend (that )?you (are|have)|'
                 r'disregard.{0,30}instructions|HARDRULES|^CONSTRAINTS:')


def prepare_chat_sample(parquet, destination):
    """Keep only selected source IDs and user text, never identity/location data."""
    import pyarrow.parquet as pq
    seen, samples, scanned = set(), [], 0
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=256, columns=['conversation_hash', 'language', 'toxic', 'redacted', 'conversation']):
        for row in batch.to_pylist():
            scanned += 1
            messages = row['conversation']
            if row['language'] != 'English' or row['toxic'] or row['redacted'] or not messages or messages[0]['role'] != 'user':
                continue
            text = messages[0]['content']
            # Metadata language can refer to the whole conversation; reject
            # non-English opening text for this explicitly English subset.
            if not 3000 <= len(text) <= 8000 or len(re.findall('[\u4e00-\u9fff]', text)) > 20:
                continue
            if re.search(CHAT_EXCLUDED, text, re.I) or re.search(r'\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b|api[_ -]?key|password\s*[:=]', text, re.I):
                continue
            content_key = re.sub(r'\s+', ' ', text[:200].casefold())
            if content_key in seen:
                continue
            seen.add(content_key)
            samples.append(dict(id=row['conversation_hash'], initial_user=text))
            if len(samples) == 24:
                break
        if len(samples) == 24:
            break
    if len(samples) < 24:
        raise ValueError('Not enough eligible public chat inputs')
    result = dict(schema='stage6-public-chat-sample-v1', repository='allenai/WildChat-1M',
                  revision='7d6490e462285cf85d91eabea0f9a954fbddcd1f',
                  file='data/train-00000-of-00014.parquet', license='odc-by',
                  filter=dict(language='English', toxic=False, redacted=False, initial_chars=[3000, 8000],
                              excluded_regex=CHAT_EXCLUDED, deduplicate='normalized first 200 characters',
                              identity_metadata='omitted', email_or_secret_pattern='excluded'),
                  rows_scanned=scanned, samples=samples)
    atomic_json(destination, result)
    return result


def chat_corpus(sample, partition):
    data = load(sample)
    offset = {'development': 0, 'confirmation': 8, 'repair': 16}[partition]
    sessions = []
    for index, item in enumerate(data['samples'][offset:offset + 8]):
        sessions.append(dict(id='wildchat-' + item['id'], context='',
            turns=[item['initial_user'],
                   'Explain the main reasoning behind your response, using the same task and context.',
                   'Give one concrete example that helps clarify your response.',
                   'Identify a limitation or assumption in your response and explain its effect.',
                   'Refine your response in light of that limitation.',
                   'Summarize the most useful conclusion in a concise final answer.'],
            wait_s=[1, 5, 15, 1, 5], publish_hints=False,
            quality_reference=dict(kind='coherence_and_instruction_adherence', initial_user=item['initial_user'])))
    return validate_corpus(dict(schema='semantic-serving-corpus-v1', synthetic=False,
        provenance=f'Filtered text-only WildChat-1M first-user requests at {data["revision"]}; selected source IDs and filtering retained. Five follow-ups and 1/5/15-second pauses are authored, not production timing. No historical assistant text.',
        source_selection={key: value for key, value in data.items() if key != 'samples'},
        partition=partition, cohort_mode='fixed', sessions=sessions))


def longbench_candidates(archive, tokenizer):
    import zipfile
    candidates = []
    with zipfile.ZipFile(archive) as stream:
        for filename in ('data/multifieldqa_en.jsonl', 'data/qasper.jsonl', 'data/triviaqa.jsonl'):
            for line in stream.read(filename).decode().splitlines():
                row = json.loads(line)
                answers = [a for a in row['answers'] if isinstance(a, str) and len(a.strip()) > 2 and a.casefold() not in ('yes', 'no', 'unanswerable')]
                context = row['context']
                for answer in answers:
                    found = context.casefold().find(answer.casefold().rstrip('.'))
                    if found < 0:
                        continue
                    # Whole paragraphs around the reference. This is a declared
                    # serving adaptation, never a published LongBench score.
                    start = context.rfind('\n\n', 0, max(0, found - 400))
                    start = 0 if start < 0 else start + 2
                    end = context.find('\n\n', found + len(answer) + 400)
                    end = len(context) if end < 0 else end
                    excerpt = context[start:end]
                    count = len(tokenizer.encode(excerpt))
                    if count > 900:
                        # Some source documents have no paragraph breaks. Select
                        # complete surrounding sentences, retaining the answer.
                        boundaries = [0] + [m.end() for m in re.finditer(r'(?<=[.!?])\s+|\n', context)] + [len(context)]
                        start = max((b for b in boundaries if b <= max(0, found - 600)), default=0)
                        end = min((b for b in boundaries if b >= found + len(answer) + 600), default=len(context))
                        excerpt = context[start:end]
                        count = len(tokenizer.encode(excerpt))
                    if 128 <= count <= 1000:
                        candidates.append(dict(source_id=row['_id'], dataset=filename, context=excerpt,
                                               question=row['input'], answers=answers, source_offsets=[start, end]))
                        break
    return candidates


def longbench_corpus(archive, tokenizer, partition, kind):
    import zipfile
    revision = '5e628be450b7e67fb7ae6e201bd6d8f7056f7672'
    sessions = []
    if kind == 'qa':
        candidates = longbench_candidates(archive, tokenizer)
        offset, count = {'development': (0, 64), 'confirmation': (64, 64), 'repair': (128, 16)}[partition]
        selected = candidates[offset:offset + count]
        if len(selected) != count:
            raise ValueError(f'Insufficient adapted LongBench short QA inputs: {len(candidates)}')
        for item in selected:
            sessions.append(dict(id='longbench-' + item['source_id'], context=item['context'],
                turns=[item['question'] + '\nReturn a JSON object with one string field "answer". Use the supplied evidence.'],
                publish_hints=False, quality_reference=dict(kind='answer_json', answers=item['answers'],
                    source_id=item['source_id'], dataset=item['dataset'], source_offsets=item['source_offsets'])))
    elif kind == 'summary':
        candidates = []
        with zipfile.ZipFile(archive) as stream:
            for filename in ('data/multi_news.jsonl', 'data/gov_report.jsonl'):
                for line in stream.read(filename).decode().splitlines():
                    row = json.loads(line)
                    tokens = len(tokenizer.encode(row['context']))
                    if 3500 <= tokens <= 7000:
                        candidates.append((filename, row))
        offset = {'development': 0, 'confirmation': 16, 'repair': 32}[partition]
        for filename, row in candidates[offset:offset + 16]:
            sessions.append(dict(id='longbench-summary-' + row['_id'], context=row['context'],
                turns=['Summarize the supplied documents in at most 300 words. State key facts and uncertainties; do not introduce unsupported claims.'],
                publish_hints=False, quality_reference=dict(kind='source_grounded_summary', references=row['answers'],
                    source_id=row['_id'], dataset=filename)))
        if len(sessions) != 16:
            raise ValueError('Insufficient intact LongBench summaries in useful context envelope')
    else:
        raise ValueError('Unknown LongBench adapter')
    return validate_corpus(dict(schema='semantic-serving-corpus-v1', synthetic=False,
        provenance=f'Adapted public THUDM/LongBench {kind} inputs at {revision}; exact source IDs/references retained. Whole document or declared answer-preserving paragraph excerpt. Authored serving instructions; not a published LongBench score.',
        partition=partition, cohort_mode='fixed', sessions=sessions))


def squad_corpus(path, tokenizer, partition):
    """One intact paragraph/question per article; article-disjoint partitions."""
    import random
    if file_hash(path)!=SQUAD_SHA256:
        raise ValueError('SQuAD v1.1 source content differs from preparation pin')
    articles = list(load(path)['data'])
    random.Random(17011).shuffle(articles)
    eligible = []
    for article in articles:
        found = None
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            if not 128<=len(tokenizer.encode(context))<=1000:
                continue
            for qa in paragraph['qas']:
                if qa['answers'] and all(context[a['answer_start']:a['answer_start']+len(a['text'])]==a['text'] for a in qa['answers']):
                    found = dict(id='squad-'+qa['id'],context=context,
                        turns=[qa['question']+'\nReturn a JSON object with one string field "answer". Use the supplied evidence.'],
                        publish_hints=False,quality_reference=dict(kind='answer_json',answers=[a['text'] for a in qa['answers']],
                            source_id=qa['id'],article=article['title'],source_answers=qa['answers']))
                    break
            if found:
                break
        if found:
            eligible.append(found)
    offset = {'development':0,'confirmation':64,'repair':128}[partition]
    sessions = eligible[offset:offset+64]
    if len(sessions)!=64:
        raise ValueError('Insufficient article-disjoint short QA sources')
    return validate_corpus(dict(schema='semantic-serving-corpus-v1',synthetic=False,
        provenance=f'SQuAD v1.1 public training source {SQUAD_URL}, sha256 {SQUAD_SHA256}; CC BY-SA 4.0. Intact paragraphs, original human questions/reference spans. Authored JSON extraction instruction; article-disjoint partitions after seeded shuffle; not an official SQuAD score.',
        source=dict(url=SQUAD_URL,sha256=SQUAD_SHA256,license='CC-BY-SA-4.0',selection_seed=17011,
                    paragraph_tokens=[128,1000],one_question_per_article=True),
        partition=partition,cohort_mode='fixed',sessions=sessions))


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    fetch = sub.add_parser('prepare')
    fetch.add_argument('--cache',type=Path,required=True)
    sample = sub.add_parser('chat-sample')
    sample.add_argument('--parquet', type=Path, required=True)
    sample.add_argument('--output', type=Path, required=True)
    build = sub.add_parser('build')
    build.add_argument('--cache', type=Path, required=True)
    build.add_argument('--python', required=True, help='Pinned read-only tool interpreter')
    build.add_argument('--model', required=True, help='Prepared local tokenizer for source selection')
    build.add_argument('--archive', type=Path, required=True, help='Pinned LongBench data.zip')
    build.add_argument('--chat-sample', type=Path, required=True)
    build.add_argument('--qa-source', type=Path, required=True, help='Pinned official SQuAD v1.1 training JSON')
    args = p.parse_args()
    if args.command == 'prepare':
        result = dict(sources=prepare_sources(args.cache),datasets=prepare_datasets(args.cache))
        print(json.dumps(result,indent=2))
    elif args.command == 'chat-sample':
        result = prepare_chat_sample(args.parquet, args.output)
        print(json.dumps(dict(samples=len(result['samples']), rows_scanned=result['rows_scanned'])))
    else:
        from transformers import AutoTokenizer
        sources = load(args.cache/'sources.json')
        tokenizer = AutoTokenizer.from_pretrained(args.model,local_files_only=True)
        for partition in ('development','confirmation','repair'):
            target = args.cache/'corpora'/partition
            atomic_json(target/'repository.json', repository_tasks(sources['more-itertools']['path'],partition,args.python))
            atomic_json(target/'documents.json', document_tasks(sources['peps']['path'],partition,args.python,target/'document-references.json'))
            atomic_json(target/'chat.json', chat_corpus(args.chat_sample,partition))
            atomic_json(target/'qa.json',squad_corpus(args.qa_source,tokenizer,partition))
            atomic_json(target/'summary.json',longbench_corpus(args.archive,tokenizer,partition,'summary'))
        print(json.dumps(dict(partitions=['development','confirmation','repair'], cache=str(args.cache))))


if __name__ == '__main__':
    main()
