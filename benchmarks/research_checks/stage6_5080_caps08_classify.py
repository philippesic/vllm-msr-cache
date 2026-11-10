"""Find unread contexts in partial all-attempt cap snapshot08; never rate them."""
from collections import Counter
from pathlib import Path
import sys

repo = Path(__file__).resolve().parents[2]
source = repo / 'output/stage6-5080-20261108/primary-execution-bundle-01/source'
assert Path.cwd() == source.resolve() and sys.dont_write_bytecode
assert sys.executable == '/home/philip/.cache/semantic-offload/stage6-5080-runtimes-01/local-env/bin/python'
sys.path[:] = [str(source)] + [v for v in sys.path if Path(v).resolve() != Path(__file__).resolve().parent]
from benchmarks.serving_suite.campaign_contracts import load, digest, atomic_json

b = repo / 'output/stage6-5080-20261109/caps-all-checkpoint-08'
e = repo / 'evidence/stage6-5080-20261109'
prior = load(e / 'caps-all-review-index-07.json')
extended, fields = prior['signature_fields'], prior['quality_signature_fields']
old_packet = load(Path(prior['raw_packet']))
assert digest(old_packet['items']) == prior['packet_digest']
old_bindings = load(e / 'caps-instruction-bindings-07.json')
known = set()
for stem, no in [('screen', '07')] + [('confirm', f'{i:02}') for i in range(1, 18)]:
    index = load(e / f'{stem}-review-index-{no}.json')
    assert index['signature_fields'] == fields
    assert index['executed_system_instruction_digest'] == old_bindings['executed_system_instruction_digest']
    assert index['executed_instruction_source_sha256'] == old_bindings['executed_instruction_source_sha256']
    known.update(c['signature'] for c in index['contexts'])
known.update(digest({k:x[k] for k in fields}) for x in old_packet['items'])
old = Counter(digest({k:x[k] for k in extended}) for x in old_packet['items'])
packet = load(b / 'blind-packet.json')
summary = load(b / 'summary.json')
assert digest(packet['items']) == summary['packet_digest']
current = Counter(digest({k:x[k] for k in extended}) for x in packet['items'])
assert all(current[sig] >= count for sig, count in old.items()), 'Earlier sealed caps must remain included'
novels, seen, reused, new_items = [], set(), 0, 0
for x in packet['items']:
    ext = digest({k:x[k] for k in extended})
    q = digest({k:x[k] for k in fields})
    if ext in old:
        reused += 1
        continue
    new_items += 1
    if q not in known and q not in seen:
        novels.append(x)
        seen.add(q)
target = b / 'novel-quality-contexts.json'
assert not target.exists()
atomic_json(target, novels)
print(dict(capped=summary['capped_responses'], old_extended_reuse=reused,
    new_extended_occurrences=new_items, novel_full_quality_contexts=len(novels),
    all_previous418_capped_occurrences_preserved=True,
    reuse_union='screen07, confirm01-through17 and completed cap07 with identical archived instruction identity',
    scope='Classification only; every novel answer requires explicit review before ratings. Partial live-acquisition inventory.'))
