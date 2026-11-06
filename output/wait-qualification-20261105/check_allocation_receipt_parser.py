"""False-positive prevention using synthetic startup records, not GPU evidence."""
import copy, json, sys, tempfile
from pathlib import Path

b = Path(__file__).resolve().parent
sys.path.insert(0, str(b.parents[1]))
from benchmarks.research_checks.qualification_allocation_receipts import check_observation, report, PREFIX, SCHEMA

good = dict(schema=SCHEMA, available=True, kv_bytes_per_chunk=32,
    requested_cpu_bytes=135, num_cpu_blocks=4, blocks_per_chunk=1,
    cpu_buffers=[dict(device='cpu', pinned=True, nbytes=64,
        load_uses_same_storage=True, shape=[4,16], dtype='torch.int8') for _ in range(2)])
checks = []
def check(name, value, expected):
    actual = check_observation(value)['passed']
    assert actual is expected, (name, value)
    checks.append(dict(check=name, passed=True, expected_path=expected))
check('valid rounded byte budget and pinned shared CPU buffers', good, True)
for name, field, value in [('unpinned', 'pinned', False), ('GPU is not CPU backing', 'device', 'cuda:0'),
                         ('load/store differ', 'load_uses_same_storage', False), ('wrong allocated bytes', 'nbytes', 63)]:
    bad = copy.deepcopy(good); bad['cpu_buffers'][0][field] = value
    check(name, bad, False)
bad = copy.deepcopy(good); bad['requested_cpu_bytes'] = 160
check('budget mismatch', bad, False)
bad = copy.deepcopy(good); bad['cpu_buffers'] = []
check('empty buffer set is not a positive observation', bad, False)
with tempfile.TemporaryDirectory(prefix='allocation-report-check-', dir=b) as temporary:
    root = Path(temporary); case = root/'00-wait'; case.mkdir()
    (root/'result.json').write_text(json.dumps({'cases':[{'name':'00-wait'}]}))
    for name, text, expected in [
        ('missing startup receipt', 'pin_memory requested True', False),
        ('single valid startup receipt', PREFIX+json.dumps(good), True),
        ('multiple receipts require investigation', (PREFIX+json.dumps(good)+'\n')*2, False),
        ('malformed receipt fails closed', PREFIX+'{}invalid', False),
    ]:
        (case/'server.log').write_text(text)
        assert report(root)['all_actual_pinned_paths_established'] is expected, name
        checks.append(dict(check=name, passed=True, expected_path=expected))
(b/'allocation-receipt-parser-checks.json').write_text(json.dumps(dict(passed=True,
    checks=checks, scope='Synthetic metadata interpretation only. This checks false-positive prevention and budget arithmetic, not actual pinning, allocation or GPU transfers.'), indent=2)+'\n')
print('Passed', len(checks), 'allocation receipt interpretation checks; owned temporary fixture removed')
