"""Bounded recovery of the fourth case; first three raw cases stay untouched."""
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/home/philip/vllm-semantic-cache-ext')
sys.path.insert(0, str(ROOT))
from harness.gpu_lock import GPULock
from benchmarks.serving_suite.contracts import write_json
from benchmarks.serving_suite.published import check_source, comparisons
from benchmarks.serving_suite.report import render_report
from benchmarks.serving_suite.runner import available_gpu, source_identity, plan_for_variant, run_case, select_sessions
from benchmarks.serving_suite.verify import verify, seal


def main():
    directory = ROOT/'output/fair-four-20261104/comparison-final'
    with GPULock():
        import psutil
        started = time.time()
        before_verification = verify(directory)
        result = json.loads((directory/'result.json').read_text())
        plan = result['plan']
        initial = json.loads((directory/'source.json').read_text())
        assert source_identity()['digest'] == initial['digest']
        assert [c['variant'] for c in result['cases']] == ['wait', 'arc', 'continuum']
        assert all(c['evaluation']['status'] == 'passed' and c['summary']['completed'] == 64
                   and c['cleanup']['verified'] for c in result['cases'])
        assert not (directory/'00-upstream').exists()
        for name, runtime in plan['published_runtimes'].items():
            check_source(name, runtime)
        hardware = available_gpu()
        original_manifest = json.loads((directory/'artifacts.json').read_text())
        record = dict(reason=result['error'], original_verification=before_verification,
            source_digest=initial['digest'], plan_sha256=hashlib.sha256((directory/'plan.json').read_bytes()).hexdigest(),
            coordinator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            command=[sys.executable, str(Path(__file__).resolve())], started_unix=started,
            gap_since_third_case_cleanup_s=started-(result['cases'][-1]['started_unix']+result['cases'][-1]['wall_s']),
            hardware_before=hardware, host_memory_before=psutil.virtual_memory()._asdict(),
            swap_before=psutil.swap_memory()._asdict(),
            device_before=subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu,clocks.sm,power.draw,memory.used','--format=csv'],text=True),
            scope='Exactly one not-yet-started upstream case; same frozen source, plan, selected inputs, policy settings and measurements.')
        for old, new in [('result.json','orchestration-failure.json'), ('REPORT.md','REPORT-before-continuation.md'), ('artifacts.json','artifacts-before-continuation.json')]:
            shutil.copyfile(directory/old, directory/new)
        shutil.copyfile(__file__, directory/'continuation-coordinator.py')
        write_json(directory/'continuation.json', record)
        selected = select_sessions(json.loads((directory/'corpus.json').read_text()), plan, 0)
        case_plan = plan_for_variant(plan, 'upstream')
        case = run_case(case_plan, selected, 'candidate', 0, directory, variant='upstream')
        assert source_identity()['digest'] == initial['digest']
        for name, runtime in plan['published_runtimes'].items():
            check_source(name, runtime)
        assert case['cleanup']['verified']
        for name, digest in original_manifest['sha256'].items():
            if name.startswith(('00-wait/', '00-arc/', '00-continuum/')):
                assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == digest, name
        result['cases'].append(case)
        result['orchestration_status'] = 'interrupted_after_three_complete_cases_then_bounded_continuation'
        result['orchestration_failure'] = result.pop('error')
        result['comparisons'] = comparisons(result['cases'])
        result['comparison'] = result['comparisons']['wait_vs_arc']
        states = [c['evaluation']['status'] for c in result['cases']]
        result['status'] = 'failed' if 'failed' in states else 'inconclusive' if 'inconclusive' in states else 'passed'
        result['not_started'] = []
        result['wall_s'] += time.time()-started+record['gap_since_third_case_cleanup_s']
        record.update(finished_unix=time.time(), previous_case_files_unchanged=True,
            source_unchanged=True, hardware_after=available_gpu(),
            host_memory_after=psutil.virtual_memory()._asdict(), swap_after=psutil.swap_memory()._asdict())
        write_json(directory/'continuation.json', record)
        write_json(directory/'result.json', result)
        (directory/'REPORT.md').write_text('The initial orchestration stopped after three complete arms with a Python import error. The fourth arm ran once through the same unchanged frozen run_case code under both GPU locks. Original failure, unchanged prior cases, gap/host state and recovery source are retained in continuation.json and orchestration-failure.json. This was an interrupted and explicitly continued comparison, not an uninterrupted suite.\n\n'+render_report(result))
        seal(directory)
        print(json.dumps(verify(directory), indent=2), flush=True)
        print(json.dumps(dict(variant=case['variant'], evaluation=case['evaluation'], summary=case['summary'])), flush=True)

if __name__ == '__main__':
    main()
