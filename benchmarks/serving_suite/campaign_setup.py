"""Resolve per-model workload/plan inputs using explicit development profiles."""
from __future__ import annotations

import copy
from pathlib import Path

from .campaign_contracts import atomic_json, digest, fields, load, positive
from .campaign_workload import linear_graph, team_graph
from .contracts import SCHEMA, validate_plan


def validate_profile(profile):
    fields(profile, ('gpu_blocks', 'cpu_bytes', 'max_model_len', 'gpu_memory_utilization', 'batch_tokens', 'status'),
           ('model_format', 'text_only', 'roomy_cpu_bytes', 'roomy_gpu_blocks', 'runtime_python', 'allocation_receipts', 'kv_bytes_per_token'))
    for name in ('gpu_blocks', 'cpu_bytes', 'max_model_len', 'batch_tokens'):
        positive(profile[name], name)
    if profile['gpu_blocks'] * 16 < profile['max_model_len']:
        raise ValueError('GPU KV capacity is smaller than one declared context')
    if profile['status'] not in ('development', 'measured') or not 0 < profile['gpu_memory_utilization'] < 1:
        raise ValueError('Invalid resource profile status/utilization')
    return profile


def b04(corpus):
    source = corpus['sessions']
    if len(source) < 20:
        raise ValueError('Hot/scan/return requires four hot and sixteen cold documents')
    result = []
    for phase in ('hot', 'cold_scan', 'hot_return'):
        cohort = source[4:20] if phase == 'cold_scan' else [source[i % 4] for i in range(16)]
        for index, original in enumerate(cohort):
            session = copy.deepcopy(original)
            session.update(id=f'{phase}-{index}-{original["id"]}', phase_class=phase)
            result.append(session)
    return {**corpus, 'sessions': result, 'workload_kind': 'hot_scan_return',
            'provenance': corpus['provenance'] + ' Authored three-phase exact-prefix hot-set/scan/return schedule; repeated hot questions are intentional.'}


def assemble(campaign, catalogs, assets, host, partition, destination):
    """Write explicit effective inputs; return bindings, never a success receipt."""
    destination = Path(destination).resolve()
    cache = Path(host['asset_cache']) / 'corpora' / partition
    bindings = {}
    for model, cells in campaign['matrix'].items():
        artifact = assets['models'][model]
        if artifact['status'] != 'prepared':
            raise ValueError('Prepare pinned model artifacts before workload resolution')
        profile = validate_profile(host['model_profiles'][model])
        model_path = host['models'].get(model, artifact['path'])
        for cell in cells:
            settings = catalogs['workloads'][cell]
            adapter = settings['adapter']
            if adapter == 'chat':
                corpus = load(cache/'chat.json')
                corpus['sessions'] = corpus['sessions'][:1]
            elif adapter in ('independent_qa', 'summary'):
                corpus = load(cache/('qa.json' if adapter == 'independent_qa' else 'summary.json'))
            elif adapter == 'hot_scan':
                corpus = b04(load(cache/'qa.json'))
            elif adapter == 'documents':
                corpus = load(cache/'documents.json')
            else:
                corpus = load(cache/'repository.json')
                if adapter == 'team':
                    corpus['sessions'] = corpus['sessions'][:4]
                elif adapter == 'mixed':
                    ordinary = copy.deepcopy(load(cache/'qa.json')['sessions'][:16])
                    for session in ordinary:
                        session.update(id='mixed-unhinted-' + session['id'], publish_hints=False, traffic_class='unhinted')
                    corpus['sessions'] = [s for pair in zip(corpus['sessions'], ordinary) for s in pair]
            for session in corpus['sessions']:
                if 'publish_hints' not in session:
                    session['publish_hints'] = settings['hints'] and bool(session.get('tools'))
                if adapter == 'mixed' and session.get('tools'):
                    session['traffic_class'] = 'hinted'
            configuration = copy.deepcopy(catalogs['policy']['configuration'])
            extra = configuration['kv_connector_extra_config']
            extra['admission_namespace'].update(model_revision=artifact['revision'], tokenizer_revision=artifact['revision'],
                layout_namespace=model + '-dense-tp1-fp16-block16')
            cpu = profile['cpu_bytes']
            gpu = profile['gpu_blocks']
            if cell == 'B09':
                if cpu % 2:
                    raise ValueError('Full CPU-byte profile cannot be halved exactly')
                cpu //= 2
            if cell == 'B07':
                if 'roomy_cpu_bytes' not in profile or 'roomy_gpu_blocks' not in profile:
                    raise ValueError('B07 requires an explicit matched roomy envelope')
                cpu, gpu = profile['roomy_cpu_bytes'], profile['roomy_gpu_blocks']
            extra['cpu_bytes_to_use'] = cpu
            plan = dict(schema=SCHEMA, tier='screen', campaign_adapter=True, cohort_mode='fixed', repeats=1,
                sessions=len(corpus['sessions']), turns=max(len(s['turns']) for s in corpus['sessions']),
                concurrency=settings['concurrency'], client_concurrency=0, tool_concurrency=settings['tool_concurrency'],
                workflow_concurrency=1 if cell in ('B01', 'B05') else 0,
                waves=1, max_tokens=settings['max_tokens'], model=model_path,
                gpu_blocks=gpu, cpu_bytes=cpu, max_model_len=profile['max_model_len'],
                gpu_memory_utilization=profile['gpu_memory_utilization'], batch_tokens=profile['batch_tokens'],
                seed=campaign['seed'], execution='eager', sample_interval_s=0.25,
                startup_timeout_s=host['limits']['startup_timeout_s'], request_timeout_s=host['limits']['request_timeout_s'],
                tool_timeout_s=host['limits']['tool_timeout_s'], case_timeout_s=host['limits']['case_timeout_s'],
                tool_output_chars=2000, continuation_credit_ms=8000, publish_hints=True,
                candidate_config=configuration, qualification_trace=True, compact_events=True, observe_runtime=True,
                arrival_interval_s=0, wave_interval_s=0, natural_eos=True,
                generation_mode='live non-thinking greedy generation to EOS with declared finite output cap',
                chat_template_kwargs={'enable_thinking': False}, synthetic=False,
                corpus_provenance=corpus['provenance'], model_format=profile.get('model_format'), text_only=profile.get('text_only', False),
                required_activity=['store', 'restore', 'cpu_recency'] if cell in ('B06','B08','B09','B10','B11','B12') else [])
            # Generic/no-opportunity cells intentionally allow no activity. This
            # does not satisfy pressure or native correctness gates elsewhere.
            for session in corpus['sessions']:
                session['graph'] = (team_graph if adapter == 'team' else linear_graph)(session, plan)
            plan['expected_requests_per_wave'] = sum(n['kind'] == 'request' for s in corpus['sessions'] for n in s['graph']['nodes'])
            plan['expected_tools_per_wave'] = sum(n['kind'] == 'tool' for s in corpus['sessions'] for n in s['graph']['nodes'])
            if len(cells[cell]) == 4:
                runtimes = copy.deepcopy(host.get('resolved_published_runtimes'))
                if not runtimes:
                    raise ValueError('Resolve published runtime identities before four-system assembly')
                for system, runtime in runtimes.items():
                    runtime['cpu_bytes'] = cpu
                    runtime['kv_transfer_config'].setdefault('kv_connector_extra_config', {})['cpu_bytes_to_use'] = cpu
                    if system == 'continuum':
                        runtime['env']['LMCACHE_MAX_LOCAL_CPU_SIZE'] = str(cpu/2**30)
                plan['published_runtimes'] = runtimes
                plan['published_variants'] = ['arc', 'wait', 'continuum', 'upstream']
            validate_plan(plan)
            directory = destination/model/cell
            atomic_json(directory/'plan.json', plan)
            atomic_json(directory/'corpus.json', corpus)
            bindings[model + ':' + cell] = dict(plan=str(directory/'plan.json'), corpus=str(directory/'corpus.json'),
                plan_digest=digest(plan), corpus_digest=digest(corpus), partition=partition, cohort_mode='fixed')
    return bindings


def resolve(campaign_path, host_path, destination):
    from .campaign import catalogs
    from .campaign_contracts import validate_host
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError('Resolution requires a new destination; old effective inputs are immutable')
    spec, records = catalogs(campaign_path)
    host = validate_host(load(host_path))
    assets = load(Path(host['asset_cache'])/(spec['id']+'-assets.json'))
    host['workload_bindings'] = assemble(spec, records, assets, host, 'development', destination/'development')
    host['confirmation_bindings'] = assemble(spec, records, assets, host, 'confirmation', destination/'confirmation')
    path = destination/'host.private.json'
    atomic_json(path, host)
    path.chmod(0o600)
    return dict(host=str(path), development_cells=len(host['workload_bindings']),
                confirmation_cells=len(host['confirmation_bindings']),
                scope='Prepared inputs and proposed profiles; native resource/quality gates remain required')


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--host', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(resolve(args.campaign,args.host,args.out),indent=2))


if __name__ == '__main__':
    main()
