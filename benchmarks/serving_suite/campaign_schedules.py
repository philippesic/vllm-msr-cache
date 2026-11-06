"""Fixed controls outside the throughput matrix; generated before protocol lock."""
from .campaign_contracts import atomic_json


def supplementary(out, resolved):
    overhead, operational = [], []
    control_orders = [('O0','O1','O2'),('O1','O2','O0'),('O2','O0','O1'),
                      ('O2','O1','O0'),('O1','O0','O2'),('O0','O2','O1')]
    for cell in resolved['cells']:
        model, name = cell['model'],cell['cell']
        if model == 'qwen3_8b_awq' and name in ('B01','B02'):
            for repeat,order in enumerate(control_orders):
                overhead.append(dict(model=model,cell=name,repeat=repeat,systems=list(order)))
        elif model != 'qwen3_8b_awq' and name == 'B02':
            overhead.append(dict(model=model,cell=name,repeat=0,systems=['O2']))
        if name == 'B06':
            operational.append(dict(model=model,cell=name,repeat=0,systems=['arc','wait']))
    result = dict(overhead=overhead,operational=operational)
    for phase,blocks in result.items():
        atomic_json(out/(phase+'-schedule.json'),dict(schema='serving-campaign-supplementary-schedule-v1',blocks=blocks))
    return result
