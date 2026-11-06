"""Run only when the GPU owner permits; random score equivalence, not task quality."""
import argparse,json
from pathlib import Path
import torch
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();torch.set_num_threads(1)
assert torch.cuda.is_available()
generator=torch.Generator().manual_seed(9061)
results=[]
for name,requests,blocks,heads,dim in [('host',3,199,2,128),('larger',8,1024,8,128),('near_ties',3,199,2,128)]:
    queries=torch.randn(requests,heads,dim,generator=generator)
    means=torch.randn(blocks,heads,dim,generator=generator)
    if name=='near_ties':
        means[1::2]=means[:blocks-1:2]+torch.randn(blocks//2,heads,dim,generator=generator)*1e-7
    q=queries.cuda();m=means.cuda()
    gpu=(q[:,None]*m[None]).sum(-1).mean(-1).cpu()
    q_cpu=q.detach().to(device='cpu',copy=True);m_cpu=m.detach().to(device='cpu',copy=True)
    cpu=(q_cpu[:,None]*m_cpu[None]).sum(-1).mean(-1)
    gpu_rank=gpu.argsort(dim=1,descending=True,stable=True)
    cpu_rank=cpu.argsort(dim=1,descending=True,stable=True)
    wrong=(q_cpu.mean(1)[:,None]*m_cpu.mean(1)[None]).sum(-1)
    top=min(10,blocks)
    results.append({'case':name,'shape':[requests,blocks,heads,dim],
        'max_absolute_error':(cpu-gpu).abs().max().item(),
        'mean_absolute_error':(cpu-gpu).abs().mean().item(),
        'top1_agreement':(gpu_rank[:,0]==cpu_rank[:,0]).float().mean().item(),
        'top10_set_overlap':sum(len(set(a[:top].tolist())&set(b[:top].tolist())) for a,b in zip(gpu_rank,cpu_rank))/(requests*top),
        'full_rank_equal':torch.equal(gpu_rank,cpu_rank),
        'score_tolerance_passed':torch.allclose(cpu,gpu,rtol=2e-5,atol=2e-5),
        'wrong_head_pooling_control_rejected':not torch.allclose(cpu,wrong,rtol=2e-5,atol=2e-5)})
args.output.write_text(json.dumps({'device':torch.cuda.get_device_name(),
    'seed':9061,'torch_version':torch.__version__,'results':results,
    'scope':'Synthetic numerical audit only; near-tie ranking differences can occur. No task-level retention or KV-correctness claim.'},indent=2)+'\n')
