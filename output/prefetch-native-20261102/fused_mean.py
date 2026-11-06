"""Owned FP32 block means without materializing gathered full KV pages.

The caller must retain source page ownership until the current stream completes.
This helper neither modifies KV nor publishes cache identities. CUDA/Triton dense
one-page-per-key fast path only; existing worker handles all other layouts.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _mean_pages(KV, IDS, OUT, S0:tl.constexpr, S1:tl.constexpr,
                S2:tl.constexpr, S3:tl.constexpr, HEADS:tl.constexpr,
                TOKENS:tl.constexpr, DIM:tl.constexpr,
                BT:tl.constexpr, BD:tl.constexpr):
 row=tl.program_id(0)
 head=tl.program_id(1)
 block=tl.load(IDS+row)
 t=tl.arange(0,BT)
 d=tl.arange(0,BD)
 values=tl.load(KV+block*S0+head*S1+t[:,None]*S2+d[None,:]*S3,
                mask=(t[:,None]<TOKENS)&(d[None,:]<DIM),other=0).to(tl.float32)
 mean=tl.sum(values,axis=0)/TOKENS
 tl.store(OUT+row*HEADS*DIM+head*DIM+d,mean,mask=d<DIM)

def owned_mean_pages(kv,block_ids,head_size):
 if (kv.ndim!=4 or not kv.is_cuda or kv.dtype!=torch.float16
     or any(s<=0 for s in kv.shape) or not 0<head_size<=kv.shape[3]):
  raise ValueError('fused means require CUDA FP16 [blocks,heads,tokens,KVdim]')
 ids=[int(i) for i in block_ids]
 if any(i<0 or i>=kv.shape[0] for i in ids):
  raise IndexError('summary source block is outside the KV cache')
 output=torch.empty((len(ids),kv.shape[1],head_size),device=kv.device,dtype=torch.float32)
 if not ids:return output
 # Pinned staging and an asynchronous index copy avoid a device-wide fence.
 # PyTorch's pinned allocator holds host storage until this copy completes.
 host_ids=torch.tensor(ids,dtype=torch.int64,pin_memory=True)
 device_ids=host_ids.to(kv.device,non_blocking=True)
 _mean_pages[(len(ids),kv.shape[1])](kv,device_ids,output,*kv.stride(),
     kv.shape[1],kv.shape[2],head_size,triton.next_power_of_2(kv.shape[2]),
     triton.next_power_of_2(head_size),num_warps=4)
 return output
