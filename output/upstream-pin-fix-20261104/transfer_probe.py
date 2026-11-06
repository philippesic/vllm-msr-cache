import os,sys,json
sys.path.insert(0, '/home/philip/vllm-semantic-cache-ext')
from harness.gpu_lock import GPULock
with GPULock():
 import torch
 from vllm import _custom_ops as ops
 n=int(sys.argv[1]); page=16384
 src=torch.arange(n*page, device='cuda',dtype=torch.int32).to(torch.uint8).reshape(n,page)
 if len(sys.argv)>2:
  from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
  from vllm.v1.kv_offload.cpu.gpu_worker import pin_mmap_region
  region=SharedOffloadRegion('pinfix-'+str(os.getpid()),n,0,page,page,barrier=lambda:None)
  if sys.argv[2]=='torch':
   result=torch.cuda.cudart().cudaHostRegister(region._base.data_ptr(),region.total_size_bytes,0)
   print('torch registration',result,flush=True)
  else:
   pin_mmap_region(region)
  print('region pinned',region.is_pinned,flush=True)
  host=region.create_next_worker_view(page).view(torch.uint8)
 else:
  host=torch.empty((n,page),dtype=torch.uint8,pin_memory=True)
 import ctypes
 from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
 lib=CudaRTLibrary().lib
 class Attr(ctypes.Structure):
  _fields_=[('type',ctypes.c_int),('device',ctypes.c_int),('devicePointer',ctypes.c_void_p),('hostPointer',ctypes.c_void_p)]
 attr=Attr(); lib.cudaPointerGetAttributes.argtypes=[ctypes.POINTER(Attr),ctypes.c_void_p]
 code=lib.cudaPointerGetAttributes(ctypes.byref(attr),ctypes.c_void_p(host.data_ptr()))
 print(json.dumps({'attributes':code,'type':attr.type,'devicePointer':attr.devicePointer,'hostPointer':attr.hostPointer,'tensorPointer':host.data_ptr()}),flush=True)
 dst=torch.zeros_like(src)
 def desc(t):
  return torch.tensor([t.data_ptr()+i*page for i in range(n)],dtype=torch.int64,pin_memory=True)
 a,b,c=desc(src),desc(host),desc(dst)
 if len(sys.argv)>3 and sys.argv[3]=='alias':
  b += attr.devicePointer-host.data_ptr()
 sizes=torch.full((n,),page,dtype=torch.int64,pin_memory=True)
 stream=torch.cuda.Stream()
 torch.cuda.synchronize()
 with torch.cuda.stream(stream):
  if len(sys.argv)>3 and sys.argv[3]=='copy':
   host.copy_(src,non_blocking=True)
  else:
   ops.swap_blocks_batch(a,b,sizes)
 stream.synchronize()
 print(json.dumps({'stage':'store','n':n,'pinned':host.is_pinned(),'equal':torch.equal(host,src.cpu())}),flush=True)
 with torch.cuda.stream(stream):
  if len(sys.argv)>3 and sys.argv[3]=='copy':
   dst.copy_(host,non_blocking=True)
  else:
   ops.swap_blocks_batch(b,c,sizes,is_src_access_order_any=True)
 stream.synchronize()
 assert torch.equal(src,dst)
 print(json.dumps({'stage':'restore','n':n,'equal':True}),flush=True)
