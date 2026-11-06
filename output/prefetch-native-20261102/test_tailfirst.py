import ast
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
H=Path(__file__).resolve().parent

def load(root):
 t=ast.parse((H/root/'vllm/v1/core/block_pool.py').read_text());ns={'make_block_hash_with_group_id':lambda h,g:h}
 methods=[n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name in {'publish_completed_prefix','publish_completed_suffix'}]
 tree=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+methods,type_ignores=[])
 exec(compile(ast.fix_missing_locations(tree),'<native-publication>','exec'),ns);return ns

class Pool:
 def __init__(self,local=0):
  self.enable_caching=True;self.num_gpu_blocks=8;self.hash_block_size=16;self.cached_block_hashes_by_block={};self.queue=[];self.cached={}
  self.blocks=[NS(block_id=i,is_null=(i==0),ref_cnt=1,block_hash=None,block_hash_num_tokens=None,data=f'owned-KV-{i}') for i in range(8)]
  self.pages=self.blocks[1:4]
  for i,b in enumerate(self.pages[:local]):b.block_hash=b'abc'[i:i+1];b.block_hash_num_tokens=(i+1)*16;self.cached[b.block_hash]=b
 def get_cached_block(self,k,g):return [self.cached[k]] if k in self.cached else None
 def cache_full_blocks(self,request,blocks,first,last,size,group):
  for i,b in enumerate(blocks[first:last],first):b.block_hash=request.block_hashes[i];b.block_hash_num_tokens=(i+1)*size;self.cached[b.block_hash]=b
 def free_blocks(self,blocks):
  for b in blocks:b.ref_cnt-=1;self.queue.append(b)
 def evict_blocks(self,ids):
  for b in self.pages:
   if b.block_id in ids:self.cached.pop(b.block_hash,None);b.block_hash=None
 def evict_one(self):
  b=self.queue.pop(0);self.cached.pop(b.block_hash);b.block_hash=None;return b.block_id
 def longest(self):
  n=0
  for k in [b'a',b'b',b'c']:
   if k not in self.cached:break
   n+=1
  return n

def request():return NS(skip_reading_prefix_cache=False,num_tokens=49,block_hashes=[b'a',b'b',b'c'])
@pytest.mark.parametrize('local',[0,1])
def test_native_tail_eviction_retains_longest_prefix(local):
 outcomes=[]
 for root in ['runtime-shared-lookup','runtime-prefix-order']:
  ns=load(root);p=Pool(local);data=[b.data for b in p.pages]
  fn=ns['publish_completed_suffix' if local else 'publish_completed_prefix'];args=(request(),p.pages[:local],p.pages[local:]) if local else (request(),p.pages)
  assert fn(p,*args,block_size=16,kv_cache_group_id=0,completion_verified=True)
  assert [b.data for b in p.pages]==data
  assert all(b.ref_cnt==0 for b in p.pages[local:]);assert p.pages[0].ref_cnt==(1 if local else 0)
  p.evict_one();outcomes.append(p.longest())
 assert outcomes==([0,2] if local==0 else [1,2]),'old forward-order negative control must lose the leading prefix'
@pytest.mark.parametrize('local',[0,1])
def test_unverified_copy_keeps_all_ownership(local):
 p=Pool(local);fn=load('runtime-prefix-order')['publish_completed_suffix' if local else 'publish_completed_prefix'];args=(request(),p.pages[:local],p.pages[local:]) if local else (request(),p.pages)
 assert not fn(p,*args,block_size=16,kv_cache_group_id=0,completion_verified=False)
 assert not p.queue and all(b.ref_cnt==1 for b in p.pages)
def test_only_publication_order_changed():
 def funcs(root):return {n.name:ast.dump(n) for n in ast.walk(ast.parse((H/root/'vllm/v1/core/block_pool.py').read_text())) if isinstance(n,ast.FunctionDef)}
 a,b=funcs('runtime-shared-lookup'),funcs('runtime-prefix-order')
 assert set(a)==set(b)
 assert {k for k in a if a[k]!=b[k]}=={'publish_completed_prefix','publish_completed_suffix'}
