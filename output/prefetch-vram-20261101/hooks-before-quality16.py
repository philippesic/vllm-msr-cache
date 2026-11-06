"""Task-local runtime seam and passive measurement. No cache-policy substitution."""
import functools, importlib.abc, importlib.machinery, importlib.util, json, os, sys, time
from collections import Counter
from pathlib import Path
H=Path(__file__).resolve().parents[1]
DIRECTORY=Path(os.environ['MATRIX_RUN_DIR']) if os.getenv('MATRIX_RUN_DIR') else None
COUNTS=Counter();TIMES={};SAMPLES=[];MANAGER=None;SCHEDULER=None;LAST_MULTI_TOKEN=False
RUNTIME={'vllm.v1.engine.async_llm','vllm.v1.engine.core','vllm.v1.engine.core_client','vllm.v1.core.sched.scheduler','vllm.v1.core.block_pool'}
TARGETS=RUNTIME|{'starlette.applications','vllm.v1.worker.gpu_worker','semantic_offload.worker','semantic_offload.manager','semantic_offload.policy','semantic_offload.connector','vllm.v1.kv_offload.cpu.manager','vllm.v1.worker.gpu.model_runner'}
def emit(kind,value):
    print('MATRIX_'+kind+' '+json.dumps(value,sort_keys=True),flush=True)
def wrap(cls,name,label,after=None):
    original=cls.__dict__.get(name)
    if original is None:return
    @functools.wraps(original)
    def call(self,*a,**kw):
        t=time.perf_counter_ns();c=time.thread_time_ns()
        try:
            result=original(self,*a,**kw)
            if after:after(self,a,kw,result)
            return result
        finally:
            stat=TIMES.setdefault(label,{'calls':0,'host_wall_ns':0,'thread_cpu_ns':0})
            stat['calls']+=1;stat['host_wall_ns']+=time.perf_counter_ns()-t;stat['thread_cpu_ns']+=time.thread_time_ns()-c
    setattr(cls,name,call)
def sample(phase):
    if len(SAMPLES)>=20000:return
    row={'at_ns':time.monotonic_ns(),'phase':phase}
    if MANAGER is not None and hasattr(MANAGER,'_policy'):
        p=MANAGER._policy;native=getattr(p,'_lru',p)
        entries=list(native.blocks.values()) if hasattr(native,'blocks') else list(p.t1.values())+list(p.t2.values())
        row.update(cpu_resident=len(entries),cpu_ready=sum(b.is_ready for b in entries),cpu_pending=sum(not b.is_ready for b in entries),cpu_capacity=MANAGER._num_blocks,cpu_evictable=MANAGER._num_evictable_cache_blocks,policy_type=type(p).__name__)
    pool=getattr(SCHEDULER,'gpu_block_pool',None)
    if pool is not None:
        row.update(gpu_capacity=pool.num_gpu_blocks,gpu_free=pool.get_num_free_blocks(),gpu_reserved=SCHEDULER._prefetch_reserved_blocks,gpu_referenced=sum(b.ref_cnt>0 for b in pool.blocks if b.block_id!=0),gpu_hashed=sum(b.block_hash is not None for b in pool.blocks if b.block_id!=0))
    SAMPLES.append(row)
def state():return {'pid':os.getpid(),'counts':dict(COUNTS),'method_times':TIMES,'samples':SAMPLES,'manager_type':type(MANAGER).__name__ if MANAGER else None,'policy_type':type(getattr(MANAGER,'_policy',None)).__name__ if MANAGER else None,'inclusive_times_overlap':True}
def flush(label):
    if DIRECTORY:
        DIRECTORY.mkdir(parents=True,exist_ok=True)
        (DIRECTORY/f'{label}-{os.getpid()}.json').write_text(json.dumps(state()))
def patch(m):
    global MANAGER,SCHEDULER
    n=m.__name__
    if n=='starlette.applications':
        original=m.Starlette.__call__
        async def call(self,scope,receive,send):
            if scope.get('type')!='http' or scope.get('path')!='/_matrix/control':return await original(self,scope,receive,send)
            from starlette.requests import Request
            from starlette.responses import JSONResponse
            body=await Request(scope,receive).json();engine=self.state.engine_client
            workers=await engine.collective_rpc('matrix_control',timeout=60,args=(body['operation'],))
            scheduler=await engine.engine_core.call_utility_async('matrix_export')
            await JSONResponse({'workers':workers,'scheduler':scheduler})(scope,receive,send)
        m.Starlette.__call__=call
    elif n=='vllm.v1.engine.core':
        def export(self):
            sample('postload_export');flush('scheduler')
            from semantic_offload.profiling import PROFILE
            PROFILE.flush()
            return state()
        m.EngineCore.matrix_export=export
    elif n=='vllm.v1.worker.gpu_worker':
        def control(self,operation):
            import torch
            from vllm.distributed.kv_transfer import get_kv_transfer_group
            from semantic_offload.profiling import PROFILE
            w=get_kv_transfer_group().connector_worker.worker
            w._drain_async_scores(wait=True)
            torch.cuda.synchronize()
            if operation=='reset':
                torch.cuda.reset_peak_memory_stats();COUNTS.clear();TIMES.clear();SAMPLES.clear()
                PROFILE.drain();PROFILE.metrics.clear();PROFILE.counters.clear();PROFILE.calls.clear()
            PROFILE.flush();flush('worker')
            tensors=[value for summaries in w.durable_summaries.values() for summary in summaries for field in ['min','max','mean','mad'] if isinstance((value:=getattr(summary,field,None)),torch.Tensor)]
            storages={t.untyped_storage().data_ptr():t.untyped_storage().nbytes() for t in tensors}
            metadata={'logical_bytes':sum(t.numel()*t.element_size() for t in tensors),'retained_storage_bytes':sum(storages.values()),'storage_count':len(storages)}
            return {**state(),'summary_memory':metadata,'operation':operation,'capture':getattr(w,'_matrix_capture',{}),'durable_summary_blocks':len(w.durable_summaries),'async_stats':w._async_scoring.stats() if getattr(w,'_async_scoring',None) else None,'pending_async_observations':len(getattr(w,'_pending_async_score_observations',[])),'pending_profile_cuda':len(PROFILE.pending),'gpu_allocated_bytes':torch.cuda.memory_allocated(),'gpu_reserved_bytes':torch.cuda.memory_reserved(),'load_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'load_peak_reserved_bytes':torch.cuda.max_memory_reserved()}
        m.Worker.matrix_control=control
    elif n=='vllm.v1.kv_offload.cpu.manager':
        cls=m.CPUOffloadingManager
        def prepared(self,a,kw,out):
            global MANAGER
            MANAGER=self;COUNTS['store_offered_keys']+=len(a[0]);COUNTS['store_prepare_calls']+=1
            if out is None:COUNTS['store_prepare_refused']+=1
            else:COUNTS['store_accepted_keys']+=len(out.keys_to_store);COUNTS['cpu_evicted_blocks']+=len(out.evicted_keys)
            if out is None or (out and (out.evicted_keys or out.keys_to_store)):sample('store_prepare')
        def completed(self,a,kw,out):
            COUNTS['store_completed_keys' if kw.get('success',True) else 'store_failed_keys']+=len(a[0]);sample('store_complete')
        wrap(cls,'prepare_store','manager.prepare_store',prepared);wrap(cls,'complete_store','manager.complete_store',completed)
        def lookup(self,a,kw,out):COUNTS['lookup_'+out.name]+=1
        wrap(cls,'lookup','manager.lookup',lookup)
    elif n=='semantic_offload.manager':
        for cls in [m.SemanticOffloadingManager,m.PreservationOffloadingManager]:
            def updated(self,a,kw,out):
                COUNTS['relevance_update_calls']+=1
                COUNTS['relevance_scored_rows']+=sum(len(rows) for reqs in a[0].values() for rows in reqs.values())
            wrap(cls,'update_relevance','manager.update_relevance',updated)
            def applied(self,a,kw,out):
                COUNTS['relevance_apply_calls']+=1
                COUNTS['relevance_apply_keys']+=sum(len(values) for values in a[0].values())
            wrap(cls,'apply_relevance_updates','manager.apply_relevance_updates',applied)
    elif n=='semantic_offload.policy':
        original=m.SemanticPolicy.evict
        @functools.wraps(original)
        def evict(self,*a,**kw):
            COUNTS['semantic_evict_calls']+=1
            COUNTS['semantic_evict_calls_with_scores']+=bool(self._relevance_ema.get(self._method))
            relevance=self._relevance_ema.get(self._method,{})
            COUNTS['semantic_evict_calls_with_resident_scores']+=any(k in relevance for k in self._lru.evictable_blocks)
            return original(self,*a,**kw)
        m.SemanticPolicy.evict=evict
    elif n=='semantic_offload.worker':
        cls=m.SemanticOffloadingWorker
        original=cls._on_queries_captured
        @functools.wraps(original)
        def captured(self,ids,tensor):
            d=self.__dict__.setdefault('_matrix_capture',{'calls':0,'rows':0,'first_batches':[]});d['calls']+=1;d['rows']+=len(ids)
            if len(d['first_batches'])<8:d['first_batches'].append({'ids':list(ids),'shape':list(tensor.shape),'dtype':str(tensor.dtype),'at_ns':time.monotonic_ns()})
            result=original(self,ids,tensor)
            if os.getenv('MATRIX_QUALITY')=='1':
                state=self.__dict__.setdefault('_quality_state',{'sequence':0,'history':{},'metrics':{}})
                sequence=state['sequence'];state['sequence']+=1
                current=self._pending_scores.get(self._method,{})
                for stride in [8,32,"multi_token"]:
                    hist=state['history'].setdefault(stride,{})
                    metric=state['metrics'].setdefault(stride,{'observations':0,'top10_overlap_sum':0.,'rank_corr_sum':0.,'compared_keys':0})
                    for rid in ids:
                        rows=current.get(rid,[])
                        if not rows:continue
                        if ((LAST_MULTI_TOKEN if stride=="multi_token" else sequence%stride==0) or rid not in hist):hist[rid]=list(rows)
                        prior=hist[rid];live={k for k,v in rows};old={k for k,v in prior};common=live&old
                        if len(common)<2:continue
                        a=[k for k,v in rows if k in common];b=[k for k,v in prior if k in common];pos={k:i for i,k in enumerate(b)};n=len(a);top=min(10,n)
                        metric['observations']+=1;metric['compared_keys']+=n
                        metric['top10_overlap_sum']+=len(set(a[:top])&set(b[:top]))/top
                        metric['rank_corr_sum']+=1-6*sum((i-pos[k])**2 for i,k in enumerate(a))/(n*(n*n-1))
                if DIRECTORY:(DIRECTORY/'quality.json').write_text(json.dumps({'proxy':'hold last query ranking on common resident keys; not actual policy outcomes','metrics':state['metrics']}))
            return result
        cls._on_queries_captured=captured
        # No hot-path timings beyond passive aggregates in headline runs.
        for method in ['_on_queries_captured','_score_queries','_execute_async_score_job','_rebuild_stack_cache']:
            if os.getenv('MATRIX_DIAGNOSTIC')=='1':wrap(cls,method,'worker.'+method)
    elif n=='semantic_offload.connector':
        cls=m.SemanticOffloadingConnectorScheduler
        original=cls.__init__
        def init(self,*a,**kw):
            global SCHEDULER,MANAGER
            original(self,*a,**kw);SCHEDULER=self;MANAGER=self.manager
            emit('CONFIGURATION',{'manager':type(self.manager).__name__,'policy':type(getattr(self.manager,'_policy',None)).__name__,'scoring_schedule':self._scoring_schedule,'trigger':self._prefetch_trigger,'selector':self._prefetch_selector,'budget_fraction':self._prefetch_budget_fraction})
        cls.__init__=init
        original_execution=cls._record_scheduler_execution
        def execution(self,out):
            r=original_execution(self,out);COUNTS['scheduler_steps']+=1
            if COUNTS['scheduler_steps']%16==0:sample('periodic16')
            return r
        cls._record_scheduler_execution=execution
    elif n=='vllm.v1.worker.gpu.model_runner':
        if os.getenv('MATRIX_QUALITY')=='1':
            prepare=m.GPUModelRunner.prepare_inputs
            def prepare_quality(self,*a,**kw):
                global LAST_MULTI_TOKEN
                result=prepare(self,*a,**kw)
                LAST_MULTI_TOKEN=any(int(x)>1 for x in result.num_scheduled_tokens)
                return result
            m.GPUModelRunner.prepare_inputs=prepare_quality
        if os.getenv('MATRIX_DIAGNOSTIC')=='1':
            for method in ['execute_model','prepare_inputs','prepare_attn']:wrap(m.GPUModelRunner,method,'runner.'+method)
class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname not in TARGETS:return
        if fullname in RUNTIME:spec=importlib.util.spec_from_file_location(fullname,H/'runtime'/Path(*fullname.split('.')).with_suffix('.py'))
        else:spec=importlib.machinery.PathFinder.find_spec(fullname,path)
        if spec is None or spec.loader is None:return
        loader=spec.loader
        class Loader(importlib.abc.Loader):
            def create_module(self,spec):return loader.create_module(spec)
            def exec_module(self,module):loader.exec_module(module);patch(module)
        spec.loader=Loader();return spec
if DIRECTORY:sys.meta_path.insert(0,Finder())
