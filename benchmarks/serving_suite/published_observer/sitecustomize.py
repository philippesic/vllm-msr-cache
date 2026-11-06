"""Read-only return observation; never replaces engine methods or decisions."""
import os

_policy_path = os.environ.get('SUITE_PUBLISHED_OBSERVATION')
_runtime_path = os.environ.get('SUITE_RUNTIME_OBSERVATION')
if _policy_path or _runtime_path:
    import json
    import sys
    import time

    _monitor = sys.monitoring
    _tool = 4
    _monitor.use_tool_id(_tool, 'serving-suite-observation')

    def _write(path, event):
        event.update(monotonic_ns=time.monotonic_ns(), pid=os.getpid())
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(descriptor, (json.dumps(event)+'\n').encode())
        finally:
            os.close(descriptor)

    def _returned(code, offset, value):
        arguments = sys._getframe(1).f_locals
        obj = arguments.get('self')
        if _runtime_path and (code.co_name == 'build_app' or
                code.co_name == '__init__' and code.co_filename.endswith('/core/sched/scheduler.py')):
            import threading
            _write(_runtime_path, dict(event='loaded_extension_modules', scope=code.co_filename.rsplit('/vllm/',1)[-1],
                modules={name:getattr(module,'__file__',None) for name,module in list(sys.modules.items())
                         if name == 'semantic_offload' or name.startswith('semantic_offload.')},
                threads=[thread.name for thread in threading.enumerate()]))
        if code.co_name == 'apply_g1_priority':
            _write(_policy_path, dict(event='priority_application', status=value.status,
                affected_blocks=len(value.block_ids),
                priority=getattr(arguments.get('action'), 'value', None),
                request_id=getattr(arguments.get('request'), 'request_id', None)))
        elif code.co_name == 'take_blocks':
            _write(_policy_path, dict(event='priority_selection', selected_blocks=len(value)))
        elif code.co_name == 'from_request_tracker':
            tracker = arguments.get('tracker')
            if (value is not None and arguments.get('save_decode_cache')
                    and tracker.is_decode_phase and value.save_spec.can_save
                    and len(value.token_ids) > tracker.prompt_len):
                _write(_runtime_path, dict(event='lmcache_decode_publication',
                    request_id=tracker.req_id, prompt_tokens=tracker.prompt_len,
                    published_tokens=len(value.token_ids),
                    decode_tokens_in_publication=len(value.token_ids)-tracker.prompt_len,
                    skip_leading_tokens=value.save_spec.skip_leading_tokens))
        elif code.co_name == 'get_finished':
            if not hasattr(obj, 'gpu_to_cpu'):
                return
            for item in value:
                _write(_runtime_path, dict(event='completed_transfer',
                    direction='store' if obj.gpu_to_cpu else 'restore',
                    success=item.success, bytes=item.transfer_size))
        elif code.co_filename.endswith('/cpu/spec.py'):
            _write(_runtime_path, dict(event='native_cpu_geometry',
                chunks=getattr(obj, 'num_chunks', getattr(obj, 'num_blocks', None)),
                bytes_per_chunk=obj.kv_bytes_per_chunk,
                blocks_per_chunk=obj.blocks_per_chunk,
                eviction_policy=obj.eviction_policy))
        elif code.co_filename.endswith('/cpu/gpu_worker.py'):
            cpu = arguments.get('cpu_tensors')
            gpu = arguments.get('gpu_tensors')
            if cpu and gpu and 'kv_caches' in arguments:
                region = arguments.get('mmap_region')
                _write(_runtime_path, dict(event='native_worker_geometry',
                    cpu_bytes=sum(t.numel()*t.element_size() for t in cpu),
                    gpu_bytes=sum(t.numel()*t.element_size() for t in gpu),
                    blocks_per_chunk=arguments['blocks_per_chunk'],
                    pinned=bool(region.is_pinned) if region else all(t.is_pinned() for t in cpu),
                    device_aliases=sum(getattr(t, '_offload_device_ptr', t.data_ptr()) != t.data_ptr() for t in cpu)))
        elif type(obj).__name__ == 'MixedMemoryAllocator':
            _write(_runtime_path, dict(event='lmcache_cpu_geometry', bytes=obj.size,
                pinned=obj.buffer.is_pinned(), allocator=type(obj).__name__))

    def _started(code, offset):
        policy = (_policy_path and code.co_name in {'apply_g1_priority', 'take_blocks'}
                  and code.co_filename.endswith('/vllm/v1/core/kv_cache_priority.py'))
        geometry = (_runtime_path and code.co_name == '__init__' and (
            code.co_filename.endswith('/vllm/v1/kv_offload/cpu/spec.py')
            or code.co_filename.endswith('/vllm/v1/kv_offload/cpu/gpu_worker.py')
            or code.co_filename.endswith('/lmcache/v1/memory_management.py')))
        transfers = (_runtime_path and code.co_name == 'get_finished'
                     and code.co_filename.endswith('/vllm/v1/kv_offload/cpu/gpu_worker.py'))
        decode = (_runtime_path and code.co_name == 'from_request_tracker'
                  and code.co_filename.endswith('/lmcache/integration/vllm/vllm_v1_adapter.py'))
        imports = (_runtime_path and (code.co_name == 'build_app' and code.co_filename.endswith('/openai/api_server.py')
                   or code.co_name == '__init__' and code.co_filename.endswith('/core/sched/scheduler.py')))
        if policy or geometry or transfers or decode or imports:
            _monitor.set_local_events(_tool, code, _monitor.events.PY_RETURN)
        return _monitor.DISABLE

    _monitor.register_callback(_tool, _monitor.events.PY_START, _started)
    _monitor.register_callback(_tool, _monitor.events.PY_RETURN, _returned)
    _monitor.set_events(_tool, _monitor.events.PY_START)
