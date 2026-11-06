from pathlib import Path
import shutil
H=Path(__file__).resolve().parent;S=H/'sources/nov3-gpu-coverage/semantic_offload';D=H/'sources/nov3-gpu-gate/semantic_offload';D.mkdir(parents=True,exist_ok=False)
for p in S.glob('*.py'):shutil.copyfile(p,D/p.name)
p=D/'connector.py';s=p.read_text();needle='        if gpu_observation and (\n';assert s.count(needle)==1
s=s.replace(needle,'''        gpu_gate = extra_config.get("wait_hint_cpu_retention_gpu_gate", False)
        if not isinstance(gpu_gate, bool):
            raise TypeError("wait_hint_cpu_retention_gpu_gate must be a JSON boolean")
        if (gpu_observation or gpu_gate) and (
''').replace('        self._wait_hint_gpu_prefix_observation = gpu_observation\n','        self._wait_hint_gpu_prefix_observation = gpu_observation\n        self._wait_hint_cpu_retention_gpu_gate = gpu_gate\n',1)
needle='            ready_count = refresh(ready) if callable(refresh) else 0\n';assert s.count(needle)==1
s=s.replace(needle,'''            gate_details = {}
            refresh_keys = ready
            if getattr(self, "_wait_hint_cpu_retention_gpu_gate", False):
                gpu_count, _, gpu_available = _local_gpu_prefix(self.gpu_block_pool, source.keys)
                skip = gpu_available and gpu_count == len(source.keys)
                gate_details = {
                    "cpu_retention_gpu_gate_policy": "skip_CPU_advice_while_full_GPU_prefix_cached",
                    "cpu_retention_gpu_index_available": gpu_available,
                    "cpu_retention_gpu_cached_prefix_keys": gpu_count if gpu_available else None,
                    "cpu_retention_source_keys": len(source.keys),
                    "cpu_retention_ready_prefix_keys": len(ready),
                    "cpu_retention_skipped_ready_prefix_keys": len(ready) if skip else 0,
                    "cpu_retention_skipped_full_gpu": skip,
                }
                if skip:
                    refresh_keys = []
            ready_count = refresh(refresh_keys) if callable(refresh) else 0
''')
needle='                **coverage_details,\n';assert s.count(needle)==1;s=s.replace(needle,needle+'                **gate_details,\n');p.write_text(s)
s=(H/'run_live_gpu_coverage_nov3.py').read_text();needle=" if a.observe_gpu_prefix:\n";assert s.count(needle)==1;s=s.replace(needle," if a.gpu_gated_cpu_retention and a.arm!='candidate':raise ValueError('GPU-gated CPU retention is candidate-only')\n"+needle,1)
needle=" if a.observe_gpu_prefix:cfg['kv_connector_extra_config']['wait_hint_gpu_prefix_observation']=True\n";assert s.count(needle)==1;s=s.replace(needle,needle+" if a.gpu_gated_cpu_retention:cfg['kv_connector_extra_config']['wait_hint_cpu_retention_gpu_gate']=True\n")
s=s.replace("ap=argparse.ArgumentParser();","ap=argparse.ArgumentParser();ap.add_argument('--gpu-gated-cpu-retention',action='store_true');",1)
p=H/'run_live_gpu_gate_nov3.py';assert not p.exists();p.write_text(s)
print('Isolated read-only GPU-state gate prepared; no frozen files changed, no tests/GPU run yet')
