from pathlib import Path
import shutil,json,hashlib
H=Path(__file__).resolve().parent;S=H/'sources/nov3-gpu-bridge/semantic_offload';D=H/'sources/nov3-gpu-coverage/semantic_offload';D.mkdir(parents=True,exist_ok=False)
for p in S.glob('*.py'):shutil.copyfile(p,D/p.name)
p=D/'connector.py';s=p.read_text();needle='        self._wait_hint_cpu_retention_gpu_aware = gpu_aware_retention\n';assert s.count(needle)==1
extra='''        gpu_observation = extra_config.get("wait_hint_gpu_prefix_observation", False)
        if not isinstance(gpu_observation, bool):
            raise TypeError("wait_hint_gpu_prefix_observation must be a JSON boolean")
        if gpu_observation and (
            not retain_only or not self._wait_hint_native_adoption_layout_supported
            or self.config.num_workers != 1
            or len(self.config.kv_group_configs) != 1
            or config_blocks_per_chunk(self.config) != 1
            or self.config.kv_group_configs[0].group_idx != 0
            or group_config_tokens_per_chunk(self.config.kv_group_configs[0])
            != group_config_tokens_per_block(self.config.kv_group_configs[0])
            or getattr(self.config.kv_group_configs[0], "is_eagle_group", False)
        ):
            raise ValueError("GPU prefix observation requires dense TP1 single-block CPU retention layout")
        self._wait_hint_gpu_prefix_observation = gpu_observation
'''
s=s.replace(needle,needle+extra)
needle='            refresh = getattr(self.manager, "refresh_wait_hint_cpu_recency", None)\n';assert s.count(needle)==1
s=s.replace(needle,'''            coverage_details = {}
            if getattr(self, "_wait_hint_gpu_prefix_observation", False):
                from semantic_offload.gpu_prefix_coverage import describe_gpu_prefix
                coverage_details = describe_gpu_prefix(self.gpu_block_pool, source.keys)
'''+needle)
needle='                **bridge_details,\n';assert s.count(needle)==1;s=s.replace(needle,needle+'                **coverage_details,\n');p.write_text(s)
s=(H/'run_live_bridge_nov3.py').read_text();s=s.replace(' if a.gpu_aware_retention and'," if a.observe_gpu_prefix:\n  if a.arm!='candidate':raise ValueError('GPU observation is candidate-only')\n  a.feasibility_only=True\n if a.gpu_aware_retention and",1)
needle=" if a.gpu_aware_retention:cfg['kv_connector_extra_config']['wait_hint_cpu_retention_gpu_aware']=True\n";assert s.count(needle)==1;s=s.replace(needle,needle+" if a.observe_gpu_prefix:cfg['kv_connector_extra_config']['wait_hint_gpu_prefix_observation']=True\n")
s=s.replace("ap=argparse.ArgumentParser();","ap=argparse.ArgumentParser();ap.add_argument('--observe-gpu-prefix',action='store_true');",1)
p=H/'run_live_gpu_coverage_nov3.py';assert not p.exists();p.write_text(s)
print('Prepared isolated observation-only source and performance-ineligible driver; helper/tests pending')
