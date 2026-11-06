from pathlib import Path
import shutil
H=Path(__file__).resolve().parent;S=H/'sources/nov3-gpu-coverage/semantic_offload';D=H/'sources/nov3-gpu-recency/semantic_offload';D.mkdir(parents=True,exist_ok=False)
for p in S.glob('*.py'):shutil.copyfile(p,D/p.name)
p=D/'connector.py';s=p.read_text();needle='        if gpu_observation and (\n';assert s.count(needle)==1
s=s.replace(needle,'''        gpu_recency = extra_config.get("wait_hint_gpu_cache_recency", False)
        if not isinstance(gpu_recency, bool):
            raise TypeError("wait_hint_gpu_cache_recency must be a JSON boolean")
        if (gpu_observation or gpu_recency) and (
''');s=s.replace('        self._wait_hint_gpu_prefix_observation = gpu_observation\n','        self._wait_hint_gpu_prefix_observation = gpu_observation\n        self._wait_hint_gpu_cache_recency = gpu_recency\n',1)
needle='            if getattr(self, "_wait_hint_gpu_prefix_observation", False):\n';assert s.count(needle)==1
s=s.replace(needle,'''            if getattr(self, "_wait_hint_gpu_cache_recency", False):
                from semantic_offload.gpu_cache_recency import refresh_gpu_prefix_recency
                coverage_details = refresh_gpu_prefix_recency(self.gpu_block_pool, source.keys)
            elif getattr(self, "_wait_hint_gpu_prefix_observation", False):
''');p.write_text(s)
s=(H/'run_live_gpu_coverage_nov3.py').read_text();needle=" if a.observe_gpu_prefix:\n";assert s.count(needle)==1;s=s.replace(needle," if a.gpu_recency and a.arm!='candidate':raise ValueError('GPU cache recency is candidate-only')\n"+needle,1)
needle=" if a.observe_gpu_prefix:cfg['kv_connector_extra_config']['wait_hint_gpu_prefix_observation']=True\n";assert s.count(needle)==1;s=s.replace(needle,needle+" if a.gpu_recency:cfg['kv_connector_extra_config']['wait_hint_gpu_cache_recency']=True\n")
s=s.replace("ap=argparse.ArgumentParser();","ap=argparse.ArgumentParser();ap.add_argument('--gpu-recency',action='store_true');",1)
# Operational queue validation is part of the policy, not an external observer.
needle=" result['measurement_configuration']={'lean':True,";assert s.count(needle)==1
s=s.replace(needle," result['gpu_recency_operational_validation']={'configured':a.gpu_recency,'scope':'Policy uses a bounded native free-queue membership walk per valid hint; no external residency observer or GPU allocation.'}\n"+needle)
p=H/'run_live_gpu_recency_nov3.py';assert not p.exists();p.write_text(s)
print('Prepared separate opt-in GPU eviction-policy source; mutation helper/tests pending; no frozen dependencies changed')
