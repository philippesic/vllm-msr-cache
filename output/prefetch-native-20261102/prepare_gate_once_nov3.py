from pathlib import Path
H=Path(__file__).resolve().parent
s=(H/'run_live_integrated_gpu_gate_nov3.py').read_text()
s=s.replace(" if a.gpu_gated_cpu_retention and a.arm!='candidate':", " if a.stop_after_first_cpu_touch and not a.gpu_gated_cpu_retention:raise ValueError('One-touch policy requires explicit GPU gate')\n if a.gpu_gated_cpu_retention and a.arm!='candidate':",1)
s=s.replace(" out=H/'runs'/a.name;", " workload_module='live_once_service_workload_nov3' if a.stop_after_first_cpu_touch else 'live_service_workload_nov3'\n out=H/'runs'/a.name;",1)
s=s.replace("('workload.py',H/'live_service_workload_nov3.py')", "('workload.py',H/(workload_module+'.py'))")
s=s.replace("    from live_service_workload_nov3 import exercise\n", "    import importlib\n    exercise=importlib.import_module(workload_module).exercise\n")
s=s.replace("ap=argparse.ArgumentParser();", "ap=argparse.ArgumentParser();ap.add_argument('--stop-after-first-cpu-touch',action='store_true');",1)
p=H/'run_live_gate_once_nov3.py';assert not p.exists();p.write_text(s)
s=(H/'qualify_once_ready_nov3.py').read_text().replace('live-nov3-once-ready-cpu-qualification.json','live-nov3-gate-once-cpu-qualification.json').replace("'once-ready-producer-CPU-qualification-v1'","'GPU-gate-once-positive-producer-CPU-qualification-v1'")
s=s.replace("('first_positive',[ack(2)],True),", "('GPU_full_skip_then_partial_positive',[dict(ack(0),cpu_retention_skipped_full_gpu=True,cpu_retention_ready_prefix_keys=5),dict(ack(2),cpu_retention_skipped_full_gpu=False)],True),('repeated_full_skips_then_zero_until_return',[dict(ack(0),cpu_retention_skipped_full_gpu=True),dict(ack(0),cpu_retention_skipped_full_gpu=True),dict(ack(0),cpu_retention_skipped_full_gpu=False)],False),('first_positive',[ack(2)],True),")
p=H/'qualify_gate_once_nov3.py';assert not p.exists();p.write_text(s)
print('New isolated runner and 8-case CPU qualification prepared; server source unchanged.')
