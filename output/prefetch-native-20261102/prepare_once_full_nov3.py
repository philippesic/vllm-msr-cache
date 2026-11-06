from pathlib import Path
H=Path(__file__).resolve().parent
s=(H/'live_once_service_workload_nov3.py').read_text()
s=s.replace("and ready>0):","and ready==len(observed)//16 and ready>0):").replace('first_positive_cpu_recency_ack','first_complete_cpu_recency_ack').replace('retry_while_pending_until_first_positive_cpu_recency_ack','scheduled_advice_while_pending_until_first_complete_cpu_recency_ack')
p=H/'live_once_full_service_workload_nov3.py';assert not p.exists();p.write_text(s)
s=(H/'run_live_once_ready_nov3.py').read_text().replace('live_once_service_workload_nov3','live_once_full_service_workload_nov3')
p=H/'run_live_once_full_nov3.py';assert not p.exists();p.write_text(s)
s=(H/'qualify_once_ready_nov3.py').read_text().replace('live_once_service_workload_nov3','live_once_full_service_workload_nov3').replace('first_positive_cpu_recency_ack','first_complete_cpu_recency_ack').replace("'first_positive',[ack(2)],True","'first_complete',[ack(1)],True").replace("'zero_then_ready',[ack(0),ack(3)],True","'zero_partial_then_complete',[ack(0),ack(0),ack(1)],True").replace("'observed':[1]*16","'observed':[1]*16").replace('once-ready-producer-CPU-qualification-v1','once-complete-producer-CPU-qualification-v1').replace('live-nov3-once-ready-cpu-qualification','live-nov3-once-full-cpu-qualification')
p=H/'qualify_once_full_nov3.py';assert not p.exists();p.write_text(s)
