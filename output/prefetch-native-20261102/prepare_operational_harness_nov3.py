"""Create new operational-only files; never modify qualified campaign inputs."""
from pathlib import Path
H = Path(__file__).resolve().parent
ROOT = H.parents[1]


def replace_once(text, old, new):
    assert text.count(old) == 1, (old[:100], text.count(old))
    return text.replace(old, new)


def create(path, text):
    if path.exists():
        if path.read_text() == text:
            return
        # These four generated H-only files are owned by this preparation task.
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


work = (H/'live_return_priority_workload_nov3.py').read_text()
work = replace_once(work, 'from return_priority_nov3 import gateway_priority', 'from semantic_offload.request_priority import gateway_priority')
work = replace_once(work, "sid=session['id']", "sid=a.application_namespace+'::'+session['id']")
work = replace_once(work, "gateway_priority(submitted_ns,result['load_start_ns'],turn,a.return_priority_ms)", "gateway_priority(submitted_ns,a.gateway_origin_ns,turn,a.return_priority_ms)")
work = replace_once(work, "'priority_origin_ns':result['load_start_ns']", "'priority_origin_ns':a.gateway_origin_ns")
work = replace_once(work, "'priority_clock':'single_gateway_host_monotonic_relative_to_load_origin'", "'priority_clock':'single_gateway_host_monotonic_relative_to_persistent_server_case_origin'")
create(H/'live_operational_workload_nov3.py', work)

hook = (H/'hooks-live-lean/sitecustomize.py').read_text()
hook = replace_once(hook, "H=Path(__file__).resolve().parents[1]", "H=Path(__file__).resolve().parents[1]\nsys.path.insert(0,str(H))")
hook = replace_once(hook, "original_schedule=m.Scheduler.schedule", "from operational_abort_nov3 import install\n        install(m.Scheduler)\n        original_schedule=m.Scheduler.schedule")
hook = replace_once(hook, 'def state():return {', "def state():\n    from operational_state_nov3 import snapshot\n    from operational_abort_nov3 import snapshot as abort_snapshot\n    return {'operational_abort_transitions':abort_snapshot(),'operational_scalar_state':snapshot(SCHEDULER,MANAGER,NATIVE_SCHEDULER),")
hook = replace_once(hook, "return {**state(),'prefix_mean_scoring'", "from operational_state_nov3 import snapshot\n            return {**state(),'operational_worker_state':snapshot(worker=w),'prefix_mean_scoring'")
create(H/'hooks-operational/sitecustomize.py', hook)

driver = (H/'run_live_return_priority_nov3.py').read_text()
driver = driver.replace('live_return_priority_workload_nov3', 'live_operational_workload_nov3')
driver = driver.replace('hooks-live-lean', 'hooks-operational')
driver = driver.replace("H/'return_priority_nov3.py'", "H/'operational_waves_nov3.py',H/'operational_state_nov3.py',H/'operational_abort_nov3.py',H/'operational_cancel_probe_nov3.py'")
driver = replace_once(driver, "if a.no_audit:cfg['kv_connector_extra_config']['preservation_audit']=False", "if a.no_audit:cfg['kv_connector_extra_config']['preservation_audit']=False\n if a.arm=='candidate':cfg['kv_connector_extra_config']['observed_session_registry_size']=a.registry_limit")
driver = replace_once(driver, "('hooks.py',H/'hooks-operational/sitecustomize.py')", "('hooks.py',H/'hooks-operational/sitecustomize.py'),('public_priority_helper.py',ROOT/'semantic_offload/request_priority.py'),('wave_driver.py',H/'operational_waves_nov3.py'),('scalar_telemetry.py',H/'operational_state_nov3.py'),('abort_observer.py',H/'operational_abort_nov3.py'),('cancel_probe.py',H/'operational_cancel_probe_nov3.py')")
driver = replace_once(driver, "from live_operational_workload_nov3 import exercise", "from operational_waves_nov3 import exercise")
driver = replace_once(driver, "'performance_eligible':not (a.audit or a.quality or a.boundary_diagnostic or a.feasibility_only or a.observe_native_queries)", "'performance_eligible':False,'operational_qualification':True,'performance_ineligibility_reason':'Persistent-engine operational qualification with explicit boundary exports; no confirmation inference'")
driver = replace_once(driver, "if a.service_profile.startswith('repository_triage_') and not a.support_profile:", "if a.qualification or a.audit or a.open_rate or a.abandon_every:raise ValueError('Operational wave runner excludes unrelated modes')\n if a.registry_limit!=4096 and a.arm!='candidate':raise ValueError('Accelerated registry cap case is candidate-only correctness qualification')\n if a.return_priority_ms!=8000 or (a.arm=='candidate' and not a.no_audit) or a.clients!=24 or a.turns!=3:raise ValueError('Operational plan requires qualified credit, bounded mode, 24 sessions and three turns')\n if a.service_profile.startswith('repository_triage_') and not a.support_profile:")
driver = replace_once(driver, "ap=argparse.ArgumentParser();", "ap=argparse.ArgumentParser();ap.add_argument('--wave-release-interval-s',type=float,default=0);ap.add_argument('--cancel-probe',action='store_true');ap.add_argument('--waves',type=int,required=True);ap.add_argument('--wave-seeds',required=True);ap.add_argument('--wave-settle-s',type=float,default=2.0);ap.add_argument('--registry-limit',type=int,choices=[128,4096],default=4096);")
driver = replace_once(driver, "result['native_scheduler_source']=", "result['public_priority_helper']={'path':str(ROOT/'semantic_offload/request_priority.py'),'sha256':sha(ROOT/'semantic_offload/request_priority.py'),'archive':'public_priority_helper.py'}\n result['native_scheduler_source']=")
create(H/'run_live_operational_nov3.py', driver)
