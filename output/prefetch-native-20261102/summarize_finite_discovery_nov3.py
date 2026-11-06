"""Source-pinned descriptive report; no confidence intervals from discovery."""
import collections,gzip,hashlib,json,re,statistics
from pathlib import Path
from analyze_live_nov3 import metrics
H=Path(__file__).resolve().parent

def summarize(path):
 path=path.resolve()
 r=json.load(gzip.open(path,"rt"));row=metrics(path)
 row.update(raw_path=str(path.relative_to(H)),raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),args=r.get("args",{}),source_pins=r.get("source_pins",{}),runtime_pins=r.get("runtime_pins",{}),driver_pins=r.get("pins",{}),service_prompt_profile=r.get("service_prompt_profile"),workload=r.get("workload",{}))
 s=r.get("postload_export",{}).get("scheduler",{})
 row["scheduler_counts"]=s.get("counts",{})
 row["scheduler_export_available"]=bool(r.get("postload_export",{}).get("scheduler"))
 row["memory"]={"serving_sampled_nvml_peak_mib":row["peak_nvml_mib"],"startup_sampled_nvml_peak_mib":max((x.get("gpu",{}).get("memory_used_bytes",0) for x in r.get("startup_resource_samples",[])),default=0)/2**20 if r.get("startup_resource_samples") else None,"startup_measurement_scope":r.get("startup_measurement_scope","not sampled by frozen discovery driver"),"before_launch_nvml_used_mib":r.get("gpu_before",{}).get("used_bytes",0)/2**20,"worker_load_cuda":[{k:v for k,v in w.items() if k in ["gpu_allocated_bytes","gpu_reserved_bytes","load_peak_allocated_bytes","load_peak_reserved_bytes"]} for w in r.get("postload_export",{}).get("workers",[])]}
 def transfer_values(text):
  totals=collections.Counter()
  for line in text.splitlines():
   if line.startswith("vllm:kv_offload_total_bytes_total{"):
    match=re.search(r'transfer_type="([^"]+)"',line)
    if match:totals[match.group(1)]+=float(line.rsplit(" ",1)[1])
  return totals
 before=transfer_values(r.get("metrics_initial",""));after=transfer_values(r.get("metrics_final",""))
 row["load_transfer_bytes"]={k:after[k]-before[k] for k in set(before)|set(after)} if "metrics_initial" in r and "metrics_final" in r else None
 row["retained_diagnostic_samples"]=len(s.get("samples",[]))
 row["gpu_residency_scans"]=sum("gpu_hashed" in x for x in s.get("samples",[]))
 row["hint_acknowledgements"]=dict(collections.Counter(x.get("response",{}).get("reason",x.get("error","unknown")) for x in r.get("hints",[])))
 lengths=[len(x.get("output_token_ids",[])) for x in r.get("requests",[])]
 row["output_lengths"]={"median":statistics.median(lengths),"max":max(lengths)} if lengths else {}
 row["longest_responses"]=[{"index":x["index"],"turn":x["turn"],"tokens":len(x.get("output_token_ids",[])),"finish_reasons":x.get("finish_reasons"),"end_offset_s":(x["request_end_ns"]-r["load_start_ns"])/1e9} for x in sorted(r.get("requests",[]),key=lambda x:len(x.get("output_token_ids",[])),reverse=True)[:3]]
 row["first_prompt_digests"]={str(x["index"]):hashlib.sha256(json.dumps(x["payload"]["prompt"]).encode()).hexdigest() for x in r.get("requests",[]) if x.get("turn")==0}
 row["unique_source_dirs"]=r.get("args",{}).get("source_dir")
 return row

def main():
 progress=json.loads((H/"live-nov3-followup-progress.json").read_text())
 continuation=H/"live-nov3-discovery-completion-progress.json"
 if continuation.exists():progress.extend(json.loads(continuation.read_text()))
 names=["live-nov3-"+x["name"] for x in progress]
 rows={name:summarize(H/"runs"/name/"raw.json.gz") for name in names if (H/"runs"/name/"raw.json.gz").exists()}
 pairs=[]
 for label,a,b in [
 ("graph_a8_cpu4096","support-fast-graph","support-native-graph"),
 ("graph_a16_cpu4096","support-fast-graph-a16","support-native-graph-a16"),
 ("graph_a16_cpu1024","support-fast-graph-a16-cpu1024","support-native-graph-a16-cpu1024"),
 ("eager_a16_cpu1024","support-fast-eager-a16-cpu1024","support-native-eager-a16-cpu1024"),
 ("graph_gpu1024_cpu1024","support-fast-graph-gpu1024-cpu1024","support-native-graph-gpu1024-cpu1024"),
 ("graph_gpu1024_cpu1024_native_retry1_original_failure_retained","support-fast-graph-gpu1024-cpu1024","support-native-graph-gpu1024-cpu1024-retry1"),
 ("graph_a16_matched_gpu2048_cpu1024","support-fast-graph-a16-cpu1024","support-native-graph-matched2048-cpu1024"),
 ("graph_a16_hint1000_matched_gpu2048_cpu1024","support-fast-graph-a16-cpu1024-h1000","support-native-graph-matched2048-cpu1024"),
 ]:
  a=rows.get("live-nov3-"+a);b=rows.get("live-nov3-"+b)
  if not a or not b:continue
  p={"comparison":label,"production_confirmation_eligible":False,"scope":"asymmetric diagnostic instrumentation and stock prerelease native; descriptive discovery only","candidate":a["name"],"native":b["name"],"both_workflows_eligible":bool(a["performance_eligible"] and b["performance_eligible"]),"first_prompts_equal":a["first_prompt_digests"]==b["first_prompt_digests"]}
  if p["both_workflows_eligible"]:
   p.update(workflow_rate_gain_fraction=b["load_s"]/a["load_s"]-1,token_rate_gain_fraction=a["tokens_s"]/b["tokens_s"]-1,output_token_ratio=a["tokens"]/b["tokens"],vram_saving_mib=b["peak_nvml_mib"]-a["peak_nvml_mib"],mean_ttft_change_ms=a["ttft_ms"]["mean"]-b["ttft_ms"]["mean"],p95_ttft_change_ms=a["ttft_ms"]["p95"]-b["ttft_ms"]["p95"],mean_e2e_change_ms=a["e2e_ms"]["mean"]-b["e2e_ms"]["mean"])
  pairs.append(p)
 report={"schema":"nov3-finite-discovery-report-v1","status":"descriptive_discovery_only","limitations":["One partition, one workflow per configuration; no statistical inference.","Natural outputs and later prompts vary, so token rate and workflow rate measure different work.","Asymmetric diagnostic GPU residency scans add candidate overhead; use lean symmetric hooks for confirmation.","Concise support system prompt is an explicit service profile; original unconstrained chat/agent failures remain separate negative evidence.","Native ARC uses a common task-local runtime overlay, not an unpatched wheel.","Stock native final-token store boundary differs from the extension clamp; normalized controls are required for confirmation.","Original native GPU1024 run failed on a client disconnect; one whole-workflow retry is reported separately without erasing that failure.","No quality or task-success rating established."],"progress":progress,"rows":rows,"comparisons":pairs}
 (H/"live-nov3-finite-discovery-report.json").write_text(json.dumps(report,indent=2)+"\n")
 print(json.dumps({"runs":len(rows),"comparisons":pairs},indent=2))
if __name__=="__main__":main()
