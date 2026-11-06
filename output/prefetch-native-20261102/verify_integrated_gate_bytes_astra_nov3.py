"""Integrated gate byte qualification extends independent all-component checks."""
import gzip,json
from pathlib import Path
import verify_selected_bytes_astra_nov3 as base_module
from gpu_gate_receipt_astra_nov3 import verify_gpu_gate
from http_confirmation_receipt_astra_nov3 import verify_http

def verify_bytes(path):
 # Use the qualified current cancellation-aware transport definition throughout.
 base_module.verify_http=verify_http
 r=json.load(gzip.open(path,'rt'));out=base_module.verify_bytes(path);issues=out['issues']
 out['schema']='astra-integrated-gate-byte-qualification-v1'
 http=verify_http(r);out['HTTP_receipt']=http
 if not http['qualified']:issues.append('current_HTTP_receipt_failed')
 candidate=r['args']['arm']=='candidate'
 cfg=r.get('config',{}).get('kv_connector_extra_config',{})
 out['preservation_audit_configured']=cfg.get('preservation_audit')
 out['intrusive_byte_audit_configured']=r['args'].get('audit')
 if candidate and (cfg.get('preservation_audit') is not False or r['args'].get('no_audit') is not True):issues.append('candidate_bounded_mode_not_active')
 gate=verify_gpu_gate(r,'gate' if candidate else 'native');out['gate_receipt']=gate
 if not gate['qualified']:issues.append('gate_activation_or_causal_receipt_failed')
 if candidate:
  rows=gate['fresh_gate_rows']
  if not any(x['skipped_full_GPU'] and x['CPU_ready_before']>0 for x in rows):issues.append('no_positive_ready_full_GPU_skip')
  if not any(x['GPU_index_available'] and x['GPU_cached_prefix_keys']<x['source_keys'] and x['CPU_actual_refreshed']>0 for x in rows):issues.append('no_actual_partial_GPU_fallback_refresh')
 for d,v in out['native_aggregate_reconciliation'].items():
  if v['bytes_match'] is not True or v['counts_match'] is not True:issues.append('native_final_aggregate_not_observed_and_reconciled_'+d)
  v['load_bytes_match']=v['load_delta_bytes']==v['audit_bytes_at_or_after_load_start'] if v['load_delta_bytes'] is not None else None
  v['load_counts_match']=v['load_delta_job_count']==v['audit_jobs_at_or_after_load_start'] if v['load_delta_job_count'] is not None else None
  if v['load_bytes_match'] is not True or v['load_counts_match'] is not True:issues.append('native_load_aggregate_not_observed_and_reconciled_'+d)
 if r.get('owned_descendants_remaining')!=[]:issues.append('cleanup_receipt_missing_or_nonempty')
 out['qualified']=not issues
 return out
