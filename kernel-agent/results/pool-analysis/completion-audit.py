import datetime,hashlib,json
from pathlib import Path
root=Path('/Users/sskchen/Documents/ChatGPT/kernel agent/kernel-agent')
folder=root/'results/pool-study';before_root=root/'results/pool-before-handoff'
s=json.loads((folder/'summary.json').read_text());before=json.loads((folder/'handoff/before.json').read_text())
assert s['status']=='complete'and s['resolved']==24
exits=json.loads((folder/'workers-exit.json').read_text());assert exits=={'18118':0,'54575':0,'60953':0}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(folder/'plan.json')==sha(before_root/'plan.json')
unchanged_results={};unchanged_final={};old_ids=[];reservations={'duration_seconds':0,'output_tokens':0}
for worker in json.loads((folder/'plan.json').read_text())['workers']:
 name=f"p{worker['port']}";old=before[name]
 state=json.loads((folder/'workers'/name/'generation.json').read_text())
 by_id={r['job']['id']:r for r in state['jobs']}
 assert all(r['status']in {'completed','failed','interrupted_unknown'}for r in state['jobs'])
 for job_id,expected in old['committed_results'].items():
  assert by_id[job_id]['result_sha256']==expected
  unchanged_results[job_id]=expected;old_ids.append(job_id)
  old_wire=before_root/'workers'/name/'wire'/job_id;new_wire=folder/'workers'/name/'wire'/job_id
  assert {str(p.relative_to(old_wire))for p in old_wire.rglob('*.json')}=={str(p.relative_to(new_wire))for p in new_wire.rglob('*.json')}
  for p in (before_root/'workers'/name/'final'/job_id).rglob('*'):
   if p.is_file():
    rel=p.relative_to(before_root);assert sha(p)==sha(folder/rel);unchanged_final[str(rel)]=sha(p)
 for k in reservations:reservations[k]+=state['reservations'][k]
assert len(unchanged_results)==15 and reservations=={'duration_seconds':21600,'output_tokens':480000}
for name,expected in before['request_file_sha256'].items():assert sha(folder/name)==expected
owned_checkpoint=folder/'workers/p18118/generation.json'
assert sha(owned_checkpoint)==before['p18118']['checkpoint_sha256']
value={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'verified','scheduled':24,'resolved':24,'worker_exits':exits,'plan_file_sha256':sha(folder/'plan.json'),'plan_semantic_sha256':s['plan_sha256'],'preserved_result_sha256':unchanged_results,'unchanged_old_final_files_sha256':unchanged_final,'old_request_admission_files_unchanged':len(before['request_file_sha256']),'additional_request_files_for_old_jobs':0,'owned_completed_checkpoint_sha256':sha(owned_checkpoint),'unchanged_owned_jobs':8,'newly_started_jobs':9,'budget_reservations':reservations,'outcome_types':s['outcome_types'],'pooled_contrasts_suppressed':s['pooled_contrasts_suppressed'],'cross_restart_pairs':s['cross_restart_pairs'],'scope':'Post-completion file/ledger verification; no model or GPU execution. Unknown responses remain unknown; no claim of exactly-once external execution or continuous model processes.'}
(root/'results/pool-completion-audit.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps({k:v for k,v in value.items()if k not in ['preserved_result_sha256','unchanged_old_final_files_sha256']}))
