"""Reconstruct completion and descriptive failures; zero model/GPU work."""
import collections,datetime,hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]).absolute();sys.path[:0]=[str(root),str(root/'src')]
from scripts.summarize_operational import analyze
from scripts.summarize_campaign import digest
folder=root/'results/operational-study'
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(v,m):
 if not v:raise ValueError(m)
s=analyze(folder);plan=load(folder/'plan.json');state=load(folder/'checkpoint.json')
require(s==load(folder/'summary.json'),'summary not reproducible')
require(s['status']==state['status']=='complete' and s['resolved']==44,'study incomplete')
require(state['allocations']=={'model_calls':96,'output_tokens':196608},'allocation changed')
require(s['dispatched_requests']<=96,'physical cap exceeded')
launch=load(folder/'launch.json');exited=load(folder/'supervisor-exit.json')
require(launch['plan_sha256']==s['plan_sha256'] and exited['status']=='complete' and exited['exit_code']==0,'supervisor did not finish declared plan')
require(sha(folder/'supervisor.py')==launch['supervisor_source_sha256'],'supervisor source changed')
for name,expected in plan['source_sha256'].items():require(sha(root/name)==expected,'executed source changed: '+name)
bank=load(folder/'sources/data/semantic-failure-bank.json');oldpath=folder/'sources/results/repair-study/summary.json';old=load(oldpath)
expected={b['block']['id']+'-'+g for b in old['blocks'] for g,a in b['result']['initial'].items() if a.get('source_sha256') and a['evaluation']['status'] in {'numeric_error','compile_error','contract_error','runtime_error','policy_error'}}
require({c['id']for c in bank['cases']}==expected and len(expected)==7,'incomplete failure bank')
require(sha(oldpath)==bank['parent_summary_sha256'],'parent summary changed')
for c in bank['cases']:require(sha(folder/'sources'/c['source_path'])==c['source_sha256'],'parent source changed')
statuses,finish,omissions,wirecounts,patcherrors=collections.Counter(),collections.Counter(),collections.Counter(),collections.Counter(),collections.Counter()
initials=[];infra=[];duplicates=0;final_sources=0;retained_after_rejection=[];repair_pairs=[]
for b in s['blocks']:
 require(b['status']=='complete' and b['result'],'unsealed block')
 r=b['result'];path=folder/'blocks'/b['block']['id'];unique={}
 initials.append({'block':b['block']['id'],'stratum':b['block']['stratum'],'task_id':b['block']['task_id'],'initial_status':r['initial']['evaluation']['status'],'imported':r['initial'].get('imported',False)})
 if r['infrastructure_stop'] or any(x['infrastructure']for x in s['rows']if x['block']==b['block']['id']):infra.append({'block':b['block']['id'],'stop':r['infrastructure_stop'],'final_statuses':{a:v['final']['status']if v['final']else None for a,v in r['arms'].items()}})
 for arm,a in r['arms'].items():
  require(len(a['model_opportunities'])==2 and load(path/'final'/f'{arm}.json')==a,'opportunity/final artifact mismatch')
  final_sources+=bool(a['final']) and 'final_reused_from' not in a
  for t in a['attempts']:
   key=t['request_id'];require(key not in unique or unique[key]==t,'shared initial changed');unique[key]=t
  if arm=='operational_patch':
   for first,second in zip(a['attempts'],a['attempts'][1:]):
    if first['evaluation']['status']=='patch_error':retained_after_rejection.append({'block':b['block']['id'],'rejected':first['request_id'],'next':second['request_id'],'base_sha256':second['parent_sha256'],'next_status':second['evaluation']['status'],'final_correct':a['correct']})
 for t in unique.values():
  require(load(path/'attempts'/f"{t['request_id']}.json")==t,'attempt artifact differs')
  statuses[t['evaluation']['status']]+=1;duplicates+=bool(t.get('duplicate_of'))
  if t['evaluation']['status']=='patch_error':patcherrors[t['evaluation'].get('message','')]+=1
 dispatched=0
 for p in sorted((path/'requests').glob('*.json')):
  if p.name.endswith(('.request.json','.response.json')):continue
  receipt=load(p);omissions.update(receipt.get('omitted_sections',[]));wirecounts[receipt['status']]+=1
  dispatched+=bool(receipt.get('dispatched_utc'))
  if receipt['status']=='completed':finish[str(receipt.get('finish_reason'))]+=1
 require(dispatched==r['model_calls']<=b['block']['model_call_cap'],'call accounting changed')
 repair_pairs.append({'block':b['block'],'initial_status':r['initial']['evaluation']['status'],'rewrite_correct':r['arms']['rewrite']['correct'],'patch_correct':r['arms']['operational_patch']['correct'],'selected':{a:v['selected_sha256']for a,v in r['arms'].items()}})
for st in s['strata']:
 for c in st['cells']:require(c['scheduled']==(14 if st['stratum']=='regression'else 8) and c['resolved']==c['scheduled'],'stratum denominator changed')
 require(st['contrasts']['interval']['clusters']==(5 if st['stratum']=='regression'else 4),'clusters changed')
smoke={p.stem:load(p)for p in (root/'results/operational-smoke').glob('*.json')}
require(smoke['oracles']['status']=='transfer_oracles_passed' and smoke['oracles']['independent_checks']==32,'independent oracle missing')
for task,route in [('silu_mul','algebraic_relation'),('transpose','layout')]:require(smoke[task]['status']=='numeric_error' and smoke[task]['diagnosis']['route']==route,'real bad-source routing missing')
require(all(smoke[k]['device_binding']['verified']for k in ['oracles','silu_mul','transpose']),'smoke physical device mismatch')
require(smoke['context']['status']=='passed' and len(smoke['context']['rows'])==6,'common context preflight missing')
value={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'verified','scope':'Post-run integrity and descriptive failure accounting; zero additional model/GPU work','plan_sha256':digest(plan),'scheduled_blocks':22,'terminal_arm_outcomes':44,'allocated_calls':96,'dispatched_calls':s['dispatched_requests'],'request_statuses':dict(wirecounts),'unique_code_opportunity_statuses':dict(statuses),'duplicate_search_sources':duplicates,'independent_final_sources':final_sources,'patch_statuses':s['patch_statuses'],'patch_error_reasons':dict(patcherrors),'retained_after_rejection':retained_after_rejection,'finish_reasons':dict(finish),'omitted_optional_sections':dict(omissions),'initials':initials,'infrastructure_blocks':infra,'strata':s['strata'],'paired_outcomes':repair_pairs,'source_bank_selection_verified':True,'generation_policy_sources_match_frozen':True,'analysis_only_exceptions':[],'mechanism_smoke_verified':True}
(root/'results/operational-completion-audit.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:value[k]for k in ['status','terminal_arm_outcomes','dispatched_calls','patch_statuses']}))
