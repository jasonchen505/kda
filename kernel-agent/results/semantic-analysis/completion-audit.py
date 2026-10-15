"""Post-run integrity and failure accounting; no model calls or policy changes."""
import collections,datetime,hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]).absolute();sys.path[:0]=[str(root),str(root/'src')]
from scripts.summarize_semantic import analyze
from scripts.summarize_campaign import digest
folder=root/'results/semantic-study'
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(v,m):
 if not v:raise ValueError(m)
s=analyze(folder);plan=load(folder/'plan.json');state=load(folder/'checkpoint.json')
require(s==load(folder/'summary.json'),'summary not reproducible')
require(s['status']==state['status']=='complete' and s['resolved']==66,'study incomplete')
require(state['allocations']=={'model_calls':118,'output_tokens':241664},'budget changed')
require(s['dispatched_requests']<=118,'dispatch cap exceeded')
launch=load(folder/'launch.json');exited=load(folder/'supervisor-exit.json')
require(launch['plan_sha256']==s['plan_sha256'] and exited['status']=='complete' and exited['exit_code']==0,'supervisor did not complete the declared plan')
require(sha(folder/'supervisor.py')==launch['supervisor_source_sha256'],'supervisor driver changed')
correction=load(root/'results/semantic-analysis-correction.json')
require(correction['changed_file']=='scripts/summarize_semantic.py' and correction['executed_sha256']==plan['source_sha256'][correction['changed_file']] and sha(root/correction['changed_file'])==correction['corrected_sha256'],'analysis-only correction unbound')
for name,expected in plan['source_sha256'].items():
 if name!=correction['changed_file']:require(sha(root/name)==expected,'executed generation source changed: '+name)
original=load(folder/'summary.original-order.json')
require(sha(folder/'summary.original-order.json')==correction['original_summary_sha256'] and {k:v for k,v in original.items()if k!='timeline'}=={k:v for k,v in s.items()if k!='timeline'},'analysis correction changed outcomes')
require(sorted(original['timeline'],key=lambda x:json.dumps(x,sort_keys=True))==sorted(s['timeline'],key=lambda x:json.dumps(x,sort_keys=True)),'analysis correction changed timeline records')
bank=load(folder/'sources/data/semantic-failure-bank.json');oldpath=folder/'sources/results/repair-study/summary.json';old=load(oldpath)
expected={b['block']['id']+'-'+g for b in old['blocks'] for g,a in b['result']['initial'].items() if a.get('source_sha256') and a['evaluation']['status'] in {'numeric_error','compile_error','contract_error','runtime_error','policy_error'}}
require({c['id']for c in bank['cases']}==expected and len(expected)==7,'bank completeness changed')
require(sha(oldpath)==bank['parent_summary_sha256'],'parent summary changed')
for c in bank['cases']:require(sha(folder/'sources'/c['source_path'])==c['source_sha256'],'parent source changed')
statuses,finish,omissions=collections.Counter(),collections.Counter(),collections.Counter()
reviews=collections.Counter();reference_rejections=collections.Counter();schema_errors=collections.Counter();wirecounts=collections.Counter()
initials=[];infra=[];duplicates=0;final_sources=0;known_good_unchanged=0
for b in s['blocks']:
 require(b['status']=='complete' and b['result'],'block not sealed')
 r=b['result'];path=folder/'blocks'/b['block']['id'];unique={}
 initial=r['initial'];initials.append({'block':b['block']['id'],'stratum':b['block']['stratum'],'task_id':b['block']['task_id'],'initial_status':initial['evaluation']['status'],'imported':initial.get('imported',False)})
 if r['infrastructure_stop']:infra.append({'block':b['block']['id'],'stop':r['infrastructure_stop']})
 reviews[r['review']['validation']['status']]+=1
 if not r['review']['validation']['reference']['valid']:
  reasons={p.get('error',p.get('status','unknown'))for p in r['review']['validation']['reference'].get('probes',[])}
  reference_rejections.update(reasons or {r['review']['validation']['status']})
 for arm,a in r['arms'].items():
  require(len(a['model_opportunities'])==2,'repair opportunities changed')
  require(load(path/'final'/f'{arm}.json')==a,'final artifact mismatch')
  final_sources+=bool(a['final']) and 'final_reused_from' not in a
  for t in a['attempts']:
   key=t['request_id'];require(key not in unique or unique[key]==t,'shared initial modified');unique[key]=t
 for t in unique.values():
  require(load(path/'attempts'/f"{t['request_id']}.json")==t,'attempt artifact differs')
  statuses[t['evaluation']['status']]+=1;duplicates+=bool(t.get('duplicate_of'))
  if t['evaluation']['status']=='protocol_error':schema_errors[t['evaluation'].get('message','')[:180]]+=1
 dispatched=0
 for p in (path/'requests').glob('*.json'):
  if p.name.endswith(('.request.json','.response.json')):continue
  receipt=load(p);omissions.update(receipt.get('omitted_sections',[]));wirecounts[receipt['status']]+=1
  if receipt.get('dispatched_utc'):dispatched+=1
  if receipt['status']=='completed':finish[str(receipt.get('finish_reason'))]+=1
 require(dispatched==r['model_calls']<=b['block']['model_call_cap'],'block call accounting changed')
for st in s['strata']:
 for c in st['cells']:require(c['scheduled']==(14 if st['stratum']=='regression'else 8) and c['resolved']==c['scheduled'],'stratum denominator changed')
 for contrast in st['contrasts'].values():require(contrast['interval']['clusters']==(5 if st['stratum']=='regression'else 8),'task clusters changed')
smoke={p.stem:load(p)for p in (root/'results/semantic-smoke').glob('*.json')}
require(smoke['oracle']['status']=='semantic_mechanisms_passed' and smoke['oracle']['oracle_probe_checks']==16,'oracle mechanism check missing')
require(smoke['bad_silu']['status']=='numeric_error' and smoke['good_add']['status']=='correct','real source smoke missing')
require(all(smoke[k]['device_binding']['verified']for k in ['oracle','bad_silu','good_add']),'smoke device mismatch')
value={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'verified','scope':'Post-run integrity and descriptive failure accounting; no new generation or GPU work','plan_sha256':digest(plan),'scheduled_blocks':22,'terminal_arm_outcomes':66,'allocated_calls':118,'dispatched_calls':s['dispatched_requests'],'request_statuses':dict(wirecounts),'unique_code_opportunity_statuses':dict(statuses),'duplicate_search_sources':duplicates,'independent_final_sources':final_sources,'review_validation_statuses':dict(reviews),'reference_rejection_reasons_per_review':dict(reference_rejections),'protocol_errors':dict(schema_errors),'finish_reasons':dict(finish),'omitted_optional_sections':dict(omissions),'initials':initials,'infrastructure_blocks':infra,'strata':s['strata'],'source_bank_selection_verified':True,'generation_policy_sources_match_frozen':True,'analysis_only_exceptions':[correction['changed_file']],'mechanism_smoke_verified':True}
(root/'results/semantic-completion-audit.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:value[k] for k in ['status','terminal_arm_outcomes','dispatched_calls','review_validation_statuses']}))
