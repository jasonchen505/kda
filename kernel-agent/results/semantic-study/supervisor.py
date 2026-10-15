import json,sys,traceback
from pathlib import Path
sys.path[:0]=[str(Path.cwd()),str(Path.cwd()/'src')]
from scripts.run_semantic_study import SemanticStudy,make_plan
from kernel_agent.resource_lock import acquire
from kernel_agent.campaign import atomic_json
from kernel_agent.model_pool import utc
out=Path('/data/home/chenyizhou/kernel-agent-semantic-experiment-v1')
receipt={'started_utc':utc()}
try:
 with acquire(out/'supervisor.lock',timeout=0):
  study=SemanticStudy(out,'/data/home/chenyizhou/kda-humanize-portfolio/.venv/bin/python',plan=make_plan())
  study.run()
 receipt.update(status=study.state['status'],exit_code=0)
except BaseException as error:
 receipt.update(status='supervisor_error',exit_code=1,error=str(error),traceback=traceback.format_exc())
finally:
 receipt['finished_utc']=utc();atomic_json(out/'supervisor-exit.json',receipt)
sys.exit(receipt['exit_code'])
