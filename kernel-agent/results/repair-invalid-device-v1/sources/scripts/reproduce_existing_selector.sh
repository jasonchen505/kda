#!/usr/bin/env bash
# Reuse an idle owner-provided local base service without modifying its processes.
set -euo pipefail
TASK_ROOT=$(cd "$(dirname "$0")/.." && pwd)
HUMANIZE_PYTHON=${HUMANIZE_PYTHON:-/data/home/chenyizhou/humanize/.venv/bin/python}
SERVICE_PORT=${1:-8016}
SERVICE_PID=${2:-3097365}
export PATH="/data/home/chenyizhou/.local/bin:$PATH"
export PYTHONPATH="$TASK_ROOT:$TASK_ROOT/src"
cd "$TASK_ROOT"
mkdir -p runs results
test ! -e results/agent-selector-validation.json
"$HUMANIZE_PYTHON" - "$SERVICE_PORT" "$SERVICE_PID" <<'PY'
import hashlib,json,os,socket,subprocess,sys,urllib.request
from datetime import UTC,datetime
from pathlib import Path
port,pid=map(int,sys.argv[1:])
endpoint=f'http://127.0.0.1:{port}'
proc=Path(f'/proc/{pid}')
assert proc.stat().st_uid==os.getuid(),'service must belong to this user'
argv=proc.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0')
assert '--port' in argv and argv[argv.index('--port')+1]==str(port)
assert argv[argv.index('--host')+1]=='127.0.0.1'
models=json.load(urllib.request.urlopen(endpoint+'/v1/models',timeout=5))
base=next(m for m in models['data'] if m['id']=='qwen38-27b-fp8')
asset=json.loads(Path('results/qwen27-asset-manifest.json').read_text())
assert base['parent'] is None and base['root']==asset['directory']
metrics=urllib.request.urlopen(endpoint+'/metrics',timeout=5).read().decode()
queue={line.split('{')[0]:float(line.rsplit(' ',1)[1]) for line in metrics.splitlines()
       if line.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{'))}
assert len(queue)==2 and not any(queue.values()),f'service is busy: {queue}'
used=subprocess.check_output(['nvidia-smi','-i','6','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
assert int(used)<256,'GPU6 evaluator must be idle'
with socket.socket() as probe:probe.bind(('127.0.0.1',18116))
package=Path(argv[0]).parents[1]/'lib/python3.13/site-packages/vllm'
routing=package/'entrypoints/serve/engine/serving.py'
source=routing.read_text()
assert 'def _maybe_get_adapters(' in source and 'is_base_model' in source
source_paths=[routing,package/'entrypoints/openai/models/serving.py',
              package/'entrypoints/openai/chat_completion/serving.py']
manifest={'mode':'existing_owner_service','captured_utc':datetime.now(UTC).isoformat(),
          'endpoint':endpoint,'pid':pid,'process_start_ticks':proc.joinpath('stat').read_text().split()[21],
          'argv':argv,'generation_gpus':'2,3','base_model_id':base['id'],'adapter_requested':None,
          'model_listing':models,'initial_queue':queue,'vllm_version':'0.27.1',
          'routing_source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
          'runtime_differences_from_parent':['GPU2/3 instead of GPU4/5','65536 instead of 16384 context',
             'FP8 KV cache','prefix caching','LoRA-enabled runtime with base ID requested',
             'memory utilization 0.90 instead of 0.85'],
          'scope':'integration replication; runtime differs from the original strategy comparison'}
Path('results/selector-runtime-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'preflight':'passed','base_model_id':base['id'],'initial_queue':queue}))
PY
"$HUMANIZE_PYTHON" -m pytest tests/unit tests/integration -q
"$HUMANIZE_PYTHON" scripts/local_proxy.py --model-id kernel-qwen-local-27b \
  --upstream-model-id qwen38-27b-fp8 --upstream "http://127.0.0.1:$SERVICE_PORT" \
  --seed 20261012 --trace-dir runs/selector-validation-wire > runs/proxy-selector.log 2>&1 &
PROXY_SERVER_PID=$!
trap 'kill "$PROXY_SERVER_PID" 2>/dev/null || true' EXIT
"$HUMANIZE_PYTHON" - <<'PY'
import time,urllib.request
for _ in range(30):
    try:
        urllib.request.urlopen('http://127.0.0.1:18116/v1/models',timeout=2).close()
        break
    except OSError:time.sleep(.2)
else:raise RuntimeError('local proxy did not become ready')
PY
"$HUMANIZE_PYTHON" scripts/run_selector_validation.py --runtime-manifest results/selector-runtime-manifest.json
"$HUMANIZE_PYTHON" scripts/summarize_strategy.py --input results/agent-selector-validation.json
"$HUMANIZE_PYTHON" scripts/build_report.py --input results/agent-selector-validation.json
