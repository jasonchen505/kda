"""Shared representative-task evaluator with host-compute rejection and JIT witnesses."""
import json
import os
import subprocess

from kernel_agent.representative import ROOT,RepresentativeMixin
from kernel_agent.representative_tasks import specs


class VerifiedMixin(RepresentativeMixin):
    def evaluate(self,row,final=False):
        if self.task_id not in specs():
            return super().evaluate(row,final)
        if not final:
            if self.gpu_calls>=9:
                raise ValueError('search GPU budget exhausted')
            self.gpu_calls+=1
            self.save()
        command=[self.gpu_python,str(ROOT/'scripts/evaluate_representative_verified_serial.py'),
                 '--lock-file',str(self.lock_file),self.task_id,row['path']]
        if final:
            command.append('--final')
        child=subprocess.run(command,capture_output=True,text=True,timeout=400,
            env={**os.environ,'PYTHONPATH':str(ROOT/'src')+':'+str(ROOT)})
        result=json.loads(child.stdout) if child.returncode==0 else {
            'status':'evaluator_error','message':(child.stdout+child.stderr)[-1800:]}
        if result.get('status')=='correct':
            assert result['source_sha256']==row['source_sha256'] and sum(result['candidate_jit_dispatches'].values())>0
        return result
