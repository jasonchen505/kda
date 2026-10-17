"""Actual candidate-only CUDA activity; no unavailable counter or causal claim."""
import argparse
import importlib.util
import json
import sys
import os
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]

from kernel_agent.protocol import admit  # noqa: E402
from kernel_agent.representative_tasks import specs  # noqa: E402
from scripts.evaluate_representative import inputs,run  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('task')
    p.add_argument('source')
    p.add_argument('--trace',required=True)
    a = p.parse_args()
    admit(Path(a.source).read_text())
    spec = importlib.util.spec_from_file_location('activity_candidate',a.source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    shape = specs()[a.task]['timing_shapes'][1]
    torch.manual_seed(771019)
    values = inputs(a.task,shape,torch.bfloat16)
    for _ in range(3):
        run(a.task,module.run,values,a.task=='attention')
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:
        run(a.task,module.run,values,a.task=='attention')
        torch.cuda.synchronize()
    profile.export_chrome_trace(a.trace)
    activity = [{'name':e.name,'device_type':str(e.device_type),'duration_us':e.time_range.elapsed_us()}
        for e in profile.events() if 'CUDA' in str(e.device_type) or 'cudaLaunch' in e.name]
    index = torch.cuda.current_device()
    result = {'status':'profiled_activity' if any('CUDA' in e['device_type'] for e in activity) else 'profile_unavailable',
        'shape':shape,'activity':activity,'hardware_counters_available':False,
        'scope':'Actual instrumented CUDA activity only; no hardware counters or causal attribution',
        'device_audit':{'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'device_index':index,
            'device_count':torch.cuda.device_count(),'gpu_uuid':str(torch.cuda.get_device_properties(index).uuid)}}
    print(json.dumps(result))


if __name__ == '__main__':
    main()
