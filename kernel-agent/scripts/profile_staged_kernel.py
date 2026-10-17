"""A single candidate-only CUDA profiler range, invoked under Nsight Compute."""
import argparse
import importlib.util
import json
import os
from pathlib import Path

import torch

from kernel_agent.protocol import admit
from scripts.evaluate_candidate import inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task')
    parser.add_argument('source', type=Path)
    parser.add_argument('--shape', required=True)
    parser.add_argument('--activity-output', type=Path)
    args = parser.parse_args()
    source = args.source.read_text()
    admit(source)
    spec = importlib.util.spec_from_file_location('profile_candidate', args.source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(771019)
    values = inputs(args.task, json.loads(args.shape), torch.bfloat16)
    for _ in range(3):
        module.run(*values)
    torch.cuda.synchronize()
    index = torch.cuda.current_device()
    proof = {'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'device_index': index,
             'device_count': torch.cuda.device_count(),
             'gpu_uuid': str(torch.cuda.get_device_properties(index).uuid)}
    result = {'device_audit': proof, 'shape': json.loads(args.shape), 'range': 'candidate_only'}
    if args.activity_output:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as profile:
            module.run(*values)
            torch.cuda.synchronize()
        profile.export_chrome_trace(str(args.activity_output))
        rows = []
        for event in profile.events():
            if 'CUDA' in str(event.device_type) or 'cudaLaunch' in event.name:
                rows.append({'name': event.name, 'device_type': str(event.device_type),
                             'duration_us': event.time_range.elapsed_us()})
        result.update(activity=rows, hardware_counters_available=False,
                      scope='CUDA activity durations/launch events only; no SM/DRAM/stall counters')
    else:
        torch.cuda.cudart().cudaProfilerStart()
        module.run(*values)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    print(json.dumps(result))


if __name__ == '__main__':
    main()
