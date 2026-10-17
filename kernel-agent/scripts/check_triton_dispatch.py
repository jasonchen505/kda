"""Positive JIT-witness integration using an existing immutable model-authored kernel."""
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from kernel_agent.protocol import admit  # noqa: E402
from kernel_agent.representative_policy import track_dispatches,validate_host  # noqa: E402
from scripts.evaluate_candidate import inputs,reference  # noqa: E402


def main():
    folder=Path('/data/home/chenyizhou/kernel-agent-storage/experiments/staged-contract-core-v1/blocks/generation-affine_silu-s2026101801/guarded')
    result=json.loads((folder/'result.json').read_text())
    sha=result['selection']['source_sha256']
    source=folder/'candidates'/(sha+'.py')
    code=source.read_text()
    assert admit(code)==sha
    validate_host(code)
    spec=importlib.util.spec_from_file_location('dispatch_integration_candidate',source)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values=inputs('affine_silu',[2,3],torch.float32)
    with track_dispatches('dispatch_integration_candidate') as counts:
        actual=module.run(*values)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual,reference('affine_silu',*values),atol=2e-5,rtol=2e-4)
        assert sum(counts.values())>0
    index=torch.cuda.current_device()
    print(json.dumps({'status':'dispatch_verified','source_sha256':sha,'candidate_jit_dispatches':dict(counts),
        'scope':'Existing frozen model kernel integration only; not a new agent success',
        'device_audit':{'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'device_index':index,
            'device_count':torch.cuda.device_count(),'gpu_uuid':str(torch.cuda.get_device_properties(index).uuid)}}))


if __name__=='__main__':
    main()
