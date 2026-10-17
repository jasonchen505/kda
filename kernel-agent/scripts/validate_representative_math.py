"""Independent NumPy FP64 checks of new GPU oracles; no model-authored kernels."""
import json
import sys
from pathlib import Path
import os

import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]

from scripts.evaluate_representative import inputs,reference  # noqa: E402


def numpy_reference(task,values,causal):
    x=[v.float().cpu().numpy().astype(np.float64) for v in values]
    if task=='gemm':
        return x[0]@x[1]
    q,k,v=x
    z=q@k.swapaxes(-1,-2)/np.sqrt(q.shape[-1])
    if causal:
        z=np.where(np.arange(z.shape[-1])[None,:]>np.arange(z.shape[-2])[:,None],-np.inf,z)
    p=np.exp(z-z.max(axis=-1,keepdims=True))
    p/=p.sum(axis=-1,keepdims=True)
    return p@v


def main():
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(2026101800)
    rows=[]
    for task,shapes in [('gemm',[[3,5,7],[17,33,29]]),('attention',[[1,2,7,24],[2,1,17,32]])]:
        for dtype in [torch.float32,torch.bfloat16]:
            for shape in shapes:
                values=inputs(task,shape,dtype)
                for causal in ([False,True] if task=='attention' else [False]):
                    expected=numpy_reference(task,values,causal)
                    actual=reference(task,values,causal).float().cpu().numpy()
                    tol=(.015,.015) if dtype==torch.bfloat16 else (1e-4,1e-3)
                    np.testing.assert_allclose(actual,expected,atol=tol[0],rtol=tol[1])
                    if task=='attention':
                        sdpa=F.scaled_dot_product_attention(*values,is_causal=causal)
                        np.testing.assert_allclose(sdpa.float().cpu().numpy(),expected,atol=tol[0],rtol=tol[1])
                        if causal:
                            np.testing.assert_allclose(actual[:,:,0,:],values[2][:,:,0,:].float().cpu().numpy(),atol=tol[0],rtol=tol[1])
                    rows.append({'task':task,'shape':shape,'dtype':str(dtype),'causal':causal,'status':'passed'})
    i=torch.cuda.current_device()
    print(json.dumps({'status':'oracle_validated','checks':rows,
        'scope':'NumPy FP64 versus GPU math oracle and SDPA; integration only, no agent success',
        'device_audit':{'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'device_index':i,
            'device_count':torch.cuda.device_count(),'gpu_uuid':str(torch.cuda.get_device_properties(i).uuid)}}))


if __name__=='__main__':
    main()
