"""Fresh-process GEMM/self-attention correctness and conditional library timing."""
import importlib.util
import json
import math
import os
import random
import statistics
import sys
import traceback
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import triton.testing

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]

from kernel_agent.protocol import admit  # noqa: E402
from kernel_agent.representative_policy import validate_host,track_dispatches  # noqa: E402
from kernel_agent.representative_tasks import specs  # noqa: E402


def inputs(task, shape, dtype, device='cuda'):
    if task == 'gemm':
        m,k,n = shape
        return (torch.randn((m,k), dtype=dtype, device=device), torch.randn((k,n), dtype=dtype, device=device))
    return tuple(torch.randn(shape, dtype=dtype, device=device) for _ in range(3))


def reference(task, values, causal=False):
    if task == 'gemm':
        return (values[0].float() @ values[1].float()).to(values[0].dtype)
    q,k,v = (x.float() for x in values)
    scores = q @ k.transpose(-1,-2) / math.sqrt(q.shape[-1])
    if causal:
        indices = torch.arange(q.shape[-2], device=q.device)
        scores = scores.masked_fill(indices[None,:] > indices[:,None], float('-inf'))
    return (torch.softmax(scores, dim=-1) @ v).to(values[0].dtype)


def run(task, fn, values, causal=False):
    return fn(*values, causal=causal) if task == 'attention' else fn(*values)


def finite(value):
    result=float(value)
    return result if math.isfinite(result) else str(result)


def numeric_witness(actual, expected, atol, rtol):
    actual=actual.detach().float()
    expected=expected.detach().float()
    mismatch=~torch.isclose(actual,expected,atol=atol,rtol=rtol,equal_nan=False)
    indices=mismatch.nonzero()
    error=(actual-expected).abs().nan_to_num(nan=float('inf'),posinf=float('inf'))
    coordinates=[]
    if indices.numel():
        flat=int(error.reshape(-1).argmax())
        index=[]
        for dim in reversed(actual.shape):
            index.append(flat % dim)
            flat//=dim
        coordinates.append(tuple(reversed(index)))
        for row in indices[:2].tolist():
            if tuple(row) not in coordinates:
                coordinates.append(tuple(row))
    return {'scope':'Actual failing search output; bounded values, no root-cause assertion',
            'atol':atol,'rtol':rtol,'mismatches':int(mismatch.sum()),'elements':actual.numel(),
            'nonfinite_actual':int((~torch.isfinite(actual)).sum()),
            'coordinates':[{'index':list(i),'actual':finite(actual[i]),'expected':finite(expected[i]),
                            'absolute_error':finite(error[i])} for i in coordinates],
            'mismatch_by_head':mismatch.sum(dim=(0,2,3)).tolist() if actual.ndim==4 else None}


def semantic_relations(task, fn):
    if task!='attention':
        return []
    rows=[]
    # Trusted search-only observations, executed identically in both study arms.
    # These are small algebraic identities, not another generated kernel/reference.
    torch.manual_seed(551003)
    for kind in ['zero_q_uniform','constant_v_by_head','causal_first_row']:
        values=inputs(task,[1,2,7,24],torch.float32)
        q,k,v=values
        causal=kind=='causal_first_row'
        if kind=='zero_q_uniform':
            q.zero_()
            expected=v.mean(dim=-2,keepdim=True).expand_as(v)
        elif kind=='constant_v_by_head':
            v[:,0].fill_(-1.)
            v[:,1].fill_(2.)
            expected=v
        else:
            expected=v[:,:,:1,:]
        try:
            actual=run(task,fn,values,causal)
            torch.cuda.synchronize()
            if causal:
                actual=actual[:,:,:1,:]
            witness=numeric_witness(actual,expected,1e-4,1e-3)
            rows.append({'kind':kind,'seed':551003,'shape':[1,2,7,24],
                         'passed':witness['mismatches']==0,'max_error_sample':witness['coordinates'][:1],
                         'mismatches':witness['mismatches']})
        except Exception as error:  # noqa: BLE001 -- a diagnostic must not replace the scored failure
            rows.append({'kind':kind,'passed':False,'probe_error':str(error)[-180:]})
    return rows


def main(task, source, final=False):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    spec = specs()[task]
    code=Path(source).read_text()
    try:
        sha=admit(code)
        validate_host(code)
    except (ValueError,SyntaxError) as error:
        return {'status':'policy_error','message':str(error)}
    module_spec = importlib.util.spec_from_file_location('representative_candidate', source)
    module = importlib.util.module_from_spec(module_spec)
    try:
        module_spec.loader.exec_module(module)
    except Exception as error:  # noqa: BLE001 -- candidate import failure is preserved
        return {'status': 'compile_error', 'message': str(error)[-1800:]}
    with track_dispatches('representative_candidate') as dispatches:
        shapes = spec['final_correctness_shapes' if final else 'correctness_shapes'] + spec['final_timing_shapes' if final else 'timing_shapes']
        torch.manual_seed(992771 if final else 771019)
        checks = 0
        for dtype in [torch.float32, torch.bfloat16]:
            for shape in shapes:
                for scale in [1.,3.]:
                    for causal in ([False,True] if task == 'attention' else [False]):
                        values = inputs(task, shape, dtype)
                        values[0].mul_(scale)
                        copies = [v.clone() for v in values]
                        expected = reference(task, values, causal)
                        try:
                            dispatch_before=sum(dispatches.values())
                            actual = run(task, module.run, values, causal)
                            torch.cuda.synchronize()
                            if sum(dispatches.values())<=dispatch_before:
                                return {'status':'policy_error','message':'No actual candidate Triton JIT dispatch observed'}
                            if not isinstance(actual,torch.Tensor) or actual.shape != expected.shape or actual.dtype != dtype or actual.device != values[0].device or not actual.is_contiguous():
                                return {'status': 'contract_error', 'shape': shape, 'message': 'shape/dtype/device/contiguity mismatch'}
                            if any(not torch.equal(v,c) for v,c in zip(values,copies)):
                                return {'status': 'contract_error', 'message': 'input mutation'}
                            if any(actual.data_ptr() == v.data_ptr() for v in values):
                                return {'status': 'contract_error', 'message': 'output aliases input'}
                            atol,rtol = (.015,.015) if dtype == torch.bfloat16 else (1e-4,1e-3)
                            failure_pass='first_call'
                            torch.testing.assert_close(actual,expected,atol=atol,rtol=rtol)
                            actual=run(task,module.run,values,causal)
                            failure_pass='repeat_call'
                            torch.testing.assert_close(actual,expected,atol=atol,rtol=rtol)
                            if any(not torch.equal(v,c) for v,c in zip(values,copies)):
                                return {'status': 'contract_error', 'message': 'input mutation on repeated call'}
                            checks += 1
                        except AssertionError as error:
                            result={'status':'numeric_error','shape':shape,'dtype':str(dtype),'causal':causal,
                                    'checks_passed':checks,'message':str(error)[-2200:],'failure_pass':failure_pass,
                                    'source_sha256':sha}
                            if not final:
                                result['numeric_witness']=numeric_witness(actual,expected,atol,rtol)
                                result['semantic_relations']=semantic_relations(task,module.run)
                                result['diagnostic_jit_dispatches']=dict(dispatches)
                            return result
                        except Exception as error:  # noqa: BLE001 -- generated Triton failures remain scored
                            return {'status': 'compile_error' if 'Compilation' in type(error).__name__ else 'runtime_error',
                                'shape': shape, 'dtype': str(dtype), 'causal': causal, 'checks_passed': checks,
                                'message': str(error)[-2200:]}
        baseline = (lambda a,b: a@b) if task == 'gemm' else (lambda q,k,v: F.scaled_dot_product_attention(q,k,v,is_causal=True))
        compiled = torch.compile(baseline, fullgraph=True)
        rows = []
        rng = random.Random(20261018)
        for shape in spec['final_timing_shapes' if final else 'timing_shapes']:
            values = inputs(task,shape,torch.bfloat16)
            funcs = {'compiled': partial(compiled,*values),
                     'candidate': partial(run,task,module.run,values,task=='attention')}
            for f in funcs.values():
                f()
            torch.cuda.synchronize()
            samples = {k:[] for k in funcs}
            orders = []
            for _ in range(3):
                order = list(funcs)
                rng.shuffle(order)
                orders.append(order)
                for name in order:
                    samples[name].append(triton.testing.do_bench_cudagraph(funcs[name],rep=50)*1000)
            rows.append({'shape':shape,'dtype':'bfloat16','causal':task=='attention',
                'compiled_us':statistics.median(samples['compiled']), 'candidate_us':statistics.median(samples['candidate']),
                'samples_us':samples,'orders':orders})
        return {'status':'correct','source_sha256':sha,'checks_passed':checks,'timings':rows,
                'candidate_jit_dispatches':dict(dispatches),'performance_reference':spec['performance_reference'],'torch':torch.__version__,'gpu':torch.cuda.get_device_name()}


if __name__ == '__main__':
    index = torch.cuda.current_device()
    proof = {'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'device_index':index,
             'device_count':torch.cuda.device_count(),'gpu_uuid':str(torch.cuda.get_device_properties(index).uuid)}
    try:
        result = main(sys.argv[1],sys.argv[2],final='--final' in sys.argv)
    except Exception as error:  # noqa: BLE001 -- evaluator failures must not disappear
        result = {'status':'evaluator_error','message':str(error)[-2000:],'traceback':traceback.format_exc()[-2500:]}
    result['device_audit'] = proof
    print(json.dumps(result,allow_nan=False))
