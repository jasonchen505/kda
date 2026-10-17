"""CPU tensor checks inside GPU runtime; no model requests or scored kernels."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]

import torch  # noqa: E402

from scripts import evaluate_attention_observed as observer  # noqa: E402


def main():
    actual=torch.ones(1,2,7,24)
    expected=actual.clone()
    actual[0,1,5,21]=float('nan')
    witness=observer.numeric_witness(actual,expected,1e-4,1e-3)
    assert witness['mismatches']==1 and witness['nonfinite_actual']==1
    assert witness['coordinates'][0]['index']==[0,1,5,21]
    assert witness['mismatch_by_head']==[0,1]
    json.dumps(witness,allow_nan=False)
    original=observer.inputs
    def cpu_inputs(task,shape,dtype):
        return original(task,shape,dtype,device='cpu')
    with patch.object(observer,'inputs',cpu_inputs),patch.object(torch.cuda,'synchronize',lambda:None):
        rows=observer.semantic_relations('attention',lambda *values,causal=False:observer.reference('attention',values,causal))
        assert len(rows)==3 and all(row['passed'] for row in rows)
        failed=observer.semantic_relations('attention',lambda q,k,v,causal=False:torch.zeros_like(q))
        assert not any(row['passed'] for row in failed)
    result={'status':'passed','checks':['finite coordinate and head binding','nonfinite-safe JSON',
            'trusted relation reference passes','wrong output fails all relations'],
            'scope':'CPU tensors only in existing GPU runtime; zero model/GPU evaluation calls'}
    (ROOT/'results/attention-observer-checks.json').write_text(json.dumps(result)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
