"""The real self-test regression and source-line formatting errors are actionable."""
import pytest
from pydantic import ValidationError

from kernel_agent import staged_contract_v2 as c


def test_source_policy_diagnostics_point_to_real_self_test_calls():
    source='import torch\ndef run(x):\n    return torch.empty_like(x)\nif __name__ == "__main__":\n    torch.manual_seed(0)\n    x = torch.randn(3)\n    torch.allclose(x, x)\n'
    issues=c.source_issues(source)
    assert {i.get('call') for i in issues if 'call' in i}=={'torch.manual_seed','torch.randn','torch.allclose'}
    assert next(i for i in issues if i['kind']=='embedded_self_test')['line']==4
    assert all(i.get('call')!='torch.empty_like' for i in issues)


def test_schema_names_the_operation_and_rejects_embedded_newlines():
    schema=c.action_schema(['submit_and_evaluate'])
    good={'tool':'submit_and_evaluate','argument':'','code_lines':['import triton',''],'reason':'source only'}
    assert schema.model_validate(good).tool=='submit_and_evaluate'
    with pytest.raises(ValidationError):
        schema.model_validate({**good,'code_lines':['import triton\ndef run(): pass']})
    with pytest.raises(ValidationError):
        schema.model_validate({**good,'tool':'test'})
    with pytest.raises(ValidationError):
        schema.model_validate({'action':good})
    with pytest.raises(ValidationError):
        schema.model_validate({**good,'name':'submit_and_evaluate'})


def test_evidence_compaction_retains_measurements_without_sample_duplication():
    row={'shape':[3,5],'candidate_us':2.,'compiled_us':3.,'samples_us':{'candidate':[1.,2.,3.]},'orders':[['a','b']]}
    assert c.trim_timings([row])==[{'shape':[3,5],'candidate_us':2.,'compiled_us':3.}]


def test_main_section_is_rejected_before_gpu_and_read_content_is_retained(tmp_path):
    tools=c.Tools(tmp_path,'affine_silu','staged_transfer','unused',tmp_path/'lock')
    source='import torch\nimport triton\n@triton.jit\ndef kernel():\n    pass\ndef run(x):\n    return x\nif __name__ == "__main__":\n    x = torch.empty((1,))\n'
    event=tools.execute({'tool':'submit_and_evaluate','argument':'','code_lines':source.splitlines(),'reason':'fixture'},'research')
    assert event['result']['status']=='policy_error' and tools.gpu_calls==0
    assert tools.proposals[-1]['code']==source
    tools.execute({'tool':'read','argument':'api','code_lines':[],'reason':'fixture'},'research')
    assert 'Host launch rules' in tools.packet('research')['retained_references'][-1]['text_excerpt']
