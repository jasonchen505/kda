"""Both real tool classes must read new task metadata without calling the old task map."""
import pytest

from kernel_agent import staged_tools,staged_verified_guard,staged_contract_v4
from kernel_agent.representative_reference_router import routed_references
from kernel_agent.representative_tasks import specs


@pytest.mark.parametrize('cls',[staged_verified_guard.Tools,staged_contract_v4.Tools])
@pytest.mark.parametrize('task_id',['gemm','attention'])
def test_actual_api_example_routes_preserve_guard_memory_and_duplicate_refusal(monkeypatch,tmp_path,cls,task_id):
    def old(*args):
        raise AssertionError('Representative read must not reach the legacy task map')
    monkeypatch.setattr(staged_tools,'retrieve',old)
    monkeypatch.setattr(staged_tools,'retrieve_example',old)
    tools=cls(tmp_path,task_id,'staged_transfer','unused',tmp_path/'lock')
    with routed_references():
        for arg in ['api','example']:
            event=tools.execute({'tool':'read','argument':arg,'code_lines':[],'edits':[],'reason':'fixture'},'research')
            assert event['result']['text']==specs()[task_id]['api_notes']
            assert event['resolved_argument']==arg
        repeated=tools.execute({'tool':'read','argument':'api','code_lines':[],'edits':[],'reason':'fixture'},'research')
        assert repeated['result']['status']=='repeat_refused'
    assert staged_tools.retrieve is old and staged_tools.retrieve_example is old
    assert len(tools.events)==3


def test_legacy_references_pass_through_and_restore_even_on_error(monkeypatch):
    seen=[]
    def old(result,task):
        seen.append(task)
        return {'context':'legacy fixture'}
    monkeypatch.setattr(staged_tools,'retrieve',old)
    with pytest.raises(RuntimeError),routed_references():
        assert staged_tools.retrieve({},'softmax')=={'context':'legacy fixture'}
        raise RuntimeError('fixture')
    assert seen==['softmax'] and staged_tools.retrieve is old


def test_original_attention_example_defect_is_reproduced_without_router(tmp_path):
    tools=staged_verified_guard.Tools(tmp_path,'attention','staged_transfer','unused',tmp_path/'lock')
    with pytest.raises(KeyError,match='attention'):
        tools.execute({'tool':'read','argument':'example','code_lines':[],'edits':[],'reason':'observed route'},'research')
    assert not tools.events  # The host exception occurred before the tool event was appended.
