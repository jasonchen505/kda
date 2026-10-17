"""Budget adaptation must preserve source/measurement bindings and add no completions."""
import copy

import pytest

from kernel_agent import grounded_budget, model_pool
from kernel_agent.protocol import sha

SOURCE = 'import triton\n@triton.jit\ndef kernel(X):\n    pass\ndef run(x):\n    return x\n'
HISTORY = [{'request_id': 'imported', 'code': SOURCE, 'source_sha256': sha(SOURCE),
            'evaluation': {'status': 'numeric_error'}}]
CAP = {'language_symbols': []}
ENDPOINT = {'port': 1234, 'model': 'fake'}


@pytest.mark.parametrize('counts,expected', [([1000], 'full'), ([7000,1000], 'compact'),
                                            ([7000,7000,1000], 'baseline')])
def test_first_fitting_view_and_exact_baseline(monkeypatch, tmp_path, counts, expected):
    sent = []
    def http(port, path, body=None):
        assert path == '/tokenize'
        sent.append(body)
        return {'count': counts[len(sent)-1]}
    monkeypatch.setattr(model_pool, 'http', http)
    parts, settings, receipt = grounded_budget.choose(ENDPOINT,tmp_path/'context.json','task','contract',
                                                      HISTORY,CAP,'span_patch','grounded')
    assert receipt['selected_view'] == expected
    assert grounded_budget.verify(receipt,'task','contract',HISTORY,CAP,'span_patch','grounded') == expected
    assert len(sent) == len(counts)
    if expected == 'baseline':
        assert (parts,settings) == grounded_budget.context('task','contract',HISTORY,CAP,'span_patch','indexed','baseline')
    tampered = copy.deepcopy(receipt)
    tampered['views'][-1]['context_sha256'] = 'forged'
    with pytest.raises(ValueError,match='source or measured count'):
        grounded_budget.verify(tampered,'task','contract',HISTORY,CAP,'span_patch','grounded')


def test_unfit_context_is_explicit_policy_local_and_verifiable(monkeypatch,tmp_path):
    monkeypatch.setattr(model_pool,'http',lambda *args,**kwargs:{'count':9000})
    with pytest.raises(grounded_budget.ContextBudgetError) as caught:
        grounded_budget.choose(ENDPOINT,tmp_path/'context.json','task','contract',HISTORY,CAP,'span_patch','grounded')
    assert caught.value.receipt['selected_view'] is None
    assert grounded_budget.verify(caught.value.receipt,'task','contract',HISTORY,CAP,'span_patch','grounded') is None


def test_compact_packet_does_not_change_parent():
    before = copy.deepcopy(HISTORY)
    value = grounded_budget.packet(SOURCE,'compact')
    assert value['source_sha256'] == sha(SOURCE) and HISTORY == before
    assert grounded_budget.packet(SOURCE,'baseline') is None
