"""Fit source facts to the actual prompt budget without spending model calls."""
import json

from kernel_agent import grounded_repair, model_pool, operational
from kernel_agent.campaign import atomic_json, digest
from kernel_agent.transport import PROPOSER_PREFIX

ARMS = grounded_repair.ARMS
assignment = grounded_repair.assignment


class ContextBudgetError(RuntimeError):
    def __init__(self, receipt):
        super().__init__('no mandatory context view fits; policy-local refusal before dispatch')
        self.receipt = receipt


def packet(source, view):
    if view == 'baseline':
        return None
    value = grounded_repair.facts(source)
    if view == 'full':
        return value
    if view != 'compact':
        raise ValueError('unknown source-fact view')
    value['scope'] = 'Budget-compact partial source facts; complete parent source remains mandatory'
    original_stores = value['stores']
    value['stores'] = [{**v, 'dependencies': {k: t if len(t) <= 65 else t[:62] + '...'
                                            for k, t in list(v['dependencies'].items())[:4]},
                        'dependencies_omitted': max(0, len(v['dependencies']) - 4)} for v in original_stores[:1]]
    value['stores_omitted'] += len(original_stores) - len(value['stores'])
    value['budget_view'] = 'compact'
    while len(json.dumps(value, separators=(',', ':'))) > 1700 and value['stores']:
        value['stores'].pop()
        value['stores_omitted'] += 1
    return value


def views(arm, history):
    return ('full', 'compact', 'baseline') if arm == 'grounded' and operational.anchor(history) else ('baseline',)


def context(task_id, contract, history, capability, mode, arm, view='full'):
    if view == 'baseline':
        return grounded_repair.context(task_id, contract, history, capability, mode, 'indexed')
    parts, settings = grounded_repair.context(task_id, contract, history, capability, mode, arm)
    if view == 'compact' and arm == 'grounded' and operational.anchor(history):
        value = packet(operational.anchor(history)['code'], view)
        parts = [(name, 'Verified source facts:\n' + json.dumps(value, sort_keys=True, separators=(',', ':'))
                  + '\nRepair obligations:\n' + grounded_repair.RULES if name == 'source_facts' else text, required)
                 for name, text, required in parts]
    return parts, settings


def messages(parts, settings):
    return [{'role': 'system', 'content': settings.get('role_prefix', PROPOSER_PREFIX)
             + ' No tools. Follow the supplied contract.'},
            {'role': 'user', 'content': '\n\n'.join(text for _, text, _ in parts)}]


def choose(endpoint, path, task_id, contract, history, capability, mode, arm):
    receipt = {'arm': arm, 'mode': mode, 'endpoint': endpoint, 'views': [], 'selected_view': None,
               'context_limit': 8192, 'max_output_tokens': 2048, 'status': 'checking'}
    for view in views(arm, history):
        parts, settings = context(task_id, contract, history, capability, mode, arm, view)
        try:
            count = model_pool.http(endpoint['port'], '/tokenize', {'model': endpoint['model'],
                                    'messages': messages(parts, settings), 'add_generation_prompt': True,
                                    'chat_template_kwargs': {'enable_thinking': False}})['count']
        except Exception as error:  # noqa: BLE001 -- before any model completion request
            receipt.update(status='tokenizer_unavailable', error=str(error))
            atomic_json(path, receipt)
            raise model_pool.NotDispatched('tokenizer unavailable before context selection') from error
        receipt['views'].append({'view': view, 'input_tokens': count,
                                 'context_sha256': digest({'parts': parts, 'settings': settings})})
        if count + 2048 <= 8192:
            receipt.update(selected_view=view, status='selected')
            atomic_json(path, receipt)
            return parts, settings, receipt
    receipt['status'] = 'context_budget_error'
    atomic_json(path, receipt)
    raise ContextBudgetError(receipt)


def verify(receipt, task_id, contract, history, capability, mode, arm):
    expected = views(arm, history)
    seen = receipt['views']
    if ([r['view'] for r in seen] != list(expected[:len(seen)])
            or receipt['arm'] != arm or receipt['mode'] != mode
            or receipt['context_limit'] != 8192 or receipt['max_output_tokens'] != 2048):
        raise ValueError('context adaptation order/settings changed')
    for row in seen:
        parts, settings = context(task_id, contract, history, capability, mode, arm, row['view'])
        if (row['context_sha256'] != digest({'parts': parts, 'settings': settings})
                or type(row['input_tokens']) is not int or row['input_tokens'] < 0):
            raise ValueError('context adaptation source or measured count changed')
    if receipt['status'] == 'selected':
        if (not seen or receipt['selected_view'] != seen[-1]['view'] or seen[-1]['input_tokens'] + 2048 > 8192
                or any(r['input_tokens'] + 2048 <= 8192 for r in seen[:-1])):
            raise ValueError('context adaptation did not choose first fitting view')
    elif receipt['status'] == 'context_budget_error':
        if (receipt['selected_view'] is not None or len(seen) != len(expected)
                or any(r['input_tokens'] + 2048 <= 8192 for r in seen)):
            raise ValueError('context budget refusal is inconsistent')
    elif receipt['status'] != 'tokenizer_unavailable':
        raise ValueError('unrecognized context selection outcome')
    return receipt['selected_view']
