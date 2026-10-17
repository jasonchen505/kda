"""Phase-local acceptance and an honest tool-event view for the corrected 4090 port."""
import json
import time

from kernel_agent.campaign import digest
from kernel_agent.staged_tools import Tools as BaseTools, compact

GOALS = {
    'research': ('THIS PHASE ONLY: create and independently evaluate a correct Triton implementation. '
        'One actual correct search result with its event id is sufficient. Do NOT require a speedup, '
        'profiling, a second variant or shape routing here; those belong to later phases. '
        'The workspace starts with NO generated kernel. For references use read api or read example; '
        'read source is available only after a submission. Once correctness is observed, request finish.'),
    'profile_refine': ('THIS PHASE ONLY: start from accepted correct source, use profile on its SHA, '
        'and justify an optimization or keep/reject decision with actual profile/search evidence. '
        'Successful CUDA activity fallback is acceptable with an explicit hardware-counter limitation. '
        'Do NOT require shape routing or two variants yet. Do not require a guaranteed speedup; '
        'negative optimization results are valid evidence. Never infer unavailable SM/DRAM/stall counters.'),
    'shape_tune': ('THIS PHASE ONLY: compare at least two distinct correct source variants on three '
        'search shapes. Use real measurements to choose specialization/dispatch or retain the generic '
        'winner. Keep a generic path for all contracted shapes. Do not claim a router was implemented '
        'unless code contains one. Final hidden shapes remain inaccessible.'),
}


def goal(stage):
    return GOALS[stage]


def recoverable_writer_reply(directory, ticket, schema):
    """Only a fully received, hash-bound invalid response is eligible; never retry uncertainty."""
    path = directory / 'requests' / (ticket + '.json')
    response = directory / 'requests' / (ticket + '.response.json')
    if not path.exists() or not response.exists():
        return False
    receipt, value = json.loads(path.read_text()), json.loads(response.read_text())
    if receipt.get('status') != 'completed' or digest(value) != receipt.get('response_sha256'):
        return False
    try:
        schema.model_validate_json(value['choices'][0]['message']['content'])
    except ValueError:
        return True
    return False


def record_reply_error(tools, stage, ticket):
    event = {'id': f'e{len(tools.events)+1}', 'tool': 'decode_reply', 'stage': stage,
             'source_sha256': tools.proposals[-1]['source_sha256'] if tools.proposals else None,
             'reason': 'Host rejected a fully received malformed/truncated writer response',
             'result': {'status': 'protocol_error', 'model_call_id': ticket,
                        'message': 'Your previous received response could not be decoded under the schema. '
                        'It spent one opportunity. In a remaining action, emit concise complete JSON '
                        'within2048 output tokens. No partial code was reconstructed; the previous source is retained.'},
             'started': time.time(), 'finished': time.time()}
    tools.events.append(event)
    tools.save()
    return event


def review_events(events):
    """Include failed reads/empty research, not just successful numerical measurements."""
    rows = []
    for event in events:
        result = event['result']
        if event['tool'] == 'evaluate':
            result = compact(result, probes=False)
        elif event['tool'] == 'profile':
            result = {k: v for k, v in result.items() if k in {
                'status', 'source_sha256', 'metrics', 'activity', 'hardware_counters_available', 'scope'}}
        elif event['tool'] in {'read', 'research'}:
            result = {'status': result.get('status', 'returned'),
                      'message': str(result.get('message', ''))[:250],
                      'query': result.get('query'), 'pages': [p['path'] for p in result.get('pages', [])],
                      'source_sha256': result.get('sha256'),
                      'text_excerpt': str(result.get('text', result.get('context', '')))[:160]}
        rows.append({'id': event['id'], 'tool': event['tool'], 'stage': event['stage'],
                     'source_sha256': event['source_sha256'], 'result': result})
    return rows


class Tools(BaseTools):
    def execute(self, action, stage):
        event = super().execute(action, stage)
        if event['tool'] == 'read' and event['result'].get('status') == 'tool_error' and not self.proposals:
            event['result']['message'] = ('No generated source exists in this fresh workspace. '
                'Use read api/example for references, submit a complete run implementation, '
                'then evaluate its returned SHA. Repeating read source cannot create a kernel.')
            self.save()
        if event['tool'] == 'research' and 'pages' in event['result'] and not event['result']['pages']:
            event['result'].update(status='no_matches', message='No matching pinned pages were returned. '
                'Use a short literal keyword, or read api/example. This is not successful research evidence.')
            self.save()
        return event
