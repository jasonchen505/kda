"""The actual loopback protocol refuses an SDK replay before a second dispatch."""
import json
import urllib.error
import urllib.request

import pytest

from kernel_agent import staged_gateway
from kernel_agent.campaign import atomic_json


def test_sdk_ticket_dispatches_once_even_when_retried(monkeypatch, tmp_path):
    calls = []

    class FakeClient:
        def __init__(self, endpoint, directory, deadline):
            self.directory = directory

        def propose(self, sections, seed, request_id, **kwargs):
            calls.append(request_id)
            atomic_json(self.directory / (request_id + '.json'), {'usage': {'prompt_tokens': 10,
                        'completion_tokens': 4, 'total_tokens': 14}})
            return '{"text":"fake transport fixture"}'

    monkeypatch.setattr(staged_gateway.model_pool, 'Client', FakeClient)
    atomic_json(tmp_path / 'control.json', {'id': 'a', 'allocated': 1, 'schema': {}, 'role': 'planner'})
    server, _ = staged_gateway.start(tmp_path, {}, 123)
    packet = {'ticket': 'a', 'sections': [['x', 'test fixture', True]]}
    wire = json.dumps({'messages': [{'role': 'user', 'content': staged_gateway.MARKER + json.dumps(packet)}]}).encode()

    def send():
        return urllib.request.urlopen(urllib.request.Request(
            f'http://127.0.0.1:{server.server_port}/v1/chat/completions', wire,
            {'Content-Type': 'application/json'}), timeout=3)

    try:
        with send() as response:
            stream = response.read().decode()
        assert '[DONE]' in stream and 'fake transport fixture' in stream
        with pytest.raises(urllib.error.HTTPError) as refused:
            send()
        assert refused.value.code == 400
        assert calls == ['a']
    finally:
        server.shutdown()
        server.server_close()
