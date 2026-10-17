"""Loopback Humanize/LiteLLM adapter with one durable audited Qwen dispatch per ticket."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from kernel_agent import model_pool, staged
from kernel_agent.campaign import atomic_json

MARKER = 'STAGED_REQUEST_JSON\n'


def start(directory, endpoint, seed):
    client = model_pool.Client(endpoint, directory / 'requests', time.monotonic() + 2400)
    used = set()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                content = body['messages'][-1]['content']
                packet, _ = json.JSONDecoder().raw_decode(content.split(MARKER, 1)[1])
                ticket = json.loads((directory / 'control.json').read_text())
                if (packet['ticket'] != ticket['id'] or ticket['id'] in used
                        or ticket['allocated'] > staged.CALL_CAP or len(body['messages']) != 1):
                    raise ValueError('ticket replay, extra SDK history or exhausted hard budget')
                used.add(ticket['id'])
                atomic_json(directory / 'requests' / (ticket['id'] + '.humanize.json'), body)
                text = client.propose(packet['sections'], seed + ticket['allocated'], ticket['id'],
                    schema=ticket['schema'], schema_name='staged_' + ticket['role'],
                    role_prefix='You are a kernel research ' + ticket['role'] + '. '
                    'Follow the user task and supplied JSON schema. Planning means writing a document, '
                    'review means judging supplied evidence, and hosted tools are requested only '
                    'when explicitly asked for a JSON action. Native tool calls are unavailable.')
                receipt = json.loads((directory / 'requests' / (ticket['id'] + '.json')).read_text())
                # Humanize's real LiteLLM driver consumes an OpenAI SSE stream.
                common = {'id': ticket['id'], 'object': 'chat.completion.chunk',
                          'created': int(time.time()), 'model': 'staged-qwen'}
                chunks = [dict(common, choices=[{'index': 0, 'delta': {'role': 'assistant', 'content': text},
                                                'finish_reason': None}]),
                          dict(common, choices=[{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]),
                          dict(common, choices=[], usage=receipt.get('usage') or {})]
                stream = ''.join('data: ' + json.dumps(c) + '\n\n' for c in chunks) + 'data: [DONE]\n\n'
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Content-Length', str(len(stream.encode())))
                self.end_headers()
                self.wfile.write(stream.encode())
            except Exception as error:  # noqa: BLE001 -- never let SDK retry dispatch an uncertain ticket
                data = json.dumps({'error': {'message': str(error), 'type': 'staged_gateway_refusal'}}).encode()
                self.send_response(400)
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(data)

    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, client
