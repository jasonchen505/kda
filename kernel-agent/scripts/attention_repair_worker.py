"""One isolated Humanize process, pinned to one authorized Qwen service through a hard gateway."""
import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from kernel_agent.campaign import atomic_json  # noqa: E402
from kernel_agent.staged_gateway import start  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configuration', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.configuration.read_text())
    directory = Path(config['directory'])
    directory.mkdir(parents=True, exist_ok=True)
    server, client = start(directory, config['endpoint'], config['seed'])
    os.environ.update(OPENAI_API_BASE=f'http://127.0.0.1:{server.server_port}/v1',
                      OPENAI_BASE_URL=f'http://127.0.0.1:{server.server_port}/v1',
                      OPENAI_API_KEY='local-loopback-placeholder', HUMANIZE_PRICES='off',
                      LITELLM_LOCAL_MODEL_COST_MAP='True', DO_NOT_TRACK='1')
    from hmz.sdk import Hmz
    workspace = directory / 'workspace'
    workspace.mkdir(exist_ok=True)
    versions = {n: importlib.metadata.version(n) for n in ['hmz', 'litellm', 'openai', 'pydantic']}
    atomic_json(directory / 'runtime.json', {'python': sys.executable, 'packages': versions,
                'route': 'Humanize Agent.run -> real LiteLLM -> loopback audited gateway -> local Qwen',
                'bound_endpoint': config['endpoint'], 'native_tool_api': False,
                'hosted_tools': True, 'smoke_only': args.smoke})
    try:
        reply = Hmz(workspace).run(str(ROOT / 'flows' / config.get('flow', 'staged_kernel_contract.py')), config['contract'],
            agents={role: 'litellm/openai/staged-qwen' for role in ['planner', 'writer', 'reviewer']},
            params={k: config[k] for k in ['directory', 'task_id', 'contract', 'arm', 'gpu_python', 'lock_file']}
                   | {'smoke': args.smoke, 'initial_path': config.get('initial_path', '')}
                   | {'policy': config['policy']},
            budget={'duration': 2400, 'output_tokens': 49152, 'graceful': False}, backup=[]).run()
        atomic_json(directory / 'sdk-exit.json', {'status': 'returned', 'reply': reply,
                                                'physical_calls': client.calls})
        print(json.dumps({'status': 'returned', 'calls': client.calls, 'directory': str(directory)}))
    finally:
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    from kernel_agent.representative_reference_router import routed_references
    with routed_references():
        main()
