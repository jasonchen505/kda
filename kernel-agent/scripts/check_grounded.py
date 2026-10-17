"""Context-only preflight on historical sources; no new completions or GPU calls."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from kernel_agent import model_pool, grounded_repair  # noqa: E402
from kernel_agent.campaign import atomic_json  # noqa: E402
from kernel_agent.transport import PROPOSER_PREFIX  # noqa: E402
from scripts.run_grounded_study import task_specs  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    directory = Path('/data/home/chenyizhou/kernel-agent-storage/experiments/span-study-v1')
    capability = json.loads((ROOT / 'data/runtime-triton-api.json').read_text())
    specs = task_specs()
    endpoints = model_pool.discover()
    seen, rows = set(), []
    for path in sorted((directory / 'blocks').glob('*/result.json')):
        block = json.loads(path.read_text())
        for attempt in [block['initial']] + block['arms']['span_repair']['attempts'][1:]:
            if not attempt.get('code') or attempt['source_sha256'] in seen:
                continue
            seen.add(attempt['source_sha256'])
            # A rejected-patch second-slot context exercises retained parent plus failure memory.
            history = [attempt, {'request_id': 'grounded-r1', 'mode': 'span_patch',
                                'evaluation': {'status': 'patch_error', 'message': 'context-only synthetic rejection'}}]
            for arm in grounded_repair.ARMS:
                parts, settings = grounded_repair.context(block['block']['task_id'], specs[block['block']['task_id']]['contract'],
                                                           history, capability, 'full_source_fallback', arm)
                for endpoint in endpoints:
                    messages = [{'role': 'system', 'content': settings.get('role_prefix', PROPOSER_PREFIX) + ' No tools. Follow the supplied contract.'},
                                {'role': 'user', 'content': '\n\n'.join(text for _, text, _ in parts)}]
                    count = model_pool.http(endpoint['port'], '/tokenize', {'model': endpoint['model'], 'messages': messages,
                                            'add_generation_prompt': True, 'chat_template_kwargs': {'enable_thinking': False}})['count']
                    rows.append({'source_sha256': attempt['source_sha256'], 'port': endpoint['port'], 'arm': arm, 'input_tokens': count})
                    if count + 2048 > 8192:
                        atomic_json(args.output, {'status': 'failed', 'rows': rows})
                        raise ValueError('mandatory source facts exceed common context')
    atomic_json(args.output, {'status': 'passed', 'scope': 'Tokenization only, zero model completions', 'sources': len(seen), 'rows': rows})
    print(json.dumps({'status': 'passed', 'sources': len(seen), 'tokenize_calls': len(rows),
                      'maximum_input_tokens': max(r['input_tokens'] for r in rows)}))


if __name__ == '__main__':
    main()
