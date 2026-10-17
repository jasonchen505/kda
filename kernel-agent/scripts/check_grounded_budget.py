"""Exercise actual v1 histories/modes on three tokenizers; zero completion calls."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from kernel_agent import grounded_budget, model_pool  # noqa: E402
from kernel_agent.campaign import atomic_json  # noqa: E402
from scripts.run_grounded_study import task_specs  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    source = Path('/data/home/chenyizhou/kernel-agent-storage/experiments/grounded-study-v1')
    cap = json.loads((ROOT / 'data/runtime-triton-api.json').read_text())
    specs = task_specs()
    endpoints = model_pool.discover()
    choices, first_over, rows = Counter(), 0, []
    for path in sorted((source / 'blocks').glob('*/result.json')):
        result = json.loads(path.read_text())
        task = result['block']['task_id']
        for arm, value in result['arms'].items():
            for index, attempt in enumerate(value['attempts'][1:], 1):
                history = value['attempts'][:index]
                mode, _ = grounded_budget.assignment(arm, history)
                for endpoint in endpoints:
                    target = args.output.parent / 'grounded-budget-contexts' / (path.parent.name + '-' + attempt['request_id'] + '-p' + str(endpoint['port']) + '.json')
                    _, _, receipt = grounded_budget.choose(endpoint, target, task, specs[task]['contract'], history, cap, mode, arm)
                    grounded_budget.verify(receipt, task, specs[task]['contract'], history, cap, mode, arm)
                    choices[arm + ':' + receipt['selected_view']] += 1
                    first_over += receipt['views'][0]['input_tokens'] + 2048 > 8192
                    rows.append({'block': path.parent.name, 'call': attempt['request_id'], 'port': endpoint['port'],
                                 'selected_view': receipt['selected_view'], 'input_tokens': receipt['views'][-1]['input_tokens']})
    value = {'status': 'passed', 'scope': 'Actual historical modes/histories, tokenization only, zero completions',
             'policy_preflights': len(rows), 'first_view_overflows': first_over, 'choices': dict(choices),
             'maximum_selected_input_tokens': max(r['input_tokens'] for r in rows), 'rows': rows}
    atomic_json(args.output, value)
    print(json.dumps({k: v for k, v in value.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
