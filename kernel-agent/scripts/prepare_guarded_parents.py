"""Revalidate three declared old model-authored failures under SEARCH only, without new models."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from kernel_agent import staged  # noqa: E402
from kernel_agent.campaign import atomic_json, digest  # noqa: E402
from kernel_agent.protocol import admit  # noqa: E402
from kernel_agent.staged_tools import Tools  # noqa: E402

STORE = Path('/data/home/chenyizhou/kernel-agent-storage')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--gpu-python', required=True)
    args = parser.parse_args()
    if not args.directory.resolve().is_relative_to(STORE / 'experiments'):
        raise ValueError('raw parents must remain remote')
    args.directory.mkdir(parents=True, exist_ok=False)
    choices = {}
    old = STORE / 'experiments/grounded-budget-v2/blocks'
    for task in ['scale_transpose', 'affine_silu']:
        path = old / f'reg-{task}-source-s20261221' / 'result.json'
        row = json.loads(path.read_text())['initial']
        choices[task] = {'code': row['code'], 'source_sha256': row['source_sha256'],
                         'origin': str(path), 'selection': 'Declared prior budget-study failed parent, selected before new model calls.'}
    path = ROOT / 'results/operational-study/blocks/new-l2_normalize-s20261124/result.json'
    row = json.loads(path.read_text())['initial']
    choices['l2_normalize'] = {'code': row['code'], 'source_sha256': row['source_sha256'],
        'origin': str(path), 'selection': 'Declared old dtype-related compile failure; selected before new model calls.'}
    rows = {}
    for task, row in choices.items():
        assert admit(row['code']) == row['source_sha256'] == staged.text_hash(row['code'])
        folder = args.directory / task
        folder.mkdir()
        source = folder / (row['source_sha256'] + '.py')
        source.write_text(row['code'])
        tools = Tools(folder, task, 'staged_transfer', args.gpu_python, args.directory / 'gpu4.lock')
        verdict = tools.evaluate({'path': str(source), 'source_sha256': row['source_sha256']})
        atomic_json(folder / 'search-revalidation.json', verdict)
        row.update(evaluation=verdict, model_calls=0, scope='Selected diagnostic failure, not fresh generation or blind generalization.')
        atomic_json(folder / 'parent.json', row)
        rows[task] = {'path': str(folder / 'parent.json'), 'source_sha256': row['source_sha256'],
                      'parent_digest': digest(row), 'status': verdict['status'], 'origin': row['origin']}
    atomic_json(args.directory / 'parents.json', {'parents': rows, 'model_calls': 0, 'search_evaluations': 3})
    assert all(v['status'] in {'numeric_error', 'compile_error', 'runtime_error'} for v in rows.values()), 'Not all declared parents are actual search failures'
    print(json.dumps({k: {'sha': v['source_sha256'], 'status': v['status']} for k, v in rows.items()}))


if __name__ == '__main__':
    main()
