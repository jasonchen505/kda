"""Zero-model-call retrieval diagnostic; never changes or rescues an agent outcome."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from kernel_agent.campaign import atomic_json, digest  # noqa: E402

STORE = Path('/data/home/chenyizhou/kernel-agent-storage')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    output = args.directory / 'analysis-diagnostics/kernelwiki'
    output.mkdir(parents=True, exist_ok=False)
    base = STORE / 'upstreams/kda/skills/KernelWiki'
    queries = {}
    for path in sorted((STORE / 'experiments/staged-study-v2/blocks').glob('*/*/tools.json')):
        for event in json.loads(path.read_text())['events']:
            result = event['result']
            if event['tool'] == 'research' and result.get('query') and not result.get('pages'):
                queries.setdefault(result['query'], str(path))
    rows = []
    for index, (query, origin) in enumerate(list(queries.items())[:3]):
        command = [sys.executable, str(base / 'scripts/query.py'), query, '--limit', '2', '--paths-only']
        child = subprocess.run(command, cwd=base, capture_output=True, text=True, timeout=60)
        (output / f'{index}.stdout.txt').write_text(child.stdout)
        (output / f'{index}.stderr.txt').write_text(child.stderr)
        pages = []
        if child.returncode == 0:
            for name in child.stdout.splitlines():
                path = (base / name).resolve()
                if not path.is_relative_to(base.resolve()) or not path.is_file():
                    raise ValueError('upstream query returned a non-page path')
                pages.append({'path': str(path.relative_to(base)),
                              'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        rows.append({'query': query, 'original_record': origin, 'literal_search_pages': 0,
                     'upstream_query_exit_code': child.returncode, 'upstream_query_pages': pages,
                     'error_excerpt': child.stderr[-300:] if child.returncode else ''})
    value = {'schema_version': 1, 'kind': 'retrieval_interface_diagnostic', 'rows': rows,
             'kernelwiki_commit': subprocess.check_output(['git', '-C', str(base), 'rev-parse', 'HEAD'], text=True).strip(),
             'query_script_sha256': hashlib.sha256((base / 'scripts/query.py').read_bytes()).hexdigest(),
             'model_calls': 0, 'gpu_evaluations': 0,
             'scope': 'Offline replay of recorded failed query strings against the original local query tool. '
                      'Not integrated into scored agent runs, no generation/performance effect established. '
                      'The pinned corpus targets Hopper/Blackwell; returned pages are not verified Ada recipes.'}
    value['summary_sha256'] = digest(value)
    atomic_json(output / 'diagnostic.json', value)
    source = Path(__file__).read_bytes()
    (output / 'probe_staged_knowledge.py').write_bytes(source)
    atomic_json(ROOT / 'results/staged-knowledge-diagnostic.json', value)
    print(json.dumps({'queries': len(rows), 'returned_page_counts': [len(r['upstream_query_pages']) for r in rows],
                      'model_calls': 0, 'gpu_evaluations': 0}))


if __name__ == '__main__':
    main()
