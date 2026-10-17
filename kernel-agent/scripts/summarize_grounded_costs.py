"""Post-run resource accounting only; never changes generation, selection or outcomes."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from kernel_agent.campaign import atomic_json, digest  # noqa: E402
from scripts.summarize_grounded import analyze  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/grounded-costs.json')
    args = parser.parse_args()
    full = analyze(args.directory)
    if full['status'] != 'complete':
        raise ValueError('cost report requires all terminal outcomes')
    plan = json.loads((args.directory / 'plan.json').read_text())
    for name, expected in plan['source_sha256'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError('current executed source differs from its frozen identity: ' + name)
    manifest = args.directory / 'evidence-manifest.json'
    raw = json.loads(manifest.read_text())
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert raw['sha256']['analysis-tools/summarize_grounded_costs.py'] == script_sha
    arms = {}
    for path in sorted((args.directory / 'blocks').glob('*/requests/*.json')):
        if path.name.endswith(('.request.json', '.response.json')) or path.name.startswith('.'):
            continue
        r = json.loads(path.read_text())
        if not r.get('dispatched_utc'):
            continue
        key = 'initial' if r['request_id'] == 'initial' else r['request_id'].rsplit('-r', 1)[0]
        a = arms.setdefault(key, {'dispatched': 0, 'tokenized_input_tokens': 0,
                                 'reported_prompt_tokens': 0, 'reported_output_tokens': 0,
                                 'missing_prompt_usage': 0, 'missing_output_usage': 0,
                                 'finish_reasons': {}})
        a['dispatched'] += 1
        a['tokenized_input_tokens'] += r['input_tokens']
        usage = r.get('usage') or {}
        a['reported_prompt_tokens'] += usage.get('prompt_tokens', 0)
        a['reported_output_tokens'] += usage.get('completion_tokens', 0)
        a['missing_prompt_usage'] += 'prompt_tokens' not in usage
        a['missing_output_usage'] += 'completion_tokens' not in usage
        reason = r.get('finish_reason') or 'unreported'
        a['finish_reasons'][reason] = a['finish_reasons'].get(reason, 0) + 1
    assert sum(a['dispatched'] for a in arms.values()) == full['dispatched_requests']
    assert sum(a['reported_output_tokens'] for a in arms.values()) == full['reported_output_tokens']
    result = {'scope': 'Post-run resource aggregation; no policy/selection changes or throughput claim',
              'plan_sha256': full['plan_sha256'], 'full_reconstruction_sha256': digest(full),
              'raw_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
              'analysis_script_sha256': script_sha, 'arms': arms,
              'original_frozen_files_currently_matching': len(plan['source_sha256']),
              'notes': 'Tokenized inputs and server-reported prompt usage are separate observations; reserved output is not actual usage.'}
    result['sha256'] = digest(result)
    atomic_json(args.output, result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
