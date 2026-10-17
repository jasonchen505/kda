"""Read-only reconstruction of the staged workflow study; export bounded derived summaries."""
import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from kernel_agent import line_protocol, span_repair, staged  # noqa: E402
from kernel_agent.campaign import atomic_json, digest  # noqa: E402
from kernel_agent.protocol import admit, decision  # noqa: E402
from kernel_agent.staged_gateway import MARKER  # noqa: E402


def load(path):
    return json.loads(path.read_text())


def audit_arm(folder, value):
    tickets = load(folder / 'calls.json') if (folder / 'calls.json').exists() else []
    assert len(tickets) <= 24
    receipts = [load(p) for p in sorted((folder / 'requests').glob('call-*.json'))
                if re.fullmatch(r'call-\d+\.json', p.name)]
    assert len(receipts) <= len(tickets)
    by_id = {r['request_id']: r for r in receipts}
    counts = collections.Counter(r['status'] for r in receipts)
    role_calls, tool_counts = collections.Counter(), collections.Counter()
    input_tokens = output_tokens = 0
    for ticket in tickets:
        receipt = by_id.get(ticket['id'])
        if not receipt:
            continue
        role_calls[ticket['role']] += receipt['status'] in {'completed', 'dispatched', 'response_unknown'}
        if receipt['status'] != 'completed':
            continue
        request_path = folder / 'requests' / (ticket['id'] + '.request.json')
        request = load(request_path)
        response = load(folder / 'requests' / (ticket['id'] + '.response.json'))
        assert hashlib.sha256(request_path.read_bytes()).hexdigest() == receipt['request_sha256']
        assert digest(response) == receipt['response_sha256']
        assert request['max_tokens'] == 2048 and receipt['input_tokens'] + 2048 <= 8192
        assert request['response_format']['json_schema']['schema'] == ticket['schema']
        sdk = load(folder / 'requests' / (ticket['id'] + '.humanize.json'))
        packet, _ = json.JSONDecoder().raw_decode(sdk['messages'][-1]['content'].split(MARKER, 1)[1])
        assert packet['ticket'] == ticket['id']
        kept = [text for name, text, _ in packet['sections'] if name not in receipt['omitted_sections']]
        assert request['messages'][1]['content'] == '\n\n'.join(kept)
        if ticket['status'] == 'sdk_completed':
            assert json.loads(response['choices'][0]['message']['content']) == ticket['reply']
        usage = receipt.get('usage') or {}
        input_tokens += usage.get('prompt_tokens', 0)
        output_tokens += usage.get('completion_tokens', 0)
    ledger = load(folder / 'tools.json') if (folder / 'tools.json').exists() else {'candidates': [], 'events': []}
    assert ledger.get('submissions', 0) <= 6 and ledger.get('gpu_calls', 0) <= 9 and ledger.get('profiles', 0) <= 2
    for row in ledger['candidates']:
        assert admit(row['code']) == row['source_sha256']
        assert Path(row['path']).read_text() == row['code']
    for row in ledger.get('proposals', []):
        assert staged.text_hash(row['code']) == row['source_sha256']
        assert Path(row['proposal_path']).read_text() == row['code']
    proposals = [t['reply'] for t in tickets if t.get('status') == 'sdk_completed'
                 and t['role'] == 'writer' and ('code_lines' in t['reply'])
                 and t['reply'].get('tool', 'submit') in {'submit', 'patch'}]
    edits = [e for e in ledger['events'] if e['tool'] in {'submit', 'patch'}]
    assert len(proposals) == len(edits)
    parent = None
    for proposed, event in zip(proposals, edits):
        if event['tool'] == 'patch':
            if event['result'].get('status') != 'admitted':
                continue
            decoded = span_repair.apply(json.dumps({'draft': proposed.get('reason') or 'model patch',
                'plan': 'apply exact requested edit', 'base_sha256': proposed['argument'],
                'edits': proposed['edits']}), parent)
        else:
            try:
                decoded = line_protocol.proposal(json.dumps({'draft': proposed.get('draft') or proposed.get('reason')
                    or 'model source', 'plan': proposed.get('plan') or 'independent tool evaluation',
                    'code_lines': proposed['code_lines']}))
            except ValueError:
                assert event['result'].get('status') == 'tool_error'
                continue
        if event['result'].get('status') == 'admitted':
            assert admit(decoded['code']) == event['source_sha256']
        if staged.text_hash(decoded['code']) == event['source_sha256']:
            parent = decoded['code']
    for event in ledger['events']:
        tool_counts[event['tool']] += 1
        if event['tool'] == 'evaluate' and event['result'].get('status') == 'correct':
            assert event['result']['source_sha256'] == event['source_sha256']
            assert event['result']['device_binding']['verified']
        if event['tool'] == 'profile' and event['result'].get('status') == 'profiled_activity':
            assert event['result']['activity_device_verified']
            assert event['result']['hardware_counters_available'] is False
            assert event['result']['activity']
    selected = staged.select(ledger['candidates'])
    if value['selection'].get('committed_utc'):
        assert (selected['source_sha256'] if selected else None) == value['selection']['source_sha256']
    else:
        assert value['selection']['source_sha256'] is None  # uncommitted work is never selected post hoc
    if value.get('final') and value['final'].get('status') == 'correct':
        assert value['final']['source_sha256'] == value['selection']['source_sha256']
        assert value['final']['device_binding']['verified']
    phases = value.get('stages', [])
    for phase in phases:
        for kind in ['draft', 'plan']:
            assert staged.text_hash((folder / 'docs' / phase['stage'] / (kind+'.md')).read_text()) == phase[kind+'_sha256']
    return {'allocated_opportunities': len(tickets), 'actual_calls': sum(counts[k] for k in
            ['completed', 'dispatched', 'response_unknown']), 'completed_calls': counts['completed'],
            'unknown_calls': counts['response_unknown'] + counts['dispatched'],
            'not_dispatched': counts['not_dispatched'], 'input_tokens': input_tokens, 'output_tokens': output_tokens,
            'role_calls': dict(role_calls), 'tool_calls': dict(tool_counts),
            'submissions': ledger.get('submissions', 0), 'search_gpu_calls': ledger.get('gpu_calls', 0),
            'profiles': [{'status': e['result'].get('status'), 'hardware_counters_available':
                          e['result'].get('hardware_counters_available'), 'activity': e['result'].get('activity', [])}
                         for e in ledger['events'] if e['tool'] == 'profile'],
            'stages': [{'stage': p['stage'], 'accepted': p['accepted'], 'reviews': p['reviews'],
                        'draft_sha256': p['draft_sha256'], 'plan_sha256': p['plan_sha256'],
                        'draft_excerpt': (folder / 'docs' / p['stage'] / 'draft.md').read_text()[:220],
                        'plan_excerpt': (folder / 'docs' / p['stage'] / 'plan.md').read_text()[:320]}
                       for p in phases],
            'trace': [{'id': e['id'], 'tool': e['tool'], 'stage': e['stage'], 'source_sha256': e['source_sha256'],
                       'status': e['result'].get('status', 'returned'), 'reason': e.get('reason', '')[:220],
                       'message': str(e['result'].get('message', ''))[:350]} for e in ledger['events']],
            'failure': value.get('failure'), 'selected_sha256': value['selection']['source_sha256'],
            'final_decision': value.get('final_decision', {'correct': False, 'promoted': False})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--finalize', action='store_true')
    args = parser.parse_args()
    directory = args.directory.resolve()
    plan, checkpoint = load(directory / 'plan.json'), load(directory / 'checkpoint.json')
    assert digest(plan) == checkpoint['plan_sha256']
    for name, expected in plan['source_sha256'].items():
        assert hashlib.sha256((directory / 'sources' / name).read_bytes()).hexdigest() == expected
    totals = {arm: {'correct': 0, 'promoted': 0, 'resolved': 0, 'calls': 0,
                   'input_tokens': 0, 'output_tokens': 0, 'unknown_calls': 0,
                   'accepted_stages': dict.fromkeys(staged.STAGES, 0)} for arm in staged.ARMS}
    blocks = []
    for scheduled in checkpoint['blocks']:
        block = scheduled['block']
        folder = directory / 'blocks' / block['id']
        path = folder / 'result.json'
        if not path.exists():
            if scheduled['status'] not in {'failed', 'interrupted_unknown'}:
                continue
            result = {'endpoint': scheduled.get('endpoint', {'port': None}), 'arms': {},
                      'generation_committed_utc': '', 'block_failure': scheduled.get('error', scheduled['status'])}
            for arm in staged.ARMS:
                partial = folder / arm / 'result.json'
                result['arms'][arm] = load(partial) if partial.exists() else {
                    'selection': {'source_sha256': None}, 'final': None,
                    'failure': {'block_status': scheduled['status'], 'detail': scheduled.get('error')}}
        else:
            result = load(path)
            if scheduled.get('result_sha256'):
                assert digest(result) == scheduled['result_sha256']
        compact = {'id': block['id'], 'task_id': block['task_id'], 'seed': block['seed'],
                   'port': result['endpoint']['port'], 'block_status': scheduled['status'],
                   'block_failure': result.get('block_failure'), 'arms': {}}
        for arm, value in result['arms'].items():
            if result['generation_committed_utc']:
                assert value['selection'].get('committed_utc', result['generation_committed_utc']) <= result['generation_committed_utc']
            audited = audit_arm(folder / arm, value)
            verdict = decision(value['final']) if value.get('final') else {'correct': False, 'promoted': False}
            audited['final_decision'] = verdict
            compact['arms'][arm] = audited
            total = totals[arm]
            total['correct'] += verdict['correct']
            total['promoted'] += verdict['promoted']
            total['resolved'] += 1
            total['calls'] += audited['actual_calls']
            total['input_tokens'] += audited['input_tokens']
            total['output_tokens'] += audited['output_tokens']
            total['unknown_calls'] += audited['unknown_calls']
            for phase in audited['stages']:
                total['accepted_stages'][phase['stage']] += phase['accepted']
        blocks.append(compact)
    summary = {'schema_version': 1, 'kind': plan['kind'], 'status': checkpoint['status'],
               'directory': str(directory), 'plan_sha256': checkpoint['plan_sha256'],
               'completed_blocks': len(blocks), 'scheduled_blocks': len(plan['blocks']),
               'scheduled_outcomes': plan['scheduled_arms'],
               'arms': totals, 'blocks': blocks, 'services': checkpoint['services'],
               'scope': plan['scope'], 'current_sources_match': all(hashlib.sha256((ROOT / n).read_bytes()).hexdigest()
                       == h for n, h in plan['source_sha256'].items())}
    if args.finalize:
        assert checkpoint['status'] == 'complete' and len(blocks) == len(plan['blocks'])
        assert all(total['resolved'] == len(plan['blocks']) for total in totals.values())
        files = {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(directory.rglob('*')) if p.is_file()
                 and p.name not in {'evidence-manifest.json', 'analysis-full.json', 'supervisor.log'}
                 and '.humanize' not in p.parts and '.hmz' not in p.parts and '__pycache__' not in p.parts
                 and not p.name.endswith('.lock')}
        manifest = {'schema_version': 1, 'files': files, 'scope': 'Remote raw/source/SDK/trace records; '
                    'ephemeral Humanize runtime databases, locks and supervisor console excluded'}
        existing = directory / 'evidence-manifest.json'
        if existing.exists():
            assert load(existing) == manifest
        else:
            atomic_json(existing, manifest)
        summary['manifest_sha256'] = hashlib.sha256(existing.read_bytes()).hexdigest()
        summary['manifest_file_count'] = len(files)
    summary['summary_sha256'] = digest(summary)
    atomic_json(ROOT / 'results/staged-summary.json', summary)
    print(json.dumps({'status': summary['status'], 'blocks': len(blocks), 'arms': totals}))


if __name__ == '__main__':
    main()
