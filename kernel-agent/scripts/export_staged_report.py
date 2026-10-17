"""Add bounded public projections and migration diagnostics to the frozen raw auditor."""
import hashlib
import json
import re
import runpy
import sys
import copy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from kernel_agent.campaign import atomic_json, digest  # noqa: E402
from kernel_agent import staged  # noqa: E402


def audit_phases(folder, stages):
    tickets = json.loads((folder / 'calls.json').read_text()) if (folder / 'calls.json').exists() else []
    ledger = json.loads((folder / 'tools.json').read_text()) if (folder / 'tools.json').exists() else {'events': [], 'candidates': []}
    by_id = {r['id']: r for r in ledger['candidates']}
    previous_passed = True
    for phase in stages:
        assert previous_passed, 'phase advanced despite a rejected predecessor'
        planners = [t for t in tickets if t['stage'] == phase['stage'] and t['role'] == 'planner']
        assert len(planners) == 2
        assert staged.text_hash(planners[0]['reply']['text']) == phase['draft_sha256']
        assert staged.text_hash(planners[1]['reply']['text']) == phase['plan_sha256']
        assert planners[0]['finished_utc'] <= planners[1]['started_utc']
        for reading in phase['reviews']:
            history, events = [], []
            for event in ledger['events']:
                events.append(event)
                if event['tool'] in {'submit', 'patch'} and event['result'].get('status') == 'admitted':
                    row = copy.deepcopy(by_id[event['result']['candidate_id']])
                    row['evaluation'] = {'status': 'not_evaluated'}
                    history.append(row)
                if event['tool'] == 'evaluate':
                    row = next((r for r in reversed(history) if r['source_sha256'] == event['source_sha256']), None)
                    if row is not None:
                        row['evaluation'] = event['result']
                if event['id'] == reading['after_event']:
                    break
            else:
                raise ValueError('review points to a nonexistent tool event')
            review = staged.Review.model_validate(reading['review'])
            passed, reason = staged.stage_gate(phase['stage'], history, events, review)
            assert passed == reading['gate_passed'] and reason == reading['gate_reason']
        assert phase['accepted'] == any(r['gate_passed'] for r in phase['reviews'])
        previous_passed = phase['accepted']


def main():
    index = sys.argv.index('--directory')
    directory = Path(sys.argv[index+1]).resolve()
    source = Path(__file__).read_bytes()
    identity = hashlib.sha256(source).hexdigest()
    archive = directory / 'analysis-tools' / ('export_staged_report-' + identity[:12] + '.py')
    archive.parent.mkdir(exist_ok=True)
    if archive.exists():
        assert archive.read_bytes() == source
    else:
        archive.write_bytes(source)
    runpy.run_path(str(ROOT / 'scripts/summarize_staged.py'), run_name='__main__')
    path = ROOT / 'results/staged-summary.json'
    value = json.loads(path.read_text())
    value.pop('summary_sha256')
    model_intervals = []
    for block in value['blocks']:
        for arm, row in block['arms'].items():
            folder = directory / 'blocks' / block['id'] / arm
            audit_phases(folder, row['stages'])
            for phase in row['stages']:
                for reading in phase['reviews']:
                    review = reading['review']
                    for key in ['missing', 'next_step']:
                        text = review[key]
                        review[key] = text[:260]
                        review[key+'_truncated'] = len(text) > 260
            tickets = json.loads((folder / 'calls.json').read_text()) if (folder / 'calls.json').exists() else []
            row['received_reply_recoveries'] = sum(t['status'] == 'received_reply_error' for t in tickets)
            facts = 0
            omitted = 0
            model_seconds = 0.0
            for request in (folder / 'requests').glob('call-*.request.json'):
                receipt = json.loads(request.with_name(request.name.replace('.request.json', '.json')).read_text())
                sdk = json.loads(request.with_name(request.name.replace('.request.json', '.humanize.json')).read_text())
                packet, _ = json.JSONDecoder().raw_decode(sdk['messages'][-1]['content'].split('STAGED_REQUEST_JSON\n', 1)[1])
                actual_dispatch = receipt['status'] in {'completed', 'dispatched', 'response_unknown'}
                facts += (actual_dispatch and any(name == 'source_facts' for name, _, _ in packet['sections'])
                          and 'source_facts' not in receipt.get('omitted_sections', []))
                omitted += actual_dispatch and 'source_facts' in receipt.get('omitted_sections', [])
                if receipt['status'] == 'completed':
                    model_seconds += receipt.get('elapsed_seconds', 0)
                    model_intervals.append((receipt['dispatched_utc'], 1))
                    model_intervals.append((receipt['finished_utc'], -1))
            row['source_facts_dispatched'] = facts
            row['source_facts_omitted'] = omitted
            proposals = folder / 'tools.json'
            if proposals.exists():
                ledger = json.loads(proposals.read_text())
                row['rejected_source_preserved'] = sum(r['evaluation']['status'] == 'protocol_error'
                                                      for r in ledger.get('proposals', []))
                tool_seconds = {}
                for event in ledger['events']:
                    name = event['tool']
                    tool_seconds[name] = tool_seconds.get(name, 0) + max(0, event['finished'] - event['started'])
                row['resource_usage'] = {'completed_model_wall_seconds': round(model_seconds, 3),
                    'host_tool_wall_seconds': {k: round(v, 3) for k, v in tool_seconds.items()},
                    'scope': 'Summed recorded wall times; host tools include queue/lock waits. '
                             'Not GPU compute duration or isolated latency. Unknown requests excluded from duration.'}
            for profile in row['profiles']:
                for event in profile['activity']:
                    event['name'] = event['name'][:200]
    active = peak = 0
    for _, change in sorted(model_intervals, key=lambda event: (event[0], event[1])):
        active += change
        peak = max(peak, active)
    value['completed_request_peak_concurrency'] = peak
    value['public_projection'] = {'exporter_sha256': identity,
        'scope': 'Bounded plan/review/tool excerpts and counts, no full replies/kernels/traces. '
                 'Raw reconstruction is server-only; not a new model/GPU execution.'}
    value['summary_sha256'] = digest(value)
    atomic_json(path, value)
    store = directory.parent
    environment = {'schema_version': 1, 'preflights': [], 'unscored_pilot': None,
                   'hardware_counters_available': False, 'gpu': 'NVIDIA GeForce RTX4090',
                   'scope': 'Integration data separate from the scored cohort. Handwritten profiler fixture is not an agent result.'}
    environment['prepared_zero_dispatch_ledger'] = {
        'remote_directory': str(store / 'staged-study-v1'), 'model_calls': 0,
        'scope': 'Superseded prospective preparation; no efficacy outcome or model dispatch'}
    for name in ['staged-preflight-v1', 'staged-preflight-v2', 'staged-preflight-v3', 'staged-preflight-v4',
                 'staged-pilot-v1']:
        folder = store / name
        receipts = [json.loads(p.read_text()) for p in folder.rglob('call-*.json')
                    if re.fullmatch(r'call-\d+\.json', p.name)]
        row = {'name': name, 'remote_directory': str(folder),
               'actual_calls': sum(r['status'] in {'completed', 'response_unknown', 'dispatched'} for r in receipts),
               'completed_calls': sum(r['status'] == 'completed' for r in receipts),
               'not_dispatched': sum(r['status'] == 'not_dispatched' for r in receipts)}
        preflight = folder / 'preflight.json'
        if preflight.exists():
            data = json.loads(preflight.read_text())
            row.update(status=data['status'], sdk_services=len(data['sdk']),
                       profile_status=data.get('profile', {}).get('result', {}).get('status'))
            environment['preflights'].append(row)
        else:
            result = folder / 'result.json'
            row.update(status='unscored_integration_pilot',
                       failure=json.loads(result.read_text()).get('failure') if result.exists() else None)
            environment['unscored_pilot'] = row
    runtime = store / 'staged-preflight-v4/sdk-0/runtime.json'
    if runtime.exists():
        environment['humanize_runtime'] = json.loads(runtime.read_text())['packages']
    environment['summary_sha256'] = digest(environment)
    atomic_json(ROOT / 'results/staged-environment.json', environment)
    print(json.dumps({'public_summary_sha256': value['summary_sha256'], 'bounded_blocks': len(value['blocks'])}))


if __name__ == '__main__':
    main()
