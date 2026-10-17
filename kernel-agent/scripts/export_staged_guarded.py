"""Reconstruct the guarded paired study remotely and publish bounded derived evidence."""
import argparse
import collections
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from kernel_agent import staged, staged_guard as guard, line_protocol, span_repair  # noqa: E402
from kernel_agent.campaign import atomic_json, digest  # noqa: E402
from kernel_agent.protocol import admit, decision  # noqa: E402
from kernel_agent.staged_gateway import MARKER  # noqa: E402


def load(path):
    return json.loads(path.read_text())


def prefix(ledger, end):
    obj = guard.Tools.__new__(guard.Tools)
    obj.rows, obj.proposals, obj.events = [], [], []
    available = iter(ledger.get('proposals', []))
    pending = next(available, None)
    by_id = {r['id']: r for r in ledger.get('candidates', [])}
    for e in ledger['events'][:end]:
        if pending and (e.get('shared_parent') or e['tool'] in {'submit', 'patch'}) and e['source_sha256'] == pending['source_sha256']:
            row = copy.deepcopy(pending)
            if row['id'] in by_id:
                row.update(copy.deepcopy(by_id[row['id']]))
                row['evaluation'] = {'status': 'not_evaluated'}
                obj.rows.append(row)
            obj.proposals.append(row)
            pending = next(available, None)
        if e['tool'] == 'evaluate' and e['result'].get('status') not in {'tool_error', 'repeat_refused'}:
            row = next((r for r in reversed(obj.rows) if r['source_sha256'] == e['source_sha256']), None)
            assert row is not None
            row['evaluation'] = e['result']
        obj.events.append(e)
    return obj


def audit(folder, value, policy, parent):
    tickets = load(folder/'calls.json') if (folder/'calls.json').exists() else []
    ledger = load(folder/'tools.json') if (folder/'tools.json').exists() else {'events': [], 'candidates': [], 'proposals': []}
    assert len(tickets) <= 24
    for key, limit in [('submissions', 6), ('gpu_calls', 9), ('profiles', 2)]:
        assert ledger.get(key, 0) <= limit
    receipts = []
    roles = collections.Counter()
    omitted, facts = collections.Counter(), 0
    intervals = []
    for t in tickets:
        path = folder/'requests'/(t['id']+'.json')
        if not path.exists():
            continue
        r = load(path)
        receipts.append(r)
        if r['status'] not in {'completed', 'response_unknown', 'dispatched'}:
            continue
        roles[t['role']] += 1
        wire = path.with_name(t['id']+'.request.json')
        request = load(wire)
        assert hashlib.sha256(wire.read_bytes()).hexdigest() == r['request_sha256']
        assert request['max_tokens'] == 2048 and r['input_tokens'] + 2048 <= 8192
        assert request['response_format']['json_schema']['schema'] == t['schema']
        packet, _ = json.JSONDecoder().raw_decode(load(path.with_name(t['id']+'.humanize.json'))['messages'][-1]['content'].split(MARKER, 1)[1])
        assert packet['ticket'] == t['id']
        assert request['messages'][1]['content'] == '\n\n'.join(text for name, text, _ in packet['sections'] if name not in r['omitted_sections'])
        omitted.update(r['omitted_sections'])
        facts += any(name == 'source_facts' for name, _, _ in packet['sections']) and 'source_facts' not in r['omitted_sections']
        if r['status'] == 'completed':
            response = load(path.with_name(t['id']+'.response.json'))
            assert digest(response) == r['response_sha256']
            if t['status'] == 'sdk_completed':
                assert json.loads(response['choices'][0]['message']['content']) == t['reply']
            intervals.extend([(r['dispatched_utc'], 1), (r['finished_utc'], -1)])
    for row in ledger.get('proposals', []):
        assert staged.text_hash(row['code']) == row['source_sha256']
        assert Path(row['proposal_path']).read_text() == row['code']
    for row in ledger['candidates']:
        assert admit(row['code']) == row['source_sha256']
        assert Path(row['path']).read_text() == row['code']
    if parent:
        initial = ledger['proposals'][0]
        assert initial['id'] == 'c0' and initial['code'] == parent['code']
        assert ledger['events'][0]['shared_parent'] and ledger['events'][0]['result'] == parent['evaluation']
    # Reconstruct model-authored source/patch transitions from exact SDK replies.
    actions = iter(t for t in tickets if t['role'] == 'writer' and t['status'] == 'sdk_completed')
    source = parent['code'] if parent else None
    for e in ledger['events']:
        if e.get('shared_parent') or e['tool'] == 'decode_reply':
            continue
        ticket = next(actions)
        a = ticket['reply']
        assert a['tool'] == e['tool']
        if a['tool'] in {'submit', 'patch'}:
            if e['result'].get('status') == 'repeat_refused':
                continue
            try:
                if a['tool'] == 'submit':
                    decoded = line_protocol.proposal(json.dumps({'draft': a['reason'] or 'source', 'plan': 'evaluate', 'code_lines': a['code_lines']}))
                else:
                    expected = staged.text_hash(source)
                    assert a['argument'] == expected or (policy == 'guarded' and a['argument'] in {'latest', 'latest source SHA', 'source'})
                    decoded = span_repair.apply(json.dumps({'draft': a['reason'] or 'patch', 'plan': 'evaluate',
                        'base_sha256': expected, 'edits': a['edits']}), source)
                assert staged.text_hash(decoded['code']) == e['source_sha256']
                source = decoded['code']
            except (ValueError, SyntaxError):
                assert e['result']['status'] == 'tool_error'
        if e['tool'] == 'evaluate' and e['result'].get('status') == 'correct':
            assert e['result']['source_sha256'] == e['source_sha256'] and e['result']['device_binding']['verified']
    assert next(actions, None) is None
    phases = []
    for p in value.get('stages', []):
        docs = folder/'docs'/p['stage']
        for kind in ['draft', 'plan']:
            assert staged.text_hash((docs/(kind+'.md')).read_text()) == p[kind+'_sha256']
        planners = [t for t in tickets if t['stage'] == p['stage'] and t['role'] == 'planner' and t['status'] == 'sdk_completed']
        assert planners and planners[0]['reply']['text'] == (docs/'draft.md').read_text()
        plan_reply = planners[-1]['reply']
        plan_text = (docs/'plan.md').read_text()
        if policy == 'guarded' and 'steps' in plan_reply:
            assert json.loads(plan_text) == plan_reply
        else:
            assert plan_text == ('Invalid plan: see validation errors' if policy == 'guarded' else plan_reply['text'])
        assert len(planners) <= 3 if policy == 'guarded' else len(planners) == 2
        stage_events = [i for i, e in enumerate(ledger['events']) if e['stage'] == p['stage'] and not e.get('shared_parent')]
        start = min(stage_events) if stage_events else len(ledger['events'])
        before = prefix(ledger, start)
        if policy == 'guarded':
            limit = 6 if p['stage'] == 'research' else 4
            validations = [docs/'plan-validation.json'] + ([docs/'plan-revision.json'] if (docs/'plan-revision.json').exists() else [])
            for n, path in enumerate(validations):
                v = load(path)
                parsed = guard.Plan.model_validate(v['plan']) if v['plan'] else None
                assert guard.plan_issues(parsed, limit-n, p['stage'], bool(before.proposals), before.compared()) == v['errors']
            writers = [t for t in tickets if t['stage'] == p['stage'] and t['role'] == 'writer']
            assert len(writers) + len(validations)-1 <= limit
        checked_reviews = []
        for rr in p['reviews']:
            end = next(i+1 for i, e in enumerate(ledger['events']) if e['id'] == rr['after_event'])
            state = prefix(ledger, end)
            state.phase_start = start
            state.phase_entry = before.proposals[-1]['source_sha256'] if before.proposals else None
            if policy == 'guarded':
                passed, why, claims = guard.gate(p['stage'], state, guard.Review.model_validate(rr['review']))
            else:
                passed, why = staged.stage_gate(p['stage'], state.rows, state.events, staged.Review.model_validate(rr['review']))
                claims = []
            assert passed == rr['gate_passed'] and why == rr['gate_reason']
            reading = copy.deepcopy(rr)
            for key in ['missing', 'next_step']:
                reading['review'][key] = reading['review'][key][:300]
            for claim in reading['review'].get('claims', []):
                claim['hypothesis'] = claim['hypothesis'][:240]
            for claim in claims:
                if 'text' in claim:
                    claim['text'] = claim['text'][:240]
            reading['checked_claims'] = claims
            checked_reviews.append(reading)
        assert p['accepted'] == any(r['gate_passed'] for r in p['reviews'])
        contract_checks = [{'file': path.name, 'errors': load(path)['errors']} for path in
                           [docs/'plan-validation.json', docs/'plan-revision.json'] if path.exists()]
        entry_ast = guard.ast_identity(before.proposals[-1]['code']) if before.proposals else None
        known_sources = {r['source_sha256']: r['code'] for r in ledger.get('proposals', [])}
        real_variants = [e for e in ledger['events'][start:] if e['stage'] == p['stage'] and e['tool'] == 'evaluate'
            and e['result'].get('status') in {'correct', 'compile_error', 'numeric_error', 'runtime_error', 'mutation_error', 'alias_error'}
            and e['source_sha256'] in known_sources and guard.ast_identity(known_sources[e['source_sha256']]) != entry_ast]
        phases.append({'stage': p['stage'], 'accepted': p['accepted'], 'reviews': checked_reviews,
            'plan_contract_checks': contract_checks,
            'actual_variant_evaluations': len(real_variants),
            'draft_excerpt': (docs/'draft.md').read_text()[:220], 'plan_excerpt': (docs/'plan.md').read_text()[:700]})
    assert all(p['accepted'] for p in value.get('stages', [])[:-1])
    selected = staged.select(ledger['candidates'])
    assert (selected['source_sha256'] if selected else None) == value['selection']['source_sha256']
    if value.get('final') and value['final']['status'] == 'correct':
        assert value['final']['source_sha256'] == value['selection']['source_sha256'] and value['final']['device_binding']['verified']
    counts = collections.Counter(r['status'] for r in receipts)
    plan_files = list((folder/'docs').glob('*/plan-validation.json')) + list((folder/'docs').glob('*/plan-revision.json'))
    verdict = decision(value['final']) if value.get('final') else {'correct': False, 'promoted': False}
    return {'actual_calls': sum(counts[k] for k in ['completed', 'response_unknown', 'dispatched']),
        'unknown_calls': counts['response_unknown']+counts['dispatched'], 'not_dispatched': counts['not_dispatched'],
        'opportunities': len(tickets), 'input_tokens': sum((r.get('usage') or {}).get('prompt_tokens', 0) for r in receipts),
        'output_tokens': sum((r.get('usage') or {}).get('completion_tokens', 0) for r in receipts), 'roles': dict(roles),
        'submissions': ledger.get('submissions', 0), 'new_searches': ledger.get('gpu_calls', 0),
        'tool_counts': dict(collections.Counter(e['tool'] for e in ledger['events'] if not e.get('shared_parent'))),
        'tool_errors': sum(e['result'].get('status') in {'tool_error', 'no_matches'} for e in ledger['events']),
        'blocked_repeats_or_references': sum(e['result'].get('status') == 'repeat_refused' for e in ledger['events']),
        'plan_contract_rejections': sum(bool(load(p)['errors']) for p in plan_files),
        'plan_revisions': len(list((folder/'docs').glob('*/plan-revision.json'))), 'source_facts_requests': facts,
        'omitted_sections': dict(omitted), 'model_wall_seconds': round(sum(r.get('elapsed_seconds', 0) for r in receipts), 3),
        'search_tool_wall_seconds': round(sum(e['finished']-e['started'] for e in ledger['events']), 3),
        'stages': phases, 'final_decision': verdict, 'selected_sha256': value['selection']['source_sha256'],
        'changed_parent_ast': bool(parent and selected and guard.ast_identity(selected['code']) != guard.ast_identity(parent['code'])),
        'failure': value.get('failure'), 'trace': [{'id': e['id'], 'tool': e['tool'], 'stage': e['stage'],
            'status': e['result'].get('status', 'returned'), 'reason_hypothesis': e.get('reason', '')[:220],
            'message': str(e['result'].get('message', ''))[:280], 'source_sha256': e['source_sha256']}
            for e in ledger['events']]}, intervals


def main():
    global guard
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--finalize', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    directory = args.directory.resolve()
    plan, checkpoint = load(directory/'plan.json'), load(directory/'checkpoint.json')
    corrected = plan['kind'].endswith('_corrected')
    if corrected:
        from kernel_agent import staged_guard_v2
        guard = staged_guard_v2
    assert digest(plan) == checkpoint['plan_sha256']
    assert digest({k: v for k, v in checkpoint.items() if k != 'state_sha256'}) == checkpoint['state_sha256']
    for name, sha in plan['source_sha256'].items():
        assert hashlib.sha256((directory/'sources'/name).read_bytes()).hexdigest() == sha
    parents = {t: load(Path(p['path'])) for t, p in plan['parents']['parents'].items()}
    for task, value in parents.items():
        assert digest(value) == plan['parents']['parents'][task]['parent_digest']
    blocks, intervals = [], []
    for entry in checkpoint['blocks']:
        if entry['status'] not in {'complete', 'failed', 'interrupted_unknown'}:
            continue
        b = entry['block']
        folder = directory/'blocks'/b['id']
        path = folder/'result.json'
        raw = load(path) if path.exists() else {'arms': {}, 'generation_committed_utc': None}
        if path.exists():
            assert digest(raw) == entry['result_sha256']
        compact = {**b, 'status': entry['status'], 'port': entry.get('endpoint', {}).get('port'), 'arms': {}}
        for arm in plan['arms']:
            value = raw['arms'].get(arm)
            if value is None:
                raise ValueError('Terminal block lacks complete arm records; retain as unresolved and inspect, never drop denominator')
            assert value['selection']['committed_utc'] <= raw['generation_committed_utc']
            compact['arms'][arm], spans = audit(folder/arm, value, arm, parents[b['task_id']] if b['stratum'] == 'repair' else None)
            intervals.extend(spans)
        blocks.append(compact)
    active = peak = 0
    for _, delta in sorted(intervals, key=lambda e: (e[0], e[1])):
        active += delta
        peak = max(peak, active)
    summary = {'kind': plan['kind'], 'status': checkpoint['status'], 'directory': str(directory),
        'plan_sha256': checkpoint['plan_sha256'], 'scheduled_blocks': 6, 'scheduled_outcomes': 12,
        'completed_blocks': len(blocks), 'blocks': blocks, 'services': checkpoint['services'],
        'peak_completed_request_concurrency': peak, 'scope': plan['scope'],
        'parents': plan['parents'], 'source_files': len(plan['source_sha256']),
        'current_sources_match': all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest() == h for n, h in plan['source_sha256'].items())}
    if args.verify_only:
        assert checkpoint['status'] == 'complete' and len(blocks) == 6
        manifest = load(directory/'evidence-manifest.json')
        assert all(hashlib.sha256((directory/name).read_bytes()).hexdigest() == sha for name, sha in manifest.items())
        print(json.dumps({'status': 'passed', 'blocks': len(blocks), 'raw_files': len(manifest),
                          'scope': 'Read-only recorded wire/source/gate/selection reconstruction; no model/GPU execution or data writes.'}))
        return
    source = Path(__file__).read_bytes()
    archive = directory/'analysis-tools'/('export-guarded-'+hashlib.sha256(source).hexdigest()[:12]+'.py')
    archive.parent.mkdir(exist_ok=True)
    if archive.exists():
        assert archive.read_bytes() == source
    else:
        archive.write_bytes(source)
    if args.finalize:
        assert checkpoint['status'] == 'complete' and len(blocks) == 6
        files = {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.rglob('*'))
            if p.is_file() and p.name not in {'evidence-manifest.json', 'supervisor.log'}
            and '.hmz' not in p.parts and '.humanize' not in p.parts and '__pycache__' not in p.parts and not p.name.endswith('.lock')}
        path = directory/'evidence-manifest.json'
        if path.exists():
            assert load(path) == files
        else:
            atomic_json(path, files)
        summary.update(raw_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), raw_manifest_files=len(files))
    summary['summary_sha256'] = digest(summary)
    name = 'staged-guarded-corrected-summary.json' if corrected else 'staged-guarded-summary.json'
    atomic_json(ROOT/'results'/name, summary)
    store = directory.parent
    integration = []
    for folder_name in ['guarded-preflight-v1', 'guarded-preflight-v2']:
        folder = store/folder_name
        path = folder/'preflight.json'
        if not path.exists():
            continue
        data = load(path)
        receipts = [load(p) for p in folder.glob('sdk-*/requests/call-*.json')
                    if p.stem.removeprefix('call-').isdigit()]
        integration.append({'directory': str(folder), 'status': data['status'],
            'model_calls': sum(r['status'] in {'completed', 'response_unknown', 'dispatched'} for r in receipts),
            'services': len(data['sdk']), 'profile_status': data.get('profile', {}).get('result', {}).get('status'),
            'scope': 'Schema smokes and handwritten profiler fixture; excluded from agent success scores.'})
    environment = {'preflights': integration, 'shared_failed_parents': plan['parents'],
        'superseded_preparations': [{'directory': str(store/'guarded-parents-v1'), 'model_calls': 0,
            'search_evaluations': 1, 'status': 'old_affine_source_passed_search', 'raw_result_missing': True},
            {'directory': str(store/'guarded-parents-v2'), 'model_calls': 0, 'search_evaluations': 1,
             'status': 'old_transpose_source_passed_search', 'raw_result_missing': False}],
        'scope': 'Integration/selection data kept separate from each scored cohort.'}
    environment['summary_sha256'] = digest(environment)
    atomic_json(ROOT/'results/staged-guarded-environment.json', environment)
    print(json.dumps({'status': summary['status'], 'blocks': len(blocks), 'summary_sha256': summary['summary_sha256']}))


if __name__ == '__main__':
    main()
