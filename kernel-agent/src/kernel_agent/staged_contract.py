"""Capability-aligned stage artifacts and atomic model-authored experiments."""
import json
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from kernel_agent import staged
from kernel_agent.staged_guard import ast_identity, observations
from kernel_agent.staged_guard_v2 import Tools as GuardTools

CAPABILITY = ('Search correctness and per-shape timing, plus CUDA activity ONLY. '
              'No hardware counters, causal bottleneck isolation or guaranteed speedup. '
              'A failed or slower optimization is a valid experiment; never invent a cause.')
GOALS = {
    'research': 'Obtain a correct model-authored source through test. No optimization or profiling is required.',
    'profile_refine': 'Profile the accepted correct baseline, then test an AST-distinct model-authored variant. '
                      'Conclude keep/reject from actual outcomes; a slower or incorrect variant is a valid negative result. '
                      'Do not wait for unavailable hardware counters or require a speedup/causal isolation.',
    'shape_tune': 'Obtain two AST-distinct correct implementations measured on three common search shapes. '
                  'Run compare, choose the search gate-first generic winner and explain the per-shape tradeoff. '
                  'Keeping a generic implementation is valid; no unmeasured router or final shapes.',
}


class Step(BaseModel):
    model_config = ConfigDict(extra='forbid')
    tool: Literal['test', 'profile', 'compare', 'read', 'research', 'finish']
    purpose: str


class Plan(BaseModel):
    model_config = ConfigDict(extra='forbid')
    hypothesis: str
    measurement: Literal['search_only', 'activity_and_search', 'common_shape_search']
    steps: list[Step] = Field(min_length=1, max_length=4)
    negative_result: str


class Action(BaseModel):
    model_config = ConfigDict(extra='forbid')
    tool: Literal['test', 'profile', 'compare', 'read', 'research', 'finish']
    argument: str
    code_lines: list[str]
    reason: str


class Review(BaseModel):
    model_config = ConfigDict(extra='forbid')
    accept: bool
    evidence_ids: list[str]
    selected_candidate: str
    decision: Literal['keep_generic', 'tested_negative', 'tested_candidate', 'insufficient']
    limitation: Literal['activity_only_no_causal_attribution']
    hypotheses: list[str]
    reason: str


def review_schema(tools):
    events = [e['id'] for e in tools.events]
    correct = [r['id'] for r in tools.rows if r['evaluation'].get('status') == 'correct']
    return create_model('BoundReview', __base__=Review,
        evidence_ids=(list[Literal[tuple(events or ['none'])]], ...),
        selected_candidate=(Literal[tuple(correct + ['none'])], ...))


def action_schema(allowed):
    # Distinct schemas prevent one tool's reference from leaking into another's argument.
    choices = []
    for name in allowed:
        args = {'test': [''], 'profile': ['latest_correct'], 'compare': [''],
                'read': ['api', 'example', 'guide'], 'finish': ['']}.get(name)
        fields = {'tool': (Literal[name], ...)}
        if args is not None:
            fields['argument'] = (Literal[tuple(args)], ...)
        if name != 'test':
            fields['code_lines'] = (list[str], Field(max_length=0))
        choices.append(create_model(name.title()+'Request', __base__=Action, **fields))
    if len(choices) == 1:
        return choices[0]
    # Root unions are not supported by every provider. A single wrapper is accepted by
    # Humanize and both local structured-output engines.
    from typing import Union
    return create_model('ToolRequest', action=(Union[tuple(choices)], ...))


def unwrap(action):
    return getattr(action, 'action', action)


def plan_issues(plan, stage, compared):
    names = [s.tool for s in plan.steps]
    errors = []
    measurement = {'research': 'search_only', 'profile_refine': 'activity_and_search', 'shape_tune': 'common_shape_search'}[stage]
    if plan.measurement != measurement:
        errors.append('Use the active phase measurement capability: '+measurement)
    required = {'research': {'test'}, 'profile_refine': {'profile', 'test'},
                'shape_tune': {'compare'} | (set() if compared else {'test'})}[stage]
    if not required <= set(names):
        errors.append('Missing required real actions: '+', '.join(sorted(required-set(names))))
    if stage == 'profile_refine' and 'profile' in names and 'test' in names and names.index('profile') > names.index('test'):
        errors.append('Profile the accepted baseline before testing its variant.')
    if stage == 'shape_tune' and 'test' in names and 'compare' in names and names.index('compare') < names.index('test'):
        errors.append('Test before compare when another correct source is needed.')
    if any(names.count(n) > 1 for n in ['profile', 'compare', 'read', 'research']):
        errors.append('Repeated reference/profile/compare does not create an experiment.')
    return errors


class Tools(GuardTools):
    def start_phase(self):
        best = staged.select(self.rows)
        self.phase_entry = best['source_sha256'] if best else None
        self.phase_start = len(self.events)

    def emit(self, tool, stage, result, reason='', sha=None):
        now = time.time()
        event = {'id': f'e{len(self.events)+1}', 'tool': tool, 'stage': stage,
                 'source_sha256': sha, 'reason': reason, 'started': now, 'finished': now, 'result': result}
        self.events.append(event)
        self.save()
        return event

    def comparison(self):
        good = [r for r in self.rows if r['evaluation'].get('status') == 'correct']
        distinct = {}
        for r in good:
            distinct.setdefault(ast_identity(r['code']), r)
        rows = list(distinct.values())
        if len(rows) < 2:
            return None
        shapes = [{json.dumps(t['shape']) for t in r['evaluation']['timings']} for r in rows]
        common = set.intersection(*shapes)
        if len(common) < 3:
            return None
        best = staged.select(good)
        return {'status': 'compared', 'common_shapes': [json.loads(s) for s in sorted(common)],
            'candidates': [{'id': r['id'], 'sha': r['source_sha256'],
                'timings': r['evaluation']['timings']} for r in rows],
            'selected_candidate': best['id'], 'selected_sha': best['source_sha256'],
            'rule': 'search gate-first then geomean; stable ties', 'scope': 'Shared-GPU search observations, not causal attribution'}

    def readiness(self, stage):
        successful = [e for e in self.events if e['tool'] == 'evaluate' and e['result'].get('status') == 'correct']
        best = staged.select(self.rows)
        required, missing = [], []
        if not best:
            missing.append('correct independently searched source')
        else:
            required += [next(e['id'] for e in reversed(successful) if e['source_sha256'] == best['source_sha256'])]
        if stage == 'profile_refine':
            profiles = [e for e in self.events[self.phase_start:] if e['tool'] == 'profile'
                and e['source_sha256'] == self.phase_entry and e['result'].get('status') in {'profiled', 'profiled_activity'}]
            if not profiles:
                missing.append('actual activity profile of accepted baseline')
            else:
                required.append(profiles[-1]['id'])
            tested = [e for e in self.events[self.phase_start:] if e['tool'] == 'evaluate'
                and e['result'].get('status') in {'correct', 'numeric_error', 'compile_error', 'runtime_error', 'contract_error'}
                and e.get('resolved_argument') == e['source_sha256']
                and any(r['source_sha256'] == e['source_sha256'] and r['source_sha256'] != self.phase_entry
                    and ast_identity(r['code']) != ast_identity(next(x['code'] for x in self.rows if x['source_sha256'] == self.phase_entry))
                    for r in self.rows)] if self.phase_entry else []
            if not tested:
                missing.append('actual AST-distinct variant evaluation')
            else:
                required.append(tested[-1]['id'])
        if stage == 'shape_tune':
            comparisons = [e for e in self.events[self.phase_start:] if e['tool'] == 'compare' and e['result'].get('status') == 'compared']
            if not comparisons:
                missing.append('two correct AST-distinct sources on three common shapes, recorded by compare')
            else:
                required.append(comparisons[-1]['id'])
                for r in comparisons[-1]['result']['candidates']:
                    required.append(next(e['id'] for e in reversed(successful) if e['source_sha256'] == r['sha']))
        return {'ready': not missing, 'missing': missing, 'required_evidence_ids': sorted(set(required)),
                'search_winner': best['id'] if best else 'none', 'capability': CAPABILITY}

    def allowed(self, stage, remaining):
        if self.readiness(stage)['ready']:
            return ['finish']
        choices = []
        if self.submissions < staged.CANDIDATE_CAP and self.gpu_calls < 9:
            choices.append('test')
        if stage == 'profile_refine' and not self.profile_ready() and self.phase_entry and self.profiles < 2:
            choices.append('profile')
        if stage == 'shape_tune' and self.comparison():
            choices.append('compare')
        # Required experiment first: discretionary lookup only when more than the minimum slots remain.
        need = 1 if stage == 'research' else 2
        if remaining > need and not any(e['tool'] in {'read', 'research'} for e in self.events[self.phase_start:]):
            choices += ['read', 'research']
        return choices or ['finish']

    def execute(self, action, stage):
        action = unwrap(action)
        action = action.model_dump(mode='json') if isinstance(action, BaseModel) else dict(action)
        name, reason = action['tool'], action.get('reason', '')
        if name == 'test':
            event = super().execute({'tool': 'submit', 'argument': '', 'edits': [],
                'code_lines': action['code_lines'], 'reason': reason}, stage)
            if event['result'].get('status') != 'admitted':
                return event
            row = self.rows[-1]
            repeats = [r for r in self.rows[:-1] if ast_identity(r['code']) == ast_identity(row['code'])]
            if repeats:
                row['evaluation'] = {'status': 'semantic_repeat', 'message': 'Same executable AST already exists; change the actual configuration or implementation. No GPU measurement executed.'}
                return self.emit('test', stage, row['evaluation'], reason, row['source_sha256'])
            return super().execute({'tool': 'evaluate', 'argument': row['source_sha256'], 'reason': reason}, stage)
        if name == 'compare':
            value = self.comparison() or {'status': 'not_ready', 'message': 'Need two AST-distinct correct measured sources.'}
            return self.emit('compare', stage, value, reason)
        if name == 'profile':
            # The active experiment baseline is immutable, even after a failed variant.
            action['argument'] = self.phase_entry or 'latest_correct'
        if name == 'finish':
            return self.emit(name, stage, {'status': 'review_requested'}, reason)
        return super().execute(action, stage)

    def packet(self, stage):
        return {'readiness': self.readiness(stage), 'observations': observations(self),
            'comparison': self.comparison(), 'events': [{'id': e['id'], 'tool': e['tool'],
                'sha': e['source_sha256'], 'status': e['result'].get('status', 'returned')}
                for e in self.events], 'capability': CAPABILITY}


def gate(stage, tools, review):
    ready = tools.readiness(stage)
    errors = list(ready['missing'])
    if not set(review.evidence_ids) <= {e['id'] for e in tools.events}:
        errors.append('Unknown event citation')
    if not review.accept or review.decision == 'insufficient':
        errors.append('Reviewer rejected the active-phase artifact.')
    if not set(ready['required_evidence_ids']) <= set(review.evidence_ids):
        errors.append('Review must cite all required actual events from the single evidence list.')
    if review.selected_candidate != ready['search_winner']:
        errors.append('Select the measured search gate-first winner; final workloads are unavailable.')
    return not errors, errors
