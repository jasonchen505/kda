"""Executable plans, progress-aware tools and capability-bound evidence for staged agents."""
import ast
import json
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from kernel_agent import staged
from kernel_agent.campaign import atomic_json
from kernel_agent.staged_phase_interface import Tools as PreviousTools, review_events


class Step(BaseModel):
    model_config = ConfigDict(extra='forbid')
    tool: Literal['read', 'research', 'submit', 'patch', 'evaluate', 'profile', 'finish']
    argument: str
    purpose: str


class Plan(BaseModel):
    model_config = ConfigDict(extra='forbid')
    objective: str
    steps: list[Step] = Field(min_length=1, max_length=6)
    fallback: str


class Claim(BaseModel):
    model_config = ConfigDict(extra='forbid')
    kind: Literal['search_correct', 'activity_observed', 'search_compared', 'hypothesis']
    evidence_ids: list[str]
    hypothesis: str


class Review(staged.Review):
    claims: list[Claim]


def ast_identity(source):
    try:
        return ast.dump(ast.parse(source), include_attributes=False)
    except SyntaxError:
        return None


def plan_issues(plan, limit, stage, have_parent=False, comparison_ready=False):
    if plan is None:
        return ['No valid typed executable plan was received.']
    errors = []
    names = [s.tool for s in plan.steps]
    if len(names) > limit:
        errors.append(f'{len(names)} tool actions exceed {limit} remaining writer actions; reviews are automatic at the last action.')
    if stage == 'research' and not {'submit', 'patch'} & set(names):
        errors.append('Research needs a source submission or patch, not references alone.')
    if (stage == 'research' or (stage == 'shape_tune' and not comparison_ready)) and 'evaluate' not in names:
        errors.append('Plan must obtain a real search evaluation.')
    if stage == 'shape_tune' and comparison_ready and names != ['finish']:
        errors.append('Two AST-distinct correct variants are already measured on three common shapes. Plan only finish/review, no redundant measurement.')
    if stage == 'profile_refine' and not {'profile', 'evaluate'} <= set(names):
        errors.append('Refinement needs profile plus an actually evaluated variant; negative results are valid.')
    written = have_parent
    dirty = False
    reads = set()
    for step in plan.steps:
        if step.tool in {'submit', 'patch'}:
            if step.tool == 'patch' and not written:
                errors.append('Patch has no parent; use submit first.')
            written, dirty = True, True
        elif step.tool == 'evaluate':
            if not written:
                errors.append('Evaluation has no source; submit first.')
            dirty = False
        elif step.tool == 'read':
            if step.argument not in {'api', 'example', 'guide', 'source', 'evidence'}:
                errors.append('Read accepts only api/example/guide/source/evidence; no file or shell paths.')
            if step.argument == 'source' and not written:
                errors.append('Fresh workspace has no generated source to read.')
            if step.argument in reads:
                errors.append('Repeated reference read does not create new evidence.')
            reads.add(step.argument)
    if dirty:
        errors.append('The last written source has no evaluation action.')
    return list(dict.fromkeys(errors))


def import_parent(tools, path):
    """Import an unchanged, shared failed source/search witness; never a generated success."""
    value = json.loads(path.read_text())
    code, verdict = value['code'], value['evaluation']
    sha = staged.text_hash(code)
    assert sha == value['source_sha256'] and verdict['status'] != 'correct'
    target = tools.directory / 'candidates' / (sha + '.py')
    proposal = tools.directory / 'proposals' / (sha + '.py.txt')
    target.parent.mkdir(exist_ok=True)
    proposal.parent.mkdir(exist_ok=True)
    target.write_text(code)
    proposal.write_text(code)
    row = {'id': 'c0', 'code': code, 'source_sha256': sha, 'parent_sha256': None,
           'stage': 'research', 'path': str(target), 'proposal_path': str(proposal),
           'evaluation': verdict, 'imported_parent': True}
    tools.rows.append(row)
    tools.proposals.append(row)
    now = time.time()
    tools.events.append({'id': 'e1', 'tool': 'evaluate', 'stage': 'research',
        'source_sha256': sha, 'reason': 'Shared independently revalidated failed parent, not an arm model/tool call',
        'shared_parent': True, 'started': now, 'finished': now, 'result': verdict})
    tools.save()


def observations(tools):
    """Only host-rendered facts may be presented as observations, never model attribution."""
    facts = []
    for e in tools.events:
        r = e['result']
        if e['tool'] == 'evaluate':
            facts.append({'id': e['id'], 'kind': 'search', 'sha': e['source_sha256'],
                'status': r.get('status'), 'timings': r.get('timings', []),
                'scope': 'Independent search program; conditional shared-GPU timings.'})
        elif e['tool'] == 'profile':
            facts.append({'id': e['id'], 'kind': 'profile', 'sha': e['source_sha256'],
                'status': r.get('status'), 'shape': r.get('shape'),
                'hardware_counters_available': r.get('hardware_counters_available', False),
                'activity': r.get('activity', [])[:4],
                'scope': 'Instrumented CUDA activity. Do not compare its durations to independent search timings. '
                         'No observed launch/occupancy/bandwidth bottleneck follows from this data.'})
    return facts


def validate_claims(claims, tools):
    by_id = {e['id']: e for e in tools.events}
    shas = {r['source_sha256']: r['code'] for r in tools.rows}
    errors, rendered = [], []
    for c in claims:
        if c.kind == 'hypothesis':
            rendered.append({'kind': 'hypothesis', 'text': c.hypothesis, 'verified': False})
            continue
        if c.hypothesis or not c.evidence_ids or not set(c.evidence_ids) <= by_id.keys():
            errors.append('Observation requires real event ids and no free-form hypothesis text.')
            continue
        es = [by_id[i] for i in c.evidence_ids]
        successful = [e for e in es if e['tool'] == 'evaluate' and e['result'].get('status') == 'correct']
        if c.kind == 'search_correct':
            valid = bool(successful)
        elif c.kind == 'activity_observed':
            valid = any(e['tool'] == 'profile' and e['result'].get('status') in {'profiled', 'profiled_activity'} for e in es)
        else:
            measured = [e for e in successful if len(e['result'].get('timings', [])) >= 3]
            identities = {ast_identity(shas[e['source_sha256']]) for e in measured}
            shapes = [{json.dumps(t['shape']) for t in e['result']['timings']} for e in measured]
            valid = (len(identities - {None}) >= 2 and bool(shapes)
                     and len(set.intersection(*shapes)) >= 3)
        if not valid:
            errors.append(c.kind + ' lacks matching source-bound evidence.')
        else:
            rendered.append({'kind': c.kind, 'evidence_ids': c.evidence_ids, 'verified': True,
                'scope': 'Host-checked event/capability binding only; not a causal bottleneck proof.'})
    return errors, rendered


class Tools(PreviousTools):
    def __init__(self, *args):
        super().__init__(*args)
        self.phase_entry = None
        self.phase_start = 0

    def start_phase(self):
        self.phase_entry = self.proposals[-1]['source_sha256'] if self.proposals else None
        self.phase_start = len(self.events)

    def reference_keys(self):
        return {(e['tool'], e.get('resolved_argument')) for e in self.events
                if e['tool'] in {'read', 'research'} and e.get('resolved_argument')}

    def resolve(self, name, argument):
        if name in {'evaluate', 'patch', 'profile'}:
            if argument in {'latest', 'latest source SHA', 'source', 'latest_correct'}:
                rows = self.rows if name == 'profile' or argument == 'latest_correct' else self.proposals
                if name == 'profile' or argument == 'latest_correct':
                    rows = [r for r in rows if r['evaluation'].get('status') == 'correct']
                if not rows:
                    raise ValueError('No eligible source exists. Submit then evaluate; known source list is in the state.')
                argument = rows[-1]['source_sha256']
            known = self.proposals if name == 'patch' else self.rows
            if argument not in {r['source_sha256'] for r in known}:
                raise ValueError('Unknown source reference. Use latest/latest_correct or an exact SHA from the supplied state.')
        return argument

    def profile_ready(self):
        return any(e['tool'] == 'profile' and e['result'].get('status') in {'profiled', 'profiled_activity'}
                   for e in self.events[self.phase_start:])

    def variant_tested(self):
        entry = next((r for r in self.proposals if r['source_sha256'] == self.phase_entry), None)
        identity = ast_identity(entry['code']) if entry else None
        return any(e['tool'] == 'evaluate' and not e.get('shared_parent')
                   and any(r['source_sha256'] == e['source_sha256'] and ast_identity(r['code']) != identity
                           for r in self.proposals) for e in self.events[self.phase_start:])

    def compared(self):
        ids = [e['id'] for e in self.events if e['tool'] == 'evaluate' and e['result'].get('status') == 'correct']
        return not validate_claims([Claim(kind='search_compared', evidence_ids=ids, hypothesis='')], self)[0]

    def allowed(self, stage, remaining):
        latest = self.proposals[-1] if self.proposals else None
        status = latest['evaluation'].get('status') if latest else None
        correct = any(r['evaluation'].get('status') == 'correct' for r in self.rows)
        unevaluated = status == 'not_evaluated'
        if stage == 'research':
            if correct:
                return ['finish']
            need = 1 if unevaluated else 2
        elif stage == 'profile_refine':
            if self.profile_ready() and self.variant_tested():
                return ['finish']
            need = int(not self.profile_ready()) + (0 if self.variant_tested() else 1 if unevaluated else 2)
        else:
            if self.compared():
                return ['finish']
            need = 1 if unevaluated else 2
        choices = []
        if remaining > need:
            if any(('read', a) not in self.reference_keys() for a in ['api', 'example', 'guide']):
                choices.append('read')
            if not any(e['tool'] == 'research' for e in self.events[self.phase_start:]):
                choices.append('research')
        if self.submissions < staged.CANDIDATE_CAP and remaining >= 2:
            choices.append('submit')
            if latest:
                choices.append('patch')
        if unevaluated and self.gpu_calls < 9:
            choices.append('evaluate')
        if stage == 'profile_refine' and correct and not self.profile_ready() and self.profiles < 2:
            choices.append('profile')
        return choices or ['finish']  # a failed goal is reviewed honestly at the existing boundary

    def execute(self, action, stage):
        action = action.model_dump(mode='json') if isinstance(action, staged.Action) else dict(action)
        name = action['tool']
        try:
            argument = self.resolve(name, action.get('argument', ''))
            if name == 'read' and argument not in {'api', 'example', 'guide'}:
                raise ValueError('Read accepts api/example/guide. Complete source and evidence are already retained in state.')
            key = (name, argument)
            if name in {'read', 'research'} and key in self.reference_keys():
                raise ValueError('Unchanged reference was already read; use retained reference memory or submit code.')
            if name in {'profile', 'evaluate'} and any(e['tool'] == name and e['source_sha256'] == argument
                    and e['result'].get('status') not in {'tool_error', 'repeat_refused'} for e in self.events):
                raise ValueError('This immutable source already has that measurement; reuse its event or propose a variant.')
            action['argument'] = argument
        except ValueError as error:
            now = time.time()
            event = {'id': f'e{len(self.events)+1}', 'tool': name, 'stage': stage,
                'source_sha256': self.proposals[-1]['source_sha256'] if self.proposals else None,
                'reason': action.get('reason', ''), 'started': now, 'finished': now,
                'requested_argument': action.get('argument', ''), 'result': {'status': 'repeat_refused', 'message': str(error)}}
            self.events.append(event)
            self.save()
            return event
        event = super().execute(action, stage)
        if name == 'read' and argument == 'guide':
            event['result'] = {'text': 'Current port capability: independent search correctness/per-shape timing, '
                'plus instrumented CUDA activity fallback after NCU permission refusal. No SM/DRAM/stall/occupancy '
                'counters are available. Do not subtract CUDA activity durations from independent search times. '
                'Bottleneck explanations are hypotheses. Test a model-authored variant, then keep/reject using '
                'real search results. All changes require submit/patch then evaluate; no extra actions.'}
        event['resolved_argument'] = argument
        self.save()
        return event

    def state(self, stage, remaining):
        references = []
        for e in self.events:
            if e['tool'] in {'read', 'research'} and e['result'].get('status') not in {'tool_error', 'repeat_refused'}:
                references.append({'id': e['id'], 'argument': e.get('resolved_argument'),
                    'result_excerpt': json.dumps(e['result'])[:2000]})
        return {'remaining_actions': remaining, 'allowed_tools': self.allowed(stage, remaining),
            'read_arguments_remaining': [a for a in ['api', 'example', 'guide'] if ('read', a) not in self.reference_keys()],
            'source_aliases': {'latest': self.proposals[-1]['source_sha256'] if self.proposals else None,
                'latest_correct': next((r['source_sha256'] for r in reversed(self.rows) if r['evaluation'].get('status') == 'correct'), None)},
            'retained_references': references[-2:], 'observations': observations(self),
            'phase_variant_tested': self.variant_tested(), 'phase_profile_observed': self.profile_ready(),
            'scope': 'Reason/draft prose is a hypothesis. Only source-bound host records are observations.'}


def action_schema(allowed):
    return create_model('AdmissibleAction', __base__=staged.Action, tool=(Literal[tuple(allowed)], ...))


def gate(stage, tools, review):
    errors, facts = validate_claims(review.claims, tools)
    if any(not set(c.evidence_ids) <= set(review.evidence_ids) for c in review.claims if c.kind != 'hypothesis'):
        errors.append('Claim ids must also appear in review evidence_ids.')
    passed, reason = staged.stage_gate(stage, tools.rows, tools.events, review)
    required = {'research': 'search_correct', 'profile_refine': 'activity_observed', 'shape_tune': 'search_compared'}[stage]
    if required not in {f['kind'] for f in facts if f['verified']}:
        errors.append('Accepted review lacks the phase-required bound observation: '+required)
    if stage == 'profile_refine' and not tools.variant_tested():
        errors.append('No distinct AST variant was actually evaluated in refinement.')
    if stage == 'shape_tune' and not tools.compared():
        errors.append('No distinct AST variants measured on three common search shapes.')
    return passed and not errors, reason if passed and not errors else '; '.join(errors) or reason, facts


def review_packet(tools):
    return {'events': review_events(tools.events), 'observations': observations(tools),
            'scope': 'Free-form tool reasons are hypotheses, not measurements. '
                     'Activity and search use different instrumentation and cannot establish launch overhead by subtraction. '
                     'Missing hardware counters are an explicit capability limit, not a stage failure by themselves.'}


def save_plan(path, plan, errors):
    atomic_json(path, {'plan': plan.model_dump(mode='json') if plan else None, 'errors': errors})
