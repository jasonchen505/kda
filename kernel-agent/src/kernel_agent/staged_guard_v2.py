"""Close parameter/schema and measured-transition gaps in the first guarded interface."""
from typing import Literal

from pydantic import Field, create_model

from kernel_agent.staged_guard import (  # noqa: F401
    Claim, Plan, Review, Step, ast_identity, gate, import_parent, observations,
    review_packet, save_plan, validate_claims,
)
from kernel_agent.staged_guard import Tools as FirstTools, plan_issues as first_plan_issues


class Tools(FirstTools):
    def variant_tested(self):
        entry = next((r for r in self.proposals if r['source_sha256'] == self.phase_entry), None)
        identity = ast_identity(entry['code']) if entry else None
        # A refused tool request is not an actual compiler/numerical evaluation.
        observed = {'correct', 'numeric_error', 'compile_error', 'runtime_error', 'mutation_error', 'alias_error'}
        return any(e['tool'] == 'evaluate' and not e.get('shared_parent')
                   and e['result'].get('status') in observed
                   and e.get('resolved_argument') == e['source_sha256']
                   and any(r['source_sha256'] == e['source_sha256'] and ast_identity(r['code']) != identity
                           for r in self.proposals) for e in self.events[self.phase_start:])


def plan_schema(limit):
    return create_model('BudgetedPlan', __base__=Plan,
                        steps=(list[Step], Field(min_length=1, max_length=limit)))


def action_schema(allowed):
    fields = {'tool': (Literal[tuple(allowed)], ...)}
    # Research queries need arbitrary text. Once reference actions are closed, source
    # arguments can only be the advertised host-resolved aliases, never copied prose.
    if 'research' not in allowed:
        args = ['']
        if 'read' in allowed:
            args += ['api', 'example', 'guide']
        if set(allowed) & {'patch', 'evaluate', 'profile'}:
            args += ['latest', 'latest_correct']
        fields['argument'] = (Literal[tuple(args)], ...)
    from kernel_agent.staged import Action
    return create_model('ReferencedAction', __base__=Action, **fields)


def plan_issues(plan, limit, stage, have_parent=False, comparison_ready=False):
    errors = first_plan_issues(plan, limit, stage, have_parent, comparison_ready)
    if plan:
        for step in plan.steps:
            if step.tool == 'read' and step.argument not in {'api', 'example', 'guide'}:
                errors.append('Read schema and executor agree: api/example/guide only; source/evidence already retained.')
    return list(dict.fromkeys(errors))
