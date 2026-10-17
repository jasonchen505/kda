"""Evidence-directed next actions and conservative complete-reply recovery."""
from kernel_agent import staged
from kernel_agent import staged_contract_v3 as previous
from kernel_agent.grounded_repair import facts
from kernel_agent.staged_contract_v3 import (  # noqa: F401
    CAPABILITY,GOALS,MODULE_CONTRACT,Plan,Review,action_schema,compact_verdict,diagnose_reply,
    gate,plan_issues,review_schema,unwrap,wire_format,
)


class Tools(previous.Tools):
    def allowed(self,stage,remaining):
        # Required evidence first; every choice still requires a real model action.
        if self.readiness(stage)['ready']:
            return ['finish']
        if stage=='profile_refine':
            profiles=[e for e in self.events[self.phase_start:] if e['tool']=='profile'
                and e['source_sha256']==self.phase_entry and e['result'].get('status') in {'profiled','profiled_activity'}]
            if not profiles and self.phase_entry and self.profiles<2:
                return ['profile']
        if stage=='shape_tune' and self.comparison():
            return ['compare']
        if self.submissions<staged.CANDIDATE_CAP and self.gpu_calls<9:
            return ['submit_and_evaluate']
        return ['finish']

    def packet(self,stage):
        packet=super().packet(stage)
        packet['next_action_contract']={
            'research':'Submit complete source. After failure, repair the actual failing source; no new self-tests.',
            'profile_refine':'Profile accepted baseline first. Then change an executable launch configuration or implementation and submit the WHOLE changed module. Copying the baseline does not create a variant.',
            'shape_tune':'With two correct measured variants, compare now. Otherwise generate one genuinely changed implementation, preserving contracted semantics.'}[stage]
        if self.events and self.events[-1]['result'].get('status')=='semantic_repeat':
            packet['repeat_feedback']='The submitted executable AST is identical to an existing source. Change the executable configuration/implementation in code_lines, not the explanation, comments, names or formatting. No GPU evaluation ran.'
        return packet

    def source_facts(self):
        # Optional bounded observations; no code is filled in and no task-id answer map.
        if not self.proposals:
            return None
        return facts(self.proposals[-1]['code'])
