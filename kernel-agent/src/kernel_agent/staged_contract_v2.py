"""Unambiguous kernel-module experiments, actionable source-policy diagnostics and lean evidence."""
import ast
import copy
from typing import Annotated, Literal

from pydantic import Field, create_model,model_validator

from kernel_agent import staged_contract as first
from kernel_agent.protocol import TORCH_HOST_CALLS
from kernel_agent.representative_verified import VerifiedMixin
from kernel_agent.staged_contract import CAPABILITY, gate, unwrap  # noqa: F401
from kernel_agent.staged_tools import compact

MODULE_CONTRACT = ('KERNEL MODULE ONLY: imports, Triton JIT helpers, and run entry point. '
    'Do not include tests, __main__, example invocations, random input generation, test assertions, '
    'reference implementations or benchmarks. The HOST owns all testing outside this module. '
    'torch is only for allocation and metadata; no torch.randn/manual_seed/allclose/compute calls anywhere. '
    'One Python line per code_lines item, with indentation preserved. Never pass torch.dtype as a '
    'kernel constexpr type: store converts to output pointer dtype automatically; tl types are kernel types.')
GOALS={k:v.replace('through test.', 'through submit_and_evaluate.').replace('test an AST-distinct',
       'submit_and_evaluate an AST-distinct') for k,v in first.GOALS.items()}
Line = Annotated[str, Field(pattern=r'^[^\r\n]*$')]
Short = Annotated[str, Field(max_length=300)]


class Step(first.Step):
    tool: Literal['submit_and_evaluate','profile','compare','read','research','finish']
    purpose: str = Field(max_length=140)


class Plan(first.Plan):
    hypothesis: str = Field(max_length=400)
    steps: list[Step] = Field(min_length=1,max_length=4)
    negative_result: str = Field(max_length=300)


class Action(first.Action):
    tool: Literal['submit_and_evaluate','profile','compare','read','research','finish']
    code_lines: list[Line] = Field(max_length=120)
    reason: str = Field(max_length=240)
    argument: str = Field(max_length=90)

    @model_validator(mode='after')
    def check_operation(self):
        fixed={'submit_and_evaluate':[''],'profile':['latest_correct'],'compare':[''],
               'read':['api','example','guide'],'finish':['']}
        if self.tool in fixed and self.argument not in fixed[self.tool]:
            raise ValueError('Invalid argument for '+self.tool+': '+str(fixed[self.tool]))
        if self.tool!='submit_and_evaluate' and self.code_lines:
            raise ValueError('Non-source operations require code_lines=[]')
        if self.tool=='submit_and_evaluate' and not self.code_lines:
            raise ValueError('Source operation requires complete code_lines')
        return self


class Review(first.Review):
    hypotheses: list[Short] = Field(max_length=3)
    reason: str = Field(max_length=500)


def action_schema(allowed):
    # Keep one flat wire shape across every action. Requested strict schemas are not
    # assumed to be enforced by every local backend; host validation is authoritative.
    return create_model('KernelAction',__base__=Action,tool=(Literal[tuple(allowed)],...))


def wire_format(role,document=False):
    if role=='writer':
        return ('Return ONE JSON object with EXACTLY four keys: tool,argument,code_lines,reason. '
            'Shape: {"tool":"chosen_allowed_tool","argument":"tool_argument","code_lines":["one Python line","next line"],"reason":"brief"}. '
            'Those values are placeholders: use an allowed tool and its real argument. For non-source tools code_lines=[]. '
            'Do not use name or an action wrapper. No markdown, extra keys, trailing prose or copied instructions. End immediately after the JSON object.')
    if document:
        return 'Return exactly {"text":"your draft"}, no extra keys, prose or markdown outside JSON.'
    if role=='planner':
        return ('Return exactly the JSON keys hypothesis,measurement,steps,negative_result. '
            'measurement is exactly search_only,activity_and_search,or common_shape_search for the active phase. '
            'Each steps item has exactly tool,purpose. No extra fields or trailing prose.')
    return ('Return exactly JSON keys accept,evidence_ids,selected_candidate,decision,limitation,hypotheses,reason. '
            'accept is a JSON boolean. evidence_ids is the array of actual required event ids. '
            'selected_candidate is the recorded candidate id, not its SHA. decision is exactly '
            'keep_generic,tested_negative,tested_candidate,or insufficient. limitation is exactly '
            'activity_only_no_causal_attribution. hypotheses is an array of unverified strings or[]. '
            'reason is a short string. No extra fields, markdown or trailing prose.')


def review_schema(tools):
    ids=[e['id'] for e in tools.events] or ['none']
    sources=[r['id'] for r in tools.rows if r['evaluation'].get('status')=='correct']+['none']
    return create_model('BoundKernelReview',__base__=Review,
        evidence_ids=(list[Literal[tuple(ids)]],...),selected_candidate=(Literal[tuple(sources)],...))


def plan_issues(plan,stage,compared):
    data=plan.model_dump(mode='json')
    for step in data['steps']:
        if step['tool']=='submit_and_evaluate':
            step['tool']='test'
    return first.plan_issues(first.Plan.model_validate(data),stage,compared)


def source_issues(source):
    """Actual source locations only; no task-id answer map or host code correction."""
    try:
        tree=ast.parse(source)
    except SyntaxError as error:
        return [{'kind':'syntax','line':error.lineno,'message':error.msg}]
    issues=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name):
            if node.func.value.id=='torch' and node.func.attr not in TORCH_HOST_CALLS:
                issues.append({'kind':'forbidden_host_call','line':node.lineno,'call':'torch.'+node.func.attr})
        if isinstance(node,ast.If) and any(isinstance(n,ast.Name) and n.id=='__name__' for n in ast.walk(node.test)):
            issues.append({'kind':'embedded_self_test','line':node.lineno,'message':'Omit this self-test/main section from the kernel module; host owns tests.'})
    return issues[:12]


def trim_timings(rows):
    return [{k:v for k,v in row.items() if k in {'shape','dtype','causal','candidate_us','compiled_us'}} for row in rows]


def compact_verdict(value,probes=False):
    result=compact(value,probes=probes)
    if result.get('timings'):
        result['timings']=trim_timings(result['timings'])
    return result


class Tools(VerifiedMixin,first.Tools):
    def allowed(self,stage,remaining):
        return ['submit_and_evaluate' if name=='test' else name for name in super().allowed(stage,remaining)]

    def execute(self,action,stage):
        value=unwrap(action)
        value=value.model_dump(mode='json') if hasattr(value,'model_dump') else dict(value)
        if value['tool']=='submit_and_evaluate':
            issues=source_issues('\n'.join(value['code_lines'])+'\n')
            if any(i['kind']=='embedded_self_test' for i in issues):
                # Preserve the exact model proposal and spend its submission slot, but
                # never run an embedded self-test module as a scored kernel.
                event=super().execute({'tool':'submit','argument':'','edits':[],
                    'code_lines':value['code_lines'],'reason':value['reason']},stage)
                if event['result'].get('status')=='admitted':
                    row=self.rows[-1]
                    row['evaluation']={'status':'policy_error','message':'Kernel module contains a self-test/main block; remove it in a model-authored revision.',
                                       'source_policy_issues':issues}
                    return self.emit('test',stage,row['evaluation'],value['reason'],row['source_sha256'])
                event['result']['source_policy_issues']=issues
                self.save()
                return event
            value['tool']='test'
        event=super().execute(value,stage)
        if event['result'].get('status')=='tool_error' and self.proposals:
            issues=source_issues(self.proposals[-1]['code'])
            if issues:
                event['result']['source_policy_issues']=issues
                self.save()
        return event

    def packet(self,stage):
        packet=super().packet(stage)
        for item in packet['observations']:
            if item.get('timings'):
                item['timings']=trim_timings(item['timings'])
        if packet.get('comparison'):
            for row in packet['comparison']['candidates']:
                row['timings']=trim_timings(row['timings'])
        packet['kernel_module_contract']=MODULE_CONTRACT
        packet['current_source_issues']=source_issues(self.proposals[-1]['code']) if self.proposals else []
        # References must actually reach the next role. Metadata saying that a read
        # happened is not a substitute for its returned API/knowledge content.
        refs=[e for e in self.events if e['tool'] in {'read','research'}]
        packet['retained_references']=[]
        for event in refs[-2:]:
            result=event['result']
            text=result.get('text','')
            if not text and result.get('pages'):
                text='\n'.join(p['path']+': '+p['excerpt'][:600] for p in result['pages'][:2])
            packet['retained_references'].append({'id':event['id'],'text_excerpt':text[:1800],
                'truncated':len(text)>1800,'scope':'Actual returned reference excerpt, not a measured claim'})
        if self.events:
            result=copy.deepcopy(self.events[-1]['result'])
            packet['latest_tool_result']={k:v for k,v in result.items() if k in {'status','message','source_policy_issues'}}
            if 'message' in packet['latest_tool_result']:
                packet['latest_tool_result']['message']=str(packet['latest_tool_result']['message'])[-800:]
        return packet
