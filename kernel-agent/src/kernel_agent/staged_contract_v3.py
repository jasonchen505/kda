"""Make argument preconditions and received validation failures visible to the next role."""
import json
from typing import Literal

from pydantic import ValidationError,create_model

from kernel_agent import staged
from kernel_agent import staged_contract_v2 as previous
from kernel_agent.staged_contract_v2 import (  # noqa: F401
    CAPABILITY,GOALS,MODULE_CONTRACT,Plan,Review,gate,plan_issues,review_schema,unwrap,
)

ARGUMENT_RULES={
    'submit_and_evaluate':{'argument':'','code_lines':'complete kernel module; never a source alias'},
    'profile':{'argument':'latest_correct','code_lines':[]},
    'compare':{'argument':'','code_lines':[]},'finish':{'argument':'','code_lines':[]},
    'read':{'argument':['api','example','guide'],'code_lines':[]},
    'research':{'argument':'short literal query,1..90characters','code_lines':[]},
}


def action_schema(allowed):
    fields={'tool':(Literal[tuple(allowed)],...)}
    if 'research' not in allowed:
        arguments=[]
        for name in allowed:
            value=ARGUMENT_RULES[name]['argument']
            arguments.extend(value if isinstance(value,list) else [value])
        fields['argument']=(Literal[tuple(dict.fromkeys(arguments))],...)
    return create_model('ExecutableKernelAction',__base__=previous.Action,**fields)


def wire_format(role,document=False,schema=None):
    if role=='writer':
        prop=schema.model_json_schema()['properties']['tool'] if schema else {}
        allowed=prop.get('enum',[prop['const']] if 'const' in prop else list(ARGUMENT_RULES))
        examples=[]
        for tool in allowed:
            argument=ARGUMENT_RULES[tool]['argument']
            if tool=='research':
                argument='block'
            elif isinstance(argument,list):
                argument=argument[0]
            examples.append({'tool':tool,'argument':argument,
                'code_lines':['<replace with complete Python source, one line per item>'] if tool=='submit_and_evaluate' else [],
                'reason':'short rationale'})
        return ('Return ONE flat JSON action. Valid shapes for the CURRENT allowed tools: '+
            json.dumps(examples,separators=(',',':'))+'. Choose ONE object, not the surrounding list. '
            'Replace the source placeholder with complete kernel code. Source submission ALWAYS has argument=""; '
            'never use latest_correct, a SHA or the tool name as its argument. Other tools have code_lines=[]. '
            'reason<=240characters. No name key, action wrapper, markdown, trailing prose or copied instructions.')
    text=previous.wire_format(role,document)
    if role=='planner' and not document:
        text+=' hypothesis<=400characters, each purpose<=140, negative_result<=300.'
    elif role=='reviewer':
        text+=' reason<=500characters, at most3hypotheses each<=300.'
    return text


def diagnose_reply(content,schema):
    try:
        value=json.loads(content)
    except json.JSONDecodeError as error:
        return {'errors':[{'kind':'invalid_json','line':error.lineno,'column':error.colno,'message':error.msg}]}
    try:
        schema.model_validate(value)
    except ValidationError as error:
        tool=value.get('tool') if isinstance(value,dict) else None
        argument=value.get('argument') if isinstance(value,dict) else None
        return {'received_tool':tool[:90] if isinstance(tool,str) else type(tool).__name__,
                'received_argument':argument[:90] if isinstance(argument,str) else type(argument).__name__,
                'received_value_truncated':any(isinstance(v,str) and len(v)>90 for v in [tool,argument]),
                'errors':[{'location':list(e['loc']),'kind':e['type'],'message':e['msg'][:300]}
                          for e in error.errors(include_input=False)[:4]]}
    return {'errors':[]}


def compact_verdict(value,probes=False):
    result=previous.compact_verdict(value,probes=probes)
    witness=staged.common_probe(value)
    if witness and len(json.dumps(witness))<=2500:
        result['finite_search_witness']=witness
    return result


class Tools(previous.Tools):
    def packet(self,stage):
        packet=super().packet(stage)
        packet['argument_contracts']=ARGUMENT_RULES
        if self.events and self.events[-1]['tool']=='decode_reply':
            packet['latest_received_validation_failure']=self.events[-1]['result'].get('validation_diagnostics',{})
        return packet
