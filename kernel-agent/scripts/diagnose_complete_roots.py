"""Post-hoc protocol comparison on identical received bytes; no model/GPU/source rewrite."""
import argparse
import collections
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from kernel_agent import line_protocol,staged_contract_v3 as c  # noqa: E402
from kernel_agent.campaign import atomic_json,digest  # noqa: E402
from kernel_agent.structured_recovery import recover  # noqa: E402
from kernel_agent.protocol import admit  # noqa: E402


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--directory',type=Path,required=True)
    a=p.parse_args()
    rows=[]
    for calls_path in a.directory.glob('blocks/*/contract/calls.json'):
        folder=calls_path.parent
        for ticket in json.loads(calls_path.read_text()):
            if ticket['role']!='writer':
                continue
            path=folder/'requests'/(ticket['id']+'.response.json')
            receipt=folder/'requests'/(ticket['id']+'.json')
            if not path.exists() or not receipt.exists():
                continue
            r=json.loads(receipt.read_text())
            if r['status']!='completed':
                continue
            response=json.loads(path.read_text())
            assert digest(response)==r['response_sha256']
            content=response['choices'][0]['message']['content']
            prop=ticket['schema']['properties']['tool']
            allowed=prop.get('enum',[prop['const']] if 'const' in prop else [])
            schema=c.action_schema(allowed)
            assert schema.model_json_schema()==ticket['schema']
            row={'block':folder.parent.name,'call':ticket['id'],'strict_valid':False,
                 'complete_first_root_valid':False,'admissible_model_source':False,'tail_chars':0,
                 'raw_response_sha256':r['response_sha256']}
            recovered=recover(content,schema,allowed)
            row['unique_contract_valid']=bool(recovered)
            row['recovered_source_admissible']=False
            if recovered:
                action,meta=recovered
                row['recovery']=meta
                if action.tool=='submit_and_evaluate':
                    try:
                        proposal=line_protocol.proposal(json.dumps({'draft':'exact model bytes','plan':'protocol diagnostic','code_lines':action.code_lines}))
                        row['recovered_source_sha256']=admit(proposal['code'])
                        row['recovered_source_admissible']=True
                    except (ValueError,SyntaxError):
                        pass
            try:
                schema.model_validate_json(content)
                row['strict_valid']=True
            except ValueError:
                pass
            try:
                stripped=content.lstrip()
                assert stripped.startswith('{')
                value,end=json.JSONDecoder().raw_decode(stripped)
                action=schema.model_validate(value)
                row.update(complete_first_root_valid=True,tail_chars=len(stripped[end:]),
                           prefix_digest=digest(value),tool=action.tool)
                if action.tool=='submit_and_evaluate':
                    proposal=line_protocol.proposal(json.dumps({'draft':'model bytes','plan':'protocol diagnostic',
                                                              'code_lines':action.code_lines}))
                    row['source_sha256']=admit(proposal['code'])
                    row['admissible_model_source']=True
            except (ValueError,SyntaxError,AssertionError) as error:
                row['reason']=str(error)[:180]
            rows.append(row)
    counts=collections.Counter()
    for r in rows:
        for k in ['strict_valid','complete_first_root_valid','admissible_model_source','unique_contract_valid','recovered_source_admissible']:
            counts[k]+=r[k]
        counts['additional_complete_roots']+=r['complete_first_root_valid'] and not r['strict_valid']
    result={'received_writer_replies':len(rows),'counts':dict(counts),'rows':rows,
        'scope':'Post-hoc parser diagnostic on identical received bytes. No new model/GPU calls; complete-root admissibility is not numeric correctness or full-flow success; frozen cohort unchanged.'}
    result['summary_sha256']=digest(result)
    atomic_json(ROOT/'results/structural-recovery-diagnostic.json',result)
    print(json.dumps({'replies':len(rows),'counts':dict(counts),'scope':result['scope']}))


if __name__=='__main__':
    main()
