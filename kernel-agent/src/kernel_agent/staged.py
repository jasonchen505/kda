"""4090 staged-agent contracts: distinct planning, model-chosen tools and fresh review."""
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from kernel_agent import grounded_repair, operational, repair
from kernel_agent.protocol import decision, prefers

ARMS = ('fixed', 'staged', 'staged_transfer')
STAGES = ('research', 'profile_refine', 'shape_tune')
CALL_CAP = 24
CANDIDATE_CAP = 6


class Edit(BaseModel):
    model_config = ConfigDict(extra='forbid')
    start_line: int
    end_line: int
    replacement_lines: list[str]


class Action(BaseModel):
    """A real hosted tool request, not a claim that a command already ran."""
    model_config = ConfigDict(extra='forbid')
    tool: Literal['read', 'research', 'submit', 'patch', 'evaluate', 'profile', 'finish']
    argument: str
    code_lines: list[str]
    edits: list[Edit]
    reason: str


class Document(BaseModel):
    model_config = ConfigDict(extra='forbid')
    text: str


class Review(BaseModel):
    model_config = ConfigDict(extra='forbid')
    accept: bool
    evidence_ids: list[str]
    missing: str
    next_step: str


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def select(rows):
    best = None
    for row in rows:
        if prefers(decision(row['evaluation']), decision(best['evaluation']) if best else None):
            best = row
    return best


def visible_result(row, enriched):
    result = row.get('evaluation', {})
    if enriched:
        evidence = operational.diagnose(result)
        return {'id': row['id'], 'sha256': row.get('source_sha256'),
                'evidence': evidence, 'result': result}
    # Keep identical oracle verdict/timing. Only the added interpretation differs.
    return {'id': row['id'], 'sha256': row.get('source_sha256'), 'result': result}


def source_context(rows, enriched):
    parent = rows[-1] if rows else None
    if not parent:
        return []
    parts = [('parent', 'Latest immutable source, SHA ' + parent['source_sha256'] + ':\n'
              + parent['code'], True)]
    if enriched:
        facts = grounded_repair.facts(parent['code'])
        parts.append(('source_facts', 'Partial source observations; not semantic proof:\n'
                      + json.dumps(facts, separators=(',', ':')), False))
    return parts


def brief(task_id, contract, stage, arm):
    return (f'Task {task_id}. Stage {stage}.\n{contract}\n'
            'RTX4090 Ada, Torch2.6/Triton3.2. Only torch allocation and metadata; compute in Triton. '
            'FP32 atol2e-5/rtol2e-4; BF16 atol/rtol.015. Fresh contiguous output, immutable inputs. '
            'Search promotion: geomean>=1.05, worst>=.90 vs torch.compile. '
            'Final workloads are unavailable and must never be queried. '
            'B200 TMEM/tcgen05 and Blackwell-only recipes do not transfer to Ada. '
            'Tools execute through the host; model statements are not measurements. '
            f'Arm {arm}. All roles use the same Qwen weights in separate fresh sessions; '
            'reviewer independence means context separation, not independent model errors.\n')


TOOLS = (
    'Choose ONE hosted tool via JSON, fill unused code_lines/edits with []. No native tool API. '
    'read argument=api|example|guide|source|source:SHA|evidence; research argument=short keyword for pinned KernelWiki; '
    'submit code_lines=complete Python source defining run, argument=""; '
    'patch argument=parent SHA, edits=[{"start_line":1,"end_line":2,"replacement_lines":[...]}] using '
    '1-based inclusive NONOVERLAPPING ranges in the exact parent (only transfer arm); '
    'evaluate argument=latest source SHA; profile argument=latest SEARCH-CORRECT source SHA; '
    'finish requests fresh review, does not accept a stage. Never submit documentation in place of code. '
    'A source proposal is not automatically evaluated. Use evaluate to obtain actual evidence. '
    'Rejected submitted source is preserved exactly for repair. After patch rejection you may '
    'submit full source in a remaining action; no extra retries are granted. '
    'You may inspect, query, profile, edit/backtrack and vary configurations inside these tools. '
    'Phase1: correct implementation. Phase2: measured diagnosis and improvement. '
    'Phase3: compare shapes; implement shape-based configuration/dispatch only if evidence supports it. '
    'Do not invent results, use another project candidate, or edit the oracle. '
    'A failed profiler is an explicit blocker, never a measured performance diagnosis.'
)


def stage_gate(stage, rows, events, review):
    """A reviewer cannot bless absent, stale, failed or unmeasured work."""
    evaluations = [r for r in rows if r.get('evaluation', {}).get('status') == 'correct']
    known = {e['id'] for e in events}
    cited = set(review.evidence_ids)
    if not review.accept or not cited or not cited <= known or not evaluations:
        return False, 'review rejected, missing correct source, or nonexistent evidence citation'
    correct_shas = {r['source_sha256'] for r in evaluations}
    evidence = [e for e in events if e['id'] in cited]
    if not any(e['tool'] == 'evaluate' and e['result'].get('status') == 'correct'
               and e['source_sha256'] in correct_shas for e in evidence):
        return False, 'review must cite actual successful search evaluation'
    if stage == 'profile_refine' and not any(e['tool'] == 'profile'
            and e['result'].get('status') in {'profiled', 'profiled_activity'} and e['source_sha256'] in correct_shas
            for e in evidence):
        return False, 'profiling stage needs actual source-bound counters or explicitly degraded CUDA activity'
    if stage == 'shape_tune':
        compared = {e['source_sha256'] for e in evidence if e['tool'] == 'evaluate'
                    and e['result'].get('status') == 'correct'
                    and len(e['result'].get('timings', [])) >= 3}
        if len(compared) < 2:
            return False, 'shape stage needs two distinct correct sources measured on three search shapes'
    return True, 'review citations and deterministic stage requirements satisfied'


def checked_path(directory, name):
    base = Path(directory).resolve()
    path = (base / name).resolve()
    if not path.is_relative_to(base):
        raise ValueError('tool path escapes the isolated workspace')
    return path


def fixed_sections(task_id, contract, history):
    from kernel_agent.staged_tools import compact
    visible = [{**r, 'evaluation': compact(r['evaluation'], probes=False)} for r in history]
    parts = repair.sections(task_id, contract, 'baseline', visible, 'baseline')
    probe = common_probe(history[-1]['evaluation']) if history else None
    if probe:
        parts.append(('common_probe', 'Exact finite search witness:\n'+json.dumps(probe), True))
    return parts


def common_probe(evaluation):
    if evaluation.get('status') == 'correct':
        return None
    probes = evaluation.get('semantic_probes', [])
    return probes[0] if probes else evaluation.get('counterexample')
