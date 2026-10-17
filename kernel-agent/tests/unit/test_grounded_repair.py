"""Source facts must expose evidence without executing or rewriting a candidate."""
import json

import pytest

from kernel_agent import grounded_repair, span_repair
from kernel_agent.protocol import sha

SOURCE = '''import triton
import triton.language as tl
@triton.jit
def kernel(X, Y, M, N, BLOCK: tl.constexpr, OUT_DTYPE: tl.constexpr):
    m = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    n = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    xp = X + m[:, None] * N + n[None, :]
    value = tl.load(xp)
    yp = Y + n[:, None] * M + m[None, :]
    tl.store(yp, value)
def run(x):
    out = x
    kernel[(1, 1)](x, out, 3, 5, BLOCK=32, OUT_DTYPE=x.dtype)
    return out
'''


def test_compile_catches_duplicate_formals_beyond_ast_parse():
    value = grounded_repair.facts(SOURCE.replace('M, N, BLOCK:', 'M, M, BLOCK:'))
    assert value['python_compile']['message'].startswith('duplicate argument')
    assert value['source_sha256'] == sha(SOURCE.replace('M, N, BLOCK:', 'M, M, BLOCK:'))


def test_signature_launch_dtype_and_store_dependencies_are_source_bound():
    value = grounded_repair.facts(SOURCE)
    assert value['python_compile'] == 'passed'
    assert 'OUT_DTYPE:tl.constexpr' in value['functions'][0]['parameters']
    assert value['launches'][0]['bindings']['OUT_DTYPE'] == 'x.dtype'
    deps = value['stores'][0]['dependencies']
    assert deps['value'] == 'tl.load(xp)' and deps['xp'] == 'X + m[:, None] * N + n[None, :]'
    assert 'n[:, None]' in deps['yp']
    assert len(json.dumps(value, separators=(',', ':'))) <= 3300


def test_malformed_source_is_reported_and_arbitrary_source_never_executes():
    assert isinstance(grounded_repair.facts('def bad(')['python_compile'], dict)
    assert grounded_repair.facts('raise RuntimeError("must never run")')['python_compile'] == 'passed'


def test_dynamic_launch_expansion_is_explicitly_partial():
    result = grounded_repair.facts(SOURCE.replace('x, out, 3, 5, BLOCK=32, OUT_DTYPE=x.dtype', '*args, **kwargs'))
    assert result['launches'][0]['partial']


def test_unresolved_kernel_name_and_bounded_slices():
    result = grounded_repair.facts(SOURCE.replace('tl.load(xp)', 'tl.load(xp) + missing_width'))
    assert result['functions'][0]['unresolved_names_partial'] == ['missing_width']
    result = grounded_repair.facts(SOURCE.replace('tl.store(yp, value)', '\n'.join(['tl.store(yp, value)'] + ['    tl.store(yp, value)'] * 4)))
    assert len(result['stores']) <= 2 and result['stores_omitted'] >= 3


def test_control_is_unchanged_and_new_packet_never_changes_schema_or_source():
    parent = {'code': SOURCE, 'source_sha256': sha(SOURCE), 'request_id': 'initial',
              'evaluation': {'status': 'numeric_error'}}
    history = [parent]
    cap = {'language_symbols': ['load', 'store']}
    base = span_repair.context('transpose', 'contract', history, cap, 'span_patch')
    assert grounded_repair.context('transpose', 'contract', history, cap, 'span_patch', 'indexed') == base
    parts, settings = grounded_repair.context('transpose', 'contract', history, cap, 'span_patch', 'grounded')
    assert settings == base[1] and parts[:-1] == base[0]
    assert parts[-1][0] == 'source_facts' and parts[-1][2]
    assert parent['code'] == SOURCE
    assert grounded_repair.assignment('grounded', history) == ('span_patch', parent)
    history.append({'mode': 'span_patch', 'evaluation': {'status': 'patch_error'}})
    assert grounded_repair.assignment('grounded', history) == ('full_source_fallback', parent)
    with pytest.raises(ValueError):
        grounded_repair.assignment('unknown', history)
