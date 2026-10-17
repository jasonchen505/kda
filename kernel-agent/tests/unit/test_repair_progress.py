"""Refuse renames, but permit real type/binding/constant changes and ambiguity."""
from kernel_agent.repair_progress import alpha_identity, compiler_observation

SOURCE = '''import triton
import triton.language as tl
@triton.jit
def kernel(X, D, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    scale = D ** 0.5
    tl.store(X + i, scale)
def run(x):
    d = x.numel()
    kernel[(1,)](x, d, BLOCK=32)
    return x
'''


def test_renames_and_formatting_only_are_equivalent():
    changed = SOURCE.replace('kernel', 'calculate').replace('scale', 'factor').replace('BLOCK', 'TILE').replace('D', 'WIDTH')
    changed = changed.replace('d =', 'length =').replace('(x, d,', '(x, length,') + '# comment\n'
    assert alpha_identity(SOURCE) == alpha_identity(changed)


def test_real_repairs_are_not_refused():
    for changed in [SOURCE.replace('D, BLOCK', 'D: tl.constexpr, BLOCK'),
                    SOURCE.replace('** 0.5', '* 0.5'), SOURCE.replace('BLOCK=32', 'BLOCK=64'),
                    SOURCE.replace('(x, d,', '(x, 24,'), SOURCE.replace('x.numel()', 'x.shape[-1]'),
                    SOURCE.replace('import triton\n', 'import triton\nVALUE = 1\n')]:
        assert alpha_identity(SOURCE) != alpha_identity(changed)


def test_external_or_unresolved_names_remain_literal():
    assert alpha_identity(SOURCE.replace('scale = D', 'scale = UNDEFINED')) != alpha_identity(SOURCE)
    assert alpha_identity(SOURCE.replace('tl.store', 'tl.load')) != alpha_identity(SOURCE)
    assert alpha_identity(SOURCE.replace('scale = D', 'scale = locals()["D"]')) is None
    assert alpha_identity('def run(x):\n    return [a for a in x]\n') is None


def test_compiler_line_is_bound_by_unique_excerpt_not_relative_line():
    result = compiler_observation(SOURCE, {'status':'compile_error',
        'message':"at 3:14:\n    scale = D ** 0.5\n              ^\nAttributeError('no pow')"})
    assert result['source_line'] == 6 and result['function'] == 'kernel'
    assert result['parameters'] == ['X', 'D', 'BLOCK: tl.constexpr']
    ambiguous = compiler_observation(SOURCE+'    scale = D ** 0.5\n', {'status':'compile_error',
        'message':"    scale = D ** 0.5\n              ^\nAttributeError('no pow')"})
    assert ambiguous['source_line'] is None
