import torch
import triton
import triton.language as tl


@triton.jit
def _l2_norm_kernel(
    X,
    Y,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride
    Y += row * stride
    ss = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        ss += x * x
    total = tl.sum(ss, axis=0)
    rstd = tl.rsqrt(total + eps)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        y = x * rstd
        tl.store(Y + cols, y, mask=mask)


def run(x):
    assert x.is_cuda
    assert x.is_contiguous()
    assert x.dim() == 2
    rows, width = x.shape
    out = torch.empty_like(x)
    eps = 1e-6
    BLOCK_SIZE = 1024
    grid = (rows,)
    _l2_norm_kernel[grid](
        x, out, width, width, eps, BLOCK_SIZE
    )
    return out
