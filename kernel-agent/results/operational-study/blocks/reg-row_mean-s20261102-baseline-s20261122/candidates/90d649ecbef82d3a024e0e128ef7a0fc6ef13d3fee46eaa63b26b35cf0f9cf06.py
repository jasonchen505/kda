import torch
import triton
import triton.language as tl


@triton.jit
def _row_mean_kernel(
    X,
    Y,
    stride,
    N,
    BLOCK_SIZE: tl.constexpr,
    N_ITERS: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in tl.static_range(N_ITERS):
        off = i * BLOCK_SIZE
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        a = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        acc += a
    total = tl.sum(acc, axis=0)
    mean = total / N
    tl.store(Y + row, mean)


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def run(x):
    assert x.is_cuda
    assert x.dim() == 2
    assert x.is_contiguous()
    rows, width = x.shape
    out = torch.empty((rows,), dtype=x.dtype, device=x.device)
    if rows == 0 or width == 0:
        return out
    block = _next_pow2(width)
    if block < 1024:
        block = 1024
    if block > 8192:
        block = 8192
    n_iters = triton.cdiv(width, block)
    grid = (rows,)
    _row_mean_kernel[grid](x, out, width, width, BLOCK_SIZE=block, N_ITERS=n_iters)
    return out
