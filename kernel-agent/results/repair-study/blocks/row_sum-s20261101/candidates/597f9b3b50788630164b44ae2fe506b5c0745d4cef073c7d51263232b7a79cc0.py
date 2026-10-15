import torch
import triton
import triton.language as tl


@triton.jit
def _row_sum_single(
    X,
    Y,
    stride_x,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride_x
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    a = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
    total = tl.sum(a, axis=0)
    tl.store(Y + row, total)


@triton.jit
def _row_sum_multi(
    X,
    Y,
    stride_x,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    col_blk = tl.program_id(1)
    X += row * stride_x + col_blk * BLOCK_SIZE
    cols = col_blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    a = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
    total = tl.sum(a, axis=0)
    tl.atomic_add(Y + row, total)


def run(x):
    assert x.is_cuda
    assert x.dim() == 2
    assert x.is_contiguous()
    rows, cols = x.shape
    out = torch.empty((rows,), dtype=x.dtype, device=x.device)
    if rows == 0 or cols == 0:
        return out
    BLOCK_SIZE = triton.next_power_of_2(cols)
    if BLOCK_SIZE <= 1024:
        grid = (rows,)
        _row_sum_single[grid](x, out, x.stride(0), cols, BLOCK_SIZE=BLOCK_SIZE)
    else:
        BLOCK_SIZE = 1024
        num_col_blocks = triton.cdiv(cols, BLOCK_SIZE)
        out.zero_()
        grid = (rows, num_col_blocks)
        _row_sum_multi[grid](x, out, x.stride(0), cols, BLOCK_SIZE=BLOCK_SIZE)
    return out
