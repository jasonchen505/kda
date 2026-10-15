import torch
import triton
import triton.language as tl


@triton.jit
def _row_mean_1d(
    X,
    Y,
    stride,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    a = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
    total = tl.sum(a, axis=0)
    mean = total / N
    tl.store(Y + row, mean)


@triton.jit
def _row_mean_2d(
    X,
    Y,
    stride,
    N,
    ROWS_PER_BLOCK: tl.constexpr,
    COLS_PER_BLOCK: tl.constexpr,
):
    row_start = tl.program_id(0) * ROWS_PER_BLOCK
    rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
    cols = tl.arange(0, COLS_PER_BLOCK)
    mask = cols[None, :] < N
    ptrs = X + rows[:, None] * stride + cols[None, :]
    a = tl.load(ptrs, mask=mask, other=0.).to(tl.float32)
    total = tl.sum(a, axis=1)
    mean = total / N
    tl.store(Y + rows, mean)


def run(x):
    assert x.is_cuda
    assert x.dim() == 2
    assert x.is_contiguous()
    rows, width = x.shape
    out = torch.empty((rows,), dtype=x.dtype, device=x.device)
    if rows == 0 or width == 0:
        return out
    if width <= 1024:
        BLOCK_SIZE = max(1, triton.next_power_of_2(width))
        grid = (rows,)
        _row_mean_1d[grid](x, out, width, width, BLOCK_SIZE=BLOCK_SIZE)
    else:
        COLS_PER_BLOCK = 1024
        ROWS_PER_BLOCK = max(1, triton.cdiv(width, COLS_PER_BLOCK))
        grid = (triton.cdiv(rows, ROWS_PER_BLOCK),)
        _row_mean_2d[grid](x, out, width, width, ROWS_PER_BLOCK=ROWS_PER_BLOCK, COLS_PER_BLOCK=COLS_PER_BLOCK)
    return out
