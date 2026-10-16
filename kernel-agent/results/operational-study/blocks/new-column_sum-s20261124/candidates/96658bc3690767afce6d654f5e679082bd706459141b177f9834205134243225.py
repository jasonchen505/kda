import torch
import triton
import triton.language as tl


@triton.jit
def _sum_rows_kernel(
    X,
    Y,
    rows,
    width,
    row_chunk,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)
    cols = pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < width
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    row_start = pid_row * ROWS_PER_PROG
    row_end = tl.minimum(row_start + ROWS_PER_PROG, rows)
    for r in range(row_start, row_end):
        ptr = X + r * width + cols
        val = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
        acc += val
    tl.atomic_add(Y + cols, acc, mask=mask)


def run(x):
    assert x.is_cuda
    assert x.dim() == 2
    assert x.is_contiguous()
    rows, width = x.shape
    y = torch.zeros((width,), dtype=torch.float32, device=x.device)
    BLOCK_SIZE = 1024
    ROWS_PER_PROG = 16
    grid = (triton.cdiv(rows, ROWS_PER_PROG), triton.cdiv(width, BLOCK_SIZE))
    _sum_rows_kernel[grid](x, y, rows, width, ROWS_PER_PROG, BLOCK_SIZE=BLOCK_SIZE, ROWS_PER_PROG=ROWS_PER_PROG, DTYPE=x.dtype)
    return y.to(x.dtype)
