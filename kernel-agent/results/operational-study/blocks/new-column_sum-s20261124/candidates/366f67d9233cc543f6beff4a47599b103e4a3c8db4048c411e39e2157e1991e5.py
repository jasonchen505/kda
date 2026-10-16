import torch
import triton
import triton.language as tl


@triton.jit
def _sum_rows_kernel(
    X,
    Y,
    rows,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < width
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for r in range(rows):
        ptr = X + r * width + cols
        val = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
        acc += val
    tl.store(Y + cols, acc, mask=mask)


def run(x):
    assert x.is_cuda
    assert x.dim() == 2
    assert x.is_contiguous()
    rows, width = x.shape
    y = torch.empty((width,), dtype=x.dtype, device=x.device)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(width, BLOCK_SIZE),)
    _sum_rows_kernel[grid](x, y, rows, width, BLOCK_SIZE=BLOCK_SIZE)
    return y
