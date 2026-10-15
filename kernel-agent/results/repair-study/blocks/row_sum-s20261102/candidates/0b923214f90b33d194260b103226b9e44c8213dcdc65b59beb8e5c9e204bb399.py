import torch
import triton
import triton.language as tl


@triton.jit
def _row_sum_kernel(
    X,
    Y,
    cols,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    base = X + row * cols
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, cols, BLOCK_SIZE):
        cols_off = off + tl.arange(0, BLOCK_SIZE)
        mask = cols_off < cols
        x = tl.load(base + cols_off, mask=mask, other=0.0).to(tl.float32)
        acc += x
    total = tl.sum(acc, axis=0)
    tl.store(Y + row, total)
    

def run(x):
    assert x.is_cuda
    assert x.dim() == 2
    assert x.is_contiguous()
    rows, cols = x.shape
    out = torch.empty((rows,), dtype=x.dtype, device=x.device)
    BLOCK_SIZE = 1024
    grid = (rows,)
    _row_sum_kernel[grid](x, out, cols, BLOCK_SIZE=BLOCK_SIZE)
    return out
