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
        c = off + tl.arange(0, BLOCK_SIZE)
        mask = c < cols
        v = tl.load(base + c, mask=mask, other=0.).to(tl.float32)
        acc += v
    s = tl.sum(acc, axis=0)
    tl.store(Y + row, s)


def run(x):
    assert x.is_cuda
    assert x.dim() == 2
    assert x.is_contiguous()
    rows, cols = x.shape
    y = torch.empty((rows,), dtype=x.dtype, device=x.device)
    BLOCK_SIZE = 1024
    grid = (rows,)
    _row_sum_kernel[grid](x, y, cols, BLOCK_SIZE=BLOCK_SIZE)
    return y

