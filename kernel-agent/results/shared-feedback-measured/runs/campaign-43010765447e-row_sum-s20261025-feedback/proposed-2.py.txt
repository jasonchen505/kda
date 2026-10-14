import torch
import triton
import triton.language as tl


@triton.jit
def _row_sum_kernel(
    X,  # pointer to input (2D contiguous)
    Y,  # pointer to output (1D)
    cols,  # number of columns
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * cols
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, cols, BLOCK_SIZE):
        c = off + tl.arange(0, BLOCK_SIZE)
        mask = c < cols
        v = tl.load(X + c, mask=mask, other=0.0).to(tl.float32)
        acc += v
    s = tl.sum(acc, axis=0)
    tl.store(Y + row, s)


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2 and x.is_contiguous()
    rows, cols = x.shape
    out = torch.empty((rows,), device=x.device, dtype=x.dtype)
    if cols > 1024:
        BLOCK_SIZE = 1024
        num_warps = 8
    else:
        BLOCK_SIZE = _next_pow2(cols)
        num_warps = 4 if BLOCK_SIZE >= 256 else 2
    grid = (rows,)
    _row_sum_kernel[grid](x, out, cols, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    return out
