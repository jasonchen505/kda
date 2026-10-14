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


def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2 and x.is_contiguous()
    rows, cols = x.shape
    out = torch.empty((rows,), device=x.device, dtype=x.dtype)
    BLOCK_SIZE = 1024
    grid = (rows,)
    _row_sum_kernel[grid](x, out, cols, BLOCK_SIZE=BLOCK_SIZE)
    return out

