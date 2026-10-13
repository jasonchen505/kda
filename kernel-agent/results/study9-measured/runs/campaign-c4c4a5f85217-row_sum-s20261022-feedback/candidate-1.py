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
    X += row * cols
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, cols, BLOCK_SIZE):
        col_offs = off + tl.arange(0, BLOCK_SIZE)
        mask = col_offs < cols
        vals = tl.load(X + col_offs, mask=mask, other=0.0).to(tl.float32)
        acc += vals
    total = tl.sum(acc, axis=0)
    tl.store(Y + row, total)


def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "input must be a CUDA tensor"
    assert x.dim() == 2, "input must be 2D"
    assert x.is_contiguous(), "input must be contiguous"
    rows, cols = x.shape
    out = torch.empty(rows, dtype=x.dtype, device=x.device)
    if rows == 0:
        return out
    BLOCK_SIZE = 1024
    grid = (rows,)
    _row_sum_kernel[grid](x, out, cols, BLOCK_SIZE=BLOCK_SIZE)
    return out
