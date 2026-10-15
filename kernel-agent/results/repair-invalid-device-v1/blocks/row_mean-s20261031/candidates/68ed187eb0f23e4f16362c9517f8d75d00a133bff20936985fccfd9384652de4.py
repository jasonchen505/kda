import torch
import triton
import triton.language as tl

#
@triton.jit
def _sum_kernel(
    X,
    S,
    stride_x,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    X += row * stride_x + col_block * BLOCK_SIZE
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N - col_block * BLOCK_SIZE
    a = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    s = tl.sum(a, axis=0)
    tl.atomic_add(S + row, s)

#
@triton.jit
def _finalize_kernel(
    S,
    Y,
    N,
    OUT_DTYPE: tl.constexpr,
):
    row = tl.program_id(0)
    s = tl.load(S + row)
    mean = s / N
    tl.store(Y + row, mean.to(OUT_DTYPE))

#
def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    assert x.dim() == 2
    assert x.is_contiguous()
    rows, cols = x.shape
    out = torch.empty((rows,), dtype=x.dtype, device=x.device)
    if rows == 0 or cols == 0:
        return out
    scratch = torch.zeros((rows,), dtype=torch.float32, device=x.device)
    if cols <= 256:
        BLOCK_SIZE = 256
    else:
        BLOCK_SIZE = 1024
    col_blocks = triton.cdiv(cols, BLOCK_SIZE)
    grid = (rows, col_blocks)
    _sum_kernel[grid](x, scratch, cols, cols, BLOCK_SIZE=BLOCK_SIZE)
    _finalize_kernel[(rows,)](scratch, out, cols, OUT_DTYPE=x.dtype)
    return out
