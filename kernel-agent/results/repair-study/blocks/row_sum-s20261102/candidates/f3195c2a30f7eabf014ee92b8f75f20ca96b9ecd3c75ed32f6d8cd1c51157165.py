import torch
import triton
import triton.language as tl

@triton.jit
def _row_sum_kernel(
    X,  # pointer to input
    Y,  # pointer to output
    stride_x,  # row stride of input
    N,  # number of columns
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride_x
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        acc += x
    result = tl.sum(acc, axis=0)
    Y += row
    tl.store(Y, result)

def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input must be a CUDA tensor"
    assert x.dim() == 2, "Input must be 2D"
    assert x.is_contiguous(), "Input must be contiguous"
    rows, cols = x.shape
    out = torch.empty(rows, dtype=x.dtype, device=x.device)
    if rows == 0:
        return out
    BLOCK_SIZE = triton.next_power_of_2(cols)
    grid = (rows,)
    _row_sum_kernel[grid](x, out, x.stride(0), cols, BLOCK_SIZE=BLOCK_SIZE)
    return out
