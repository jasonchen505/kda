import torch
import triton
import triton.language as tl

@triton.jit
def _row_sum_kernel(
    X,  # pointer to input (rows, cols)
    Y,  # pointer to output (rows,)
    stride,  # row stride in elements
    N,  # number of columns
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        a = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        acc += a
    total = tl.sum(acc, axis=0)
    tl.store(Y + row, total)


def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    assert x.dim() == 2
    assert x.is_contiguous()
    rows, cols = x.shape
    y = torch.empty((rows,), dtype=x.dtype, device=x.device)
    BLOCK_SIZE = 1024
    grid = (rows,)
    _row_sum_kernel[grid](x, y, x.stride(0), cols, BLOCK_SIZE=BLOCK_SIZE)
    return y

