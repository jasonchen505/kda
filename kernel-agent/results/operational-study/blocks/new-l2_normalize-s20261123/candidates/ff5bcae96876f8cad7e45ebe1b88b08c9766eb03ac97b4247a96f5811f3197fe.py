import torch
import triton
import triton.language as tl

@triton.jit
def _l2_norm_kernel(
    X,  # pointer to input
    Y,  # pointer to output
    stride,  # row stride
    N,  # width
    eps,  # epsilon
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride
    Y += row * stride

    # Compute sum of squares
    sum_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        sum_sq += x * x

    total_sum = tl.sum(sum_sq, axis=0)
    rstd = tl.rsqrt(total_sum + eps)

    # Scale and store
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        y = x * rstd
        tl.store(Y + cols, y, mask=mask)

def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    assert x.is_contiguous()
    assert x.dim() == 2

    rows, width = x.shape
    y = torch.empty_like(x)

    BLOCK_SIZE = 1024
    grid = (rows,)

    _l2_norm_kernel[grid](
        x, y, x.stride(0), width, 1e-6, BLOCK_SIZE
    )

    return y
