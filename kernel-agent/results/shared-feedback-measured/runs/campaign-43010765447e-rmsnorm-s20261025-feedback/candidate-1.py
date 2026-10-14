import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    X,
    W,
    Y,
    stride_x,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride_x
    Y += row * stride_x

    # Compute sum of squares in FP32
    _sum_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        _sum_sq += x * x
    mean_sq = tl.sum(_sum_sq, axis=0) / N
    rstd = tl.rsqrt(mean_sq + eps)

    # Normalize, cast to input dtype, multiply by weight, store
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        normalized = (x * rstd).to(DTYPE)
        y = (w * normalized.to(tl.float32)).to(DTYPE)
        tl.store(Y + cols, y, mask=mask)


def run(x, weight):
    rows, width = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(width)
    grid = (rows,)
    _rmsnorm_kernel[grid](
        x, weight, y,
        width,
        width,
        1e-6,
        BLOCK_SIZE=BLOCK_SIZE,
        DTYPE=x.dtype,
    )
    return y
