import torch
import triton
import triton.language as tl

@triton.jit
def _rmsnorm_kernel(
    X,  # pointer to input (rows, width)
    Y,  # pointer to output (rows, width)
    W,  # pointer to weight (width,)
    stride,  # row stride in elements
    N,  # width
    eps,  # epsilon
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride
    Y += row * stride

    # Compute sum of squares in FP32
    sum_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        a = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        sum_sq += a * a
    mean_sq = tl.sum(sum_sq, axis=0) / N
    rstd = tl.rsqrt(mean_sq + eps)

    # Normalize, cast to input dtype, multiply by weight, store
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = x * rstd
        x_hat = x_hat.to(DTYPE)
        w = tl.load(W + cols, mask=mask)
        y = w * x_hat
        tl.store(Y + cols, y, mask=mask)

def run(x, weight):
    rows, width = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(width)
    if BLOCK_SIZE > 1024:
        BLOCK_SIZE = 1024
    grid = (rows,)
    _rmsnorm_kernel[grid](
        x, y, weight,
        width, width, 1e-6,
        BLOCK_SIZE=BLOCK_SIZE,
        DTYPE=x.dtype,
    )
    return y

