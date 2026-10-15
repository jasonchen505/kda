import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    X,
    Y,
    W,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride
    Y += row * stride

    # Compute sum of squares in FP32
    _sum_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        a = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        _sum_sq += a * a
    sum_sq = tl.sum(_sum_sq, axis=0)
    mean_sq = sum_sq / N
    rstd = 1.0 / tl.sqrt(mean_sq + eps)

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
    out = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(width)
    if BLOCK_SIZE > 4096:
        BLOCK_SIZE = 4096

    grid = (rows,)
    _rmsnorm_kernel[grid](
        x,
        out,
        weight,
        width,
        width,
        1e-6,
        BLOCK_SIZE=BLOCK_SIZE,
        DTYPE=x.dtype,
    )
    return out
