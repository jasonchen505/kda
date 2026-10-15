import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    X,
    W,
    Y,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride
    Y += row * stride
    # Accumulate sum of squares in fp32
    ss = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        a = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        ss += a * a
    mean_sq = tl.sum(ss, axis=0) / N
    rstd = tl.rsqrt(mean_sq + eps)
    # Normalize, cast to input dtype, multiply by weight
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        a = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        normed = (a * rstd).to(tl.bfloat16 if IS_BF16 else tl.float32)
        w = tl.load(W + cols, mask=mask)
        y = w * normed
        tl.store(Y + cols, y, mask=mask)


def run(x, weight):
    rows, width = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(width)
    IS_BF16 = x.dtype == torch.bfloat16
    grid = (rows,)
    _rmsnorm_kernel[grid](
        x, weight, y,
        width, width, 1e-6,
        BLOCK_SIZE=BLOCK_SIZE,
        IS_BF16=IS_BF16,
    )
    return y
