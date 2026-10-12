import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel_small(
    X,  # pointer to input (rows, width)
    Y,  # pointer to output (rows, width)
    W,  # pointer to weight (width,)
    stride,  # row stride in elements
    N,  # width
    eps,  # epsilon
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride
    Y += row * stride

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)

    # Compute sum of squares in FP32
    sum_sq = tl.sum(x * x, axis=0)
    mean_sq = sum_sq / N
    rstd = tl.rsqrt(mean_sq + eps)

    # Normalize, cast to input dtype, multiply by weight, store
    w = tl.load(W + cols, mask=mask)
    normalized = (x * rstd).to(X.dtype.element_ty)
    y = w * normalized
    tl.store(Y + cols, y, mask=mask)


@triton.jit
def _rmsnorm_kernel_large(
    X,  # pointer to input (rows, width)
    Y,  # pointer to output (rows, width)
    W,  # pointer to weight (width,)
    stride,  # row stride in elements
    N,  # width
    eps,  # epsilon
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride
    Y += row * stride

    # First pass: compute sum of squares
    sum_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        sum_sq += x * x
    mean_sq = tl.sum(sum_sq, axis=0) / N
    rstd = tl.rsqrt(mean_sq + eps)

    # Second pass: normalize and store
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        w = tl.load(W + cols, mask=mask)
        normalized = (x * rstd).to(X.dtype.element_ty)
        y = w * normalized
        tl.store(Y + cols, y, mask=mask)


def run(x, weight):
    rows, width = x.shape
    y = torch.empty_like(x)
    stride = width
    eps = 1e-6

    if width <= 1024:
        BLOCK_SIZE = triton.next_power_of_2(width)
        _rmsnorm_kernel_small[(rows,)](
            x, y, weight,
            stride, width, eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        BLOCK_SIZE = 1024
        _rmsnorm_kernel_large[(rows,)](
            x, y, weight,
            stride, width, eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    return y
