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
    SINGLE_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride
    Y += row * stride

    if SINGLE_BLOCK:
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        sum_sq = tl.sum(x * x, axis=0)
        mean_sq = sum_sq / N
        rstd = tl.rsqrt(mean_sq + eps)
        w = tl.load(W + cols, mask=mask)
        x_norm = x * rstd
        x_norm_cast = x_norm.to(X.dtype.element_ty)
        y = x_norm_cast * w
        tl.store(Y + cols, y, mask=mask)
    else:
        _sum_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
            _sum_sq += x * x
        sum_sq = tl.sum(_sum_sq, axis=0)
        mean_sq = sum_sq / N
        rstd = tl.rsqrt(mean_sq + eps)

        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
            w = tl.load(W + cols, mask=mask)
            x_norm = x * rstd
            x_norm_cast = x_norm.to(X.dtype.element_ty)
            y = x_norm_cast * w
            tl.store(Y + cols, y, mask=mask)

def run(x, weight):
    rows, width = x.shape
    y = torch.empty_like(x)
    if width <= 2048:
        BLOCK_SIZE = triton.next_power_of_2(width)
        SINGLE_BLOCK = True
    else:
        BLOCK_SIZE = 4096
        SINGLE_BLOCK = False
    grid = (rows,)
    _rmsnorm_kernel[grid](
        x, y, weight,
        width,  # stride for contiguous 2D tensor
        width,
        1e-6,
        BLOCK_SIZE=BLOCK_SIZE,
        SINGLE_BLOCK=SINGLE_BLOCK,
    )
    return y
