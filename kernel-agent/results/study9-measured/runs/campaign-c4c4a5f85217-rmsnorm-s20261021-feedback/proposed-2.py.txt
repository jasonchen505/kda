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
        normalized = (x * rstd).to(x.dtype)
        y = w * normalized
        tl.store(Y + cols, y, mask=mask)
    else:
        sum_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
            sum_sq += x * x
        mean_sq = tl.sum(sum_sq, axis=0) / N
        rstd = tl.rsqrt(mean_sq + eps)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
            w = tl.load(W + cols, mask=mask)
            normalized = (x * rstd).to(x.dtype)
            y = w * normalized
            tl.store(Y + cols, y, mask=mask)


def run(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda
    assert x.is_contiguous() and weight.is_contiguous()
    assert x.dtype == weight.dtype
    rows, width = x.shape
    y = torch.empty_like(x)
    if rows == 0 or width == 0:
        return y
    if width <= 1024:
        BLOCK_SIZE = triton.next_power_of_2(width)
        SINGLE_BLOCK = True
    else:
        BLOCK_SIZE = 1024
        SINGLE_BLOCK = False
    _rmsnorm_kernel[(rows,)](
        x, y, weight,
        width,  # stride for contiguous 2D tensor
        width,
        1e-6,
        BLOCK_SIZE=BLOCK_SIZE,
        SINGLE_BLOCK=SINGLE_BLOCK,
    )
    return y
