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
):
    row = tl.program_id(0)
    X += row * stride_x
    Y += row * stride_x

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # Load x once, compute sum of squares in FP32
    x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
    sum_sq = tl.sum(x * x, axis=0)
    mean_sq = sum_sq / N
    rstd = tl.rsqrt(mean_sq + eps)

    # Load weight
    w = tl.load(W + cols, mask=mask).to(tl.float32)

    # Normalize, cast to input dtype, multiply by weight, store
    normalized = (x * rstd).to(Y.dtype.element_ty)
    y = w * normalized.to(tl.float32)
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
    )
    return y
