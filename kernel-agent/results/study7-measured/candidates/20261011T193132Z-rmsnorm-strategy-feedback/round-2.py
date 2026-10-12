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
):
    row = tl.program_id(0)
    X += row * stride
    Y += row * stride

    # Load entire row into registers (padded to BLOCK_SIZE)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)

    # Compute sum of squares
    sum_sq = tl.sum(x * x, axis=0)
    mean_sq = sum_sq / N
    rstd = tl.rsqrt(mean_sq + eps)

    # Normalize, cast to input dtype, multiply by weight, store
    w = tl.load(W + cols, mask=mask)
    normalized = (x * rstd).to(X.dtype.element_ty)
    y = w * normalized
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
    )
    return y
