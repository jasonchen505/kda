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
    IS_BF16: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride
    Y += row * stride
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    a = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
    sum_sq = tl.sum(a * a, axis=0)
    mean_sq = sum_sq / N
    rstd = tl.rsqrt(mean_sq + eps)
    norm = a * rstd
    if IS_BF16:
        norm = norm.to(tl.bfloat16)
    else:
        norm = norm.to(tl.float32)
    w = tl.load(W + cols, mask=mask)
    y = w * norm
    tl.store(Y + cols, y, mask=mask)


def run(x, weight):
    rows, width = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(width)
    is_bf16 = x.dtype == torch.bfloat16
    grid = (rows,)
    _rmsnorm_kernel[grid](
        x,
        out,
        weight,
        width,
        width,
        1e-6,
        BLOCK_SIZE=BLOCK_SIZE,
        IS_BF16=is_bf16,
    )
    return out
