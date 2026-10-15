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
    x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)

    sum_sq = tl.sum(x * x, axis=0)
    mean_sq = sum_sq / N
    rstd = 1.0 / tl.sqrt(mean_sq + eps)

    x_hat = x * rstd
    if IS_BF16:
        x_hat = x_hat.to(tl.bfloat16)
    else:
        x_hat = x_hat.to(tl.float32)

    w = tl.load(W + cols, mask=mask)
    y = w * x_hat
    tl.store(Y + cols, y, mask=mask)


def run(x, weight):
    rows, width = x.shape
    out = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(width)
    if BLOCK_SIZE > 4096:
        BLOCK_SIZE = 4096

    is_bf16 = 1 if x.dtype == torch.bfloat16 else 0

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
