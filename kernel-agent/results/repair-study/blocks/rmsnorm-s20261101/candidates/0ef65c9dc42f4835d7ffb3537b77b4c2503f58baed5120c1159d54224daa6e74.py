import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    X,
    W,
    Y,
    stride_x,
    stride_y,
    width,
    eps,
    BLOCK: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr = X + row * stride_x
    y_ptr = Y + row * stride_y
    cols = tl.arange(0, BLOCK)
    mask = cols < width
    x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x2 = x * x
    sum_x2 = tl.sum(x2, axis=0)
    mean_x2 = sum_x2 / width
    rstd = tl.rsqrt(mean_x2 + eps)
    normalized = x * rstd
    if IS_BF16:
        normalized = normalized.to(tl.bfloat16)
    else:
        normalized = normalized.to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0)
    y = w * normalized
    tl.store(y_ptr + cols, y, mask=mask)


def run(x, weight):
    rows, width = x.shape
    out = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(width)
    is_bf16 = x.dtype == torch.bfloat16
    grid = (rows,)
    _rmsnorm_kernel[grid](
        x,
        weight,
        out,
        width,
        width,
        width,
        1e-6,
        BLOCK=BLOCK,
        IS_BF16=is_bf16,
    )
    return out
