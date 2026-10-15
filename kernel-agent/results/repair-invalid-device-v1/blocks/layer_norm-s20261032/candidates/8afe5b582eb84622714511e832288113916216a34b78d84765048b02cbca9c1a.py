import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_kernel(
    X,
    Y,
    W,
    B,
    stride_x,
    stride_y,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr = X + row * stride_x
    y_ptr = Y + row * stride_y
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / N
    rstd = tl.rsqrt(var + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = centered * rstd * w + b
    tl.store(y_ptr + cols, y.to(OUT_DTYPE), mask=mask)


def run(x, weight, bias):
    rows, width = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(width)
    grid = (rows,)
    _layer_norm_kernel[grid](
        x, out, weight, bias,
        width, width,
        width, 1e-5,
        BLOCK_SIZE=BLOCK_SIZE,
        OUT_DTYPE=x.dtype,
    )
    return out
