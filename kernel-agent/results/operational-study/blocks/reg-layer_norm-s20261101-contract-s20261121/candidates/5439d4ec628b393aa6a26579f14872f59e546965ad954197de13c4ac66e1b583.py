import torch
import triton
import triton.language as tl


@triton.jit
def _ln_kernel(
    X, Y, W, B,
    stride_x, stride_y,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
    ROWS: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK_SIZE)
    rmask = rows < tl.num_programs(0) * ROWS
    cmask = cols < N
    mask = rmask[:, None] & cmask[None, :]
    x_ptr = X + rows[:, None] * stride_x + cols[None, :]
    x = tl.load(x_ptr, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=cmask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=cmask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) / N
    centered = tl.where(cmask[None, :], x - mean[:, None], 0.0)
    var = tl.sum(centered * centered, axis=1) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    y = centered * rstd[:, None] * w[None, :] + b[None, :]
    y_ptr = Y + rows[:, None] * stride_y + cols[None, :]
    tl.store(y_ptr, y.to(OUT_DTYPE), mask=mask)


def run(x, weight, bias):
    assert x.is_cuda and x.dim() == 2
    R, W = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(W)
    if R * W <= 8192:
        ROWS = 4
    else:
        ROWS = 1
    grid = (triton.cdiv(R, ROWS),)
    out_dtype = tl.bfloat16 if x.dtype == torch.bfloat16 else tl.float32
    _ln_kernel[grid](
        x, out, weight, bias,
        x.stride(0), out.stride(0),
        W, 1e-5,
        BLOCK_SIZE=BLOCK_SIZE,
        ROWS=ROWS,
        OUT_DTYPE=out_dtype,
    )
    return out
