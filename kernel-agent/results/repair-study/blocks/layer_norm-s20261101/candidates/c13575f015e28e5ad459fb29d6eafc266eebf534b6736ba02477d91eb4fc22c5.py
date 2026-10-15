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
):
    row = tl.program_id(0)
    x_ptr = X + row * stride_x
    y_ptr = Y + row * stride_y
    # Single pass: accumulate sum and sumsq
    _sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    _sumsq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        a = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        _sum += a
        _sumsq += a * a
    mean = tl.sum(_sum, axis=0) / N
    var = tl.sum(_sumsq, axis=0) / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    # Store pass
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(y_ptr + cols, y, mask=mask)


def run(x, weight, bias):
    rows, width = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(width)
    grid = (rows,)
    _layer_norm_kernel[grid](
        x, out, weight, bias,
        x.stride(0), out.stride(0),
        width, 1e-5,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
