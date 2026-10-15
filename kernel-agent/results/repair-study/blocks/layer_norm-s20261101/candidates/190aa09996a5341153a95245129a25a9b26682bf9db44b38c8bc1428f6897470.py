import torch
import triton
import triton.language as tl


@triton.jit
def _ln_single(
    X, Y, W, B,
    stride_x, stride_y,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr = X + row * stride_x
    y_ptr = Y + row * stride_y
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    y = centered * rstd * w + b
    tl.store(y_ptr + cols, y, mask=mask)


@triton.jit
def _ln_loop(
    X, Y, W, B,
    stride_x, stride_y,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr = X + row * stride_x
    y_ptr = Y + row * stride_y
    # Pass 1: compute mean
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        a = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # Pass 2: compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        a = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        c = tl.where(mask, a - mean, 0.0)
        _var += c * c
    var = tl.sum(_var, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    # Pass 3: normalize and store
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        a = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
        c = tl.where(mask, a - mean, 0.0)
        y = c * rstd * w + b
        tl.store(y_ptr + cols, y, mask=mask)


def run(x, weight, bias):
    assert x.is_cuda and x.dim() == 2
    R, W = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(W)
    grid = (R,)
    if BLOCK_SIZE <= 1024:
        _ln_single[grid](
            x, out, weight, bias,
            x.stride(0), out.stride(0),
            W, 1e-5,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        _ln_loop[grid](
            x, out, weight, bias,
            x.stride(0), out.stride(0),
            W, 1e-5,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    return out
