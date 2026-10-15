import torch
import triton
import triton.language as tl


@triton.jit
def _ln_small(
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
    mean = tl.sum(x, axis=0) / N
    centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / N
    rstd = tl.rsqrt(var + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = centered * rstd * w + b
    tl.store(y_ptr + cols, y, mask=mask)


@triton.jit
def _ln_large(
    X, Y, W, B,
    stride_x, stride_y,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    x_ptr = X + row * stride_x
    y_ptr = Y + row * stride_y
    cols = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Two-pass: first compute mean and var across all chunks via atomics is complex.
    # Instead use a single kernel that loops over chunks.
    # For simplicity and correctness, use loop-based approach.
    pass


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
    _sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        _sum += x
    mean = tl.sum(_sum, axis=0) / N
    # Pass 2: compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.where(mask, x - mean, 0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = tl.rsqrt(var + eps)
    # Pass 3: normalize and store
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(y_ptr + cols, y, mask=mask)


def run(x, weight, bias):
    rows, width = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(width)
    if BLOCK_SIZE <= 1024:
        grid = (rows,)
        _ln_small[grid](
            x, out, weight, bias,
            width, width,
            width, 1e-5,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        grid = (rows,)
        _ln_loop[grid](
            x, out, weight, bias,
            width, width,
            width, 1e-5,
            BLOCK_SIZE=1024,
        )
    return out
