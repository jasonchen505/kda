import torch
import triton
import triton.language as tl


@triton.jit
def _ln_kernel(
    X,
    Y,
    W,
    B,
    stride_x,
    stride_y,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
    EVEN: tl.constexpr,
    SINGLE: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr = X + row * stride_x
    y_ptr = Y + row * stride_y

    if SINGLE:
        cols = tl.arange(0, BLOCK_SIZE)
        if EVEN:
            a = tl.load(x_ptr + cols).to(tl.float32)
        else:
            a = tl.load(x_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
        s = tl.sum(a, axis=0)
        ss = tl.sum(a * a, axis=0)
    else:
        s = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        ss = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            if EVEN:
                a = tl.load(x_ptr + cols).to(tl.float32)
            else:
                a = tl.load(x_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
            s += a
            ss += a * a
        s = tl.sum(s, axis=0)
        ss = tl.sum(ss, axis=0)
    mean = s / N
    var = ss / N - mean * mean
    rstd = tl.rsqrt(var + eps)

    if SINGLE:
        if EVEN:
            w = tl.load(W + cols).to(tl.float32)
            b = tl.load(B + cols).to(tl.float32)
            y = (a - mean) * rstd * w + b
            tl.store(y_ptr + cols, y)
        else:
            w = tl.load(W + cols, mask=cols < N, other=0.0).to(tl.float32)
            b = tl.load(B + cols, mask=cols < N, other=0.0).to(tl.float32)
            y = (a - mean) * rstd * w + b
            tl.store(y_ptr + cols, y, mask=cols < N)
    else:
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            if EVEN:
                a = tl.load(x_ptr + cols).to(tl.float32)
                w = tl.load(W + cols).to(tl.float32)
                b = tl.load(B + cols).to(tl.float32)
                y = (a - mean) * rstd * w + b
                tl.store(y_ptr + cols, y)
            else:
                a = tl.load(x_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
                w = tl.load(W + cols, mask=cols < N, other=0.0).to(tl.float32)
                b = tl.load(B + cols, mask=cols < N, other=0.0).to(tl.float32)
                y = (a - mean) * rstd * w + b
                tl.store(y_ptr + cols, y, mask=cols < N)


def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def run(x, weight, bias):
    rows, width = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = _next_pow2(min(4096, width))
    EVEN = (width % BLOCK_SIZE) == 0
    SINGLE = width <= BLOCK_SIZE
    grid = (rows,)
    _ln_kernel[grid](
        x, out, weight, bias,
        width, width,
        width, 1e-5,
        BLOCK_SIZE=BLOCK_SIZE,
        EVEN=EVEN,
        SINGLE=SINGLE,
    )
    return out
