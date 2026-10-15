import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(
    X,
    Y,
    W,
    B,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
    SINGLE: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride
    Y += row * stride
    if SINGLE:
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        a = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(a, axis=0) / N
        c = tl.where(mask, a - mean, 0.0)
        var = tl.sum(c * c, axis=0) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
        y = (a - mean) * rstd * w + b
        tl.store(Y + cols, y, mask=mask)
    else:
        mean = 0.0
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            a = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
            mean += tl.sum(a, axis=0)
        mean = mean / N
        var = 0.0
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            a = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
            c = tl.where(mask, a - mean, 0.0)
            var += tl.sum(c * c, axis=0)
        var = var / N
        rstd = 1.0 / tl.sqrt(var + eps)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            a = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
            b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
            y = (a - mean) * rstd * w + b
            tl.store(Y + cols, y, mask=mask)


def run(x, weight, bias):
    assert x.is_cuda and x.dim() == 2 and x.is_contiguous()
    assert weight.is_cuda and weight.dim() == 1 and weight.is_contiguous()
    assert bias.is_cuda and bias.dim() == 1 and bias.is_contiguous()
    rows, width = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(width)
    SINGLE = BLOCK_SIZE <= 1024
    grid = (rows,)
    _rms_norm_kernel[grid](
        x,
        out,
        weight,
        bias,
        width,
        width,
        1e-5,
        BLOCK_SIZE=BLOCK_SIZE,
        SINGLE=SINGLE,
    )
    return out
