import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_kernel(
    X,
    Y,
    W,
    B,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride
    Y += row * stride
    # Pass 1: compute mean and variance
    sum_val = 0.0
    sum_sq = 0.0
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        a = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(a, axis=0)
        sum_sq += tl.sum(a * a, axis=0)
    mean = sum_val / N
    var = sum_sq / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    # Pass 2: normalize and store
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
    grid = (rows,)
    _layer_norm_kernel[grid](
        x,
        out,
        weight,
        bias,
        width,
        width,
        1e-5,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
