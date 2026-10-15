import torch
import triton
import triton.language as tl


@triton.jit
def _ln_kernel(
    X,
    Y,
    W,
    B,
    stride,
    N,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride
    Y += row * stride
    s = tl.zeros([BLOCK], dtype=tl.float32)
    ss = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        a = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        s += a
        ss += a * a
    mean = tl.sum(s, axis=0) / N
    var = tl.sum(ss, axis=0) / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
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
    BLOCK = 1024
    grid = (rows,)
    _ln_kernel[grid](
        x,
        out,
        weight,
        bias,
        width,
        width,
        1e-5,
        BLOCK=BLOCK,
    )
    return out
