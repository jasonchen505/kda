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
    s = 0.0
    ss = 0.0
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        s += tl.sum(x, axis=0)
        ss += tl.sum(x * x, axis=0)
    mean = s / N
    var = ss / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(y_ptr + cols, y, mask=mask)


def run(x, weight, bias):
    assert x.is_cuda and x.dim() == 2 and x.is_contiguous()
    assert weight.is_cuda and weight.dim() == 1 and weight.is_contiguous()
    assert bias.is_cuda and bias.dim() == 1 and bias.is_contiguous()
    R, W = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = min(triton.next_power_of_2(W), 1024)
    grid = (R,)
    _layer_norm_kernel[grid](
        x, out, weight, bias,
        x.stride(0), out.stride(0),
        W, 1e-5,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
