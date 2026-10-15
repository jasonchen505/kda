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

    _sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    _sum_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        a = tl.load(x_ptr + cols, mask=mask, other=0.).to(tl.float32)
        _sum += a
        _sum_sq += a * a
    mean = tl.sum(_sum, axis=0) / N
    var = tl.sum(_sum_sq, axis=0) / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.).to(tl.float32)
        b = tl.load(B + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(y_ptr + cols, y, mask=mask)


def run(x, weight, bias):
    assert x.is_cuda and x.dim() == 2 and x.is_contiguous()
    assert weight.is_cuda and weight.dim() == 1 and weight.is_contiguous()
    assert bias.is_cuda and bias.dim() == 1 and bias.is_contiguous()
    rows, width = x.shape
    assert weight.shape[0] == width and bias.shape[0] == width
    out = torch.empty_like(x)
    if rows == 0 or width == 0:
        return out
    eps = 1e-5
    BLOCK_SIZE = 2048
    grid = (rows,)
    _layer_norm_kernel[grid](
        x, out, weight, bias,
        x.stride(0), out.stride(0),
        width, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
