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
    TWO_DIM: tl.constexpr,
):
    if TWO_DIM:
        row = tl.program_id(0)
        col_block = tl.program_id(1)
    else:
        row = tl.program_id(0)
        col_block = 0
    x_ptr = X + row * stride_x
    y_ptr = Y + row * stride_y
    sum_x = 0.0
    sum_sq = 0.0
    for off in range(col_block * BLOCK_SIZE, N, BLOCK_SIZE * (1 if not TWO_DIM else 1)):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        sum_x += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
    mean = sum_x / N
    var = sum_sq / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    for off in range(col_block * BLOCK_SIZE, N, BLOCK_SIZE * (1 if not TWO_DIM else 1)):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(y_ptr + cols, y, mask=mask)


def run(x, weight, bias):
    assert x.is_cuda and x.dim() == 2
    R, W = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = min(triton.next_power_of_2(W), 1024)
    if W <= 1024:
        grid = (R, triton.cdiv(W, BLOCK_SIZE))
        two_dim = True
    else:
        grid = (R,)
        two_dim = False
    _layer_norm_kernel[grid](
        x, out, weight, bias,
        x.stride(0), out.stride(0),
        W, 1e-5,
        BLOCK_SIZE=BLOCK_SIZE,
        TWO_DIM=two_dim,
    )
    return out
