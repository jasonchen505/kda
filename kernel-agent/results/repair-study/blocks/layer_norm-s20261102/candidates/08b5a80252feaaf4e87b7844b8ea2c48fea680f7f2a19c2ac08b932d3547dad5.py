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
    EVEN: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride
    Y += row * stride
    
    # Single pass: accumulate sum and sum of squares
    _sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    _sum_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        if EVEN:
            a = tl.load(X + cols).to(tl.float32)
        else:
            mask = cols < N
            a = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        _sum += a
        _sum_sq += a * a
    sum_val = tl.sum(_sum, axis=0)
    sum_sq_val = tl.sum(_sum_sq, axis=0)
    mean = sum_val / N
    var = sum_sq_val / N - mean * mean
    rstd = 1 / tl.sqrt(var + eps)
    
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        if EVEN:
            w = tl.load(W + cols).to(tl.float32)
            b = tl.load(B + cols).to(tl.float32)
            x = tl.load(X + cols).to(tl.float32)
            x_hat = (x - mean) * rstd
            y = x_hat * w + b
            tl.store(Y + cols, y)
        else:
            mask = cols < N
            w = tl.load(W + cols, mask=mask, other=0.).to(tl.float32)
            b = tl.load(B + cols, mask=mask, other=0.).to(tl.float32)
            x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
            x_hat = (x - mean) * rstd
            y = x_hat * w + b
            tl.store(Y + cols, y, mask=mask)
    

def run(x, weight, bias):
    rows, width = x.shape
    output = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(width)
    EVEN = (width == BLOCK_SIZE)
    grid = (rows,)
    _layer_norm_kernel[grid](
        x,
        output,
        weight,
        bias,
        width,
        width,
        1e-5,
        BLOCK_SIZE=BLOCK_SIZE,
        EVEN=EVEN,
    )
    return output
