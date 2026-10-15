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
    OUT_BF16: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride
    Y += row * stride
    
    # Single-pass Welford mean/var
    mean = 0.0
    m2 = 0.0
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        a = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        n = tl.sum(tl.where(mask, 1.0, 0.0), axis=0)
        delta = a - mean
        mean += tl.sum(delta, axis=0) / n
        delta2 = a - mean
        m2 += tl.sum(delta * delta2, axis=0)
    var = m2 / N
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Fused normalize + affine + store
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask, other=0.).to(tl.float32)
        b = tl.load(B + cols, mask=mask, other=0.).to(tl.float32)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        y = (x - mean) * rstd * w + b
        if OUT_BF16:
            tl.store(Y + cols, y.to(tl.bfloat16), mask=mask)
        else:
            tl.store(Y + cols, y, mask=mask)


def run(x, weight, bias):
    rows, width = x.shape
    output = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(width)
    out_bf16 = x.dtype == torch.bfloat16
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
        OUT_BF16=out_bf16,
    )
    return output
