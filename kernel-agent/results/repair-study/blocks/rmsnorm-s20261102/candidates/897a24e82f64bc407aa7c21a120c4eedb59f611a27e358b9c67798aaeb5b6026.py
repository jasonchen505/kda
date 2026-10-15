import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    X,
    W,
    Y,
    stride_x,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr = X + row * stride_x
    y_ptr = Y + row * stride_x
    ss = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        ss += x * x
    mean_sq = tl.sum(ss, axis=0) / N
    rstd = tl.rsqrt(mean_sq + eps)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0)
        normalized = x * rstd
        if IS_BF16:
            normalized = normalized.to(tl.bfloat16)
        else:
            normalized = normalized.to(tl.float32)
        y = normalized * w
        tl.store(y_ptr + cols, y, mask=mask)


def run(x, weight):
    assert x.is_cuda
    assert x.is_contiguous()
    assert weight.is_cuda
    assert weight.is_contiguous()
    rows, width = x.shape
    assert weight.shape == (width,)
    assert x.dtype == weight.dtype
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(width)
    IS_BF16 = x.dtype == torch.bfloat16
    _rmsnorm_kernel[(rows,)](
        x,
        weight,
        y,
        width,
        width,
        1e-6,
        BLOCK_SIZE=BLOCK_SIZE,
        IS_BF16=IS_BF16,
    )
    return y
