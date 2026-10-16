import torch
import triton
import triton.language as tl

@triton.jit
def _l2_norm_kernel(
    X,
    Y,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
    EVEN: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride
    Y += row * stride
    cols = tl.arange(0, BLOCK_SIZE)
    if EVEN:
        x = tl.load(X + cols).to(tl.float32)
    else:
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
    sum_sq = tl.sum(x * x, axis=0)
    rstd = 1.0 / tl.sqrt(sum_sq + eps)
    y = x * rstd
    if EVEN:
        tl.store(Y + cols, y)
    else:
        tl.store(Y + cols, y, mask=cols < N)
    
def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    assert x.is_contiguous()
    assert x.dim() == 2
    rows, width = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = max(1, triton.next_power_of_2(min(width, 2048)))
    EVEN = (width == BLOCK_SIZE)
    grid = (rows,)
    _l2_norm_kernel[grid](
        x, y, x.stride(0), width, 1e-6, BLOCK_SIZE, EVEN
    )
    return y
