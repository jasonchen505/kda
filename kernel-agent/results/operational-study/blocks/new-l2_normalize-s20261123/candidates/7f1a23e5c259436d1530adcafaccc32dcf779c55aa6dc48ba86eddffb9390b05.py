import torch
import triton
import triton.language as tl

@triton.jit
def _l2_norm_kernel(
    X,  # pointer to input
    Y,  # pointer to output
    stride,  # row stride
    N,  # width
    eps,  # epsilon
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride
    Y += row * stride

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)

    sum_sq = tl.sum(x * x, axis=0)
    rstd = tl.rsqrt(sum_sq + eps)

    y = x * rstd
    tl.store(Y + cols, y, mask=mask)

def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    assert x.is_contiguous()
    assert x.dim() == 2

    rows, width = x.shape
    y = torch.empty_like(x)

    BLOCK_SIZE = 1 << (width - 1).bit_length()
    grid = (rows,)

    _l2_norm_kernel[grid](
        x, y, x.stride(0), width, 1e-6, BLOCK_SIZE
    )

    return y
