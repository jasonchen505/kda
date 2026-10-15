import torch
import triton
import triton.language as tl


@triton.jit
def _mean_atomic_kernel(
    X,
    S,
    stride,
    N,
    BLOCK_SIZE: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    X += row * stride + chunk * BLOCK_SIZE
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < (N - chunk * BLOCK_SIZE)
    a = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
    s = tl.sum(a, axis=0)
    tl.atomic_add(S + row, s)


@triton.jit
def _finalize_kernel(
    S,
    Y,
    N,
    IS_BF16: tl.constexpr,
):
    row = tl.program_id(0)
    s = tl.load(S + row)
    mean = s / N
    if IS_BF16:
        mean = mean.to(tl.bfloat16)
    tl.store(Y + row, mean)


def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    assert x.dim() == 2
    assert x.is_contiguous()
    rows, width = x.shape
    y = torch.empty((rows,), dtype=x.dtype, device=x.device)
    s = torch.zeros((rows,), dtype=torch.float32, device=x.device)
    BLOCK_SIZE = 1024
    n_chunks = triton.cdiv(width, BLOCK_SIZE)
    grid = (rows, n_chunks)
    _mean_atomic_kernel[grid](x, s, width, width, BLOCK_SIZE=BLOCK_SIZE, IS_BF16=(x.dtype == torch.bfloat16))
    _finalize_kernel[(rows,)](s, y, width, IS_BF16=(x.dtype == torch.bfloat16))
    return y
