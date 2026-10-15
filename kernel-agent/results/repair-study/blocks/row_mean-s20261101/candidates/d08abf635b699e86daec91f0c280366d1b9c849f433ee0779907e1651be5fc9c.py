import torch
import triton
import triton.language as tl


@triton.jit
def _mean_kernel(
    X,
    Y,
    stride,
    N,
    BLOCK_SIZE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        a = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        acc += a
    total = tl.sum(acc, axis=0)
    mean = total / N
    tl.store(Y + row, mean.to(OUT_DTYPE))


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    assert x.dim() == 2
    assert x.is_contiguous()
    rows, width = x.shape
    y = torch.empty((rows,), dtype=x.dtype, device=x.device)
    BLOCK_SIZE = _next_pow2(min(width, 2048))
    grid = (rows,)
    out_dtype = tl.float32 if x.dtype == torch.float32 else tl.bfloat16
    _mean_kernel[grid](x, y, width, width, BLOCK_SIZE=BLOCK_SIZE, OUT_DTYPE=out_dtype)
    return y
