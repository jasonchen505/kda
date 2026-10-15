import torch
import triton
import triton.language as tl


@triton.jit
def _row_sum_kernel(
    X,
    Y,
    stride,
    N,
    BLOCK_SIZE: tl.constexpr,
    SINGLE: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride
    if SINGLE:
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        a = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        s = tl.sum(a, axis=0)
    else:
        acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            a = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
            acc += a
        s = tl.sum(acc, axis=0)
    tl.store(Y + row, s)


def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    assert x.dim() == 2
    assert x.is_contiguous()
    rows, cols = x.shape
    out = torch.empty((rows,), dtype=x.dtype, device=x.device)
    BLOCK_SIZE = 1
    while BLOCK_SIZE < cols:
        BLOCK_SIZE *= 2
    SINGLE = cols <= BLOCK_SIZE
    grid = (rows,)
    _row_sum_kernel[grid](x, out, x.stride(0), cols, BLOCK_SIZE=BLOCK_SIZE, SINGLE=SINGLE)
    return out
