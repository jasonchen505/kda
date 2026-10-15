import torch
import triton
import triton.language as tl


@triton.jit
def _row_mean_single(
    X,
    Y,
    stride,
    N,
    BLOCK_SIZE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    a = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    total = tl.sum(a, axis=0)
    mean = total / N
    tl.store(Y + row, mean.to(OUT_DTYPE))


@triton.jit
def _row_mean_loop(
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
        a = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        acc += a
    total = tl.sum(acc, axis=0)
    mean = total / N
    tl.store(Y + row, mean.to(OUT_DTYPE))


def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    assert x.dim() == 2
    assert x.is_contiguous()
    rows, width = x.shape
    out = torch.empty((rows,), dtype=x.dtype, device=x.device)
    if rows == 0 or width == 0:
        return out
    BLOCK_SIZE = triton.next_power_of_2(width)
    grid = (rows,)
    out_dtype = tl.bfloat16 if x.dtype == torch.bfloat16 else tl.float32
    if BLOCK_SIZE <= 1024:
        _row_mean_single[grid](x, out, width, width, BLOCK_SIZE=BLOCK_SIZE, OUT_DTYPE=out_dtype)
    else:
        _row_mean_loop[grid](x, out, width, width, BLOCK_SIZE=BLOCK_SIZE, OUT_DTYPE=out_dtype)
    return out
