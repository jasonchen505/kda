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
    SINGLE: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    if SINGLE:
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        a = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        acc += a
    else:
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            a = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
            acc += a
    mean = tl.sum(acc, axis=0) / N
    tl.store(Y + row, mean)


def run(x):
    rows, width = x.shape
    out = torch.empty(rows, dtype=x.dtype, device=x.device)
    BLOCK_SIZE = min(triton.next_power_of_2(width), 512)
    SINGLE = width <= BLOCK_SIZE
    grid = (rows,)
    _mean_kernel[grid](x, out, width, width, BLOCK_SIZE=BLOCK_SIZE, SINGLE=SINGLE)
    return out
