import torch
import triton
import triton.language as tl


@triton.jit
def _row_sum_kernel(
    X,
    Y,
    stride_x,
    N,
    BLOCK_SIZE: tl.constexpr,
    NUM_ITERS: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride_x
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in tl.static_range(NUM_ITERS):
        cols = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        a = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        acc += a
    total = tl.sum(acc, axis=0)
    Y += row
    tl.store(Y, total)


def run(x):
    assert x.is_cuda
    assert x.dim() == 2
    assert x.is_contiguous()
    rows, cols = x.shape
    out = torch.empty((rows,), dtype=x.dtype, device=x.device)
    if rows == 0 or cols == 0:
        return out
    BLOCK_SIZE = triton.next_power_of_2(cols)
    if BLOCK_SIZE > 1024:
        BLOCK_SIZE = 1024
    NUM_ITERS = triton.cdiv(cols, BLOCK_SIZE)
    grid = (rows,)
    _row_sum_kernel[grid](x, out, x.stride(0), cols, BLOCK_SIZE=BLOCK_SIZE, NUM_ITERS=NUM_ITERS)
    return out
