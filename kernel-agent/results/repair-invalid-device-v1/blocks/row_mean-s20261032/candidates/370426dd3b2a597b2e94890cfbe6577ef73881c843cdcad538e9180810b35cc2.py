import torch
import triton
import triton.language as tl


@triton.jit
def _mean_kernel(
    X,
    Y,
    stride_x,
    N,
    BLOCK_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride_x
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        a = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        acc += a
    s = tl.sum(acc, axis=0)
    y = s / N
    tl.store(Y + row, y)


def run(x):
    assert x.is_cuda
    assert x.dim() == 2
    assert x.is_contiguous()
    rows, width = x.shape
    out = torch.empty(rows, dtype=x.dtype, device=x.device)
    if width <= 128:
        BLOCK_SIZE = 128
    elif width <= 256:
        BLOCK_SIZE = 256
    elif width <= 512:
        BLOCK_SIZE = 512
    elif width <= 1024:
        BLOCK_SIZE = 1024
    elif width <= 2048:
        BLOCK_SIZE = 2048
    else:
        BLOCK_SIZE = 4096
    grid = (rows,)
    _mean_kernel[grid](x, out, width, width, BLOCK_SIZE=BLOCK_SIZE, NUM_STAGES=2)
    return out
