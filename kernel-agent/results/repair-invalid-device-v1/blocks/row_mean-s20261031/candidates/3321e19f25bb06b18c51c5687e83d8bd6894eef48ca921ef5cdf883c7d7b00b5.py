import torch
import triton
import triton.language as tl


@triton.jit
def _row_mean_kernel(
    X,
    Y,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    base = X + row * N
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        a = tl.load(base + cols, mask=mask, other=0.0).to(tl.float32)
        acc += a
    s = tl.sum(acc, axis=0)
    mean = s / N
    tl.store(Y + row, mean)


def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    assert x.dim() == 2
    assert x.is_contiguous()
    rows, cols = x.shape
    out = torch.empty((rows,), dtype=x.dtype, device=x.device)
    if rows == 0 or cols == 0:
        return out
    if cols <= 256:
        BLOCK_SIZE = 256
    elif cols <= 1024:
        BLOCK_SIZE = 1024
    else:
        BLOCK_SIZE = 2048
    _row_mean_kernel[(rows,)](x, out, cols, BLOCK_SIZE=BLOCK_SIZE)
    return out
