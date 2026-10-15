import torch
import triton
import triton.language as tl


@triton.jit
def _row_mean_kernel(
    X,
    Y,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    base = row * width
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, width, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < width
        x = tl.load(X + base + cols, mask=mask, other=0.).to(tl.float32)
        acc += x
    s = tl.sum(acc, axis=0)
    mean = s / width
    tl.store(Y + row, mean)


def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2 and x.is_contiguous()
    assert x.dtype in (torch.float32, torch.bfloat16)
    rows, width = x.shape
    out = torch.empty((rows,), dtype=x.dtype, device=x.device)
    if rows == 0 or width == 0:
        return out
    BLOCK_SIZE = 1024
    grid = (rows,)
    _row_mean_kernel[grid](x, out, width, BLOCK_SIZE=BLOCK_SIZE)
    return out
