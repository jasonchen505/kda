import torch
import triton
import triton.language as tl


@triton.jit
def _row_mean_kernel(
    X,
    Y,
    stride,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    a = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
    total = tl.sum(a, axis=0)
    mean = total / N
    tl.store(Y + row, mean)


def run(x):
    assert x.is_cuda
    assert x.dim() == 2
    assert x.is_contiguous()
    rows, width = x.shape
    out = torch.empty((rows,), dtype=x.dtype, device=x.device)
    if rows == 0 or width == 0:
        return out
    BLOCK_SIZE = triton.next_power_of_2(width)
    grid = (rows,)
    _row_mean_kernel[grid](x, out, width, width, BLOCK_SIZE=BLOCK_SIZE)
    return out
