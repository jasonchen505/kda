import torch
import triton
import triton.language as tl


@triton.jit
def _mean_kernel(
    X,
    Y,
    stride_x,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride_x
    Y += row
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < width
    x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
    s = tl.sum(x, axis=0)
    mean = s / width
    y = mean.to(Y.dtype.element_ty)
    tl.store(Y, y)


def run(x):
    assert x.is_cuda
    assert x.dim() == 2
    assert x.is_contiguous()
    rows, width = x.shape
    out = torch.empty((rows,), dtype=x.dtype, device=x.device)
    BLOCK_SIZE = triton.next_power_of_2(width)
    grid = (rows,)
    _mean_kernel[grid](x, out, x.stride(0), width, BLOCK_SIZE=BLOCK_SIZE)
    return out
