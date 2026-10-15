import torch
import triton
import triton.language as tl


@triton.jit
def _mean_single(
    X,
    Y,
    stride_x,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride_x
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < width
    x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
    s = tl.sum(x, axis=0)
    mean = s / width
    y = mean.to(Y.dtype.element_ty)
    tl.store(Y + row, y)


@triton.jit
def _mean_multi(
    X,
    Y,
    stride_x,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    X += row * stride_x + chunk * BLOCK_SIZE
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < width - chunk * BLOCK_SIZE
    x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
    s = tl.sum(x, axis=0)
    tl.atomic_add(Y + row, s)


def run(x):
    assert x.is_cuda
    assert x.dim() == 2
    assert x.is_contiguous()
    rows, width = x.shape
    out = torch.zeros((rows,), dtype=torch.float32, device=x.device)
    BLOCK = 1024
    n_chunks = triton.cdiv(width, BLOCK)
    if n_chunks == 1:
        BLOCK_SIZE = triton.next_power_of_2(width)
        grid = (rows,)
        _mean_single[grid](x, out, x.stride(0), width, BLOCK_SIZE=BLOCK_SIZE)
        out = out.to(x.dtype)
    else:
        grid = (rows, n_chunks)
        _mean_multi[grid](x, out, x.stride(0), width, BLOCK_SIZE=BLOCK)
        out = (out / width).to(x.dtype)
    return out
