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
    N_CHUNKS: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride_x
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in tl.static_range(N_CHUNKS):
        cols = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = cols < width
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        acc += x
    s = tl.sum(acc, axis=0)
    mean = s / width
    tl.store(Y + row, mean.to(Y.dtype.element_ty))


def run(x):
    assert x.is_cuda
    assert x.dim() == 2
    assert x.is_contiguous()
    rows, width = x.shape
    out = torch.empty((rows,), dtype=x.dtype, device=x.device)
    BLOCK = 1024
    n_chunks = triton.cdiv(width, BLOCK)
    grid = (rows,)
    _mean_kernel[grid](x, out, x.stride(0), width, BLOCK_SIZE=BLOCK, N_CHUNKS=n_chunks)
    return out
