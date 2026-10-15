import torch
import triton
import triton.language as tl
LOG2E = 1.4426950408889634


@triton.jit
def _softmax_kernel(
    out_ptr,
    in_ptr,
    in_row_stride,
    out_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    row = tl.program_id(0)
    in_row = in_ptr + row * in_row_stride
    out_row = out_ptr + row * out_row_stride
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    x = tl.load(in_row + offs, mask=mask, other=-float('inf')).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp2((x - m) * LOG2E)
    d = tl.sum(e, axis=0)
    y = e / d
    tl.store(out_row + offs, y, mask=mask)


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def run(x):
    assert x.is_cuda
    assert x.dim() == 2
    assert x.is_contiguous()
    rows, cols = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = _next_pow2(cols)
    if cols <= 512:
        num_warps = 4
    elif cols <= 1024:
        num_warps = 8
    else:
        num_warps = 16
    grid = (rows,)
    _softmax_kernel[grid](
        out,
        x,
        x.stride(0),
        out.stride(0),
        cols,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_STAGES=2,
        num_warps=num_warps,
    )
    return out
