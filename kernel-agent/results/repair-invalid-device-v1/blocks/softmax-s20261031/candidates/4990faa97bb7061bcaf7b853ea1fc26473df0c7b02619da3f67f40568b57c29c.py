import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    out_ptr,
    in_ptr,
    in_row_stride,
    out_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    in_row = in_ptr + row * in_row_stride
    out_row = out_ptr + row * out_row_stride
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(in_row + cols, mask=mask, other=-float('inf'))
    x = x.to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    d = tl.sum(e, axis=0)
    y = e / d
    tl.store(out_row + cols, y, mask=mask)
    return

def run(x):
    assert x.is_cuda
    assert x.dim() == 2
    assert x.is_contiguous()
    R, C = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(C)
    grid = (R,)
    _softmax_kernel[grid](
        out,
        x,
        x.stride(0),
        out.stride(0),
        C,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
