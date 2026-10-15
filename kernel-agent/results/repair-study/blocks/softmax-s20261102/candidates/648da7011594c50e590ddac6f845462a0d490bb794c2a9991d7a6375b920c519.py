import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    out_ptr,
    in_ptr,
    n_rows,
    n_cols,
    in_row_stride,
    out_row_stride,
    BLOCK_SIZE: tl.constexpr,
    NUM_WARPS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    in_ptrs = in_ptr + row * in_row_stride + cols
    x = tl.load(in_ptrs, mask=mask, other=-float('inf')).to(tl.float32)
    x_max = tl.max(x, axis=0)
    e = tl.exp(x - x_max)
    denom = tl.sum(e, axis=0)
    y = e / denom
    out_ptrs = out_ptr + row * out_row_stride + cols
    tl.store(out_ptrs, y, mask=mask)
    return

def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2 and x.is_contiguous()
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # Choose warps: small rows use fewer warps to cut launch/sync overhead
    if BLOCK_SIZE <= 256:
        num_warps = 1
    elif BLOCK_SIZE <= 1024:
        num_warps = 2
    else:
        num_warps = 4
    grid = (n_rows,)
    _softmax_kernel[grid](
        out,
        x,
        n_rows,
        n_cols,
        x.stride(0),
        out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_WARPS=num_warps,
        num_warps=num_warps,
    )
    return out
