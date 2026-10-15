import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    input_ptr,
    output_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row_ptrs = input_ptr + row_idx * n_cols + col_offsets
    row = tl.load(row_ptrs, mask=mask, other=-float('inf')).to(tl.float32)
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator
    out_ptrs = output_ptr + row_idx * n_cols + col_offsets
    tl.store(out_ptrs, result, mask=mask)


def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2 and x.is_contiguous()
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows,)
    _softmax_kernel[grid](x, out, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out
