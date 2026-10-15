import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    input_row_stride,
    output_row_stride,
    BLOCK_SIZE: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_in = input_ptr + row_idx * input_row_stride
    row_start_out = output_ptr + row_idx * output_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    in_ptrs = row_start_in + col_offsets
    out_ptrs = row_start_out + col_offsets
    row = tl.load(in_ptrs, mask=mask, other=-float('inf'))
    if IS_BF16:
        row = row.to(tl.float32)
    row_max = tl.max(row, axis=0)
    row_minus_max = row - row_max
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator
    if IS_BF16:
        result = result.to(tl.bfloat16)
    tl.store(out_ptrs, result, mask=mask)
    

def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2 and x.is_contiguous()
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    is_bf16 = x.dtype == torch.bfloat16
    grid = (n_rows,)
    _softmax_kernel[grid](
        x,
        out,
        n_rows,
        n_cols,
        x.stride(0),
        out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        IS_BF16=is_bf16,
    )
    return out
