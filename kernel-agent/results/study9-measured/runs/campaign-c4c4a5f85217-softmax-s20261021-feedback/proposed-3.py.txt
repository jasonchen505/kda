import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr, ROWS_PER_BLOCK: tl.constexpr):
    row_start = tl.program_id(0) * ROWS_PER_BLOCK
    row_offsets = row_start + tl.arange(0, ROWS_PER_BLOCK)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    row_mask = row_offsets < n_rows
    col_mask = col_offsets < n_cols
    mask = row_mask[:, None] & col_mask[None, :]
    input_ptrs = input_ptr + row_offsets[:, None] * input_row_stride + col_offsets[None, :]
    row = tl.load(input_ptrs, mask=mask, other=-float('inf')).to(tl.float32)
    row_max = tl.max(row, axis=1)
    row_minus_max = row - row_max[:, None]
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=1)
    softmax_output = numerator / denominator[:, None]
    output_ptrs = output_ptr + row_offsets[:, None] * output_row_stride + col_offsets[None, :]
    tl.store(output_ptrs, softmax_output, mask=mask)

def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    assert x.dim() == 2
    assert x.is_contiguous()
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    if n_rows <= 8:
        ROWS_PER_BLOCK = 1
    elif n_rows <= 32:
        ROWS_PER_BLOCK = 4
    elif n_rows <= 128:
        ROWS_PER_BLOCK = 8
    else:
        ROWS_PER_BLOCK = 16
    grid = (triton.cdiv(n_rows, ROWS_PER_BLOCK),)
    softmax_kernel[grid](
        output, x, x.stride(0), output.stride(0), n_rows, n_cols, BLOCK_SIZE, ROWS_PER_BLOCK
    )
    return output
