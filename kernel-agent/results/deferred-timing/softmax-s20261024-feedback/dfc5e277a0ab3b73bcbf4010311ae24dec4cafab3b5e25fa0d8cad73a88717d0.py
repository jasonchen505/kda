import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf')).to(tl.float32)
        row_max = tl.max(row, axis=0)
        row_minus_max = row - row_max
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)

def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    assert x.is_contiguous()
    assert x.dim() == 2
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    max_blocks = 1024
    grid_size = min(n_rows, max_blocks)
    grid = (grid_size,)
    softmax_kernel[grid](
        out,
        x,
        x.stride(0),
        out.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
