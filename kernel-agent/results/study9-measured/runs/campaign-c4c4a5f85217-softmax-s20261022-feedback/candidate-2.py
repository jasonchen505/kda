import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    output_ptr, input_ptr, input_row_stride, output_row_stride,
    n_rows, n_cols,
    BLOCK_ROWS: tl.constexpr, BLOCK_COLS: tl.constexpr
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_ROWS
    row_offsets = row_start + tl.arange(0, BLOCK_ROWS)
    col_offsets = tl.arange(0, BLOCK_COLS)

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
    assert x.is_cuda, "input must be a CUDA tensor"
    assert x.dim() == 2, "input must be 2D"
    assert x.is_contiguous(), "input must be contiguous"
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)

    BLOCK_COLS = triton.next_power_of_2(n_cols)
    BLOCK_ROWS = 4
    grid = (triton.cdiv(n_rows, BLOCK_ROWS),)

    softmax_kernel[grid](
        output, x, x.stride(0), output.stride(0),
        n_rows, n_cols,
        BLOCK_ROWS=BLOCK_ROWS, BLOCK_COLS=BLOCK_COLS
    )
    return output
