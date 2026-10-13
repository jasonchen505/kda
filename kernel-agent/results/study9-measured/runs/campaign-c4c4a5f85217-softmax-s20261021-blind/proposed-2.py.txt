import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    input_ptrs = input_ptr + row_idx * input_row_stride + col_offsets
    row = tl.load(input_ptrs, mask=mask, other=-float('inf')).to(tl.float32)
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "input must be a CUDA tensor"
    assert x.dim() == 2, "input must be 2D"
    assert x.is_contiguous(), "input must be contiguous"
    assert x.dtype in (torch.float32, torch.bfloat16), "dtype must be float32 or bfloat16"

    n_rows, n_cols = x.shape
    out = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    _softmax_kernel[(n_rows,)](
        x,
        out,
        x.stride(0),
        out.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out
