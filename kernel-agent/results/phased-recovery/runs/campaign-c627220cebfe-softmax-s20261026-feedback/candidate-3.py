import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    input_ptrs = input_ptr + row_idx * input_row_stride + col_offsets
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

    if IS_BF16:
        row = row.to(tl.float32)

    row_max = tl.max(row, axis=0)
    row_minus_max = row - row_max
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    if IS_BF16:
        softmax_output = softmax_output.to(tl.bfloat16)

    output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input must be a CUDA tensor"
    assert x.dim() == 2, "Input must be 2D"
    assert x.is_contiguous(), "Input must be contiguous"
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    IS_BF16 = x.dtype == torch.bfloat16

    # Tune num_warps based on block size for optimal occupancy
    if BLOCK_SIZE <= 128:
        num_warps = 1
    elif BLOCK_SIZE <= 256:
        num_warps = 2
    elif BLOCK_SIZE <= 512:
        num_warps = 4
    else:
        num_warps = 8

    grid = (n_rows,)
    softmax_kernel[grid](
        output,
        x,
        x.stride(0),
        output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        IS_BF16=IS_BF16,
        num_warps=num_warps,
    )
    return output
