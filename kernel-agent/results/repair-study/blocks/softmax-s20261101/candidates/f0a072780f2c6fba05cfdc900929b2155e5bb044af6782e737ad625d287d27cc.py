import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    IS_BF16: tl.constexpr,
    IS_FP32: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
    if IS_BF16:
        row = row.to(tl.float32)
    elif IS_FP32:
        row = row.to(tl.float32)
    else:
        row = row.to(tl.float32)
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    if IS_BF16:
        softmax_output = softmax_output.to(tl.bfloat16)
    elif IS_FP32:
        softmax_output = softmax_output.to(tl.float32)
    else:
        softmax_output = softmax_output.to(tl.float32)
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)
    return


def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2 and x.is_contiguous()
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    BLOCK_SIZE = 1 << (n_cols - 1).bit_length()
    grid = (n_rows,)
    is_bf16 = x.dtype == torch.bfloat16
    is_fp32 = x.dtype == torch.float32
    _softmax_kernel[grid](
        output,
        x,
        n_cols,
        n_cols,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        IS_BF16=is_bf16,
        IS_FP32=is_fp32,
    )
    return output
