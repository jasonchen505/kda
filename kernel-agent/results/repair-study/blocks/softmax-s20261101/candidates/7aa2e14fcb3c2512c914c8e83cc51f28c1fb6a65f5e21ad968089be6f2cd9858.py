import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    NUM_PROGS: tl.constexpr,
):
    row_start = tl.program_id(0)
    row_step = NUM_PROGS
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=NUM_STAGES):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf')).to(tl.float32)
        row_max = tl.max(row, axis=0)
        numerator = tl.exp(row - row_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)
    return


def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2 and x.is_contiguous()
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    BLOCK_SIZE = triton.cdiv(n_cols, 1)
    BLOCK_SIZE = 1 << (BLOCK_SIZE - 1).bit_length()
    NUM_PROGS = min(n_rows, 256)
    grid = (NUM_PROGS,)
    _softmax_kernel[grid](
        output,
        x,
        n_cols,
        n_cols,
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_STAGES=2,
        NUM_PROGS=NUM_PROGS,
    )
    return output
