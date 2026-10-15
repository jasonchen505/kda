import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    NUM_SMS: tl.constexpr,
    num_stages: tl.constexpr,
):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        input_ptrs = input_ptr + row_idx * n_cols + col_offsets
        row = tl.load(input_ptrs, mask=mask, other=-float('inf')).to(tl.float32)
        row_max = tl.max(row, axis=0)
        numerator = tl.exp(row - row_max)
        denominator = tl.sum(numerator, axis=0)
        result = numerator / denominator
        output_ptrs = output_ptr + row_idx * n_cols + col_offsets
        tl.store(output_ptrs, result, mask=mask)


def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2 and x.is_contiguous()
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    NUM_SMS = torch.cuda.get_device_properties(x.device).multi_processor_count
    grid = (min(NUM_SMS, n_rows),)
    _softmax_kernel[grid](x, out, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE, NUM_SMS=NUM_SMS, num_stages=2)
    return out
