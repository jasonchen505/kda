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
    IS_BF16: tl.constexpr,
    num_stages: tl.constexpr,
):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        in_ptrs = input_ptr + row_idx * n_cols + col_offsets
        row_vals = tl.load(in_ptrs, mask=mask, other=-float('inf'))
        row_f32 = row_vals.to(tl.float32)
        row_max = tl.max(row_f32, axis=0)
        exp_vals = tl.exp(row_f32 - row_max)
        denom = tl.sum(exp_vals, axis=0)
        out_vals = exp_vals / denom
        if IS_BF16:
            out_vals = out_vals.to(tl.bfloat16)
        out_ptrs = output_ptr + row_idx * n_cols + col_offsets
        tl.store(out_ptrs, out_vals, mask=mask)


def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2 and x.is_contiguous()
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (min(n_rows, 1024),)
    _softmax_kernel[grid](x, out, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE, IS_BF16=(x.dtype == torch.bfloat16), num_stages=2)
    return out
