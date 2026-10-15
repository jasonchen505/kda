import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_partial(
    input_ptr,
    max_ptr,
    sum_ptr,
    input_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    block_id: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_start = block_id * BLOCK_SIZE
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row_ptr = input_ptr + row_idx * input_row_stride
    row = tl.load(row_ptr + col_offsets, mask=mask, other=-float('inf')).to(tl.float32)
    local_max = tl.max(row, axis=0)
    numerator = tl.exp(row - local_max)
    local_sum = tl.sum(numerator, axis=0)
    tl.store(max_ptr + row_idx * NUM_BLOCKS + block_id, local_max)
    tl.store(sum_ptr + row_idx * NUM_BLOCKS + block_id, local_sum)


@triton.jit
def _softmax_finalize(
    input_ptr,
    output_ptr,
    max_ptr,
    sum_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    block_id: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_start = block_id * BLOCK_SIZE
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row_ptr = input_ptr + row_idx * input_row_stride
    row = tl.load(row_ptr + col_offsets, mask=mask, other=-float('inf')).to(tl.float32)
    local_max = tl.load(max_ptr + row_idx * NUM_BLOCKS + block_id)
    local_sum = tl.load(sum_ptr + row_idx * NUM_BLOCKS + block_id)
    numerator = tl.exp(row - local_max)
    out_ptr = output_ptr + row_idx * output_row_stride
    result = numerator / local_sum
    tl.store(out_ptr + col_offsets, result, mask=mask)


def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2 and x.is_contiguous()
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    if n_cols <= 1024:
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        grid = (n_rows,)
        num_warps = 8 if BLOCK_SIZE >= 1024 else 4
        _softmax_single[grid](
            x,
            out,
            x.stride(0),
            out.stride(0),
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
    else:
        BLOCK_SIZE = 1024
        NUM_BLOCKS = triton.cdiv(n_cols, BLOCK_SIZE)
        max_ws = torch.empty(n_rows * NUM_BLOCKS, device=x.device, dtype=torch.float32)
        sum_ws = torch.empty(n_rows * NUM_BLOCKS, device=x.device, dtype=torch.float32)
        for b in range(NUM_BLOCKS):
            _softmax_partial[(n_rows,)](
                x,
                max_ws,
                sum_ws,
                x.stride(0),
                n_cols,
                BLOCK_SIZE=BLOCK_SIZE,
                NUM_BLOCKS=NUM_BLOCKS,
                block_id=b,
                num_warps=8,
            )
        for b in range(NUM_BLOCKS):
            _softmax_finalize[(n_rows,)](
                x,
                out,
                max_ws,
                sum_ws,
                x.stride(0),
                out.stride(0),
                n_cols,
                BLOCK_SIZE=BLOCK_SIZE,
                NUM_BLOCKS=NUM_BLOCKS,
                block_id=b,
                num_warps=8,
            )
    return out


@triton.jit
def _softmax_single(
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
    row_ptr = input_ptr + row_idx * input_row_stride
    row = tl.load(row_ptr + col_offsets, mask=mask, other=-float('inf')).to(tl.float32)
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator
    out_ptr = output_ptr + row_idx * output_row_stride
    tl.store(out_ptr + col_offsets, result, mask=mask)
