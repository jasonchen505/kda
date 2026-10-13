import torch
import triton
import triton.language as tl


@triton.jit
def transpose_kernel(x_ptr, out_ptr, M, N, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    pid_r = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)
    row_offsets = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    col_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    row_mask = row_offsets < M
    col_mask = col_offsets < N
    mask = row_mask[:, None] & col_mask[None, :]
    x_ptrs = x_ptr + row_offsets[:, None] * N + col_offsets[None, :]
    tile = tl.load(x_ptrs, mask=mask, other=0.0)
    tile_t = tl.trans(tile)
    out_row_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    out_col_offsets = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    out_row_mask = out_row_offsets < N
    out_col_mask = out_col_offsets < M
    out_mask = out_row_mask[:, None] & out_col_mask[None, :]
    out_ptrs = out_ptr + out_row_offsets[:, None] * M + out_col_offsets[None, :]
    tl.store(out_ptrs, tile_t, mask=out_mask)


def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2
    M, N = x.shape
    out = torch.empty((N, M), dtype=x.dtype, device=x.device)
    # Choose block sizes based on matrix dimensions
    if M <= 64 and N <= 64:
        BLOCK_R = 32
        BLOCK_C = 32
    elif M <= 256 and N <= 256:
        BLOCK_R = 64
        BLOCK_C = 64
    else:
        BLOCK_R = 128
        BLOCK_C = 128
    grid = (triton.cdiv(M, BLOCK_R), triton.cdiv(N, BLOCK_C))
    transpose_kernel[grid](x, out, M, N, BLOCK_R=BLOCK_R, BLOCK_C=BLOCK_C)
    return out
