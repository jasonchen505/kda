import torch
import triton
import triton.language as tl


@triton.jit
def transpose_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    row_off = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = row_off < M
    col_mask = col_off < N
    mask = row_mask[:, None] & col_mask[None, :]
    x_ptrs = x_ptr + row_off[:, None] * stride_xm + col_off[None, :] * stride_xn
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    out_ptrs = out_ptr + col_off[:, None] * stride_om + row_off[None, :] * stride_on
    out_mask = col_mask[:, None] & row_mask[None, :]
    tl.store(out_ptrs, x, mask=out_mask)


def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    assert x.dim() == 2
    assert x.is_contiguous()
    M, N = x.shape
    out = torch.empty((N, M), dtype=x.dtype, device=x.device)
    if M >= 64 and N >= 64:
        BLOCK_M = 64
        BLOCK_N = 64
    else:
        BLOCK_M = 32
        BLOCK_N = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    transpose_kernel[grid](
        x, out,
        M, N,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return out
