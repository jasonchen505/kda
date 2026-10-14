import torch
import triton
import triton.language as tl


@triton.jit
def _transpose_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    x_ptrs = x_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = mask_m[:, None] & mask_n[None, :]
    tile = tl.load(x_ptrs, mask=mask, other=0.0)

    out_ptrs = out_ptr + offs_n[:, None] * M + offs_m[None, :]
    out_mask = mask_n[:, None] & mask_m[None, :]
    tl.store(out_ptrs, tl.trans(tile), mask=out_mask)


def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2 and x.is_contiguous()
    M, N = x.shape[0], x.shape[1]
    out = torch.empty((N, M), dtype=x.dtype, device=x.device)

    # Choose block size based on matrix dimensions
    # Smaller blocks reduce wasted work for small/odd matrices
    if M <= 64 and N <= 64:
        BLOCK_M = 32
        BLOCK_N = 32
    elif M <= 128 and N <= 128:
        BLOCK_M = 32
        BLOCK_N = 32
    else:
        BLOCK_M = 64
        BLOCK_N = 64

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _transpose_kernel[grid](x, out, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    return out
