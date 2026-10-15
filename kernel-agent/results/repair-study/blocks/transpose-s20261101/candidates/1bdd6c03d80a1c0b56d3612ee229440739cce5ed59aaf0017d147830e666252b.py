import torch
import triton
import triton.language as tl


@triton.jit
def transpose_kernel(x_ptr, y_ptr, M, N,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_2d = mask_m[:, None] & mask_n[None, :]
    x_ptrs = x_ptr + offs_m[:, None] * M + offs_n[None, :]
    x = tl.load(x_ptrs, mask=mask_2d, other=0.0)
    y_ptrs = y_ptr + offs_n[:, None] * N + offs_m[None, :]
    tl.store(y_ptrs, x, mask=mask_2d)
    

def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2 and x.is_contiguous()
    M, N = x.shape
    y = torch.empty((N, M), dtype=x.dtype, device=x.device)
    BLOCK_M = 32
    BLOCK_N = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    transpose_kernel[grid](x, y, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    return y
