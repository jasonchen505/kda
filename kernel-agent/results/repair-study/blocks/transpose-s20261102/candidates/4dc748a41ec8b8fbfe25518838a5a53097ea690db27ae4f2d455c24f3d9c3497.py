import torch
import triton
import triton.language as tl


@triton.jit
def transpose_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    x_ptrs = x_ptr + offs_m[:, None] * N + offs_n[None, :]
    x = tl.load(x_ptrs, mask=mask)
    out_ptrs = out_ptr + offs_n[:, None] * M + offs_m[None, :]
    tl.store(out_ptrs, x.T, mask=mask.T)


def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    assert x.dim() == 2
    assert x.is_contiguous()
    M, N = x.shape
    out = torch.empty((N, M), dtype=x.dtype, device=x.device)
    BLOCK_M = 64
    BLOCK_N = 64
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    transpose_kernel[grid](x, out, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    return out
