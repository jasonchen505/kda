import torch
import triton
import triton.language as tl


@triton.jit
def transpose_kernel(x_ptr, out_ptr, M, N, BLOCK: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK + tl.arange(0, BLOCK)
    offs_n = pid_n * BLOCK + tl.arange(0, BLOCK)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    x_ptrs = x_ptr + offs_m[:, None] * N + offs_n[None, :]
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    out_ptrs = out_ptr + offs_n[:, None] * M + offs_m[None, :]
    tl.store(out_ptrs, x, mask=mask)
    

def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2
    M, N = x.shape
    out = torch.empty((N, M), dtype=x.dtype, device=x.device)
    BLOCK = 32
    grid = (triton.cdiv(M, BLOCK), triton.cdiv(N, BLOCK))
    transpose_kernel[grid](x, out, M, N, BLOCK=BLOCK)
    return out
