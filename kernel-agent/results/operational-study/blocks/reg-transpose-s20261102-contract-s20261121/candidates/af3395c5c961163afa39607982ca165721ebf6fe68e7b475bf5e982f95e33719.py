import torch
import triton
import triton.language as tl


@triton.jit
def transpose_kernel(x_ptr, out_ptr, M, N, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    out_mask = offs < (M * N)
    i = offs // M
    j = offs % M
    src_flat = j * N + i
    x = tl.load(x_ptr + src_flat, mask=out_mask, other=0.0)
    tl.store(out_ptr + offs, x, mask=out_mask)
    

def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2
    M, N = x.shape
    out = torch.empty((N, M), dtype=x.dtype, device=x.device)
    BLOCK = 32
    grid = (triton.cdiv(M * N, BLOCK),)
    transpose_kernel[grid](x, out, M, N, BLOCK=BLOCK)
    return out
