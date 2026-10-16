import torch
import triton
import triton.language as tl


@triton.jit
def transpose_kernel(x_ptr, out_ptr, M, N, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M * N
    out_row = offs // M
    out_col = offs % M
    in_flat = out_col * N + out_row
    x = tl.load(x_ptr + in_flat, mask=mask)
    tl.store(out_ptr + offs, x, mask=mask)
    

def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2
    M, N = x.shape
    out = torch.empty((N, M), dtype=x.dtype, device=x.device)
    BLOCK = 128
    grid = (triton.cdiv(M * N, BLOCK),)
    transpose_kernel[grid](x, out, M, N, BLOCK=BLOCK)
    return out
