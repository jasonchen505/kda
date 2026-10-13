import torch
import triton
import triton.language as tl

@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n_elements, OUT_DTYPE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask).to(tl.float32)
    s = x + y
    tl.store(out_ptr + offsets, s.to(OUT_DTYPE), mask=mask)

def run(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda
    assert x.shape == y.shape
    assert x.dtype in (torch.float32, torch.bfloat16)
    assert x.dtype == y.dtype
    out = torch.empty_like(x)
    n = out.numel()
    if n == 0:
        return out
    out_dtype = tl.bfloat16 if x.dtype == torch.bfloat16 else tl.float32
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    _add_kernel[grid](x, y, out, n, OUT_DTYPE=out_dtype, BLOCK_SIZE=1024)
    return out
