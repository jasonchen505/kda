import torch
import triton
import triton.language as tl


@triton.jit
def _zsig_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    col = offsets % width
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + col, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + col, mask=mask, other=0.0).to(tl.float32)
    z = x * w + b
    sig = 1.0 / (1.0 + tl.exp(-z))
    res = z * sig
    tl.store(out_ptr + offsets, res.to(OUT_DTYPE), mask=mask)
    

def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    assert x.dtype == weight.dtype == bias.dtype
    assert x.is_contiguous() and weight.is_contiguous() and bias.is_contiguous()
    rows, width = x.shape
    out = torch.empty_like(x)
    n_elements = rows * width
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _zsig_kernel[grid](x, weight, bias, out, n_elements, width=width, BLOCK_SIZE=1024, OUT_DTYPE=x.dtype)
    return out
