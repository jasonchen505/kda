import torch
import triton
import triton.language as tl


@triton.jit
def _swish_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + (offsets % width), mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + (offsets % width), mask=mask, other=0.0).to(tl.float32)
    
    z = x * w + b
    out = z * tl.sigmoid(z)
    
    tl.store(out_ptr + offsets, out, mask=mask)


def run(x, weight, bias):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    assert x.dtype == weight.dtype == bias.dtype
    assert x.is_contiguous() and weight.is_contiguous() and bias.is_contiguous()
    
    out = torch.empty_like(x)
    n = x.numel()
    width = x.shape[1]
    
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    _swish_kernel[grid](x, weight, bias, out, n, width, BLOCK_SIZE=1024)
    
    return out
