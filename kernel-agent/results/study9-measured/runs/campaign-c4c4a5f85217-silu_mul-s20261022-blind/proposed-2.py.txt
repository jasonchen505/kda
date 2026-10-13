import torch
import triton
import triton.language as tl

@triton.jit
def silu_mul_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask).to(tl.float32)
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    result = x * sigmoid_x * y
    tl.store(output_ptr + offsets, result, mask=mask)

def run(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda
    assert x.shape == y.shape
    assert x.is_contiguous() and y.is_contiguous()
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    silu_mul_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
