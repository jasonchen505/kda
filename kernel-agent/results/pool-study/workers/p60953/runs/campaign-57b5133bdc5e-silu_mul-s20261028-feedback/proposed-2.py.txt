import torch
import triton
import triton.language as tl


@triton.jit
def silu_mul_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    # sigmoid(x) = 1 / (1 + exp(-x)) = 1 / (1 + exp2(-x * log2(e)))
    LOG2E: tl.constexpr = 1.4426950408889634
    sigmoid_x = 1.0 / (1.0 + tl.exp2(-x * LOG2E))
    result = x * sigmoid_x * y
    tl.store(out_ptr + offsets, result, mask=mask)


def run(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda
    assert x.shape == y.shape
    assert x.is_contiguous() and y.is_contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    if n_elements <= 4096:
        BLOCK_SIZE = 256
    elif n_elements <= 65536:
        BLOCK_SIZE = 1024
    else:
        BLOCK_SIZE = 2048
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    silu_mul_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out
