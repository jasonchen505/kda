import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr,
               y_ptr,
               output_ptr,
               n_elements,
               BLOCK_SIZE: tl.constexpr,
               ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask).to(tl.float32)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def run(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda
    assert x.is_contiguous() and y.is_contiguous()
    assert x.shape == y.shape
    assert x.dtype in (torch.float32, torch.bfloat16)
    output = torch.empty_like(x)
    n_elements = output.numel()
    if n_elements <= 1024:
        block_size = 128
    elif n_elements <= 131072:
        block_size = 1024
    else:
        block_size = 4096
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=block_size)
    return output
