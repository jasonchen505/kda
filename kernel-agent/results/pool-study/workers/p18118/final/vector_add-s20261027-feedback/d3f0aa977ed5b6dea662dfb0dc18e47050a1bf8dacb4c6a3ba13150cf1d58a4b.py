import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask).to(tl.float32)
    out = (x + y).to(x_ptr.dtype.element_ty)
    tl.store(output_ptr + offsets, out, mask=mask)


def run(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    if n_elements >= 65536:
        BLOCK_SIZE = 1024
        num_warps = 4
    elif n_elements >= 8192:
        BLOCK_SIZE = 512
        num_warps = 4
    else:
        BLOCK_SIZE = 256
        num_warps = 2
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    return output
