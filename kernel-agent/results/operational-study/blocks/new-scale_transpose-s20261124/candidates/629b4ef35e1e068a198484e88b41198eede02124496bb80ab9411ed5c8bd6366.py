import torch
import triton
import triton.language as tl


@triton.jit
def _transpose_scale_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    rows,
    width,
    BLOCK_I: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    pid_i = tl.program_id(axis=0)
    pid_j = tl.program_id(axis=1)
    i0 = pid_i * BLOCK_I
    j0 = pid_j * BLOCK_J
    i_off = i0 + tl.arange(0, BLOCK_I)
    j_off = j0 + tl.arange(0, BLOCK_J)
    i_mask = i_off < rows
    j_mask = j_off < width
    x_ptrs = x_ptr + i_off[:, None] * width + j_off[None, :]
    x = tl.load(x_ptrs, mask=i_mask[:, None] & j_mask[None, :], other=0.0)
    w_ptrs = w_ptr + j_off
    w = tl.load(w_ptrs, mask=j_mask, other=0.0)
    w = w.to(tl.float32)
    x = x.to(tl.float32)
    prod = x * w[None, :]
    out_ptrs = out_ptr + j_off[:, None] * rows + i_off[None, :]
    tl.store(out_ptrs, prod, mask=j_mask[:, None] & i_mask[None, :])


def run(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda
    assert x.dtype == weight.dtype
    assert x.is_contiguous() and weight.is_contiguous()
    rows, width = x.shape
    out = torch.empty((width, rows), dtype=x.dtype, device=x.device)
    BLOCK_I = 32
    BLOCK_J = 32
    grid = (triton.cdiv(rows, BLOCK_I), triton.cdiv(width, BLOCK_J))
    _transpose_scale_kernel[grid](x, weight, out, rows, width, BLOCK_I=BLOCK_I, BLOCK_J=BLOCK_J)
    return out
