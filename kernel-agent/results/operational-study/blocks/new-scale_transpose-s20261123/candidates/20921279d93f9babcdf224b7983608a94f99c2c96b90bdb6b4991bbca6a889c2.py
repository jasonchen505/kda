import torch
import triton
import triton.language as tl


@triton.jit
def _scale_transpose_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    rows: tl.constexpr,
    width: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_r = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)
    r_off = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    c_off = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    r_mask = r_off < rows
    c_mask = c_off < width
    x_ptrs = x_ptr + r_off[:, None] * width + c_off[None, :]
    x_mask = r_mask[:, None] & c_mask[None, :]
    x = tl.load(x_ptrs, mask=x_mask, other=0.0)
    w = tl.load(w_ptr + c_off, mask=c_mask, other=0.0)
    y = x * w[None, :]
    out_ptrs = out_ptr + c_off[:, None] * rows + r_off[None, :]
    out_mask = c_mask[:, None] & r_mask[None, :]
    tl.store(out_ptrs, y.T, mask=out_mask)
    return


def run(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda
    assert x.dtype == weight.dtype
    assert x.is_contiguous() and weight.is_contiguous()
    rows, width = x.shape
    out = torch.empty((width, rows), dtype=x.dtype, device=x.device)
    if rows >= 128 and width >= 512:
        BLOCK_R = 64
        BLOCK_C = 64
    elif rows >= 64 and width >= 256:
        BLOCK_R = 32
        BLOCK_C = 64
    elif rows >= 32 and width >= 128:
        BLOCK_R = 32
        BLOCK_C = 32
    else:
        BLOCK_R = 16
        BLOCK_C = 16
    grid = (triton.cdiv(rows, BLOCK_R), triton.cdiv(width, BLOCK_C))
    _scale_transpose_kernel[grid](x, weight, out, rows, width, BLOCK_R=BLOCK_R, BLOCK_C=BLOCK_C)
    return out
