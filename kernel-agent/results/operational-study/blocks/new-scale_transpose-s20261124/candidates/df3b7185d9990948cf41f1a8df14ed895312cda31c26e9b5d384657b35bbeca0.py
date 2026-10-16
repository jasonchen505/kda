import torch
import triton
import triton.language as tl


@triton.jit
def _k(
    x_ptr, w_ptr, out_ptr, rows, width,
    BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr,
):
    pi = tl.program_id(0)
    pj = tl.program_id(1)
    i0 = pi * BLOCK_I
    j0 = pj * BLOCK_J
    i = i0 + tl.arange(0, BLOCK_I)
    j = j0 + tl.arange(0, BLOCK_J)
    im = i < rows
    jm = j < width
    xp = x_ptr + i[:, None] * width + j[None, :]
    x = tl.load(xp, mask=im[:, None] & jm[None, :], other=0.0)
    w = tl.load(w_ptr + j, mask=jm, other=0.0)
    r = x.to(tl.float32) * w.to(tl.float32)[None, :]
    op = out_ptr + j[:, None] * rows + i[None, :]
    tl.store(op, r, mask=jm[:, None] & im[None, :])


def run(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda
    assert x.dtype == weight.dtype
    assert x.is_contiguous() and weight.is_contiguous()
    rows, width = x.shape
    out = torch.empty((width, rows), dtype=x.dtype, device=x.device)
    BI = 32
    BJ = 32
    grid = (triton.cdiv(rows, BI), triton.cdiv(width, BJ))
    _k[grid](x, weight, out, rows, width, BLOCK_I=BI, BLOCK_J=BJ)
    return out
