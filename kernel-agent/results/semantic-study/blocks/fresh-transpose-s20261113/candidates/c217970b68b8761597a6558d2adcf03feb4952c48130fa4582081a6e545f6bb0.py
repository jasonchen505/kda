import torch
import triton
import triton.language as tl


@triton.jit
def _transpose_kernel(
    x_ptr,
    y_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    mask = mask_m[:, None] & mask_n[None, :]
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    y_ptrs = y_ptr + offs_n[:, None] * stride_ym + offs_m[None, :] * stride_yn
    tl.store(y_ptrs, x, mask=mask)
    

def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "input must be CUDA"
    assert x.dim() == 2, "input must be 2D"
    assert x.is_contiguous(), "input must be contiguous"
    assert x.dtype in (torch.float32, torch.bfloat16), "unsupported dtype"
    M, N = x.shape
    y = torch.empty((N, M), dtype=x.dtype, device=x.device)
    BLOCK_N = 32
    BLOCK_M = 32
    grid = (triton.cdiv(N, BLOCK_N),)
    _transpose_kernel[grid](
        x,
        y,
        M,
        N,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_N=BLOCK_N,
        BLOCK_M=BLOCK_M,
    )
    return y
