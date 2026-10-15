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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    y_ptrs = y_ptr + offs_n[:, None] * stride_ym + offs_m[None, :] * stride_yn
    tl.store(y_ptrs, tl.trans(x), mask=mask)
    

def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "input must be CUDA"
    assert x.dim() == 2, "input must be 2D"
    assert x.is_contiguous(), "input must be contiguous"
    assert x.dtype in (torch.float32, torch.bfloat16), "unsupported dtype"
    M, N = x.shape
    y = torch.empty((N, M), dtype=x.dtype, device=x.device)
    BLOCK_M = 32
    BLOCK_N = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _transpose_kernel[grid](
        x,
        y,
        M,
        N,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return y
