import torch
import triton
import triton.language as tl


@triton.jit
def transpose_kernel(x_ptr, out_ptr,
                       M, N,
                       stride_xm, stride_xn,
                       stride_om, stride_on,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    x_t = tl.trans(x)
    out_ptrs = out_ptr + offs_n[:, None] * stride_om + offs_m[None, :] * stride_on
    out_mask = mask_n[:, None] & mask_m[None, :]
    tl.store(out_ptrs, x_t, mask=out_mask)


def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "input must be CUDA"
    assert x.dim() == 2, "input must be 2D"
    assert x.is_contiguous(), "input must be contiguous"
    M, N = x.shape
    out = torch.empty((N, M), dtype=x.dtype, device=x.device)
    if M == 0 or N == 0:
        return out
    # Choose block sizes based on matrix dimensions for optimal performance
    if M >= 256 and N >= 256:
        BLOCK_M = 64
        BLOCK_N = 64
    elif M >= 128 or N >= 128:
        BLOCK_M = 32
        BLOCK_N = 64
    else:
        BLOCK_M = 32
        BLOCK_N = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    transpose_kernel[grid](x, out, M, N,
                           x.stride(0), x.stride(1),
                           out.stride(0), out.stride(1),
                           BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    return out
