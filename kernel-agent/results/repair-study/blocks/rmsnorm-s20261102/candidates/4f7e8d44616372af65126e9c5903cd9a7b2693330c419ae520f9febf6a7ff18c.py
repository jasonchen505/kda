import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    X,
    W,
    Y,
    stride_x,
    stride_y,
    width,
    eps,
    BLOCK: tl.constexpr,
    SINGLE: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr = X + row * stride_x
    y_ptr = Y + row * stride_y
    if SINGLE:
        cols = tl.arange(0, BLOCK)
        mask = cols < width
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        acc = tl.sum(x * x, axis=0)
        mean = acc / width
        rstd = tl.rsqrt(mean + eps)
        w = tl.load(W + cols, mask=mask, other=0.0)
        normed = (x * rstd).to(x.dtype)
        y = w * normed
        tl.store(y_ptr + cols, y, mask=mask)
    else:
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for off in range(0, width, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            mask = cols < width
            x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            acc += x * x
        mean = tl.sum(acc, axis=0) / width
        rstd = tl.rsqrt(mean + eps)
        for off in range(0, width, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            mask = cols < width
            x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(W + cols, mask=mask, other=0.0)
            normed = (x * rstd).to(x.dtype)
            y = w * normed
            tl.store(y_ptr + cols, y, mask=mask)


def run(x, weight):
    rows, width = x.shape
    y = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(width)
    if width <= 256:
        BLOCK = 256
    single = BLOCK == width
    _rmsnorm_kernel[(rows,)](
        x,
        weight,
        y,
        x.stride(0),
        y.stride(0),
        width,
        1e-6,
        BLOCK=BLOCK,
        SINGLE=single,
    )
    return y
