"""Catch a Torch matmul hidden behind an unused JIT decoration before scoring."""
import pytest

from kernel_agent.representative_policy import validate_host


def test_host_compute_rejected_but_jit_compute_and_metadata_allowed():
    with pytest.raises(ValueError,match='Host matrix multiply'):
        validate_host('import triton\n@triton.jit\ndef unused(): pass\ndef run(a,b): return a@b\n')
    with pytest.raises(ValueError,match='softmax'):
        validate_host('def run(x): return x.softmax(-1)\n')
    validate_host('def metadata(x):\n    return x.transpose(0,1).permute(1,0)\n')
    validate_host('import triton\n@triton.jit\ndef gemm(a,b):\n    x = a @ b\n    return x.to(tl.float32)\ndef run(a,b):\n    m,k = a.shape\n    out = torch.empty((m,b.shape[1]),device=a.device,dtype=a.dtype)\n    gemm[(1,)](a,b,out)\n    return out\n')
