import hashlib

from kernel_agent.references import retrieve_example


def test_retrieved_code_is_attributed_and_hash_linked():
    result = retrieve_example("vector_add")
    assert "01-vector-add.py" in result["url"]
    assert hashlib.sha256(result["excerpt"].encode()).hexdigest() == result["excerpt_sha256"]
    assert "@triton.jit" in result["context"]


def test_rmsnorm_reference_requires_adaptation():
    result = retrieve_example("rmsnorm")
    assert "remove centering/bias" in result["context"]
    assert "05-layer-norm" in result["name"]
