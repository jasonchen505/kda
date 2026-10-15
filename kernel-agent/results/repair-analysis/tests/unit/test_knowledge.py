from kernel_agent.knowledge import retrieve


def test_runtime_feedback_selects_host_and_memory_api_contracts():
    result = retrieve({"status": "runtime_error", "message": "torch.dtype has no dtype"}, "vector_add")
    assert "launch" in result["card_ids"]
    assert "pass torch tensors directly" in result["context"]
    assert "v3.2.0" in result["sources"]["launch"]


def test_protocol_feedback_addresses_python_newlines():
    result = retrieve({"status": "protocol_error", "message": "invalid syntax"}, "rmsnorm")
    assert "format" in result["card_ids"] and "reduce" in result["card_ids"]
    assert "Keep each import" in result["context"]
