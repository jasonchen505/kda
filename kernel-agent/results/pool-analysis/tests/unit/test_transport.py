import copy

import pytest

from kernel_agent.transport import PROPOSER_PREFIX, normalize


def request(system=PROPOSER_PREFIX):
    return {"model": "kernel-qwen3-4b", "messages": [
        {"role": "system", "content": system}, {"role": "user", "content": "contract"}],
        "tools": [{"type": "function", "function": {"name": "read"}}],
        "tool_choice": "auto", "parallel_tool_calls": True, "stream": True}


def test_tool_free_transport_preserves_contract_and_does_not_mutate_input():
    original = request()
    before = copy.deepcopy(original)
    result = normalize(original, "tool-free")
    assert result["messages"] == before["messages"] and result["stream"]
    assert not any(key in result for key in ("tools", "tool_choice", "parallel_tool_calls"))
    assert original == before


def test_schema_transport_requires_all_proposal_fields_and_forbids_extra_fields():
    result = normalize(request(), "json-schema")
    schema = result["response_format"]["json_schema"]["schema"]
    assert set(schema["required"]) == {"draft", "plan", "code"}
    assert schema["additionalProperties"] is False and "tools" not in result


def test_title_requests_are_not_rewritten_even_when_user_contains_task_prefix():
    original = request("You are a title generator.")
    original["messages"][1]["content"] = PROPOSER_PREFIX
    assert normalize(original, "json-schema") == original


def test_tool_history_is_rejected_rather_than_erased():
    original = request()
    original["messages"].append({"role": "tool", "content": "result", "tool_call_id": "call"})
    with pytest.raises(ValueError, match="tool conversation"):
        normalize(original, "json-schema")


def test_passthrough_retains_offered_tools():
    assert normalize(request(), "passthrough") == request()


def test_line_schema_requires_an_array_without_the_conflicting_code_string():
    schema = normalize(request(), "json-lines")["response_format"]["json_schema"]["schema"]
    assert "code" not in schema["properties"]
    assert schema["properties"]["code_lines"]["type"] == "array"
