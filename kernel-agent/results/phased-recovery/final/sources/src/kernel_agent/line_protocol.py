"""Represent Python source as explicit lines, without interpreting escape sequences."""
import json


def proposal(text):
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except ValueError:
            continue
        if not isinstance(value, dict):
            continue
        if not all(isinstance(value.get(key), str) and value[key].strip() for key in ("draft", "plan")):
            continue
        lines = value.get("code_lines")
        if ("code" in value or not isinstance(lines, list) or not 1 <= len(lines) <= 2000
                or not all(isinstance(line, str) and "\n" not in line and "\r" not in line for line in lines)):
            continue
        source = "\n".join(lines) + "\n"
        if not source.strip() or len(source) > 60000:
            continue
        return {"draft": value["draft"], "plan": value["plan"], "code": source}
    raise ValueError("reply must contain draft/plan strings and code_lines, one Python line per item")


FORMAT_RULES = ("Return a JSON object with draft and plan strings, plus code_lines: an array of strings. "
                "Each array item is exactly one Python source line. Preserve leading indentation spaces; "
                "use an empty string for a blank line. Never concatenate source lines. Do not use a code field. "
                "Do not embed newline characters inside an array item. Example of formatting only: "
                '{"draft":"short draft","plan":"short plan","code_lines":'
                '["import math","","def example(x):","    return x"]}. ')


def first_prompt(base):
    old = "Return a JSON object with exactly three string fields: draft, plan, code. "
    return base.replace(old, FORMAT_RULES).replace("The code must import", "The resulting source must import")


def feedback(base):
    return base.replace("Return valid JSON with draft, plan and complete code strings.", FORMAT_RULES).replace(
        "Return a new complete proposal JSON.", FORMAT_RULES)


def reference_context(base):
    # The older formatting card describes code-string escaping; it conflicts with this protocol.
    if "[format]" not in base:
        return base
    return base[:base.index("[format]")] + "[format] " + FORMAT_RULES
