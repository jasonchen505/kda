"""Run current-source checks through the consolidated server launcher."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--preflight-span", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--focused", action="store_true")
    args = parser.parse_args()
    if args.preflight_span:
        from kernel_agent import model_pool, span_repair
        from kernel_agent.campaign import atomic_json
        capability = json.loads((ROOT / "data/runtime-triton-api.json").read_text())
        specs = {**json.loads((ROOT / "data/tasks.json").read_text()),
                 **json.loads((ROOT / "data/repair-transfer-tasks.json").read_text())}
        endpoints = model_pool.discover()
        seen, rows = set(), []
        for p in sorted((ROOT / "results/operational-study/blocks").glob("*/result.json")):
            result = json.loads(p.read_text())
            initial = result["initial"]
            source_sha = initial.get("source_sha256")
            if not source_sha or source_sha in seen:
                continue
            seen.add(source_sha)
            for mode in ("span_patch", "full_source_fallback"):
                history = [initial] + ([{"request_id": "span_repair-r1", "mode": "span_patch",
                                        "evaluation": {"status": "patch_error", "message": "synthetic protocol-preflight only"}}]
                                       if mode == "full_source_fallback" else [])
                parts, settings = span_repair.context(result["block"]["task_id"], specs[result["block"]["task_id"]]["contract"], history, capability, mode)
                for e in endpoints:
                    from kernel_agent.transport import PROPOSER_PREFIX
                    messages = [{"role": "system", "content": settings.get("role_prefix", PROPOSER_PREFIX) + " No tools. Follow the supplied contract."},
                                {"role": "user", "content": "\n\n".join(t for _, t, _ in parts)}]
                    n = model_pool.http(e["port"], "/tokenize", {"model": e["model"], "messages": messages,
                                      "add_generation_prompt": True, "chat_template_kwargs": {"enable_thinking": False}})["count"]
                    rows.append({"port": e["port"], "source_sha256": source_sha, "mode": mode, "input_tokens": n})
                    if n + 2048 > 8192:
                        raise ValueError("mandatory indexed context exceeds common limit")
        value = {"status": "passed", "scope": "Tokenization only, zero model completions", "sources": len(seen), "rows": rows}
        if args.output:
            atomic_json(args.output, value)
        print(json.dumps({"status": "passed", "sources": len(seen), "tokenize_calls": len(rows),
                          "maximum_input_tokens": max(r["input_tokens"] for r in rows)}))
        return
    if args.inventory:
        from kernel_agent import model_pool
        rows = []
        for e in model_pool.discover():
            try:
                rows.append({"endpoint": e, "load": model_pool.validate(e)})
            except (OSError, ValueError) as error:
                rows.append({"endpoint": e, "error": str(error)})
        print(json.dumps(rows))
        return
    commands = [[sys.executable, "-m", "pytest", "-q"] + (
        ["tests/unit/test_span_repair.py", "tests/integration/test_span_study.py"] if args.focused else []),
        [sys.executable, "-m", "ruff", "check", "src", "scripts", "tests", "flows"]]
    rows = []
    for command in commands:
        p = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=180,
                           env={**os.environ, "PYTHONPATH": str(ROOT / "src") + ":" + str(ROOT)})
        rows.append({"command": command, "exit_code": p.returncode, "stdout": p.stdout, "stderr": p.stderr})
        print(json.dumps(rows[-1]), flush=True)
        if p.returncode:
            break
    if args.output:
        from kernel_agent.campaign import atomic_json
        atomic_json(args.output, {"status": "passed" if len(rows) == len(commands) and all(r["exit_code"] == 0 for r in rows) else "failed",
                                  "scope": "CPU fake-driver mechanisms and lint; no agent/model/GPU scores", "commands": rows})
    sys.exit(0 if len(rows) == len(commands) and all(r["exit_code"] == 0 for r in rows) else 1)


if __name__ == "__main__":
    main()
