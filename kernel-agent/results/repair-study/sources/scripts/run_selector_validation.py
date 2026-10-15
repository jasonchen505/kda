"""Run a fresh, gate-first integration replication without overwriting strategy evidence."""

import argparse
import hashlib
import json
import os
import random
from datetime import UTC, datetime
from pathlib import Path

from hmz.sdk import Hmz

ROOT = Path(__file__).resolve().parents[1]
GPU_PYTHON = "/data/home/chenyizhou/kda-humanize-portfolio/.venv/bin/python"


def resolve_result(name: str) -> Path:
    path = Path(name)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--output", default="results/agent-selector-validation.json")
    parser.add_argument("--runtime-manifest", help="Recorded existing-service configuration, relative to project")
    args = parser.parse_args()

    os.environ["OPENCODE_DISABLE_AUTOUPDATE"] = "true"
    os.environ["OPENCODE_DISABLE_DEFAULT_PLUGINS"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    os.environ["PYTHONPATH"] = str(ROOT / "src")
    config = json.loads((ROOT / "data/opencode-qwen27.json").read_text())
    config["provider"]["kernel-local"]["options"]["baseURL"] = "http://127.0.0.1:18116/json-lines/v1"
    os.environ["OPENCODE_CONFIG_CONTENT"] = json.dumps(config)
    fixture = json.loads((ROOT / "data/tasks.json").read_text())
    source_names = ["flows/kernel_loop.py", "src/kernel_agent/protocol.py", "src/kernel_agent/knowledge.py",
                    "scripts/evaluate_candidate.py", "scripts/local_proxy.py", "scripts/run_selector_validation.py",
                    "src/kernel_agent/transport.py", "src/kernel_agent/line_protocol.py", "data/tasks.json",
                    "data/opencode-qwen27.json", "results/qwen27-asset-manifest.json",
                    "scripts/start_cached_qwen27.sh", "scripts/inventory_model.py",
                    "src/kernel_agent/references.py", "references/triton-3.2/manifest.json",
                    "docs/selector-validation-plan.md"]
    runtime = None
    if args.runtime_manifest:
        runtime_path = resolve_result(args.runtime_manifest)
        source_names.append(str(runtime_path.relative_to(ROOT)))
        source_names.append("scripts/reproduce_existing_selector.sh")
        runtime = json.loads(runtime_path.read_text())
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_names}
    schedule = [(task, "feedback") for task in ("rmsnorm", "row_sum", "silu_mul")]
    random.Random(args.seed).shuffle(schedule)
    run_stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    target = resolve_result(args.output)
    if target.exists():
        raise RuntimeError(f"result exists; archive it explicitly before a fresh replication: {target}")
    task_split = {"rmsnorm": "development_replication", "row_sum": "task_replication",
                  "silu_mul": "task_replication"}
    evidence = {
        "schema_version": 1,
        "status": "running",
        "started_utc": run_stamp,
        "metadata": {
            "study_kind": "selector_validation",
            "parent_study_sha256": hashlib.sha256((ROOT / "results/agent-strategy-study.json").read_bytes()).hexdigest(),
            "model": "Cached Qwen 27B FP8 (config: qwen3_5)",
            "model_id": "kernel-qwen-local-27b",
            "model_adapter": "none",
            "model_revision": None,
            "model_asset": json.loads((ROOT / "results/qwen27-asset-manifest.json").read_text()),
            "generation_gpus": runtime["generation_gpus"] if runtime else "4,5",
            "generation_service": runtime or {"mode": "dedicated", "endpoint": "http://127.0.0.1:18118"},
            "gpu": "RTX 4090, physical GPU 6",
            "backend": "Humanize Agent → OpenCode 1.18.32 → local vLLM",
            "temperature": 0,
            "seed": args.seed,
            "thinking": False,
            "max_tokens_per_request": 4096,
            "rounds": 3,
            "selection_policy": "gate_first",
            "transport_policies": ["json-lines"],
            "study_policies": ["feedback"],
            "task_split": task_split,
            "source_sha256": hashes,
            "opencode_config": "data/opencode-qwen27.json",
            "study_scope": ("three fresh feedback-agent replications using gate-first selection; "
                            "no blind arm and no claim of a feedback-policy effect"),
        },
        "schedule": schedule,
        "runs": [],
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(evidence, indent=2) + "\n")
    for index, (task, policy) in enumerate(schedule):
        run_id = f"{run_stamp}-{task}-selector-{policy}"
        print(json.dumps({"starting": index + 1, "task": task, "study_policy": policy}), flush=True)
        try:
            result = Hmz(ROOT).run(
                str(ROOT / "flows/kernel_loop.py"),
                fixture[task]["contract"] + "\nVisible examples: " + json.dumps(fixture[task]["examples"]),
                agents={"builder": "opencode/kernel-local/kernel-qwen-local-27b"},
                budget={"duration": 900, "output_tokens": 20000, "graceful": False},
                params={"task_id": task, "mode": "reference_guided", "run_id": run_id,
                        "proposal_format": "lines", "study_policy": policy, "rounds": 3,
                        "selection_policy": "gate_first",
                        "evaluator": [GPU_PYTHON, str(ROOT / "scripts/evaluate_candidate.py")]},
            ).run()
        except Exception as error:  # noqa: BLE001 -- preserve failed scheduled work in the denominator
            result = {"task_id": task, "mode": "reference_guided", "run_id": run_id,
                      "terminal": "run_error", "correct": False, "promoted": False,
                      "attempts": [], "error": str(error)[-4000:]}
        result.update(study_policy=policy, transport_policy="json-lines", task_split=task_split[task],
                      effective_config=config, effective_config_sha256=hashlib.sha256(json.dumps(config).encode()).hexdigest())
        evidence["runs"].append(result)
        target.write_text(json.dumps(evidence, indent=2) + "\n")
        print(json.dumps({"finished": index + 1, **{key: result[key] for key in
                         ("task_id", "study_policy", "terminal", "correct", "promoted")}}), flush=True)
    if any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest for name, digest in hashes.items()):
        raise RuntimeError("measured sources changed during the selector-validation study")
    evidence["status"] = "complete"
    target.write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
