"""Run real local-model sessions, retaining every candidate, failure and usage record."""
import argparse
import hashlib
import json
import os
import random
from datetime import UTC, datetime
from pathlib import Path

from hmz.sdk import Hmz

from kernel_agent.transport import POLICIES

ROOT = Path(__file__).resolve().parents[1]
GPU_PYTHON = "/data/home/chenyizhou/kda-humanize-portfolio/.venv/bin/python"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["vector_add", "softmax", "rmsnorm", "transpose"])
    parser.add_argument("--modes", nargs="+", choices=["kda_prompt", "evidence_loop", "api_guided", "reference_guided"],
                        default=["kda_prompt", "evidence_loop", "api_guided"])
    parser.add_argument("--config", default="data/opencode-local.json")
    parser.add_argument("--output-name", default="agent-pilot.json")
    parser.add_argument("--transport-policies", nargs="+", choices=POLICIES, default=["passthrough"])
    parser.add_argument("--model-id", default="kernel-qwen3-4b")
    parser.add_argument("--model-name")
    parser.add_argument("--model-revision")
    parser.add_argument("--model-asset-manifest")
    parser.add_argument("--generation-gpus", default="5")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = json.loads(config_path.read_text())
    if args.model_id not in config["provider"]["kernel-local"]["models"]:
        parser.error("model must be declared in the local-only provider config")
    manifest = json.loads((ROOT / args.model_asset_manifest).read_text()) if args.model_asset_manifest else None
    output_path = ROOT / "results" / args.output_name
    os.environ["OPENCODE_DISABLE_AUTOUPDATE"] = "true"
    os.environ["OPENCODE_DISABLE_DEFAULT_PLUGINS"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    os.environ["PYTHONPATH"] = str(ROOT / "src")
    fixture = json.loads((ROOT / "data/tasks.json").read_text())
    run_stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    sources = ["flows/kernel_loop.py", "src/kernel_agent/protocol.py", "src/kernel_agent/knowledge.py", "scripts/evaluate_candidate.py",
               "scripts/local_proxy.py", "scripts/run_pilot.py", "src/kernel_agent/transport.py",
               "src/kernel_agent/line_protocol.py", "data/tasks.json", "data/opencode-local.json"]
    if args.model_asset_manifest:
        sources += [args.model_asset_manifest, "scripts/start_cached_qwen27.sh", "scripts/inventory_model.py"]
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}
    hashes[args.config] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if "reference_guided" in args.modes:
        for name in ["src/kernel_agent/references.py", "references/triton-3.2/manifest.json"]:
            hashes[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    evidence = {"schema_version": 1, "status": "running", "started_utc": run_stamp,
                "metadata": {"model": args.model_name or config["provider"]["kernel-local"]["models"][args.model_id]["name"],
                             "model_id": args.model_id, "model_adapter": "none",
                             "model_revision": args.model_revision or (
                                 "cdbee75f17c01a7cc42f958dc650907174af0554" if args.model_id == "kernel-qwen3-4b" else None),
                             "model_asset": manifest, "generation_gpus": args.generation_gpus,
                             "backend": "Humanize Agent → OpenCode 1.18.32 → local vLLM",
                             "temperature": 0, "seed": 20261011, "thinking": False,
                             "max_tokens_per_request": 4096, "gpu": "RTX 4090, physical GPU 6",
                             "opencode_config": args.config,
                             "transport_policies": args.transport_policies,
                             "source_sha256": hashes,
                             "study_scope": "one run per task/mode/transport policy; development task families; descriptive small sample"},
                "runs": []}
    schedule = [(task, mode, policy) for task in args.tasks for mode in args.modes
                for policy in args.transport_policies]
    random.Random(20261011).shuffle(schedule)
    evidence["schedule"] = schedule
    for index, (task, mode, policy) in enumerate(schedule):
        effective_config = json.loads(config_path.read_text())
        effective_config["provider"]["kernel-local"]["options"]["baseURL"] = f"http://127.0.0.1:18116/{policy}/v1"
        os.environ["OPENCODE_CONFIG_CONTENT"] = json.dumps(effective_config)
        run_id = f"{run_stamp}-{task}-{mode}-{policy}"
        print(json.dumps({"starting": index + 1, "task": task, "mode": mode, "transport_policy": policy}), flush=True)
        try:
            result = Hmz(ROOT).run(str(ROOT / "flows/kernel_loop.py"),
                                   fixture[task]["contract"] + "\nVisible examples: " + json.dumps(fixture[task]["examples"]),
                                   agents={"builder": f"opencode/kernel-local/{args.model_id}"},
                                   budget={"duration": 900, "output_tokens": 20000, "graceful": False},
                                   params={"task_id": task, "mode": mode, "run_id": run_id,
                                           "proposal_format": "lines" if policy == "json-lines" else "string",
                                           "evaluator": [GPU_PYTHON, str(ROOT / "scripts/evaluate_candidate.py")]}).run()
        except Exception as error:  # noqa: BLE001 -- a failed run stays in the denominator
            result = {"task_id": task, "mode": mode, "run_id": run_id, "terminal": "run_error",
                      "correct": False, "promoted": False, "attempts": [], "error": str(error)[-4000:]}
        result["transport_policy"] = policy
        result["effective_config_sha256"] = hashlib.sha256(json.dumps(effective_config).encode()).hexdigest()
        result["effective_config"] = effective_config
        evidence["runs"].append(result)
        output_path.write_text(json.dumps(evidence, indent=2) + "\n")
        print(json.dumps({"finished": index + 1, **{k: result[k] for k in
                         ("task_id", "mode", "terminal", "correct", "promoted")}}), flush=True)
    if any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest for name, digest in hashes.items()):
        raise RuntimeError("agent or evaluator source changed during the pilot")
    evidence["status"] = "complete"
    output_path.write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
