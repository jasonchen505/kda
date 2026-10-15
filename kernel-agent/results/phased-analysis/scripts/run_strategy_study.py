"""Equal-proposal-budget study of evaluator feedback and candidate selection."""
import hashlib
import json
import os
import random
from datetime import UTC, datetime
from pathlib import Path

from hmz.sdk import Hmz

ROOT = Path(__file__).resolve().parents[1]
GPU_PYTHON = "/data/home/chenyizhou/kda-humanize-portfolio/.venv/bin/python"


def main():
    os.environ["OPENCODE_DISABLE_AUTOUPDATE"] = "true"
    os.environ["OPENCODE_DISABLE_DEFAULT_PLUGINS"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    os.environ["PYTHONPATH"] = str(ROOT / "src")
    config = json.loads((ROOT / "data/opencode-qwen27.json").read_text())
    config["provider"]["kernel-local"]["options"]["baseURL"] = "http://127.0.0.1:18116/json-lines/v1"
    os.environ["OPENCODE_CONFIG_CONTENT"] = json.dumps(config)
    fixture = json.loads((ROOT / "data/tasks.json").read_text())
    run_stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    sources = ["flows/kernel_loop.py", "src/kernel_agent/protocol.py", "src/kernel_agent/knowledge.py",
               "scripts/evaluate_candidate.py", "scripts/local_proxy.py", "scripts/run_strategy_study.py",
               "src/kernel_agent/transport.py", "src/kernel_agent/line_protocol.py", "data/tasks.json",
               "data/opencode-qwen27.json", "results/qwen27-asset-manifest.json",
               "scripts/start_cached_qwen27.sh", "scripts/inventory_model.py",
               "src/kernel_agent/references.py", "references/triton-3.2/manifest.json",
               "docs/strategy-study-plan.md"]
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}
    schedule = [(task, policy) for task in ("rmsnorm", "row_sum", "silu_mul") for policy in ("blind", "feedback")]
    random.Random(20261011).shuffle(schedule)
    evidence = {"schema_version": 1, "status": "running", "started_utc": run_stamp,
                "metadata": {"study_kind": "strategy_ablation", "model": "Cached Qwen 27B FP8 (config: qwen3_5)",
                             "model_id": "kernel-qwen-local-27b", "model_adapter": "none", "model_revision": None,
                             "model_asset": json.loads((ROOT / "results/qwen27-asset-manifest.json").read_text()),
                             "generation_gpus": "4,5", "gpu": "RTX 4090, physical GPU 6",
                             "backend": "Humanize Agent → OpenCode 1.18.32 → local vLLM",
                             "temperature": 0, "seed": 20261011, "thinking": False,
                             "max_tokens_per_request": 4096, "rounds": 3,
                             "selection_policy": "geomean",
                             "transport_policies": ["json-lines"], "study_policies": ["blind", "feedback"],
                             "task_split": {"rmsnorm": "development", "row_sum": "prospective", "silu_mul": "prospective"},
                             "source_sha256": hashes, "opencode_config": "data/opencode-qwen27.json",
                             "study_scope": "six scheduled runs, one per task/policy; three proposal opportunities; fixed references; selection audit is not independent trials"},
                "schedule": schedule, "runs": []}
    target = ROOT / "results/agent-strategy-study.json"
    if target.exists():
        raise RuntimeError("study result already exists; archive it explicitly before a new campaign")
    target.write_text(json.dumps(evidence, indent=2) + "\n")
    for index, (task, policy) in enumerate(schedule):
        run_id = f"{run_stamp}-{task}-strategy-{policy}"
        print(json.dumps({"starting": index + 1, "task": task, "study_policy": policy}), flush=True)
        try:
            result = Hmz(ROOT).run(str(ROOT / "flows/kernel_loop.py"),
                                   fixture[task]["contract"] + "\nVisible examples: " + json.dumps(fixture[task]["examples"]),
                                   agents={"builder": "opencode/kernel-local/kernel-qwen-local-27b"},
                                   budget={"duration": 900, "output_tokens": 20000, "graceful": False},
                                   params={"task_id": task, "mode": "reference_guided", "run_id": run_id,
                                           "proposal_format": "lines", "study_policy": policy, "rounds": 3,
                                           "selection_policy": "geomean",
                                           "evaluator": [GPU_PYTHON, str(ROOT / "scripts/evaluate_candidate.py")]}).run()
        except Exception as error:  # noqa: BLE001 -- keep budget/harness failures in the scheduled denominator
            result = {"task_id": task, "mode": "reference_guided", "run_id": run_id,
                      "terminal": "run_error", "correct": False, "promoted": False,
                      "attempts": [], "error": str(error)[-4000:]}
        result.update(study_policy=policy, transport_policy="json-lines", task_split=evidence["metadata"]["task_split"][task],
                      effective_config=config, effective_config_sha256=hashlib.sha256(json.dumps(config).encode()).hexdigest())
        evidence["runs"].append(result)
        target.write_text(json.dumps(evidence, indent=2) + "\n")
        print(json.dumps({"finished": index + 1, **{k: result[k] for k in
                         ("task_id", "study_policy", "terminal", "correct", "promoted")}}), flush=True)
    if any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest for name, digest in hashes.items()):
        raise RuntimeError("measured sources changed during the strategy study")
    evidence["status"] = "complete"
    target.write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
