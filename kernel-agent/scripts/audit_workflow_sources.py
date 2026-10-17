"""Archive pinned workflow sources remotely; record a read-only capability inventory."""
import argparse
import base64
import hashlib
import json
import shutil
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

KDA = "ef6ce617693ef0782b3ecb9f37e39bbf10226a90"
CONTEST = "d61573ffb2008b6e68fb32fe8e17703e4b2448c9"
ROOT = Path(__file__).resolve().parents[1]
STORE = Path("/data/home/chenyizhou/kernel-agent-storage")


def command(argv):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    return {"exit_code": result.returncode, "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--gpu-python", required=True)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if not directory.is_relative_to(STORE / "archives/source-audits"):
        parser.error("Source archives must remain in remote storage, outside Git/Mac")
    directory.mkdir(parents=True, exist_ok=False)
    manifest = []
    groups = [
        ("kda", "NVlabs/kda", KDA,
         ["README.md", "docs/agent-flow.md", "prompts/basic-flow.md", "CLAUDE.md"]),
        ("contest", "mit-han-lab/mlsys2026-flashinfer-contest", CONTEST,
         ["README.md", "docs/reproduction.md", "prompts/README.md",
          "docs/HAN_Lab_Kernel_Mafia_Technical_Report.pdf"] +
         [f"prompts/{task}/phase{phase}.md" for task in
          ["gdn-decode", "gdn-prefill", "dsa-topk-indexer", "dsa-sparse-attention", "moe-fp8"]
          for phase in [1, 2, 3]])]
    for group, repo, commit, names in groups:
        for name in names:
            url = f"https://api.github.com/repos/{repo}/contents/{name}?ref={commit}"
            document = json.load(urllib.request.urlopen(url, timeout=30))
            if document.get("encoding") != "base64":
                document = json.load(urllib.request.urlopen(document["git_url"], timeout=30))
            if document.get("encoding") != "base64":
                raise ValueError("GitHub did not return a base64 source blob")
            data = base64.b64decode(document["content"], validate=False)
            target = directory / group / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            manifest.append({"path": str(target.relative_to(directory)), "bytes": len(data),
                             "sha256": hashlib.sha256(data).hexdigest(),
                             "url": f"https://github.com/{repo}/blob/{commit}/{name}"})
    pdf = directory / "contest/docs/HAN_Lab_Kernel_Mafia_Technical_Report.pdf"
    text = pdf.with_suffix(".txt")
    extraction = command(["pdftotext", "-layout", str(pdf), str(text)])
    if extraction["exit_code"]:
        raise RuntimeError(extraction)
    packages = command([args.gpu_python, "-c", """import json, importlib.metadata as m
versions = {}
for name in ['torch', 'triton', 'flashinfer-python', 'flashinfer-bench']:
    try:
        versions[name] = m.version(name)
    except m.PackageNotFoundError:
        versions[name] = None
print(json.dumps(versions))
"""])
    code = {}
    for name in ["flows/kernel_loop.py", "src/kernel_agent/humanize_api.py",
                 "src/kernel_agent/repair.py", "src/kernel_agent/grounded_budget.py",
                 "scripts/run_grounded_study.py", "scripts/run_grounded_budget_study.py",
                 "src/kernel_agent/model_pool.py", "src/kernel_agent/protocol.py"]:
        code[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    result = {"schema_version": 1, "kind": "workflow_source_and_platform_audit",
              "audited_utc": datetime.now(timezone.utc).isoformat(),
              "kda_commit": KDA, "contest_commit": CONTEST,
              "source_archive": str(directory), "sources": manifest,
              "pdf_text_sha256": hashlib.sha256(text.read_bytes()).hexdigest(),
              "current_code_sha256": code,
              "hardware": command(["nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv,noheader"]),
              "gpu_runtime_packages": packages, "ncu_path": shutil.which("ncu"),
              "ncu_version": command(["ncu", "--version"]) if shutil.which("ncu") else None,
              "model_calls": 0, "gpu_evaluations": 0,
              "decision": "Do not launch a new strategy study during portfolio closeout. "
              "The contest's B200/FlashInfer stack differs from this RTX4090/Triton fixture harness. "
              "A local staged-flow trial would be a port, not a full contest reproduction or matched KDA baseline.",
              "scope": "Source identity and capability inventory only; not a reproduction or efficacy experiment. "
              "ncu availability does not establish profiling permission or successful collection."}
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    (directory / "audit.json").write_text(encoded)
    target = ROOT / "results/workflow-audit.json"
    target.write_text(encoded)
    print(json.dumps({"summary": str(target), "sources": len(manifest), "archive": str(directory),
                      "runtime": packages, "model_calls": 0, "gpu_evaluations": 0}))


if __name__ == "__main__":
    main()
