"""Commit online-feedback search before a separately resumable final evaluation."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from kernel_agent.campaign import Campaign, atomic_json  # noqa: E402
from scripts import run_campaign  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--port", type=int, default=18118)
    parser.add_argument("--gpu-index", type=int, default=4)
    parser.add_argument("--gpu-python", default="/data/home/chenyizhou/kda-humanize-portfolio/.venv/bin/python")
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    directory = args.directory.absolute()
    plan = run_campaign.make_plan(run_campaign.runtime_identity(args), run_campaign.source_hashes(),
                                  tasks=["transpose", "softmax"], seeds=[20261026], resource_wait_seconds=120)
    plan["jobs"] = [job for job in plan["jobs"] if job["study_policy"] == "feedback"]
    plan["campaign_budget"] = {key: value * len(plan["jobs"]) for key, value in plan["per_job_budget"].items()}
    plan.update(study_kind="online_search_durable_final_pilot",
                scope="two development tasks; phase durability and service recovery, not feedback efficacy")
    plan["generation"].update(evaluation_mode="search_only", measurement_condition="shared_observed",
                              evaluation_guard="shared_observed")
    with Campaign(directory / "generation.json", plan) as campaign:
        run_campaign.freeze_sources(directory / "generation.json", plan["source_sha256"])
        if not args.prepare:
            campaign.run(lambda job: run_campaign.execute(plan, job, directory / "wire"),
                         ready=lambda job: run_campaign.wait_until_ready(plan, job, directory / "resource-wait.json"))
        summary = campaign.summary()
        # Generation completion is not a final verdict. The raw flow booleans remain
        # false until the independent final queue; public phase summaries use null.
        summary.update(phase="online_search", final_correct=None, final_promoted=None)
        atomic_json(directory / "search-summary.json", summary)
        print(json.dumps(summary, indent=2))
    return 2 if not args.prepare and summary["pending"] else 0


if __name__ == "__main__":
    sys.exit(main())
