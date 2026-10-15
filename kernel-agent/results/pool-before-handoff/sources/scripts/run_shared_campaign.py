"""A prospective paired study with observed GPU sharing and immediate feedback."""
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
    parser.add_argument("--port", type=int, default=8005)
    parser.add_argument("--gpu-index", type=int, default=4)
    parser.add_argument("--gpu-python", default="/data/home/chenyizhou/kda-humanize-portfolio/.venv/bin/python")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    plan = run_campaign.make_plan(run_campaign.runtime_identity(args), run_campaign.source_hashes(),
                                  seeds=[20261025], resource_wait_seconds=120)
    plan.update(study_kind="shared_feedback", scope="six development families, one seed, observed sharing; descriptive paired replication")
    plan["generation"].update(measurement_condition="shared_observed", evaluation_guard="shared_observed")
    with Campaign(args.checkpoint, plan) as campaign:
        run_campaign.freeze_sources(args.checkpoint, plan["source_sha256"])
        if not args.prepare:
            campaign.run(lambda job: run_campaign.execute(plan, job, args.checkpoint.parent / "wire"),
                         ready=lambda job: run_campaign.wait_until_ready(plan, job, args.checkpoint.parent / "resource-wait.json"))
        summary = campaign.summary()
        atomic_json(args.checkpoint.parent / "summary.json", summary)
        print(json.dumps(summary, indent=2))
    return 2 if not args.prepare and summary["pending"] else 0


if __name__ == "__main__":
    sys.exit(main())
