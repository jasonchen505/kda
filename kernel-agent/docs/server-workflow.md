# Canonical server workflow and lean local review

Current code lives in `/data/home/chenyizhou/kda-kernel-agent-harness/kernel-agent`.
Supporting runtimes, upstream clones, services and archives live in the sibling
`/data/home/chenyizhou/kernel-agent-storage`. Old home-level workspace paths are retired.
Historical frozen records retain their literal paths; do not rewrite them for migration.

Run project scripts through the external consolidated launcher:

```bash
python3 /data/home/chenyizhou/kernel-agent-storage/bin/run.py check_project.py
python3 /data/home/chenyizhou/kernel-agent-storage/bin/run.py check_project.py --inventory
python3 /data/home/chenyizhou/kernel-agent-storage/bin/run.py verify_evidence.py
```

The launcher uses `runtimes/controller`, sets imports to the canonical source and supplies
`runtimes/gpu/bin/python` to scripts with a GPU-Python option. Use its `--runtime gpu`
mode for a standalone evaluator. Do not resolve an environment's Python entry point or
reuse a default pointing to a retired path. Do not relocate/restart existing Qwen processes
or edit foreign configurations as part of this workflow.

## New experiments

New raw data stay OUTSIDE Git and outside Mac, under
`kernel-agent-storage/experiments/<new-study-id>/`. A completed ledger is immutable.
For the prospective indexed-patch study, after tests/context/GPU preflight:

```bash
python3 /data/home/chenyizhou/kernel-agent-storage/bin/run.py run_span_study.py \
  --directory /data/home/chenyizhou/kernel-agent-storage/experiments/NEW-ID --background
```

A new study must use an unused directory. The launcher child acquires a single-supervisor
lock. Source/task/plan hashes freeze before model calls; source drift is checked before
and after each paired block. Budgets persist before dispatch; unknown requests are not
replayed. Idle admission is observed, not a reservation of other users' services.

The scheduler dynamically discovers the three explicitly authorized cached Qwen services,
chooses unused idle replicas before latency EWMA, and keeps each paired block on one
process identity. Project GPU4 evaluation is cooperatively serialized and physically
UUID-bound. Sharing is allowed, so kernel timing ratios are conditional observations.

## Evidence and summaries

Audit from the full remote ledger and write a compact result into the canonical project:

```bash
python3 /data/home/chenyizhou/kernel-agent-storage/bin/run.py export_span_report.py \
  --directory /data/home/chenyizhou/kernel-agent-storage/experiments/span-study-v1
# Once the supervisor has exited successfully and all66 outcomes are terminal:
python3 /data/home/chenyizhou/kernel-agent-storage/bin/run.py export_span_report.py \
  --directory /data/home/chenyizhou/kernel-agent-storage/experiments/span-study-v1 --finalize
python3 /data/home/chenyizhou/kernel-agent-storage/bin/run.py build_report.py
```

Full raw requests/replies, kernels, evaluator outputs, exact source freezes, checkpoint and
`analysis-full.json` stay in the experiment directory. Its immutable `evidence-manifest.json`
binds the complete records. The compact summary records that manifest's SHA/root path,
source-plan identity and bounded derived diffs; it does not contain the full reply/kernel
corpus. Manifest roots establish identity, not public availability of raw files.

Only sync current code, documentation, compact `span-summary.json`/checks/audits and latest
`reports/kernel-agent-report.html` to Mac. Do not expand historical snapshots, disable
sparse checkout, run `restore_evidence.py --full`, or mirror `results/` recursively as a
routine development step. Original historical data are already available remotely.

On the lean Mac, use the summary-only checker:

```bash
python3 scripts/verify_span_summary.py results/span-summary.json
```

Its scope is self-hash and count consistency. Full raw reconstruction and historical
evidence verification run on the server through `bin/run.py`; the lean Mac is not a full
evidence checkout. New raw evidence is private to the owner's server, not included in
this branch. Public review can inspect code/protocol/summary/diffs; independent raw
reconstruction requires access to that server. Do not describe compact verification as
re-executing models, validating CUDA numerics or auditing unavailable raw files.
