# Frozen next experiment: replicated feedback versus blind refinement

The research object is the kernel-generating agent controller. Six existing development
task families are instruments for evaluating it; no new kernel is hand-written here.
This is a new experiment design, not a relabeling of the historical three-task results.

## Design, fixed before any new proposals

- Six task families: vector addition, row softmax, RMSNorm, transpose, row sum, fused SiLU
  multiply. All have been exposed during development: none is a new held-out task.
- Three stochastic seeds per task (20261021/22/23), each with blind and evaluator-feedback
  arms: **36 scheduled jobs, 108 proposal opportunities**. Temperature 0.2 replaces greedy
  decoding for this study; changing the seed at temperature zero alone is not a convincing
  independent replication. GPU generation may remain nondeterministic despite fixed seeds.
- Both arms use the same cached base Qwen asset, frozen references, line-array protocol,
  gate-first selector, fresh bounded sessions, previous source, three opportunities,
  4096 output tokens/request and 900 seconds / 20000 SDK output tokens/job. Only the actual
  search verdict in rounds two and three differs between policies. System workdir context
  varies; first candidates are not guaranteed identical.
- Pairs are stratified by task/seed; arm order and global schedule are deterministically
  randomized. One sequential worker keeps a constant runtime rather than adding distributed
  scheduling as another experimental variable. The local service must be initially idle,
  and the evaluator uses a separate idle GPU. Shared-service interference during a job remains
  a limitation: record it if observed and do not selectively rerun slow or failed jobs.
- The immutable plan pins controller/reference hashes, service PID/start identity, actual
  base-model root, weight metadata/shard inventory, Humanize public API source hashes,
  OpenCode/vLLM versions and evaluator versions. A source/runtime change requires a new
  checkpoint. No paid API, weight download, adapter or training is requested.

## Outcomes and analysis

Primary outcomes are final correctness and final promotion (geomean >=1.05 AND worst
>=0.90 versus torch.compile), per scheduled job. Final data cannot alter search selection
or be fed back to the proposer. Report counts by task/arm first; also report paired differences
per seed, source diversity, duplicates, reported usage, elapsed time, and failure categories.
Paired seeds do not imply matched initial candidates or a strict deterministic causal control.

Tasks, not the 108 dependent proposals, are the generalization unit. Pooling all jobs
cannot create 36 independent task discoveries. Three seeds and six development families
give a small descriptive replication; report task-clustered uncertainty only with this limit
made explicit. Do not combine new scores with earlier model/runtime campaigns. No numerical
results or improvement claims are registered before execution.

Interrupted jobs with unknown outcomes and failed jobs remain in the scheduled denominator
and count as zero successes in the conservative primary summary. Show these categories
separately from observed GPU numerical failures. A completed task's recorded verdict is reused;
an in-flight external request is never auto-replayed. Missing provider usage is unknown, not zero.

## Recovery contract and resource handling

`scripts/run_campaign.py` implements durable scheduling **at task boundaries**, around the
existing Humanize flow. Atomic fsync/rename checkpoints, a POSIX single-owner lock, immutable
plans and SHA-linked events protect completed results and budget allocations. Checksums detect
accidental edits, not malicious tampering. This is not native Humanize FlowState restoration:
candidate/session state inside an interrupted task is not resumed. Orphaned external requests
may still finish; there is no exactly-once API execution guarantee.

Before each started job, its full configured budget is reserved durably. Reservations survive
interruptions. They are conservative scheduler allocations, **not measured provider token totals
or an absolute billing guarantee**; OpenCode background calls and interrupted requests may have
unknown usage. The SDK enforces its per-job budget independently. Busy resources stop before
allocation or a model call, leaving pending tasks and a visible `awaiting_resources` status.
Device checks are readiness checks, not a GPU lock that excludes other users.

## Commands on the server

```bash
cd /data/home/chenyizhou/kernel-agent
export PATH=/data/home/chenyizhou/.local/bin:$PATH
# Supply the PID of your existing loopback base service (example: 3813970 on port 8005).
.venv/bin/python scripts/run_campaign.py --prepare --pid 3813970
# Once both service and GPU6 are idle, start at most two jobs; repeat to resume.
.venv/bin/python scripts/run_campaign.py --run --pid 3813970 --max-new-runs 2
# Run all remaining pending jobs with the exact same plan/source/runtime:
.venv/bin/python scripts/run_campaign.py --run --pid 3813970
```

The wrapper never starts or terminates the model service. A pending/blocked run exits 2,
and checkpoint/summary files show the reason. Changing the service, source or study design
requires a new `--checkpoint` directory. No scheduled automation is created.
