# Online feedback, durable finalization: engineering pilot

Written before new model requests, 2026-10-15 (America/Los_Angeles).

The previous 12-job study keeps its 11 committed results and one 900-second budget
failure. This pilot never replaces that denominator. It tests a controller change:
retain search timing feedback, commit generation before final-data access, import
the exact search measurements into a durable queue, and run only the selected
source's independent final check outside the Humanize generation budget.

- Two development tasks: transpose (previous budget failure family) and softmax.
- Feedback arm only, seed 20261026, three proposal opportunities each, temperature
  0.2, reference-guided line protocol, search-only gate-first selection.
- Per job: 900 seconds / 20,000 SDK output tokens; 1,800 / 40,000 total allocation.
- Base cached Qwen FP8, new owned service on GPUs 5/6, port 18118; evaluation GPU4
  permits sharing with telemetry and at least 1,024 MiB free.
- Model directory alias says Qwen3.8; config says qwen3_5. Full weight hashes and
  upstream model revision remain unverified. A new service identity and frozen
  source plan prevent pooling this pilot with previous service runs.
- Persist search selection before final access. No final feedback, alternative
  candidate audits or reselection. Reopening complete queues performs no work.
- Inject a real process kill after the first final queue marks its evaluation
  running, before its evaluator child starts. Resume pure final evaluation; record
  the interrupted evaluation. Never repeat generation or committed search.

Success requires two committed search results, an independently finalized queue
for each eligible selected source, matching source hashes, and zero extra model
requests/search evaluator calls during recovery. Numerical failures, budget
failures and missing results remain visible. Correctness and promotion are not
success requirements for this engineering test. It cannot establish feedback
benefit, generalization, speed superiority over the prior runtime, or recovery
from arbitrary in-flight model calls.

Commands (use a fresh directory; keep its sources unchanged):

```bash
python scripts/run_phased.py --pid "$SERVICE_PID" --directory runs/phased-v1 --prepare
python scripts/run_phased.py --pid "$SERVICE_PID" --directory runs/phased-v1
python scripts/finish_deferred.py --generation runs/phased-v1/generation.json --directory runs/phased-final-v1
```

Finalization verifies source/runtime identity and does not depend on a live model
service. Resource failures leave the pure evaluation queued for a later retry.
