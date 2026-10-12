# Fixed-candidate SiLU recheck

Question: does the recorded final performance contrast between the original selected SiLU
candidate and the discarded gate-eligible candidate repeat under new evaluator processes?

This is not a model-generation run. It accepts existing immutable candidate files, verifies
their SHA against the evaluator response, then invokes the final evaluator once per fresh
process. It does not feed results to the model, reselect a candidate, or replace the original
six-run strategy outcome.

The recorded run uses GPU6, three final evaluations per source, and these two paths:

```bash
PYTHONPATH=src /data/home/chenyizhou/humanize/.venv/bin/python \
  scripts/recheck_candidates.py silu_mul \
  runs/20261011T193132Z-silu_mul-strategy-feedback/candidate-1.py \
  runs/20261011T193132Z-silu_mul-strategy-feedback/candidate-2.py \
  --repeat 3 --cuda-visible-devices 6 \
  --output results/silu-selector-recheck.json
```

Results: candidate 1 is correct 3/3 and promoted 0/3, with 0.926–0.932x geometric mean
and 0.802–0.816x worst-shape ratio. Candidate 2 is correct 3/3 and promoted 3/3, with
1.289–1.299x geometric mean and 0.996–1.005x worst-shape ratio. Timing variability remains;
this narrow recheck does not measure new model samples, feedback policy, task generalization,
or the success rate of the patched selector.
