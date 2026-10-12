# Fixed-model kernel-agent strategy study, frozen before generation

Question: does independent evaluator feedback improve a three-proposal kernel agent
relative to blind refinement, and does retaining the search-best correct candidate
protect against deterioration in later proposals?

Use the same cached 27B FP8 Qwen asset, no adapter or training, json-lines protocol,
reference-guided fresh sessions, temperature 0 / model seed 20261011, 4096-token cap
per request, three requests per run, 900s / 20000 output-token outer budget. Run both
policies on RMSNorm (development), row_sum and silu_mul (new task families reserved
before this campaign). One run per task/policy; shuffle the six runs with seed 20261011.
These are prospective new-task checks, not a large independently curated benchmark.

Policies share the first proposal prompt, task, reference excerpt, API cards, previous
proposed source and generic revision instruction. For proposals two/three only, feedback
adds the actual independent search evaluation and typed repair instruction. API/reference
cards remain frozen within both runs; no adaptive retrieval confound. Work directories
and round counters differ. Feedback also reveals search timing shapes; final shapes never
reach the model. Equal request caps do not imply equal token consumption or prompt length.

Both policies use exactly three proposal opportunities even if an earlier proposal is
promoted or repeated. Duplicate source reuses its search measurement and is explicitly
marked; it is not a new GPU-success trial. A harness/budget failure remains a failure in
its scheduled denominator. Both select the best search-correct geomean, with stable tie
handling. Never use final results to change that choice or to repair the model.

After generation/selection, run final checks on every unique search-correct source,
including first, last and selected candidates. A search-rejected source remains rejected;
we do not reclassify it by cherry-picking final shapes. Main results use the search-chosen
source's final result. First/last/best diagnostics use this same fixed artifact set and
are labelled post-generation selection audits, not independent trials. Preserve every
reply, duplicate, source SHA, candidate parent, usage, search and final measurement.

GPU4/5 generate; independent evaluator uses GPU6. Row-sum and SiLU-multiply references
are PyTorch specifications kept outside generated candidates. Their routed official
examples teach generic reduction/load/launch patterns, not complete solutions. Existing
four-task evaluator behavior and tolerance/gates remain unchanged. Archive all previous
measured source snapshots before edits, freeze new source hashes throughout this campaign.

Report observed counts, final geomean and worst-shape ratios, unique-source count, actual
usage and first/last/best selection. Do not infer statistically significant gains, stable
success, model-training benefit or SOTA from six development/prospective runs.
