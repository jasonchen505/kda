# Reproduce or review indexed kernel repair

Read the [prospective protocol](span-study-plan.md) before interpreting the results. The
comparison has three arms, two repair calls per arm,22 blocks/66 outcomes and at most140
physical calls. Regression and fresh generation on existing recent families remain separate.
The new interface changes numbered spans, exact LF decoding and a second-slot fallback
together; it is not a component-isolation ablation or a new external task benchmark.

## Lightweight review

```bash
python3 scripts/verify_span_summary.py results/span-summary.json
```

Open `reports/kernel-agent-report.html#span-study`. The HTML embeds the compact summary and
bounded model-candidate diff excerpts, so charts/case switching work offline. It does not
contain this round's complete model replies or kernel corpus. The checker validates the
summary's self-hash and row/cell consistency, not unavailable raw records or GPU numerics.

## Full server verification

```bash
LAUNCH=/data/home/chenyizhou/kernel-agent-storage/bin/run.py
RUN=/data/home/chenyizhou/kernel-agent-storage/experiments/span-study-v1
python3 "$LAUNCH" summarize_span.py "$RUN"     # full output stays remote
python3 "$LAUNCH" export_span_report.py --directory "$RUN" --finalize
python3 "$LAUNCH" verify_span_summary.py results/span-summary.json
python3 "$LAUNCH" verify_evidence.py
```

Raw reconstruction binds each physical request/response, assigned schema and entire
prompt, seed/budget/process identity, shared source, exact old/new patch application,
compiled source SHA, API preflight, GPU UUID/serialized intervals, all three pre-final
selections and final accept/veto. It makes no new model/GPU calls. `analysis-full.json`
and `evidence-manifest.json` remain remote; the compact result records their identity.
Never restart the completed supervisor or alter its checkpoint to obtain new outcomes.

## Mechanism and new execution

```bash
python3 "$LAUNCH" check_project.py --focused
python3 "$LAUNCH" check_project.py
python3 "$LAUNCH" check_project.py --inventory
python3 "$LAUNCH" check_project.py --preflight-span \
  --output /data/home/chenyizhou/kernel-agent-storage/checks/NEW-context.json
```

Unit fixtures test stale SHA, repeated source lines, atomic original offsets, bounds,
overlap, whole no-op rejection, unchanged sub-edits, exact embedded LF and fallback mode.
The full native-client integration fixture uses fake HTTP/GPU verdicts, forces first-patch
rejections, requires the same second slot to recover, exercises all66 outcomes/140 fake
calls, reverses filesystem iteration and tampers with a response while re-signing its
receipt. These are controller tests, not real agent successes. Real GPU evidence is separate.

For a new model run use a NEW experiment directory through `bin/run.py`, as documented in
[the canonical server workflow](server-workflow.md). Source hashes freeze before inference;
canonical generation files must remain unchanged until all blocks finish. A new prospective
plan is required for policy changes. The original source bank, task definitions, trusted
Torch oracles and final-shape split are unchanged in this iteration. All model-authored
candidate edits remain unmodified by humans; only the documented decoder joins strings.

Future work should investigate whether reduced refusals translate to correct semantics
on additional families. Neither successful transaction application nor one rescue trajectory
proves that the repair architecture improves final correctness or generalizes.

## Reporting-only correction

The frozen `export_span.py` assumed that a missing parent SHA identified an admitted source.
A real protocol-failed initial exposed that assumption. The additive
`export_span_report.py` requires both code and a source SHA before deriving a diff;
no-parent diffs stay empty. Generation, search/final evaluation, selection, raw outcomes
and the original98 frozen files are unchanged. Its additional formatting fixture raises
the CPU count from204 to205. The corrected analysis entry point and receipt are retained
in the remote run's `analysis-tools/`, separately from the original executed-source freeze.
