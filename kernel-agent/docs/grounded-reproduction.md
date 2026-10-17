# Review source-grounded repair

The agent repair controller, not manually tuned kernels, is the subject. Read the
[prospective plan](grounded-study-plan.md). `indexed` is the previous indexed/fallback
policy; `grounded` changes only the prompt by adding a deterministic source-bound packet
and obligations. Candidate schema, admission, evaluator, selector and two-call repair
budget are shared. Source facts never automatically fix or reject a kernel.

The packet compiles Python without executing it, lists direct JIT signatures and launch
arguments, and traces bounded store dependencies from straight-line assignments. It
marks partial observations/omissions. It is not a full Triton type checker, lane simulator,
semantic proof or arbitrary-code sandbox. Source-derived names and expressions help a model
ground its repair, but do not establish the correctness of its proposed edits.

## Compact local review

```bash
python3 scripts/verify_grounded_summary.py results/grounded-summary.json
```

Open `reports/kernel-agent-report.html#grounded-study`. The embedded summary and bounded
diffs work offline. The companion JSON supplies the download link. Compact checks validate
self-hash/count consistency, not absent raw records or CUDA numerics.

## Server audit and new runs

All project scripts run through the consolidated launcher:

```bash
LAUNCH=/data/home/chenyizhou/kernel-agent-storage/bin/run.py
RUN=/data/home/chenyizhou/kernel-agent-storage/experiments/grounded-study-v1
python3 "$LAUNCH" summarize_grounded.py "$RUN"
python3 "$LAUNCH" export_grounded.py --directory "$RUN" --finalize
python3 "$LAUNCH" verify_grounded_summary.py results/grounded-summary.json
python3 "$LAUNCH" build_report.py
```

Finalize only after the supervisor has exited successfully. The exporter reconstructs
exact prompts, response/schema bindings, model-authored patch application, retained parents,
source facts and candidate SHAs, all selections before final-data access, physical GPU4
proof/serialized windows, admitted service identities and durable budgets. A finalized
raw manifest is immutable. Full replies, kernels, measurements, parent attempts, source
freezes and full reconstruction stay remote, outside Git and Mac.

For a new study use an unused directory and a separately frozen plan. Never restart a
completed experiment or replay an unknown call to replace an outcome. Do not change
generation source until the active study has ended. Older frozen results retain their
literal paths and are not edited during this iteration.

```bash
python3 "$LAUNCH" check_project.py
python3 "$LAUNCH" check_grounded.py --output /data/home/chenyizhou/kernel-agent-storage/checks/NEW-context.json
python3 "$LAUNCH" run_grounded_study.py --directory /data/home/chenyizhou/kernel-agent-storage/experiments/NEW-ID --background
```

The current CPU suite has 212 checks and lint passes. The added integration fixture drives
all 28 outcomes/64 fake calls, forces first-patch rejection and same-slot fallback, checks
final measurement reuse, reverses filesystem order and forges a source-fact prompt while
re-signing its request receipt. The auditor must reject the forged policy. These are
mechanism tests, not model successes. Context preflight tokenized 48 historical sources
through both policies on all three services: 288 tokenizations, maximum 5843 input tokens,
zero completions. GPU preflight verified physical UUID4 on a known bad source; its expected
numeric failure is not a new agent outcome.

## Interpretation

The eight fresh starts are primary; six selected prior-source repairs are diagnostic and
separate. All families have development history. One fresh seed and three regression
clusters are small exploratory samples. Report conditional rescues, costs, protocol and
semantic failures alongside final correctness. A more informative prompt can still hurt.
Do not change defaults or claim generalization merely because the checks exist.

## Source-fact boundaries

The packet keeps at most two direct JIT functions, two launch observations and two store
slices. Long expressions are marked with an ellipsis; omitted observations are counted.
If necessary, store slices are omitted to keep the serialized packet near3300 characters.
Only direct language aliases and straight-line assignments are traced. Dynamic argument
expansion is explicitly partial. Unresolved names are observations from a limited scope,
not automatic errors or candidate rejection. These limits keep real search evidence and
the complete parent source mandatory inside the common8192-token context.

## Initial-run limitation and follow-up

The initial run dispatched56/64 calls: two policy-caused context refusals incorrectly
cancelled counterpart opportunities. Keep its0/6 vs3/6 regression records but do not
interpret that contrast as clean efficacy. The fresh-start7/8 vs8/8 comparison is separate.
Initial source preflight had212 checks; the current219-check suite adds actual-context
budget adaptation and forced policy isolation. The portable fake-parent fixture does not
require private parent records. See [the separate follow-up](grounded-budget-reproduction.md).
