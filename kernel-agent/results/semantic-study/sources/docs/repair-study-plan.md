# Contract planning and counterexample repair: prospective factorial study

Frozen before corrected v2 generation, 2026-10-15, America/Los_Angeles.

The initial v1 native-backend pre-run lacked CUDA device binding. It observed GPU4
but actually used default GPU0, as confirmed by physical UUID. It was stopped, retained
in `results/repair-invalid-device-v1/`, and excluded from efficacy/performance evidence.
The policies are unchanged. V2 uses new seeds and a new ledger; v1 allocations and
uncertain requests are retained, not reset or automatically replayed. The two new task
families were already seen in that invalid pre-run, after policy freeze; they are not
claimed as blind held-out tasks in v2. The 160-call cap below is for v2 only.

The research question is whether task-independent contract planning and structured
counterexample repair improve the kernel-writing agent under the same model-call and
output ceilings. Scheduler improvements are shared by all arms, not a treatment.

Eight families × two new seeds (20261101/20261102) form 16 blocks. Six families have
development history; row_mean and layer_norm are prospective transfer tasks, defined
before this study and not used to tune the treatment. They are small task extensions,
not an independent external benchmark or proof of broad held-out generalization.

Each block remains on one model process and has two independently generated first
proposals: existing KDA/reference prompting versus an explicit contract checklist.
Each first proposal and its original search verdict are then shared unchanged between
two repair arms: ordinary error text versus structured counterexample/memory repair.
Each repair arm gets two further opportunities. This is 32 actual initial calls,
128 actual repair opportunities, and 64 scheduled end-to-end arm outcomes. Initial
generation is executed once per generation condition, not once per repair arm.

Generation order and per-round repair order are seeded/randomized. Both repair arms
start with byte-identical source and evaluation. They receive identical task contracts,
references, backend, temperature 0.2, no thinking, 8192 context and at most 2048 output
tokens per call. New generation asks for output/index/cast/mask reasoning in the existing
draft/plan fields, not an extra paid call. New repair supplies the actual search failure,
a deterministic tiny numerical witness, rejected-source memory and an optional correct
source anchor. It never includes final shapes or results. Neither controller edits a
generated kernel. These are bundled prompt/context policies; the study does not isolate
each instruction or prove latent reasoning.

The native loopback chat backend removes OpenCode auxiliary title calls for every arm.
Consequently old OpenCode experiments are not a causal backend comparison. Requests,
responses, usage and admission receipts are preserved. Exact tokenization may remove
whole optional references/memory/anchor blocks, recorded by name; contract, previous
source and active feedback remain intact. Oversized mandatory context is rejected.

The scheduler discovers allowed Qwen FP8 roots from public vLLM process arguments and
HTTP metadata, maps GPU workers using public parent chains/NVML, and admits idle services.
It first uses each idle replica, then uses measured latency EWMA for ready work. One
block per process is active; faster services can claim more pending blocks. API PID/start
ticks and declaration bind a block; changed/unavailable services never cause in-flight
blocks to migrate or uncertain calls to replay. Newly ready replicas may take untouched
blocks. This idle observation does not reserve another user's server. No foreign model,
environment, private files, weights or logs are modified/read. Root names do not prove
identical weights; vLLM configurations differ, hence comparisons stay within process.

Every admitted block reserves ten calls / 20480 output tokens and has a 1800-second
model-phase admission deadline; finalization is separate. Total allocations are 160
calls / 327680 tokens, not measured usage. No automatic retry after dispatch uncertainty.
Failed/interrupted blocks remain in the planned denominators. An interrupted active
block is marked unknown on supervisor restart; task-internal model calls are not replayed.

GPU4 permits sharing and serializes this project's evaluators. The wrapper owns
CUDA_VISIBLE_DEVICES, binding it to NVML's physical GPU4 UUID before spawning Python.
Each child reports its CUDA UUID, visible-device string, count=1 and logical index=0;
the wrapper rejects a missing or mismatched proof before accepting any verdict. Search uses the same
correctness/performance thresholds for every arm. Search-only selection uses gate-first
ranking and is committed for all four arms before any final result is observed. Identical
sources within a block share original search and final measurements, explicitly marked;
these reuse references are not new GPU executions. Final data only accept/veto the
search-selected source. The tiny witness uses search-only synthetic inputs and contains
no suggested kernel or manually corrected source.

Primary generation metric: first-proposal search correctness (16 per generation policy).
Primary end-to-end metric: independent final correctness (16 per factorial cell), failures
retained. Report the generation and repair contrasts within block, average seeds per task,
then use eight task clusters for descriptive bootstrap intervals. They are exploratory,
not a powered confirmatory significance test. Conditional repair rescue compares the two
repair arms only on their shared failed initial candidates, with explicit planned and
fully observed denominators. Infrastructure-stopped cases are shown separately; partial
observations cannot establish repair superiority. Performance promotion, protocol rate,
duplicates, output tokens, model proxy time and independent GPU counts are secondary.

Keep every negative outcome. Do not tune this treatment after seeing these results or
replace failed trials with retries. A null/negative result is a completed experiment.
