# API-aware agent pilot

The first measured campaign produced zero correct final candidates in eight real sessions.
Actual failures were dominated by Triton API misuse and malformed Python source, rather
than numerical performance. Preserve that full campaign in results/agent-pilot-unguided.json.

Before the second campaign, add versioned reference cards covering host launch, memory,
reductions and source formatting. Their selector uses task family and failure information.
They are concise API notes, not complete implementations or hand-written task kernels.
The generator still creates all source and the independent evaluator still owns decisions.

Run ordinary evidence_loop and api_guided with the same three-turn cap, model, source
admission, inputs, evaluator, timing criteria and final check. Schedule the eight mode/task
pairs in randomized order. Ordinary feedback is unchanged. API-guided starts with launch
and memory context (plus reduction rules for normalization tasks), then attaches selected
cards to actual failure feedback. Each response has the same 4096-token cap and the flow
has the same 20k output-token / 900-second ceiling. Actual consumed tokens may differ.

The API notes were developed from the first campaign's failures on these same four task
families. This is a development pilot, not a held-out task-family generalization claim.
The final tensor shapes/random seed remain separate from search feedback. Report all
outcomes, including a null improvement or a correct kernel that is slower than baseline.
