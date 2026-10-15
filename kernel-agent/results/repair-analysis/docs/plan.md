# Kernel agent reproduction and improvement

The research object is the agent that writes kernels. Hand-written operator results in
the sibling portfolio are auxiliary evaluation material, not agent results.

## Reproduction boundary
KDA at ef6ce617693ef0782b3ecb9f37e39bbf10226a90 is a workflow and prompt reference,
not a packaged autonomous kernel generator. Humanize at
77e3d0658d9dd4c54cd9497e45940593bf6a43ff supplies the actual coding-agent driver.
Reproduce its real Agent.spawn/run path with OpenCode 1.18.32 and the existing local
Qwen model. Initial attempts used qwen38-27b-fp8, with no LoRA adapter, but the
shared service queued requests. The measured pilot instead uses a dedicated cached
Qwen3-4B-Instruct-2507 snapshot cdbee75f17c01a7cc42f958dc650907174af0554 on GPU 5;
kernel evaluation uses GPU 6 because GPU 4 acquired another research workload. A local protocol adapter disables thinking, fixes
temperature/seed and caps every request at 4096 tokens. Paid providers are disabled.

## Downstream agent changes
The model proposes a plan and source. The controller stores candidates and runs a
separate GPU evaluator through Humanize's LocalEnv. The model cannot select its own
promotion decision. Add typed failure feedback, bounded repair, SHA-linked artifacts,
duplicate detection, explicit terminal states, and a quality-before-timing gate.
Humanize upstream and KDA upstream remain unmodified.

## Predeclared pilot
Compare a KDA-prompt adaptation with one generation turn against the same prompt plus
an evidence-driven loop of at most three turns. Use four generation tasks: vector add,
row softmax, RMSNorm and matrix transpose. Same base model, dtype/shape contracts,
initial task prompts, and evaluator. Each mode/task has one fresh session. This is
a small descriptive pilot; the larger repair budget is a confound, and this comparison
does not isolate structured feedback from additional inference compute.

The evaluator owns the hidden input tensors and references. It checks shape, dtype,
input mutation, repeat calls, finite values, odd sizes, and Triton implementation.
Correct kernels are benchmarked against torch.compile on evaluation shapes with three
randomized provider repeats using CUDA graphs. Performance promotion requires geomean
>=1.05 and worst >=0.90. A correct kernel that misses these thresholds remains valid
but is not performance-promoted. Record failures and token usage as actually available;
never infer successful model generation from mocked flow tests.

## Evidence and presentation
Keep prompts, model replies, source hashes, failure feedback, candidate parent links,
Humanize session identifiers and final decisions. Report generation success, performance
promotion, attempts, time and observed usage separately. Search evaluation feedback is
allowed to reveal its cases. After selection, a fresh process evaluates distinct shapes
and random inputs once, without sending the final results back to the agent. The headline
success/promotion metrics use this final check. Deliver an offline Chinese HTML
report with architecture, selectable trajectories, charts, raw downloads, limitations
and reproduction commands. Prefer short sentences and fixed terms inspired by the
Karpathy explanation recommendation; do not claim ASD-STE100 conformance in Chinese.
