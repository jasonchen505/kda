# Larger cached local Qwen capability pilot

The 4B model repeatedly fails basic Python/API/interface repair even with reference code.
After completing the frozen representation comparison, test the same latest controller
on the already-cached local 27B FP8 model. This is a capacity/integration pilot, not evidence
that the controller alone improves success. Do not pool models or prior studies.

The local directory is named `Qwen3.8-27B-FP8`, but its config says `qwen3_5`, architecture
`Qwen3_5ForConditionalGeneration`, 64 text layers, hidden size 5120 and FFN size 17408.
Report the path alias and actual config separately; do not infer an official model release
from the directory name. It contains 66 weight shards totalling 28.75 GiB. Record config,
index/tokenizer hashes and a size/mtime inventory. The upstream weight revision is unknown;
these records identify this cached asset, not a verified release or full weight digest.

Create an isolated loopback-only service using free GPUs 4/5 after stopping our 4B service.
Retain the user's existing GPU0–3 services. Evaluate generated kernels on GPU6. The local
service serves the base model without an adapter; no paid API, new weight download or training.

Run four task families using reference-guided mode with json-lines, the existing three-turn
and 4096-token limits, immutable evaluator and independent final checks. Seed remains
20261011 and thinking remains disabled. Record actual usage and every failure. This is a
single run per task on development families; reliability/generalization remains unproven.
