# GPU sharing: execution versus performance evidence

Sharing a GPU is permitted. There is no CUDA requirement that a kernel agent must own
an otherwise empty device. This project's strict PID/VRAM gate is an experimental policy,
not a universal hardware or numerical-correctness requirement.

Generation, source validation, compilation and numerical checks can normally share a
device, subject to available memory and operational timeouts. Correctness failures must
be distinguished from OOM/resource failures. A resident CUDA process can be idle: its PID
or allocated memory alone does not prove active contention.

Performance comparisons answer a different question. Other GPU activity can affect shared
SMs, L2/DRAM resources and clocks, and need not affect the candidate and compiled reference
equally. Repetition or medians do not remove a systematic background-load effect. NVIDIA's
[Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#reproducibility)
describes shared-resource activity and clock/cache controls for reproducible measurement.
Its recommendations are measurement guidance, not a prohibition on ordinary concurrent CUDA
execution. This project uses a separate CUDA-graph evaluator, not Nsight Compute itself.

## Two implemented execution modes

The frozen guarded campaign uses the deliberately conservative policy documented in
`measurement-guard.md`: wait for a nearly empty device and reject foreign PIDs at evaluator
boundaries. This can reject an idle resident model. It cannot detect all short-lived
interference. Neither property should be presented as perfect resource isolation.

The new deferred mode separates numerical validation from timing: shared generation and
search correctness checks produce a durable candidate queue, then a different invocation
performs guarded search measurements and the final veto. Its implementation, tradeoff and
separate pilot are described in `deferred-evaluation.md`. The model receives numerical
feedback only in this mode. It does not receive delayed timing or final-data feedback.

The original integrated campaign remains frozen and uses its original immediate performance
feedback and conservative occupancy policy. Its source/runtime plan is not changed to make
remaining jobs easier to run. The two modes' results are not pooled. A controlled co-running
workload could be another experimental condition, but shared-environment timings would need
their own label and must not be called standalone kernel speedups.

## Observed sharing, authorized on 2026-10-14

`evaluate_coexecuted.py` permits resident CUDA processes and measures both reference and
candidate under the observed environment. It records memory, utilization, SM clock, power
and PIDs at evaluator boundaries and nominally every 0.5 seconds during execution. It keeps
the same numerical checks, interleaved CUDA-graph samples and final veto. Monitor faults or
insufficient memory remain infrastructure failures. Presence of a process is no longer a veto.

The record says `measurement_condition=shared_observed`. Neither the queue nor the offline
auditor can import those measurements into an old uncontended plan. The existing strict
queues are preserved; a newly frozen plan reuses committed proposals without model calls.
This condition records ambient sharing rather than a controlled co-running workload, so
uncontrolled load and observation overhead limit the interpretation. See `shared-study-plan.md`.
