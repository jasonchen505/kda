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

## Current versus intended architecture

The frozen guarded campaign uses the deliberately conservative policy documented in
`measurement-guard.md`: wait for a nearly empty device and reject foreign PIDs at evaluator
boundaries. This can reject an idle resident model. It cannot detect all short-lived
interference. Neither property should be presented as perfect resource isolation.

The more resource-efficient next architecture separates numerical validation from timing:
allow shared generation/compilation/correctness work, then queue timing for a controlled
measurement window. Alternatively, fix and record an explicit co-running workload and
report shared-environment performance as its own experimental condition. Those measurements
must not be described as standalone kernel speedups. This separation is not yet implemented
in the frozen campaign, whose source/runtime plan must not change mid-experiment.
