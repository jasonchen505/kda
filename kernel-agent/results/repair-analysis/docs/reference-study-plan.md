# Reference-assisted agent development pilot

The first two campaigns produced no correct candidates. The second also exposed a real
OpenCode ContextOverflowError after long model replies accumulated. Do not hide these
outcomes; archive both full campaigns and their measured controller sources.

The next development version combines three changes: a source-proposer system prompt
aligned with the read-only role, licensed Triton example retrieval, and a new conversation
per repair carrying only the task, last admitted source (12k characters maximum), key
failure feedback (3500 characters maximum), API notes and the reference excerpt.
This is a combined intervention. It cannot identify each component's independent effect.

Reference examples come from official Triton v3.2.0 tutorials, with source/license/SHA
provenance. They contain implementation patterns and, for vector addition, a complete
example kernel. Outputs are reference-assisted, not evidence of unassisted or novel
kernel synthesis. The model still supplies the actual task implementation; never manually
fix generated candidates or count a handwritten evaluator fixture as model success.

Run the same four tasks, three-turn cap, 4096-token request limit, evaluator and final
shapes. Generation remains on Qwen3-4B-Instruct-2507. This is a third, sequential development
campaign on already-seen task families, not a randomized causal comparison or SOTA test.
