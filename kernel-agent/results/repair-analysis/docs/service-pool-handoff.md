# A recorded endpoint restart, without replaying completed work

The initial pool reached 15 committed generation/final outcomes. Ports 54575 and
60953 then refused connections before the next jobs were admitted, leaving nine
untouched jobs. The owned endpoint completed all eight of its assigned jobs.
The pre-handoff snapshot is retained in `results/pool-before-handoff/`.

Public process arguments showed replacement Qwen API processes on 56825 and 43619.
Public NVML compute PIDs and `/proc` parent chains associated them with GPU0/1 and
GPU2/3 respectively. Their HTTP model root, served base alias, parent, context cap
and vLLM version matched the recorded declarations. No environment, model weights,
private files, logs or other users' prompts were read. No model process was stopped
or modified. These checks do not establish identical weights or cache/process continuity.

`scripts/service_gateway.py` restores the two original logical loopback entrances:

| Logical entrance | Replacement upstream | Physical GPUs |
|---|---|---|
| 54575 | 56825 | 0/1 |
| 60953 | 43619 | 2/3 |

The gateway forwards only the five required API routes, preserves streaming bytes,
does not log payloads/headers and never retries a request. The original proxy still
checks identity, idle metrics and context before every model dispatch. The original
source manifests, immutable plans and budgets are unchanged. Workers reopen their
ledgers: completed jobs are reused, only pending jobs start, and unknown jobs are
never replayed. The eight completed owned-service jobs are also reopened without
new model requests or measurements.

`handoff/before.json` records all 15 result hashes, pending job IDs, budget reservations
and old request/admission hashes. `gateway-receipt.json`, the exact gateway source
and `gpu-mapping.json` preserve the transport change. Offline verification checks that
old results and requests remain unchanged and budgets are not reset.

Five planned seed/strategy pairs straddle the restart. The report therefore suppresses
the pooled feedback-effect interval. Final job counts describe engineering coverage
across a documented restart; they do not complete the originally intended continuous
service-condition comparison. The pre-handoff partial comparison and the restart
evidence remain accessible independently. This is a harness recovery result, not a
claim that feedback improves kernel quality or that different cached roots are equivalent.

## Completion and task-owned gateway cleanup

All 24 engineering outcomes resolved and all three workers exited zero. The completion
audit verifies 15 unchanged committed results, 178 unchanged request/admission files,
zero new request files for old jobs and all eight owned-service checkpoint bytes unchanged.
See `results/pool-completion-audit.json`. Final correctness is 20/24 and conditional
promotion 6/24; both arms have 10/12 correct and 3/12 promoted.

The replacement 56825 endpoint disappeared again before the last RMSNorm job's third
proposal could pass preflight. No third proposer request reached that service; two
previously retained candidates allowed independent finalization. Together with the
earlier Softmax and transpose incidents, three jobs ended search with `harness_error`.
Two still finalized correct retained candidates; transpose had no eligible candidate.
This is observed candidate preservation, not a clean three-proposal strategy comparison.

After all work ended, the owned 18118 endpoint and borrowed 43619 answered metadata
checks; 56825 was unavailable. The temporary task-owned gateway (PID 2259772) was then
stopped and its logical ports released. The owned Qwen was kept running and no foreign
model process was changed. Current endpoint continuity is not assumed; the receipt is
`results/pool-runtime-status.json`.
