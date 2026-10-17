"""Common extension contract; no conflicting historical numeric tolerance boilerplate."""
from kernel_agent.representative_tasks import specs


def brief(task_id,contract,stage,arm):
    return (f'Task {task_id}; active stage {stage}; arm {arm}.\n{contract}\n'
        'RTX4090 Ada, Torch2.6/Triton3.2. Triton compute only; torch allocation/metadata. '
        'No tests/examples/main block inside a submitted kernel module. '
        'Numeric tolerances are exactly those in this task contract above. '
        'Search promotion geomean>=1.05,worst>=.90; final shapes are unavailable and must not be queried. '
        'No Blackwell-only TMEM/tcgen05. Roles are fresh contexts of the same model process, '
        'not independent model weights. '+specs()[task_id]['api_notes'])
