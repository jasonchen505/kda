"""Process-local task-reference injection before the legacy retrieval lookup."""
from contextlib import contextmanager

from kernel_agent import staged_tools
from kernel_agent.representative_tasks import specs


@contextmanager
def routed_references():
    # Each real arm has its own isolated SDK worker. Preserve all legacy guard,
    # alias, deduplication and event semantics; replace only the reference backend.
    old_retrieve,old_example=staged_tools.retrieve,staged_tools.retrieve_example
    task_specs=specs()
    def retrieve(result,task_id):
        if task_id in task_specs:
            return {'context':task_specs[task_id]['api_notes']}
        return old_retrieve(result,task_id)
    def example(task_id):
        if task_id in task_specs:
            return {'text':task_specs[task_id]['api_notes'],
                    'scope':'Task API notes only; no complete generated candidate supplied'}
        return old_example(task_id)
    staged_tools.retrieve,staged_tools.retrieve_example=retrieve,example
    try:
        yield
    finally:
        staged_tools.retrieve,staged_tools.retrieve_example=old_retrieve,old_example
