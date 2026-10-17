"""Apply the common task-reference backend inside each isolated real SDK worker."""
import runpy
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from kernel_agent.representative_reference_router import routed_references  # noqa: E402

if __name__=='__main__':
    with routed_references():
        runpy.run_path(str(ROOT/'scripts/staged_contract_worker.py'),run_name='__main__')
