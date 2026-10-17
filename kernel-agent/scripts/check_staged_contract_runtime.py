"""Run the identical checks inside the isolated real-SDK environment."""
import subprocess
from pathlib import Path

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    raise SystemExit(subprocess.call(['/data/home/chenyizhou/kernel-agent-storage/runtimes/multistage/bin/python',
                                    str(root/'scripts/check_staged_contract.py'),'--runtime'],cwd=root))
