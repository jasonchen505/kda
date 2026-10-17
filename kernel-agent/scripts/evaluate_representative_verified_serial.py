"""UUID-bound representative evaluator under the same cooperative GPU4 lock."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]

from kernel_agent.resource_lock import acquire  # noqa: E402
from scripts.evaluate_coexecuted import evaluate  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lock-file',type=Path,required=True)
    parser.add_argument('task')
    parser.add_argument('source')
    parser.add_argument('--final',action='store_true')
    args = parser.parse_args()
    command = [sys.executable,str(Path(__file__).with_name('evaluate_representative_verified.py')),args.task,args.source]
    if args.final:
        command.append('--final')
    with acquire(args.lock_file) as (lock,waited):
        result = evaluate(command,4,pass_fds=(lock.fileno(),),require_device_proof=True)
    result['cooperative_queue'] = {'lock_file':str(args.lock_file),'wait_seconds':waited}
    print(json.dumps(result,allow_nan=False))


if __name__ == '__main__':
    main()
