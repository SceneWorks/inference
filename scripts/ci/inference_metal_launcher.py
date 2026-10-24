#!/usr/bin/env python3
"""Inference adapter: shared cooperative lease plus existing Metal process census.

The portable lease helper stays byte-identical across repos. This adapter uses this repo's
existing census. It does not turn a process-name census into universal device ownership.
"""
import argparse
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ci.metal_execution_lock import execution_lease, run_owned
from scripts.ci.yue2_precision_proof import metal_census, physical_busy_message


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-seconds', type=float, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = args.command[1:] if args.command[:1] == ['--'] else args.command
    try:
        with execution_lease(args.wait_seconds) as lease:
            raw, busy = metal_census()
            if busy:
                raise ValueError(physical_busy_message(raw, busy, 'unwrapped Metal worker detected'))
            return run_owned(argv, lease)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print('Metal launcher refused: ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
