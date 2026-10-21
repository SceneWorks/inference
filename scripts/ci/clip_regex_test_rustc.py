#!/usr/bin/env python3
"""Use exact built rlibs only for Cargo's injected primary regex test harness.

Normal compiler invocations are forwarded unchanged. No crate source, predicate, feature,
provenance marker, error, or test is replaced or suppressed. Metadata filenames and their rlib
counterparts must have the identical Cargo hash; no glob/nearest-artifact fallback is allowed.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path


def linkable_test_args(args: list[str]) -> list[str]:
    crate = None
    for index, arg in enumerate(args):
        if arg == "--crate-name" and index + 1 < len(args):
            crate = args[index + 1]
        elif arg.startswith("--crate-name="):
            crate = arg.split("=", 1)[1]
    if crate != "regex" or "--test" not in args:
        return list(args)
    result = list(args)
    for index, arg in enumerate(args):
        if index == 0 or args[index - 1] != "--extern" or "=" not in arg:
            continue
        name, raw = arg.split("=", 1)
        metadata = Path(raw)
        if metadata.suffix != ".rmeta":
            continue
        rlib = metadata.with_suffix(".rlib")
        if not rlib.is_file():
            raise RuntimeError(f"required same-hash test rlib is absent: {rlib}")
        result[index] = name + "=" + str(rlib)
    return result


def main() -> None:
    if len(sys.argv) < 2:
        raise RuntimeError("Cargo must supply the actual rustc command")
    rustc, *args = sys.argv[1:]
    os.execvp(rustc, [rustc, *linkable_test_args(args)])


if __name__ == "__main__":
    main()
