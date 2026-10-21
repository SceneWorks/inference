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
    # Cargo considers the target a library and passes library output/filename flags even
    # though --test supplies the actual harness. With an explicit -o, require a real executable
    # at that exact path: bin + link only, no library suffix or Cargo extra-filename. Keep the
    # crate metadata disambiguator, features, cfgs, source, and all semantic/codegen flags.
    explicit_output = "-o" in args
    result = []
    index = 0
    while index < len(args):
        arg = args[index]
        if explicit_output and arg == "--crate-type" and index + 1 < len(args):
            if args[index + 1] not in ("lib", "rlib", "bin"):
                raise RuntimeError("unexpected primary regex test crate type")
            result.extend([arg, "bin"])
            index += 2
        elif explicit_output and arg.startswith("--crate-type="):
            if arg.split("=", 1)[1] not in ("lib", "rlib", "bin"):
                raise RuntimeError("unexpected primary regex test crate type")
            result.append("--crate-type=bin")
            index += 1
        elif explicit_output and arg == "--emit" and index + 1 < len(args):
            result.extend([arg, "link"])
            index += 2
        elif explicit_output and arg.startswith("--emit="):
            result.append("--emit=link")
            index += 1
        elif explicit_output and arg == "-C" and index + 1 < len(args) and args[index + 1].startswith("extra-filename="):
            index += 2
        elif explicit_output and arg.startswith("-Cextra-filename="):
            index += 1
        elif explicit_output and arg == "--out-dir" and index + 1 < len(args):
            index += 2
        elif explicit_output and arg.startswith("--out-dir="):
            index += 1
        else:
            result.append(arg)
            index += 1
    for index, arg in enumerate(result):
        if index == 0 or result[index - 1] != "--extern" or "=" not in arg:
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
