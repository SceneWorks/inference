"""Which GPU architectures the CUDA build actually covers, derived from the build itself.

sc-19545. There are two questions people keep conflating, and only the second one can make a
render silently wrong:

  1. "Does `CUDA_COMPUTE_CAP` match the GPU in the box?"  -- it is NOT supposed to. The variable
     names the *packaging baseline*, the bottom rung of an architecture ladder, not the hardware.
     A build that declared the runner's own arch would stop running on every older customer GPU.
  2. "Is the runner's arch covered by the code the build actually emitted?"  -- this is the real
     invariant. When it is false for the QUANTIZED kernels the failure is silent: no compatible
     cubin, no PTX to JIT, and every Q4/Q8 `QMatMul` returns zeros while the process exits 0.

Question 2 is what sc-7544 fixed and what this module exists to keep fixed. The authority is
`crates/media/candle-gen/vendor/candle-kernels/build.rs` -- the file that emits the flags -- so
everything here is PARSED out of it rather than restated. Restating it is how the two drift apart.

## The two compile paths cover different arch sets

`candle-kernels` compiles twice, and the portability story is different on each side:

* **dense** kernels go through cudaforge `build_ptx()` -> `nvcc --ptx`, emitting `compute_<cap>`
  PTX. The driver JITs that to the runtime GPU's SASS, so the dense path covers every arch
  **>= cap**. A cap/hardware mismatch here is harmless by construction.
* **quantized + MoE** kernels (`mmq_gguf/*`, `moe/*`, `mmvq_gguf` -- the GGUF `QMatMul`) go
  through `build_lib()` -> `nvcc -c`, a SASS **object with no PTX**. cudaforge emits exactly one
  `-gencode` from `CUDA_COMPUTE_CAP`; the vendored fork appends more. Coverage here is the
  explicit ladder and nothing else -- there is no JIT fallback unless a `code=compute_N` line
  puts one there.

So the hazard lives entirely on the quant path, and `quant_path_covers()` is the predicate that
matters. `dense_path_covers()` is included because reporting only half the story is how the
original diagnosis went wrong.

## Compute capabilities are packed integers here

`80` is sm_8.0, `86` is sm_8.6, `120` is sm_12.0: `major = cap // 10`, `minor = cap % 10`. This
is CUDA's own `CUDA_COMPUTE_CAP` / `-gencode` spelling, so no conversion is needed anywhere.

Two compatibility rules, both CUDA's, not ours:

* **SASS** is binary-compatible upward *within one major* only. An sm_80 cubin runs on sm_86 and
  sm_89; it does not run on sm_90 or sm_120.
* **PTX** JITs forward across majors. `compute_80` PTX runs on anything >= 80.

The consequence worth knowing before it bites: **datacenter Blackwell sm_100 (B100/B200) is NOT
covered on the quant path.** 100 is major 10, so no sm_90 or sm_120 cubin serves it, and the
`compute_120` PTX floor is above it. `build.rs` says this is deliberate and out of scope. If that
hardware ever appears in the pool this module fails loudly instead of rendering black.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
VENDORED_BUILD_RS = (
    REPO_ROOT
    / "crates"
    / "media"
    / "candle-gen"
    / "vendor"
    / "candle-kernels"
    / "build.rs"
)

# `-gencode=arch=compute_90,code=sm_90` (native SASS) and
# `-gencode=arch=compute_120,code=compute_120` (embedded PTX, JITs forward).
# The `code=` half is the one that decides coverage; `arch=` is only the virtual ISA it was
# compiled from.
_GENCODE = re.compile(
    r"-gencode\s*=\s*arch\s*=\s*compute_(\d+)\s*,\s*code\s*=\s*(sm|compute)_(\d+)"
)

# build.rs disables the bf16 WMMA kernels below this cap:
#     if compute_cap < 80 { moe_builder = moe_builder.arg("-DNO_BF16_KERNEL"); }
# Parsed rather than hardcoded so that moving the floor upstream moves this floor with it.
_BF16_FLOOR = re.compile(r"if\s+compute_cap\s*<\s*(\d+)\s*\{")


def _read(build_rs: Path | None = None) -> str:
    return (build_rs or VENDORED_BUILD_RS).read_text(encoding="utf-8")


def _emitted_gencode_flags(source: str) -> list[str]:
    """The `-gencode` flags build.rs actually passes to nvcc, ignoring commentary.

    `.arg("-gencode=...")` is the only way a flag reaches the compiler, so anchor on the call and
    not on the bare string -- the same file's comment block quotes the ladder in prose, and a
    parser that matched prose would keep reporting coverage after the real flags were deleted.
    """
    return [
        match.group(1)
        for match in re.finditer(r"\.arg\(\s*\"(-gencode=[^\"]*)\"\s*\)", source)
    ]


def explicit_arches(source: str | None = None) -> tuple[frozenset[int], frozenset[int]]:
    """`(native_sass, ptx_floors)` added by the vendored fork, excluding the cudaforge baseline."""
    source = source if source is not None else _read()
    sass: set[int] = set()
    ptx: set[int] = set()
    for flag in _emitted_gencode_flags(source):
        match = _GENCODE.search(flag)
        if match is None:
            continue
        kind, cap = match.group(2), int(match.group(3))
        (sass if kind == "sm" else ptx).add(cap)
    return frozenset(sass), frozenset(ptx)


def bf16_kernel_floor(source: str | None = None) -> int:
    """The cap below which build.rs compiles the bf16 WMMA kernels OUT (`-DNO_BF16_KERNEL`)."""
    source = source if source is not None else _read()
    match = _BF16_FLOOR.search(source)
    if match is None:
        raise ValueError(
            "vendored candle-kernels build.rs no longer carries the `compute_cap < N` bf16 guard; "
            "the CUDA_COMPUTE_CAP lower bound is derived from it and can no longer be computed"
        )
    return int(match.group(1))


def baseline_bounds(source: str | None = None) -> tuple[int, int]:
    """`(lowest_allowed, first_disallowed)` for `CUDA_COMPUTE_CAP`, both read out of build.rs.

    The declared cap is not free, and it is not the hardware. It is pinned from both sides by
    build.rs's own code:

    * **Lower bound** -- `if compute_cap < 80 { -DNO_BF16_KERNEL }`. Declaring anything below that
      floor compiles the bf16 WMMA kernels out of `libmoe.a` entirely.
    * **Upper bound** -- the cap contributes the cudaforge `-gencode`, i.e. the ladder's BOTTOM
      rung. It must therefore sit strictly below the lowest rung the fork adds explicitly. Raise it
      to the runner's own 120 and the ladder loses its Ampere rung: the shipped worker stops
      running quantized models on every pre-Blackwell customer GPU, and the dense PTX floor moves
      to `compute_120` so even dense models stop loading there. That is the reason "fix the number
      to match the box" is a regression and not a fix.
    """
    source = source if source is not None else _read()
    sass, _ = explicit_arches(source)
    if not sass:
        raise ValueError(
            "vendored candle-kernels build.rs emits no explicit `code=sm_NN` gencode; the "
            "multi-arch fatbin (sc-7544) is gone and quantized matmuls will silently return zeros "
            "on any GPU whose arch is not exactly CUDA_COMPUTE_CAP"
        )
    return bf16_kernel_floor(source), min(sass)


def expected_baseline(source: str | None = None) -> int:
    """The one `CUDA_COMPUTE_CAP` every CI site must declare, derived from `baseline_bounds()`.

    The lowest cap satisfying both of build.rs's constraints -- lowest because the cap seeds the
    ladder's bottom rung, and the bottom rung is what buys support for the oldest GPU we ship to.
    """
    low, high = baseline_bounds(source)
    if low >= high:
        raise ValueError(
            f"vendored build.rs leaves no valid CUDA_COMPUTE_CAP: the bf16 floor ({low}) is not "
            f"below the lowest explicit gencode rung ({high})"
        )
    return low


def quant_path_arches(
    declared_cap: int, source: str | None = None
) -> tuple[frozenset[int], frozenset[int]]:
    """`(native_sass, ptx_floors)` for `libmoe.a` -- the fork's ladder plus the cudaforge rung."""
    sass, ptx = explicit_arches(source)
    return frozenset(sass | {declared_cap}), ptx


def _sass_covers(cubin_cap: int, device_cap: int) -> bool:
    """SASS is binary-compatible upward within one major version, and no further."""
    return cubin_cap // 10 == device_cap // 10 and cubin_cap % 10 <= device_cap % 10


def quant_path_covers(
    device_cap: int, declared_cap: int, source: str | None = None
) -> tuple[bool, str]:
    """Can the GGUF `QMatMul` kernels run on `device_cap`? `(covered, human-readable reason)`.

    False is the silent-zeros condition: no compatible cubin and no PTX to JIT means the kernel
    launch does not fail, it just produces nothing.
    """
    sass, ptx = quant_path_arches(declared_cap, source)
    native = sorted(cap for cap in sass if _sass_covers(cap, device_cap))
    if native:
        return True, f"native sm_{native[0]} cubin covers sm_{device_cap}"
    floors = sorted(floor for floor in ptx if floor <= device_cap)
    if floors:
        return True, f"compute_{floors[-1]} PTX JITs forward to sm_{device_cap}"
    return False, (
        f"NO compatible code for sm_{device_cap}: libmoe.a holds SASS for "
        f"{{{', '.join(f'sm_{cap}' for cap in sorted(sass))}}} and PTX floors "
        f"{{{', '.join(f'compute_{cap}' for cap in sorted(ptx)) or 'none'}}}. Quantized matmuls "
        f"will SILENTLY RETURN ZEROS on this device -- dense models still render, quantized "
        f"models come out black, and the process exits 0"
    )


def dense_path_covers(device_cap: int, declared_cap: int) -> tuple[bool, str]:
    """Dense kernels ship as `compute_<cap>` PTX, which JITs forward to anything at or above it."""
    if device_cap >= declared_cap:
        return True, f"compute_{declared_cap} PTX JITs forward to sm_{device_cap}"
    return False, (
        f"sm_{device_cap} is BELOW the compute_{declared_cap} PTX floor; the dense kernels cannot "
        f"load at all on this device"
    )


def parse_device_cap(text: str) -> int:
    """Parse `nvidia-smi --query-gpu=compute_cap` output (`12.0`) or a bare packed cap (`120`)."""
    text = text.strip()
    if not text:
        raise ValueError("empty compute capability")
    if "." in text:
        major, _, minor = text.partition(".")
        return int(major) * 10 + int(minor)
    return int(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--device-cap",
        action="append",
        default=[],
        metavar="CAP",
        help="a GPU's compute capability, as `12.0` or `120`. Repeat for multiple GPUs. Accepts "
        "`nvidia-smi --query-gpu=compute_cap --format=csv,noheader` output one value per flag.",
    )
    parser.add_argument(
        "--declared-cap",
        default=None,
        metavar="CAP",
        help="the CUDA_COMPUTE_CAP the build used. Defaults to $CUDA_COMPUTE_CAP, then to the "
        "value derived from build.rs.",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = parser.parse_args(argv)

    source = _read()
    declared = (
        int(args.declared_cap)
        if args.declared_cap
        else int(os.environ.get("CUDA_COMPUTE_CAP") or expected_baseline(source))
    )
    sass, ptx = quant_path_arches(declared, source)

    report: dict[str, object] = {
        "declared_cap": declared,
        "expected_baseline": expected_baseline(source),
        "quant_native_sass": sorted(sass),
        "quant_ptx_floors": sorted(ptx),
        "devices": [],
    }
    failed = False
    for raw in args.device_cap:
        for token in raw.replace(",", " ").split():
            device = parse_device_cap(token)
            quant_ok, quant_why = quant_path_covers(device, declared, source)
            dense_ok, dense_why = dense_path_covers(device, declared)
            report["devices"].append(
                {
                    "device_cap": device,
                    "quant_covered": quant_ok,
                    "quant_reason": quant_why,
                    "dense_covered": dense_ok,
                    "dense_reason": dense_why,
                }
            )
            failed = failed or not (quant_ok and dense_ok)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(
            f"CUDA_COMPUTE_CAP={declared} (derived baseline {report['expected_baseline']}); "
            f"libmoe.a SASS={sorted(sass)} PTX floors={sorted(ptx)}"
        )
        for entry in report["devices"]:
            status = "OK  " if entry["quant_covered"] and entry["dense_covered"] else "FAIL"
            print(f"  [{status}] sm_{entry['device_cap']} quant: {entry['quant_reason']}")
            print(f"         dense: {entry['dense_reason']}")

    if failed:
        print(
            "::error::the CUDA build does not cover every GPU on this runner -- see above. This is "
            "the sc-7544 silent-zeros condition, which does NOT fail a render, it just produces "
            "them wrong. Add the missing arch to the -gencode ladder in "
            "crates/media/candle-gen/vendor/candle-kernels/build.rs.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
