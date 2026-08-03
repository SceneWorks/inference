#!/usr/bin/env python3
"""Expand a leading `~/` in runner-local snapshot paths to the runner's own home.

WHY THIS EXISTS: every snapshot path reaches a job as a repository variable, and a
repository variable holds exactly ONE value for every runner that resolves it. That was
survivable while each label pool was a single box, and the workflow leaned on it — the
macOS pool reads `MAGE_SNAPSHOT` while the Windows pool reads `CANDLE_MAGE_SNAPSHOT`
under `E:\\huggingface\\hub`, the same checkpoint carried as two absolute variables.

A SECOND macOS box breaks that convention rather than extending it. The two Macs do not
share a home directory (`/Users/michael` and `/Users/MTrefry`), so a single absolute
`/Users/michael/...` value is only correct on whichever box happens to claim the job, and
`runs-on` label matching deliberately does not say which one that will be. Duplicating
every variable per box (`MAGE_SNAPSHOT_2`, ...) does not help either: the job would have
to know which machine it landed on to pick the right one.

A `~/`-relative value sidesteps all of it — it names the same location on both boxes and
is resolved here, per job, against the home of the runner that actually took the work.

DELIBERATELY ONLY A LEADING `~/`. Absolute values pass through untouched, which is what
keeps this migration incremental: the workflow can carry this step everywhere before a
single variable is rewritten, and each variable can be flipped from absolute to `~/` on
its own without a coordinated cutover or a window where the lane is red. It is also what
leaves the Windows pool's `E:\\...` values alone.

If a box stores its Hugging Face cache off the home volume, symlink `~/.cache/huggingface`
at it rather than reintroducing an absolute per-machine value.
"""

from __future__ import annotations

import argparse
import os
from pathlib import PurePosixPath


def resolve(value: str, home: str) -> str:
    """Return `value` with a leading `~/` replaced by `home`, else unchanged.

    PurePosixPath, not Path: a `~/`-relative value is by construction a POSIX runner home -- the
    convention exists for the two macOS boxes -- but `Path` takes the flavour of whatever host runs
    this. On the Windows CUDA dev box that turned `~/a/b` into `\\a\\b`, so the repository's own
    `scripts/tests` suite failed there while staying green on the ubuntu runner CI executes it on.
    """
    if value == "~" or value.startswith("~/"):
        return str(PurePosixPath(home) / value[2:]) if value.startswith("~/") else home
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "names",
        nargs="+",
        help="Environment variable names to resolve in place.",
    )
    args = parser.parse_args()

    home = os.environ.get("HOME", "")
    if not home:
        print("HOME is unset; cannot resolve a `~/`-relative snapshot path", flush=True)
        return 1

    github_env = os.environ.get("GITHUB_ENV")
    if not github_env:
        print("GITHUB_ENV is unset; this step only runs inside GitHub Actions", flush=True)
        return 1

    # An unset or empty variable is left exactly as it is. Several jobs fail closed on a
    # missing snapshot path with a message naming the variable an operator forgot to set;
    # substituting a bare home directory here would turn that into a confusing downstream
    # "revision mismatch" against whatever happens to sit in `~`.
    updates: list[tuple[str, str]] = []
    for name in args.names:
        value = os.environ.get(name, "")
        if not value:
            continue
        resolved = resolve(value, home)
        if resolved != value:
            updates.append((name, resolved))
            print(f"{name}: {value} -> {resolved}", flush=True)

    if updates:
        # `GITHUB_ENV` is append-only and last-write-wins, so writing only the changed
        # variables leaves every untouched value at its original definition.
        with open(github_env, "a", encoding="utf-8") as handle:
            for name, resolved in updates:
                if "\n" in resolved:
                    raise RuntimeError(f"{name} resolves to a value containing a newline")
                handle.write(f"{name}={resolved}\n")

    print(f"resolved {len(updates)} of {len(args.names)} snapshot paths", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
