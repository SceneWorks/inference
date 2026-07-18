#!/usr/bin/env python3
"""Prepare a runtime release-candidate PR, and (with --release) cut the release.

Default (prepare):    compute the next runtime-YYYY.MM.patch-rc.N from existing tags, branch off
                      main, record it in release/VERSION, commit, push, and open a PR for approval.
--release (execute):  read the approved version from release/VERSION, build + verify the release
                      bundle, then create and push the immutable tag.
--promote (execute):  promote that rc to its final tag -- rebuild from the rc's exact revision and
                      create/push the bare runtime-YYYY.MM.patch tag (rc -> release).

The version lives only in the git tag (see release/README.md); release/VERSION is the in-repo
record of the version being prepared, so the approval PR has a reviewable diff and the release step
cuts exactly the approved version. Patch increments within a calendar month and resets on a new
month; a candidate is `-rc.N`, promoted to the bare final tag later from the same revision.

Safety: preview any run with --dry-run. --release/--promote push an immutable tag (never moved or
reused) and require --yes; --promote also requires --gates-passed. This script builds the source
bundle + SBOM and tags/pushes; it does NOT run the multi-platform or real-weight release gates --
those are CI's job per release/README.md.
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VERSION_FILE = ROOT / "release" / "VERSION"
BUILD_RELEASE = ROOT / "scripts" / "release" / "build_release.py"
VERIFY_RELEASE = ROOT / "scripts" / "release" / "verify_release.py"
DEFAULT_BASE = "main"
# Kept in lockstep with build_release.py's TAG_PATTERN (asserted by test_cut_release.py).
TAG_PATTERN = re.compile(r"^runtime-(\d{4})\.(\d{2})\.(\d+)(?:-rc\.(\d+))?$")


class AmbiguousBump(Exception):
    """The next candidate can't be chosen without an explicit re-spin/new-patch decision."""


# --- version algebra (pure: unit-tested offline) --------------------------------------------


@dataclass(frozen=True)
class Version:
    year: int
    month: int
    patch: int
    rc: int | None  # None => final release

    @property
    def sort_key(self) -> tuple[int, int, int, float]:
        # A final release sorts above every candidate of the same patch.
        return (self.year, self.month, self.patch, math.inf if self.rc is None else self.rc)

    def format(self) -> str:
        base = f"runtime-{self.year:04d}.{self.month:02d}.{self.patch}"
        return base if self.rc is None else f"{base}-rc.{self.rc}"


def parse(tag: str) -> Version | None:
    match = TAG_PATTERN.fullmatch(tag.strip())
    if not match:
        return None
    rc = None if match.group(4) is None else int(match.group(4))
    return Version(int(match.group(1)), int(match.group(2)), int(match.group(3)), rc)


def next_candidate(
    versions: list[Version],
    today: tuple[int, int],
    *,
    respin: bool = False,
    new_patch: bool = False,
) -> tuple[Version, str]:
    """Return (next candidate, human-readable reason) for the current calendar month.

    - First release of the month  -> patch 0, rc.0.
    - Top patch already finalized  -> new patch, rc.0.
    - Top patch has an in-flight rc (no final): ambiguous -- re-spinning that candidate
      (rc.N+1) and starting a fresh patch (patch+1, rc.0) are both valid, so require an
      explicit --respin/--new-patch choice rather than guess.
    """
    if respin and new_patch:
        raise AmbiguousBump("choose either --respin or --new-patch, not both")

    year, month = today
    this_month = [v for v in versions if (v.year, v.month) == (year, month)]
    if not this_month:
        return Version(year, month, 0, 0), f"first release of {year:04d}.{month:02d}"

    top_patch = max(v.patch for v in this_month)
    top = [v for v in this_month if v.patch == top_patch]
    finalized = any(v.rc is None for v in top)
    label = f"{year:04d}.{month:02d}.{top_patch}"

    if new_patch or (finalized and not respin):
        reason = "forced new patch" if new_patch else f"{label} is final"
        return Version(year, month, top_patch + 1, 0), f"{reason} -> new patch {top_patch + 1}, rc.0"

    max_rc = max((v.rc for v in top if v.rc is not None), default=None)
    if max_rc is None:
        raise AmbiguousBump(f"{label} has no candidate to re-spin; use --new-patch")
    if not respin and not finalized:
        raise AmbiguousBump(
            f"{label}-rc.{max_rc} has no final tag: pass --respin for {label}-rc.{max_rc + 1} "
            f"or --new-patch for {year:04d}.{month:02d}.{top_patch + 1}-rc.0"
        )
    return Version(year, month, top_patch, max_rc + 1), f"re-spin {label} -> rc.{max_rc + 1}"


def promote_target(rc_tag: str) -> str:
    """The final tag a candidate promotes to (strip -rc.N). Raise if it isn't a candidate."""
    version = parse(rc_tag)
    if version is None:
        raise ValueError(f"{rc_tag!r} is not a runtime release tag")
    if version.rc is None:
        raise ValueError(f"{rc_tag} is already a final release; nothing to promote")
    return Version(version.year, version.month, version.patch, None).format()


# --- derived artifacts (pure) ---------------------------------------------------------------


def branch_name(tag: str) -> str:
    return f"release/{tag}"


def version_file_content(tag: str) -> str:
    return f"{tag}\n"


def pr_title(tag: str) -> str:
    return f"release: {tag}"


def pr_body(tag: str) -> str:
    return (
        f"Prepared release candidate **`{tag}`** (auto-generated by "
        "`scripts/release/cut_release.py`).\n\n"
        f"- `release/VERSION` records the version being cut.\n"
        f"- Approve/merge this PR, then cut the release from the approved revision:\n\n"
        f"  ```sh\n"
        f"  python3 scripts/release/cut_release.py --release --yes\n"
        f"  ```\n\n"
        "The release step builds the source bundle + SBOM, verifies it, and pushes the immutable "
        "tag. Multi-platform and real-weight gates run in CI (see `release/README.md`)."
    )


# --- executable step plan -------------------------------------------------------------------


@dataclass(frozen=True)
class Command:
    description: str
    argv: list[str]


@dataclass(frozen=True)
class WriteFile:
    description: str
    path: Path
    content: str


Step = Command | WriteFile


def plan_prepare(tag: str, base: str, *, fetch: bool = True) -> list[Step]:
    branch = branch_name(tag)
    rel_version = str(VERSION_FILE.relative_to(ROOT))
    steps: list[Step] = []
    if fetch:
        steps.append(Command(f"fetch origin/{base}", ["git", "fetch", "origin", base, "--tags"]))
    steps += [
        Command(f"create branch {branch} off origin/{base}", ["git", "switch", "-c", branch, f"origin/{base}"]),
        WriteFile(f"record {tag} in {rel_version}", VERSION_FILE, version_file_content(tag)),
        Command(f"stage {rel_version}", ["git", "add", rel_version]),
        Command(f"commit {tag}", ["git", "commit", "-m", f"release: prepare {tag}"]),
        Command(f"push {branch}", ["git", "push", "-u", "origin", branch]),
        Command(
            "open PR for approval",
            ["gh", "pr", "create", "--base", base, "--head", branch,
             "--title", pr_title(tag), "--body", pr_body(tag)],
        ),
    ]
    return steps


def plan_release(tag: str, *, offline: bool = False) -> list[Step]:
    build = [sys.executable, str(BUILD_RELEASE), "--tag", tag]
    verify = [sys.executable, str(VERIFY_RELEASE), "dist/release"]
    if offline:
        build.append("--offline")
        verify.append("--offline")
    return [
        Command(f"build release bundle for {tag}", build),
        Command("verify release bundle", verify),
        Command(f"create annotated tag {tag}", ["git", "tag", "-a", tag, "-m", f"Runtime release {tag}"]),
        Command(f"push tag {tag}", ["git", "push", "origin", tag]),
    ]


def plan_promote(rc_tag: str, final_tag: str, revision: str, *, offline: bool = False) -> list[Step]:
    # Rebuild the final bundle from the candidate's EXACT revision (source unchanged), then create
    # and push the final tag at that same revision. Build first so a build failure never leaves a
    # dangling local tag.
    build = [sys.executable, str(BUILD_RELEASE), "--tag", final_tag, "--source-ref", revision]
    verify = [sys.executable, str(VERIFY_RELEASE), "dist/release"]
    if offline:
        build.append("--offline")
        verify.append("--offline")
    return [
        Command(f"build release bundle for {final_tag} from {rc_tag} ({revision})", build),
        Command("verify release bundle", verify),
        Command(f"create final tag {final_tag} at {rc_tag}'s revision",
                ["git", "tag", "-a", final_tag, revision, "-m", f"Runtime release {final_tag}"]),
        Command(f"push tag {final_tag}", ["git", "push", "origin", final_tag]),
    ]


def execute(steps: list[Step], *, dry_run: bool) -> int:
    for step in steps:
        print(f"• {step.description}")
        if isinstance(step, WriteFile):
            print(f"    write {step.path.relative_to(ROOT)}: {step.content!r}")
            if not dry_run:
                step.path.write_text(step.content, encoding="utf-8")
        else:
            print(f"    $ {' '.join(step.argv)}")
            if not dry_run:
                result = subprocess.run(step.argv, cwd=ROOT, check=False)
                if result.returncode:
                    print(f"cut_release: step failed (exit {result.returncode}); stopping.", file=sys.stderr)
                    return result.returncode
    return 0


# --- git queries + guards -------------------------------------------------------------------


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=False, capture_output=True, encoding="utf-8"
    )
    if result.returncode:
        raise SystemExit(f"cut_release: `git {' '.join(args)}` failed: {result.stderr.strip()}")
    return result.stdout


def existing_versions(*, fetch: bool) -> list[Version]:
    if fetch:
        subprocess.run(["git", "fetch", "--tags", "--quiet"], cwd=ROOT, check=False)
    tags = _git("tag", "--list", "runtime-*").splitlines()
    return [v for v in (parse(line) for line in tags) if v is not None]


def working_tree_dirty() -> bool:
    return bool(_git("status", "--porcelain").strip())


def tag_exists(tag: str) -> bool:
    return bool(_git("tag", "--list", tag).strip())


def resolve_commit(ref: str) -> str | None:
    """The commit a ref points at, or None if the ref does not exist (non-raising)."""
    result = subprocess.run(
        ["git", "rev-parse", "-q", "--verify", f"{ref}^{{commit}}"],
        cwd=ROOT, check=False, capture_output=True, encoding="utf-8",
    )
    return result.stdout.strip() or None


def read_prepared_version() -> str:
    if not VERSION_FILE.is_file():
        raise SystemExit(f"cut_release: {VERSION_FILE.relative_to(ROOT)} not found; run prepare first.")
    tag = VERSION_FILE.read_text(encoding="utf-8").strip()
    if parse(tag) is None:
        raise SystemExit(f"cut_release: {VERSION_FILE.relative_to(ROOT)} holds a non-release tag: {tag!r}")
    return tag


# --- command line ---------------------------------------------------------------------------


def run_prepare(args: argparse.Namespace) -> int:
    versions = existing_versions(fetch=not args.no_fetch)
    today = date.today()
    try:
        candidate, reason = next_candidate(
            versions, (today.year, today.month), respin=args.respin, new_patch=args.new_patch
        )
    except AmbiguousBump as error:
        print(f"cut_release: {error}", file=sys.stderr)
        return 2
    tag = candidate.format()
    latest = max(versions, key=lambda v: v.sort_key).format() if versions else "(none)"
    print(f"latest existing: {latest}  ->  preparing: {tag}  ({reason})\n")

    if not args.dry_run and working_tree_dirty():
        print("cut_release: working tree is dirty; commit/stash before preparing a release.", file=sys.stderr)
        return 1
    return execute(plan_prepare(tag, args.base, fetch=not args.no_fetch), dry_run=args.dry_run)


def run_release(args: argparse.Namespace) -> int:
    tag = read_prepared_version()
    print(f"releasing prepared version: {tag}\n")

    if not args.dry_run:
        if not args.yes:
            print("cut_release: --release pushes an immutable tag; re-run with --yes to proceed "
                  "(or --dry-run to preview).", file=sys.stderr)
            return 1
        if tag_exists(tag):
            print(f"cut_release: tag {tag} already exists and is never reused; refusing.", file=sys.stderr)
            return 1
        if working_tree_dirty():
            print("cut_release: working tree is dirty; release must be cut from a clean revision.", file=sys.stderr)
            return 1
    return execute(plan_release(tag, offline=args.offline), dry_run=args.dry_run)


def print_gate_checklist(rc_tag: str) -> None:
    print(
        f"Release gates that MUST be green on {rc_tag}'s revision before promoting "
        "(see release/README.md):\n"
        "  1. CI lanes: workspace, contracts, affected backend/platform, docs, supply-chain\n"
        "  2. Required real-weight profiles (these run in CI, not here)\n"
        "  3. verify_release.py passed on the rc bundle (no --allow-dirty/--skip-smoke)\n"
        "  4. Uploaded artifact hashes match SHA256SUMS\n",
        file=sys.stderr,
    )


def run_promote(args: argparse.Namespace) -> int:
    rc_tag = read_prepared_version()
    try:
        final_tag = promote_target(rc_tag)
    except ValueError as error:
        print(f"cut_release: {error}", file=sys.stderr)
        return 1
    print(f"promoting {rc_tag}  ->  {final_tag}\n")
    print_gate_checklist(rc_tag)

    revision = resolve_commit(rc_tag)
    if not args.dry_run:
        if not args.gates_passed:
            print(f"cut_release: confirm the gates above passed on {rc_tag}, then re-run with "
                  "--gates-passed.", file=sys.stderr)
            return 1
        if not args.yes:
            print("cut_release: --promote pushes an immutable final tag; re-run with --yes.", file=sys.stderr)
            return 1
        if revision is None:
            print(f"cut_release: candidate tag {rc_tag} not found; cut it first with --release.", file=sys.stderr)
            return 1
        if tag_exists(final_tag):
            print(f"cut_release: final tag {final_tag} already exists and is never reused; refusing.", file=sys.stderr)
            return 1
        if working_tree_dirty():
            print("cut_release: working tree is dirty; promote from a clean checkout.", file=sys.stderr)
            return 1
    plan = plan_promote(rc_tag, final_tag, revision or f"<{rc_tag} revision>", offline=args.offline)
    return execute(plan, dry_run=args.dry_run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--release", action="store_true",
                      help="execute the release for the rc in release/VERSION (default: prepare a PR)")
    mode.add_argument("--promote", action="store_true",
                      help="promote the rc in release/VERSION to its final tag (rc -> release)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and change nothing")
    parser.add_argument("--yes", action="store_true", help="required to actually push the immutable tag (--release/--promote)")
    parser.add_argument("--gates-passed", action="store_true",
                        help="acknowledge the release gates passed on the rc (required by --promote)")
    parser.add_argument("--base", default=DEFAULT_BASE, help=f"PR base branch (default: {DEFAULT_BASE})")
    parser.add_argument("--no-fetch", action="store_true", help="skip fetching tags/base first (local may be stale)")
    parser.add_argument("--offline", action="store_true", help="pass --offline to the release build/verify")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--respin", action="store_true", help="prepare: re-spin the in-flight candidate (rc+1)")
    group.add_argument("--new-patch", action="store_true", help="prepare: start a new patch even if the top rc isn't final")
    args = parser.parse_args(argv)

    if args.promote:
        return run_promote(args)
    return run_release(args) if args.release else run_prepare(args)


if __name__ == "__main__":
    raise SystemExit(main())
