#!/usr/bin/env python3
"""Fail closed when a feature-epic pull request violates branch topology.

Feature work is integrated through a repository-local branch train:

* ``story/sc-<story>-epic-<epic>-<slug>`` -> matching
  ``feature/sc-<epic>-<slug>``;
* ``sync/sc-<epic>-main-<date>`` -> matching feature branch; and
* a feature branch -> ``main``.

Ordinary pull requests remain unaffected.  The explicit epic marker on story
branches is what lets CI prove ownership without trusting mutable PR prose or
requiring Shortcut credentials on a runner.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping


FEATURE_RE = re.compile(
    r"feature/sc-(?P<epic>[1-9][0-9]*)-(?P<slug>[a-z0-9]+(?:-[a-z0-9]+)*)"
)
STORY_RE = re.compile(
    r"story/sc-(?P<story>[1-9][0-9]*)-epic-(?P<epic>[1-9][0-9]*)-"
    r"(?P<slug>[a-z0-9]+(?:-[a-z0-9]+)*)"
)
SYNC_RE = re.compile(
    r"sync/sc-(?P<epic>[1-9][0-9]*)-main-(?P<day>[0-9]{4}-[0-9]{2}-[0-9]{2})"
)
SHA_RE = re.compile(r"[0-9a-f]{40}")

PROTECTED_PREFIXES = ("feature/", "sync/")
PULL_REQUEST_ACTIONS = frozenset({"opened", "reopened", "synchronize"})


class PolicyError(ValueError):
    """An event is malformed or violates the feature-epic policy."""


@dataclass(frozen=True)
class TrainBranch:
    kind: str
    epic: int


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PolicyError(f"{field} must be an object")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise PolicyError(f"{field} must be a non-empty string without surrounding whitespace")
    return value


def _nested_mapping(payload: Mapping[str, Any], *path: str) -> Mapping[str, Any]:
    value: Any = payload
    traversed: list[str] = []
    for component in path:
        traversed.append(component)
        current = _mapping(value, ".".join(traversed[:-1]) or "event")
        if component not in current:
            raise PolicyError(f"{'.'.join(traversed)} is missing")
        value = current[component]
    return _mapping(value, ".".join(path))


def _nested_string(payload: Mapping[str, Any], *path: str) -> str:
    value: Any = payload
    traversed: list[str] = []
    for component in path:
        traversed.append(component)
        current = _mapping(value, ".".join(traversed[:-1]) or "event")
        if component not in current:
            raise PolicyError(f"{'.'.join(traversed)} is missing")
        value = current[component]
    return _string(value, ".".join(path))


def _same_repository(actual: str, expected: str, field: str) -> None:
    if actual.casefold() != expected.casefold():
        raise PolicyError(f"{field} must be {expected!r}, found {actual!r}")


def _event_repository(payload: Mapping[str, Any], expected: str | None) -> str:
    repository = _nested_string(payload, "repository", "full_name")
    if expected is not None:
        expected = _string(expected, "repository")
        _same_repository(repository, expected, "repository.full_name")
        return expected
    return repository


def _parse_train_branch(branch: str) -> TrainBranch | None:
    feature = FEATURE_RE.fullmatch(branch)
    if feature:
        return TrainBranch("feature", int(feature.group("epic")))

    story = STORY_RE.fullmatch(branch)
    if story:
        return TrainBranch("story", int(story.group("epic")))

    sync = SYNC_RE.fullmatch(branch)
    if sync:
        try:
            date.fromisoformat(sync.group("day"))
        except ValueError as error:
            raise PolicyError(f"sync branch has an invalid calendar date: {branch!r}") from error
        return TrainBranch("sync", int(sync.group("epic")))

    looks_protected = branch.startswith(PROTECTED_PREFIXES) or (
        branch.startswith("story/") and "-epic-" in branch
    )
    if looks_protected:
        raise PolicyError(
            "malformed feature-epic branch; expected feature/sc-<epic>-<slug>, "
            "story/sc-<story>-epic-<epic>-<slug>, or "
            f"sync/sc-<epic>-main-<YYYY-MM-DD>: {branch!r}"
        )
    return None


def _validate_pull_request(payload: Mapping[str, Any], repository: str) -> str:
    action = _nested_string(payload, "action")
    if action not in PULL_REQUEST_ACTIONS:
        raise PolicyError(
            f"unsupported pull_request action {action!r}; expected one of "
            f"{sorted(PULL_REQUEST_ACTIONS)!r}"
        )

    pull_request = _nested_mapping(payload, "pull_request")
    head = _nested_mapping(pull_request, "head")
    base = _nested_mapping(pull_request, "base")
    head_ref = _nested_string(head, "ref")
    base_ref = _nested_string(base, "ref")

    head_train = _parse_train_branch(head_ref)
    base_train = _parse_train_branch(base_ref)
    protected_train = head_train is not None or base_train is not None
    if not protected_train:
        return f"ordinary pull request {head_ref!r} -> {base_ref!r}; no feature-epic policy applies"

    head_repository = _nested_string(head, "repo", "full_name")
    base_repository = _nested_string(base, "repo", "full_name")
    _same_repository(head_repository, repository, "pull_request.head.repo.full_name")
    _same_repository(base_repository, repository, "pull_request.base.repo.full_name")

    if head_train is None:
        raise PolicyError(
            f"feature branch {base_ref!r} accepts only an epic-bearing story "
            "or matching sync branch"
        )

    if head_train.kind == "feature":
        if base_ref != "main":
            raise PolicyError(
                f"feature branch {head_ref!r} may target only 'main', not {base_ref!r}"
            )
        return f"feature epic sc-{head_train.epic} targets main"

    if head_train.kind in {"story", "sync"}:
        if base_train is None or base_train.kind != "feature":
            raise PolicyError(
                f"{head_train.kind} branch {head_ref!r} must target its matching feature branch"
            )
        if head_train.epic != base_train.epic:
            raise PolicyError(
                f"epic mismatch: {head_ref!r} belongs to sc-{head_train.epic}, "
                f"but {base_ref!r} belongs to sc-{base_train.epic}"
            )
        return f"{head_train.kind} branch targets matching feature epic sc-{head_train.epic}"

    raise AssertionError(f"unhandled train branch kind: {head_train.kind}")


def _branch_from_full_ref(ref: str, field: str) -> str:
    prefix = "refs/heads/"
    if not ref.startswith(prefix) or len(ref) == len(prefix):
        raise PolicyError(f"{field} must be a full branch ref under {prefix!r}, found {ref!r}")
    return ref.removeprefix(prefix)


def _validate_merge_group(payload: Mapping[str, Any]) -> str:
    action = _nested_string(payload, "action")
    if action != "checks_requested":
        raise PolicyError(
            f"unsupported merge_group action {action!r}; expected 'checks_requested'"
        )

    merge_group = _nested_mapping(payload, "merge_group")
    head_ref = _nested_string(merge_group, "head_ref")
    base_ref = _nested_string(merge_group, "base_ref")
    head_sha = _nested_string(merge_group, "head_sha")
    base_sha = _nested_string(merge_group, "base_sha")
    if SHA_RE.fullmatch(head_sha) is None:
        raise PolicyError("merge_group.head_sha must be a lowercase 40-hex commit")
    if SHA_RE.fullmatch(base_sha) is None:
        raise PolicyError("merge_group.base_sha must be a lowercase 40-hex commit")

    base_branch = _branch_from_full_ref(base_ref, "merge_group.base_ref")
    head_branch = _branch_from_full_ref(head_ref, "merge_group.head_ref")
    base_train = _parse_train_branch(base_branch)
    if base_branch != "main" and (
        base_train is None or base_train.kind != "feature"
    ):
        raise PolicyError(
            "merge queues governed by this workflow may target only 'main' or a valid "
            f"feature-epic branch, found {base_branch!r}"
        )

    expected_prefix = f"gh-readonly-queue/{base_branch}/"
    if not head_branch.startswith(expected_prefix):
        raise PolicyError(
            f"merge_group.head_ref must be the queue ref for {base_branch!r}; "
            f"expected prefix {expected_prefix!r}, found {head_branch!r}"
        )
    if head_branch == expected_prefix:
        raise PolicyError("merge_group.head_ref is missing its queue entry suffix")

    # merge_group does not expose the source PR ref. The corresponding pull_request run proves
    # source topology, while this run proves that GitHub constructed a local queue ref for the
    # only permitted targets.
    return f"merge queue ref targets {base_branch!r}"


def _validate_push(payload: Mapping[str, Any]) -> str:
    ref = _nested_string(payload, "ref")
    if ref == "refs/heads/main":
        return "main push is outside pull-request topology"
    if ref.startswith("refs/tags/") and len(ref) > len("refs/tags/"):
        return "tag push is outside pull-request topology"
    raise PolicyError(
        f"branch push {ref!r} is not an allowed CI event; feature trains merge "
        "through pull requests"
    )


def validate_event(
    event_name: str,
    payload: Mapping[str, Any],
    *,
    repository: str | None = None,
) -> str:
    """Validate a GitHub event and return a human-readable acceptance reason."""

    event_name = _string(event_name, "event_name")
    payload = _mapping(payload, "event")
    resolved_repository = _event_repository(payload, repository)

    if event_name == "pull_request":
        return _validate_pull_request(payload, resolved_repository)
    if event_name == "merge_group":
        return _validate_merge_group(payload)
    if event_name == "push":
        return _validate_push(payload)
    if event_name == "workflow_dispatch":
        return "operator dispatch is outside pull-request topology"
    raise PolicyError(f"unsupported GitHub event {event_name!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--event-name",
        default=os.environ.get("GITHUB_EVENT_NAME"),
        help="GitHub event name (defaults to GITHUB_EVENT_NAME)",
    )
    parser.add_argument(
        "--event-path",
        type=Path,
        default=os.environ.get("GITHUB_EVENT_PATH"),
        help="GitHub event JSON path (defaults to GITHUB_EVENT_PATH)",
    )
    parser.add_argument(
        "--repository",
        default=os.environ.get("GITHUB_REPOSITORY"),
        help="Expected owner/repository (defaults to GITHUB_REPOSITORY)",
    )
    args = parser.parse_args(argv)

    if args.event_name is None:
        parser.error("--event-name or GITHUB_EVENT_NAME is required")
    if args.event_path is None:
        parser.error("--event-path or GITHUB_EVENT_PATH is required")
    if args.repository is None:
        parser.error("--repository or GITHUB_REPOSITORY is required")

    try:
        with args.event_path.open(encoding="utf-8") as event_file:
            payload = json.load(event_file)
        reason = validate_event(args.event_name, payload, repository=args.repository)
    except (OSError, json.JSONDecodeError, PolicyError) as error:
        print(f"::error title=Feature epic branch policy::{error}")
        return 1

    print(f"feature epic branch policy: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
