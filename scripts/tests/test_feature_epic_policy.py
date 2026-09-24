import functools
import json
import os
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from scripts.ci.feature_epic_policy import (
    PolicyError,
    resolve_local_commit,
    resolve_local_merge_parents,
    resolve_remote_feature_branch,
    validate_event,
)


REPOSITORY = "SceneWorks/inference"
SCRIPT = Path(__file__).resolve().parents[1] / "ci" / "feature_epic_policy.py"
FEATURE = "feature/sc-18304-pipeline-flexibility-mlx-perf"
STORY = "story/sc-18419-epic-18304-pipeline-flexibility-mlx-perf"
PR_HEAD_SHA = "1" * 40
PR_BASE_SHA = "2" * 40
PR_MERGE_SHA = "3" * 40
MERGE_GROUP_HEAD_SHA = "4" * 40
MERGE_GROUP_BASE_SHA = "5" * 40
QUEUE_SUFFIX_SHA = "6" * 40
# An annotated tag push names the tag OBJECT in `after`; GITHUB_SHA is the commit it peels to.
TAG_OBJECT_SHA = "a" * 40
TAGGED_COMMIT_SHA = "b" * 40
OTHER_COMMIT_SHA = "c" * 40
RELEASE_TAG = "runtime-2026.09.1-rc.0"
HERMETIC_GIT_ENV = {
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "feature-epic-policy-test",
    "GIT_AUTHOR_EMAIL": "feature-epic-policy-test@example.invalid",
    "GIT_COMMITTER_NAME": "feature-epic-policy-test",
    "GIT_COMMITTER_EMAIL": "feature-epic-policy-test@example.invalid",
}


def canonical_feature(epic: int) -> str:
    branches = {
        18304: FEATURE,
        18305: "feature/sc-18305-other",
        99999: "feature/sc-99999-other-epic",
    }
    return branches.get(epic, f"feature/sc-{epic}-canonical")


def pull_request_event(
    head: str,
    base: str,
    *,
    head_repository: str = REPOSITORY,
    base_repository: str = REPOSITORY,
) -> dict:
    return {
        "action": "synchronize",
        "repository": {"full_name": REPOSITORY},
        "pull_request": {
            "head": {
                "ref": head,
                "sha": PR_HEAD_SHA,
                "repo": {"full_name": head_repository},
            },
            "base": {
                "ref": base,
                "sha": PR_BASE_SHA,
                "repo": {"full_name": base_repository},
            },
            "merge_commit_sha": PR_MERGE_SHA,
        },
    }


def merge_group_event(base: str) -> dict:
    return {
        "action": "checks_requested",
        "repository": {"full_name": REPOSITORY},
        "merge_group": {
            "head_ref": f"refs/heads/gh-readonly-queue/{base}/pr-42-{QUEUE_SUFFIX_SHA}",
            "base_ref": f"refs/heads/{base}",
            "head_sha": MERGE_GROUP_HEAD_SHA,
            "base_sha": MERGE_GROUP_BASE_SHA,
        },
    }


def tag_push_event(after: str, tag: str = RELEASE_TAG) -> dict:
    return {
        "repository": {"full_name": REPOSITORY},
        "ref": f"refs/tags/{tag}",
        "after": after,
    }


def peel_fixture(obj: str) -> str:
    """Model `git rev-parse <obj>^{commit}` over a store holding one annotated tag object."""
    return {TAG_OBJECT_SHA: TAGGED_COMMIT_SHA}.get(obj, obj)


def absent_tag_object(obj: str) -> str:
    raise PolicyError(f"could not peel {obj} to a commit in the checkout")


def hermetic_git_env() -> dict[str, str]:
    return {**os.environ, **HERMETIC_GIT_ENV}


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=hermetic_git_env(),
    ).stdout.strip()


def annotated_tag_checkout(root: Path, tag: str) -> tuple[Path, str, str]:
    """Reproduce what actions/checkout (fetch-depth: 0) leaves behind on an annotated-tag push.

    It first fetches every tag, so the pushed tag object lands in the object store; then, because
    the local tag resolves to the tag object rather than GITHUB_SHA, it force-refetches the peeled
    commit onto the same ref. The ref now names the commit, but the tag object is still present.
    """
    origin = root / "origin"
    origin.mkdir()
    git(origin, "init", "-q")
    git(origin, "commit", "-q", "--allow-empty", "--no-verify", "-m", "release")
    git(origin, "tag", "-a", "-m", "release", tag)
    tag_object = git(origin, "rev-parse", f"refs/tags/{tag}")
    commit = git(origin, "rev-parse", f"refs/tags/{tag}^{{commit}}")

    checkout = root / "checkout"
    checkout.mkdir()
    git(checkout, "init", "-q")
    git(checkout, "remote", "add", "origin", str(origin))
    git(
        checkout,
        "fetch",
        "-q",
        "--prune",
        "origin",
        "+refs/heads/*:refs/remotes/origin/*",
        "+refs/tags/*:refs/tags/*",
    )
    git(checkout, "fetch", "-q", "--no-tags", "--prune", "origin", f"+{commit}:refs/tags/{tag}")
    git(checkout, "checkout", "-q", "--force", f"refs/tags/{tag}")
    return checkout, tag_object, commit


def active_sha_for(event_name: str, payload: dict) -> str | None:
    if event_name == "pull_request":
        return payload["pull_request"]["merge_commit_sha"]
    if event_name == "merge_group":
        return payload["merge_group"]["head_sha"]
    if event_name == "push":
        return payload["after"]
    return None


def validate(
    event_name: str,
    payload: dict,
    *,
    repository: str = REPOSITORY,
    active_sha: str | None = None,
    use_event_sha: bool = True,
    feature_resolver=canonical_feature,
    commit_parent_resolver=lambda _commit: (PR_BASE_SHA, PR_HEAD_SHA),
    commit_peeler=None,
) -> str:
    if use_event_sha:
        active_sha = active_sha_for(event_name, payload)
    return validate_event(
        event_name,
        payload,
        repository=repository,
        active_sha=active_sha,
        feature_resolver=feature_resolver,
        commit_parent_resolver=commit_parent_resolver,
        commit_peeler=commit_peeler,
    )


class FeatureEpicPolicyTests(unittest.TestCase):
    def assert_rejected(
        self,
        event_name: str,
        payload: dict,
        pattern: str,
        **kwargs,
    ) -> None:
        with self.assertRaisesRegex(PolicyError, pattern):
            validate(event_name, payload, **kwargs)

    def test_epic_bearing_story_targets_matching_feature(self) -> None:
        reason = validate(
            "pull_request",
            pull_request_event(
                STORY,
                "feature/sc-18304-pipeline-flexibility-mlx-perf",
            ),
        )
        self.assertIn("matching feature epic sc-18304", reason)

    def test_story_cannot_target_main_or_the_wrong_epic(self) -> None:
        story = STORY
        self.assert_rejected(
            "pull_request",
            pull_request_event(story, "main"),
            "must target its matching feature branch",
        )
        self.assert_rejected(
            "pull_request",
            pull_request_event(story, "feature/sc-99999-other-epic"),
            "epic mismatch",
        )

    def test_legacy_story_name_cannot_enter_a_feature_branch(self) -> None:
        self.assert_rejected(
            "pull_request",
            pull_request_event(
                "story/sc-18419-automate-feature-policy",
                "feature/sc-18304-pipeline-flexibility-mlx-perf",
            ),
            "accepts only an epic-bearing story",
        )

    def test_story_slug_must_match_the_live_canonical_feature_slug(self) -> None:
        self.assert_rejected(
            "pull_request",
            pull_request_event(
                "story/sc-18419-epic-18304-wrong-slug",
                FEATURE,
            ),
            "not the canonical feature slug",
        )

    def test_malformed_epic_bearing_story_fails_closed(self) -> None:
        self.assert_rejected(
            "pull_request",
            pull_request_event(
                "story/sc-18419-epic-18304-Bad_Slug",
                "feature/sc-18304-pipeline-flexibility-mlx-perf",
            ),
            "malformed feature-epic branch",
        )

    def test_epic_word_inside_an_ordinary_story_slug_is_not_reserved(self) -> None:
        reason = validate(
            "pull_request",
            pull_request_event("story/sc-18419-feature-epic-policy", "main"),
        )
        self.assertIn("ordinary pull request", reason)

    def test_sync_targets_only_its_matching_feature(self) -> None:
        sync = "sync/sc-18304-main-2026-08-10"
        validate(
            "pull_request",
            pull_request_event(sync, "feature/sc-18304-pipeline-flexibility-mlx-perf"),
        )
        self.assert_rejected(
            "pull_request",
            pull_request_event(sync, "feature/sc-18305-other"),
            "epic mismatch",
        )
        self.assert_rejected(
            "pull_request",
            pull_request_event(sync, "main"),
            "must target its matching feature branch",
        )

        validate(
            "pull_request",
            pull_request_event(
                "sync/sc-18304-main-2026-08-10-2",
                "feature/sc-18304-pipeline-flexibility-mlx-perf",
            ),
        )

    def test_sync_date_must_be_a_real_calendar_date(self) -> None:
        self.assert_rejected(
            "pull_request",
            pull_request_event(
                "sync/sc-18304-main-2026-02-30",
                "feature/sc-18304-pipeline-flexibility-mlx-perf",
            ),
            "invalid calendar date",
        )

    def test_feature_targets_main_only(self) -> None:
        feature = "feature/sc-18304-pipeline-flexibility-mlx-perf"
        validate(
            "pull_request",
            pull_request_event(feature, "main"),
        )
        for base in ("release/next", "feature/sc-18305-other"):
            with self.subTest(base=base):
                self.assert_rejected(
                    "pull_request",
                    pull_request_event(feature, base),
                    "may target only 'main'",
                )

    def test_duplicate_noncanonical_feature_slugs_are_rejected_everywhere(self) -> None:
        duplicate = "feature/sc-18304-duplicate"
        self.assert_rejected(
            "pull_request",
            pull_request_event(STORY, duplicate),
            "not the unique live canonical",
        )
        self.assert_rejected(
            "pull_request",
            pull_request_event(duplicate, "main"),
            "not the unique live canonical",
        )
        self.assert_rejected(
            "merge_group",
            merge_group_event(duplicate),
            "not the unique live canonical",
        )

    def test_protected_topology_requires_a_live_resolver(self) -> None:
        self.assert_rejected(
            "pull_request",
            pull_request_event(STORY, FEATURE),
            "live canonical feature-branch resolver is required",
            feature_resolver=None,
        )

    def test_protected_train_heads_must_come_from_this_repository(self) -> None:
        for head, base in (
            (STORY, FEATURE),
            (
                "sync/sc-18304-main-2026-08-10",
                "feature/sc-18304-pipeline-flexibility-mlx-perf",
            ),
            ("feature/sc-18304-pipeline-flexibility-mlx-perf", "main"),
        ):
            with self.subTest(head=head):
                self.assert_rejected(
                    "pull_request",
                    pull_request_event(head, base, head_repository="attacker/inference"),
                    "head.repo.full_name must be",
                )

    def test_protected_train_base_must_be_this_repository(self) -> None:
        self.assert_rejected(
            "pull_request",
            pull_request_event(
                STORY,
                "feature/sc-18304-pipeline-flexibility-mlx-perf",
                base_repository="other/inference",
            ),
            "base.repo.full_name must be",
        )

    def test_ordinary_pull_requests_remain_compatible_with_forks(self) -> None:
        reason = validate(
            "pull_request",
            pull_request_event(
                "fix/cuda-build",
                "main",
                head_repository="contributor/inference",
            ),
        )
        self.assertIn("ordinary pull request", reason)

    def test_pull_request_shape_and_action_fail_closed(self) -> None:
        event = pull_request_event("fix/cuda-build", "main")
        del event["pull_request"]["base"]["ref"]
        self.assert_rejected("pull_request", event, "ref is missing")

        event = pull_request_event("fix/cuda-build", "main")
        event["action"] = "closed"
        self.assert_rejected("pull_request", event, "unsupported pull_request action")

    def test_pull_request_binds_complete_commit_identity_to_active_sha(self) -> None:
        for container, field in (
            ("head", "sha"),
            ("base", "sha"),
            ("pull_request", "merge_commit_sha"),
        ):
            with self.subTest(missing=f"{container}.{field}"):
                event = pull_request_event("fix/cuda-build", "main")
                if container == "pull_request":
                    del event["pull_request"][field]
                else:
                    del event["pull_request"][container][field]
                self.assert_rejected(
                    "pull_request",
                    event,
                    f"{field} is missing",
                    use_event_sha=False,
                    active_sha=PR_MERGE_SHA,
                )

        event = pull_request_event("fix/cuda-build", "main")
        event["pull_request"]["base"]["sha"] = "A" * 40
        self.assert_rejected("pull_request", event, "lowercase 40-hex")

        event = pull_request_event("fix/cuda-build", "main")
        self.assert_rejected(
            "pull_request",
            event,
            "GITHUB_SHA is required",
            use_event_sha=False,
            active_sha=None,
        )
        self.assert_rejected(
            "pull_request",
            event,
            "exact payload base/head parents",
            use_event_sha=False,
            active_sha="7" * 40,
            commit_parent_resolver=lambda _commit: ("7" * 40, PR_HEAD_SHA),
        )
        self.assert_rejected(
            "pull_request",
            event,
            "GITHUB_SHA must be a lowercase 40-hex",
            use_event_sha=False,
            active_sha="not-a-commit",
        )

    def test_pull_request_binds_unavailable_payload_merge_sha_to_local_parents(self) -> None:
        for unavailable in (None, ""):
            with self.subTest(unavailable=unavailable):
                event = pull_request_event(STORY, FEATURE)
                event["pull_request"]["merge_commit_sha"] = unavailable
                reason = validate(
                    "pull_request",
                    event,
                    use_event_sha=False,
                    active_sha=PR_MERGE_SHA,
                )
                self.assertIn("story", reason)

        event = pull_request_event(STORY, FEATURE)
        event["pull_request"]["merge_commit_sha"] = None
        self.assert_rejected(
            "pull_request",
            event,
            "exact payload base/head parents",
            use_event_sha=False,
            active_sha=PR_MERGE_SHA,
            commit_parent_resolver=lambda _commit: ("7" * 40, PR_HEAD_SHA),
        )

        stale = pull_request_event(STORY, FEATURE)
        stale["pull_request"]["merge_commit_sha"] = "8" * 40
        reason = validate(
            "pull_request",
            stale,
            use_event_sha=False,
            active_sha=PR_MERGE_SHA,
        )
        self.assertIn("story", reason)

    def test_local_merge_parent_resolver_requires_exactly_two_parents(self) -> None:
        result = SimpleNamespace(returncode=0, stdout=f"{PR_BASE_SHA} {PR_HEAD_SHA}\n", stderr="")
        self.assertEqual(
            resolve_local_merge_parents(PR_MERGE_SHA, runner=lambda *args, **kwargs: result),
            (PR_BASE_SHA, PR_HEAD_SHA),
        )

        one_parent = SimpleNamespace(returncode=0, stdout=f"{PR_HEAD_SHA}\n", stderr="")
        with self.assertRaisesRegex(PolicyError, "two-parent test merge"):
            resolve_local_merge_parents(
                PR_MERGE_SHA,
                runner=lambda *args, **kwargs: one_parent,
            )

    def test_merge_group_accepts_main_and_feature_queue_refs(self) -> None:
        for base in ("main", "feature/sc-18304-pipeline-flexibility-mlx-perf"):
            with self.subTest(base=base):
                reason = validate("merge_group", merge_group_event(base))
                self.assertIn(base, reason)

    def test_merge_group_rejects_wrong_queue_ref_and_unsupported_target(self) -> None:
        event = merge_group_event("feature/sc-18304-pipeline-flexibility-mlx-perf")
        event["merge_group"]["head_ref"] = (
            f"refs/heads/gh-readonly-queue/main/pr-42-{QUEUE_SUFFIX_SHA}"
        )
        self.assert_rejected("merge_group", event, "must be the queue ref")
        self.assert_rejected(
            "merge_group",
            merge_group_event("release/next"),
            "may target only 'main' or a valid feature-epic branch",
        )

        event = merge_group_event("main")
        event["merge_group"]["head_ref"] = (
            "refs/heads/gh-readonly-queue/main/../main/"
            f"pr-42-{QUEUE_SUFFIX_SHA}"
        )
        self.assert_rejected("merge_group", event, "must be the queue ref")

    def test_merge_group_requires_complete_refs_and_lowercase_shas(self) -> None:
        event = merge_group_event("main")
        event["merge_group"]["base_ref"] = "main"
        self.assert_rejected("merge_group", event, "must be a full branch ref")

        event = merge_group_event("main")
        event["merge_group"]["head_sha"] = "A" * 40
        self.assert_rejected("merge_group", event, "lowercase 40-hex")

        event = merge_group_event("main")
        del event["merge_group"]["base_sha"]
        self.assert_rejected("merge_group", event, "base_sha is missing")

        event = merge_group_event("main")
        self.assert_rejected(
            "merge_group",
            event,
            "GITHUB_SHA is required",
            use_event_sha=False,
            active_sha=None,
        )
        self.assert_rejected(
            "merge_group",
            event,
            "GITHUB_SHA must equal merge_group.head_sha",
            use_event_sha=False,
            active_sha="7" * 40,
        )

    def test_main_and_tag_pushes_do_not_break_the_gate(self) -> None:
        for ref in ("refs/heads/main", "refs/tags/runtime-0.9.0"):
            with self.subTest(ref=ref):
                payload = {
                    "repository": {"full_name": REPOSITORY},
                    "ref": ref,
                    "after": PR_MERGE_SHA,
                }
                reason = validate(
                    "push",
                    payload,
                )
                self.assertIn("push", reason)

        self.assert_rejected(
            "push",
            {
                "repository": {"full_name": REPOSITORY},
                "ref": "refs/heads/feature/sc-18304-pipeline-flexibility-mlx-perf",
                "after": PR_MERGE_SHA,
            },
            "merge through pull requests",
        )

    def test_push_binds_after_to_active_sha(self) -> None:
        for ref in ("refs/heads/main", "refs/tags/runtime-0.9.0"):
            with self.subTest(ref=ref):
                event = {
                    "repository": {"full_name": REPOSITORY},
                    "ref": ref,
                    "after": PR_MERGE_SHA,
                }
                self.assert_rejected(
                    "push",
                    event,
                    "GITHUB_SHA is required",
                    use_event_sha=False,
                    active_sha=None,
                    commit_peeler=peel_fixture,
                )
                del event["after"]
                self.assert_rejected(
                    "push",
                    event,
                    "after is missing",
                    use_event_sha=False,
                    active_sha=PR_MERGE_SHA,
                    commit_peeler=peel_fixture,
                )

        self.assert_rejected(
            "push",
            {
                "repository": {"full_name": REPOSITORY},
                "ref": "refs/heads/main",
                "after": PR_MERGE_SHA,
            },
            "GITHUB_SHA must equal after",
            use_event_sha=False,
            active_sha="7" * 40,
        )
        # A branch push names a commit in `after`, so nothing is peeled: a tag object that
        # happens to peel to GITHUB_SHA must not satisfy a main push.
        self.assert_rejected(
            "push",
            {
                "repository": {"full_name": REPOSITORY},
                "ref": "refs/heads/main",
                "after": TAG_OBJECT_SHA,
            },
            "GITHUB_SHA must equal after",
            use_event_sha=False,
            active_sha=TAGGED_COMMIT_SHA,
            commit_peeler=peel_fixture,
        )

    def test_annotated_tag_push_binds_the_peeled_commit_to_active_sha(self) -> None:
        # Runs 35812806857 (runtime-2026.09.0) and 35796817963 (runtime-2026.09.0-rc.8) refused
        # exactly this shape: `after` is the tag object, GITHUB_SHA the commit it peels to.
        reason = validate(
            "push",
            tag_push_event(TAG_OBJECT_SHA),
            use_event_sha=False,
            active_sha=TAGGED_COMMIT_SHA,
            commit_peeler=peel_fixture,
        )
        self.assertIn("tag push", reason)

    def test_lightweight_tag_push_binds_after_directly_without_a_local_peel(self) -> None:
        for peeler in (None, peel_fixture):
            with self.subTest(peeler=getattr(peeler, "__name__", None)):
                reason = validate(
                    "push",
                    tag_push_event(TAGGED_COMMIT_SHA),
                    use_event_sha=False,
                    active_sha=TAGGED_COMMIT_SHA,
                    commit_peeler=peeler,
                )
                self.assertIn("tag push", reason)

    def test_tag_push_refuses_a_revision_the_pushed_tag_does_not_name(self) -> None:
        names_other = "GITHUB_SHA must equal the commit that tag push after names"
        cases = (
            ("annotated tag peels to another commit", TAG_OBJECT_SHA, OTHER_COMMIT_SHA,
             peel_fixture, names_other),
            ("lightweight tag names another commit", TAGGED_COMMIT_SHA, OTHER_COMMIT_SHA,
             peel_fixture, names_other),
            ("no peeler to prove what a tag object names", TAG_OBJECT_SHA, TAGGED_COMMIT_SHA,
             None, "no local commit peeler"),
            ("tag object absent from the checkout", TAG_OBJECT_SHA, TAGGED_COMMIT_SHA,
             absent_tag_object, "could not peel"),
        )
        for label, after, active_sha, peeler, pattern in cases:
            with self.subTest(label):
                self.assert_rejected(
                    "push",
                    tag_push_event(after),
                    pattern,
                    use_event_sha=False,
                    active_sha=active_sha,
                    commit_peeler=peeler,
                )

    def test_local_commit_peeler_peels_tag_objects_and_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkout, tag_object, commit = annotated_tag_checkout(Path(temp_dir), RELEASE_TAG)
            runner = functools.partial(subprocess.run, cwd=checkout, env=hermetic_git_env())
            self.assertNotEqual(tag_object, commit)
            self.assertEqual(resolve_local_commit(tag_object, runner=runner), commit)
            self.assertEqual(resolve_local_commit(commit, runner=runner), commit)

            tree = git(checkout, "rev-parse", f"{commit}^{{tree}}")
            for label, obj in (("absent object", OTHER_COMMIT_SHA), ("tree", tree)):
                with self.subTest(label):
                    with self.assertRaisesRegex(PolicyError, "could not peel"):
                        resolve_local_commit(obj, runner=runner)

    def test_cli_binds_an_annotated_release_tag_push_through_the_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkout, tag_object, commit = annotated_tag_checkout(root, RELEASE_TAG)
            # actions/checkout has repointed the local ref at the commit; only the object store
            # still carries the tag object that `after` names.
            self.assertEqual(git(checkout, "cat-file", "-t", f"refs/tags/{RELEASE_TAG}"), "commit")
            self.assertEqual(git(checkout, "cat-file", "-t", tag_object), "tag")
            other = git(checkout, "commit-tree", "-m", "other", f"{commit}^{{tree}}")
            event_path = root / "event.json"
            event_path.write_text(json.dumps(tag_push_event(tag_object)), encoding="utf-8")

            for active_sha, expected_code, expected_text in (
                (commit, 0, "feature epic branch policy: tag push"),
                (other, 1, "::error title=Feature epic branch policy::"),
            ):
                with self.subTest(expected_code=expected_code):
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(SCRIPT),
                            "--event-name",
                            "push",
                            "--event-path",
                            str(event_path),
                            "--repository",
                            REPOSITORY,
                            "--active-sha",
                            active_sha,
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        cwd=checkout,
                        env=hermetic_git_env(),
                    )
                    self.assertEqual(
                        result.returncode, expected_code, result.stdout + result.stderr
                    )
                    self.assertIn(expected_text, result.stdout)

    def test_dispatch_is_allowed_but_unknown_events_fail_closed(self) -> None:
        payload = {"repository": {"full_name": REPOSITORY}}
        reason = validate(
            "workflow_dispatch",
            payload,
            use_event_sha=False,
            active_sha=None,
        )
        self.assertIn("operator dispatch", reason)
        self.assert_rejected("schedule", payload, "unsupported GitHub event")

    def test_repository_context_must_match_event_payload(self) -> None:
        event = pull_request_event("fix/cuda-build", "main")
        with self.assertRaisesRegex(PolicyError, "repository.full_name must be"):
            validate("pull_request", event, repository="other/inference")

    def test_cli_reports_success_and_annotations_for_policy_errors(self) -> None:
        valid = {"repository": {"full_name": REPOSITORY}}
        invalid = pull_request_event("fix/ordinary", "main")
        invalid["pull_request"]["merge_commit_sha"] = "not-a-commit"

        for event_name, payload, expected_code, expected_text in (
            ("workflow_dispatch", valid, 0, "feature epic branch policy:"),
            ("pull_request", invalid, 1, "::error title=Feature epic branch policy::"),
        ):
            with self.subTest(expected_code=expected_code):
                with tempfile.TemporaryDirectory() as temp_dir:
                    event_path = Path(temp_dir) / "event.json"
                    event_path.write_text(json.dumps(payload), encoding="utf-8")
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(SCRIPT),
                            "--event-name",
                            event_name,
                            "--event-path",
                            str(event_path),
                            "--repository",
                            REPOSITORY,
                            "--active-sha",
                            PR_MERGE_SHA,
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                    )
                self.assertEqual(result.returncode, expected_code, result.stdout + result.stderr)
                self.assertIn(expected_text, result.stdout)

    def test_remote_resolver_uses_fixed_origin_and_safe_argument_vector(self) -> None:
        calls = []

        def runner(command, **kwargs):
            calls.append((command, kwargs))
            return SimpleNamespace(
                returncode=0,
                stdout=f"{PR_HEAD_SHA}\trefs/heads/{FEATURE}\n",
                stderr="",
            )

        self.assertEqual(resolve_remote_feature_branch(18304, runner=runner), FEATURE)
        self.assertEqual(
            calls[0][0],
            [
                "git",
                "ls-remote",
                "--heads",
                "origin",
                "refs/heads/feature/sc-18304-*",
            ],
        )
        self.assertNotIn("shell", calls[0][1])
        self.assertEqual(calls[0][1]["timeout"], 30)

    def test_remote_resolver_fails_on_zero_multiple_or_noncanonical_refs(self) -> None:
        cases = (
            ("", "no live feature branch"),
            (
                f"{PR_HEAD_SHA}\trefs/heads/{FEATURE}\n"
                f"{PR_BASE_SHA}\trefs/heads/feature/sc-18304-duplicate\n",
                "multiple live feature branches",
            ),
            (
                f"{PR_HEAD_SHA}\trefs/heads/{FEATURE}/nested\n",
                "invalid feature ref",
            ),
            (
                f"{PR_HEAD_SHA}\trefs/heads/{FEATURE}\n"
                f"{PR_BASE_SHA}\trefs/heads/{FEATURE}\n",
                "divergent commits",
            ),
        )
        for stdout, pattern in cases:
            with self.subTest(pattern=pattern):
                runner = lambda command, **kwargs: SimpleNamespace(
                    returncode=0, stdout=stdout, stderr=""
                )
                with self.assertRaisesRegex(PolicyError, pattern):
                    resolve_remote_feature_branch(18304, runner=runner)


if __name__ == "__main__":
    unittest.main()
