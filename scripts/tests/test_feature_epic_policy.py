import json
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from scripts.ci.feature_epic_policy import PolicyError, validate_event


REPOSITORY = "SceneWorks/inference"
SCRIPT = Path(__file__).resolve().parents[1] / "ci" / "feature_epic_policy.py"
HEAD_SHA = "1" * 40
BASE_SHA = "2" * 40


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
            "head": {"ref": head, "repo": {"full_name": head_repository}},
            "base": {"ref": base, "repo": {"full_name": base_repository}},
        },
    }


def merge_group_event(base: str) -> dict:
    return {
        "action": "checks_requested",
        "repository": {"full_name": REPOSITORY},
        "merge_group": {
            "head_ref": f"refs/heads/gh-readonly-queue/{base}/pr-42-{HEAD_SHA}",
            "base_ref": f"refs/heads/{base}",
            "head_sha": HEAD_SHA,
            "base_sha": BASE_SHA,
        },
    }


class FeatureEpicPolicyTests(unittest.TestCase):
    def assert_rejected(self, event_name: str, payload: dict, pattern: str) -> None:
        with self.assertRaisesRegex(PolicyError, pattern):
            validate_event(event_name, payload, repository=REPOSITORY)

    def test_epic_bearing_story_targets_matching_feature(self) -> None:
        reason = validate_event(
            "pull_request",
            pull_request_event(
                "story/sc-18419-epic-18304-automate-feature-policy",
                "feature/sc-18304-pipeline-flexibility-mlx-perf",
            ),
            repository=REPOSITORY,
        )
        self.assertIn("matching feature epic sc-18304", reason)

    def test_story_cannot_target_main_or_the_wrong_epic(self) -> None:
        story = "story/sc-18419-epic-18304-automate-feature-policy"
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

    def test_malformed_epic_bearing_story_fails_closed(self) -> None:
        self.assert_rejected(
            "pull_request",
            pull_request_event(
                "story/sc-18419-epic-18304-Bad_Slug",
                "feature/sc-18304-pipeline-flexibility-mlx-perf",
            ),
            "malformed feature-epic branch",
        )

    def test_sync_targets_only_its_matching_feature(self) -> None:
        sync = "sync/sc-18304-main-2026-08-10"
        validate_event(
            "pull_request",
            pull_request_event(sync, "feature/sc-18304-pipeline-flexibility-mlx-perf"),
            repository=REPOSITORY,
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
        validate_event(
            "pull_request",
            pull_request_event(feature, "main"),
            repository=REPOSITORY,
        )
        for base in ("release/next", "feature/sc-18305-other"):
            with self.subTest(base=base):
                self.assert_rejected(
                    "pull_request",
                    pull_request_event(feature, base),
                    "may target only 'main'",
                )

    def test_protected_train_heads_must_come_from_this_repository(self) -> None:
        for head, base in (
            (
                "story/sc-18419-epic-18304-automate-feature-policy",
                "feature/sc-18304-pipeline-flexibility-mlx-perf",
            ),
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
                "story/sc-18419-epic-18304-automate-feature-policy",
                "feature/sc-18304-pipeline-flexibility-mlx-perf",
                base_repository="other/inference",
            ),
            "base.repo.full_name must be",
        )

    def test_ordinary_pull_requests_remain_compatible_with_forks(self) -> None:
        reason = validate_event(
            "pull_request",
            pull_request_event(
                "fix/cuda-build",
                "main",
                head_repository="contributor/inference",
            ),
            repository=REPOSITORY,
        )
        self.assertIn("ordinary pull request", reason)

    def test_pull_request_shape_and_action_fail_closed(self) -> None:
        event = pull_request_event("fix/cuda-build", "main")
        del event["pull_request"]["base"]["ref"]
        self.assert_rejected("pull_request", event, "ref is missing")

        event = pull_request_event("fix/cuda-build", "main")
        event["action"] = "closed"
        self.assert_rejected("pull_request", event, "unsupported pull_request action")

    def test_merge_group_accepts_main_and_feature_queue_refs(self) -> None:
        for base in ("main", "feature/sc-18304-pipeline-flexibility-mlx-perf"):
            with self.subTest(base=base):
                reason = validate_event(
                    "merge_group", merge_group_event(base), repository=REPOSITORY
                )
                self.assertIn(base, reason)

    def test_merge_group_rejects_wrong_queue_ref_and_unsupported_target(self) -> None:
        event = merge_group_event("feature/sc-18304-pipeline-flexibility-mlx-perf")
        event["merge_group"]["head_ref"] = (
            f"refs/heads/gh-readonly-queue/main/pr-42-{HEAD_SHA}"
        )
        self.assert_rejected("merge_group", event, "must be the queue ref")
        self.assert_rejected(
            "merge_group",
            merge_group_event("release/next"),
            "may target only 'main' or a valid feature-epic branch",
        )

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

    def test_main_and_tag_pushes_do_not_break_the_gate(self) -> None:
        for ref in ("refs/heads/main", "refs/tags/runtime-0.9.0"):
            with self.subTest(ref=ref):
                reason = validate_event(
                    "push",
                    {"repository": {"full_name": REPOSITORY}, "ref": ref},
                    repository=REPOSITORY,
                )
                self.assertIn("push", reason)

        self.assert_rejected(
            "push",
            {
                "repository": {"full_name": REPOSITORY},
                "ref": "refs/heads/feature/sc-18304-pipeline-flexibility-mlx-perf",
            },
            "merge through pull requests",
        )

    def test_dispatch_is_allowed_but_unknown_events_fail_closed(self) -> None:
        payload = {"repository": {"full_name": REPOSITORY}}
        validate_event("workflow_dispatch", payload, repository=REPOSITORY)
        self.assert_rejected("schedule", payload, "unsupported GitHub event")

    def test_repository_context_must_match_event_payload(self) -> None:
        event = pull_request_event("fix/cuda-build", "main")
        with self.assertRaisesRegex(PolicyError, "repository.full_name must be"):
            validate_event("pull_request", event, repository="other/inference")

    def test_cli_reports_success_and_annotations_for_policy_errors(self) -> None:
        valid = pull_request_event(
            "story/sc-18419-epic-18304-automate-feature-policy",
            "feature/sc-18304-pipeline-flexibility-mlx-perf",
        )
        invalid = deepcopy(valid)
        invalid["pull_request"]["base"]["ref"] = "main"

        for payload, expected_code, expected_text in (
            (valid, 0, "feature epic branch policy:"),
            (invalid, 1, "::error title=Feature epic branch policy::"),
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
                            "pull_request",
                            "--event-path",
                            str(event_path),
                            "--repository",
                            REPOSITORY,
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                    )
                self.assertEqual(result.returncode, expected_code, result.stdout + result.stderr)
                self.assertIn(expected_text, result.stdout)


if __name__ == "__main__":
    unittest.main()
