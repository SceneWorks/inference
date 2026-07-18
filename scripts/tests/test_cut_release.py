import unittest

from scripts.release.build_release import TAG_PATTERN as BUILD_TAG_PATTERN
from scripts.release.cut_release import (
    TAG_PATTERN,
    AmbiguousBump,
    Command,
    Version,
    WriteFile,
    branch_name,
    next_candidate,
    parse,
    plan_prepare,
    plan_promote,
    plan_release,
    pr_title,
    promote_target,
    version_file_content,
)


def versions(*tags: str) -> list[Version]:
    return [v for v in (parse(tag) for tag in tags) if v is not None]


JULY = (2026, 7)
# The real tag history: latest is runtime-2026.07.7 (final), with an abandoned .5-rc.0.
REAL = (
    "runtime-2026.07.0", "runtime-2026.07.1", "runtime-2026.07.1-rc.0",
    "runtime-2026.07.2", "runtime-2026.07.3", "runtime-2026.07.4",
    "runtime-2026.07.5-rc.0", "runtime-2026.07.6", "runtime-2026.07.6-rc.0",
    "runtime-2026.07.7", "runtime-2026.07.7-rc.0",
)


class VersionAlgebraTests(unittest.TestCase):
    def test_parses_final_and_rc(self) -> None:
        self.assertEqual(parse("runtime-2026.07.7"), Version(2026, 7, 7, None))
        self.assertEqual(parse("runtime-2026.07.8-rc.0"), Version(2026, 7, 8, 0))

    def test_rejects_non_release_tags(self) -> None:
        self.assertIsNone(parse("v1.2.3"))
        self.assertIsNone(parse("runtime-2026.7.0"))  # month must be zero-padded to 2 digits
        self.assertIsNone(parse("2026.07.0"))         # missing runtime- prefix

    def test_final_sorts_above_its_candidates(self) -> None:
        self.assertGreater(parse("runtime-2026.07.7").sort_key, parse("runtime-2026.07.7-rc.0").sort_key)

    def test_finalized_top_yields_next_patch_rc0(self) -> None:
        candidate, _ = next_candidate(versions(*REAL), JULY)
        self.assertEqual(candidate.format(), "runtime-2026.07.8-rc.0")

    def test_first_release_of_a_new_month(self) -> None:
        candidate, reason = next_candidate(versions(*REAL), (2026, 8))
        self.assertEqual(candidate.format(), "runtime-2026.08.0-rc.0")
        self.assertIn("first release", reason)

    def test_no_tags_at_all(self) -> None:
        candidate, _ = next_candidate([], JULY)
        self.assertEqual(candidate.format(), "runtime-2026.07.0-rc.0")

    def test_unfinalized_top_is_ambiguous_without_a_choice(self) -> None:
        inflight = versions("runtime-2026.07.7", "runtime-2026.07.8-rc.0")
        with self.assertRaises(AmbiguousBump):
            next_candidate(inflight, JULY)

    def test_respin_and_new_patch_resolve_the_ambiguity(self) -> None:
        inflight = versions("runtime-2026.07.7", "runtime-2026.07.8-rc.0")
        self.assertEqual(next_candidate(inflight, JULY, respin=True)[0].format(), "runtime-2026.07.8-rc.1")
        self.assertEqual(next_candidate(inflight, JULY, new_patch=True)[0].format(), "runtime-2026.07.9-rc.0")


class PreparePlanTests(unittest.TestCase):
    TAG = "runtime-2026.07.8-rc.0"

    def test_records_the_version_file(self) -> None:
        writes = [s for s in plan_prepare(self.TAG, "main") if isinstance(s, WriteFile)]
        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0].path.name, "VERSION")
        self.assertEqual(writes[0].content, f"{self.TAG}\n")
        self.assertEqual(version_file_content(self.TAG), f"{self.TAG}\n")

    def test_branch_commit_push_and_pr_in_order(self) -> None:
        cmds = [s for s in plan_prepare(self.TAG, "main") if isinstance(s, Command)]
        joined = [" ".join(c.argv) for c in cmds]
        # branch off origin/main, then push that branch, then open the PR against main.
        self.assertIn(f"git switch -c {branch_name(self.TAG)} origin/main", joined)
        self.assertIn(f"git push -u origin {branch_name(self.TAG)}", joined)
        pr = next(c for c in cmds if c.argv[:3] == ["gh", "pr", "create"])
        self.assertIn("--base", pr.argv)
        self.assertIn(pr_title(self.TAG), pr.argv)
        # no tag *creation/push* sneaks into the prepare plan (a `git fetch --tags` is fine)
        self.assertFalse(any(c.argv[:2] == ["git", "tag"] for c in cmds))
        self.assertFalse(any(c.argv[:2] == ["git", "push"] and "origin" in c.argv
                             and any(a.startswith("runtime-") for a in c.argv) for c in cmds))

    def test_no_fetch_drops_the_fetch_step(self) -> None:
        with_fetch = plan_prepare(self.TAG, "main", fetch=True)
        without = plan_prepare(self.TAG, "main", fetch=False)
        self.assertEqual(len(with_fetch) - len(without), 1)


class ReleasePlanTests(unittest.TestCase):
    TAG = "runtime-2026.07.8-rc.0"

    def test_builds_verifies_then_tags_and_pushes_in_order(self) -> None:
        joined = [" ".join(s.argv) for s in plan_release(self.TAG)]
        build = next(i for i, j in enumerate(joined) if "build_release.py" in j and self.TAG in j)
        verify = next(i for i, j in enumerate(joined) if "verify_release.py" in j)
        tag = next(i for i, j in enumerate(joined) if j.startswith("git tag -a"))
        push = next(i for i, j in enumerate(joined) if j == f"git push origin {self.TAG}")
        self.assertLess(build, verify)
        self.assertLess(verify, tag)
        self.assertLess(tag, push)

    def test_offline_propagates_to_build_and_verify(self) -> None:
        joined = [" ".join(s.argv) for s in plan_release(self.TAG, offline=True)]
        self.assertTrue(all("--offline" in j for j in joined if "release.py" in j))


class PromoteTests(unittest.TestCase):
    RC = "runtime-2026.07.8-rc.0"
    FINAL = "runtime-2026.07.8"
    REV = "0123456789abcdef0123456789abcdef01234567"

    def test_promote_target_strips_the_rc_suffix(self) -> None:
        self.assertEqual(promote_target(self.RC), self.FINAL)
        self.assertEqual(promote_target("runtime-2026.07.8-rc.3"), self.FINAL)

    def test_promote_target_rejects_a_final_or_non_tag(self) -> None:
        with self.assertRaises(ValueError):
            promote_target(self.FINAL)  # already final
        with self.assertRaises(ValueError):
            promote_target("v1.2.3")

    def test_plan_builds_from_the_rc_revision_then_tags_and_pushes(self) -> None:
        joined = [" ".join(s.argv) for s in plan_promote(self.RC, self.FINAL, self.REV)]
        build = next(i for i, j in enumerate(joined) if "build_release.py" in j)
        verify = next(i for i, j in enumerate(joined) if "verify_release.py" in j)
        tag = next(i for i, j in enumerate(joined) if j.startswith("git tag -a"))
        push = next(i for i, j in enumerate(joined) if j == f"git push origin {self.FINAL}")
        self.assertLess(build, verify)
        self.assertLess(verify, tag)   # build first so a failure leaves no dangling tag
        self.assertLess(tag, push)
        # the final bundle + tag are pinned to the rc's exact revision, using the FINAL tag name
        self.assertIn(f"--tag {self.FINAL} --source-ref {self.REV}", joined[build])
        self.assertIn(f"git tag -a {self.FINAL} {self.REV}", joined[tag])

    def test_offline_propagates(self) -> None:
        joined = [" ".join(s.argv) for s in plan_promote(self.RC, self.FINAL, self.REV, offline=True)]
        self.assertTrue(all("--offline" in j for j in joined if "release.py" in j))


class DriftGuardTests(unittest.TestCase):
    def test_tag_pattern_matches_the_release_builder(self) -> None:
        self.assertEqual(TAG_PATTERN.pattern, BUILD_TAG_PATTERN.pattern)


if __name__ == "__main__":
    unittest.main()
