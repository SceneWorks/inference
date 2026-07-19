"""Regression tests for trust boundaries around persistent self-hosted CI runners."""

import re
import unittest
from pathlib import Path


WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"


def job_if_expression(workflow: str, job: str) -> str:
    match = re.search(
        rf"^  {re.escape(job)}:\n(?P<body>(?:^    .*\n|^\n)*)",
        workflow,
        flags=re.MULTILINE,
    )
    if match is None:
        raise AssertionError(f"missing workflow job: {job}")

    lines = match.group("body").splitlines()
    for index, line in enumerate(lines):
        if line.startswith("    if: >-"):
            expression = []
            for continuation in lines[index + 1 :]:
                if not continuation.startswith("      "):
                    break
                expression.append(continuation.strip())
            return " ".join(expression)
        if line.startswith("    if: "):
            return line.removeprefix("    if: ").strip()
    raise AssertionError(f"missing if policy for workflow job: {job}")


def evaluate_policy(
    expression: str,
    *,
    lanes_include_cuda: bool,
    event_name: str,
    head_repository: str,
    repository: str = "SceneWorks/inference",
) -> bool:
    values = {
        "needs.changes.outputs.windows_cuda": str(lanes_include_cuda).lower(),
        "github.event_name": event_name,
        "github.event.pull_request.head.repo.full_name": head_repository,
        "github.repository": repository,
    }
    rendered = expression
    for name in sorted(values, key=len, reverse=True):
        rendered = rendered.replace(name, repr(values[name]))
    rendered = rendered.replace("&&", " and ").replace("||", " or ")
    if re.search(r"[A-Za-z_]\w*(?:\.\w+)+", rendered):
        raise AssertionError(f"unrecognized workflow context in policy: {rendered}")
    return bool(eval(rendered, {"__builtins__": {}}, {}))


class CiWorkflowPolicyTests(unittest.TestCase):
    def test_windows_cuda_check_rejects_fork_prs_but_preserves_trusted_events(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        expression = job_if_expression(workflow, "windows-cuda-check")

        cases = (
            ("pull_request", "external/fork", True, False),
            ("pull_request", "SceneWorks/inference", True, True),
            ("push", "", True, True),
            ("workflow_dispatch", "", True, True),
            ("push", "", False, False),
        )
        for event, head_repository, selected, expected in cases:
            with self.subTest(event=event, head_repository=head_repository, selected=selected):
                self.assertEqual(
                    evaluate_policy(
                        expression,
                        lanes_include_cuda=selected,
                        event_name=event,
                        head_repository=head_repository,
                    ),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
