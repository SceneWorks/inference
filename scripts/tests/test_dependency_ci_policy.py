import re
import unittest
from pathlib import Path

import yaml


WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"
METAL_CLIPPY = (
    "cargo clippy --locked -p candle-llm -p 'candle-gen*' -p 'candle-audio*' "
    "--all-targets --features metal -- -D warnings"
)


def normalized(command: str) -> str:
    return " ".join(command.split())


def run_commands(workflow: dict) -> list[str]:
    return [
        step["run"]
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if "run" in step
    ]


class DependencyCiPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))

    def test_supply_chain_policy_is_blocking_and_unconditional(self) -> None:
        jobs = self.workflow["jobs"]
        job = jobs["supply-chain"]
        self.assertNotIn("needs", job)
        self.assertNotIn("if", job)
        self.assertNotIn("supply_chain", jobs["changes"]["outputs"])
        self.assertIn("supply-chain", jobs["gate"]["needs"])

        commands = [normalized(step["run"]) for step in job["steps"] if "run" in step]
        governance = "python3 scripts/ci/check_advisory_policy.py"
        audit = "cargo deny --locked check advisories bans licenses sources"
        self.assertEqual(commands.count(governance), 1)
        self.assertEqual(commands.count(audit), 1)
        self.assertLess(commands.index(governance), commands.index(audit))

    def test_unique_metal_clippy_coverage_is_preserved_exactly(self) -> None:
        job = self.workflow["jobs"]["macos-metal"]
        matching = [
            step
            for step in job["steps"]
            if step.get("name") == "Clippy Candle Metal packages"
        ]
        self.assertEqual(len(matching), 1)
        self.assertEqual(normalized(matching[0]["run"]), METAL_CLIPPY)

    def test_retired_redundant_workspace_cargo_check_cannot_return(self) -> None:
        for command in run_commands(self.workflow):
            with self.subTest(command=normalized(command)[:100]):
                self.assertIsNone(
                    re.search(r"\bcargo\s+check\b.*?\s--workspace\b", normalized(command)),
                    "ordinary CI must not restore the redundant cargo check --workspace",
                )


if __name__ == "__main__":
    unittest.main()
