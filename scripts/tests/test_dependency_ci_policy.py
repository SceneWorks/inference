import copy
import json
import re
import subprocess
import tomllib
import unittest
from collections.abc import Mapping
from pathlib import Path

import yaml


WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"
DENY = WORKFLOW.parents[2] / "deny.toml"
METAL_CLIPPY = (
    "cargo clippy --locked -p candle-llm -p 'candle-gen*' -p 'candle-audio*' "
    "--all-targets --features metal -- -D warnings"
)
GOVERNANCE = "python3 scripts/ci/check_advisory_policy.py"
AUDIT = "cargo deny --locked check advisories bans licenses sources"
FEATURE_ONLY_POLICY_PACKAGES = frozenset(
    {
        "candle-flash-attn",
        "candle-metal-kernels",
        "cudarc",
        "metal",
        "objc2-metal",
        "ug-metal",
    }
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


def triggers(workflow: dict) -> Mapping:
    # PyYAML follows YAML 1.1 and parses GitHub Actions' `on` key as boolean true.
    # Accept either spelling so this guard stays correct if the parser changes.
    value = workflow.get("on", workflow.get(True))
    if not isinstance(value, Mapping):
        raise AssertionError("workflow must declare structured event triggers")
    return value


def step_for_command(workflow: dict, command: str) -> dict:
    matching = [
        step
        for step in workflow["jobs"]["supply-chain"]["steps"]
        if "run" in step and normalized(step["run"]) == command
    ]
    if len(matching) != 1:
        raise AssertionError(f"expected exactly one supply-chain step for {command!r}")
    return matching[0]


class DependencyCiPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))

    def assert_supply_chain_policy_is_blocking_and_unconditional(self, workflow: dict) -> None:
        jobs = workflow["jobs"]
        job = jobs["supply-chain"]
        self.assertNotIn("needs", job)
        self.assertNotIn("if", job)
        self.assertNotIn("continue-on-error", job)
        self.assertNotIn("supply_chain", jobs["changes"]["outputs"])
        self.assertIn("supply-chain", jobs["gate"]["needs"])

        event_triggers = triggers(workflow)
        self.assertIn("pull_request", event_triggers)
        pull_request = event_triggers["pull_request"]
        if pull_request is not None:
            self.assertIsInstance(pull_request, Mapping)
            self.assertFalse(
                pull_request,
                "pull_request must be null or an empty mapping so no branch, action, or path "
                "filter can skip dependency policy",
            )

        commands = [normalized(step["run"]) for step in job["steps"] if "run" in step]
        self.assertEqual(commands.count(GOVERNANCE), 1)
        self.assertEqual(commands.count(AUDIT), 1)
        self.assertLess(commands.index(GOVERNANCE), commands.index(AUDIT))
        for command in (GOVERNANCE, AUDIT):
            step = step_for_command(workflow, command)
            self.assertNotIn("if", step)
            self.assertNotIn("continue-on-error", step)

    def test_supply_chain_policy_is_blocking_and_unconditional(self) -> None:
        self.assert_supply_chain_policy_is_blocking_and_unconditional(self.workflow)
        workflow = copy.deepcopy(self.workflow)
        triggers(workflow)["pull_request"] = {}
        self.assert_supply_chain_policy_is_blocking_and_unconditional(workflow)

    def test_skip_and_non_blocking_mutations_are_rejected(self) -> None:
        mutations = (
            (
                "governance step if",
                lambda workflow: step_for_command(workflow, GOVERNANCE).__setitem__("if", False),
            ),
            (
                "audit step if",
                lambda workflow: step_for_command(workflow, AUDIT).__setitem__("if", False),
            ),
            (
                "governance continue-on-error",
                lambda workflow: step_for_command(workflow, GOVERNANCE).__setitem__(
                    "continue-on-error", True
                ),
            ),
            (
                "audit continue-on-error",
                lambda workflow: step_for_command(workflow, AUDIT).__setitem__(
                    "continue-on-error", True
                ),
            ),
            (
                "job continue-on-error",
                lambda workflow: workflow["jobs"]["supply-chain"].__setitem__(
                    "continue-on-error", True
                ),
            ),
            (
                "pull-request branches",
                lambda workflow: triggers(workflow).__setitem__(
                    "pull_request", {"branches": ["main"]}
                ),
            ),
            (
                "pull-request branches-ignore",
                lambda workflow: triggers(workflow).__setitem__(
                    "pull_request", {"branches-ignore": ["feature/**"]}
                ),
            ),
            (
                "pull-request types",
                lambda workflow: triggers(workflow).__setitem__(
                    "pull_request", {"types": ["closed"]}
                ),
            ),
            (
                "pull-request paths",
                lambda workflow: triggers(workflow).__setitem__(
                    "pull_request", {"paths": ["crates/**"]}
                ),
            ),
            (
                "pull-request paths-ignore",
                lambda workflow: triggers(workflow).__setitem__(
                    "pull_request", {"paths-ignore": ["docs/**"]}
                ),
            ),
            (
                "pull-request trigger removed",
                lambda workflow: triggers(workflow).pop("pull_request"),
            ),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                workflow = copy.deepcopy(self.workflow)
                mutate(workflow)
                with self.assertRaises(AssertionError):
                    self.assert_supply_chain_policy_is_blocking_and_unconditional(workflow)

    def assert_policy_includes_feature_only_platform_graphs(self, policy: dict) -> None:
        graph = policy.get("graph")
        self.assertIsInstance(graph, Mapping)
        all_features = graph.get("all-features")
        self.assertIsInstance(all_features, bool)
        metadata_command = ["cargo", "metadata", "--locked"]
        if all_features:
            metadata_command.append("--all-features")
        metadata_command.extend(["--format-version", "1"])
        result = subprocess.run(
            metadata_command,
            cwd=WORKFLOW.parents[2],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        metadata = json.loads(result.stdout)
        package_names = {package["id"]: package["name"] for package in metadata["packages"]}
        resolved = {package_names[node["id"]] for node in metadata["resolve"]["nodes"]}
        self.assertFalse(
            FEATURE_ONLY_POLICY_PACKAGES - resolved,
            "cargo-deny graph lost representative optional Metal/CUDA packages",
        )

    def test_advisory_policy_includes_feature_only_platform_graphs(self) -> None:
        with DENY.open("rb") as source:
            policy = tomllib.load(source)
        self.assert_policy_includes_feature_only_platform_graphs(policy)

        mutated = copy.deepcopy(policy)
        mutated["graph"]["all-features"] = False
        with self.assertRaises(AssertionError):
            self.assert_policy_includes_feature_only_platform_graphs(mutated)

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
