"""Exercise suite routing with synthetic libtest transcripts; never run a model."""

import json
import subprocess
import sys
import unittest
from unittest.mock import patch

from scripts.ci import test_suites as suites


class TestSuites(unittest.TestCase):
    def case(self, **changes):
        return {
            "package": "fixture",
            "kind": "lib",
            "target": "fixture",
            "test": "numerical::sweep",
            "source": "scripts/ci/test_suites.py",
            "platforms": [sys.platform],
            **changes,
        }

    def artifact(self, *, target="fixture", kind="lib", features=()):
        return {
            "reason": "compiler-artifact",
            "profile": {"test": True},
            "package_id": "fixture-id",
            "target": {"kind": [kind], "name": target},
            "executable": "fixture-binary",
            "features": list(features),
        }

    def fake_run(self, calls, *, names=None, ignored=None, result=None, artifact=None):
        names = names or {"numerical::sweep", "contract::shape", "weights::external"}
        ignored = {"weights::external"} if ignored is None else ignored
        artifact = artifact or self.artifact()

        def run(command, timeout, **context):
            calls.append(command)
            self.assertGreater(timeout, 0)
            if command[0:2] == ["cargo", "metadata"]:
                return json.dumps({"target_directory": "/target", "packages": [{"id": "fixture-id", "name": "fixture", "manifest_path": str(suites.ROOT / "Cargo.toml"), "version": "0.0.0"}]})
            if command[0] == "rustc":
                return "/rust/lib\n"
            if command[0:2] == ["cargo", "test"]:
                self.assertIn("--no-run", command)
                self.assertIn("--message-format=json", command)
                return json.dumps(artifact)
            if "--list" in command:
                self.assertEqual(context["cwd"], str(suites.ROOT))
                self.assertEqual(context["env"]["RUST_TEST_THREADS"], "1")
                selected = ignored if "--ignored" in command else names
                return "\n".join(f"{name}: test" for name in sorted(selected))
            if result:
                return result
            if "--exact" in command:
                return "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 2 filtered out;"
            return "test result: ok. 1 passed; 0 failed; 1 ignored; 0 measured; 1 filtered out;"

        return run

    def execute(self, mode="routine", *, cases=None, **fake):
        calls = []
        with patch.object(suites, "run", self.fake_run(calls, **fake)):
            suites.execute(mode, ["cargo", "test", "--lib", "--tests", "-p", "fixture"], cases or [self.case()], 60)
        return calls

    def test_routine_compiles_full_graph_and_excludes_only_exact_reviewed_cases(self):
        calls = self.execute()
        self.assertIn(["fixture-binary", "--test-threads=1", "--skip", "numerical::sweep"], calls)
        self.assertFalse(any("--ignored" in call and "--list" not in call for call in calls))

    def test_explicit_exhaustive_runs_every_case_and_never_real_weight_ignores(self):
        calls = self.execute("exhaustive")
        self.assertIn(["fixture-binary", "--exact", "numerical::sweep", "--test-threads=1"], calls)
        self.assertFalse(any("--skip" in call for call in calls))

    def test_missing_ignored_and_ambiguous_names_fail_before_execution(self):
        for names, ignored in [
            ({"contract::shape"}, set()),
            ({"numerical::sweep"}, {"numerical::sweep"}),
            ({"numerical::sweep", "numerical::sweep_more"}, set()),
        ]:
            with self.subTest(names=names, ignored=ignored), self.assertRaises(ValueError):
                self.execute(names=names, ignored=ignored)

    def test_target_rename_fails_instead_of_losing_all_registered_cases(self):
        with self.assertRaisesRegex(ValueError, "targets absent"):
            self.execute(artifact=self.artifact(target="renamed"))

    def test_platform_and_feature_gates_do_not_hide_stale_active_cases(self):
        other = "linux" if sys.platform != "linux" else "darwin"
        cases = [self.case(platforms=[other]), self.case(test="cuda::sweep", features=["cuda"])]
        calls = self.execute(cases=cases, names={"contract::shape"}, ignored=set(), result="test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;")
        self.assertFalse(any("--skip" in call for call in calls))
        with self.assertRaisesRegex(ValueError, "stale/ignored"):
            self.execute(cases=cases, artifact=self.artifact(features=["cuda"]))

    def test_wrong_counts_and_empty_explicit_suite_fail(self):
        for mode in ("routine", "exhaustive"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.execute(mode, result="test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out;")
        with self.assertRaisesRegex(ValueError, "incomplete exhaustive"):
            self.execute("exhaustive", cases=[self.case(platforms=[])])

    def test_outer_summary_wins_over_nested_child_transcripts(self):
        output = "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;\n"
        suites.verify_result(output + "test result: ok. 2 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out;", 2, 1)
        with self.assertRaises(ValueError):
            suites.verify_result(output + "child failed without an outer libtest summary", 2, 1)

    def test_timeout_terminates_the_posix_child_process_group(self):
        if sys.platform == "win32":
            self.skipTest("POSIX process group contract")
        process = unittest.mock.Mock(pid=1234)
        process.communicate.side_effect = [subprocess.TimeoutExpired("fixture", .01), ("timed out", None)]
        with patch.object(suites.subprocess, "Popen", return_value=process), patch.object(suites.os, "killpg") as killpg:
            with self.assertRaises(TimeoutError):
                suites.run(["fixture"], .01)
            killpg.assert_called_once_with(1234, suites.signal.SIGKILL)
            process.kill.assert_called_once()

    def test_direct_libtest_preserves_package_and_loader_environment(self):
        package = {"name": "fixture", "version": "1.2.3", "manifest_path": "/repo/crate/Cargo.toml"}
        artifact = self.artifact()
        artifact["executable"] = "/target/debug/deps/fixture-binary"
        output = json.dumps({"reason": "build-script-executed", "package_id": "fixture-id", "out_dir": "/target/debug/build/fixture/out", "linked_paths": ["native=/target/debug/build/native", "native=/outside/not-cargo-runtime"]})
        with patch.dict(suites.os.environ, {"RUST_TEST_THREADS": "8", "CUDA_VISIBLE_DEVICES": "GPU-fixture"}):
            env = suites.test_environment({"target_directory": "/target"}, package, artifact, output, "/rust/lib\n")
        self.assertEqual(env["RUST_TEST_THREADS"], "1")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "GPU-fixture")
        self.assertEqual(env["CARGO_MANIFEST_DIR"], "/repo/crate")
        self.assertEqual(env["OUT_DIR"], "/target/debug/build/fixture/out")
        key = "PATH" if suites.os.name == "nt" else "DYLD_FALLBACK_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
        paths = env[key].split(suites.os.pathsep)
        for path in ("/target/debug/deps", "/target/debug", "/target/debug/build/native", "/rust/lib"):
            self.assertIn(path, paths)
        self.assertNotIn("/outside/not-cargo-runtime", paths)

    def test_no_fail_fast_collects_failure_without_masking_suite_status(self):
        calls = []
        run = self.fake_run(calls)

        def failing(command, timeout, **context):
            if command[0] == "fixture-binary" and "--list" not in command:
                raise RuntimeError("test failure")
            return run(command, timeout, **context)

        with patch.object(suites, "run", failing), self.assertRaisesRegex(RuntimeError, "suite failures"):
            suites.execute("routine", ["cargo", "test", "--lib", "--no-fail-fast"], [self.case()], 60)

    def test_checked_in_registry_and_workflow_routes_are_reachable(self):
        cases = suites.registry()
        self.assertTrue(any(case["package"].startswith("mlx") for case in cases))
        self.assertTrue(any(case["package"].startswith("candle") for case in cases))
        import yaml

        workflow = yaml.safe_load((suites.ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
        dispatch = workflow.get("on", workflow.get(True))["workflow_dispatch"]["inputs"]
        self.assertEqual(dispatch["test_suite"]["options"], ["routine", "exhaustive"])
        self.assertEqual(dispatch["test_suite"]["default"], "routine")
        for name, job in workflow["jobs"].items():
            self.assertIn("timeout-minutes", job, name)
        for name in ("candle-cpu-test", "macos-metal", "macos-nax", "windows-cuda"):
            job = workflow["jobs"][name]
            steps = [step for step in job["steps"] if "test_suites.py" in step.get("run", "")]
            self.assertTrue(steps, name)
            for step in steps:
                self.assertEqual(step["env"]["TEST_SUITE"], "${{ inputs.test_suite || 'routine' }}")
            self.assertIn(name, workflow["jobs"]["gate"]["needs"])
        self.assertIn("inputs.test_suite == 'exhaustive'", workflow["concurrency"]["group"])

    def test_restoration_checkpoint_matrix_is_exhaustive_only_on_mlx(self):
        name = "restoration_train_lifecycle::activation_checkpointing_leaves_the_run_unchanged"
        cases = {
            (case["package"], case["kind"], case["target"], case["test"]): case
            for case in suites.registry()
        }
        mlx = ("mlx-gen-iris", "test", "integration", name)
        self.assertIn(mlx, cases)
        self.assertEqual(cases[mlx]["platforms"], ["darwin"])
        self.assertNotIn(("candle-gen-iris", "test", "integration", name), cases)

    def test_cpu_matrix_preserves_both_groups_budgets_caches_and_failure_propagation(self):
        import yaml

        jobs = yaml.safe_load((suites.ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))["jobs"]
        cpu = jobs["candle-cpu-test"]
        self.assertEqual(cpu["timeout-minutes"], "${{ inputs.test_suite == 'exhaustive' && 150 || 90 }}")
        self.assertEqual(cpu["env"]["TEST_SUITE_DEADLINE_SECONDS"], "${{ inputs.test_suite == 'exhaustive' && '7200' || '4500' }}")
        steps = [step for step in cpu["steps"] if "test_suites.py" in step.get("run", "")]
        self.assertEqual(cpu["strategy"], {"fail-fast": False, "matrix": {"group": ["base", "gen"]}})
        self.assertNotIn("continue-on-error", cpu)
        self.assertEqual(cpu["runs-on"], "ubuntu-latest")
        cache = next(step for step in cpu["steps"] if step.get("name") == "Cache Cargo")
        self.assertEqual(cache["with"]["key"], "${{ matrix.group }}")
        self.assertNotIn("shared-key", cache["with"])
        self.assertEqual(cache["with"]["save-if"], "${{ github.ref == 'refs/heads/main' }}")
        self.assertEqual(len(steps), 2)
        self.assertEqual([step["if"] for step in steps], ["matrix.group == 'base'", "matrix.group == 'gen'"])
        selectors = [
            "-p candle-llm -p candle-quant-kernels -p 'candle-audio*' -p runtime-cpu",
            "-p 'candle-gen*'",
        ]
        targets = ["--lib", "--lib --tests"]
        for step, selector, target in zip(steps, selectors, targets):
            self.assertIn(f'select --scope "$TEST_SCOPE" {selector})', step["run"])
            self.assertIn(f'cargo test --locked {target} -j 1 "${{packages[@]}}"', step["run"])
        self.assertIn("candle-cpu-test", jobs["gate"]["needs"])
        self.assertIn("always()", jobs["gate"]["if"])
        gate = jobs["gate"]["steps"][0]
        self.assertEqual(gate["env"]["RESULTS"], "${{ join(needs.*.result, ' ') }}")
        self.assertIn("success|skipped) ;;", gate["run"])
        self.assertIn("exit 1", gate["run"])
        for step in steps:
            self.assertNotIn("continue-on-error", step)
            self.assertIn("-j 1", step["run"])
            self.assertIn('test_suites.py --deadline-seconds "$TEST_SUITE_DEADLINE_SECONDS" "$TEST_SUITE" -- cargo test', step["run"])
        self.assertEqual(jobs["macos-metal"]["timeout-minutes"], "${{ inputs.test_suite == 'exhaustive' && 180 || 75 }}")

    def test_hosted_mlx_full_graph_has_explicit_suite_headroom_only_at_its_step(self):
        import yaml

        jobs = yaml.safe_load((suites.ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))["jobs"]
        metal = jobs["macos-metal"]
        self.assertEqual(metal["timeout-minutes"], "${{ inputs.test_suite == 'exhaustive' && 180 || 75 }}")
        self.assertNotIn("TEST_SUITE_DEADLINE_SECONDS", metal.get("env", {}))
        step = next(step for step in metal["steps"] if step.get("name") == "Test MLX packages")
        self.assertEqual(step["env"]["TEST_SUITE_DEADLINE_SECONDS"], "${{ inputs.test_suite == 'exhaustive' && '7200' || '3600' }}")
        self.assertIn('test_suites.py --deadline-seconds "$TEST_SUITE_DEADLINE_SECONDS" "$TEST_SUITE" -- cargo test --locked --lib --tests "${packages[@]}"', step["run"])
        self.assertIn("-p mlx-llm -p mlx-llm-server -p mlx-gen -p 'mlx-gen-*' -p runtime-macos", step["run"])
        for name, job in jobs.items():
            if name == "candle-cpu-test":
                continue
            for other in job["steps"]:
                if name == "macos-metal" and other.get("name") == "Test MLX packages":
                    continue
                self.assertNotIn("TEST_SUITE_DEADLINE_SECONDS", other.get("env", {}))
                self.assertNotIn("--deadline-seconds", other.get("run", ""))

    def test_cuda_rank_fixture_preserves_the_parent_device_binding(self):
        source = (suites.ROOT / "crates/media/candle-gen/candle-gen-iris/tests/train_distributed.rs").read_text(encoding="utf-8")
        start = source.index("fn a_rank_without_its_cuda_device_is_refused_unless_sharing_is_opted_in()")
        end = source.index("fn inherited_single_cuda_device(", start)
        body = source[start:end]
        self.assertIn('std::env::var("CUDA_VISIBLE_DEVICES")', body)
        self.assertIn("inherited_single_cuda_device(visible.as_deref())", body)
        self.assertIn('json!({"CUDA_VISIBLE_DEVICES": visible})', body)
        self.assertNotIn('"CUDA_VISIBLE_DEVICES": "0"', body)
        self.assertIn('#[cfg(feature = "cuda")]\n#[test]\n', source[start - 50:start])
        self.assertNotIn("return;", body)
        self.assertIn("fn rank_children_preserve_only_the_single_bound_gpu1_mapping()", source)


if __name__ == "__main__":
    unittest.main()
