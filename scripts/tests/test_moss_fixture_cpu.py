"""CPU-only harness protocol and actual owned-process safety; no Cargo/device/model execution."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from scripts.ci import moss_fixture_cpu as fixture
from scripts.ci.moss_cuda_unit import UnitFailure, supervise


class FixtureProtocolTests(unittest.TestCase):
    def test_compile_requires_explicit_feature_exact_package_and_lib_json(self):
        for backend, package in (("candle", "candle-gen-moss-image"), ("mlx", "mlx-gen-moss-image")):
            argv = fixture.compile_argv(backend)
            self.assertEqual(argv[argv.index("-p") + 1], package)
            self.assertEqual(argv[argv.index("--features") + 1], "test-support")
            for argument in ("--locked", "--lib", "--no-run", "--message-format=json"):
                self.assertIn(argument, argv)
            self.assertNotIn("cuda", argv)

    def output(self, name=None):
        name = fixture.CASES[0] if name is None else name
        return f"running 1 test\ntest {name} ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 29 filtered out; finished in 0.1s\n"

    def test_positive_exact_result_and_crlf(self):
        output = self.output()
        for positive in (output, output.replace("\n", "\r\n"), output.replace("29 filtered", "901 filtered")):
            try:
                fixture.verify(positive, fixture.CASES[0])
            except UnitFailure as error:
                self.fail(str(error))

    def test_zero_ignored_wrong_duplicate_and_failed_results_refuse(self):
        output = self.output()
        for mutant in (
            output.replace("running 1 test", "running 0 tests"),
            output.replace("1 passed; 0 failed; 0 ignored", "0 passed; 0 failed; 1 ignored"),
            output.replace("0 ignored", "1 ignored"),
            output.replace(fixture.CASES[0], "different::test"),
            output + output,
            output.replace("test result: ok", "test result: FAILED"),
            output.replace(" ... ok", " ... ignored"),
        ):
            with self.subTest(mutant=mutant), self.assertRaises(UnitFailure):
                fixture.verify(mutant, fixture.CASES[0])

    def test_execution_uses_the_declared_owned_cpu_envelope_and_exact_filter(self):
        owner = mock.Mock(peak=173)
        owner.collect.return_value = {}
        with tempfile.TemporaryDirectory() as root, mock.patch("builtins.print"), mock.patch.object(fixture, "CpuOwner", return_value=owner), mock.patch.object(fixture, "supervise", return_value=self.output()) as run:
            fixture.execute(Path(root) / "actual-binary", fixture.CASES[0], Path(root) / "run.log")
        argv = run.call_args.args[0]
        self.assertEqual(argv[1], fixture.CASES[0])
        self.assertIn("--exact", argv)
        self.assertIn("--test-threads=1", argv)
        self.assertIs(run.call_args.args[1], owner)
        self.assertEqual(run.call_args.kwargs["deadline"], 180.0)
        self.assertEqual(run.call_args.kwargs["rss_cap"], 2 * 1024**3)
        self.assertEqual(run.call_args.kwargs["env"]["RUST_TEST_THREADS"], "1")
        self.assertEqual(run.call_args.kwargs["env"]["VECLIB_MAXIMUM_THREADS"], "1")
        self.assertEqual(run.call_args.kwargs["env"]["OMP_NUM_THREADS"], "1")

    def demand(self):
        return {"cpu_reference": True, "token_bound_case_host_envelope_bytes": 1024,
                "cases": [{"tensor_peak_bytes": 173, "host_envelope_bytes": 1024} for _ in fixture.CASES]}

    def test_actual_demand_marker_can_interleave_with_libtest_prefix(self):
        record = self.demand()
        output = self.output(fixture.DEMAND_TEST).replace(" ... ok", " ... MOSS_FIXTURE_DEMAND=" + json.dumps(record) + "\nok")
        fixture.verify(output, fixture.DEMAND_TEST)
        self.assertEqual(fixture.validate_demand(output), record)

    def test_demand_unknown_nonpositive_excessive_or_duplicate_refuses(self):
        for value in (0, -1, None, "173", fixture.RSS_CAP + 1):
            record = self.demand()
            record["cases"][0]["host_envelope_bytes"] = value
            with self.subTest(value=value), self.assertRaises(UnitFailure):
                fixture.validate_demand("MOSS_FIXTURE_DEMAND=" + json.dumps(record))
        record = self.demand()
        record["token_bound_case_host_envelope_bytes"] = fixture.RSS_CAP + 1
        with self.assertRaises(UnitFailure):
            fixture.validate_demand("MOSS_FIXTURE_DEMAND=" + json.dumps(record))
        text = "MOSS_FIXTURE_DEMAND=" + json.dumps(self.demand())
        with self.assertRaises(UnitFailure):
            fixture.validate_demand(text + "\n" + text)

    def test_cargo_artifact_must_match_package_lib_profile_and_owned_path(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root).resolve() / "target"
            target.mkdir()
            (target.parent / "foreign-test").write_text("existing foreign target", encoding="utf-8")
            binary = target / "moss-test"
            binary.write_text("fixture", encoding="utf-8")
            package = fixture.PACKAGES["candle"]
            record = {"reason": "compiler-artifact", "package_id": f"path+file:///repo/{package}#0.0.0",
                      "target": {"name": package.replace("-", "_"), "kind": ["lib"]},
                      "profile": {"test": True}, "executable": str(binary)}
            self.assertEqual(fixture.select_binary(json.dumps(record), package, target), binary)
            for key, value in (("package_id", "another-package#0.0.0"), ("profile", {"test": False}),
                               ("target", {"name": package.replace("-", "_"), "kind": ["test"]}),
                               ("executable", str(target.parent / "foreign-test"))):
                mutant = dict(record)
                mutant[key] = value
                with self.subTest(key=key), self.assertRaises(UnitFailure):
                    fixture.select_binary(json.dumps(mutant), package, target)
            with self.assertRaises(UnitFailure):
                fixture.select_binary(json.dumps(record) + "\n" + json.dumps(record), package, target)


@unittest.skipUnless(os.name == "posix", "owned POSIX process protocol")
class OwnedCpuSafetyTests(unittest.TestCase):
    def test_current_process_identity_is_uid_and_os_birth(self):
        uid, birth = fixture.process_identity(os.getpid())
        self.assertEqual(uid, os.geteuid())
        self.assertGreater(birth, 0)

    def test_real_child_success_is_reaped_without_gpu_probe(self):
        with tempfile.TemporaryDirectory() as root:
            owner = fixture.CpuOwner()
            output = supervise([sys.executable, "-c", "print('CPU_ONLY_OWNED')"], owner, Path(root) / "output.log",
                               deadline=5, rss_cap=128 * 1024**2, poll=0.01)
            self.assertIn("CPU_ONLY_OWNED", output)
            self.assertEqual(owner.child.returncode, 0)
            self.assertEqual(owner.collect(), {})

    def test_real_deadline_and_rss_breaches_release_owned_child(self):
        for deadline, cap, expected in ((0.1, 128 * 1024**2, "deadline"), (5, 1, "above")):
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as root:
                owner = fixture.CpuOwner()
                with self.assertRaisesRegex(UnitFailure, expected):
                    supervise([sys.executable, "-c", "import time; time.sleep(30)"], owner, Path(root) / "output.log",
                              deadline=deadline, rss_cap=cap, poll=0.01, grace=2)
                self.assertEqual(owner.collect(), {})
                self.assertIsNotNone(owner.child.returncode)

    def test_deadline_reaps_actual_descendant_in_a_new_session(self):
        with tempfile.TemporaryDirectory() as root:
            marker=Path(root)/"leaf.json"
            leaf="import os,json,time;from pathlib import Path;Path("+repr(str(marker))+").write_text(json.dumps({'pid':os.getpid(),'group':os.getpgrp()}),encoding='utf-8');time.sleep(30)"
            parent="import subprocess,sys,time;subprocess.Popen([sys.executable,'-c',"+repr(leaf)+"],start_new_session=True);time.sleep(30)"
            owner=fixture.CpuOwner()
            with self.assertRaisesRegex(UnitFailure,"deadline"):
                supervise([sys.executable,"-c",parent],owner,Path(root)/"output.log",deadline=0.8,rss_cap=128*1024**2,poll=0.01,grace=2)
            self.assertTrue(marker.is_file(),"actual regrouped descendant must run before breach")
            record=json.loads(marker.read_text(encoding="utf-8"))
            self.assertNotEqual(record['group'],owner.child.pid)
            self.assertIsNone(fixture.process_identity(record['pid']))
            self.assertEqual(owner.collect(),{})

    def test_changed_birth_or_uid_is_never_signaled(self):
        owner = fixture.CpuOwner()
        owner.known = {12345: (os.geteuid(), 173)}
        with mock.patch.object(owner, "collect", return_value={12345: (1, os.geteuid(), 1)}):
            for identity in ((os.geteuid(), 174), (os.geteuid() + 1, 173), None):
                with mock.patch.object(fixture, "process_identity", return_value=identity), mock.patch.object(fixture.os, "kill") as kill:
                    owner._signal(15)
                    kill.assert_not_called()

    def test_metadata_denial_is_failure_not_idle(self):
        owner = fixture.CpuOwner()
        with mock.patch.object(fixture.subprocess, "run", side_effect=subprocess.CalledProcessError(1, ["ps"])):
            with self.assertRaises(subprocess.CalledProcessError):
                owner.collect()


if __name__ == "__main__":
    unittest.main()
