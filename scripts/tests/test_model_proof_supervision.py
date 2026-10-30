"""CPU-only host policy and inherited lease supervision; no NN/device/model execution."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import media_campaign_supervisor as safety
from scripts.tests.test_media_campaign_supervisor import Probe, FixtureDeadlineClock, published_fixture_pid


class NoGpuProbe(Probe):
    def gpu_free(self):
        raise AssertionError('a host-only proof must not probe a GPU')

    def gpu_free_and_tree_bytes(self, owner):
        raise AssertionError('a host-only proof must not probe GPU process memory')


def data(backend):
    return {'schemaVersion': 1, 'backend': backend, 'deadlineSeconds': 5,
            'pollMillis': 10, 'termGraceMillis': 50, 'hostFreeReserveBytes': 100,
            'childFootprintCapBytes': 1000000, 'stdoutCapBytes': 10000,
            'stderrCapBytes': 10000, 'eventCapBytes': 1000000}


class HostOnlyPolicies(unittest.TestCase):
    def test_linux_windows_cpu_policy_has_no_cuda_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'policy.json'
            for backend in ('linux-cpu', 'windows-cpu'):
                value = data(backend); path.write_text(json.dumps(value), encoding="utf-8")
                policy = safety.load_policy(path)
                self.assertIsNone(policy.cuda_device_uuid)
                value['cudaDeviceUuid'] = 'GPU-00000000-0000-0000-0000-000000000000'
                path.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaises(safety.SupervisionError):
                    safety.load_policy(path)

    @unittest.skipUnless(os.name == 'posix', 'POSIX lease FD; Windows uses Job ownership')
    def test_host_only_guard_passes_exact_lease_fd_without_gpu_probes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            path = root / 'policy.json'; path.write_text(json.dumps(data('linux-cpu')), encoding="utf-8")
            policy = safety.load_policy(path)
            lease_path = root / 'lease'; lease_path.write_bytes(b'cpu-only fixture')
            fd = os.open(lease_path, os.O_RDONLY)
            try:
                script = 'import os,sys;assert os.fstat(int(sys.argv[1])).st_ino == int(sys.argv[2])'
                result = safety.run_guarded([sys.executable, '-I', '-c', script, str(fd), str(os.fstat(fd).st_ino)],
                    cwd=root, env=dict(os.environ), policy=policy, stdout_path=root/'stdout', stderr_path=root/'stderr',
                    event_path=root/'events', probe=NoGpuProbe(), pass_fds=(fd,), require_gpu_sample=False)
                self.assertEqual(result.returncode, 0)
                self.assertIsNone(result.peak_gpu_bytes)
                self.assertIsNone(result.gpu_free_at_launch)
            finally:
                os.close(fd)

    def test_gpu_sample_requirement_is_explicit_boolean(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'policy.json'; path.write_text(json.dumps(data('darwin-mlx')), encoding="utf-8")
            with self.assertRaises(safety.SupervisionError):
                safety.run_guarded(['not-executed'], cwd=root, env={}, policy=safety.load_policy(path),
                    stdout_path=root/'stdout', stderr_path=root/'stderr', require_gpu_sample='false')

    def test_metadata_reaping_cannot_be_enabled_for_device_execution_or_nonboolean_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for backend, require_gpu, enabled in (('windows-cpu', False, 'true'),
                    ('windows-cpu', True, True), ('darwin-mlx', False, True)):
                with self.subTest(backend=backend, require_gpu=require_gpu, enabled=enabled):
                    path = root / 'policy.json'
                    path.write_text(json.dumps(data(backend)), encoding='utf-8')
                    with self.assertRaisesRegex(safety.SupervisionError, 'invalid-policy'):
                        safety.run_guarded(['not-executed'], cwd=root, env={}, policy=safety.load_policy(path),
                            stdout_path=root/'stdout', stderr_path=root/'stderr', require_gpu_sample=require_gpu,
                            reap_metadata_descendants=enabled)


@unittest.skipUnless(os.name == 'nt', 'native Windows Job fixture')
class WindowsMetadataDiscovery(unittest.TestCase):
    """A fake compiler exits after starting an owned persistent telemetry helper."""

    def run_fixture(self, *, enabled=True, exit_status=0, skip_reap=False):
        from scripts import media_campaign_windows as windows
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = data('windows-cpu')
            value.update(deadlineSeconds=3, termGraceMillis=2000, childFootprintCapBytes=512 * 1024**2)
            path = root / 'policy.json'; path.write_text(json.dumps(value), encoding='utf-8')
            helper = ("import os,pathlib,threading; "
                "pathlib.Path('helper.tmp').write_text(str(os.getpid()),encoding='ascii'); "
                "os.replace('helper.tmp','helper'); threading.Event().wait()")
            compiler = f'import subprocess,sys; subprocess.Popen([sys.executable,"-c",{helper!r}]); sys.exit({exit_status})'
            clock, retained, jobs, reaps = FixtureDeadlineClock(), [], [], []
            factory = windows.WindowsJob
            kernel, _ = windows._apis()
            def job_factory():
                job = factory(); jobs.append(job)
                original = job.terminate_and_reap
                def reap(child, grace):
                    reaps.append(job.members())
                    if not skip_reap:
                        original(child, grace)
                job.terminate_and_reap = reap
                return job
            def ready(root_pid, _pgid):
                def observed():
                    helper_pid = published_fixture_pid(root / 'helper')
                    if helper_pid is None:
                        return False
                    members = jobs[0].members()
                    if root_pid in members:
                        return False
                    self.assertIn(helper_pid, members)
                    handle = kernel.OpenProcess(0x00100000, False, helper_pid)
                    self.assertTrue(handle)
                    self.assertEqual(kernel.WaitForSingleObject(handle, 0), 0x102)
                    retained.append(handle)
                    return True
                clock.admit(observed)
            try:
                with patch.object(windows, 'WindowsJob', side_effect=job_factory):
                    try:
                        result = safety.run_guarded([sys.executable, '-c', compiler], cwd=root, env=dict(os.environ),
                            policy=safety.load_policy(path), stdout_path=root/'stdout', stderr_path=root/'stderr',
                            probe=NoGpuProbe(), on_spawn=ready, clock=clock, require_gpu_sample=False,
                            reap_metadata_descendants=enabled)
                    except safety.SupervisionError as error:
                        result = error
                self.assertTrue(retained, str(result))
                self.assertEqual(kernel.WaitForSingleObject(retained[0], 2000), 0,
                                 'metadata helper must signal exit before fixture completion')
                self.assertTrue(reaps)
                return result
            finally:
                for handle in retained:
                    kernel.CloseHandle(handle)

    def test_completed_compiler_reaps_owned_helper_before_success(self):
        result = self.run_fixture()
        self.assertIsInstance(result, safety.RunResult)
        self.assertEqual(result.returncode, 0)
        self.assertIsNone(result.peak_gpu_bytes)

    def test_execution_mode_keeps_live_descendant_until_deadline(self):
        result = self.run_fixture(enabled=False)
        self.assertIsInstance(result, safety.SupervisionError)
        self.assertEqual(result.reason, 'deadline')

    def test_failed_compiler_still_fails_after_helper_reaping(self):
        result = self.run_fixture(exit_status=7)
        self.assertIsInstance(result, safety.SupervisionError)
        self.assertEqual(result.reason, 'child-exit')

    def test_noop_reaper_cannot_accept_an_owned_live_helper(self):
        result = self.run_fixture(skip_reap=True)
        self.assertIsInstance(result, safety.SupervisionError)
        self.assertEqual(result.reason, 'cleanup-failure')


if __name__ == '__main__':
    unittest.main()
