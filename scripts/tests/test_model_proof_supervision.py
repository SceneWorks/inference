"""CPU-only host policy and inherited lease supervision; no NN/device/model execution."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import media_campaign_supervisor as safety
from scripts.tests.test_media_campaign_supervisor import Probe


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


if __name__ == '__main__':
    unittest.main()
