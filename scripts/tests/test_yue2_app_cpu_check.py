"""CPU-only source contract for the exact hosted SceneWorks check route."""

from pathlib import Path
import unittest


WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/yue2-app-precision-profile.yml"


def cpu_contract(source: str) -> None:
    assert "options: [cuda, metal, app-cpu-check]" in source
    group = source.split("concurrency:\n", 1)[1].split("\njobs:\n", 1)[0]
    assert "inputs.backend == 'app-cpu-check' && format('yue2-app-cpu-check-{0}', github.run_id)" in group
    assert "queue: max" in group and "cancel-in-progress: false" in group
    cpu = source.split("\n  app-cpu-check:\n", 1)[1].split("\n  cuda:\n", 1)[0]
    assert "if: inputs.backend == 'app-cpu-check'" in cpu
    assert "runs-on: ubuntu-latest" in cpu
    assert "timeout-minutes: 20" in cpu
    assert "test \"$EXPECTED_CONTROL_SHA\" = \"$GITHUB_SHA\"" in cpu
    assert "git -C app rev-parse HEAD" in cpu
    assert "git -C control rev-parse HEAD" in cpu
    assert "git -C \"$INFERENCE_REPO\" rev-parse HEAD" in cpu
    assert "('candle-core', 'candle-kernels')" in cpu
    assert "cache-dependency-path: app/package-lock.json" in cpu
    assert "node-version: \"20\"" in cpu
    assert "npm ci --ignore-scripts" in cpu
    assert "0b084cbf46b5b4f4f561305ef116268b57eecf7f" in cpu
    assert "node scripts/anchor-loader-closure.mjs --anchor-revisions" in cpu
    assert "npm run check 2>&1 | tee \"$CPU_EVIDENCE/npm-check.log\"" in cpu
    assert "if: always()" in cpu and "actions/upload-artifact@" in cpu
    assert "job_status=" in cpu and "packageSha256" in cpu
    assert "CUDA_VISIBLE_DEVICES" not in cpu
    assert "nvidia-smi" not in cpu
    assert "cargo test" not in cpu
    assert "\n  cuda:\n    if: inputs.backend == 'cuda'\n" in source
    assert "\n  metal:\n    if: inputs.backend == 'metal'\n" in source


class HostedAppCpuCheckTests(unittest.TestCase):
    def test_exact_cpu_route_preserves_full_check_and_old_hardware_routes(self):
        source = WORKFLOW.read_text(encoding="utf-8")
        cpu_contract(source)
        for mutant in (
            source.replace("runs-on: ubuntu-latest", "runs-on: [self-hosted, Windows, cuda]", 1),
            source.replace("npm run check 2>&1", "node --test scripts/check-source-control-bytes.test.mjs 2>&1", 1),
            source.replace("node scripts/anchor-loader-closure.mjs --anchor-revisions", "echo no-anchors", 1),
            source.replace("('candle-core', 'candle-kernels')", "('candle-kernels',)", 1),
            source.replace("if: inputs.backend == 'app-cpu-check'", "if: inputs.backend == 'cuda'", 1),
            source.replace("if: inputs.backend == 'metal'", "if: inputs.backend == 'app-cpu-check'", 1),
        ):
            with self.subTest(mutant=mutant[:100]):
                with self.assertRaises(AssertionError):
                    cpu_contract(mutant)


if __name__ == "__main__":
    unittest.main()
