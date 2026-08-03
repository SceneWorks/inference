"""Regression tests for trust boundaries around persistent self-hosted CI runners."""

import re
import subprocess
import textwrap
import unittest
from pathlib import Path


WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"
REAL_WEIGHTS_WORKFLOW = WORKFLOW.with_name("real-weights.yml")
RESIDENCY_SCRIPT = WORKFLOW.parents[2] / "scripts" / "release" / "run-residency-ab.ps1"
QWEN_MEMORY_STRATEGY = (
    WORKFLOW.parents[2]
    / "crates/media/candle-gen/candle-gen-qwen-image/src/memory_strategy.rs"
)
JOB_ENV_RUNNER_TEMP_EXPRESSION = re.compile(
    r"(?m)^      [A-Z][A-Z0-9_]+: \$\{\{ runner\.temp \}\}"
)


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


def workflow_job_bodies(workflow: str) -> dict[str, list[str]]:
    """Return top-level job bodies without treating nested step keys as jobs."""
    lines = workflow.splitlines()
    try:
        cursor = lines.index("jobs:") + 1
    except ValueError as error:
        raise AssertionError("workflow has no jobs mapping") from error

    jobs: dict[str, list[str]] = {}
    current: str | None = None
    for line in lines[cursor:]:
        if line and not line.startswith(" "):
            break
        match = re.fullmatch(r"  ([A-Za-z0-9_-]+):", line)
        if match is not None:
            current = match.group(1)
            jobs[current] = []
        elif current is not None:
            jobs[current].append(line)
    return jobs


def privileged_real_weight_jobs(workflow: str) -> list[str]:
    """Find ordinary-CI jobs whose job-level runner declaration names the privileged label."""
    privileged: list[str] = []
    for job, lines in workflow_job_bodies(workflow).items():
        for index, line in enumerate(lines):
            if not line.startswith("    runs-on:"):
                continue
            declaration = [line.partition(":")[2].strip()]
            for continuation in lines[index + 1 :]:
                if continuation and len(continuation) - len(continuation.lstrip()) <= 4:
                    break
                declaration.append(continuation.strip())
            if "real-weights" in " ".join(declaration).lower():
                privileged.append(job)
            break
    return privileged


class CiWorkflowPolicyTests(unittest.TestCase):
    def test_chroma_packed_build_script_is_valid_bash(self) -> None:
        workflow = REAL_WEIGHTS_WORKFLOW.read_text(encoding="utf-8")
        step = re.search(
            r"(?ms)^      - name: Build and validate packed q4/q8 tiers\n"
            r".*?^        run: \|\n(?P<script>.*?)^      - name:",
            workflow,
        )
        self.assertIsNotNone(step)
        script = textwrap.dedent(step.group("script"))
        result = subprocess.run(
            ["bash", "-n"],
            input=script,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_chroma_shipping_policy_cannot_dispatch_unsupported_t5_geometry(self) -> None:
        workflow = REAL_WEIGHTS_WORKFLOW.read_text(encoding="utf-8")
        self.assertNotIn("chroma_t5_group_size:", workflow)
        self.assertNotIn("SC16462_AUX_BITS", workflow)
        self.assertNotIn("SC8777_BITS=8", workflow)
        self.assertEqual(workflow.count('SC8777_BITS="$bits"'), 3)
        self.assertEqual(workflow.count('SC16462_T5_GROUP_SIZE: "32"'), 2)
        self.assertGreaterEqual(
            workflow.count('test "$SC16462_T5_GROUP_SIZE" = "32"'), 2
        )
        self.assertEqual(workflow.count('"bits": 8,'), 2)
        self.assertEqual(workflow.count('"group_size": 32,'), 2)

    def test_sa3_snapshot_paths_are_manifest_derived(self) -> None:
        workflow = REAL_WEIGHTS_WORKFLOW.read_text(encoding="utf-8")
        self.assertNotRegex(workflow, r"SA3_[A-Z0-9_]+[^\n]*[0-9a-f]{40}")
        self.assertEqual(workflow.count("export_model_snapshot_paths.py"), 13)
        expected_models = {
            "same-l-metal": ("same-l",),
            "same-chunked-metal": ("same-s", "same-l"),
            "sa3-small-music-metal": ("stable-audio-3-small-music",),
            "sa3-small-sfx-metal": (
                "stable-audio-3-small-sfx",
                "stable-audio-3-small-music",
            ),
            "sa3-medium-metal": (
                "stable-audio-3-medium",
                "stable-audio-3-medium-base",
                "stable-audio-3-small-music",
                "stable-audio-3-small-sfx",
                "same-l",
            ),
            "sa3-base-identity-metal": (
                "stable-audio-3-small-music",
                "stable-audio-3-small-sfx",
                "stable-audio-3-medium",
                "stable-audio-3-small-music-base",
                "stable-audio-3-small-sfx-base",
                "stable-audio-3-medium-base",
                "same-l",
            ),
            "sa3-small-base-metal": (
                "stable-audio-3-small-music-base",
                "stable-audio-3-small-sfx-base",
            ),
        }
        expected_models.update(
            {
                job.replace("-metal", "-cuda"): models
                for job, models in expected_models.items()
                if job != "same-chunked-metal"
            }
        )
        for job, models in expected_models.items():
            match = re.search(
                rf"^  {re.escape(job)}:\n(?P<body>(?:^    .*\n|^\n)*)",
                workflow,
                flags=re.MULTILINE,
            )
            self.assertIsNotNone(match, job)
            body = match["body"]
            helper_start = body.index("export_model_snapshot_paths.py")
            helper_end = body.find("\n      - name:", helper_start)
            helper = body[helper_start : helper_end if helper_end >= 0 else None]
            invocation_models = re.findall(r"--model\s+([^\s]+)", helper)
            self.assertEqual(invocation_models, list(models), job)
            self.assertEqual(len(invocation_models), len(set(invocation_models)), job)

    def test_real_weight_selection_is_informational_in_ordinary_ci(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("real_weights: ${{ steps.select.outputs.real_weights }}", workflow)
        self.assertIn("ordinary CI cannot launch privileged real-weight runners", workflow)
        self.assertEqual(privileged_real_weight_jobs(workflow), [])

    def test_privileged_runner_guard_rejects_differently_named_jobs(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        mutations = {
            "scalar": "    runs-on: real-weights",
            "list": "    runs-on: [self-hosted, macOS, ARM64, real-weights]",
            "expression": (
                "    runs-on: ${{ fromJSON('[\"self-hosted\",\"real-weights\"]') }}"
            ),
        }
        for shape, runs_on in mutations.items():
            with self.subTest(shape=shape):
                mutated = workflow + f"\n  differently-named-{shape}:\n{runs_on}\n"
                self.assertEqual(
                    privileged_real_weight_jobs(mutated),
                    [f"differently-named-{shape}"],
                )

    def test_sa3_weight_free_step_keeps_only_the_executable_invariant(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        start = workflow.index("- name: Test Stable Audio 3 weight-free quality gates")
        run = workflow.index("\n        run:", start)
        comments = [line for line in workflow[start:run].splitlines() if "#" in line]
        self.assertLessEqual(len(comments), 4)
        self.assertIn("test_sa3_ci_target_coverage.py", workflow[start:run])
        self.assertIn("SC_16605_REAL_WEIGHT_WORKFLOW_CLEANUP.md", workflow[start:run])

    def test_real_weight_concurrency_is_scoped_by_profile(self) -> None:
        workflow = REAL_WEIGHTS_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn(
            "group: inference-real-weights-${{ github.ref }}-"
            "${{ inputs.profile || 'schedule' }}",
            workflow,
        )

    def test_mage_media_lane_requires_verified_operator_cpu_oracles(self) -> None:
        workflow = REAL_WEIGHTS_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn('MAGE_REQUIRE_GOLDENS: "1"', workflow)
        self.assertIn(
            'echo "MAGE_GOLDEN_DIR=$RUNNER_TEMP/mage-flow-oracles" >> "$GITHUB_ENV"',
            workflow,
        )
        self.assertIn("scripts/release/provision_mage_oracles.py", workflow)
        self.assertIn("requirements-oracles.txt", workflow)
        self.assertIn("--require-hashes", workflow)
        self.assertIn("uv python install 3.12.10", workflow)
        self.assertIn('python_path="$(uv python find 3.12.10)"', workflow)
        self.assertIn(
            "UV_CACHE_DIR: ${{ runner.temp }}/uv-cache",
            workflow,
        )
        self.assertIn(
            "UV_PYTHON_INSTALL_DIR: ${{ runner.temp }}/python-install",
            workflow,
        )
        self.assertNotIn("uses: actions/setup-python", workflow)
        self.assertNotIn("3.12.11", workflow)
        self.assertIn("Run Mage-Flow text-encoder parity", workflow)
        self.assertIn("Run Mage-VAE all-geometry parity", workflow)
        self.assertNotIn("Regenerate Candle Mage 1024² Torch acceptance oracles", workflow)
        self.assertNotIn("MAGE_H=1024 MAGE_W=1024 MAGE_STEPS=20", workflow)
        self.assertNotIn("tools/dump_mage_flow_golden.py --stage dit", workflow)
        cache_sha = "5a3ec84eff668545956fd18022155c47e93e2684"
        self.assertIn(f"uses: actions/cache/restore@{cache_sha}", workflow)
        self.assertIn(f"uses: actions/cache/save@{cache_sha}", workflow)
        self.assertNotIn("restore-keys:", workflow)
        self.assertIn("id: mage-oracle-key", workflow)
        self.assertIn("id: mage-oracle-cache", workflow)
        self.assertIn("id: mage-oracle-seed", workflow)
        self.assertIn(
            "MAGE_ORACLE_SEED_DIR: ${{ vars.MAGE_ORACLE_SEED_DIR }}",
            workflow,
        )
        self.assertIn("refusing to run the multi-hour CPU producer", workflow)
        self.assertNotIn("Regenerate and verify shared CPU Mage oracles", workflow)
        self.assertIn(
            'if [[ ! -f "$source" || -L "$source" ]]',
            workflow,
        )
        for fingerprint_input in (
            ".github/workflows/real-weights.yml",
            "crates/media/mlx-gen/_vendor/mage_flow/**",
            "crates/media/mlx-gen/_vendor/mage_flow/assets/dog.jpg",
            "crates/media/mlx-gen/_vendor/mage_flow/requirements-oracles.txt",
            "crates/media/mlx-gen/tools/_paths.py",
            "crates/media/mlx-gen/tools/dump_mage_flow_golden.py",
            "crates/media/mlx-gen/tools/dump_mage_vae_sizes.py",
            "scripts/release/mage_reference_environment.py",
            "scripts/release/provision_mage_oracles.py",
            "scripts/release/provision_mage_edit_variants.py",
            "scripts/release/verify_mage_candle_oracles.py",
            "scripts/release/verify_mage_candle_transfer.py",
        ):
            self.assertIn(fingerprint_input, workflow)
        for snapshot in (
            "$MAGE_SNAPSHOT",
            "$MAGE_EDIT_SNAPSHOT",
            "$MAGE_EDIT_BASE_SNAPSHOT",
            "$MAGE_EDIT_TURBO_SNAPSHOT",
        ):
            self.assertIn(f'snapshot_revision "{snapshot}"', workflow)
        self.assertGreaterEqual(
            workflow.count("steps.mage-oracle-cache.outputs.cache-hit != 'true'"),
            2,
        )
        self.assertIn(
            "Verify restored or operator-provisioned Mage oracle cache",
            workflow,
        )
        self.assertGreaterEqual(workflow.count("--verify-only"), 2)
        self.assertIn("--verify-edit-artifact", workflow)
        self.assertNotIn("--write-manifest", workflow)
        self.assertIn("mage_candle_oracles_manifest.json", workflow)
        self.assertGreaterEqual(workflow.count("--edit-snapshot \"$MAGE_EDIT_SNAPSHOT\""), 3)
        self.assertEqual(workflow.count("--gen \"$MAGE_SNAPSHOT\""), 2)
        self.assertLess(
            workflow.index("Verify restored or operator-provisioned Mage oracle cache"),
            workflow.index("Save verified Mage oracle cache"),
        )
        self.assertIn("mage-flow-candle-oracles-${{ github.sha }}", workflow)
        self.assertIn("mage_edit_oracle_manifest.json", workflow)
        self.assertIn("mage_candle_transfer_manifest.json", workflow)
        # End-delimiter only — the first job of the audio lane, which follows the Mage upload
        # step. sc-16981 split the single `candle-audio` job into per-family
        # `candle-audio-<family>` jobs, so this bound moved to the first of them. If the audio
        # lane is reordered or renamed again, this is the anchor to update.
        upload_block = workflow[
            workflow.index("Upload Candle Mage acceptance oracles") :
            workflow.index("\n  candle-audio-kokoro:")
        ]
        uploaded = set(
            re.findall(
                r"\$\{\{ runner\.temp \}\}/mage-flow-oracles/([^\s]+)",
                upload_block,
            )
        )
        self.assertEqual(
            uploaded,
            {
                "mage_flow_te_golden.safetensors",
                "mage_flow_dit_golden.safetensors",
                "mage_flow_vae_f32_1024.safetensors",
                "mage_flow_e2e_golden.safetensors",
                "mage_flow_edit_golden.safetensors",
                "mage_flow_edit_base_golden.safetensors",
                "mage_flow_edit_turbo_golden.safetensors",
                "mage_edit_oracle_manifest.json",
                "mage_edit_variants_manifest.json",
                "mage_candle_oracles_manifest.json",
                "mage_candle_transfer_manifest.json",
            },
        )
        self.assertIn("mage_flow_dit_golden.safetensors", workflow)
        self.assertIn("mage_flow_e2e_golden.safetensors", workflow)
        self.assertIn("CANDLE_MAGE_SNAPSHOT: ${{ vars.CANDLE_MAGE_SNAPSHOT }}", workflow)
        self.assertIn(
            "cargo test --locked --release -p candle-gen-mage --features cuda "
            "--test real_parity -- --ignored --nocapture",
            workflow,
        )
        self.assertIn(
            "cargo test --locked --release -p candle-gen-mage --features cuda "
            "--test cuda_1024 -- --ignored --nocapture",
            workflow,
        )
        self.assertIn('set "MAGE_CONFORMANCE_FAILED=0"', workflow)
        self.assertEqual(
            workflow.count('|| set "MAGE_CONFORMANCE_FAILED=1"'),
            5,
        )
        self.assertIn("for %%T in (q4 q8 bf16) do (", workflow)
        self.assertIn(
            "--test quant_real_weights "
            "registered_tier_matches_independent_oracle_and_vram_budget "
            "-- --ignored --nocapture || set \"MAGE_CONFORMANCE_FAILED=1\"",
            workflow,
        )
        self.assertIn(
            'if not "%MAGE_CONFORMANCE_FAILED%"=="0" exit /b 1',
            workflow,
        )
        transferred_verify = workflow.index(
            "Verify transferred Candle Mage acceptance oracles"
        )
        candle_rust_acceptance = workflow.index(
            "cargo test --locked --release -p candle-gen-mage --features cuda "
            "--test real_parity"
        )
        self.assertLess(transferred_verify, candle_rust_acceptance)
        self.assertIn('"numpy==2.4.3" "safetensors==0.8.0"', workflow)
        self.assertIn("--verify-edit-artifact", workflow[transferred_verify:])
        self.assertIn(
            "provision_mage_edit_variants.py", workflow[transferred_verify:]
        )
        self.assertIn(
            "verify_mage_candle_oracles.py", workflow[transferred_verify:]
        )
        self.assertIn(
            "verify_mage_candle_transfer.py", workflow[transferred_verify:]
        )
        self.assertNotIn("MAGE_1024_GOLDEN_SHA256", workflow)
        self.assertIsNotNone(
            JOB_ENV_RUNNER_TEMP_EXPRESSION.search(
                "    env:\n      UNSAFE_PATH: ${{ runner.temp }}/shared-cache\n"
            ),
            "the policy expression must detect a synthetic job-environment mutation",
        )
        self.assertNotRegex(workflow, JOB_ENV_RUNNER_TEMP_EXPRESSION)
        for assignment in (
            r"EDIT_SRC=%RUNNER_TEMP%\sdxl-edit-src.ppm",
            r"EDIT_OUT=%RUNNER_TEMP%\sdxl-edit-out",
            r"IP_REF=%RUNNER_TEMP%\sdxl-ip-ref.ppm",
            r"IP_OUT=%RUNNER_TEMP%\sdxl-ip-out",
            r"MMAUDIO_WAV_OUT=%RUNNER_TEMP%\mmaudio_foley_16k.wav",
            r"MMAUDIO_WAV_OUT_44K=%RUNNER_TEMP%\mmaudio_foley_44k.wav",
        ):
            self.assertIn(assignment, workflow)
        self.assertNotIn("MAGE_FLOW_TE_GOLDEN: ${{ vars.", workflow)
        self.assertNotIn("MAGE_REF_PYTHON", workflow)

        ordinary_ci = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("provision_mage_oracles.py --self-test", ordinary_ci)

        lock = (
            WORKFLOW.parents[2]
            / "crates/media/mlx-gen/_vendor/mage_flow/requirements-oracles.txt"
        ).read_text(encoding="utf-8")
        requirement_lines = [
            line for line in lock.splitlines() if line and not line.startswith((" ", "#"))
        ]
        self.assertGreater(len(requirement_lines), 12)
        self.assertTrue(all(line.endswith(" \\") for line in requirement_lines))

    def test_residency_ab_is_operator_run_without_ci_model_dependencies(self) -> None:
        workflow = REAL_WEIGHTS_WORKFLOW.read_text(encoding="utf-8")
        self.assertNotIn("residency-ab", workflow)
        self.assertNotIn("QWEN_IMAGE_SNAPSHOT", workflow)
        self.assertNotIn("FLUX_DEV_DIR", workflow)

        script = RESIDENCY_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("qwen_image_probed_generate_for_offload_ab", script)
        self.assertIn("flux_dev_probed_generate_for_offload_ab", script)
        self.assertEqual(script.count("--features cuda"), 1)
        qwen_source = QWEN_MEMORY_STRATEGY.read_text(encoding="utf-8")
        fingerprint = re.search(
            r'pub const CALIBRATION_FINGERPRINT: &str =\s*"([^"]+)"',
            qwen_source,
        )
        self.assertIsNotNone(fingerprint)
        qwen_fingerprint = fingerprint.group(1)
        self.assertEqual(script.count(qwen_fingerprint), 1)
        self.assertEqual(script.count("$QwenCalibrationFingerprint"), 4)
        self.assertNotIn("qwen-image-cuda-residency-v1", script)

    def test_memory_evidence_v1_lane_is_exact_artifact_bound_and_operator_dispatched(self) -> None:
        workflow = REAL_WEIGHTS_WORKFLOW.read_text(encoding="utf-8")
        start = workflow.index("  mlx-memory-evidence-v1:")
        end = workflow.index("\n  mlx-llm:", start)
        job = workflow[start:end]

        self.assertIn("memory-evidence-v1", workflow.split("jobs:", 1)[0])
        self.assertIn("sceneworks_revision:", workflow.split("jobs:", 1)[0])
        self.assertIn(
            "if: github.event_name == 'workflow_dispatch' && "
            "inputs.profile == 'memory-evidence-v1'",
            job,
        )
        self.assertIn("INFERENCE_REVISION: ${{ github.sha }}", job)
        self.assertIn("ZIMAGE_SEQ_SIZE: 512", job)
        self.assertIn('test "$(git rev-parse HEAD)" = "$INFERENCE_REVISION"', job)
        self.assertIn("git diff --quiet", job)
        self.assertIn("git diff --cached --quiet", job)
        self.assertIn("ensure_model_snapshot.py", job)
        self.assertIn("SCENEWORKS_PROVENANCE_ROOT: ${{ runner.temp }}/sceneworks-provenance", job)
        self.assertIn("https://github.com/SceneWorks/SceneWorks.git", job)
        self.assertIn('fetch --depth=1 origin "${{ inputs.sceneworks_revision }}"', job)
        self.assertIn('git -C "$SCENEWORKS_PROVENANCE_ROOT" rev-parse HEAD', job)
        self.assertIn('echo "SCENEWORKS_REVISION=$resolved" >> "$GITHUB_ENV"', job)
        self.assertGreaterEqual(
            job.count('test -z "$(git status --porcelain --untracked-files=normal)"'),
            2,
        )
        self.assertIn("verify_model_snapshot.py", job)
        self.assertIn("--inventory-output", job)
        self.assertIn("MEMORY_MODEL_INVENTORY_SHA256", job)
        self.assertIn('echo "MEMORY_MODEL_REVISION=$model_revision" >> "$GITHUB_ENV"', job)
        self.assertIn("MEMORY_MODEL_INVENTORY_AFTER", job)
        self.assertIn('cmp -s "$MEMORY_MODEL_INVENTORY" "$MEMORY_MODEL_INVENTORY_AFTER"', job)
        self.assertIn("--model z-image-turbo", job)
        self.assertIn("sequential_bounds_peak_and_is_byte_identical", job)
        self.assertIn("--ignored --exact --test-threads=1 --nocapture", job)
        self.assertIn("set -o pipefail", job)
        self.assertIn("verify_residency_ab.py", job)
        self.assertIn("--min-reduction-mib 512", job)
        self.assertIn("--expected-fingerprint z-image-mlx-independent-materialization-v3", job)
        self.assertIn("--expected-abi 3", job)
        self.assertIn("--expected-model-revision", job)
        self.assertIn("--expected-model-inventory-sha256", job)
        self.assertIn("z_image_turbo-resident.rgb", job)
        self.assertIn("z_image_turbo-staged.rgb", job)
        self.assertIn("actions/upload-artifact@", job)
        self.assertNotIn("if: always()", job)
        self.assertIn("z-image-turbo-model-inventory.json", job)
        self.assertIn("verifier-result.txt", job)
        self.assertIn("memory-evidence-v1-z-image-${{ github.sha }}", job)

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
