"""Regression tests for trust boundaries around persistent self-hosted CI runners."""

import functools
import ntpath
import os
import re
import shutil
import subprocess
import textwrap
import unittest
from pathlib import Path


WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"
REAL_WEIGHTS_WORKFLOW = WORKFLOW.with_name("real-weights.yml")
REAL_WEIGHT_REQUIREMENTS = WORKFLOW.parent.parent / "requirements"
MACOS_HUB_LOCK = (
    ".github/requirements/real-weights-huggingface-hub-macos-arm64-py312.txt"
)
WINDOWS_HUB_LOCK = (
    ".github/requirements/real-weights-huggingface-hub-windows-x64-py312.txt"
)
WINDOWS_MAGE_LOCK = (
    ".github/requirements/real-weights-mage-verify-windows-x64-py312.txt"
)
MACOS_MAGE_LOCK = (
    "crates/media/mlx-gen/_vendor/mage_flow/requirements-oracles.txt"
)
APPROVED_REAL_WEIGHT_LOCKS = {
    MACOS_HUB_LOCK,
    WINDOWS_HUB_LOCK,
    WINDOWS_MAGE_LOCK,
    MACOS_MAGE_LOCK,
}
HUB_LOCK_PACKAGES = {
    "annotated-doc",
    "anyio",
    "certifi",
    "click",
    "filelock",
    "fsspec",
    "h11",
    "hf-xet",
    "httpcore",
    "httpx",
    "huggingface-hub",
    "idna",
    "markdown-it-py",
    "mdurl",
    "packaging",
    "pygments",
    "pyyaml",
    "rich",
    "shellingham",
    "tqdm",
    "typer",
    "typing-extensions",
}
RESIDENCY_SCRIPT = WORKFLOW.parents[2] / "scripts" / "release" / "run-residency-ab.ps1"
QWEN_MEMORY_STRATEGY = (
    WORKFLOW.parents[2]
    / "crates/media/candle-gen/candle-gen-qwen-image/src/memory_strategy.rs"
)
JOB_ENV_RUNNER_TEMP_EXPRESSION = re.compile(
    r"(?m)^      [A-Z][A-Z0-9_]+: \$\{\{ runner\.temp \}\}"
)

# Windows ships `C:\Windows\System32\bash.exe` -- the WSL launcher -- and it precedes Git for
# Windows on a default PATH, so bare `bash` resolves to it. With no WSL distro installed it exits
# non-zero before reading a byte of the script, which turns a syntax gate into a host-shape false
# red: it fails whether or not the workflow script is valid, so genuine drift in the script hides
# behind it. Resolve a shell that demonstrably parses instead of trusting PATH order, and skip
# honestly when the host has none -- a missing interpreter is not evidence of a broken script.
# Same class as the model-weight-licenses CRLF gate (72658873) and the golden model path gate
# (sc-17077). `ntpath` throughout so the stub is recognisable from Linux CI too, not only Windows.
WINDOWS_SYSTEM_DIRECTORIES = frozenset(
    ntpath.normcase(ntpath.join(os.environ.get("SystemRoot", r"C:\Windows"), name))
    for name in ("System32", "SysWOW64", "Sysnative")
)

# Git for Windows is the supported POSIX shell here; cover the machine-wide and per-user installs.
GIT_FOR_WINDOWS_ROOTS = ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)")
GIT_FOR_WINDOWS_BASH = (
    ntpath.join("Git", "bin", "bash.exe"),
    ntpath.join("Git", "usr", "bin", "bash.exe"),
)


def in_windows_system_directory(path: str) -> bool:
    """True when `path` is the WSL launcher shipped in the Windows system directory."""
    return ntpath.normcase(ntpath.dirname(str(path))) in WINDOWS_SYSTEM_DIRECTORIES


def posix_shell_candidates() -> list[str]:
    """Every plausible bash, PATH order first, with the Windows WSL stub filtered out."""
    candidates: list[str] = []

    def offer(candidate: str | None) -> None:
        if candidate is None or in_windows_system_directory(candidate):
            return
        if os.path.isfile(candidate) and candidate not in candidates:
            candidates.append(candidate)

    # `shutil.which` returns only the first hit, which on Windows is the stub -- walk PATH entry by
    # entry so a Git bash sitting behind System32 is still found.
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if directory:
            offer(shutil.which("bash", path=directory))

    roots = [os.environ[name] for name in GIT_FOR_WINDOWS_ROOTS if name in os.environ]
    if "LOCALAPPDATA" in os.environ:
        roots.append(ntpath.join(os.environ["LOCALAPPDATA"], "Programs"))
    for root in roots:
        for relative in GIT_FOR_WINDOWS_BASH:
            offer(ntpath.join(root, relative))
    return candidates


@functools.lru_cache(maxsize=1)
def posix_shell() -> str | None:
    """The first candidate that actually parses a script, or None when the host has no POSIX shell.

    The probe -- not the directory filter -- is what makes this honest: a shell that cannot parse
    `:` can never be selected, so a stub in an unanticipated location still cannot produce a red.
    That is not hypothetical. Windows also installs the WSL launcher as an App Execution Alias at
    `%LOCALAPPDATA%\\Microsoft\\WindowsApps\\bash.exe`, which is not a system directory and so
    survives the filter; only the probe rejects it and moves on to Git for Windows.
    """
    for candidate in posix_shell_candidates():
        try:
            probe = subprocess.run(
                [candidate, "-n"],
                input=":\n",
                text=True,
                encoding="utf-8",
                capture_output=True,
                check=False,
            )
        except OSError:
            continue
        if probe.returncode == 0:
            return candidate
    return None


def bash_syntax_check(shell: str, script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [shell, "-n"],
        input=script,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )


def chroma_packed_build_script() -> str:
    workflow = REAL_WEIGHTS_WORKFLOW.read_text(encoding="utf-8")
    step = re.search(
        r"(?ms)^      - name: Build and validate packed q4/q8 tiers\n"
        r".*?^        run: \|\n(?P<script>.*?)^      - name:",
        workflow,
    )
    if step is None:
        raise AssertionError("missing workflow step: Build and validate packed q4/q8 tiers")
    return textwrap.dedent(step.group("script"))


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


def real_weight_pip_policy_errors(workflow: str) -> list[str]:
    """Reject pip installs that can escape the reviewed wheel/hash inputs."""
    errors: list[str] = []
    install_lines = [
        (line_number, line.strip())
        for line_number, line in enumerate(workflow.splitlines(), start=1)
        if re.search(r"\bpip(?:\d+(?:\.\d+)*)?\s+install\b", line)
    ]
    if not install_lines:
        return ["real-weight workflow has no pip installs"]

    locks_seen: list[str] = []
    for line_number, command in install_lines:
        prefix = f"line {line_number}"
        for required_flag in ("--only-binary=:all:", "--require-hashes"):
            if required_flag not in command:
                errors.append(f"{prefix}: missing {required_flag}")
        if command.endswith(("\\", "^")):
            errors.append(f"{prefix}: pip install must be a single physical line")

        requirement = re.findall(r"(?:^|\s)(?:-r|--requirement)\s+(\S+)", command)
        if len(requirement) != 1:
            errors.append(f"{prefix}: expected exactly one requirement lock")
            continue
        lock = requirement[0].strip("'\"")
        locks_seen.append(lock)
        if lock not in APPROVED_REAL_WEIGHT_LOCKS:
            errors.append(f"{prefix}: unapproved requirement lock {lock}")

        expected_lock = None
        if "mage-reference/bin/python" in command:
            expected_lock = MACOS_MAGE_LOCK
        elif "mage-oracle-verify" in command:
            expected_lock = WINDOWS_MAGE_LOCK
        elif "python3 -m pip" in command:
            expected_lock = MACOS_HUB_LOCK
        elif "python -m pip" in command:
            expected_lock = WINDOWS_HUB_LOCK
        if expected_lock != lock:
            errors.append(
                f"{prefix}: install target expects {expected_lock}, got {lock}"
            )

        install_arguments = command.split("pip install", 1)[1]
        before_lock, after_lock = re.split(
            r"(?:^|\s)(?:-r|--requirement)\s+\S+", install_arguments, maxsplit=1
        )
        before_lock = re.sub(
            r"(?:^|\s)--target\s+(?:\"[^\"]+\"|'[^']+'|\S+)", "", before_lock
        )
        for allowed_flag in (
            "--disable-pip-version-check",
            "--only-binary=:all:",
            "--require-hashes",
        ):
            before_lock = before_lock.replace(allowed_flag, "")
        if before_lock.strip():
            errors.append(f"{prefix}: unexpected argument before requirement lock")
        if after_lock.strip() not in ("", "|| exit /b 1"):
            errors.append(f"{prefix}: unexpected argument after requirement lock")

    expected_lock_counts = {
        MACOS_HUB_LOCK: 22,
        WINDOWS_HUB_LOCK: 10,
        WINDOWS_MAGE_LOCK: 1,
        MACOS_MAGE_LOCK: 1,
    }
    actual_lock_counts = {lock: locks_seen.count(lock) for lock in set(locks_seen)}
    if actual_lock_counts != expected_lock_counts:
        errors.append(
            f"pip install lock counts differ: expected {expected_lock_counts}, "
            f"got {actual_lock_counts}"
        )
    return errors


def parse_binary_hashed_lock(lock: str) -> dict[str, tuple[str, str]]:
    """Parse the deliberately narrow one-wheel-per-package real-weight lock format."""
    lines = lock.splitlines()
    if "--only-binary=:all:" not in lines:
        raise AssertionError("lock must require binary distributions")
    requirements: dict[str, tuple[str, str]] = {}
    cursor = 0
    while cursor < len(lines):
        line = lines[cursor]
        if not line or line.startswith("#") or line == "--only-binary=:all:":
            cursor += 1
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s\\]+) \\", line)
        if match is None:
            raise AssertionError(f"unrecognized lock line: {line!r}")
        if cursor + 1 >= len(lines):
            raise AssertionError(f"missing hash for {match.group(1)}")
        hash_match = re.fullmatch(
            r"    --hash=sha256:([0-9a-f]{64})", lines[cursor + 1]
        )
        if hash_match is None:
            raise AssertionError(f"missing SHA-256 wheel hash for {match.group(1)}")
        name = match.group(1).lower().replace("_", "-")
        if name in requirements:
            raise AssertionError(f"duplicate locked package: {name}")
        requirements[name] = (match.group(2), hash_match.group(1))
        cursor += 2
    return requirements


def validate_binary_hashed_lock(
    lock: str, expected_packages: set[str]
) -> dict[str, tuple[str, str]]:
    requirements = parse_binary_hashed_lock(lock)
    if set(requirements) != expected_packages:
        missing = sorted(expected_packages - set(requirements))
        extra = sorted(set(requirements) - expected_packages)
        raise AssertionError(f"lock package set differs: missing={missing}, extra={extra}")
    return requirements


class CiWorkflowPolicyTests(unittest.TestCase):
    def require_posix_shell(self) -> str:
        shell = posix_shell()
        if shell is None:
            self.skipTest(
                "no usable POSIX shell on this host: searched every PATH entry (excluding the "
                "Windows System32 WSL stub) and the Git for Windows install locations, and none "
                "parsed a trivial script. Install Git for Windows, or a bash, to run this gate."
            )
        return shell

    def test_real_weight_python_installs_are_binary_hash_locked(self) -> None:
        workflow = REAL_WEIGHTS_WORKFLOW.read_text(encoding="utf-8")
        self.assertEqual(real_weight_pip_policy_errors(workflow), [])
        self.assertEqual(workflow.count(MACOS_HUB_LOCK), 22)
        self.assertEqual(workflow.count(WINDOWS_HUB_LOCK), 10)
        self.assertEqual(workflow.count(WINDOWS_MAGE_LOCK), 1)
        self.assertNotRegex(
            workflow,
            r"\bpip\s+install[^\n]*(?:huggingface[_-]hub|numpy|safetensors)==",
        )

    def test_real_weight_pip_policy_discriminates_bypass_mutations(self) -> None:
        workflow = REAL_WEIGHTS_WORKFLOW.read_text(encoding="utf-8")
        mutations = {
            "missing hashes": workflow.replace(" --require-hashes", "", 1),
            "missing binary only": workflow.replace(" --only-binary=:all:", "", 1),
            "new inline install": workflow
            + '\n          python3 -m pip install "requests==2.32.5"\n',
            "direct pip bypass": workflow + "\n          pip install requests\n",
            "new superficially compliant install": workflow
            + f"\n          python3 -m pip install --disable-pip-version-check "
            f"--only-binary=:all: --require-hashes -r {MACOS_HUB_LOCK}\n",
            "inline package after lock": workflow.replace(
                f"-r {MACOS_HUB_LOCK}",
                f"requests -r {MACOS_HUB_LOCK}",
                1,
            ),
            "binary override": workflow.replace(
                f"-r {MACOS_HUB_LOCK}",
                f"--no-binary=:all: -r {MACOS_HUB_LOCK}",
                1,
            ),
            "wrong platform lock": workflow.replace(
                MACOS_HUB_LOCK, WINDOWS_HUB_LOCK, 1
            ),
            "unreviewed lock": workflow.replace(
                MACOS_HUB_LOCK, ".github/requirements/unreviewed.txt", 1
            ),
        }
        for mutation, mutated_workflow in mutations.items():
            with self.subTest(mutation=mutation):
                self.assertTrue(real_weight_pip_policy_errors(mutated_workflow))

    def test_real_weight_wheel_locks_are_complete_and_hash_shaped(self) -> None:
        macos = validate_binary_hashed_lock(
            (REAL_WEIGHT_REQUIREMENTS / Path(MACOS_HUB_LOCK).name).read_text(
                encoding="utf-8"
            ),
            HUB_LOCK_PACKAGES,
        )
        windows = validate_binary_hashed_lock(
            (REAL_WEIGHT_REQUIREMENTS / Path(WINDOWS_HUB_LOCK).name).read_text(
                encoding="utf-8"
            ),
            HUB_LOCK_PACKAGES | {"colorama"},
        )
        mage = validate_binary_hashed_lock(
            (REAL_WEIGHT_REQUIREMENTS / Path(WINDOWS_MAGE_LOCK).name).read_text(
                encoding="utf-8"
            ),
            {"numpy", "safetensors"},
        )
        self.assertEqual(macos["huggingface-hub"][0], "1.20.1")
        self.assertEqual(windows["huggingface-hub"][0], "1.20.1")
        self.assertNotEqual(macos["hf-xet"][1], windows["hf-xet"][1])
        self.assertNotEqual(macos["pyyaml"][1], windows["pyyaml"][1])

    def test_real_weight_lock_policy_discriminates_mutations(self) -> None:
        lock = (REAL_WEIGHT_REQUIREMENTS / Path(MACOS_HUB_LOCK).name).read_text(
            encoding="utf-8"
        )
        typing_extensions = (
            "typing-extensions==4.16.0 \\\n"
            "    --hash=sha256:"
            "481caa481374e813c1b176ada14e97f1f67a4539ce9cfeb3f350d78d6370c2e8\n"
        )
        mutations = {
            "missing binary policy": lock.replace("--only-binary=:all:\n", "", 1),
            "unhashed requirement": lock.replace(
                "    --hash=sha256:", "    --hash=sha512:", 1
            ),
            "unpinned requirement": lock.replace("annotated-doc==", "annotated-doc>=", 1),
            "incomplete closure": lock.replace(typing_extensions, "", 1),
            "duplicate package": lock + typing_extensions,
        }
        for mutation, mutated_lock in mutations.items():
            with self.subTest(mutation=mutation):
                with self.assertRaises(AssertionError):
                    validate_binary_hashed_lock(mutated_lock, HUB_LOCK_PACKAGES)

    def test_chroma_packed_build_script_is_valid_bash(self) -> None:
        script = chroma_packed_build_script()
        # An empty or truncated capture would make `bash -n` pass vacuously, so pin the extraction
        # to the payload this gate exists to check. Runs even on a host with no bash at all.
        self.assertIn("for bits in 4 8; do", script)
        self.assertIn('SC8777_BITS="$bits"', script)

        result = bash_syntax_check(self.require_posix_shell(), script)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_bash_syntax_check_rejects_a_malformed_build_script(self) -> None:
        # The positive case alone cannot tell "the script parses" from "the checker never fails".
        # Feed the real script with an unterminated loop appended, through the same shell and the
        # same code path: genuine drift in the workflow script must still turn this gate red.
        shell = self.require_posix_shell()
        result = bash_syntax_check(shell, chroma_packed_build_script() + "\nfor bits in 4 8; do\n")
        self.assertNotEqual(result.returncode, 0, "bash -n accepted an unterminated loop")
        self.assertNotEqual(result.stderr.strip(), "", "bash -n rejected the script silently")

    def test_shell_resolution_never_selects_the_windows_wsl_stub(self) -> None:
        # The sc-17196 regression: bare `bash` resolved to the WSL launcher, which exits non-zero
        # before parsing anything. Asserted with ntpath so Linux CI covers it too.
        stub = ntpath.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "bash.exe")
        self.assertTrue(
            in_windows_system_directory(stub),
            "the resolver must recognise the Windows System32 WSL stub",
        )
        for candidate in posix_shell_candidates():
            self.assertFalse(
                in_windows_system_directory(candidate),
                f"resolved a Windows system stub as a POSIX shell: {candidate}",
            )

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
        self.assertIn(WINDOWS_MAGE_LOCK, workflow)
        self.assertNotIn('"numpy==2.4.3" "safetensors==0.8.0"', workflow)
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
