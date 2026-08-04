"""Regression tests for trust boundaries around persistent self-hosted CI runners."""

import functools
import ntpath
import os
import re
import shutil
import subprocess
import textwrap
import tomllib
import unittest
from pathlib import Path


WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"
REAL_WEIGHTS_WORKFLOW = WORKFLOW.with_name("real-weights.yml")
MODEL_MANIFEST = WORKFLOW.parents[2] / "release" / "real-weight-models.toml"
# An `unwired_reason` has to carry an actual explanation. "n/a" or "todo" would silence the gate
# while recording nothing, which is precisely the failure this exists to prevent -- a deliberate
# non-goal has to stay legible as a decision. Every real exemption in the manifest is far longer.
MINIMUM_UNWIRED_REASON = 20
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
MACOS_INTERPRETER = "python3.12"
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


def real_weight_macos_interpreter_errors(workflow: str) -> list[str]:
    """Reject a macOS real-weight step that runs Python without naming the reviewed CPython.

    The pip policy below sees only `pip install` lines, which is half the chain. The macOS hub lock
    is a reviewed CPython 3.12 set whose every pin floors at 3.10, so a bare `python3` resolves to
    whatever the runner's launchd `.path` puts first — Apple's 3.9 on a box where /usr/bin precedes
    Homebrew — and pip then filters the entire reviewed set out of the candidate list and reports
    the pin as nonexistent rather than as a version conflict. The `PYTHONPATH=… python3 script.py`
    lines fail the same way one stage later: they import the tree the pip step installed, so a
    fixed installer with an unfixed consumer is still broken. Hence every Python invocation in a
    macOS job, not just the installs.
    """
    errors: list[str] = []
    # An interpreter TOKEN: `python`/`python3`/`python3.12` standing alone as a command word. The
    # lookbehind drops `actions/setup-python`, `${{ runner.temp }}/python-bin` and `"$python_path"`
    # (paths and variables, not command words); the lookahead drops `python_path=`.
    interpreter = re.compile(r"(?<![\w./$\"'-])(python[0-9.]*)(?![\w-])")
    # Two mentions in the Mage oracle job are legitimately not `python3.12`, and both are pinned by
    # construction rather than by PATH luck:
    #   * `uv python install 3.12.10` — a uv SUBCOMMAND, not an interpreter.
    #   * the `$RUNNER_TEMP/mage-reference` venv, created by and running the interpreter that step
    #     installed and prepended to $GITHUB_PATH.
    exempt = re.compile(r"\buv python\b|mage-reference")
    for job, lines in workflow_job_bodies(workflow).items():
        runs_on = next((line for line in lines if line.startswith("    runs-on:")), "")
        if "macOS" not in runs_on:
            continue
        for line in lines:
            # Comments must not fail the gate — this workflow's own header explains the rule in
            # prose that names bare `python3`. Cutting at `#` can truncate a quoted string, which
            # costs a false negative, the right way round for a gate that blocks a merge.
            command = line.split("#", 1)[0]
            if exempt.search(command):
                continue
            for found in interpreter.finditer(command):
                name = found.group(1)
                if name != MACOS_INTERPRETER:
                    errors.append(
                        f"{job}: macOS steps must name {MACOS_INTERPRETER}, found "
                        f"{name!r} in {command.strip()!r}"
                    )
    return errors


def real_weight_pip_policy_errors(workflow: str) -> list[str]:
    """Reject pip installs that can escape the reviewed wheel/hash inputs."""
    errors: list[str] = []
    pip_token = re.compile(r"\bpip(?:\d+(?:\.\d+)*)?\b", re.IGNORECASE)
    canonical_install = re.compile(
        r"\bpip(?:\d+(?:\.\d+)*)?\s+install\b", re.IGNORECASE
    )
    # A FULL-LINE comment is prose, not a command: the workflow header explains this very policy
    # and names `pip` while doing it, which used to fail the gate it documents. Text after code on
    # a line is still scanned, so the trailing-comment decoy mutations below stay caught.
    pip_lines = [
        (line_number, line.strip())
        for line_number, line in enumerate(workflow.splitlines(), start=1)
        if pip_token.search(line) and not line.lstrip().startswith("#")
    ]
    if not pip_lines:
        return ["real-weight workflow has no pip commands"]

    install_lines: list[tuple[int, str, re.Match[str]]] = []
    for line_number, command in pip_lines:
        first_pip = pip_token.search(command)
        assert first_pip is not None
        match = canonical_install.search(command)
        if match is None or match.start() != first_pip.start():
            errors.append(
                f"line {line_number}: first pip command is not a canonical "
                "single-line install"
            )
            continue
        install_lines.append((line_number, command, match))

    locks_seen: list[str] = []
    for line_number, command, install_match in install_lines:
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
        elif f"{MACOS_INTERPRETER} -m pip" in command:
            expected_lock = MACOS_HUB_LOCK
        elif "python -m pip" in command:
            expected_lock = WINDOWS_HUB_LOCK
        if expected_lock != lock:
            errors.append(
                f"{prefix}: install target expects {expected_lock}, got {lock}"
            )

        install_arguments = command[install_match.end() :]
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
        # 27 since sc-17284 added the `mlx-qwen-image`, `mlx-qwen-image-pid` and
        # `mlx-qwen-image-producers` jobs
        # (24 since sc-17250 added the JoyCaption and MOSS-TTS-Realtime jobs; 22 before).
        MACOS_HUB_LOCK: 27,
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


def workflow_code(workflows: str) -> str:
    """The workflow text with comment lines removed.

    Over CODE only, for the same reason `test_real_weight_macos_steps_name_the_reviewed_cpython`
    is: prose documenting a variable necessarily writes the very token being searched for. Without
    this, a comment explaining why something WAS wired keeps counting as wiring long after the
    mapping itself is deleted -- exactly the rot this gate exists to catch, hiding inside the gate.
    `#` covers YAML and the bash `run:` blocks; `rem` covers the Windows `shell: cmd` ones.
    """
    return "\n".join(
        line
        for line in workflows.splitlines()
        if not line.lstrip().startswith("#") and not line.lstrip().lower().startswith("rem ")
    )


def environment_variable_is_referenced(workflows: str, name: str) -> bool:
    """True when the workflows reference `name` as something a test process can actually read.

    Whole-token, so a longer name cannot lend its wiring to a shorter one: with
    `MMAUDIO_VAE_44K_SNAPSHOT` exported, a bare substring test would call a hypothetical
    `MMAUDIO_VAE_44K` referenced too and that key's coverage gap would never surface.

    A lone `${{ vars.NAME }}` / `${{ env.NAME }}` / `${{ secrets.NAME }}` does NOT count. A
    repository variable can be mapped onto a differently-named job variable --
    `MMAUDIO_BIGVGAN_V2_SNAPSHOT: ${{ vars.MMAUDIO_BIGVGAN_SNAPSHOT }}` does exactly that -- so
    those prove a value was read out of repository settings or an earlier step, never that any
    process sees an environment variable under that name.
    """
    code = workflow_code(workflows)
    for match in re.finditer(rf"(?<![A-Z0-9_]){re.escape(name)}(?![A-Z0-9_])", code):
        if not code[: match.start()].endswith(("vars.", "env.", "secrets.")):
            return True
    return False


def models_exported_by_key(workflows: str) -> set[str]:
    """Model keys wired through `export_model_snapshot_paths.py --model <key>`.

    That helper reads the manifest and writes `<environment name>=<path>` straight into GITHUB_ENV,
    so the variable name never appears in the workflow at all. The Stable Audio 3 lanes wire every
    snapshot this way; without this channel the gate would report them as orphans. Deliberately
    scoped to that script -- `ensure_model_snapshot.py --model <key>` takes the same flag but only
    materializes weights on disk and exports nothing. Over code only, like the reference check --
    a comment recalling a lane that USED to export a key must not keep that key looking wired.
    """
    keys: set[str] = set()
    for tail in workflow_code(workflows).split("export_model_snapshot_paths.py")[1:]:
        command: list[str] = []
        for line in tail.splitlines():
            command.append(line)
            if not line.rstrip().endswith("\\"):
                break
        keys.update(re.findall(r"--model[=\s]+([A-Za-z0-9._-]+)", " ".join(command)))
    return keys


def manifest_environment_wiring_errors(models: list[dict], workflows: str) -> list[str]:
    """Report manifest `environment` keys no workflow references, and exemptions that went stale.

    Both directions matter, and only checking one is how this rots. A key nothing references means
    the model is pinned, provisioned and holding disk on a runner while the tests it gates run
    NOWHERE -- sc-17266 found eleven at once, among them the MMAudio MM-DiT and both decoder stages,
    each with real conformance gates that had never executed. An `unwired_reason` on a key that IS
    referenced means the exemption outlived its justification, which is how the record of a
    deliberate non-goal (Mochi-1's freeze) decays into something indistinguishable from an oversight.
    """
    exported_by_key = models_exported_by_key(workflows)
    errors: list[str] = []
    for model in models:
        key = model.get("key", "<unkeyed>")
        reason = model.get("unwired_reason")
        variables = model.get("environment", [])
        if not variables:
            errors.append(f"{key}: declares no environment variable")
            continue
        if reason is not None and (
            not isinstance(reason, str) or len(reason.strip()) < MINIMUM_UNWIRED_REASON
        ):
            errors.append(
                f"{key}: unwired_reason must explain the exemption, got {reason!r}"
            )
        for name in variables:
            if key in exported_by_key or environment_variable_is_referenced(workflows, name):
                if reason is not None:
                    errors.append(
                        f"{key}: {name} is referenced by a workflow but still carries "
                        "unwired_reason -- delete the stale exemption"
                    )
            elif reason is None:
                errors.append(
                    f"{key}: {name} is declared but referenced by no workflow -- wire a lane "
                    "or record an unwired_reason"
                )
    return errors


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
        self.assertEqual(workflow.count(MACOS_HUB_LOCK), 27)
        self.assertEqual(workflow.count(WINDOWS_HUB_LOCK), 10)
        self.assertEqual(workflow.count(WINDOWS_MAGE_LOCK), 1)
        self.assertNotRegex(
            workflow,
            r"\bpip\s+install[^\n]*(?:huggingface[_-]hub|numpy|safetensors)==",
        )

    def test_real_weight_macos_steps_name_the_reviewed_cpython(self) -> None:
        workflow = REAL_WEIGHTS_WORKFLOW.read_text(encoding="utf-8")
        self.assertEqual(real_weight_macos_interpreter_errors(workflow), [])
        # The gate is worthless if it inspected nothing, and `count` alone would pass on a file
        # whose installs are all Windows. Pin both: the reviewed interpreter appears on every
        # macOS hub-lock install, and no bare `python3` survives anywhere in the file.
        self.assertEqual(
            workflow.count(f"{MACOS_INTERPRETER} -m pip install"),
            workflow.count(MACOS_HUB_LOCK),
        )
        # Over CODE only. The header documents the rule in prose that necessarily writes the very
        # token being banned, so a file-wide regex would fail on its own documentation.
        code = "\n".join(
            line for line in workflow.splitlines() if not line.lstrip().startswith("#")
        )
        self.assertNotRegex(code, r"(?<![\w.])python3(?!\.12)(?![\w-])")

    def test_real_weight_macos_interpreter_policy_discriminates_mutations(self) -> None:
        workflow = REAL_WEIGHTS_WORKFLOW.read_text(encoding="utf-8")
        mutations = {
            "bare python3 installer": workflow.replace(
                f"{MACOS_INTERPRETER} -m pip install", "python3 -m pip install", 1
            ),
            "bare python3 consumer": workflow.replace(
                f"{MACOS_INTERPRETER} scripts/release/ensure_model_snapshot.py",
                "python3 scripts/release/ensure_model_snapshot.py",
                1,
            ),
            "bare python3 heredoc": workflow.replace(
                f"{MACOS_INTERPRETER} - <<'PY'", "python3 - <<'PY'", 1
            ),
            "windows interpreter on a macOS step": workflow.replace(
                f"{MACOS_INTERPRETER} scripts/release/resolve_snapshot_paths.py",
                "python scripts/release/resolve_snapshot_paths.py",
                1,
            ),
            "unreviewed minor": workflow.replace(
                f"{MACOS_INTERPRETER} -m pip install", "python3.9 -m pip install", 1
            ),
        }
        for mutation, mutated_workflow in mutations.items():
            with self.subTest(mutation=mutation):
                self.assertTrue(real_weight_macos_interpreter_errors(mutated_workflow))
        # The `uv python` / `mage-reference` exemptions are line-scoped, not job-scoped: the Mage
        # oracle job carries both, and a bare `python3` on one of its OTHER lines must still fail.
        # Without this, widening an exemption to the job would silently unguard eight steps.
        mage_regression = workflow.replace(
            f"PYTHONPATH=\"$RUNNER_TEMP/huggingface-hub\" {MACOS_INTERPRETER} "
            "scripts/release/ensure_model_snapshot.py --model z-image-turbo",
            'PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3 '
            "scripts/release/ensure_model_snapshot.py --model z-image-turbo",
            1,
        )
        self.assertNotEqual(mage_regression, workflow)
        self.assertTrue(
            any(
                error.startswith("mlx-media:")
                for error in real_weight_macos_interpreter_errors(mage_regression)
            )
        )

    def test_real_weight_pip_policy_discriminates_bypass_mutations(self) -> None:
        workflow = REAL_WEIGHTS_WORKFLOW.read_text(encoding="utf-8")
        canonical_macos_install = (
            f"{MACOS_INTERPRETER} -m pip install --disable-pip-version-check "
            "--only-binary=:all: --require-hashes --target \"$PYTHONPATH\" "
            f"-r {MACOS_HUB_LOCK}"
        )
        mutations = {
            "missing hashes": workflow.replace(" --require-hashes", "", 1),
            "missing binary only": workflow.replace(" --only-binary=:all:", "", 1),
            "new inline install": workflow
            + '\n          python3 -m pip install "requests==2.32.5"\n',
            "direct pip bypass": workflow + "\n          pip install requests\n",
            "direct pip3 bypass": workflow + "\n          pip3 install requests\n",
            "posix continuation bypass": workflow
            + "\n          python3 -m pip \\\n"
            + "            install requests\n",
            "cmd continuation bypass": workflow
            + "\n          python -m pip ^\n"
            + "            install requests\n",
            "option before install bypass": workflow
            + "\n          python3 -m pip --disable-pip-version-check install requests\n",
            "later canonical comment decoy": workflow.replace(
                canonical_macos_install,
                "python3 -m pip --disable-pip-version-check install requests # "
                + canonical_macos_install,
                1,
            ),
            "later canonical separator decoy": workflow.replace(
                canonical_macos_install,
                "python3 -m pip --disable-pip-version-check install requests ; "
                + canonical_macos_install,
                1,
            ),
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
        self.assertIn('export SC16462_BASELINE="$CHROMA_SNAPSHOT/q$bits"', script)

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

    def test_chroma_auxiliaries_cannot_be_dispatched_above_the_selected_tier(self) -> None:
        """sc-16462: the auxiliary width must FOLLOW the tier, and nothing may override it.

        The defect this guards is the one the story exists to remove: a "q4" tier whose text
        encoder is secretly wider. The width is derived in Rust from the tier's own packed
        transformer, so the workflow must expose no knob that could reintroduce a divergence,
        and must verify the published artifact declares exactly its tier's width.
        """
        workflow = REAL_WEIGHTS_WORKFLOW.read_text(encoding="utf-8")
        # No dispatch input or env var may select an auxiliary width or T5 geometry.
        for forbidden in (
            "chroma_t5_group_size:",
            "SC16462_AUX_BITS",
            "SC16462_T5_GROUP_SIZE",
            "SC8777_BITS",
            "auxiliary_bits",
        ):
            self.assertNotIn(
                forbidden,
                workflow,
                f"{forbidden} would let a dispatch put Chroma's auxiliaries above the selected tier",
            )
        # Both lanes must build through the derived-width seam, never a hand-rolled width.
        self.assertEqual(workflow.count("packed_auxiliaries_match_load_time_quantization"), 2)
        # The published artifact is verified to declare exactly its own tier's width.
        self.assertIn('if quantization.get("bits") != expected_bits:', workflow)
        self.assertIn('expected_bits = int(tier[1:])', workflow)
        self.assertIn('if quantization.get("group_size") != 32:', workflow)
        # A stale residual-era artifact must be rejected rather than silently published.
        self.assertIn('if "residual_bits" in quantization:', workflow)

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
        """The CUDA residency A/B stays operator-run — gated directly, not by variable name.

        This used to assert the strings QWEN_IMAGE_SNAPSHOT and FLUX_DEV_DIR never appear in
        real-weights.yml. That proxy held only while the A/B was the sole consumer of both names,
        and sc-17284 found it was not: 45 mlx-gen-qwen-image and 2 mlx-gen-flux `#[ignore]` tests
        read the same two names on the Mac, so the ban was blocking a lane for a *different*
        backend rather than protecting this decision. The A/B is now gated by what it actually is.

        FLUX_DEV_DIR is therefore allowed in the workflow — the macOS PiD lane sets it, and
        `flux-1-dev` names one artifact both consumers want. QWEN_IMAGE_SNAPSHOT is still banned:
        it names the ~60 GB torch original, whose only consumers are this A/B and the candle CUDA
        tests, and the MLX half was renamed off it (MLX_GEN_QWEN_SNAPSHOT) precisely because the
        re-host it was being fed is a different repository at a different revision.
        """
        # Over CODE, not prose: `mlx-qwen-image`'s header comment has to name QWEN_IMAGE_SNAPSHOT to
        # explain why the MLX half was renamed off it. Same reason `workflow_code` exists for the
        # wiring gate — a comment can document a variable but can never wire one.
        workflow = workflow_code(REAL_WEIGHTS_WORKFLOW.read_text(encoding="utf-8"))
        self.assertNotIn("residency-ab", workflow)
        self.assertNotIn("QWEN_IMAGE_SNAPSHOT", workflow)

        # The A/B's own two tests are what must never be invoked from CI. This is the assertion the
        # name ban was standing in for, and unlike the ban it cannot be satisfied by a rename.
        self.assertNotIn("qwen_image_probed_generate_for_offload_ab", workflow)
        self.assertNotIn("flux_dev_probed_generate_for_offload_ab", workflow)

        # The operator repository variables hold nax-windows paths (`E:\huggingface\hub\...`,
        # `C:\Users\...`). A macOS lane that read one would resolve to a path that cannot exist on
        # a Mac, so no workflow may consume them; the macOS PiD lane maps its own variable onto
        # FLUX_DEV_DIR instead.
        self.assertNotIn("vars.FLUX_DEV_DIR", workflow)
        self.assertNotIn("vars.QWEN_IMAGE_SNAPSHOT", workflow)

        # Every assertion above is negative and would stay true if the macOS PiD lane were deleted.
        # The positive counterpart is not duplicated here because it already exists and is stronger:
        # `test_manifest_environment_keys_are_wired_or_explicitly_exempt` fails the moment
        # FLUX_DEV_DIR stops being referenced, because `flux-1-dev` no longer carries an exemption.

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

    def test_manifest_environment_keys_are_wired_or_explicitly_exempt(self) -> None:
        models = tomllib.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))["models"]
        # Every workflow, not just the two that exist today: a lane added in a third file must
        # count as wiring, or this gate would start reporting phantom orphans.
        workflows = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted(WORKFLOW.parent.glob("*.yml"))
        )
        self.assertEqual(manifest_environment_wiring_errors(models, workflows), [])

    def test_manifest_wiring_policy_discriminates_mutations(self) -> None:
        # The detector has to detect. Each case below is a defect shape that really occurred:
        # sc-17250 found three orphans by hand, sc-17266 eight more, and Mochi-1's freeze is a
        # decision that must not read as an oversight once someone finally wires a lane for it.
        wired = {"key": "wired", "environment": ["WIRED_SNAPSHOT"]}
        exempt = {
            "key": "exempt",
            "environment": ["OFF_SNAPSHOT"],
            "unwired_reason": "operator-run by design, not a CI lane",
        }
        text = "      WIRED_SNAPSHOT: ${{ vars.WIRED_SNAPSHOT }}\n"
        self.assertEqual(manifest_environment_wiring_errors([wired, exempt], text), [])

        orphan = {"key": "orphan", "environment": ["ORPHAN_SNAPSHOT"]}
        self.assertIn(
            "referenced by no workflow",
            "\n".join(manifest_environment_wiring_errors([orphan], text)),
        )

        stale = dict(exempt, environment=["WIRED_SNAPSHOT"])
        self.assertIn(
            "delete the stale exemption",
            "\n".join(manifest_environment_wiring_errors([stale], text)),
        )

        for empty in ("", "   ", "n/a", None):
            hollow = dict(orphan, unwired_reason=empty)
            self.assertNotEqual(
                manifest_environment_wiring_errors([hollow], text),
                [],
                f"unwired_reason={empty!r} must not be able to silence the gate",
            )

        # A longer referenced name must not lend its wiring to a shorter declared one.
        prefix = {"key": "prefix", "environment": ["WIRED"]}
        self.assertIn(
            "referenced by no workflow",
            "\n".join(manifest_environment_wiring_errors([prefix], text)),
        )

        # A repository variable mapped onto a DIFFERENT job variable does not wire its own name.
        remapped = "      OTHER_SNAPSHOT: ${{ vars.ORPHAN_SNAPSHOT }}\n"
        self.assertIn(
            "referenced by no workflow",
            "\n".join(manifest_environment_wiring_errors([orphan], remapped)),
        )

        # ...but the same name on the left of the mapping does.
        self.assertEqual(
            manifest_environment_wiring_errors(
                [orphan], "      ORPHAN_SNAPSHOT: ${{ vars.SOMETHING_ELSE }}\n"
            ),
            [],
        )

        # export_model_snapshot_paths.py wires by MODEL KEY, so the variable name never appears.
        exported = "        run: python3.12 scripts/release/export_model_snapshot_paths.py --model orphan\n"
        self.assertEqual(manifest_environment_wiring_errors([orphan], exported), [])
        self.assertEqual(
            models_exported_by_key(
                "python3.12 scripts/release/export_model_snapshot_paths.py \\\n"
                "  --model alpha --model beta\n"
                "python scripts/release/ensure_model_snapshot.py --model provisioned-only\n"
            ),
            {"alpha", "beta"},
        )
        # Materializing weights is not wiring: ensure_model_snapshot.py exports nothing.
        provisioned = "        run: python scripts/release/ensure_model_snapshot.py --model orphan\n"
        self.assertIn(
            "referenced by no workflow",
            "\n".join(manifest_environment_wiring_errors([orphan], provisioned)),
        )

        # PROSE IS NOT WIRING. The comment documenting a variable necessarily writes its name, so
        # without a code-only filter a comment left behind by a deleted mapping keeps the key
        # looking wired forever -- the gate would hide the exact rot it exists to find. This one is
        # not hypothetical: sc-17266's own comments name every variable the same commit wires.
        for prose in (
            "      # ORPHAN_SNAPSHOT used to be set here\n",
            "          rem ORPHAN_SNAPSHOT is materialized elsewhere\n",
            "      # historical: export_model_snapshot_paths.py --model orphan\n",
        ):
            with self.subTest(prose=prose.strip()):
                self.assertIn(
                    "referenced by no workflow",
                    "\n".join(manifest_environment_wiring_errors([orphan], prose)),
                )
        # A real mapping still counts even when a comment above it names the same variable.
        documented = (
            "      # ORPHAN_SNAPSHOT feeds the decoder conformance test\n"
            "      ORPHAN_SNAPSHOT: ${{ vars.SOMETHING_ELSE }}\n"
        )
        self.assertEqual(manifest_environment_wiring_errors([orphan], documented), [])

        # A pure consumer proves nothing: `${{ env.X }}` reads a value someone else must define.
        self.assertIn(
            "referenced by no workflow",
            "\n".join(
                manifest_environment_wiring_errors(
                    [orphan], "        run: echo ${{ env.ORPHAN_SNAPSHOT }}\n"
                )
            ),
        )

        self.assertIn(
            "declares no environment variable",
            "\n".join(
                manifest_environment_wiring_errors([{"key": "bare", "environment": []}], text)
            ),
        )

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

    def test_krea_s18_sweep_is_operator_dispatched_and_keeps_its_evidence(self) -> None:
        """sc-17276: the S18 coherence sweep is a measurement lane, not a regression gate.

        Three properties are load-bearing and each was a live defect at some point in this job's
        history, so each is pinned here rather than left to review. It must stay OFF the weekly
        schedule (~4.3 h on a box that runs four `rw-*` label pools one at a time is the sc-16981
        head-of-line block); it must carry the seeds WITHOUT WHICH THE TEST CANNOT PASS AT ALL, since
        the verdict rule refuses to rank configs with no between-seed variance estimate; and it must
        surrender its per-cell evidence even when the run fails, because an unresolvable verdict is
        exactly the outcome whose 21 cells someone needs to read and re-running costs another
        4.3 hours.
        """
        workflow = REAL_WEIGHTS_WORKFLOW.read_text(encoding="utf-8")
        start = workflow.index("  mlx-krea-realtime-s18-sweep:")
        job = workflow[start : workflow.index("\n  candle-audio-kokoro:", start)]

        self.assertIn("krea-s18-sweep", workflow.split("jobs:", 1)[0])
        self.assertIn(
            "if: github.event_name == 'workflow_dispatch' && "
            "inputs.profile == 'krea-s18-sweep'",
            job,
        )
        # Dispatch-only is only half of it: `schedule` must not reach this job by any other route.
        self.assertNotIn("github.event_name == 'schedule'", job)
        self.assertIn("runs-on: [self-hosted, macOS, ARM64, rw-krea]", job)
        # Over GitHub's 360-minute default, which killed a long macOS lane mid-run in sc-16981.
        self.assertIn("timeout-minutes: 480", job)
        # Rows AND seeds: the run-count assertion below is their product, so the job owns both.
        self.assertIn('KREA_S18_ROWS: ABCDFEZ', job)
        self.assertIn('KREA_S18_SEEDS: "7,11,23"', job)
        # Name-selected, so `--exact` after the `--` plus a run count — the sc-17250 false-green shape.
        self.assertIn("set -o pipefail", job)
        self.assertIn("-- --exact --ignored --nocapture", job)
        self.assertIn('grep -qE "test result: ok\\. 1 passed"', job)
        self.assertIn('"$cells" -ne 21', job)
        # The evidence must outlive a failing sweep: teed to a file inside the run step, then
        # extracted and uploaded from steps that run whatever the sweep did.
        self.assertIn('tee "$RUNNER_TEMP/s18-sweep.log"', job)
        self.assertEqual(job.count("if: always()"), 2)
        self.assertIn("actions/upload-artifact@", job)
        self.assertIn("krea-s18-sweep-${{ github.sha }}", job)

    def test_krea_e2e_step_pins_its_run_count_and_excludes_the_s18_sweep(self) -> None:
        """sc-17276: the e2e step selects by `--ignored`, which is a blanket.

        Every `#[ignore]` test in `generate_smoke.rs` is conscripted into it, which is how an
        85-minute research sweep ended up inside a 20-minute regression lane. The run-count assertion
        is what makes the next such addition loud instead of silent.
        """
        workflow = REAL_WEIGHTS_WORKFLOW.read_text(encoding="utf-8")
        start = workflow.index("      - name: Run Krea Realtime real-weight e2e (Q4 tier)")
        step = workflow[
            start : workflow.index("      - name: Run Krea Realtime KV-cache residency", start)
        ]

        self.assertIn("set -o pipefail", step)
        self.assertIn("--skip kv_cache_residency_at_the_production_geometry", step)
        self.assertIn("--skip long_clip_coherence_under_the_bounded_window", step)
        self.assertIn('grep -qE "test result: ok\\. 6 passed"', step)

    def test_qwen_image_lanes_name_select_every_test_and_pin_its_run_count(self) -> None:
        """sc-17284: the three Qwen-Image jobs must keep the contract they were wired under.

        Each of the 20 selections has to survive all three traps at once. `--exact` AFTER the `--`,
        because cargo rejects it in its own argument position; a run-count assertion, because with
        `--exact` accepted a renamed test yields `0 passed; N filtered out` and cargo EXITS 0; and a
        NAME, because `--ignored` alone is a blanket that silently conscripts whatever `#[ignore]`
        test lands in the file next -- which is exactly how an 85-minute sweep joined a 20-minute
        regression lane in sc-17276.

        Four tests are deliberately absent and must stay absent while their stories are open:
        `perf.rs` x2 (sc-17513) and `fit_preview_rgb_factors` (sc-17515) FAIL on real weights, and
        `lightning_loras_apply_cleanly` (sc-17518) asserts 840 modules against a published LoRA that
        carries 720. A red weekly lane is ignored within a month, so they are recorded rather than
        wired.
        """
        workflow = REAL_WEIGHTS_WORKFLOW.read_text(encoding="utf-8")
        jobs = ("mlx-qwen-image", "mlx-qwen-image-pid", "mlx-qwen-image-producers")
        # Slice to the NEXT job key at the same indentation, not to the next Qwen job — the last of
        # the three would otherwise swallow the rest of the file and read other lanes' commands.
        lines = workflow.splitlines(keepends=True)
        starts = {}
        for index, line in enumerate(lines):
            match = re.fullmatch(r"  ([a-z0-9][a-z0-9-]*):\n", line)
            if match:
                starts[index] = match.group(1)
        keys = sorted(starts)
        bodies = {}
        for position, index in enumerate(keys):
            name = starts[index]
            if name not in jobs:
                continue
            end = keys[position + 1] if position + 1 < len(keys) else len(lines)
            bodies[name] = "".join(lines[index:end])
        self.assertEqual(sorted(bodies), sorted(jobs), "a Qwen-Image job was renamed or removed")

        selected = {
            "mlx-qwen-image": [
                "control_loads_and_emits_hints",
                "scale_zero_matches_base",
                "scale_one_changes_output",
                "public_generate_runs",
                "default_sampler_equals_explicit_euler",
                "named_sampler_dpmpp_2m_is_coherent_and_distinct",
                "sequential_bounds_peak_and_is_byte_identical",
                "sequential_repeat_job_stays_bounded",
                "edit_sequential_bounds_peak_and_is_byte_identical",
                "control_sequential_bounds_peak_and_is_byte_identical",
                "bounded_window_is_distinct_from_the_unbounded_stream_control",
                "lightning_render_is_coherent",
                "edit_lightning_render_is_coherent",
                "routing_map_covers_full_fork_surface",
                "kohya_matches_peft_on_real_tree",
            ],
            "mlx-qwen-image-pid": [
                "use_pid_without_loaded_pid_errors",
                "qwen_image_pid_decode_vs_vae",
                "qwen_image_pid_from_ldm_early_stop",
                "flux_dev_pid_decode_vs_vae",
                "flux_dev_pid_from_ldm_early_stop",
            ],
            "mlx-qwen-image-producers": ["dump_runb_latents"],
        }
        for job, names in selected.items():
            body = bodies[job]
            for name in names:
                # `run_one <name>` in the multi-test steps, `name=<name>` in the single-test ones.
                self.assertTrue(
                    f"run_one {name}\n" in body or f"name={name}\n" in body,
                    f"{job}: {name} is no longer selected by name",
                )
            # Join `\`-continued shell lines first: every invocation here spans several, and a
            # per-LINE check cannot see that `--exact` moved from after the `--` to before it.
            joined, buffer = [], ""
            for line in body.splitlines():
                stripped = line.strip()
                buffer += " " + stripped.removesuffix("\\")
                if not stripped.endswith("\\"):
                    joined.append(buffer)
                    buffer = ""
            invocations = [command for command in joined if "cargo test " in command]
            self.assertTrue(invocations, f"{job}: no cargo test invocation")
            # Per STEP, not per job: without pipefail the `| tee /dev/stderr` swallows cargo's exit
            # status, and one step losing it is invisible to a bare `assertIn`.
            self.assertEqual(
                body.count("set -o pipefail"),
                len(invocations),
                f"{job}: every step that runs cargo test needs its own `set -o pipefail`",
            )
            # Trap 1: `--exact` is a libtest flag, and cargo REJECTS it in its own argument position
            # ("error: unexpected argument '--exact' found", exit 1). Everything before the ` -- `
            # is cargo's; everything after is libtest's.
            for command in invocations:
                cargo_arguments, separator, _ = command.partition(" -- ")
                self.assertTrue(separator, f"{job}: cargo test invocation has no `--`: {command}")
                self.assertNotIn(
                    "--exact",
                    cargo_arguments,
                    f"{job}: --exact must follow the `--`, not precede it",
                )
            # Trap 2: with `--exact` accepted, a rename yields `0 passed; N filtered out` and cargo
            # exits 0. Every selection must be paired with the count assertion.
            self.assertEqual(
                body.count("-- --exact --ignored --nocapture"),
                body.count('grep -qE "test result: ok\\. 1 passed"'),
                f"{job}: every `--exact` selection needs its own run-count assertion",
            )

        # Absent, and each with an open story. Over CODE only: the steps' comments have to NAME these
        # tests to say why they are excluded, and prose can never select a test.
        for name in (
            "qwen_t2i_per_step_compiled_vs_eager",
            "qwen_edit_per_step_compiled_vs_eager",
            "fit_preview_rgb_factors",
            "lightning_loras_apply_cleanly",
            "edit_lightning_user_lora_reference_repro",
        ):
            for job in jobs:
                self.assertNotIn(
                    name,
                    workflow_code(bodies[job]),
                    f"{job}: {name} is excluded for a recorded reason",
                )

        # Trap 4: measured, and nowhere near GitHub's 360-minute ceiling (sc-16981).
        for job, cap in (("mlx-qwen-image", 90), ("mlx-qwen-image-pid", 60), ("mlx-qwen-image-producers", 60)):
            self.assertIn(f"timeout-minutes: {cap}", bodies[job])

        # The producers job is dispatch-only and must surrender its evidence, or it spends a Metal
        # box producing a gitignored file the next checkout deletes.
        producers = bodies["mlx-qwen-image-producers"]
        self.assertNotIn("github.event_name == 'schedule'", producers)
        self.assertIn("actions/upload-artifact", producers)
        self.assertIn("if-no-files-found: error", producers)

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
