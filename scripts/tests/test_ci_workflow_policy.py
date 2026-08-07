"""Regression tests for trust boundaries around persistent self-hosted CI runners."""

import functools
import ntpath
import os
import re
import shutil
import subprocess
import sys
import textwrap
import tomllib
import unittest

import yaml
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
        # 28 since sc-15520 added the `mlx-chroma-memory-ladder` job
        # (27 since sc-17284 added the `mlx-qwen-image`, `mlx-qwen-image-pid` and
        # `mlx-qwen-image-producers` jobs; 24 since sc-17250 added the JoyCaption and
        # MOSS-TTS-Realtime jobs; 22 before).
        MACOS_HUB_LOCK: 28,
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
        self.assertEqual(workflow.count(MACOS_HUB_LOCK), 28)
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
        self.assertIn("migrate_mage_edit_variant_manifest:", workflow)
        self.assertIn(
            "default: false",
            workflow[workflow.index("migrate_mage_edit_variant_manifest:") :],
        )
        migration = workflow[
            workflow.index("\n      - name: Migrate only the copied Mage edit-variant manifest") :
            workflow.index(
                "\n      - name: Verify restored or operator-provisioned Mage oracle cache"
            )
        ]
        self.assertIn("inputs.profile == 'media'", migration)
        self.assertIn("inputs.migrate_mage_edit_variant_manifest", migration)
        self.assertIn('golden_root="$(cd "$MAGE_GOLDEN_DIR" && pwd -P)"', migration)
        self.assertIn('runner_root="$(cd "$RUNNER_TEMP" && pwd -P)"', migration)
        self.assertIn('seed_root="$(cd "$MAGE_ORACLE_SEED_DIR" && pwd -P)"', migration)
        self.assertIn('"$golden_root" != "$runner_root/"*', migration)
        self.assertIn('"$golden_root" == "$seed_root"', migration)
        self.assertIn(" -ef ", migration)
        self.assertIn("--migrate-reference-environment-manifest-only", migration)
        self.assertNotIn("dump_mage_flow_golden.py", migration)
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
        self.assertEqual(workflow.count("--gen \"$MAGE_SNAPSHOT\""), 3)
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
        exactly the outcome whose measured cells someone needs to read, and re-running costs
        hours.
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
        # sc-17655 made them dispatch inputs so the sweep can be run in row-sized pieces instead of
        # one indivisible 4.3 h block, which means the count is DERIVED rather than the literal 21
        # that only ever held for the full seven-row sweep. Both halves are pinned: the job must read
        # the inputs, and the inputs must still DEFAULT to the full recorded sweep, or a bare
        # dispatch would quietly measure something narrower than this lane claims.
        self.assertIn("KREA_S18_ROWS: ${{ inputs.krea_s18_rows }}", job)
        self.assertIn("KREA_S18_SEEDS: ${{ inputs.krea_s18_seeds }}", job)
        # PARSED, not string-matched. Substring assertions over the pre-`jobs:` text were tried and
        # are false greens twice over: `assertIn("krea_s18_rows:", ...)` matches the key inside a
        # `#` comment, and `assertIn("default: ABCDFEZ", ...)` is unanchored, so swapping the two
        # defaults between the rows and seeds inputs still passed. Both were demonstrated.
        inputs = yaml.safe_load(REAL_WEIGHTS_WORKFLOW.read_text(encoding="utf-8"))[True][
            "workflow_dispatch"
        ]["inputs"]
        # NOT the full ABCDFEZ. sc-17324 established by measurement that two rows cannot run on
        # this lane's runner at the pinned 832x480: F (~49 GiB) took nax-macos-2 down twice and
        # belongs at 640x384, and E (~63-69 GiB) is unmeasurable on this infrastructure at either
        # bucket. The memory preflight refuses both, so defaulting to all seven would be a default
        # that always fails. Pinned so the convenience of "just put them back" has to argue with
        # two crashed runners first.
        self.assertEqual(inputs["krea_s18_rows"]["default"], "ABCDZ")
        self.assertNotIn("E", inputs["krea_s18_rows"]["default"])
        self.assertNotIn("F", inputs["krea_s18_rows"]["default"])
        self.assertEqual(inputs["krea_s18_seeds"]["default"], "7,11,23")
        # Name-selected, so `--exact` after the `--` plus a run count — the sc-17250 false-green shape.
        self.assertIn("set -o pipefail", job)
        self.assertIn("-- --exact --ignored --nocapture", job)
        self.assertIn('grep -qE "test result: ok\\. 1 passed"', job)
        self.assertIn("expected_cells=$(( n_rows * n_seeds ))", job)
        self.assertIn('"$cells" -ne "$expected_cells"', job)
        # A free-text row list is a new way to make the count assertion vacuous: a typo'd letter
        # selects fewer rows than it counts, and a repeated one counts a row twice. Both are
        # rejected before the four-hour cargo invocation rather than after it.
        self.assertIn("=~ ^[ABCDFEZ]+$", job)
        self.assertIn("repeats a row", job)
        # Three shell details that were each a live bug in this step, pinned because every one of
        # them fails SILENTLY — the step still exits non-zero, just with no ::error:: line and no
        # explanation, four hours in:
        #   * `|| true` on the seed count. `grep -c` exits 1 when it counts zero, and under
        #     `bash -e` + `set -o pipefail` that kills the shell at the assignment, making the
        #     "parsed to no seeds" branch unreachable.
        #   * `[:blank:]`, not `[:space:]`, in the duplicate-seed check: `[:space:]` deletes the
        #     newlines separating the seeds, collapsing `7,7` to `77` so the check never fires.
        #   * field-vs-seed count parity, so a value this shell cannot read the way Rust parses it
        #     (`+7`, or one wider than u64) fails now rather than as a cell-count mismatch later.
        # Anchored to the seed-count line: a bare `|| true` also appears on the `cells=` line
        # below, so the loose form still passed with this one deleted (demonstrated).
        self.assertIn("[0-9]{1,19}[[:blank:]]*$' || true)\"", job)
        # Same class, pre-dating sc-17655 and previously unpinned: a sweep that emits ZERO cells
        # makes this `grep -c` exit 1 too, so without `|| true` the step dies before it can report
        # "captured 0" — the one diagnosis that matters when nothing was measured.
        self.assertIn("'^S18CELL' \"$RUNNER_TEMP/s18-sweep.log\" || true)\"", job)
        self.assertIn("tr -d '[:blank:]'", job)
        self.assertIn('"$n_seeds" -ne "$n_fields"', job)
        self.assertIn("parsed to no seeds", job)
        self.assertIn("repeats a seed", job)
        # Pieces of a split sweep must be distinguishable: same sha, same inner filename, so without
        # the rows/seeds in the artifact name the same piece can be re-aggregated twice.
        self.assertIn(
            "name: krea-s18-sweep-${{ github.sha }}-${{ inputs.krea_s18_rows }}"
            "-s${{ inputs.krea_s18_seeds }}",
            job,
        )
        # sc-17324: the memory preflight must run, must see the rows actually dispatched, and must
        # NOT be continue-on-error — refusing a row this host cannot hold is its entire purpose.
        # It replaces two runner deaths, both row F at 832x480, both of which destroyed every cell
        # measured up to that point because a dying runner takes the `always()` steps with it.
        self.assertIn("scripts/ci/s18_memory_preflight.py", job)
        preflight = job.split("- name: S18 memory preflight", 1)[1].split("- name:", 1)[0]
        self.assertIn("KREA_S18_ROWS: ${{ inputs.krea_s18_rows }}", preflight)
        self.assertNotIn("continue-on-error", preflight)
        # The guard and the run must read the SAME geometry, or the preflight clears one bucket
        # while the sweep measures another — which is precisely the hole that let an unguarded
        # row F reach the runner. Both steps take it from the one dispatch input.
        self.assertIn("KREA_S18_GEOMETRY: ${{ inputs.krea_s18_geometry }}", preflight)
        sweep = job.split("- name: Run the S18 coherence sweep", 1)[1]
        self.assertIn("KREA_S18_GEOMETRY: ${{ inputs.krea_s18_geometry }}", sweep)
        self.assertNotIn('KREA_SMOKE_W: "832"', job)
        self.assertEqual(inputs["krea_s18_geometry"]["default"], "832x480")
        # The evidence must outlive a failing sweep: teed to a file inside the run step, then
        # extracted and uploaded from steps that run whatever the sweep did.
        self.assertIn('tee "$RUNNER_TEMP/s18-sweep.log"', job)
        # Three, not two: sc-17355's GPU/memory evidence report is the third `always()` step, and it
        # is counted here rather than exempted — the point of a count is that a fourth has to be
        # argued for.
        self.assertEqual(job.count("if: always()"), 3)
        self.assertIn("actions/upload-artifact@", job)
        self.assertIn("krea-s18-sweep-${{ github.sha }}", job)
        # The sampler's CSV rides along with the cells: on a 4.3-hour run the summary this job
        # prints is a fraction of what the trajectory holds, and re-running to get it costs 4.3 hours.
        self.assertIn("${{ runner.temp }}/gpu-fault-evidence/memory.csv", job)

    def test_krea_lanes_record_gpu_fault_evidence_with_a_predicate_that_parses(self) -> None:
        """sc-17355: a Metal command-buffer cascade names no cause, so the record must pre-exist.

        Run 30869410054 failed the LoRA gate with `kIOGPUCommandBufferCallbackErrorSubmissionsIgnored`
        — "ignored for causing prior/excessive GPU errors", i.e. an EARLIER submission faulted and
        this one was dropped. That earlier fault is the only thing that names a cause, and it is not
        in the run log. Nor can it be recovered afterwards: the re-dispatch that passed destroyed the
        machine state that would have explained it. So the collection has to be in place BEFORE the
        recurrence, on both lanes that drive this crate on `rw-krea`.

        The predicate is pinned because the obvious one is inert. `subsystem == "com.apple.gpu"` —
        which sc-17355 itself proposed — matches ZERO events on macOS 25.5.0: IOGPU faults carry an
        empty `subsystem` and are identifiable only by `senderImagePath`. Measured both ways on
        nax-macos over the same 3-day window: sender-based matching returns real IOGPU faults, the
        subsystem form returns nothing at all. A capture that silently matches nothing is worse than
        no capture, because an empty file reads as evidence of absence.
        """
        workflow = REAL_WEIGHTS_WORKFLOW.read_text(encoding="utf-8")
        bounds = {
            "  mlx-krea-realtime:": "\n  mlx-krea-realtime-s18-sweep:",
            "  mlx-krea-realtime-s18-sweep:": "\n  candle-audio-kokoro:",
        }
        # The sampler bound is per-caller because the two lanes are not the same length. The sweep
        # is `timeout-minutes: 480` and really runs ~4.3 h; the 7200s default (the regression lane's
        # `timeout-minutes: 120`) would stop recording less than halfway through it, silently, in
        # the half where accumulated pressure is most likely. A shared constant cannot serve both.
        expected_start = {
            "  mlx-krea-realtime:": "scripts/ci/gpu_fault_evidence.sh start\n",
            "  mlx-krea-realtime-s18-sweep:": "scripts/ci/gpu_fault_evidence.sh start 30000\n",
        }
        for header, terminator in bounds.items():
            start = workflow.index(header)
            job = workflow[start : workflow.index(terminator, start)]
            with self.subTest(job=header.strip()):
                self.assertIn(expected_start[header], job)
                self.assertIn("scripts/ci/gpu_fault_evidence.sh report", job)
                # SCOPED TO EACH STEP, not counted across the job. A `job.count(...)` of
                # `continue-on-error: true` is a false green: both evidence steps could lose the
                # key and the count would still be satisfied by unrelated steps elsewhere in the
                # job. The properties asserted here are per-step, so the slice must be per-step.
                for step_name, must_have in (
                    ("- name: Start GPU fault evidence", ("continue-on-error: true",)),
                    (
                        "- name: Report GPU fault evidence",
                        # The report explains a FAILING run, so it must not be skipped by one.
                        ("continue-on-error: true", "if: always()"),
                    ),
                ):
                    body = job[job.index(step_name) :]
                    body = body[: body.index("run:")]
                    for key in must_have:
                        self.assertIn(key, body, f"{step_name}: missing {key}")

                # THE RAW RECORD MUST OUTLIVE `$RUNNER_TEMP`, on BOTH lanes. The `report` step
                # prints a summary — a peak line, a 30-row tail, a histogram — and a diagnosis
                # needs the 1 Hz CSV and the full event list, which the runner deletes when the
                # job ends. The regression lane is the one that actually failed (30869410054), so
                # shipping retention only on the operator-dispatch sweep would have put the record
                # everywhere except where the fault was seen.
                self.assertIn("actions/upload-artifact@", job)
                self.assertIn("gpu-fault-evidence/memory.csv", job)
                self.assertIn("gpu-fault-evidence/gpu-events.txt", job)

        # ONE TEST PER PROCESS in the LoRA step, and this is a correctness constraint rather than a
        # style one. sc-17355 made `render()` call `reset_peak_memory()`, which is a process-global
        # MLX mutation. The step is safe only because `run_one` invokes cargo once per test with
        # `--exact`; collapsing both names into a single invocation would let libtest's default
        # thread pool run them concurrently, and each would rebase the other's high-water mid-render
        # — silently corrupting the per-arm figures this lane now reports.
        lora_step = workflow[workflow.index("- name: Run Krea Realtime real Wan LoRA gates") :]
        lora_step = lora_step[: lora_step.index("- name: Report GPU fault evidence")]
        self.assertIn('"$name" -- --exact --ignored --nocapture', lora_step)
        self.assertEqual(lora_step.count("run_one real_wan_"), 2)

        script_path = REAL_WEIGHTS_WORKFLOW.parents[2] / "scripts" / "ci" / "gpu_fault_evidence.sh"
        script = script_path.read_text(encoding="utf-8")
        # Comments stripped first: the header explains the inert predicate in prose, and that
        # explanation is precisely why it must not silently reappear in the command itself.
        code = "\n".join(
            line for line in script.splitlines() if not line.lstrip().startswith("#")
        )
        self.assertNotIn('subsystem == "com.apple.gpu"', code)
        self.assertIn('senderImagePath CONTAINS "IOGPU"', code)
        # `log stream` drops messages under load, and load is the only condition of interest: a
        # local run that ended in a memory kill produced 20+ "Messages dropped during live
        # streaming" markers and none of the events, while `log show` over the same window
        # returned them. The reader must stay post-hoc.
        self.assertIn("/usr/bin/log show", code)
        self.assertNotIn("log stream", code)
        # The hypothesis under test is memory pressure, and the kernel names the mechanism outright
        # (`killing due to "vm-compressor-space-shortage"`). None of those lines contain "IOGPU" or
        # "command buffer", so a GPU-only predicate throws away the most direct evidence there is.
        self.assertIn('eventMessage CONTAINS[c] "memorystatus"', code)
        # `log` records its own argument vector, which contains "IOGPU" — without this exclusion
        # every run matches itself and reports a spurious hit, which is as useless as reporting none.
        self.assertIn('processImagePath != "/usr/bin/log"', code)
        # The sibling `report_runner_disk_headroom.sh` names a lost exec bit as the one failure it
        # cannot see; git tracks the mode, so pin it here instead of discovering it on a real box.
        mode = subprocess.run(
            ["git", "ls-files", "-s", "--", "scripts/ci/gpu_fault_evidence.sh"],
            capture_output=True,
            text=True,
            # `test_script_encoding` requires this explicitly: the locale default decodes these
            # gates' output as something other than UTF-8 on Windows, where this suite also runs.
            encoding="utf-8",
            cwd=script_path.parents[2],
            check=False,
        ).stdout.split(" ", 1)[0]
        self.assertEqual(mode, "100755")

        # THE NAME OF THIS TEST HAS TO BE EARNED. Everything above is substring matching, which
        # cannot tell a working predicate from a malformed one — and a malformed predicate fails
        # the same silent way the story's inert one did: `log show` exits non-zero, the script's
        # `else` branch prints "log show failed", the lane stays green, and the capture records
        # nothing. So actually run it.
        #
        # THE NAME SAYS "parses", NOT "matches", and the difference is the point. This probe catches
        # a MALFORMED predicate; it cannot catch a well-formed one that matches nothing. Appending
        # `AND subsystem == "com.apple.iokit.IOGPUFamily"` — i.e. reintroducing exactly the inert
        # clause sc-17355 proposed — parses fine and would keep this green while collecting zero
        # events. No assertion here can close that: "matches something" needs an event known to be
        # in the archive at test time, and nothing is. The guard against inertness is the
        # `assertNotIn('subsystem == "com.apple.gpu"')` above plus the report's own unclassified
        # count, not this probe. Naming it `…_that_matches` claimed a check that does not exist.
        #
        # HONEST SCOPE, because this is the exact trap the change is about. `scripts/tests` runs on
        # `ubuntu-latest` in ci.yml and NOWHERE ELSE, so this assertion never executes in CI — it is
        # a developer-machine gate that fires for anyone running the suite on a Mac, which is where
        # this script is written and where the lanes it guards run. Do not read a green CI as having
        # checked the predicate. The runtime backstop is `report` printing "log show failed", which
        # does run on the lanes.
        if sys.platform != "darwin":
            self.skipTest(
                "`log show` is macOS-only, and this suite runs on ubuntu-latest in CI — the "
                "predicate parse check fires only on a developer Mac"
            )
        predicate = re.search(r"--predicate '(.+?)' \\\n", code, re.DOTALL)
        self.assertIsNotNone(predicate, "could not extract the predicate from the script")
        probe = subprocess.run(
            ["/usr/bin/log", "show", "--last", "1s", "--style", "compact",
             "--predicate", predicate.group(1)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        # Assert on the PARSE, not on the exit code. `log show` can fail for reasons that say
        # nothing about the predicate — a sandboxed or restricted host, a busy log archive — and
        # failing the suite on those would be a flake that teaches people to ignore this test.
        # A malformed predicate is unambiguous and specific: `log: Bad predicate (...)`.
        self.assertNotIn(
            "Bad predicate",
            probe.stderr,
            f"the shipped predicate does not parse: {probe.stderr.strip()}",
        )

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

        Each of the 24 selections has to survive all three traps at once. `--exact` AFTER the `--`,
        because cargo rejects it in its own argument position; a run-count assertion, because with
        `--exact` accepted a renamed test yields `0 passed; N filtered out` and cargo EXITS 0; and a
        NAME, because `--ignored` alone is a blanket that silently conscripts whatever `#[ignore]`
        test lands in the file next -- which is exactly how an 85-minute sweep joined a 20-minute
        regression lane in sc-17276.

        Three tests are deliberately absent and must stay absent, and the excluded tuple below is
        the list -- `perf.rs` x2 (sc-17513), which FAIL on real weights and always have, and
        `edit_lightning_user_lora_reference_repro`, a bug-repro harness needing a user LoRA and a
        reference PPM that exist in no repository and on no Hub. A red weekly lane is ignored within
        a month, so all three are recorded with their reason in `release/real-weight-models.toml`
        rather than wired red or quietly dropped.

        Keep this paragraph in step with both lists. TWO names left it in the same week, and each
        had been excluded on a number that measured the test rather than the code:
        `fit_preview_rgb_factors` (sc-17515), whose R^2 = 0.0114 was its own host readback, and
        `lightning_loras_apply_cleanly` (sc-17518), whose 840 was a host-map target count against
        pinned lightx2v files that apply 720 with zero unmatched. Both are now selected below. The
        docstring outliving either change would have made THIS test -- the enforcement point for
        doc-vs-reality drift about what runs where -- an instance of the defect class it exists to
        catch.

        sc-17519 added the 24th, `edit_generate_is_deterministic_rust`, and the arithmetic of what it
        did NOT add is the point. `edit_real_weights.rs` x12 and `vision_real_weights.rs` x7 -- 19
        tests -- were left running nowhere by sc-17284 while the per-VARIABLE manifest gate read as
        satisfied. All 19 were executed on real weights for the first time on 2026-08-06: ONE
        resolves no golden and passes (34-44s, the selection added below), and the other 18 failed at
        `Weights::from_file` with `SafeTensors(NotFile)` in 0.02s, because `tools/golden/` is
        gitignored by design and no checkout had the 10 artifacts they read (sc-17909).

        Of those 18, ONE was free to unblock and is no longer among them.
        `edit_rope_multi_image_matches_fork` reads a golden whose producer imports one symbol from the
        MIT-licensed fork and touches no snapshot, no HF cache and no torch original, so sc-17519
        minted it, committed it to `mlx-gen-qwen-image/tests/fixtures/` (78,133 bytes) and DROPPED the
        test's `#[ignore]`. It runs in the ordinary macOS lane on every PR, needs no weight set, and
        is therefore not a candidate for selection here at all. 17 still fail on the golden read, and
        they read 9 artifacts, not 10.

        Note the two different 18s, because conflating them is how this row got its last wrong number:
        18 is the count that FAILED on the golden read (19 minus the wired determinism test), and 18
        is also, separately, the count of tests `QWEN_IMAGE_EDIT_SNAPSHOT` gates -- which is 11 + 3
        from these two files plus 4 elsewhere, and excludes the 5 here that read no environment
        variable at all. The manifest row derives both.

        Of the 17, only 13 are blocked on the Edit-2511 oracle bundle (sc-17909). The other 4 -- the
        index/Gate-A gates in `vision_real_weights.rs` -- need no weights on either side, and are
        blocked solely because `tools/dump_qwen_vision_golden.py` writes its two weight-free halves
        into the same output file as a third that loads the torch original. That is sc-18085, filed
        apart so it does not wait on a ~60 GB decision it has no stake in.

        Those 17 are deliberately NOT in the excluded tuple below, and that is the same call sc-17503's
        21 T2I golden-parity tests got: the tuple pins tests that are RUNNABLE and excluded anyway on a
        measured number, so a future reader has to be told why a green lane skips them. A test blocked
        on a missing fixture should be wired the moment the fixture exists, and listing it here would
        create a second place to remember to unlist it. The manifest row carries the accounting.
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
                "lightning_loras_apply_cleanly",
                # sc-17515. Wired out of the exclusion list below: its R^2 = 0.0114 was the test's
                # own host readback scrambling the samples, not the fit. It also scores the shipping
                # `preview::RGB_FACTORS` unchanged, so it is a drift gate and belongs on the weekly
                # schedule rather than in the dispatch-only producers job.
                "fit_preview_rgb_factors",
                # sc-17519. The only one of the 19 tests behind `QWEN_IMAGE_EDIT_SNAPSHOT` that
                # resolves no golden, so the only one wirable before the sc-17909 oracle bundle. It
                # renders the same edit twice and asserts the decoded images are byte-identical
                # (0/3145728 bytes differ, measured), which makes it a real gate on the whole
                # LM + vision + transformer + VAE path rather than a load smoke.
                "edit_generate_is_deterministic_rust",
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
        # Bind the docstring's count to the list rather than to a maintainer's memory. The prose and
        # the list drifted apart for exactly as long as the docstring also called
        # `fit_preview_rgb_factors` excluded while the list above required it (sc-17515 review) --
        # stale prose in the one test whose subject is doc-vs-reality drift. Either side moving alone
        # now fails here instead of misinforming the next reader.
        documented = re.search(
            r"Each of the (\d+) selections",
            type(self).test_qwen_image_lanes_name_select_every_test_and_pin_its_run_count.__doc__,
        )
        self.assertTrue(documented, "this test's docstring no longer states a selection count")
        self.assertEqual(
            sum(len(names) for names in selected.values()),
            int(documented.group(1)),
            "this test's docstring names a selection count that no longer matches `selected`",
        )

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
            # A merge group builds a `gh-readonly-queue/main/**` ref in this repository from heads
            # that already passed the fork guard as PRs, so it is trusted exactly like a push. If
            # this case ever flips to False the CUDA lane stops gating the merge queue while still
            # looking green, which is the failure this whole boundary exists to prevent.
            ("merge_group", "", True, True),
            ("merge_group", "", False, False),
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

    def test_merge_queue_speculative_ref_can_reach_the_workflow(self) -> None:
        triggers = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))[True]
        self.assertIn(
            "merge_group",
            triggers,
            "without a merge_group trigger no run is created for the queue's speculative ref, "
            "so required checks stay pending and every queued PR is evicted on timeout",
        )

    def test_every_base_sha_resolves_on_a_merge_group_event(self) -> None:
        # merge_group carries neither `pull_request.base.sha` nor `before`. An empty base is not
        # uniformly loud: select_lanes.py hard-errors, but check-review-findings.py drops its
        # append-only comparison and still exits 0. Assert on every BASE_SHA in the file so a new
        # consumer cannot be added with the two-element chain.
        assignments = [
            line.strip()
            for line in WORKFLOW.read_text(encoding="utf-8").splitlines()
            if line.strip().startswith("BASE_SHA:")
        ]
        self.assertTrue(assignments, "expected at least one BASE_SHA assignment in the workflow")
        for assignment in assignments:
            with self.subTest(assignment=assignment):
                self.assertIn("github.event.merge_group.base_sha", assignment)

    def test_gate_aggregates_every_lane_and_runs_when_they_fail(self) -> None:
        workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
        jobs = workflow["jobs"]
        gate = jobs["gate"]

        # `if: always()` is the load-bearing half. The default `success()` would skip the gate
        # precisely when an upstream lane failed, and a skipped job satisfies a required status
        # check -- so the gate would report green on exactly the runs it exists to block.
        self.assertEqual(gate["if"], "always()")

        ungated = sorted(set(jobs) - {"gate"} - set(gate["needs"]))
        self.assertEqual(
            ungated,
            [],
            f"jobs missing from the CI gate's needs, so nothing enforces them: {ungated}",
        )

    def test_gate_distinguishes_a_path_skip_from_a_failed_dependency(self) -> None:
        # Both arrive as "the job did not run", but only one is benign: a lane skipped by its `if:`
        # is a real path-based no-op, while a lane skipped because `needs: changes` failed means the
        # verdict is unknown. The gate must accept `skipped` (or docs-only PRs deadlock) and reject
        # everything else (or a failed `changes` reads green through its skipped dependents).
        step = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]["gate"]["steps"][0]
        self.assertIn("success|skipped)", step["run"])
        self.assertIn("exit 1", step["run"])
        self.assertIn("join(needs.*.result", step["env"]["RESULTS"])


if __name__ == "__main__":
    unittest.main()
