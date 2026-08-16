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
KREA_ALTERNATE_DECODER_SMOKE = (
    WORKFLOW.parents[2] / "scripts" / "ci" / "run_krea_alternate_decoder_smoke.sh"
)
KREA_ALTERNATE_DECODER_EXAMPLE = (
    WORKFLOW.parents[2]
    / "crates"
    / "media"
    / "mlx-gen"
    / "mlx-gen-krea"
    / "examples"
    / "alternate_decoder_characterization.rs"
)
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
WINDOWS_SCAIL_HUB_LOCK = (
    ".github/requirements/real-weights-huggingface-hub-windows-x64-py314.txt"
)
WINDOWS_MAGE_LOCK = (
    ".github/requirements/real-weights-mage-verify-windows-x64-py312.txt"
)
MACOS_MAGE_LOCK = (
    "crates/media/mlx-gen/_vendor/mage_flow/requirements-oracles.txt"
)
MACOS_INTERPRETER = "python3.12"
WINDOWS_SETUP_ACTION = "astral-sh/setup-uv@d0cc045d04ccac9d8b7881df0226f9e82c39688e"
WINDOWS_UV_VERSION = 'version: "0.12.3"'
WINDOWS_UV_CACHE = "enable-cache: false"
WINDOWS_UV_INSTALL = (
    "uv python install 3.12.10 --managed-python --no-registry --no-bin --no-config "
    "|| exit /b 1"
)
WINDOWS_UV_FIND = (
    "for /f \"delims=\" %%P in ('uv python find 3.12.10 --managed-python "
    "--no-project --no-config') do set \"REVIEWED_PYTHON=%%P\""
)
WINDOWS_PYTHON_EXPORT = 'echo REVIEWED_PYTHON=%REVIEWED_PYTHON%>>"%GITHUB_ENV%"'
WINDOWS_INTERPRETER = r'"%REVIEWED_PYTHON%"'
# sc-18804 pins every Windows real-weight job to the uv-managed reviewed CPython 3.12.10 so a
# runner restart cannot swap the interpreter out from under a hash-locked requirements file.
# `candle-scail2-shared` predates that convention: it provisions against the runner's OWN CPython
# 3.14 under its own py314 hash lock, and hard-fails the job when the runner is not exactly
# CPython 3.14 x64 (`test_scail2_shared_cuda_lane_is_exact_revision_provider_exercised_and_measured`
# mutation-guards both the version assertion and the lock). That is a different answer to the same
# hazard, not an absence of one, so it is recorded here as a named exception rather than silently
# tolerated. `windows_reviewed_interpreter_exemption_errors` below re-proves the exempt job still
# carries its own pinning, so the exemption cannot decay into an unpinned job.
WINDOWS_REVIEWED_INTERPRETER_EXEMPT_JOBS = {"candle-scail2-shared"}
APPROVED_REAL_WEIGHT_LOCKS = {
    MACOS_HUB_LOCK,
    WINDOWS_HUB_LOCK,
    WINDOWS_SCAIL_HUB_LOCK,
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


def windows_reviewed_interpreter_exemption_errors(job: str, lines: list[str]) -> list[str]:
    """Re-prove that an exempt Windows job still pins its own interpreter.

    The exemption in ``WINDOWS_REVIEWED_INTERPRETER_EXEMPT_JOBS`` is a recorded exception, not a
    hole: an exempt job must still (a) refuse the registry-dependent ``setup-python``, (b) assert
    the exact interpreter it was hash-locked against and abort when the runner does not match, and
    (c) terminate the job on a failed hash-locked install. Delete any of those and this fails, so
    the exemption cannot quietly decay into an unpinned job.
    """
    errors: list[str] = []
    body = "\n".join(lines)
    if any(
        re.search(r"\buses:\s*actions/setup-python@", line.split("#", 1)[0])
        for line in lines
    ):
        errors.append(
            f"{job}: Windows jobs must not invoke the registry-dependent setup-python"
        )
    if not re.search(r"sys\.version_info|platform\.python_version\(\)", body):
        errors.append(
            f"{job}: an exempt Windows job must assert its exact interpreter version"
        )
    if not re.search(r"platform\.machine\(\)", body):
        errors.append(
            f"{job}: an exempt Windows job must assert its exact interpreter architecture"
        )
    # The abort has to live in the SAME step as the interpreter assertion. Checking the whole job
    # would be satisfied by any unrelated `exit /b 1` — the Git Bash selection carries one — which
    # would let the validation step be gutted while this still passed.
    step_boundary = re.compile(r"^      - ")
    validation_starts = [
        index
        for index, line in enumerate(lines)
        if re.search(r"sys\.version_info|platform\.python_version\(\)", line)
    ]
    for start in validation_starts:
        step_start = start
        while step_start > 0 and not step_boundary.match(lines[step_start]):
            step_start -= 1
        step_end = start + 1
        while step_end < len(lines) and not step_boundary.match(lines[step_end]):
            step_end += 1
        step = "\n".join(lines[step_start:step_end])
        if not re.search(r"\bthrow\b|exit /b 1|exit \$LASTEXITCODE", step):
            errors.append(
                f"{job}: an exempt Windows job must abort when its interpreter check fails"
            )
    if not validation_starts:
        errors.append(
            f"{job}: an exempt Windows job must abort when its interpreter check fails"
        )
    for line in lines:
        command = line.split("#", 1)[0]
        if not re.search(r"\s-m\s+pip\s+install\b", command, re.IGNORECASE):
            continue
        if "--require-hashes" not in command:
            errors.append(f"{job}: exempt install must stay hash-locked: {command.strip()!r}")
    if re.search(r"-m\s+pip\s+install", body, re.IGNORECASE) and not re.search(
        r"\$LASTEXITCODE -ne 0|\|\|\s+exit /b 1", body
    ):
        errors.append(f"{job}: exempt install must terminate the job on failure")
    return errors


def real_weight_windows_interpreter_errors(workflow: str) -> list[str]:
    """Pin every Windows real-weight Python command to the reviewed CPython 3.12.

    The Windows wheel locks are platform- and interpreter-specific. A runner restart changed bare
    ``python`` from 3.12 to 3.14, so pip selected cp314 wheels whose hashes correctly differed from
    the reviewed cp312 lock. ``setup-python`` then blocked behind an operator-observed elevation
    prompt, which unattended CI cannot satisfy. Guard the no-registry uv provisioning plus every
    producer and consumer, not only pip: mixing interpreters around a shared ``PYTHONPATH`` is
    equally unsafe. ``cmd`` also continues after a failed command, so every hash-locked install must
    explicitly terminate the step.

    Jobs in ``WINDOWS_REVIEWED_INTERPRETER_EXEMPT_JOBS`` answer the same hazard with their own
    hard interpreter validation instead; they are checked by
    ``windows_reviewed_interpreter_exemption_errors`` rather than skipped outright.
    """
    errors: list[str] = []
    interpreter = re.compile(
        rf"({re.escape(WINDOWS_INTERPRETER)}|"
        r"(?<![\w./$\"'-])py(?:\s+-\d+(?:\.\d+)?)?(?![\w-])|"
        r"(?<![\w./$\"'-])python[0-9.]*(?:\.exe)?(?![\w-]))",
        re.IGNORECASE,
    )
    setup_action = re.compile(
        rf"\s*(?:-\s*)?uses:\s*{re.escape(WINDOWS_SETUP_ACTION)}\s*$"
    )
    setup_version = re.compile(rf"\s*{re.escape(WINDOWS_UV_VERSION)}\s*$")
    setup_cache = re.compile(rf"\s*{re.escape(WINDOWS_UV_CACHE)}\s*$")
    for job, lines in workflow_job_bodies(workflow).items():
        runs_on = next((line for line in lines if line.startswith("    runs-on:")), "")
        if "windows" not in runs_on.lower():
            continue
        if job in WINDOWS_REVIEWED_INTERPRETER_EXEMPT_JOBS:
            errors.extend(windows_reviewed_interpreter_exemption_errors(job, lines))
            continue
        setup_indices = [
            index
            for index, line in enumerate(lines)
            if setup_action.fullmatch(line.split("#", 1)[0])
        ]
        if len(setup_indices) != 1:
            errors.append(
                f"{job}: expected exactly one pinned Windows setup-uv action, "
                f"found {len(setup_indices)}"
            )
            setup_index = len(lines)
        else:
            setup_index = setup_indices[0]
            setup_block = lines[setup_index : setup_index + 6]
            if not any(
                setup_version.fullmatch(line.split("#", 1)[0]) for line in setup_block
            ):
                errors.append(f"{job}: setup-uv must install exact uv 0.12.3")
            if not any(
                setup_cache.fullmatch(line.split("#", 1)[0]) for line in setup_block
            ):
                errors.append(f"{job}: setup-uv cache must remain disabled")
        if any(
            re.search(r"\buses:\s*actions/setup-python@", line.split("#", 1)[0])
            for line in lines
        ):
            errors.append(
                f"{job}: Windows jobs must not invoke the registry-dependent setup-python"
            )

        exact_commands = [line.split("#", 1)[0].strip() for line in lines]
        install_indices = [
            index for index, command in enumerate(exact_commands) if command == WINDOWS_UV_INSTALL
        ]
        find_indices = [
            index for index, command in enumerate(exact_commands) if command == WINDOWS_UV_FIND
        ]
        export_indices = [
            index
            for index, command in enumerate(exact_commands)
            if command == WINDOWS_PYTHON_EXPORT
        ]
        if len(install_indices) != 1:
            errors.append(
                f"{job}: expected exactly one fail-fast managed CPython 3.12.10 install, "
                f"found {len(install_indices)}"
            )
        if len(find_indices) != 1:
            errors.append(
                f"{job}: expected exactly one managed CPython 3.12.10 path resolution, "
                f"found {len(find_indices)}"
            )
        if len(export_indices) != 1:
            errors.append(
                f"{job}: expected exactly one reviewed Python path export, "
                f"found {len(export_indices)}"
            )
        install_index = install_indices[0] if len(install_indices) == 1 else len(lines)
        find_index = find_indices[0] if len(find_indices) == 1 else len(lines)
        export_index = export_indices[0] if len(export_indices) == 1 else len(lines)
        if not setup_index < install_index < find_index < export_index:
            errors.append(
                f"{job}: setup-uv, managed install, path resolution, and export are out of order"
            )
        for index, line in enumerate(lines):
            command = line.split("#", 1)[0]
            stripped = command.strip()
            if stripped in {WINDOWS_UV_INSTALL, WINDOWS_UV_FIND}:
                continue
            found_interpreters = list(interpreter.finditer(command))
            looks_like_python_command = bool(
                re.search(r"\s-m\s+pip\b|scripts[/\\][^\s]+\.py\b", command, re.IGNORECASE)
            )
            if looks_like_python_command and WINDOWS_INTERPRETER not in command:
                errors.append(
                    f"{job}: Windows Python command must use {WINDOWS_INTERPRETER}: "
                    f"{command.strip()!r}"
                )
            for found in found_interpreters:
                name = found.group(1)
                if name != WINDOWS_INTERPRETER:
                    errors.append(
                        f"{job}: Windows steps must name {WINDOWS_INTERPRETER}, found "
                        f"{name!r} in {command.strip()!r}"
                    )
                elif index <= find_index:
                    errors.append(
                        f"{job}: Windows Python runs before the reviewed path is resolved: "
                        f"{command.strip()!r}"
                    )
            if re.search(r"\s-m\s+pip\s+install\b", command, re.IGNORECASE) and not re.search(
                r"\|\|\s+exit /b 1\s*$", command
            ):
                errors.append(
                    f"{job}: Windows pip install must fail the cmd step: {command.strip()!r}"
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

    install_lines: list[tuple[int, str, str | None, re.Match[str]]] = []
    current_job: str | None = None
    job_at_line: dict[int, str | None] = {}
    for line_number, line in enumerate(workflow.splitlines(), start=1):
        job_match = re.fullmatch(r"  ([A-Za-z0-9_-]+):", line)
        if job_match is not None:
            current_job = job_match.group(1)
        job_at_line[line_number] = current_job
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
        install_lines.append((line_number, command, job_at_line[line_number], match))

    locks_seen: list[str] = []
    for line_number, command, job, install_match in install_lines:
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
        elif f"{WINDOWS_INTERPRETER} -m pip" in command:
            expected_lock = WINDOWS_HUB_LOCK
        elif "python -m pip" in command:
            # Only the SCAIL-2 shared-package job is allowed to reach the runner's own CPython;
            # see WINDOWS_REVIEWED_INTERPRETER_EXEMPT_JOBS. Any other job spelling it this way
            # has no expected lock and fails below.
            expected_lock = (
                WINDOWS_SCAIL_HUB_LOCK if job == "candle-scail2-shared" else None
            )
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
        # 33 since sc-18325 added the three correctness-only decode-quality jobs;
        # 30 since sc-18315 added pinned Krea license materialization;
        # 29 since sc-18249 added the `mlx-sana-drift-ceiling` job
        # (28 since sc-15520 added the `mlx-chroma-memory-ladder` job;
        # 27 since sc-17284 added the `mlx-qwen-image`, `mlx-qwen-image-pid` and
        # `mlx-qwen-image-producers` jobs; 24 since sc-17250 added the JoyCaption and
        # MOSS-TTS-Realtime jobs; 22 before).
        MACOS_HUB_LOCK: 33,
        WINDOWS_HUB_LOCK: 10,
        WINDOWS_SCAIL_HUB_LOCK: 1,
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


def decode_quality_candidate_rows(job: dict) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    candidate_env = job.get("env", {}).get("DECODE_QUALITY_CANDIDATES", "")
    if candidate_env != "${{ matrix.candidate.value }}":
        return candidate_env.split(), errors
    candidates = job.get("strategy", {}).get("matrix", {}).get("candidate", [])
    rows: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) != {"geometry", "value"}:
            errors.append(f"invalid candidate matrix row {candidate!r}")
            continue
        geometry, value = candidate["geometry"], candidate["value"]
        if not isinstance(geometry, str) or not isinstance(value, str):
            errors.append(f"candidate matrix row must use strings: {candidate!r}")
            continue
        if value.split(":", 1)[0] != geometry:
            errors.append(f"candidate geometry label does not match value: {candidate!r}")
        rows.append(value)
    return rows, errors


def decode_quality_candidate_policy_errors(workflow: str) -> list[str]:
    parsed = yaml.safe_load(workflow)
    jobs = parsed["jobs"]
    rules = {
        "mlx-decode-quality-kolors": 64,
        "mlx-decode-quality-sdxl": 64,
        # Chroma is a DiT with an explicitly validated /16 image-id grid; unlike the
        # Kolors/SDXL U-Net it has no mirrored downsample skip joins, so 720 remains valid.
        "mlx-decode-quality-chroma": 16,
    }
    errors: list[str] = []
    for job, geometry_multiple in rules.items():
        job_config = jobs[job]
        rows, matrix_errors = decode_quality_candidate_rows(job_config)
        errors.extend(f"{job}: {error}" for error in matrix_errors)
        strategy = job_config.get("strategy", {})
        if strategy.get("fail-fast") is not False or strategy.get("max-parallel") != 1:
            errors.append(f"{job}: quality cells must run serialized with fail-fast disabled")
        if not rows:
            errors.append(f"{job}: empty candidate grid")
        if len(rows) != len(set(rows)):
            errors.append(f"{job}: duplicate candidate matrix row")
        for row in rows:
            prefix = f"{job}: invalid candidate {row!r}"
            try:
                geometry, tile_text, overlap_text = row.split(":")
                width_text, height_text = geometry.split("x")
                width, height = int(width_text), int(height_text)
                tile, overlap = int(tile_text), int(overlap_text)
            except (TypeError, ValueError):
                errors.append(f"{prefix}: expected WIDTHxHEIGHT:TILE:OVERLAP")
                continue
            if width <= 0 or height <= 0:
                errors.append(f"{prefix}: geometry must be positive")
            if width % geometry_multiple or height % geometry_multiple:
                errors.append(
                    f"{prefix}: geometry must align to the family /{geometry_multiple} grid"
                )
            if not 0 < overlap < tile <= min(width, height):
                errors.append(
                    f"{prefix}: require 0 < overlap < tile <= min(width,height)"
                )
            if tile % 8 or overlap % 8:
                errors.append(f"{prefix}: tile and overlap must align to the decoder /8 grid")
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

    def test_macos_metal_reclaims_broad_test_artifacts_before_bundle_profiles(self) -> None:
        workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
        steps = workflow["jobs"]["macos-metal"]["steps"]
        names = [step.get("name") for step in steps]
        pre_test_reclaim = "Reclaim Clippy and rustdoc artifacts before linking MLX tests"
        reclaim = "Reclaim broad MLX test artifacts before bundle profiles"
        self.assertEqual(names.count(pre_test_reclaim), 1)
        self.assertEqual(names.count(reclaim), 1)
        self.assertLess(names.index("Rustdoc macOS MLX packages"), names.index(pre_test_reclaim))
        self.assertLess(names.index(pre_test_reclaim), names.index("Test MLX packages"))
        self.assertLess(names.index("Test MLX packages"), names.index(reclaim))
        self.assertLess(names.index(reclaim), names.index("Test LLM-only macOS bundle"))
        self.assertLess(names.index(reclaim), names.index("Test LLM+audio macOS bundle"))
        self.assertLess(names.index(reclaim), names.index("Clippy Candle Metal packages"))
        self.assertEqual(steps[names.index(pre_test_reclaim)]["run"], "cargo clean")
        self.assertEqual(steps[names.index(reclaim)]["run"], "cargo clean")

    def test_real_weight_python_installs_are_binary_hash_locked(self) -> None:
        workflow = REAL_WEIGHTS_WORKFLOW.read_text(encoding="utf-8")
        self.assertEqual(real_weight_pip_policy_errors(workflow), [])
        self.assertEqual(workflow.count(MACOS_HUB_LOCK), 33)
        self.assertEqual(workflow.count(WINDOWS_HUB_LOCK), 10)
        self.assertEqual(workflow.count(WINDOWS_SCAIL_HUB_LOCK), 1)
        self.assertEqual(workflow.count(WINDOWS_MAGE_LOCK), 1)
        self.assertNotRegex(
            workflow,
            r"\bpip\s+install[^\n]*(?:huggingface[_-]hub|numpy|safetensors)==",
        )

    def test_decode_quality_candidates_stay_inside_family_geometry_domains(self) -> None:
        workflow_text = REAL_WEIGHTS_WORKFLOW.read_text(encoding="utf-8")
        self.assertEqual(decode_quality_candidate_policy_errors(workflow_text), [])
        workflow = yaml.safe_load(workflow_text)
        jobs = workflow["jobs"]
        geometries: dict[str, set[tuple[int, int]]] = {}
        for job in (
            "mlx-decode-quality-kolors",
            "mlx-decode-quality-sdxl",
            "mlx-decode-quality-chroma",
        ):
            geometries[job] = set()
            rows, errors = decode_quality_candidate_rows(jobs[job])
            self.assertEqual(errors, [])
            for row in rows:
                geometry = row.split(":", 1)[0]
                width_text, height_text = geometry.split("x")
                geometries[job].add((int(width_text), int(height_text)))

        self.assertIn((1280, 768), geometries["mlx-decode-quality-kolors"])
        self.assertIn((768, 1280), geometries["mlx-decode-quality-kolors"])
        self.assertNotIn((1280, 720), geometries["mlx-decode-quality-kolors"])
        self.assertNotIn((720, 1280), geometries["mlx-decode-quality-kolors"])
        self.assertIn((1280, 720), geometries["mlx-decode-quality-chroma"])
        self.assertIn((720, 1280), geometries["mlx-decode-quality-chroma"])

        expected_cells = {
            "mlx-decode-quality-kolors": 7,
            # SC-19753 diagnostic branch: the first 30 cells are already sealed, so this
            # branch dispatches only the two remaining Illustrious families.
            "mlx-decode-quality-sdxl": 20,
            "mlx-decode-quality-chroma": 12,
        }
        for job, cells in expected_cells.items():
            with self.subTest(job=job):
                config = jobs[job]
                matrix = config["strategy"]["matrix"]
                route_count = len(matrix.get("model", ["kolors"]))
                self.assertEqual(route_count * len(matrix["candidate"]), cells)
                self.assertEqual(
                    config["env"]["DECODE_QUALITY_CANDIDATES"],
                    "${{ matrix.candidate.value }}",
                )
                self.assertIn("${{ matrix.candidate.geometry }}", config["name"])
                upload = next(
                    step
                    for step in config["steps"]
                    if str(step.get("uses", "")).startswith("actions/upload-artifact@")
                )
                artifact_name = upload["with"]["name"]
                self.assertTrue(artifact_name.startswith("decode-quality-v2-"))
                self.assertIn("${{ matrix.candidate.geometry }}", artifact_name)
                if job != "mlx-decode-quality-kolors":
                    self.assertIn("${{ matrix.model.id }}", artifact_name)
                self.assertEqual(upload["with"]["if-no-files-found"], "error")
                collector_step = next(
                    step["run"]
                    for step in config["steps"]
                    if "collect_decode_quality_admission.py" in step.get("run", "")
                )
                self.assertIn("--expected-policy-count 1", collector_step)
                self.assertIn("--expected-fixture-count 5", collector_step)
        self.assertEqual(sum(expected_cells.values()), 69)

        mutations = {
            "Kolors landscape geometry": ("1280x768:576:48", "1280x720:576:48"),
            "Kolors portrait geometry": ("768x1280:576:48", "720x1280:576:48"),
            "SDXL geometry": ("1216x832:704:160", "1216x816:704:160"),
            "Chroma geometry": ("1280x720:576:192", "1280x722:576:192"),
            "zero geometry": ("768x768:576:48", "0x768:576:48"),
            "zero overlap": ("768x768:576:48", "768x768:576:0"),
            "overlap equals tile": ("768x768:576:48", "768x768:576:576"),
            "tile exceeds geometry": ("768x768:576:48", "768x768:776:48"),
            "tile off decoder grid": ("768x768:576:48", "768x768:578:48"),
            "overlap off decoder grid": ("768x768:576:48", "768x768:576:50"),
        }
        for label, (before, after) in mutations.items():
            with self.subTest(mutation=label):
                self.assertIn(before, workflow_text)
                mutated = workflow_text.replace(before, after, 1)
                self.assertTrue(decode_quality_candidate_policy_errors(mutated))

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

    def test_real_weight_windows_steps_name_reviewed_cpython_and_fail_fast(self) -> None:
        workflow = REAL_WEIGHTS_WORKFLOW.read_text(encoding="utf-8")
        self.assertEqual(real_weight_windows_interpreter_errors(workflow), [])
        windows_python_lines = [
            line
            for lines in workflow_job_bodies(workflow).values()
            if any(
                candidate.startswith("    runs-on:")
                and "windows" in candidate.lower()
                for candidate in lines
            )
            for line in lines
            if WINDOWS_INTERPRETER in line and not line.lstrip().startswith("#")
        ]
        self.assertEqual(len(windows_python_lines), 76)
        self.assertEqual(workflow.count(f"uses: {WINDOWS_SETUP_ACTION}"), 11)
        self.assertEqual(workflow.count(WINDOWS_UV_VERSION), 11)
        self.assertEqual(workflow.count(WINDOWS_UV_CACHE), 11)
        self.assertEqual(workflow.count(WINDOWS_UV_INSTALL), 11)
        self.assertEqual(workflow.count(WINDOWS_UV_FIND), 11)
        self.assertEqual(workflow.count(WINDOWS_PYTHON_EXPORT), 11)
        self.assertIn(
            f'{WINDOWS_INTERPRETER} -c "import sys; assert sys.version_info[:3] == (3, 12, 10), sys.version"',
            workflow,
        )
        pip_installs = [
            line.strip()
            for line in windows_python_lines
            if f"{WINDOWS_INTERPRETER} -m pip install" in line
        ]
        self.assertEqual(len(pip_installs), 11)
        for install in pip_installs:
            self.assertRegex(install, r"\|\|\s+exit /b 1$")
        for replacement in ("python", "py", "py -3.14"):
            with self.subTest(replacement=replacement):
                mutated = workflow.replace(WINDOWS_INTERPRETER, replacement, 1)
                self.assertTrue(real_weight_windows_interpreter_errors(mutated))
        first_windows_install = pip_installs[0]
        no_fail_fast = workflow.replace(
            first_windows_install, first_windows_install.removesuffix(" || exit /b 1"), 1
        )
        self.assertTrue(real_weight_windows_interpreter_errors(no_fail_fast))
        whitespace_install_without_fail_fast = first_windows_install.replace(
            "pip install", "pip  install", 1
        ).removesuffix(" || exit /b 1")
        whitespace_bypass = workflow.replace(
            first_windows_install, whitespace_install_without_fail_fast, 1
        )
        self.assertNotEqual(whitespace_bypass, workflow)
        self.assertTrue(real_weight_windows_interpreter_errors(whitespace_bypass))
        missing_setup = workflow.replace(f"uses: {WINDOWS_SETUP_ACTION}", "uses: missing", 1)
        self.assertTrue(real_weight_windows_interpreter_errors(missing_setup))
        wrong_version = workflow.replace(WINDOWS_UV_VERSION, 'version: "0.99.0"', 1)
        self.assertTrue(real_weight_windows_interpreter_errors(wrong_version))
        provisioning_mutations = {
            "wrong Python": WINDOWS_UV_INSTALL.replace("3.12.10", "3.14.0"),
            "registry mutation allowed": WINDOWS_UV_INSTALL.replace(" --no-registry", ""),
            "launcher mutation allowed": WINDOWS_UV_INSTALL.replace(" --no-bin", ""),
            "system Python allowed": WINDOWS_UV_FIND.replace(" --managed-python", ""),
        }
        for name, replacement in provisioning_mutations.items():
            with self.subTest(provisioning_mutation=name):
                mutated = workflow.replace(
                    WINDOWS_UV_INSTALL
                    if replacement.startswith("uv python install")
                    else WINDOWS_UV_FIND,
                    replacement,
                    1,
                )
                self.assertTrue(real_weight_windows_interpreter_errors(mutated))
        cache_enabled = workflow.replace(WINDOWS_UV_CACHE, "enable-cache: true", 1)
        self.assertTrue(real_weight_windows_interpreter_errors(cache_enabled))
        wrong_setup_comment_decoy = workflow.replace(
            f"uses: {WINDOWS_SETUP_ACTION}",
            f"uses: astral-sh/setup-uv@{'0' * 40} # uses: {WINDOWS_SETUP_ACTION}",
            1,
        )
        self.assertTrue(real_weight_windows_interpreter_errors(wrong_setup_comment_decoy))
        wrong_version_comment_decoy = workflow.replace(
            WINDOWS_UV_VERSION,
            f'version: "0.99.0" # {WINDOWS_UV_VERSION}',
            1,
        )
        self.assertTrue(real_weight_windows_interpreter_errors(wrong_version_comment_decoy))

    def test_windows_reviewed_interpreter_exemption_still_requires_its_own_pinning(self) -> None:
        """The sc-18804 exemption for `candle-scail2-shared` must not be a blank hole.

        The exempt job answers the same hazard its own way — it hard-validates the runner's
        CPython 3.14 x64 and installs from a matching py314 hash lock. Each of those properties is
        mutated ONE AT A TIME: mutating them together would only prove the set is load-bearing,
        not that any individual member is.
        """
        workflow = REAL_WEIGHTS_WORKFLOW.read_text(encoding="utf-8")
        self.assertEqual(real_weight_windows_interpreter_errors(workflow), [])
        bodies = workflow_job_bodies(workflow)
        for job in WINDOWS_REVIEWED_INTERPRETER_EXEMPT_JOBS:
            self.assertIn(job, bodies, f"exempt job {job} no longer exists — drop the exemption")
            lines = bodies[job]
            self.assertEqual(windows_reviewed_interpreter_exemption_errors(job, lines), [])
            mutations = {
                "version assertion dropped": ("platform.python_version()", "'3.14.0'"),
                "architecture assertion dropped": ("platform.machine()", "'AMD64'"),
                "hash lock dropped": ("--require-hashes", ""),
                "failure no longer aborts": ("throw ", "Write-Host "),
                "registry setup-python reintroduced": (
                    "- uses: dtolnay/rust-toolchain@",
                    "- uses: actions/setup-python@v5\n      - uses: dtolnay/rust-toolchain@",
                ),
            }
            for mutation, (needle, replacement) in mutations.items():
                with self.subTest(job=job, mutation=mutation):
                    mutated = [line.replace(needle, replacement) for line in lines]
                    self.assertNotEqual(mutated, lines, f"{mutation} changed nothing")
                    self.assertTrue(
                        windows_reviewed_interpreter_exemption_errors(job, mutated)
                    )

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
        scail_windows = validate_binary_hashed_lock(
            (REAL_WEIGHT_REQUIREMENTS / Path(WINDOWS_SCAIL_HUB_LOCK).name).read_text(
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
        self.assertEqual(scail_windows["huggingface-hub"][0], "1.20.1")
        self.assertEqual(scail_windows["hf-xet"][0], "1.6.0")
        self.assertEqual(scail_windows["packaging"][0], "26.3")
        self.assertNotEqual(macos["hf-xet"][1], windows["hf-xet"][1])
        self.assertNotEqual(macos["pyyaml"][1], windows["pyyaml"][1])
        self.assertNotEqual(windows["pyyaml"][1], scail_windows["pyyaml"][1])
        scail_lock = (
            REAL_WEIGHT_REQUIREMENTS / Path(WINDOWS_SCAIL_HUB_LOCK).name
        ).read_text(encoding="utf-8")
        self.assertIn("--platform win_amd64 --python-version 3.14", scail_lock)
        self.assertIn("--implementation cp --abi cp314", scail_lock)

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
        # Thirteen SA3/SAME exporters, SC-18309's exact SDXL-VAE projection, and SC-18315's
        # q4 Krea correctness projection and standalone Wan donor plus exact-file materialization.
        self.assertEqual(workflow.count("export_model_snapshot_paths.py"), 15)
        self.assertEqual(workflow.count("--model krea-2-turbo-mlx-q4"), 2)
        self.assertEqual(
            workflow.count("--model krea-realtime-14b-mlx-wan-z16-vae-q8"), 2
        )
        self.assertEqual(workflow.count("--model sdxl-base-mlx-vae-bf16"), 2)
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
        mlx_media = "\n".join(workflow_job_bodies(workflow)["mlx-media"])
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
        self.assertNotIn("uses: actions/setup-python", mlx_media)
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
        seed_import = workflow[
            workflow.index("\n      - name: Require operator-provisioned Mage oracle seed") :
            workflow.index("\n      - name: Require an uncached operator seed")
        ]
        self.assertIn("mage_flow_e2e_golden.png", seed_import)
        self.assertIn("mage_flow_edit_golden.png", seed_import)
        self.assertIn("migrate_mage_edit_variant_manifest:", workflow)
        self.assertIn(
            "default: false",
            workflow[workflow.index("migrate_mage_edit_variant_manifest:") :],
        )
        self.assertIn(
            "Converge and durably certify the exact Mage manifest pair on the active rw-mage seed",
            workflow,
        )
        self.assertIn(
            "mage_seed_slot: [single]",
            workflow,
        )
        recovery_index = workflow.index(
            "\n      - name: Recover any interrupted persistent Mage seed promotion"
        )
        restore_index = workflow.index("\n      - name: Restore verified Mage oracle cache")
        seed_index = workflow.index(
            "\n      - name: Require operator-provisioned Mage oracle seed"
        )
        self.assertLess(recovery_index, restore_index)
        self.assertLess(restore_index, seed_index)
        restore = workflow[restore_index:seed_index]
        self.assertIn("id: mage-oracle-cache", restore)
        self.assertIn("github.event_name != 'workflow_dispatch'", restore)
        self.assertIn("inputs.profile != 'media'", restore)
        self.assertIn("inputs.migrate_mage_edit_variant_manifest != true", restore)
        recovery = workflow[recovery_index:restore_index]
        self.assertIn("scripts/release/promote_mage_oracle_seed.py", recovery)
        self.assertIn("--recover-only", recovery)
        self.assertIn("id: mage-seed-recovery", recovery)
        self.assertIn('--runner-name "$RUNNER_NAME"', recovery)
        self.assertIn('--slot "${{ matrix.mage_seed_slot }}"', recovery)
        self.assertIn('--revision "$GITHUB_SHA"', recovery)
        self.assertIn('echo "completed=true" >> "$GITHUB_OUTPUT"', recovery)
        prepare_index = workflow.index("\n      - name: Prepare pinned Mage reference environment")
        classify_index = workflow.index("\n      - name: Classify the copied Mage manifest pair")
        migration_index = workflow.index(
            "\n      - name: Migrate only the copied Mage edit-variant manifest"
        )
        verify_index = workflow.index(
            "\n      - name: Verify restored or operator-provisioned Mage oracle cache"
        )
        self.assertLess(prepare_index, classify_index)
        self.assertLess(classify_index, migration_index)
        self.assertLess(migration_index, verify_index)
        preparation = workflow[prepare_index:classify_index]
        self.assertIn('python -m venv "$RUNNER_TEMP/mage-reference"', preparation)
        self.assertIn(
            '"$RUNNER_TEMP/mage-reference/bin/python" -m pip install', preparation
        )
        self.assertIn("requirements-oracles.txt", preparation)
        classification = workflow[classify_index:migration_index]
        self.assertIn("id: mage-seed-state", classification)
        self.assertIn("provision_mage_edit_variants.py", classification)
        self.assertIn("verify_mage_candle_transfer.py", classification)
        self.assertIn('echo "current=true" >> "$GITHUB_OUTPUT"', classification)
        self.assertIn('echo "current=false" >> "$GITHUB_OUTPUT"', classification)
        migration = workflow[migration_index:verify_index]
        self.assertIn("inputs.profile == 'media'", migration)
        self.assertIn("inputs.migrate_mage_edit_variant_manifest", migration)
        self.assertIn(
            "steps.mage-seed-recovery.outputs.completed != 'true'", migration
        )
        self.assertIn("steps.mage-seed-state.outputs.current != 'true'", migration)
        self.assertIn('golden_root="$(cd "$MAGE_GOLDEN_DIR" && pwd -P)"', migration)
        self.assertIn('runner_root="$(cd "$RUNNER_TEMP" && pwd -P)"', migration)
        self.assertIn('seed_root="$(cd "$MAGE_ORACLE_SEED_DIR" && pwd -P)"', migration)
        self.assertIn('"$golden_root" != "$runner_root/"*', migration)
        self.assertIn('"$golden_root" == "$seed_root"', migration)
        self.assertIn(" -ef ", migration)
        self.assertIn('"$RUNNER_TEMP/mage-reference/bin/python"', migration)
        self.assertNotIn("python3.12 scripts/release/provision_mage_edit_variants.py", migration)
        self.assertIn("--migrate-reference-environment-manifest-only", migration)
        self.assertIn("--migrate-edit-variant-manifest-hash-only", migration)
        self.assertIn('transfer_links="$(stat -f \'%l\' "$transfer_manifest")"', migration)
        self.assertIn('"$transfer_links" != "1"', migration)
        self.assertNotIn("dump_mage_flow_golden.py", migration)
        precondition_index = workflow.index(
            "\n      - name: Require an uncached operator seed for durable Mage certification"
        )
        self.assertLess(seed_index, precondition_index)
        self.assertLess(precondition_index, migration_index)
        precondition = workflow[precondition_index:migration_index]
        self.assertIn("steps.mage-oracle-cache.outputs.cache-hit", precondition)
        self.assertIn("steps.mage-oracle-seed.outputs.imported", precondition)
        seed_import = workflow[seed_index:precondition_index]
        self.assertIn('echo "edit-sha=$edit_sha"', seed_import)
        self.assertIn(
            'echo "transfer-sha=$transfer_sha"', seed_import
        )
        self.assertIn('} >> "$GITHUB_OUTPUT"', seed_import)
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
            "scripts/release/promote_mage_oracle_seed.py",
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
        self.assertGreaterEqual(workflow.count("--edit-snapshot \"$MAGE_EDIT_SNAPSHOT\""), 6)
        self.assertEqual(workflow.count("--gen \"$MAGE_SNAPSHOT\""), 8)
        provider_index = workflow.index("\n      - name: Run provider conformance")
        edit_parity_index = workflow.index(
            "\n      - name: Run Mage-Flow instruction-edit parity"
        )
        promote_index = workflow.index(
            "\n      - name: Atomically promote the verified Mage manifests"
        )
        persistent_verify_index = workflow.index(
            "\n      - name: Verify the promoted persistent Mage oracle seed"
        )
        receipt_index = workflow.index(
            "\n      - name: Upload persistent Mage seed promotion receipt"
        )
        cache_save_index = workflow.index("\n      - name: Save verified Mage oracle cache")
        edit_upload_index = workflow.index("\n      - name: Upload verified Mage edit oracle")
        self.assertLess(provider_index, edit_parity_index)
        self.assertLess(edit_parity_index, promote_index)
        self.assertLess(promote_index, persistent_verify_index)
        self.assertLess(persistent_verify_index, receipt_index)
        self.assertLess(receipt_index, cache_save_index)
        self.assertLess(cache_save_index, edit_upload_index)
        promotion = workflow[promote_index:persistent_verify_index]
        self.assertIn(
            "steps.mage-seed-recovery.outputs.completed != 'true'", promotion
        )
        for argument in (
            "--source \"$MAGE_GOLDEN_DIR\"",
            "--seed \"$MAGE_ORACLE_SEED_DIR\"",
            "steps.mage-oracle-seed.outputs.edit-sha",
            "steps.mage-oracle-seed.outputs.transfer-sha",
            "--runner-name \"$RUNNER_NAME\"",
            "--slot \"${{ matrix.mage_seed_slot }}\"",
            "--revision \"$GITHUB_SHA\"",
            "--allow-already-current",
        ):
            self.assertIn(argument, promotion)
        persistent_verify = workflow[persistent_verify_index:receipt_index]
        self.assertEqual(persistent_verify.count('--output "$MAGE_ORACLE_SEED_DIR"'), 5)
        self.assertIn(
            "mage-seed-promotion-${{ matrix.mage_seed_slot }}-${{ github.sha }}",
            workflow[receipt_index:cache_save_index],
        )
        cache_save = workflow[cache_save_index:edit_upload_index]
        self.assertIn("inputs.migrate_mage_edit_variant_manifest != true", cache_save)
        self.assertNotIn("matrix.mage_seed_slot != 'secondary'", workflow)
        promotion_gate_index = workflow.index("\n  mage-seed-promotion-gate:")
        qwen_index = workflow.index("\n  # sc-17284", promotion_gate_index)
        promotion_gate = workflow[promotion_gate_index:qwen_index]
        self.assertIn("needs: mlx-media", promotion_gate)
        self.assertIn("if: always()", promotion_gate)
        self.assertIn("needs.mlx-media.result", promotion_gate)
        self.assertIn("merge-multiple: true", promotion_gate)
        self.assertIn("--verify-receipts", promotion_gate)
        self.assertIn("--revision \"$GITHUB_SHA\"", promotion_gate)
        self.assertIn("Require the exact-run Mage seed certification", promotion_gate)
        candle_media_index = workflow.index("\n  candle-media:")
        self.assertIn(
            "needs: [mlx-media, mage-seed-promotion-gate]",
            workflow[candle_media_index:],
        )
        self.assertLess(
            workflow.index("Verify restored or operator-provisioned Mage oracle cache"),
            cache_save_index,
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

    def test_withdrawn_or_gated_media_models_materialize_from_exact_public_mirrors(self) -> None:
        models = {
            model["key"]: model
            for model in tomllib.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))["models"]
        }
        expected = {
            "flux-1-dev": (
                "black-forest-labs/FLUX.1-dev",
                "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21",
                "SceneWorks/flux1-dev-mlx",
                "323fd12d79f78ad444e882e8d8e871914584f2b9",
                "bf16",
            ),
            "mage-flow": (
                "microsoft/Mage-Flow",
                "faca09c18c1c19458e7fbc3f7bce6f7a7d4d01a9",
                "SceneWorks/Mage-Flow",
                "5f6455818d8ca80ce780e9c01b9e0de1d8c5f9db",
                None,
            ),
            "mage-flow-edit": (
                "microsoft/Mage-Flow-Edit",
                "b01d524f86498b7dabcc4b3572c6d264d786a16e",
                "SceneWorks/Mage-Flow-Edit",
                "dbd4a9c07faca94491ad88ab21225d62e054d9cc",
                None,
            ),
            "mage-flow-edit-base": (
                "microsoft/Mage-Flow-Edit-Base",
                "8654a7bc0283ab2946385230b5b2eb944e0b76ea",
                "SceneWorks/Mage-Flow-Edit-Base",
                "6c119cdac7ce7cf8c1ab4990d9c8ca18641f2c5d",
                None,
            ),
            "mage-flow-edit-turbo": (
                "microsoft/Mage-Flow-Edit-Turbo",
                "14427bd7627d3a25436497a5939e1096f6a0d523",
                "SceneWorks/Mage-Flow-Edit-Turbo",
                "75c11a2957aca2c78272984375502105b2b235ab",
                None,
            ),
        }
        mage_download_files = [
            "model_index.json",
            "scheduler/*",
            "text_encoder/*",
            "transformer/*",
            "vae/*",
        ]
        mage_materialization_files = [
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/.gitattributes",
            "text_encoder/README.md",
            "text_encoder/chat_template.json",
            "text_encoder/config.json",
            "text_encoder/generation_config.json",
            "text_encoder/merges.txt",
            "text_encoder/model-00001-of-00002.safetensors",
            "text_encoder/model-00002-of-00002.safetensors",
            "text_encoder/model.safetensors.index.json",
            "text_encoder/preprocessor_config.json",
            "text_encoder/tokenizer.json",
            "text_encoder/tokenizer_config.json",
            "text_encoder/video_preprocessor_config.json",
            "text_encoder/vocab.json",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ]
        flux_materialization_files = [
            "LICENSE.md",
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/model.safetensors",
            "text_encoder_2/config.json",
            "text_encoder_2/model-00001-of-00002.safetensors",
            "text_encoder_2/model-00002-of-00002.safetensors",
            "text_encoder_2/model.safetensors.index.json",
            "tokenizer/merges.txt",
            "tokenizer/special_tokens_map.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/vocab.json",
            "tokenizer_2/special_tokens_map.json",
            "tokenizer_2/spiece.model",
            "tokenizer_2/tokenizer.json",
            "tokenizer_2/tokenizer_config.json",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
            "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
            "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ]
        for key, (
            canonical_repo,
            canonical_rev,
            mirror_repo,
            mirror_rev,
            prefix,
        ) in expected.items():
            with self.subTest(key=key):
                model = models[key]
                self.assertEqual(model["repository"], canonical_repo)
                self.assertEqual(model["revision"], canonical_rev)
                self.assertEqual(model["materialization_repository"], mirror_repo)
                self.assertEqual(model["materialization_revision"], mirror_rev)
                self.assertEqual(model.get("materialization_path_prefix"), prefix)
                self.assertNotIn("materialization_requires_auth", model)
                if key.startswith("mage-flow"):
                    self.assertEqual(model["download_files"], mage_download_files)
                    self.assertEqual(
                        model["materialization_expected_files"], mage_materialization_files
                    )
                    self.assertFalse(
                        any(
                            pattern.startswith(("bf16/", "q4/", "q8/"))
                            for pattern in model["download_files"]
                        )
                    )
                else:
                    self.assertEqual(
                        model["materialization_expected_files"], flux_materialization_files
                    )

        workflow = REAL_WEIGHTS_WORKFLOW.read_text(encoding="utf-8")
        self.assertEqual(workflow.count("--require-materialization-provenance"), 6)
        mac_start = workflow.index("  mlx-request-memory-scope:")
        windows_start = workflow.index("  candle-mage-memory-ladder:")
        self.assertEqual(
            workflow[mac_start:windows_start].count("--require-materialization-provenance"),
            6,
        )
        self.assertNotIn("--require-materialization-provenance", workflow[windows_start:])

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

    def test_native_decode_seam_real_weight_gates_are_exact_and_golden_free(self) -> None:
        workflow_text = REAL_WEIGHTS_WORKFLOW.read_text(encoding="utf-8")
        workflow = yaml.safe_load(workflow_text)
        cases = {
            "mlx-media": (
                "Prove the SDXL native decode seam on real weights",
                "mlx-gen-sdxl",
                "native_decode_seam_is_byte_exact_to_pre_seam_engine",
                'SDXL_SNAPSHOT="$SDXL_N1_SNAPSHOT/bf16"',
            ),
            "mlx-qwen-image": (
                "Prove the Qwen native decode seam on real weights",
                "mlx-gen-qwen-image",
                "native_decode_seam_is_byte_exact_and_precancelled",
                'MLX_GEN_QWEN_SNAPSHOT="$QWEN_IMAGE_MLX_SNAPSHOT/bf16"',
            ),
        }
        for job, (step_name, package, test_name, snapshot_binding) in cases.items():
            with self.subTest(job=job):
                steps = workflow["jobs"][job]["steps"]
                matching = [step for step in steps if step.get("name") == step_name]
                self.assertEqual(len(matching), 1)
                run = matching[0]["run"]
                self.assertIn("set -o pipefail", run)
                self.assertIn(snapshot_binding, run)
                self.assertIn(f"cargo test --locked --release -p {package}", run)
                self.assertIn(f"--test vae_real_weights", run)
                self.assertIn(test_name, run)
                self.assertIn("-- --exact --ignored --nocapture", run)
                self.assertIn('grep -qE "test result: ok\\. 1 passed"', run)
                self.assertNotIn("GOLDEN", run)

        models = {
            model["key"]: model
            for model in tomllib.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))["models"]
        }
        sdxl = models["sdxl-base-mlx-vae-bf16"]
        self.assertEqual(sdxl["repository"], "SceneWorks/sdxl-base-mlx")
        self.assertEqual(sdxl["revision"], "36699bb8a6353e61c920e3bf19f0e6f8e4151c55")
        self.assertEqual(sdxl["environment"], ["SDXL_N1_SNAPSHOT"])
        expected = [
            "bf16/vae/config.json",
            "bf16/vae/diffusion_pytorch_model.fp16.safetensors",
        ]
        self.assertEqual(sdxl["download_files"], expected)
        self.assertEqual(sdxl["expected_files"], expected)

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

    def test_memory_evidence_v1_lane_is_artifact_bound_tolerance_pinned_and_operator_dispatched(self) -> None:
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
        # sc-18149: the lane must NOT pin a sub-tiling geometry — 512 degenerates the
        # Sequential tiled decode to a byte-exact single tile, dodging the drift the declared
        # tolerance exists to bound.
        self.assertNotIn("ZIMAGE_SEQ_SIZE:", job)
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
        self.assertIn("sequential_bounds_peak_within_declared_decode_drift", job)
        self.assertIn("--ignored --exact --test-threads=1 --nocapture", job)
        self.assertIn("set -o pipefail", job)
        self.assertIn("verify_residency_ab.py", job)
        self.assertIn("--min-reduction-mib 512", job)
        self.assertIn("--expected-fingerprint z-image-mlx-independent-materialization-v3", job)
        self.assertIn("--expected-abi 3", job)
        # sc-18149: the lane pins the adjudicated tolerance contract from outside the harness.
        self.assertIn("--expected-parity tolerance:mean_abs_u8_subpixel:4.0", job)
        # sc-18149 review: the p99 tail pin and the isolator binding are lane-pinned so the
        # verifier — not the harness — enforces both, and deleting either seam reddens here.
        self.assertIn("--max-p99-abs-u8 13", job)
        self.assertIn(
            '--isolator-output "$MEMORY_EVIDENCE_OUTPUT_DIR/z_image_turbo-resident-tiled.rgb"',
            job,
        )
        self.assertIn("--expected-model-revision", job)
        self.assertIn("--expected-model-inventory-sha256", job)
        self.assertIn("z_image_turbo-resident.rgb", job)
        self.assertIn("z_image_turbo-staged.rgb", job)
        self.assertIn("z_image_turbo-resident-tiled.rgb", job)
        self.assertIn("actions/upload-artifact@", job)
        self.assertNotIn("if: always()", job)
        self.assertIn("z-image-turbo-model-inventory.json", job)
        self.assertIn("verifier-result.txt", job)
        self.assertIn("memory-evidence-v1-z-image-${{ github.sha }}", job)

    def test_scail2_shared_cuda_lane_is_exact_revision_provider_exercised_and_measured(self) -> None:
        workflow = REAL_WEIGHTS_WORKFLOW.read_text(encoding="utf-8")
        start = workflow.index("  candle-scail2-shared:")
        end = workflow.index("\n  candle-media:", start)
        job = workflow[start:end]

        self.assertIn("scail2", workflow.split("jobs:", 1)[0])
        self.assertIn(
            "if: github.event_name == 'workflow_dispatch' && inputs.profile == 'scail2'",
            job,
        )
        self.assertNotIn("github.event_name == 'schedule'", job)
        self.assertIn('SCAIL2_REPOSITORY: "SceneWorks/scail2-mlx"', job)
        self.assertIn(
            'SCAIL2_REVISION: "ce88cfdb1008f395e9c820e525e6db7b6695f7b3"', job
        )
        self.assertIn('allow_patterns=["bf16/**"]', job)
        self.assertIn("models--SceneWorks--scail2-mlx", job)
        git_bash = job.index("Select Git Bash")
        validate_python = job.index("Validate runner-provisioned CPython 3.14 x64")
        toolchain = job.index("uses: dtolnay/rust-toolchain@")
        initialize_evidence = job.index("Initialize exact SCAIL CUDA evidence")
        provision = job.index("Provision the exact public shared bf16 package")
        self.assertLess(git_bash, toolchain)
        self.assertLess(git_bash, initialize_evidence)
        self.assertLess(initialize_evidence, validate_python)
        self.assertLess(validate_python, toolchain)
        self.assertLess(validate_python, provision)
        self.assertLess(initialize_evidence, provision)
        self.assertIn(r'C:\Program Files\Git\bin\bash.exe', job)
        self.assertNotIn("actions/setup-python@", job)
        self.assertNotIn("py -3.12", job)
        self.assertIn("sys.implementation.name", job)
        self.assertIn("platform.python_version()", job)
        self.assertIn("platform.machine()", job)
        self.assertIn("struct.calcsize('P') * 8", job)
        self.assertIn("$python -notmatch '\\|cpython\\|3\\.14\\.\\d+\\|AMD64\\|64$'", job)
        self.assertIn(WINDOWS_SCAIL_HUB_LOCK, job)
        self.assertNotIn(WINDOWS_HUB_LOCK, job)
        for required in (
            "config.json",
            "dit.safetensors",
            "t5_encoder.safetensors",
            "tokenizer.json",
            "clip.safetensors",
            "vae.safetensors",
        ):
            self.assertIn(required, job)
        self.assertIn(
            "pipeline::tests::shared_bf16_real_weights_cuda_loads_and_renders_with_measured_peak",
            job,
        )
        self.assertIn("Load through the production provider", job)
        self.assertIn("[[SCAIL2_CUDA_VRAM]]", job)
        self.assertIn("-- --ignored --exact --nocapture", job)
        self.assertIn('"provision_status=validating_python"', job)
        self.assertIn('"provision_status=complete"', job)
        # Windows PowerShell 5.1 promotes a successful native command's stderr
        # (including Cargo build warnings) to NativeCommandError under Actions'
        # stop-on-error wrapper. Keep PowerShell evidence capture on its valid
        # FilePath parameter set, but run the Cargo profile through the selected
        # Git Bash with pipefail so warnings are logged and real failures still
        # propagate through tee.
        profile_start = job.index(
            "- name: Load through the production provider, minimally render, and measure the shared package"
        )
        profile_end = job.index("- name: Upload exact SCAIL CUDA evidence", profile_start)
        profile = job[profile_start:profile_end]
        self.assertIn("shell: bash", profile)
        self.assertIn("set -o pipefail", profile)
        self.assertIn(
            'evidence_log="$(cygpath -u "$RUNNER_TEMP")/scail2-shared-cuda.log"',
            profile,
        )
        self.assertIn('export PATH="$(cygpath -u "$CUDA_PATH")/bin:$PATH"', profile)
        self.assertIn("cargo test --locked --release", profile)
        self.assertIn('tee -a "$evidence_log"', profile)
        self.assertNotIn("shell: powershell", profile)
        self.assertNotIn("Tee-Object", profile)

        def idle_evidence_errors(value: str) -> list[str]:
            errors = []
            for required in (
                'profile_gpu="${CUDA_VISIBLE_DEVICES%%,*}"',
                'profile_gpu="${profile_gpu:-0}"',
                '[[CUDA_IDLE_RAW]] profileGpu=$profile_gpu',
                "--query-gpu=index,name,driver_version,pstate,utilization.gpu,memory.used,memory.total",
                "for _ in 1 2 3 4 5 6; do",
                'nvidia-smi pmon -i "$profile_gpu" -c 1 -s um',
            ):
                if required not in value:
                    errors.append(f"missing {required}")
            if value.count('-i "$profile_gpu"') != 3:
                errors.append("every raw GPU query must use the rendered physical ordinal")
            return errors

        self.assertEqual(idle_evidence_errors(profile), [])
        for mutation, changed in {
            "missing raw samples": profile.replace("for _ in 1 2 3 4 5 6; do", "for _ in 1; do"),
            "missing process evidence": profile.replace(
                'nvidia-smi pmon -i "$profile_gpu" -c 1 -s um', "echo pmon-omitted"
            ),
            "wrong process GPU": profile.replace(
                'nvidia-smi pmon -i "$profile_gpu"', 'nvidia-smi pmon -i "0"'
            ),
            "missing evidence marker": profile.replace("[[CUDA_IDLE_RAW]]", "[[CUDA_IDLE_OMITTED]]"),
        }.items():
            with self.subTest(idle_evidence_mutation=mutation):
                self.assertTrue(idle_evidence_errors(changed))
        self.assertEqual(job.count("Tee-Object -FilePath $log -Append"), 3)
        self.assertNotIn("Tee-Object -LiteralPath $log -Append", job)
        self.assertIn("actions/upload-artifact@", job)
        self.assertIn("scail2-shared-cuda-${{ github.sha }}", job)

        for name, mutated_job in {
            "py312 lock substitution": job.replace(
                WINDOWS_SCAIL_HUB_LOCK, WINDOWS_HUB_LOCK, 1
            ),
            "shared macOS lock substitution": job.replace(
                WINDOWS_SCAIL_HUB_LOCK, MACOS_HUB_LOCK, 1
            ),
            "unhashed install": job.replace(" --require-hashes", "", 1),
        }.items():
            with self.subTest(mutation=name):
                mutated = workflow[:start] + mutated_job + workflow[end:]
                self.assertTrue(real_weight_pip_policy_errors(mutated))

    def test_sana_drift_ceiling_lane_is_operator_dispatched_and_keeps_its_evidence(self) -> None:
        """sc-18249: the SANA 6.0 drift ceiling must be enforced by a lane a workflow can run.

        The ceiling lived only in `#[ignore]` tests no workflow referenced — the sc-17250 shape
        (pinned snapshot, gates running nowhere). These pins keep the lane from being silently
        gutted: the two ceiling-bearing tests must stay named `--exact` (a rename cannot widen or
        empty the filter), `set -o pipefail` must survive the `tee` (or a red test exits green),
        the snapshot must be inventory-verified before AND after the run, and the log, inventory
        and `SANA_SWEEP_OUT` renders must be uploaded unconditionally-on-success (no `if: always()`
        — a failed run's partial artifacts must not masquerade as evidence). There is deliberately
        NO `verify_residency_ab.py` pin here: no SANA verifier exists, and the adjudicated contract
        (sc-17863) is asserted INSIDE the tests, so the exit code is the verdict.
        """
        workflow = REAL_WEIGHTS_WORKFLOW.read_text(encoding="utf-8")
        start = workflow.index("  mlx-sana-drift-ceiling:")
        end = workflow.index("\n  mlx-memory-evidence-v1:", start)
        job = workflow[start:end]

        self.assertIn("sana-drift-ceiling", workflow.split("jobs:", 1)[0])
        self.assertIn(
            "if: github.event_name == 'workflow_dispatch' && "
            "inputs.profile == 'sana-drift-ceiling'",
            job,
        )
        self.assertIn("INFERENCE_REVISION: ${{ github.sha }}", job)
        self.assertIn("SANA_LADDER_1600M: ${{ vars.SANA_LADDER_1600M }}", job)
        self.assertIn('test "$(git rev-parse HEAD)" = "$INFERENCE_REVISION"', job)
        self.assertIn("git diff --quiet", job)
        self.assertIn("git diff --cached --quiet", job)
        self.assertGreaterEqual(
            job.count('test -z "$(git status --porcelain --untracked-files=normal)"'),
            2,
        )
        self.assertIn("resolve_snapshot_paths.py", job)
        self.assertIn("ensure_model_snapshot.py", job)
        self.assertIn("--model sana-1600m-mlx", job)
        self.assertIn("verify_model_snapshot.py", job)
        self.assertIn("--inventory-output", job)
        self.assertIn("SANA_MODEL_INVENTORY_AFTER", job)
        self.assertIn('cmp -s "$SANA_MODEL_INVENTORY" "$SANA_MODEL_INVENTORY_AFTER"', job)
        # Both ceiling-bearing tests, each `--exact`: the five-latent resample that owns the
        # 6.0 ceiling, and the published-domain sweep that bounds every published edge and
        # asserts the same ceiling on any admitted overlap-probe row.
        self.assertIn("the_tiled_decode_drift_is_resampled_across_production_latents", job)
        self.assertIn(
            "the_published_decode_tile_domain_is_swept_against_the_whole_image_decode", job
        )
        self.assertEqual(job.count("--ignored --exact --test-threads=1 --nocapture"), 2)
        # Each invocation must PROVE it ran exactly one passing test. `cargo test` exits 0 when
        # a filter matches nothing, so a Rust-side rename or a lost `#[ignore]` would leave the
        # YAML names (and every pin above) green while the lane enforces nothing — the sc-15520
        # review-round-2 guard the sibling ladder lanes carry. Per invocation, not aggregate:
        # exactly one guard per cargo run, so one invocation matching two tests can never cover
        # for the other matching none.
        self.assertEqual(job.count('grep -qE "test result: ok\\. 1 passed"'), 2)
        self.assertEqual(
            job.count("cargo test"),
            job.count('grep -qE "test result: ok\\. 1 passed"'),
        )
        self.assertIn("set -o pipefail", job)
        # The sc-17863 verdict was made with eyes on the renders; the lane must keep producing
        # them per run rather than leaving that evidence on one dev Mac.
        self.assertIn("SANA_SWEEP_OUT:", job)
        self.assertIn("actions/upload-artifact@", job)
        self.assertNotIn("if: always()", job)
        self.assertIn("sana-1600m-drift.log", job)
        self.assertIn("sana-1600m-mlx-model-inventory.json", job)
        self.assertIn("/renders", job)
        self.assertIn("sana-drift-ceiling-${{ github.sha }}", job)
        # No SANA verifier exists; the day one appears this pin should flip to a positive
        # requirement rather than being deleted.
        self.assertNotIn("verify_residency_ab.py", job)

    def test_krea_alternate_decoder_smoke_is_explicit_and_correctness_only(self) -> None:
        """SC-18315 keeps its model smoke distinct from memory/calibration capture."""
        workflow = REAL_WEIGHTS_WORKFLOW.read_text(encoding="utf-8")
        jobs = workflow_job_bodies(workflow)
        job = "\n".join(jobs["mlx-krea-alternate-decoder"])
        job_header = job.split("steps:", 1)[0]
        inputs = yaml.safe_load(workflow)[True]["workflow_dispatch"]["inputs"]
        choices = inputs["profile"]["options"]

        self.assertEqual(choices.count("krea-alternate-decoder"), 1)
        self.assertIn(
            "if: github.event_name == 'workflow_dispatch' && "
            "inputs.profile == 'krea-alternate-decoder'",
            job,
        )
        self.assertNotIn("github.event_name == 'schedule'", job_header)
        self.assertNotIn("inputs.profile == 'all'", job_header)
        self.assertIn(
            "runs-on: [self-hosted, macOS, ARM64, nax, rw-sa3]", job
        )
        self.assertIn("scripts/ci/run_krea_alternate_decoder_smoke.sh", job)
        self.assertLess(
            job.index("--model krea-2-turbo-mlx-q4"),
            job.index("scripts/ci/run_krea_alternate_decoder_smoke.sh"),
        )
        self.assertIn("scripts/release/ensure_model_snapshot_file.py", job)
        self.assertIn("--file LICENSE.pdf", job)
        self.assertIn("--model krea-realtime-14b-mlx-wan-z16-vae-q8", job)
        self.assertIn("--file q8/vae.safetensors", job)
        self.assertLess(
            job.index("scripts/release/ensure_model_snapshot_file.py"),
            job.index("scripts/ci/run_krea_alternate_decoder_smoke.sh"),
        )
        self.assertIn("actions/upload-artifact@", job)
        self.assertIn("if-no-files-found: error", job)
        self.assertNotIn("xcrun metal --version", job)
        self.assertNotIn("gpu_fault_evidence.sh", job)
        self.assertNotIn("memory.csv", job)
        self.assertNotIn("--inventory-output", job)

        models = {
            model["key"]: model
            for model in tomllib.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))["models"]
        }
        donor = models["krea-realtime-14b-mlx-wan-z16-vae-q8"]
        self.assertEqual(donor["repository"], "SceneWorks/krea-realtime-14b-mlx")
        self.assertEqual(
            donor["revision"], "e68e9a3d98187fdf6936838ffcf6df5aa48d6626"
        )
        self.assertEqual(donor["download_files"], ["q8/vae.safetensors"])
        self.assertEqual(donor["expected_files"], ["q8/vae.safetensors"])
        self.assertEqual(
            donor["environment"], ["KREA_ALTERNATE_DECODER_WAN_VAE_SNAPSHOT"]
        )

        script = KREA_ALTERNATE_DECODER_SMOKE.read_text(encoding="utf-8")
        self.assertTrue(os.access(KREA_ALTERNATE_DECODER_SMOKE, os.X_OK))
        self.assertIn("d009674080cc1bccf2b629d834c34bf5eccdb723", script)
        self.assertIn("e68e9a3d98187fdf6936838ffcf6df5aa48d6626", script)
        self.assertIn(
            "42159a8b571dbeb3ea40327b88a6161a5342c0511202af7c031360629757163d",
            script,
        )
        self.assertIn("run_characterization 512 0", script)
        self.assertIn("run_characterization 768 1", script)
        self.assertEqual(script.count(".png\n"), 4)
        self.assertIn("sha256.txt", script)
        self.assertIn("provenance.txt", script)
        self.assertNotIn("get_peak_memory", script)
        self.assertNotIn("reset_peak_memory", script)
        self.assertNotIn("gpu_fault_evidence", script)
        self.assertIn('!= "$RUNNER_TEMP"/*', script)

        example = KREA_ALTERNATE_DECODER_EXAMPLE.read_text(encoding="utf-8")
        self.assertIn("KREA_AB_OUTPUT_DIR", example)
        self.assertNotIn("get_peak_memory", example)
        self.assertNotIn("reset_peak_memory", example)
        self.assertNotIn("std::time::Instant", example)
        self.assertEqual(example.count("validate_rgb_output("), 3)
        self.assertIn("usize::try_from(size)", example)
        self.assertIn(".checked_mul(edge)", example)
        self.assertIn("pixels.checked_mul(3)", example)

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
        #
        # sc-17807 adds the GEOMETRY for the same reason, and it is not cosmetic: the geometry input
        # now also carries the KV cache tier (`640x384@q8`), so a bf16 and a q8 dispatch of the same
        # rows and seeds at the same sha would otherwise produce two artifacts with identical names
        # and identical inner filenames — two different MODELS, indistinguishable. The re-aggregator
        # refuses to pool mixed tiers, but only because each cell carries its tier; an artifact you
        # cannot tell apart is still evidence you cannot safely use.
        self.assertIn(
            "name: krea-s18-sweep-${{ github.sha }}-${{ inputs.krea_s18_geometry }}"
            "-${{ inputs.krea_s18_rows }}-s${{ inputs.krea_s18_seeds }}",
            job,
        )
        # sc-17807 — the KV cache tier rides the geometry input as an optional `@q<bits>` suffix
        # (the dispatcher is at its input cap). Both the preflight and the run must derive it, or
        # the guard prices a bf16 sweep while a quantized one runs. The regex is what keeps the two
        # `##*@q` expansions unambiguous, so pin it alongside them.
        self.assertIn("^[0-9]+x[0-9]+(@q[0-9]+)?$", job)
        self.assertEqual(job.count('KREA_S18_KV_BITS="${KREA_S18_GEOMETRY##*@q}"'), 2)
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
        self.assertIn("--skip s18_verdict_from_accumulated_cells", step)
        self.assertIn("--skip s18_kv_tier_ab_from_accumulated_cells", step)
        self.assertIn('grep -qE "test result: ok\\. 6 passed"', step)

    def test_krea_kv_residency_step_runs_the_identity_and_retention_gates(self) -> None:
        """sc-17894: both real-weight acceptance arms must be name-selected and count-pinned."""
        workflow = REAL_WEIGHTS_WORKFLOW.read_text(encoding="utf-8")
        header = workflow.split("jobs:", 1)[0]
        job_start = workflow.index("  mlx-krea-realtime:")
        job = workflow[job_start : workflow.index("\n  mlx-krea-realtime-s18-sweep:", job_start)]
        e2e_start = job.index("      - name: Run Krea Realtime real-weight e2e")
        e2e = job[
            e2e_start : job.index("      - name: Run Krea Realtime KV-cache residency", e2e_start)
        ]
        start = workflow.index("      - name: Run Krea Realtime KV-cache residency")
        step = workflow[
            start : workflow.index("      - name: Run Krea Realtime real Wan LoRA gates", start)
        ]
        lora_start = job.index("      - name: Run Krea Realtime real Wan LoRA gates")
        lora = job[lora_start : job.index("      - name: Report GPU fault evidence", lora_start)]

        self.assertIn("krea-kv-cache", header)
        self.assertIn("inputs.profile == 'krea-kv-cache'", job.split("steps:", 1)[0])
        self.assertIn("if: inputs.profile != 'krea-kv-cache'", e2e)
        self.assertNotIn("if: inputs.profile != 'krea-kv-cache'", step)
        self.assertIn("if: inputs.profile != 'krea-kv-cache'", lora)

        self.assertIn("set -o pipefail", step)
        self.assertIn("-- --exact --ignored --nocapture", step)
        self.assertIn('grep -qE "test result: ok\\. 1 passed"', step)
        invocations = re.findall(r"^\s+run_one (?:integration|lib) .+$", step, re.MULTILINE)
        self.assertEqual(
            invocations,
            [
                "          run_one integration kv_cache_residency_at_the_production_geometry",
                "          run_one lib generate::tests::next_read_eviction_is_bit_identical_to_eager_max_window_retention",
            ],
        )

    def test_qwen_image_lanes_name_select_every_test_and_pin_its_run_count(self) -> None:
        """sc-17284: the three Qwen-Image jobs must keep the contract they were wired under.

        Each of the 26 selections has to survive all three traps at once. `--exact` AFTER the `--`,
        because cargo rejects it in its own argument position; a run-count assertion, because with
        `--exact` accepted a renamed test yields `0 passed; N filtered out` and cargo EXITS 0; and a
        NAME, because `--ignored` alone is a blanket that silently conscripts whatever `#[ignore]`
        test lands in the file next -- which is exactly how an 85-minute sweep joined a 20-minute
        regression lane in sc-17276.

        ONE test is deliberately absent and must stay absent, and the excluded tuple below is the
        list -- `edit_lightning_user_lora_reference_repro`, a bug-repro harness needing a user LoRA
        and a reference PPM that exist in no repository and on no Hub. A red weekly lane is ignored
        within a month, so it is recorded with its reason in `release/real-weight-models.toml`
        rather than wired red or quietly dropped.

        Keep this paragraph in step with both lists. FOUR names left it in the same week, and each
        had been excluded on a number that measured the test rather than the code:
        `fit_preview_rgb_factors` (sc-17515), whose R^2 = 0.0114 was its own host readback;
        `lightning_loras_apply_cleanly` (sc-17518), whose 840 was a host-map target count against
        pinned lightx2v files that apply 720 with zero unmatched; and `perf.rs` x2 (sc-17513), whose
        `max|D| == 0.0` had never held on a 60-layer forward -- an 18-shape sweep plus a depth probe
        and a conditioning control showed the residual to be this stack's amplification of a
        sub-ULP rounding difference, not a fusion defect, and it now carries a peak-relative bound.
        All four are selected below. The docstring outliving any of those changes would have made
        THIS test -- the enforcement point for doc-vs-reality drift about what runs where -- an
        instance of the defect class it exists to catch.

        sc-17519 added the 26th, `edit_generate_is_deterministic_rust`, and the arithmetic of what it
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
        from these two files plus 4 elsewhere (the sequential-residency Edit arm, both Lightning Edit
        arms, and the sc-17513 `perf.rs` Edit arm), and excludes the 5 here that read no environment
        variable at all. The manifest row derives both. sc-17513 moved the `perf.rs` Edit arm from
        that row's EXCLUDED block to its WIRED block without changing the 18: the variable gates the
        same tests either way, which is the property per-variable counting is supposed to have.

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
                # sc-17513. The sc-2963 compiled-glue rollout's only real-weight gate, wired out of
                # the exclusion list once an 18-shape sweep established that its residual is this
                # stack's amplification floor and replaced the bit-exactness assertion with a
                # peak-relative bound.
                "qwen_t2i_per_step_compiled_vs_eager",
                "qwen_edit_per_step_compiled_vs_eager",
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

        # Absent, with an open story. Over CODE only: the steps' comments have to NAME this test to
        # say why it is excluded, and prose can never select a test.
        for name in ("edit_lightning_user_lora_reference_repro",):
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

    def test_windows_cuda_jobs_cap_cargo_parallelism_for_shared_host(self) -> None:
        jobs = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]

        for job_name in ("windows-cuda-check", "windows-cuda"):
            with self.subTest(job=job_name):
                self.assertEqual(
                    jobs[job_name]["env"].get("CARGO_BUILD_JOBS"),
                    "12",
                    "both Windows CUDA listeners share one 48-thread host; an uncapped Cargo "
                    "process on each listener can overwhelm Windows with rustc launches",
                )

    def test_merge_queue_speculative_ref_can_reach_the_workflow(self) -> None:
        triggers = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))[True]
        self.assertIn(
            "merge_group",
            triggers,
            "without a merge_group trigger no run is created for the queue's speculative ref, "
            "so required checks stay pending and every queued PR is evicted on timeout",
        )

    def test_push_on_main_can_produce_the_required_ci_gate_context(self) -> None:
        triggers = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))[True]
        self.assertIn(
            "main",
            (triggers.get("push") or {}).get("branches") or [],
            "ruleset 20481541 requires the `CI gate` context on main and the merge queue that used "
            "to produce it was removed on 2026-08-11, so this post-merge run is the only event that "
            "can produce it: drop `branches: [main]` and every commit landing on main carries zero "
            "check-runs while the required context becomes unproducible (sc-18825)",
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

    def test_feature_epic_policy_runs_in_the_unconditional_gate_path(self) -> None:
        workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
        changes = workflow["jobs"]["changes"]
        policy_steps = [
            step
            for step in changes["steps"]
            if step.get("run") == "python3 scripts/ci/feature_epic_policy.py"
        ]
        self.assertEqual(
            len(policy_steps),
            1,
            "the feature-epic policy must run exactly once in the always-created changes job",
        )
        self.assertNotIn(
            "if",
            policy_steps[0],
            "event or path gating the policy would let an invalid topology omit its verdict",
        )
        self.assertNotIn(
            "if",
            changes,
            "the changes job is the always-run trust boundary for feature-epic topology",
        )
        self.assertIn(
            "changes",
            workflow["jobs"]["gate"]["needs"],
            "CI gate must fail when the topology policy in changes fails",
        )

        triggers = workflow[True]
        self.assertIn("pull_request", triggers)
        self.assertIn("merge_group", triggers)

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
