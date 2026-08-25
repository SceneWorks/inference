"""Enforce that `candle-gen*` integration targets execute in an automatic CI lane.

sc-19447. Every `candle-gen*` integration target used to compile in three lanes and execute in
none of them: `candle-cpu-test` ran `--lib` only, `candle-cpu-lint` reached them through
all-target Clippy without running anything, `windows-cuda-check` built them `--no-run`, and the
only `--tests` runner for the family — `windows-cuda` — is `workflow_dispatch`-only. A whole
epic's candle-side parity evidence was green because it was never evaluated.

The fix is a glob (`cargo test --lib --tests -p 'candle-gen*'`) rather than a curated
`--test` list, so a newly added target is selected the moment it exists. This check pins the
properties that make that true, because each is one careless edit away from silently reverting:

  * the step still passes `--tests` (dropping it restores the original defect verbatim),
  * its package selectors still cover every `candle-gen*` crate that carries at least one
    non-`#[ignore]`d integration case (narrowing the glob to a hand-listed set restores the drift
    the audio side already suffered — see `test_sa3_ci_target_coverage.py`),
  * the step still lives in `candle-cpu-test`, a job gated on `needs.changes.outputs.candle_cpu`
    and required by `gate` — a step that runs in a `workflow_dispatch`-only job, or is `if: false`,
    or whose job no longer blocks the gate, is a step that does not run, however correct its
    command line reads, and
  * no counted case is a **self-skipper** (see below).

SELF-SKIPPERS. A case that prints `SKIP` and `return`s when its env var is unset or its device is
not CUDA is reported by libtest as **passed**, not as skipped. Counting such a case as coverage
reproduces the sc-19447 defect one level up: the lane reports a pass for a case that evaluated
nothing. Measured on `cargo test --locked --lib --tests -p 'candle-gen*'` (210 binaries), the
conversion moved exactly those cases out of the passed column: 3161 passed / 218 ignored before,
3147 passed / 232 ignored after, 0 failed either way. Only two exclusion mechanisms are
trustworthy, and this module recognises only those two:

  * `#[ignore = "..."]` — libtest reports `ignored`, and `--ignored` is the explicit opt-in
    (the convention the repo states for itself in
    `candle-gen-sdxl/tests/preview_real_weights.rs`: "a row that early-returns on an unset
    variable still reports SUCCESS"), and
  * a file-level `#![cfg(...)]` that is off on this lane — the target compiles to an empty binary.

14 cases across 13 files were converted to `#[ignore]` in sc-19447 for exactly this reason, and the
conversion was WRONG ON ONE OF THEM. The 14 split 13 + 1:

  * 13 are ENV-GATED and weight-bound — `candle-gen-flux2/tests/vae_encode_parity.rs` and all
    eleven `candle-gen-lens` `*_parity` targets (12 files). Their weights are on no runner, so
    `#[ignore]` is the right marker and `real-weights.yml` is the explicit opt-in.
  * 1 is DEVICE-GATED, weight-free, and was LANE-REACHABLE:
    `candle-gen/tests/cuda_quant_smoke.rs::cuda_qmatmul_matches_cpu`, the sole canary for the
    sc-7544 silent-zeros GGUF QMatMul regression. Its guard is `default_device().is_cuda()`, not an
    env var, and `ci.yml`'s `windows-cuda` job runs `--features cuda` with no `-- --ignored` — so
    `#[ignore]` silenced it on the only lane that has ever executed it. It now uses the SECOND
    mechanism above, a file-level `#![cfg(feature = "cuda")]`: empty binary on this lane, compiled
    and RUN under `--features cuda`. `compiles_empty_on_this_lane` classifies it accordingly, so it
    is neither counted as weight-free coverage here nor reported by `self_skipping_cases`.

The measured counts quoted above are sc-19447's, taken while all 14 were `#[ignore]`d; correcting
that one case moves it out of the `ignored` column as well, into "not compiled on this lane".

`test_no_counted_case_self_skips` fails on any new self-skipper rather than silently dropping it,
so the class cannot re-enter. Note the consequence, recorded rather than hidden: `candle-gen-lens`
contributes NO weight-free integration coverage at all — all eleven of its `*_parity` targets are
real-weight — so this lane proves nothing about that crate's integration surface.

Scope caveat, recorded so a green run is not over-read: this policies the **weight-free** half.
Weight-bound cases are selected by the manual `windows-cuda` dispatch and by `real-weights.yml`,
and many of them are named by no lane at all. Wiring those is a separate question that sc-19447
records rather than solves.
"""

import fnmatch
import re
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
CANDLE_GEN = ROOT / "crates" / "media" / "candle-gen"
STEP_NAME = "Test Candle gen packages and integration targets"

# The step must live HERE, in a job gated on THIS expression, and that job must block THIS gate.
# Each is independently sufficient to stop the step running while its command line still reads
# correctly, so each is asserted separately.
OWNING_JOB = "candle-cpu-test"
JOB_CONDITION = "needs.changes.outputs.candle_cpu == 'true'"
GATE_JOB = "gate"

# The story that made this concrete: its integration targets are the epic's entire candle-side
# evidence surface, so it is asserted by name as well as by the general rule.
EPIC_CRATE = "candle-gen-minimax-h3"

# `candle-cpu-test` runs on `ubuntu-latest`. Non-feature `cfg` predicates are resolved against that
# lane; see `cfg_holds_on_this_lane`.
LANE_TARGET_OS = "linux"

# The repo's self-skip marker, deliberately case-SENSITIVE. Every one of the 14 real cases printed
# `"SKIP...`; sd3's `reference_manifests_are_consistent_if_present` prints lowercase `"skip {tag}`
# and `continue`s through a loop over committed, in-repo references — it genuinely asserts on every
# run and must NOT be swept up. `test_a_conditional_case_that_still_asserts_is_not_a_self_skipper`
# is that negative control.
SKIP_MARKER = re.compile(r'(?:eprintln|println)!\(\s*(?:\n\s*)?"\s*SKIP')
SKIP_LOOKBACK_LINES = 10

# The one case sc-19447 mis-classified (see the module docstring): device-gated, weight-free, and
# reachable from `windows-cuda`, so `#[ignore]` silenced it rather than parking it. Pinned by name
# because nothing else notices when a canary stops being evaluated — that is the whole failure mode
# it guards against, one level up.
CUDA_CANARY_CRATE = "candle-gen"
CUDA_CANARY_TARGET = "cuda_quant_smoke"
CUDA_CANARY_CASE = "cuda_qmatmul_matches_cpu"
# The `windows-cuda` step that executes it. Named rather than globbed: `windows-cuda-check`'s step
# compiles the same selectors `--no-run` and would satisfy every check below while running nothing.
CUDA_STEP_NAME = "Test Candle CUDA packages"
CUDA_STEP_JOB = "windows-cuda"


def _job_spans(workflow: str) -> dict[str, tuple[int, int]]:
    """Map each top-level job key to the `[start, end)` line span of its block.

    Line-based on purpose: this repo's CI checks do not assume a YAML library is installed on the
    runner (the `workspace` job pip-installs only numpy and safetensors).
    """
    lines = workflow.splitlines()
    jobs_at = next((i for i, l in enumerate(lines) if l.rstrip() == "jobs:"), None)
    if jobs_at is None:
        raise AssertionError("ci.yml has no top-level `jobs:` block")

    key = re.compile(r"^  ([A-Za-z0-9_-]+):\s*$")
    starts: list[tuple[str, int]] = []
    for i in range(jobs_at + 1, len(lines)):
        line = lines[i]
        if line.strip() and not line.startswith(" "):
            break  # left the jobs: block
        m = key.match(line)
        if m:
            starts.append((m.group(1), i))

    spans: dict[str, tuple[int, int]] = {}
    for n, (name, start) in enumerate(starts):
        end = starts[n + 1][1] if n + 1 < len(starts) else len(lines)
        spans[name] = (start, end)
    return spans


def job_of_step(workflow: str, step_name: str = STEP_NAME) -> str:
    """The top-level job key that owns the named step.

    Finding the step by name ANYWHERE in the file is position-blind: the same step relocated into
    the `workflow_dispatch`-only `windows-cuda` job reads identically and runs never.
    """
    lines = workflow.splitlines()
    marker = f"- name: {step_name}"
    at = next((i for i, line in enumerate(lines) if line.strip() == marker), None)
    if at is None:
        raise AssertionError(f"missing ci.yml step: {step_name}")
    for name, (start, end) in _job_spans(workflow).items():
        if start <= at < end:
            return name
    raise AssertionError(f"step {step_name!r} is not inside any job block")


def job_block(workflow: str, job_name: str) -> str:
    spans = _job_spans(workflow)
    if job_name not in spans:
        raise AssertionError(f"missing ci.yml job: {job_name}")
    start, end = spans[job_name]
    return "\n".join(workflow.splitlines()[start:end])


def job_needs(workflow: str, job_name: str) -> list[str]:
    """The `needs:` entries of a job, in either the inline or the block-list form."""
    block = job_block(workflow, job_name).splitlines()
    at = next((i for i, l in enumerate(block) if l.strip().startswith("needs:")), None)
    if at is None:
        return []
    inline = block[at].split("needs:", 1)[1].strip()
    if inline:
        return [n.strip() for n in inline.strip("[]").split(",") if n.strip()]
    out: list[str] = []
    for line in block[at + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("- "):
            break
        out.append(stripped[2:].strip())
    return out


def step_block(workflow: str, step_name: str = STEP_NAME) -> str:
    """Every line of the named step below its `- name:` line."""
    lines = workflow.splitlines()
    marker = f"- name: {step_name}"
    start = next((i for i, line in enumerate(lines) if line.strip() == marker), None)
    if start is None:
        raise AssertionError(f"missing ci.yml step: {step_name}")
    indent = len(lines[start]) - len(lines[start].lstrip())
    body: list[str] = []
    for line in lines[start + 1 :]:
        if line.strip() and (len(line) - len(line.lstrip())) <= indent:
            break
        body.append(line)
    return "\n".join(body)


def step_run_block(workflow: str, step_name: str = STEP_NAME) -> str:
    """Return the `run:` scalar of the named step.

    Only the `run:` block is returned: the step carries a long `#` comment block that names
    `--tests` and `candle-gen*` in prose, and matching those would let the real command drift
    while this check stayed green. Prose is not wiring.
    """
    body = step_block(workflow, step_name).splitlines()
    run_at = next((i for i, line in enumerate(body) if line.strip().startswith("run:")), None)
    if run_at is None:
        raise AssertionError(f"ci.yml step has no run: block: {step_name}")
    return "\n".join(body[run_at:])


def selected_packages(run_block: str) -> list[str]:
    """The `-p <spec>` selectors in a run block, quotes stripped. Globs are kept as globs."""
    return [spec.strip("'\"") for spec in re.findall(r"-p\s+('[^']+'|\"[^\"]+\"|\S+)", run_block)]


def _balanced(text: str, open_at: int) -> str:
    """The substring from `open_at` (an opening paren) through its matching close."""
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[open_at : i + 1]
    raise AssertionError("unbalanced parentheses in a cfg attribute")


def inner_attribute_prologue(text: str) -> str:
    """The head of a source file where inner attributes are legal, comment lines dropped.

    PROSE IS NOT WIRING. `#![cfg(...)]` is an INNER attribute — rustc accepts it only before the
    first item — so scanning the WHOLE file for the token also matches a `//!` or `///` comment that
    *names* the attribute it is documenting. A file whose header explains its own gating would then
    be classified as compiling to an empty binary with the real attribute deleted, and silently
    dropped from this lane's coverage.

    That is not hypothetical: `candle-gen/tests/cuda_quant_smoke.rs`'s module header explains in
    words why it is `#![cfg(feature = "cuda")]` rather than `#[ignore]`d, and a whole-file scan
    returned FOUR predicates for that file's ONE attribute.

    The prologue ends at the first line that is neither blank, nor a line comment, nor part of an
    inner attribute; bracket depth is tracked so a multi-line `#![cfg(all(\\n ... \\n))]` is kept
    whole. Block comments (`/* ... */`) before the first attribute are not handled — no target in
    this tree opens with one, and the failure mode would be a truncated prologue, i.e. a file
    over-counted and this check failing loudly, not a gate silently disappearing.
    """
    out: list[str] = []
    depth = 0
    for line in text.splitlines():
        if depth == 0:
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                continue
            if not stripped.startswith("#!["):
                break
        out.append(line)
        depth = max(0, depth + line.count("[") - line.count("]"))
    return "\n".join(out)


def file_level_cfgs(text: str) -> list[str]:
    """Every inner-attribute `#![cfg(...)]` predicate, as source, multi-line safe.

    Balanced-paren scanning rather than a line regex: `#![cfg(all(\\n  feature = "cuda",\\n ...))]`
    is legal rustfmt output and a single-line pattern silently misses it, which would count a
    target that compiles to an empty binary as weight-free coverage. The wrapping parens are
    stripped, so each element is the predicate itself (`feature = "cuda"`, `all(...)`, ...).

    Scanned over [`inner_attribute_prologue`], not the raw text — see there for why a comment that
    mentions the attribute must not count as the attribute.
    """
    prologue = inner_attribute_prologue(text)
    out: list[str] = []
    for m in re.finditer(r"#!\[\s*cfg\s*\(", prologue):
        out.append(_balanced(prologue, m.end() - 1)[1:-1])
    return out


def default_features(crate_dir: Path) -> set[str]:
    """The transitive closure of the crate's `default` feature."""
    manifest = crate_dir / "Cargo.toml"
    if not manifest.is_file():
        return set()
    table = tomllib.loads(manifest.read_text(encoding="utf-8")).get("features", {})
    enabled: set[str] = set()
    frontier = list(table.get("default", []))
    while frontier:
        feature = frontier.pop()
        # `dep:foo` / `foo/bar` activate dependency features, never a `#[cfg(feature = ...)]` here.
        if feature in enabled or feature.startswith("dep:") or "/" in feature:
            continue
        enabled.add(feature)
        frontier.extend(table.get(feature, []))
    return enabled


def cfg_holds_on_this_lane(expr: str, enabled: set[str]) -> bool:
    """Evaluate a `cfg(...)` predicate as `candle-cpu-test` would see it.

    Generalised past the original `"cuda"|"metal"` literal pair: any `feature = "X"` is resolved
    against the crate's real default features, so a file gated on a feature nobody thought to list
    is classified correctly rather than miscounted.

    Non-feature predicates are resolved for THIS lane (`ubuntu-latest`, default cargo features):
    `target_os`/`target_family` must name linux/unix, and anything else — `test`, `debug_assertions`
    — is taken as holding, which is what a plain `cargo test` on that runner does. If a future
    predicate makes that assumption wrong, the file is over-counted (this check demands coverage
    the lane cannot provide) rather than under-counted, i.e. it fails loudly, not silently.
    """
    expr = expr.strip()
    m = re.match(r"^(cfg|all|any|not)\s*\(", expr)
    if m:
        inner = _balanced(expr, m.end() - 1)[1:-1]
        parts, depth, current = [], 0, ""
        for ch in inner:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if ch == "," and depth == 0:
                parts.append(current)
                current = ""
            else:
                current += ch
        if current.strip():
            parts.append(current)
        results = [cfg_holds_on_this_lane(p, enabled) for p in parts]
        if m.group(1) == "all":
            return all(results)
        if m.group(1) == "any":
            return any(results)
        if m.group(1) == "not":
            return not results[0]
        return all(results)  # bare `cfg(...)` wrapper

    feature = re.match(r'^feature\s*=\s*"([^"]+)"$', expr)
    if feature:
        return feature.group(1) in enabled
    os_pred = re.match(r'^target_(?:os|family)\s*=\s*"([^"]+)"$', expr)
    if os_pred:
        return os_pred.group(1) in {LANE_TARGET_OS, "unix"}
    return True


def compiles_empty_on_this_lane(text: str, crate_dir: Path) -> bool:
    """True when a file-level `#![cfg(...)]` switches the whole target off in this lane."""
    enabled = default_features(crate_dir)
    return any(not cfg_holds_on_this_lane(cfg, enabled) for cfg in file_level_cfgs(text))


def _fn_bodies(text: str) -> dict[str, str]:
    """Every `fn NAME` in the file mapped to its brace-matched body."""
    bodies: dict[str, str] = {}
    for m in re.finditer(r"\bfn\s+(\w+)\s*[(<]", text):
        brace = text.find("{", m.end())
        if brace == -1:
            continue
        depth, end = 0, None
        for i in range(brace, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end is not None:
            bodies[m.group(1)] = text[brace : end + 1]
    return bodies


def _returns_after_skip(body: str) -> bool:
    """A `return` whose preceding lines announce a SKIP — the self-skip idiom."""
    lines = body.splitlines()
    for i, line in enumerate(lines):
        if not re.search(r"\breturn\b", line):
            continue
        window = "\n".join(lines[max(0, i - SKIP_LOOKBACK_LINES) : i + 1])
        if SKIP_MARKER.search(window):
            return True
    return False


def _case_self_skips(body: str, helpers: dict[str, str]) -> bool:
    """Directly, or through a same-file helper that swallows the guard (sc-19447 F2).

    `candle-gen-lens/tests/encoder_quant_parity.rs` is why the second arm exists: its guard lives in
    `worst_capture_cosine`, which returns `Ok(None)`, and both cases wrap their assertions in
    `if let Some(..)`. A body-only reading calls those two cases weight-free.
    """
    if _returns_after_skip(body):
        return True
    for name, helper_body in helpers.items():
        if _returns_after_skip(helper_body) and re.search(rf"\b{re.escape(name)}\s*\(", body):
            return True
    return False


def _cases(text: str) -> list[tuple[str, list[str], str]]:
    """`(name, attributes, body)` for every `#[test]` / `#[tokio::test]` in a file."""
    lines = text.splitlines()
    helpers = _fn_bodies(text)
    out: list[tuple[str, list[str], str]] = []
    run: list[str] = []
    for i, raw in enumerate(lines):
        line = raw.strip()
        if line.startswith("#["):
            run.append(line)
            continue
        if not run:
            continue
        attrs, run = run, []
        if not any(a.startswith("#[test]") or a.startswith("#[tokio::test") for a in attrs):
            continue
        m = re.search(r"\bfn\s+(\w+)", "\n".join(lines[i : i + 4]))
        if m:
            out.append((m.group(1), attrs, helpers.get(m.group(1), "")))
    return out


def self_skipping_cases() -> list[str]:
    """Every non-`#[ignore]`d candle-gen integration case that can pass without evaluating anything.

    Reported as `crate/tests/target.rs::case`. Should stay empty: the fix is `#[ignore = "..."]`
    naming the env var or device the case needs, not a comment.
    """
    offenders: list[str] = []
    for crate in sorted(p for p in CANDLE_GEN.iterdir() if (p / "tests").is_dir()):
        for source in sorted((crate / "tests").glob("*.rs")):
            text = source.read_text(encoding="utf-8")
            if compiles_empty_on_this_lane(text, crate):
                continue
            helpers = _fn_bodies(text)
            for name, attrs, body in _cases(text):
                if any(a.startswith("#[ignore") for a in attrs):
                    continue
                if _case_self_skips(body, helpers):
                    offenders.append(f"{crate.name}/tests/{source.stem}.rs::{name}")
    return offenders


def crates_with_weight_free_integration_cases() -> dict[str, list[str]]:
    """Map each `candle-gen*` crate to the integration targets this lane can actually execute.

    A target is counted when it has at least one `#[test]` that (a) carries no `#[ignore]` in its
    contiguous attribute run, and (b) is not a self-skipper — a case that prints `SKIP` and returns
    on an unset env var or a non-CUDA device passes without evaluating anything, so counting it as
    coverage restates the sc-19447 defect. The file must also not be switched off wholesale by a
    file-level `#![cfg(...)]` that is false on this lane, since that compiles to an empty binary.
    """
    found: dict[str, list[str]] = {}
    for crate in sorted(p for p in CANDLE_GEN.iterdir() if (p / "tests").is_dir()):
        for source in sorted((crate / "tests").glob("*.rs")):
            text = source.read_text(encoding="utf-8")
            if compiles_empty_on_this_lane(text, crate):
                continue
            helpers = _fn_bodies(text)
            live = 0
            for _name, attrs, body in _cases(text):
                if any(a.startswith("#[ignore") for a in attrs):
                    continue
                if _case_self_skips(body, helpers):
                    continue
                live += 1
            if live:
                found.setdefault(crate.name, []).append(source.stem)
    return found


def uncovered_crates(workflow: str) -> list[str]:
    """Crates carrying weight-free integration cases that the step does not select."""
    run_block = step_run_block(workflow)
    if "--tests" not in run_block:
        return sorted(crates_with_weight_free_integration_cases())
    specs = selected_packages(run_block)
    return sorted(
        crate
        for crate in crates_with_weight_free_integration_cases()
        if not any(fnmatch.fnmatchcase(crate, spec) for spec in specs)
    )


class CandleGenCiTargetCoverageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = WORKFLOW.read_text(encoding="utf-8")

    def test_parsers_find_something(self) -> None:
        """A parse that silently yields nothing would make the real checks vacuously green."""
        self.assertTrue(CANDLE_GEN.is_dir(), f"missing candle-gen tree: {CANDLE_GEN}")
        crates = crates_with_weight_free_integration_cases()
        self.assertTrue(crates, f"parsed no weight-free candle-gen integration targets under {CANDLE_GEN}")
        self.assertIn(
            EPIC_CRATE,
            crates,
            f"{EPIC_CRATE} parsed with no weight-free integration case; the attribute parser is broken",
        )
        self.assertTrue(
            selected_packages(step_run_block(self.workflow)),
            f"parsed no -p selectors from the {STEP_NAME!r} step",
        )

    def test_the_step_actually_runs_integration_targets(self) -> None:
        run_block = step_run_block(self.workflow)
        self.assertIn(
            "--tests",
            run_block,
            f"the {STEP_NAME!r} step no longer passes --tests, so every candle-gen integration "
            "target compiles in lint lanes and executes nowhere automatic again (sc-19447).",
        )
        self.assertNotRegex(
            run_block,
            r"--no-run\b",
            f"the {STEP_NAME!r} step must EXECUTE its test binaries; --no-run reproduces the "
            "windows-cuda-check hole this step exists to close.",
        )

    def test_the_step_runs_in_an_automatic_required_job(self) -> None:
        """A correct command line in a job that never runs is the same hole in a new place."""
        self.assertEqual(
            job_of_step(self.workflow),
            OWNING_JOB,
            f"the {STEP_NAME!r} step moved out of {OWNING_JOB!r}. Relocating it into a "
            "workflow_dispatch-only job (windows-cuda) leaves the command reading correctly and "
            "running never — the exact sc-19447 defect.",
        )
        block = job_block(self.workflow, OWNING_JOB)
        self.assertIn(
            JOB_CONDITION,
            block,
            f"{OWNING_JOB} is no longer gated on {JOB_CONDITION!r}; an `if:` that cannot be true "
            "for a candle change (`if: false`, a renamed output) silently disables this step.",
        )
        self.assertNotRegex(
            step_block(self.workflow),
            r"^\s*if:",
            f"the {STEP_NAME!r} step gained its own `if:`. This step must run whenever its job "
            "does; a step-level condition is a second, unpinned off switch.",
        )
        self.assertIn(
            OWNING_JOB,
            job_needs(self.workflow, GATE_JOB),
            f"{OWNING_JOB} is not in {GATE_JOB}.needs, so its failure no longer blocks the merge "
            "and the step's result is advisory.",
        )

    def test_a_comment_that_names_a_file_cfg_is_not_a_file_cfg(self) -> None:
        """PROSE IS NOT WIRING — see `inner_attribute_prologue`.

        The whole-file token scan this replaced read a `//!` header explaining a gate as the gate
        itself, so a target could lose its real `#![cfg]` and still be classified as compiling to an
        empty binary — dropped from coverage with nothing red. `cuda_quant_smoke.rs` is the live
        instance: its header documents its own gating, and the old scan returned four predicates for
        one attribute.
        """
        prose_only = (
            '//! Documented, not wired: `#![cfg(feature = "cuda")]` is what this WOULD use.\n'
            "\nuse candle_gen::default_device;\n"
        )
        self.assertEqual(
            file_level_cfgs(prose_only),
            [],
            "a #![cfg] named only in a comment is being read as a real inner attribute",
        )
        self.assertFalse(
            compiles_empty_on_this_lane(prose_only, CANDLE_GEN / EPIC_CRATE),
            "a target with no real file-level #![cfg] is classified as an empty binary because a "
            "comment mentions one — it would be silently dropped from this lane's coverage",
        )
        # The real attribute still counts when the same file also discusses it in prose.
        both = '//! Gated with `#![cfg(feature = "cuda")]`.\n#![cfg(feature = "cuda")]\n'
        self.assertEqual(file_level_cfgs(both), ['feature = "cuda"'])
        # And the live file it was found on resolves to exactly its one real attribute.
        canary = (
            CANDLE_GEN / CUDA_CANARY_CRATE / "tests" / f"{CUDA_CANARY_TARGET}.rs"
        ).read_text(encoding="utf-8")
        self.assertEqual(
            file_level_cfgs(canary),
            ['feature = "cuda"'],
            "the sc-7544 canary's header prose is being counted as extra file-level cfgs",
        )

    def test_the_sc7544_cuda_canary_is_cfg_gated_and_still_runs_on_windows_cuda(self) -> None:
        """The one case sc-19447 mis-classified. Both halves are asserted, because either alone
        can be satisfied by a target that proves nothing:

          * excluded from THIS lane — by the `#![cfg]` empty-binary mechanism, not by `#[ignore]`;
          * still EXECUTED on the lane that has a GPU — `windows-cuda`'s test step selects it and
            passes no `-- --ignored`, so an `#[ignore]` here is silence, not deferral.

        `#[ignore]` was applied in sc-19447 on the belief that this was an env-var self-skipper
        parked out of reach. It is device-gated (`default_device().is_cuda()`) and weight-free, and
        `windows-cuda` ran it. Nothing noticed for a whole PR cycle — which is exactly the failure
        mode the canary itself guards against, so it is pinned by name here.
        """
        crate = CANDLE_GEN / CUDA_CANARY_CRATE
        source = crate / "tests" / f"{CUDA_CANARY_TARGET}.rs"
        self.assertTrue(source.is_file(), f"the sc-7544 CUDA canary moved or was deleted: {source}")
        text = source.read_text(encoding="utf-8")

        case = next((c for c in _cases(text) if c[0] == CUDA_CANARY_CASE), None)
        self.assertIsNotNone(
            case, f"{CUDA_CANARY_CASE} is gone from {source.name}; the sc-7544 canary is unguarded"
        )
        ignored = any(a.startswith("#[ignore") for a in case[1])

        # HALF 1 — off this lane by the empty-binary mechanism.
        self.assertTrue(
            compiles_empty_on_this_lane(text, crate),
            f"{source.name} no longer carries a file-level #![cfg] that is false on "
            f"{LANE_TARGET_OS}. Without it the canary would need a CUDA device on ubuntu-latest.",
        )
        self.assertNotIn(
            CUDA_CANARY_TARGET,
            crates_with_weight_free_integration_cases().get(CUDA_CANARY_CRATE, []),
            f"{source.name} is being counted as weight-free coverage for this lane, which cannot "
            "run it — the classification and the gating disagree.",
        )

        # HALF 2 — the lane with a GPU still evaluates it.
        cuda_step = step_run_block(self.workflow, CUDA_STEP_NAME)
        self.assertEqual(
            job_of_step(self.workflow, CUDA_STEP_NAME),
            CUDA_STEP_JOB,
            f"the {CUDA_STEP_NAME!r} step left {CUDA_STEP_JOB!r}; this check would then be reading "
            "some other job's command line.",
        )
        self.assertIn(
            "--features cuda",
            cuda_step,
            f"the {CUDA_STEP_NAME!r} step no longer enables the cuda feature, so the canary's "
            "own #![cfg] now compiles it out of EVERY lane.",
        )
        self.assertNotRegex(
            cuda_step,
            r"--no-run\b",
            f"the {CUDA_STEP_NAME!r} step gained --no-run; it would build the canary and throw it "
            "away, which is the windows-cuda-check hole, not a lane that runs it.",
        )
        self.assertTrue(
            any(
                fnmatch.fnmatchcase(CUDA_CANARY_CRATE, spec)
                for spec in selected_packages(cuda_step)
            ),
            f"{CUDA_CANARY_CRATE} is selected by no -p spec in the {CUDA_STEP_NAME!r} step "
            f"(specs: {selected_packages(cuda_step)}).",
        )
        self.assertTrue(
            (not ignored) or "--ignored" in cuda_step,
            f"{CUDA_CANARY_CASE} is #[ignore]d and the {CUDA_STEP_NAME!r} step passes no "
            "`-- --ignored`, so libtest reports it `ignored` on the only lane that has ever "
            "executed it. That is how the sc-7544 canary was silenced in PR #712. Exclude it with "
            "the file-level #![cfg(feature = \"cuda\")] instead — or add `-- --ignored` to that "
            "step and rewrite this check's rationale.",
        )

    def test_no_counted_case_self_skips(self) -> None:
        """`SKIP` + `return` reports as PASSED — the sc-19447 defect wearing a green badge."""
        offenders = self_skipping_cases()
        self.assertEqual(
            offenders,
            [],
            "candle-gen integration cases print SKIP and return without asserting, yet are neither "
            "#[ignore]d nor behind a file-level #![cfg]. libtest reports them as PASSED, so this "
            "lane would count them as evidence: "
            + ", ".join(offenders)
            + ". Add #[ignore = \"...\"] naming the env var or device each needs (sc-19447).",
        )

    def test_a_conditional_case_that_still_asserts_is_not_a_self_skipper(self) -> None:
        """Negative control for the detector, so it cannot be widened into a blanket ignore rule.

        `reference_manifests_are_consistent_if_present` skips individual SD3.5 variants whose
        reference is absent, but its references ARE committed to the repo, so it asserts on every
        run. It must stay counted.
        """
        sd3 = CANDLE_GEN / "candle-gen-sd3" / "tests" / "component_parity.rs"
        self.assertTrue(sd3.is_file(), f"negative control moved: {sd3}")
        text = sd3.read_text(encoding="utf-8")
        helpers = _fn_bodies(text)
        case = next(
            (c for c in _cases(text) if c[0] == "reference_manifests_are_consistent_if_present"),
            None,
        )
        self.assertIsNotNone(case, "negative-control case was renamed or removed")
        self.assertFalse(
            _case_self_skips(case[2], helpers),
            "the self-skip detector now swallows a case that really does assert on every run",
        )
        self.assertIn(
            "component_parity",
            crates_with_weight_free_integration_cases().get("candle-gen-sd3", []),
            "candle-gen-sd3's weight-free shape guard is no longer counted as coverage",
        )

    def test_every_crate_with_weight_free_cases_is_selected(self) -> None:
        missing = uncovered_crates(self.workflow)
        self.assertEqual(
            missing,
            [],
            "candle-gen crates carry weight-free integration cases that no automatic lane runs: "
            + ", ".join(missing)
            + f". Widen the -p selectors on the {STEP_NAME!r} step in .github/workflows/ci.yml.",
        )

    def test_the_epic_crate_is_selected_by_name_or_glob(self) -> None:
        """AC5 of sc-19447, stated directly rather than only as a consequence of the general rule."""
        specs = selected_packages(step_run_block(self.workflow))
        self.assertTrue(
            any(fnmatch.fnmatchcase(EPIC_CRATE, spec) for spec in specs),
            f"{EPIC_CRATE} is selected by no -p spec in the {STEP_NAME!r} step (specs: {specs}).",
        )

    def test_a_multiline_or_differently_featured_file_cfg_is_still_seen(self) -> None:
        """The original gate was a single-line `"cuda"|"metal"` regex; both halves were too narrow."""
        crate = CANDLE_GEN / EPIC_CRATE
        multiline = '#![cfg(all(\n    feature = "cuda",\n    feature = "testkit"\n))]\n'
        self.assertTrue(
            compiles_empty_on_this_lane(multiline, crate),
            "a multi-line file-level #![cfg] is not detected; a target that compiles to an empty "
            "binary would be counted as weight-free coverage",
        )
        self.assertTrue(
            compiles_empty_on_this_lane('#![cfg(feature = "flash-attn")]\n', crate),
            "a file-level #![cfg] on a non-default feature outside {cuda, metal} is not detected",
        )
        self.assertFalse(
            compiles_empty_on_this_lane("#![cfg(test)]\n", crate),
            "a cfg that HOLDS on this lane must not be treated as switching the target off",
        )
        self.assertFalse(
            compiles_empty_on_this_lane('#![cfg(target_os = "linux")]\n', crate),
            "candle-cpu-test runs on ubuntu-latest; a linux gate does not empty the binary there",
        )
        # Pinned against the real tree too, not only synthetic strings: an extractor that returns
        # the predicate still wrapped in its parens parses nothing, falls through to "enabled", and
        # counts all 57 cuda-gated targets as weight-free. That regressed once during sc-19447.
        mochi = crates_with_weight_free_integration_cases().get("candle-gen-mochi", [])
        self.assertIn("cpu_golden", mochi, "candle-gen-mochi's CPU target is no longer counted")
        self.assertNotIn(
            "te_parity",
            mochi,
            "candle-gen-mochi/tests/te_parity.rs is `#![cfg(feature = \"cuda\")]` — it compiles to "
            "an empty binary on this lane and must not be counted as coverage",
        )

    def test_the_coverage_check_discriminates_mutations(self) -> None:
        """Each mutation is applied ALONE — a batch would let one survivor hide behind another."""
        step = "cargo test --locked --lib --tests -j 1\n          -p 'candle-gen*'"
        mutations = {
            "--tests dropped": self.workflow.replace(
                step, "cargo test --locked --lib -j 1\n          -p 'candle-gen*'"
            ),
            "glob narrowed to one crate": self.workflow.replace(
                step, "cargo test --locked --lib --tests -j 1\n          -p candle-gen-minimax-h3"
            ),
            "glob narrowed away from the epic crate": self.workflow.replace(
                step, "cargo test --locked --lib --tests -j 1\n          -p 'candle-gen-s*'"
            ),
        }
        for label, mutant in mutations.items():
            with self.subTest(mutation=label):
                self.assertNotEqual(
                    mutant, self.workflow, f"mutation {label!r} matched nothing; the check is stale"
                )
                self.assertNotEqual(
                    uncovered_crates(mutant),
                    [],
                    f"mutation {label!r} left the coverage check green",
                )

    def test_the_step_parser_rejects_a_renamed_or_missing_step(self) -> None:
        with self.assertRaises(AssertionError):
            step_run_block(self.workflow, "Test Candle gen packages that do not exist")
