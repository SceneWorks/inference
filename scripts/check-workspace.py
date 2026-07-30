#!/usr/bin/env python3
"""Fail when the normalized inference workspace drifts from its graph invariants."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MEMBER_COUNT = 93
INTERNAL_PACKAGES = {
    "candle-audio",
    "candle-audio-catalog",
    "candle-audio-kokoro",
    "candle-audio-moss-sfx",
    "candle-gen-catalog",
    "core-llm",
    "core-llm-testkit",
    "mlx-gen-catalog",
    "mlx-llm",
    "candle-llm",
    "sceneworks-gen-core",
    "sceneworks-gen-core-testkit",
    "runtime-catalog",
    "runtime-macos",
    "runtime-cpu",
    "runtime-cuda",
}
PINNED_WORKSPACE_DEPENDENCIES = {
    "mlx-rs": ("pmetal-mlx-rs", "932beb4e60db44d378ffa1fe648defea59b5cbd0"),
    "mlx-sys": ("pmetal-mlx-sys", "932beb4e60db44d378ffa1fe648defea59b5cbd0"),
    "candle-core": ("candle-core", "1e6aa85e867eb007cba1b8bae517a10d1aaf0c0d"),
    "candle-nn": ("candle-nn", "1e6aa85e867eb007cba1b8bae517a10d1aaf0c0d"),
    "candle-transformers": ("candle-transformers", "1e6aa85e867eb007cba1b8bae517a10d1aaf0c0d"),
    "candle-flash-attn": ("candle-flash-attn", "1e6aa85e867eb007cba1b8bae517a10d1aaf0c0d"),
}
DEFAULT_GRAPH_PINNED_PACKAGES = {
    package_name: revision
    for dependency_name, (package_name, revision) in PINNED_WORKSPACE_DEPENDENCIES.items()
    if dependency_name != "candle-flash-attn"
}
FORBIDDEN_GRAPH_PACKAGES = {
    # Provider composition is ordinary, value-scoped source code. Reintroducing this crate would
    # make linker participation part of the supported runtime graph again.
    "inventory",
}
# Directory names whose subtrees are not part of this checkout's single workspace: the git store
# and build output (.git, target), plus agent tooling that nests its own gitignored worktrees --
# each a separate checkout carrying its own Cargo.lock/manifest (.claude, .codex). They must not
# be swept into the single-lockfile / single-manifest invariants below.
IGNORED_TREE_PARTS = frozenset({".git", "target", ".claude", ".codex"})

# --- epic 13657 guardrail: inference never fetches weights and never derives a download-cache
# location. Every model component is a caller-provisioned local path (WeightsSource::Dir / File);
# fetching and cache placement are the consumer's job, so user-supplied models at arbitrary paths
# must load. The three assertions below (network-client ban, HF-cache source lint, deleted
# env-side-channel pins) turn that contract into an enforced boundary. See
# docs/architecture/inference-rearchitecture.md.

# No network/HTTP client may resolve anywhere in the graph. Mirrors the `inventory` whole-graph
# ban above: a client reachable through an enabled third-party feature (e.g. a candle `hf-hub`
# feature) would reopen self-fetch, so pin the feature off rather than weaken this. All nine are
# confirmed absent on main; extend this set, never trim it. If a denied crate ever becomes a
# legitimate *transitive-only* dep of a build tool, narrow to the direct/feature-activated
# workspace-member scope reported by check_network_clients rather than deleting the entry.
FORBIDDEN_NETWORK_CLIENT_PACKAGES = {
    "hf-hub",
    "reqwest",
    "ureq",
    "curl",
    "curl-sys",
    "git2",
    "isahc",
    "attohttpc",
    "hyper",
}

# Whole-tree substring bans over every workspace member's Rust (src, tests, examples, testkits --
# NO allow-list, per the sharpened rule). These name an HF download cache or its client, which
# inference must never reference. Precision notes:
#   * `.cache/huggingface` is the SPECIFIC HF cache path -- NOT a blanket `.cache/` ban, so the
#     legitimate `~/.cache/mlx-gen-seedvr2-golden` test-golden dir does not trip.
#   * bare `HUGGINGFACE` is deliberately NOT banned: it false-positives on legitimate
#     `https://huggingface.co/...` `source_url` attribution, `huggingface-cli` doc prose, repo IDs,
#     and license text. `.cache/huggingface` + `hf_hub` already cover the real cache-derivation cases.
#   * `Api::new` is the hf_hub API constructor; no non-hf `Api::new` exists on the clean tree, so
#     the bare token is safe. If a legitimate unrelated `Api::new` ever appears, qualify it to the
#     `hf_hub`-scoped form rather than dropping the ban.
RUST_BANNED_SUBSTRINGS = (
    "HF_HOME",
    "HF_HUB_CACHE",
    ".cache/huggingface",
    "hf_hub",
    "Api::new",
)

# Env vars that were DELETED as production self-fetch / cache-derivation side channels. They must
# not return as production reads. Scoped to production `crates/**/src/**` EXCLUDING `#[cfg(test)]`
# modules, and matched only as actual `env::var("NAME")` reads (not doc-comment prose that merely
# names a removed var), because:
#   * MOSS_XY_TOKENIZER_SNAPSHOT / MOSS_AUDIO_TOKENIZER_SNAPSHOT legitimately persist TEST-SIDE
#     (sc-13660/sc-13662): each moss crate's tests/conformance.rs reads them as explicit passed-in
#     snapshot paths, and they are keys in release/real-weight-models.toml + real-weights.yml that
#     provision the weekly runner. A passed-in test path is allowed; cache DERIVATION is not.
#   * LTX_GEMMA_DIR is read only inside a `#[cfg(test)]` real-weight harness (mlx-gen-ltx
#     src/training.rs) as a test-only convenience path; its production fallback was deleted.
#   * SENSENOVA_DISTILL_LORA / PULID_FLUX_WEIGHTS / LTX_UNCENSORED_GEMMA_DIR still appear in `src/`
#     ONLY as doc prose or `#[cfg(test)]` assertions proving the deleted var is NOT resurrected --
#     never as an env::var read -- so shape-matching + cfg(test) stripping keeps them green.
# Legitimate passed-in-path env vars (MLX_LLM_TEST_MODEL, per-crate *_SNAPSHOT/*_SNAPSHOT_DIR,
# MLX_GEN_MODELS_ROOT, CANDLE_GEN_MODELS_ROOT, tuning knobs LTX_MAX_LATENT_TOKENS/LTX_VAE_BUDGET_GIB)
# are NOT banned: this targets cache-location derivation + the deleted side channels, not all reads.
DELETED_ENV_SIDE_CHANNELS = (
    "PERTH_SNAPSHOT",
    "MOSS_XY_TOKENIZER_SNAPSHOT",
    "MOSS_AUDIO_TOKENIZER_SNAPSHOT",
    "SENSENOVA_DISTILL_LORA",
    "LTX_GEMMA_DIR",
    "PULID_FLUX_WEIGHTS",
    "LTX_UNCENSORED_GEMMA_DIR",
    "PULID_EVA_WEIGHTS",
    "PULID_FACE_WEIGHTS_DIR",
)

# The crate that owns the shared PiD decode seam and, since sc-15775, the shared per-route decode
# domain. A direct dependency on it is what makes a provider PiD-eligible.
PID_SEAM_CRATE = "mlx-gen-pid"

# Evidence that a provider actually CALLS the shared per-route decode API — not that it mentions it.
#
# The first revision of this gate matched the bare substrings "DecodeRoutes" / "decode_routes", which
# the adversarial review defeated three ways: a `// TODO(sc-15775): should use DecodeRoutes one day.`
# on an otherwise non-conforming adopter passed; so did `#[allow(unused)] use
# mlx_gen_pid::decode_routes;`. A gate whose stated purpose is to replace doc-comment enforcement must
# not be satisfiable by a doc comment, so matching now happens on COMMENT-STRIPPED source and requires
# call syntax.
#
# `DecodeRoutes::new` is the *checked* constructor: since sc-15775 it returns `Result` and refuses a
# native ladder that reaches into the PiD student's tile domain, so calling it (or
# `assert_decode_routes`, its panicking test-suite form) is at once the construction evidence and the
# conformance evidence. There is deliberately no third, separately-skippable conformance call to look
# for — requiring one would be ceremony now that construction cannot be performed unchecked.
PID_DECODE_ROUTE_CONSTRUCTION_MARKERS = ("DecodeRoutes::new", "assert_decode_routes")

# The admission half. Declaring the routes and then not gating on them leaves the hazard wide open, so
# a construction call alone is not adoption. Matched as a call on a route-named receiver rather than a
# bare `.validate(` so an unrelated `foo.validate(...)` elsewhere in the crate cannot stand in for it.
PID_DECODE_ROUTE_ADMISSION_CALL = re.compile(
    r"(?:decode_routes|DecodeRoutes|routes)\b[^;{}]{0,200}?\.\s*validate\s*\(",
    re.DOTALL,
)

# Any one of these in the crate's own sources means it implements rung 2 (bounded decode) and can
# therefore reach the seam with a decode geometry.
#
# Broader than the `decode_tile_edges` field name alone, which the review defeated as attack (c): an
# adopter that builds its `MemoryParameterRanges` through a shared helper never writes that literal.
# What it cannot delegate is the executable half — its own `MemoryRequestScope::configure_decode`, the
# hook the shared runtime drives to apply a bounded-decode selection — so that spelling is in the set
# too. Deliberately fail-closed: over-triggering asks a provider for a three-line declaration, while
# under-triggering ships the defect.
PID_RUNG_TWO_MARKERS = (
    "decode_tile_edges",
    "decode_overlaps",
    "MemoryParameterRanges",
    "BoundedDecode",
    "configure_decode",
)

_RAW_STRING_START = re.compile(r'(?:b?r)(#*)"')

_CFG_TEST_ATTR = re.compile(r"#\s*\[\s*cfg\s*\(\s*test\s*\)\s*\]")
_CFG_TEST_MOD = re.compile(r"\s*(?:#\s*\[[^\]]*\]\s*)*(?:pub(?:\s*\([^)]*\))?\s+)?mod\s+\w+\s*\{")


def fail(message: str) -> None:
    raise AssertionError(message)


def strip_rust_comments(source: str) -> str:
    """Blank out Rust comments so a source-text policy check cannot be satisfied by a comment.

    Comments become runs of spaces (newlines preserved) rather than being deleted, so line and column
    positions still line up with the original if a caller ever reports them.

    String and character literals are honoured, because ``"// not a comment"`` and ``"/* nor this */"``
    are real code — including raw strings (``r"..."``, ``r#"..."#``, ``br#"..."#``), whose whole purpose
    is to contain otherwise-significant characters. Block comments nest, as they do in Rust. ``'a``
    lifetimes are distinguished from ``'a'`` char literals.
    """
    out: list[str] = []
    index = 0
    length = len(source)
    while index < length:
        char = source[index]
        if char == "/" and source.startswith("//", index):
            end = source.find("\n", index)
            end = length if end < 0 else end
            out.append(" " * (end - index))
            index = end
            continue
        if char == "/" and source.startswith("/*", index):
            depth = 0
            end = index
            while end < length:
                if source.startswith("/*", end):
                    depth += 1
                    end += 2
                    continue
                if source.startswith("*/", end):
                    depth -= 1
                    end += 2
                    if depth == 0:
                        break
                    continue
                end += 1
            out.append("".join("\n" if c == "\n" else " " for c in source[index:end]))
            index = end
            continue
        raw = _RAW_STRING_START.match(source, index)
        if raw and not (index and (source[index - 1].isalnum() or source[index - 1] == "_")):
            terminator = '"' + raw.group(1)
            end = source.find(terminator, raw.end())
            end = length if end < 0 else end + len(terminator)
            out.append(source[index:end])
            index = end
            continue
        if char == '"':
            end = index + 1
            while end < length:
                if source[end] == "\\":
                    end += 2
                    continue
                if source[end] == '"':
                    end += 1
                    break
                end += 1
            out.append(source[index:end])
            index = end
            continue
        if char == "'":
            # `'\n'` / `'\u{1f600}'`: an escape runs to the closing quote.
            if source.startswith("'\\", index):
                end = source.find("'", index + 2)
                end = length if end < 0 else end + 1
                out.append(source[index:end])
                index = end
                continue
            # `'a'` is a char literal; `'a` (no closing quote) is a lifetime, so emit just the tick and
            # let the identifier be scanned normally.
            if index + 2 < length and source[index + 2] == "'":
                out.append(source[index : index + 3])
                index += 3
                continue
            out.append(char)
            index += 1
            continue
        out.append(char)
        index += 1
    return "".join(out)


def cargo_metadata(offline: bool) -> dict:
    command = ["cargo", "metadata", "--locked", "--format-version", "1"]
    if offline:
        command.append("--offline")
    # cargo emits UTF-8 on every platform, so decode explicitly. text=True would decode with
    # the locale encoding instead, which fails on Windows (cp1252) as soon as any package in
    # the resolved graph carries non-ASCII metadata -- today a dependency author name.
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if result.returncode:
        sys.stderr.write(result.stderr.decode("utf-8", errors="replace"))
        fail(f"cargo metadata failed with exit code {result.returncode}")
    return json.loads(result.stdout.decode("utf-8"))


def _within_workspace(path: Path) -> bool:
    """True when a discovered path belongs to this checkout's own workspace tree.

    The check is on the path RELATIVE to ROOT, so running the gate from inside a nested worktree
    (whose own absolute path contains e.g. ``.claude/worktrees/...``) still counts that worktree's
    own root Cargo.lock/manifest -- only subtrees *below* ROOT named in IGNORED_TREE_PARTS drop out.
    """
    return IGNORED_TREE_PARTS.isdisjoint(path.relative_to(ROOT).parts)


def check_filesystem() -> None:
    lockfiles = sorted(
        path.relative_to(ROOT)
        for path in ROOT.rglob("Cargo.lock")
        if _within_workspace(path)
    )
    if lockfiles != [Path("Cargo.lock")]:
        fail(f"expected only the root Cargo.lock, found: {lockfiles}")

    workspace_manifests = []
    for manifest in ROOT.rglob("Cargo.toml"):
        if not _within_workspace(manifest):
            continue
        if any(
            line.strip() == "[workspace]"
            for line in manifest.read_text(encoding="utf-8").splitlines()
        ):
            workspace_manifests.append(manifest.relative_to(ROOT))
    if workspace_manifests != [Path("Cargo.toml")]:
        fail(f"expected one active root workspace manifest, found: {workspace_manifests}")

    for required in (Path(".cargo/config.toml"), Path("rust-toolchain.toml")):
        if not (ROOT / required).is_file():
            fail(f"missing root-owned configuration: {required}")

    root_manifest = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    dependencies = root_manifest["workspace"]["dependencies"]
    for dependency_name, (package_name, revision) in PINNED_WORKSPACE_DEPENDENCIES.items():
        dependency = dependencies.get(dependency_name)
        if not isinstance(dependency, dict):
            fail(f"missing structured root pin for {dependency_name}")
        if dependency.get("rev") != revision:
            fail(f"{dependency_name} is not declared at {revision}: {dependency}")
        if dependency.get("package", dependency_name) != package_name:
            fail(f"{dependency_name} no longer aliases package {package_name}: {dependency}")


def check_graph(metadata: dict) -> None:
    packages = metadata["packages"]
    packages_by_id = {package["id"]: package for package in packages}
    member_ids = metadata["workspace_members"]
    members = [packages_by_id[member_id] for member_id in member_ids]

    if len(member_ids) != EXPECTED_MEMBER_COUNT:
        fail(f"expected {EXPECTED_MEMBER_COUNT} workspace members, found {len(member_ids)}")
    if len(set(member_ids)) != len(member_ids):
        fail("workspace member IDs are not unique")

    for package in members:
        manifest = Path(package["manifest_path"]).resolve()
        if package["source"] is not None:
            fail(f"workspace member {package['name']} unexpectedly has source {package['source']}")
        if ROOT / "crates" not in manifest.parents:
            fail(f"workspace member is outside crates/: {manifest}")

    for name in sorted(INTERNAL_PACKAGES):
        matches = [package for package in packages if package["name"] == name]
        if len(matches) != 1:
            fail(f"expected one {name} package resolution, found {len(matches)}")
        if matches[0]["source"] is not None:
            fail(f"internal package {name} is not a path source: {matches[0]['source']}")

    resolved_names = {package["name"] for package in packages}
    forbidden = sorted(FORBIDDEN_GRAPH_PACKAGES & resolved_names)
    if forbidden:
        fail(f"explicit composition forbids these graph packages: {forbidden}")

    for package in members:
        for dependency in package["dependencies"]:
            if dependency["name"] not in INTERNAL_PACKAGES:
                continue
            if dependency["source"] is not None or dependency.get("path") is None:
                fail(
                    f"{package['name']} -> {dependency['name']} is not a workspace path edge: "
                    f"source={dependency['source']!r}, path={dependency.get('path')!r}"
                )

    for name, revision in DEFAULT_GRAPH_PINNED_PACKAGES.items():
        matches = [package for package in packages if package["name"] == name]
        if len(matches) != 1:
            fail(f"expected one {name} resolution, found {len(matches)}")
        source = matches[0]["source"] or ""
        if not source.endswith(f"#{revision}"):
            fail(f"{name} does not resolve at {revision}: {source}")

    tokenizer_minors = {
        ".".join(package["version"].split(".")[:2])
        for package in packages
        if package["name"] == "tokenizers"
    }
    if tokenizer_minors != {"0.21", "0.22"}:
        fail(f"expected intentional tokenizers 0.21/0.22 split, found {tokenizer_minors}")


def check_network_clients(metadata: dict) -> None:
    """No network/HTTP client may resolve in the workspace graph (epic 13657 self-fetch ban)."""
    packages = metadata["packages"]
    packages_by_id = {package["id"]: package for package in packages}
    resolved_names = {package["name"] for package in packages}

    present = sorted(FORBIDDEN_NETWORK_CLIENT_PACKAGES & resolved_names)
    if not present:
        return

    # Attribute each present client to the workspace member(s) that declare it directly, so the
    # error names the reintroduction site (the common case: someone adds it to a member manifest).
    direct = sorted(
        {
            (packages_by_id[member_id]["name"], dependency["name"])
            for member_id in metadata["workspace_members"]
            for dependency in packages_by_id[member_id]["dependencies"]
            if dependency["name"] in FORBIDDEN_NETWORK_CLIENT_PACKAGES
        }
    )
    detail = f"; direct workspace-member deps: {direct}" if direct else "; transitive only"
    fail(
        "inference never self-fetches weights: no network/HTTP client may resolve in the graph, "
        f"found {present}{detail}"
    )


def _match_brace(text: str, open_index: int) -> int:
    """Index just past the ``}`` matching the ``{`` at ``open_index``.

    Rust string/char literals (including raw strings) and comments are skipped so a brace inside a
    format string or a comment cannot unbalance the count.
    """
    depth = 0
    i = open_index
    n = len(text)
    while i < n:
        char = text[i]
        if char == "{":
            depth += 1
            i += 1
        elif char == "}":
            depth -= 1
            i += 1
            if depth == 0:
                return i
        elif char == "/" and i + 1 < n and text[i + 1] == "/":
            newline = text.find("\n", i)
            i = n if newline == -1 else newline
        elif char == "/" and i + 1 < n and text[i + 1] == "*":
            close = text.find("*/", i + 2)
            i = n if close == -1 else close + 2
        elif char == "r" and i + 1 < n and text[i + 1] in '#"':
            hashes = 0
            cursor = i + 1
            while cursor < n and text[cursor] == "#":
                hashes += 1
                cursor += 1
            if cursor < n and text[cursor] == '"':
                terminator = '"' + "#" * hashes
                close = text.find(terminator, cursor + 1)
                i = n if close == -1 else close + len(terminator)
            else:
                i += 1
        elif char == '"':
            i += 1
            while i < n:
                if text[i] == "\\":
                    i += 2
                elif text[i] == '"':
                    i += 1
                    break
                else:
                    i += 1
        elif char == "'":
            # Char literal or a lifetime. Skip an escaped or single-char literal; otherwise advance
            # one (a lifetime such as `'a` has no closing quote).
            if i + 1 < n and text[i + 1] == "\\":
                close = text.find("'", i + 2)
                i = n if close == -1 else close + 1
            elif i + 2 < n and text[i + 2] == "'":
                i += 3
            else:
                i += 1
        else:
            i += 1
    return n


def _cfg_test_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of ``#[cfg(test)] mod ... { ... }`` blocks, so the production-source scan
    ignores test code that legitimately reuses the pinned env-var names as passed-in test paths."""
    spans: list[tuple[int, int]] = []
    for attribute in _CFG_TEST_ATTR.finditer(text):
        module = _CFG_TEST_MOD.match(text, attribute.end())
        if module is None:
            continue
        end = _match_brace(text, module.end() - 1)
        spans.append((attribute.start(), end))
    return spans


def _line_of(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def check_rust_sources(root: Path) -> None:
    """Fail on any HF-cache reference in workspace Rust, or a production read of a deleted env
    side channel. See RUST_BANNED_SUBSTRINGS / DELETED_ENV_SIDE_CHANNELS for the precise scoping."""
    crates = root / "crates"
    if not crates.is_dir():
        return

    side_channel_reads = {
        name: re.compile(r"env::var(?:_os)?\s*\(\s*\"" + re.escape(name) + r"\"\s*\)")
        for name in DELETED_ENV_SIDE_CHANNELS
    }
    violations: list[str] = []
    for path in sorted(crates.rglob("*.rs")):
        relative = path.relative_to(root)
        if not IGNORED_TREE_PARTS.isdisjoint(relative.parts):
            continue
        text = path.read_text(encoding="utf-8")

        # Whole-tree HF-cache bans: every .rs of every member, tests and examples included.
        for needle in RUST_BANNED_SUBSTRINGS:
            index = text.find(needle)
            while index != -1:
                violations.append(
                    f"{relative}:{_line_of(text, index)}: banned HF-cache reference {needle!r}"
                )
                index = text.find(needle, index + 1)

        # Deleted env side channels: production `src/` reads only, `#[cfg(test)]` blocks excluded.
        if "src" in relative.parts:
            spans = _cfg_test_spans(text)
            for name, pattern in side_channel_reads.items():
                for match in pattern.finditer(text):
                    if any(start <= match.start() < end for start, end in spans):
                        continue
                    violations.append(
                        f"{relative}:{_line_of(text, match.start())}: production read of deleted "
                        f"env side channel {name!r} (inference receives model paths from the caller)"
                    )

    if violations:
        joined = "\n  ".join(violations)
        fail(f"inference source must not reference HF caches or deleted env side channels:\n  {joined}")


def check_pid_decode_route_adoption(metadata: dict, root: Path) -> None:
    """A PiD-eligible provider that adopts bounded decode must declare its routes through the shared
    ``mlx_gen_pid::DecodeRoutes`` (sc-15775).

    ``mlx_gen_pid::engine::selected_decode_tiling`` is a shared, provider-agnostic seam. Native VAE
    tile edges (256-768 output px) and PiD tile edges (512px-aligned, 1024..=4096) are disjoint by
    construction, because the student decodes a ``scale x`` super-resolved output. A provider that
    emits its native ladder into ``GenerationMemory::decode_tile_edge`` therefore turns a working
    ``use_pid`` + bounded-decode request into a hard ``budget::validate_tile`` rejection at generate
    time — and the seam must not paper over it, because silently re-planning would execute a different
    strategy than the selector chose.

    ``DecodeRoutes::new`` accepts only the native ladder and, since sc-15775, *refuses to construct* one
    that reaches into the PiD domain — so an overlapping declaration never becomes a value. That helps
    only a provider that calls it, so this is the tripwire for one that does not: it lives here, in a
    lane that fires on any workspace change, so the next adopter cannot opt out by simply not reaching
    for the shared type.

    Armed by two facts: a direct ``mlx-gen-pid`` dependency (PiD-eligible) and a
    ``register_memory_strategy`` call (its contract is resolvable, so a selector can choose a strategy
    for it), plus any sign of rung-2 adoption in the crate's own sources
    (``PID_RUNG_TWO_MARKERS``). A provider missing the first two cannot reach the hazard and is not
    asked to declare routes.

    What it verifies, precisely — and no more:

    * the crate's comment-stripped sources contain a *call* to the checked constructor, and
    * a *call* to ``validate`` on a route-named receiver.

    Both are call-syntax matches on comment-stripped text, which is what closes the three defeats the
    adversarial review found in the first revision (a ``// TODO`` mentioning ``DecodeRoutes``, an unused
    ``use``, and a trigger set narrow enough to miss a helper-built ``MemoryParameterRanges``).

    It remains a *static text* check, and two residual gaps are worth naming rather than implying away:

    1. It cannot prove the ``validate`` call is reached on every admission path — only that one exists.
       Wiring ``validate`` into ``safety_check`` is what a behavioural, registry-driven walk would
       verify; that needs two new pieces of shared contract surface (a safety-check fn pointer on
       ``MemoryRegistration``, and a way to declare PiD eligibility) and is tracked separately.
    2. The rung-2 trigger is a *text* proxy for the semantic fact "this provider's contract publishes
       non-empty ``decode_tile_edges``". A provider could in principle delegate every textual trace of
       rung 2 — the ranges, the strategy name, and its ``configure_decode`` hook — to another crate. The
       same behavioural walk is what covers that; the trigger set is deliberately wide (fail-closed) to
       shrink the window in the meantime.
    """
    packages_by_id = {package["id"]: package for package in metadata["packages"]}
    violations: list[str] = []
    for member_id in metadata["workspace_members"]:
        package = packages_by_id[member_id]
        if package["name"] == PID_SEAM_CRATE:
            continue
        if not any(
            dependency["name"] == PID_SEAM_CRATE for dependency in package["dependencies"]
        ):
            continue
        manifest_dir = Path(package["manifest_path"]).parent
        if not manifest_dir.is_dir():
            continue
        # Comments are stripped BEFORE any matching, on both the trigger conditions and the evidence:
        # a commented-out `register_memory_strategy` must not arm the gate, and a `// TODO: use
        # DecodeRoutes` must not disarm it.
        joined = strip_rust_comments(
            "\n".join(
                path.read_text(encoding="utf-8")
                for path in sorted(manifest_dir.rglob("*.rs"))
                if IGNORED_TREE_PARTS.isdisjoint(path.relative_to(root).parts)
            )
        )
        if "register_memory_strategy" not in joined:
            continue
        if not any(marker in joined for marker in PID_RUNG_TWO_MARKERS):
            continue
        missing: list[str] = []
        if not any(marker in joined for marker in PID_DECODE_ROUTE_CONSTRUCTION_MARKERS):
            spellings = " or ".join(
                f"`{marker}`" for marker in PID_DECODE_ROUTE_CONSTRUCTION_MARKERS
            )
            missing.append(f"never calls the checked constructor ({spellings})")
        if not PID_DECODE_ROUTE_ADMISSION_CALL.search(joined):
            missing.append("never calls `validate` on a declared route set to gate admission")
        if not missing:
            continue
        violations.append(
            f"{package['name']} depends on {PID_SEAM_CRATE}, registers a memory-strategy contract, "
            f"and implements bounded decode, but {', and '.join(missing)}"
        )

    if violations:
        joined = "\n  ".join(violations)
        fail(
            "a PiD-eligible provider that adopts bounded decode must construct its native ladder "
            "through the checked mlx_gen_pid::DecodeRoutes::new (or assert_decode_routes) AND gate "
            "admission on the resulting `validate` (sc-15775) — native VAE tile edges are not legal "
            "PiD tiles, so emitting the native ladder into a use_pid request is refused at generate "
            "time rather than re-planned. A mention in a comment or an unused import is not "
            "adoption; this gate matches call syntax on comment-stripped source:\n  "
            f"{joined}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--offline",
        action="store_true",
        help="require Cargo to validate entirely from its local cache",
    )
    args = parser.parse_args()

    try:
        check_filesystem()
        metadata = cargo_metadata(args.offline)
        check_graph(metadata)
        check_network_clients(metadata)
        check_rust_sources(ROOT)
        check_pid_decode_route_adoption(metadata, ROOT)
    except (AssertionError, json.JSONDecodeError) as error:
        print(f"workspace gate: FAIL: {error}", file=sys.stderr)
        return 1

    print(
        "workspace gate: OK "
        f"({EXPECTED_MEMBER_COUNT} path members, one lockfile, explicit registries, pinned backends, "
        "intentional tokenizer split, no network clients, no HF-cache references)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
