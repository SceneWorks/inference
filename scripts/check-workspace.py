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

_CFG_ATTR_START = re.compile(r"#\s*(!?)\s*\[\s*cfg\s*\(")


def fail(message: str) -> None:
    raise AssertionError(message)


def strip_rust_comments(source: str, *, strip_literals: bool = False) -> str:
    """Blank out Rust comments so a source-text policy check cannot be satisfied by a comment.

    Comments become runs of spaces (newlines preserved) rather than being deleted, so line and column
    positions still line up with the original if a caller ever reports them.

    String and character literals are honoured, because ``"// not a comment"`` and ``"/* nor this */"``
    are real code — including raw strings (``r"..."``, ``r#"..."#``, ``br#"..."#``), whose whole purpose
    is to contain otherwise-significant characters. When ``strip_literals`` is true, their spans are
    blanked too, so a policy looking for call syntax cannot be satisfied by a diagnostic or test
    fixture string. Block comments nest, as they do in Rust. ``'a`` lifetimes are distinguished from
    ``'a'`` char literals.
    """
    def emit_literal(literal: str) -> str:
        if not strip_literals:
            return literal
        return "".join("\n" if char == "\n" else " " for char in literal)

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
            out.append(emit_literal(source[index:end]))
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
            out.append(emit_literal(source[index:end]))
            index = end
            continue
        if char == "'":
            # `'\n'` / `'\u{1f600}'`: an escape runs to the closing quote.
            if source.startswith("'\\", index):
                end = source.find("'", index + 2)
                end = length if end < 0 else end + 1
                out.append(emit_literal(source[index:end]))
                index = end
                continue
            # `'a'` is a char literal; `'a` (no closing quote) is a lifetime, so emit just the tick and
            # let the identifier be scanned normally.
            if index + 2 < length and source[index + 2] == "'":
                out.append(emit_literal(source[index : index + 3]))
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


def _split_cfg_args(expression: str) -> list[str]:
    """Split one cfg predicate's arguments at top-level commas."""
    args: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(expression):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            args.append(expression[start:index].strip())
            start = index + 1
    args.append(expression[start:].strip())
    return [arg for arg in args if arg]


def _cfg_truth_without_test(expression: str) -> tuple[bool, bool]:
    """Return ``(can_be_false, can_be_true)`` with the Rust ``test`` cfg fixed false.

    Other predicates are treated as independently unknown. That is deliberately conservative: an
    item is removed from a production scan only when its cfg expression cannot possibly be true in
    a non-test build.
    """
    expression = expression.strip()
    if expression == "test":
        return True, False
    call = re.fullmatch(r"(all|any|not)\s*\((.*)\)", expression, re.DOTALL)
    if call is None:
        return True, True
    kind, body = call.groups()
    values = [_cfg_truth_without_test(arg) for arg in _split_cfg_args(body)]
    if kind == "all":
        return any(can_false for can_false, _ in values), all(
            can_true for _, can_true in values
        )
    if kind == "any":
        return all(can_false for can_false, _ in values), any(
            can_true for _, can_true in values
        )
    if len(values) != 1:
        return True, True
    can_false, can_true = values[0]
    return can_true, can_false


def _cfg_attributes_requiring_test(text: str) -> list[tuple[int, int, bool]]:
    """Attributes whose cfg expression cannot be true when ``test`` is false.

    The third tuple member identifies inner ``#![...]`` attributes, which gate the enclosing
    module/file rather than a following construct.
    """
    attributes: list[tuple[int, int, bool]] = []
    for start in _CFG_ATTR_START.finditer(text):
        open_paren = start.end() - 1
        depth = 1
        cursor = open_paren + 1
        while cursor < len(text) and depth:
            if text[cursor] == "(":
                depth += 1
            elif text[cursor] == ")":
                depth -= 1
            cursor += 1
        if depth:
            continue
        close_paren = cursor - 1
        after = cursor
        while after < len(text) and text[after].isspace():
            after += 1
        if after >= len(text) or text[after] != "]":
            continue
        _, can_be_true = _cfg_truth_without_test(text[open_paren + 1 : close_paren])
        if not can_be_true:
            attributes.append((start.start(), after + 1, start.group(1) == "!"))
    return attributes


def _cfg_item_end(syntax: str, start: int) -> int | None:
    """Find the end of the Rust item following a cfg attribute.

    ``syntax`` has comments and literals blanked, so only Rust delimiters remain. Braces inside a
    function header (notably const-generic arguments such as ``Foo<{ 1 }>``) are skipped; only a
    top-level brace starts the item's body.
    """
    prefix = syntax[start:]
    prefix = re.sub(r"^\s*(?:#\s*\[[^\]]*\]\s*)*", "", prefix)
    prefix = re.sub(r"^pub(?:\s*\([^)]*\))?\s+", "", prefix)
    # These constructs own every brace in their initializer/type/path and end at a semicolon. A
    # function body, including a qualified `const fn`, ends at its top-level brace instead. String
    # literal blanking leaves an extern ABI as whitespace, so the same pattern covers `extern fn`
    # and `extern "ABI" fn`.
    const_function = (
        re.match(r"const\s+(?:async\s+)?(?:unsafe\s+)?(?:extern\s+)?fn\b", prefix)
        is not None
    )
    semicolon_item = (
        re.match(r"(?:const|static|type|use|let)\b", prefix) is not None
        and not const_function
    )

    paren_depth = 0
    bracket_depth = 0
    angle_depth = 0
    cursor = start
    while cursor < len(syntax):
        char = syntax[cursor]
        if char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth = max(0, paren_depth - 1)
        elif char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth = max(0, bracket_depth - 1)
        elif char == "<" and paren_depth == 0 and bracket_depth == 0:
            angle_depth += 1
        elif (
            char == ">"
            and paren_depth == 0
            and bracket_depth == 0
            and not (cursor > 0 and syntax[cursor - 1] == "-")
        ):
            angle_depth = max(0, angle_depth - 1)
        elif char == "{":
            end = _match_brace(syntax, cursor)
            if (
                not semicolon_item
                and paren_depth == 0
                and bracket_depth == 0
                and angle_depth == 0
            ):
                return end
            cursor = end
            continue
        elif (
            char == ";"
            and paren_depth == 0
            and bracket_depth == 0
            # A bare comparison such as `if 1 < 2` is indistinguishable from a generic opener to
            # this lightweight scanner. Recognized semicolon items still end here; their braced
            # initializers were skipped above, and nested delimiters remain protected.
            and (semicolon_item or angle_depth == 0)
        ):
            return cursor + 1
        cursor += 1
    return None


def _cfg_test_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of items that cannot compile outside a Rust test build.

    This covers functions and other items as well as modules, including predicates such as
    ``cfg(all(test, feature = "fixture"))``. An ``any(test, feature = "shipping")`` item is retained
    because it can compile in production when the other predicate is true.
    """
    spans: list[tuple[int, int]] = []
    syntax = strip_rust_comments(text, strip_literals=True)
    for attribute_start, attribute_end, inner in _cfg_attributes_requiring_test(syntax):
        if inner:
            # File-scope inner cfg gates the rest of the file. An inline-module inner cfg could be
            # narrowed to that module's closing brace, but blanking farther is safely fail-closed:
            # trigger discovery deliberately uses a separate cfg-unblanked syntax stream.
            spans.append((attribute_start, len(text)))
            continue
        # Additional attributes and visibility modifiers are intentionally included in the scan
        # from the cfg attribute onward.
        end = _cfg_item_end(syntax, attribute_end)
        if end is None:
            continue
        spans.append((attribute_start, end))
    return spans


def _blank_spans(text: str, spans: list[tuple[int, int]]) -> str:
    """Blank selected source spans while preserving newlines and character offsets."""
    if not spans:
        return text
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    out: list[str] = []
    cursor = 0
    for start, end in merged:
        out.append(text[cursor:start])
        out.append("".join("\n" if char == "\n" else " " for char in text[start:end]))
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


def _line_of(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def _configure_decode_hooks_are_unconditional_rejections(text: str) -> bool:
    """Whether every concrete ``configure_decode`` hook is a single typed ``Err`` expression.

    ``MemoryRequestScope`` requires the method even from a provider whose contract declares rung 2
    Missing. Such a hook is not evidence that native tile geometry can reach the PiD seam. Keep the
    exemption deliberately narrow: any statement, delegation, ``?``, or success arm before the
    rejection remains an adoption signal and must use ``DecodeRoutes``.
    """
    starts = list(re.finditer(r"\bfn\s+configure_decode\b", text))
    if not starts:
        return False
    for start in starts:
        open_index = text.find("{", start.end())
        if open_index < 0:
            return False
        end = _match_brace(text, open_index)
        if not _is_single_err_expression(text[open_index + 1 : end - 1]):
            return False
    return True


def _is_single_err_expression(body: str) -> bool:
    """Whether ``body`` is exactly one ``Err(...)`` expression and nothing else.

    Shared by the two decode-hook shapes a provider can write, so "unconditional rejection" has one
    definition rather than one per seam.
    """
    body = body.strip()
    prefix = re.match(r"Err\s*\(", body)
    if prefix is None:
        return False
    outer_open = body.find("(", prefix.start())
    return _match_paren(body, outer_open) == len(body)


# The shared MLX request-scope constructor. Its LAST positional argument is the provider's decode
# validator — the closure the runtime drives to apply (or refuse) a bounded-decode selection.
#
# SC-15525 review: this is the shape that actually matters, and the first revision of the rung-2
# exemption missed it entirely. **No MLX provider writes `fn configure_decode`** — that method lives
# once, on `mlx_gen::request_scope::MlxRequestScopeCore`, and every family reaches it by handing this
# constructor a closure. So an exemption that only inspected the trait-method form was vacuous for the
# entire MLX provider family: flip the closure's `Err` to `Ok` and a native geometry reaches the PiD
# seam with the gate silent.
MLX_REQUEST_SCOPE_CONSTRUCTOR = "MlxRequestScopeConfig::new"


def _split_top_level_args(text: str) -> list[str]:
    """Split a Rust argument list on top-level commas, ignoring nesting and string literals.

    A Rust trailing comma is idiomatic and `rustfmt` inserts one, so the empty tail it produces is
    dropped — otherwise every well-formatted call site would look like it passed nothing last.
    """
    args: list[str] = []
    depth = 0
    start = 0
    i = 0
    while i < len(text):
        char = text[i]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            args.append(text[start:i])
            start = i + 1
        elif char == '"':
            i += 1
            while i < len(text):
                if text[i] == "\\":
                    i += 1
                elif text[i] == '"':
                    break
                i += 1
        elif char == "|" and depth == 0:
            # A closure parameter list at argument level. Skip to its closing `|` so a `,` between
            # closure parameters is not mistaken for an argument separator.
            close = text.find("|", i + 1)
            if close != -1:
                i = close
        i += 1
    args.append(text[start:])
    while len(args) > 1 and not args[-1].strip():
        args.pop()
    return args


def _decode_validators_are_unconditional_rejections(text: str) -> bool:
    """Whether every decode validator handed to [`MLX_REQUEST_SCOPE_CONSTRUCTOR`] refuses outright.

    The closure twin of :func:`_configure_decode_hooks_are_unconditional_rejections`, and the one an
    MLX family actually writes. Deliberately narrow, in the same way: only a literal closure whose
    entire body is one ``Err(...)`` counts. A named function, a delegation, a ``match``, or any
    closure with a success arm is an adoption signal and must go through ``DecodeRoutes``.

    Returns ``False`` — never a silent pass — when the constructor is called in a shape this reader
    cannot resolve, so an unparseable call arms the gate instead of disarming it.
    """
    for start in re.finditer(re.escape(MLX_REQUEST_SCOPE_CONSTRUCTOR), text):
        open_index = text.find("(", start.end())
        if open_index < 0:
            return False
        end = _match_paren(text, open_index)
        if end > len(text):
            return False
        args = _split_top_level_args(text[open_index + 1 : end - 1])
        validator = args[-1].strip().rstrip(",").strip()
        closure = re.match(r"(?:move\s+)?\|[^|]*\|", validator)
        if closure is None:
            return False
        if not _is_single_err_expression(validator[closure.end() :]):
            return False
    return True


# `MemoryStrategy::BoundedDecode => <expr>,` — the contract arm that states this provider's rung-2
# support. The arm is the semantic fact the SC-15525 exemption is keyed on; `<expr>` is captured so a
# declaration made through a named `const` can be followed one hop.
_BOUNDED_DECODE_SUPPORT_ARM = re.compile(
    r"BoundedDecode\s*(?:if\s+[^=]*?)?=>\s*([^,\n]+?)\s*[,\n]"
)
_MISSING_SUPPORT_EXPR = re.compile(r"^(?:MemoryStrategySupport::)?Missing$")


def _declares_bounded_decode_missing(text: str) -> bool:
    """Whether this provider's contract textually declares ``BoundedDecode`` support **Missing**.

    SC-15525 review defeat (B): the first revision keyed its exemption on the *absence* of the
    ``decode_tile_edges`` / ``decode_overlaps`` literals, which is a proxy for "publishes no domain"
    that a provider defeats simply by building its ``MemoryParameterRanges`` in another crate — while
    declaring ``BoundedDecode`` **Implemented**. Absence of evidence was doing the work of evidence.

    So the exemption is now keyed on the positive claim instead: the crate must *say* rung 2 is
    Missing. Two spellings are accepted, because both ship today — the support expression written
    inline, and the same expression behind a named ``const`` this module follows exactly one hop
    (SDXL's ``DECODE_SUPPORT``). Anything else — a computed support, a second hop, an arm this reader
    cannot parse — is not a declaration it can verify, and the provider falls through to the route
    checks.

    At least one arm must be found: a crate with no ``BoundedDecode`` arm at all has declared nothing
    and is not exempt.
    """
    arms = _BOUNDED_DECODE_SUPPORT_ARM.findall(text)
    if not arms:
        return False
    for expr in arms:
        expr = expr.strip()
        if _MISSING_SUPPORT_EXPR.match(expr):
            continue
        # One hop through a named const: `const NAME: MemoryStrategySupport = ...::Missing;`
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", expr):
            return False
        const_decl = re.search(
            r"\bconst\s+" + re.escape(expr) + r"\s*:\s*MemoryStrategySupport\s*=\s*"
            r"(?:MemoryStrategySupport::)?Missing\s*;",
            text,
        )
        if const_decl is None:
            return False
    return True


def _match_paren(text: str, open_index: int) -> int:
    """Index just past the ``)`` matching ``open_index``, ignoring Rust strings and comments."""
    depth = 0
    i = open_index
    while i < len(text):
        char = text[i]
        if char == "(":
            depth += 1
            i += 1
        elif char == ")":
            depth -= 1
            i += 1
            if depth == 0:
                return i
        elif char == '"':
            i += 1
            while i < len(text):
                if text[i] == "\\":
                    i += 2
                elif text[i] == '"':
                    i += 1
                    break
                else:
                    i += 1
        elif char == "/" and i + 1 < len(text) and text[i + 1] == "/":
            newline = text.find("\n", i)
            i = len(text) if newline == -1 else newline
        elif char == "/" and i + 1 < len(text) and text[i + 1] == "*":
            close = text.find("*/", i + 2)
            i = len(text) if close == -1 else close + 2
        else:
            i += 1
    return len(text) + 1


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

    Both are call-syntax matches on production code with comments and literals blanked, which closes
    the three defeats the
    adversarial review found in the first revision (a ``// TODO`` mentioning ``DecodeRoutes``, an unused
    ``use``, and a trigger set narrow enough to miss a helper-built ``MemoryParameterRanges``).

    It remains a *static text* check, so its limits are named rather than implied away:

    1. It cannot prove the ``validate`` call is reached on every admission path — only that one exists.
       The weights-free registry conformance walk verifies that behaviour directly through
       ``MemoryRegistration::safety_check`` and the contract's ``pid_decode_routes`` declaration.
    2. The rung-2 trigger is a *text* proxy for the semantic fact "this provider's contract publishes
       non-empty ``decode_tile_edges``". A provider could in principle delegate every textual trace of
       rung 2 — the ranges, the strategy name, and its ``configure_decode`` hook — to another crate. The
       registry walk covers the semantic declaration; the trigger set remains deliberately wide
       (fail-closed) as an earlier diagnostic.
    3. The two *exemptions* below are narrower than the trigger set, and each is keyed on a positive,
       checkable claim rather than on an absence — because SC-15525's review showed an
       absence-keyed exemption is defeated by moving the absent text into another crate. Every
       exemption additionally requires that **both** decode seams a provider can write refuse
       unconditionally: the ``configure_decode`` trait method *and* the closure handed to
       ``MlxRequestScopeConfig::new``. The latter is the one that matters in practice — no MLX
       provider writes the former.

    String/character literals and constructs that cannot compile without the Rust ``test`` cfg are
    excluded from evidence matching. Trigger matching uses a separate cfg-unblanked syntax stream,
    so a parser overmatch can fail closed but cannot silently disarm the gate.
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
        # Comments/literals are stripped before all matching. Trigger discovery deliberately does
        # NOT use cfg-item blanking: if that lightweight parser ever overblanks an unusual Rust
        # construct, the gate fails loudly for missing evidence instead of silently disarming itself
        # by erasing a real registration or rung-2 marker.
        trigger_sources: list[str] = []
        evidence_sources: list[str] = []
        for path in sorted(manifest_dir.rglob("*.rs")):
            if not IGNORED_TREE_PARTS.isdisjoint(path.relative_to(root).parts):
                continue
            source = path.read_text(encoding="utf-8")
            trigger_sources.append(strip_rust_comments(source, strip_literals=True))
            production = _blank_spans(source, _cfg_test_spans(source))
            evidence_sources.append(strip_rust_comments(production, strip_literals=True))
        triggers = "\n".join(trigger_sources)
        evidence = "\n".join(evidence_sources)
        if "register_memory_strategy" not in triggers:
            continue
        rung_two_markers = [marker for marker in PID_RUNG_TWO_MARKERS if marker in triggers]
        if not rung_two_markers:
            continue
        # SC-15800: `MemoryRequestScope` requires `configure_decode` even for a rung-4-only adopter.
        # A single-expression typed rejection cannot emit a native tile into the PiD seam, so do not
        # mistake that trait-completeness method for a bounded-decode implementation. Every broader
        # hook shape still fails closed through the route checks below.
        #
        # Both exemptions below share one precondition, and it is the one the SC-15525 review found
        # missing: **every** decode hook the crate writes — the trait method AND the closure handed to
        # the shared MLX request scope — must be an unconditional rejection. See
        # `_decode_validators_are_unconditional_rejections` for why the closure is the load-bearing
        # half on MLX.
        hooks_all_refuse = (
            "fn configure_decode" not in triggers
            or _configure_decode_hooks_are_unconditional_rejections(triggers)
        ) and _decode_validators_are_unconditional_rejections(triggers)
        if (
            rung_two_markers == ["configure_decode"]
            and _configure_decode_hooks_are_unconditional_rejections(evidence)
            and hooks_all_refuse
        ):
            continue
        # SC-15525: the same exemption, for the provider shape that declares rung 2 **Missing** after
        # measuring it. Such a provider must still NAME `BoundedDecode` (to declare the support) and
        # `MemoryParameterRanges` (the field's type, which it populates for rung 4), so the trigger
        # set alone cannot distinguish it from an adopter.
        #
        # It is keyed on the **positive claim**, not on the absence of one. The first revision keyed it
        # on the *absence* of the `decode_tile_edges` / `decode_overlaps` literals — a proxy for
        # "publishes no domain" that review defeat (B) broke by building `MemoryParameterRanges` in
        # another crate while declaring `BoundedDecode` **Implemented**: absence of evidence was doing
        # the work of evidence. Now the crate must SAY rung 2 is Missing
        # (`_declares_bounded_decode_missing`), and a provider that says so cannot also be publishing a
        # domain — so the two domain markers stay as a corroborating, fail-closed necessary condition
        # rather than as the key.
        #
        # The support declaration is read off EVIDENCE (cfg(test)-blanked): a `#[cfg(test)]` fixture
        # asserting `BoundedDecode => Missing` must not be able to buy a production exemption. The
        # domain and hook checks are read off TRIGGERS (unblanked) for the mirror-image reason — a
        # `#![cfg(test)]` file would otherwise erase the very domain (or the very accepting closure)
        # that arms the gate. Each stream is chosen so that a parser slip over-arms rather than
        # disarms.
        publishes_decode_domain = any(
            marker in triggers for marker in ("decode_tile_edges", "decode_overlaps")
        )
        if (
            _declares_bounded_decode_missing(evidence)
            and not publishes_decode_domain
            and hooks_all_refuse
        ):
            continue
        missing: list[str] = []
        if not any(marker in evidence for marker in PID_DECODE_ROUTE_CONSTRUCTION_MARKERS):
            spellings = " or ".join(
                f"`{marker}`" for marker in PID_DECODE_ROUTE_CONSTRUCTION_MARKERS
            )
            missing.append(f"never calls the checked constructor ({spellings})")
        if not PID_DECODE_ROUTE_ADMISSION_CALL.search(evidence):
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


# A path literal shaped like a model/cache store — the thing a `$HOME` read is being used to build.
# Deliberately narrow: a `$HOME` read that joins nothing store-shaped (a bare `home()` accessor, a
# `~/` expander) is not the defect and must not be flagged, or the lint gets switched off.
SNAPSHOT_STORE_LITERAL = re.compile(
    r'"\.cache/|"Library/Application Support/|"Repos/|"\.\w+/models'
)
HOME_READ = re.compile(r'env::var(?:_os)?\("HOME"\)')
# Any env read that is NOT `"HOME"` — a named override, or one whose name arrives as a binding
# (`env::var(var)`, `env::var(format!(…))`). Its presence means the chain HAS an override.
NON_HOME_ENV_READ = re.compile(r'env::var(?:_os)?\(\s*(?!"HOME")')


def _enclosing_fn(lines: list[str], index: int) -> tuple[int, list[str]]:
    """The `fn` containing line ``index``, as (start line, body lines)."""
    for start in range(index, -1, -1):
        if re.match(r"\s*(pub\s+)?(async\s+)?fn\s+\w+", lines[start]):
            depth, cursor, opened = 0, start, False
            while cursor < len(lines):
                depth += lines[cursor].count("{") - lines[cursor].count("}")
                if "{" in lines[cursor]:
                    opened = True
                if opened and depth <= 0:
                    return start, lines[start : cursor + 1]
                cursor += 1
            return start, lines[start:]
    return index, [lines[index]]


def check_snapshot_path_derivation(root: Path) -> None:
    """Fail when a snapshot/cache path is derived from ``$HOME`` with no override in the chain.

    ``check_rust_sources`` already bans HF-cache *references* under epic 13657, but only in
    production ``src/``. The same defect survived in test harnesses, where nothing linted it: a
    resolver that reads ``$HOME`` unconditionally means pointing an env var at a real store reads
    somewhere else entirely, and the suite skips or mis-resolves **while still reporting green**.
    That is the failure mode worth a gate — a red row says "fix me", a silently-skipped one says
    nothing at all.

    Two shapes hide behind a naive ``grep`` for ``env::var("HOME")`` and only one is a defect:

    * **Fallback** — an override is read first and ``$HOME`` is only the default. Harmless, and the
      shape this repo's own passing harnesses use.
    * **Derivation** — no override anywhere in the resolution chain.

    What this enforces, precisely, and no more: a ``$HOME`` read in a function that **also** joins a
    store-shaped literal (``SNAPSHOT_STORE_LITERAL``) and reads **no** other env var. All three
    conditions are needed. Dropping the store-literal one re-flags every bare ``fn home()`` accessor
    and every ``~/``-expander; dropping the env-read one re-flags every legitimate
    ``env::var(NAME).unwrap_or_else(|_| home.join(…))``.

    Two residual gaps, named rather than implied away:

    1. It is **function-scoped**, so a resolver that reads its override in one function and joins
       ``$HOME`` in another reads as a derivation. No such site exists today — the accessor shapes
       in `mlx-gen-scail2` and `mlx-gen-krea-realtime` join nothing and are excluded by the store
       literal — but a future one would need the override moved into the joining function, or an
       exemption argued here.
    2. It cannot tell a *derived cache* (correctly given a ``$HOME`` fallback) from a *provided
       input* (which should hard-fail with the epic-13657 message). Both satisfy it. Which of the
       two a path is remains a judgement the author makes; see `mlx-gen-boogu`'s
       ``BOOGU_VISION_TEST_IMAGE`` for the required shape and `converted_root()` for the fallback.
    """
    violations: list[str] = []
    for path in sorted(root.rglob("*.rs")):
        relative = path.relative_to(root)
        if not IGNORED_TREE_PARTS.isdisjoint(relative.parts):
            continue
        lines = strip_rust_comments(path.read_text(encoding="utf-8")).split("\n")
        for index, line in enumerate(lines):
            if not HOME_READ.search(line):
                continue
            start, body = _enclosing_fn(lines, index)
            text = "\n".join(body)
            if not SNAPSHOT_STORE_LITERAL.search(text):
                continue
            if NON_HOME_ENV_READ.search(text):
                continue
            violations.append(f"{relative}:{start + 1}: {lines[start].strip()}")

    if violations:
        joined = "\n  ".join(violations)
        fail(
            "a snapshot/cache path is derived from $HOME with no env override in the resolution "
            "chain, so pointing a variable at a real store cannot reach it and the row skips while "
            "looking green. Add the override (keep $HOME as the default for a derived cache; hard-"
            "fail with the epic-13657 message for a provided input):\n  " + joined
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
        check_snapshot_path_derivation(ROOT)
    except (AssertionError, json.JSONDecodeError) as error:
        print(f"workspace gate: FAIL: {error}", file=sys.stderr)
        return 1

    print(
        "workspace gate: OK "
        f"({EXPECTED_MEMBER_COUNT} path members, one lockfile, explicit registries, pinned backends, "
        "intentional tokenizer split, no network clients, no HF-cache references, "
        "no $HOME-derived snapshot paths)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
