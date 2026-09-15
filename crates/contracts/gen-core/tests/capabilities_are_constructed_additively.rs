//! sc-19561 — every `Capabilities { .. }` literal in the workspace must end in a base expression.
//!
//! **Why this is a source scan and not a type-system guard.** `Capabilities` is constructed from
//! ~70 provider crates, all of them outside `gen-core`. Both compiler-enforced options were tried
//! and both make cross-crate construction *impossible*, not merely additive:
//!
//! * `#[non_exhaustive]` — E0639, "cannot create non-exhaustive struct using functional record
//!   update syntax". The attribute forbids the literal outright outside the defining crate, so the
//!   `..Default::default()` form the struct's own docs prescribe stops compiling everywhere.
//! * a private `_non_exhaustive: ()` field — E0451, "field `_private` of struct `Caps` is private".
//!   Functional record update is refused for the same reason: the base expression's unnamed fields
//!   must all be visible at the construction site.
//!
//! A builder (`Capabilities::default().with_guidance(true)`) would compile, but it buys the same
//! additivity this scan already gives while adding a `with_*` method per field and rewriting all
//! ~70 descriptors into method chains. So the invariant is checked here, structurally: **every
//! literal is followed by a `..base` tail.** There is no maintained count anywhere in this file —
//! the assertion is over whatever literals exist at the time it runs.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// One `Capabilities { .. }` struct literal found in a source file.
#[derive(Debug)]
struct Site {
    line: usize,
    additive: bool,
}

/// Replace every comment and string-literal byte with a space, preserving byte offsets (so line
/// numbers survive) and respecting char and raw-string literals.
///
/// Load-bearing on both counts. `Capabilities`' own rustdoc contains the prescribed
/// `Capabilities { supports_guidance: true, ..Default::default() }` example, and a doc comment
/// wrapping it splits the literal across `///` prefixes — scanning raw text would report that
/// prose as an exhaustive construction site. String contents matter for the same reason: this
/// very file quotes `Capabilities { .. }` snippets inside its own fixtures.
fn strip_comments(src: &str) -> String {
    let b = src.as_bytes();
    let mut out: Vec<u8> = b.to_vec();
    let mut i = 0;
    let blank = |out: &mut Vec<u8>, from: usize, to: usize| {
        for byte in out.iter_mut().take(to).skip(from) {
            if *byte != b'\n' {
                *byte = b' ';
            }
        }
    };
    while i < b.len() {
        match b[i] {
            b'/' if i + 1 < b.len() && b[i + 1] == b'/' => {
                let start = i;
                while i < b.len() && b[i] != b'\n' {
                    i += 1;
                }
                blank(&mut out, start, i);
            }
            b'/' if i + 1 < b.len() && b[i + 1] == b'*' => {
                let start = i;
                i += 2;
                let mut depth = 1usize;
                while i < b.len() && depth > 0 {
                    if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
                        depth += 1;
                        i += 2;
                    } else if b[i] == b'*' && i + 1 < b.len() && b[i + 1] == b'/' {
                        depth -= 1;
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                blank(&mut out, start, i);
            }
            b'r' => {
                // A raw string: `r"..."` / `r#"..."#`. Skip it whole so a `//` inside cannot be
                // mistaken for a comment.
                let mut j = i + 1;
                let mut hashes = 0;
                while j < b.len() && b[j] == b'#' {
                    hashes += 1;
                    j += 1;
                }
                if j < b.len() && b[j] == b'"' {
                    let start = j;
                    j += 1;
                    loop {
                        if j >= b.len() {
                            break;
                        }
                        if b[j] == b'"' {
                            let mut k = j + 1;
                            let mut seen = 0;
                            while k < b.len() && b[k] == b'#' && seen < hashes {
                                seen += 1;
                                k += 1;
                            }
                            if seen == hashes {
                                j = k;
                                break;
                            }
                        }
                        j += 1;
                    }
                    blank(&mut out, start, j);
                    i = j;
                } else {
                    i += 1;
                }
            }
            b'"' => {
                let start = i;
                i += 1;
                while i < b.len() {
                    if b[i] == b'\\' {
                        i += 2;
                        continue;
                    }
                    if b[i] == b'"' {
                        i += 1;
                        break;
                    }
                    i += 1;
                }
                blank(&mut out, start, i);
            }
            b'\'' => {
                // Distinguish a char literal from a lifetime precisely rather than by proximity:
                // `&'a Vec<&'b T>` would otherwise look like the char literal `'a Vec<&'`. Getting
                // this wrong matters because `'"'` is real code, and mistaking it for a string
                // start would blank the rest of the file and blind the scan.
                if i + 1 < b.len() && b[i + 1] == b'\\' {
                    // `'\n'`, `'\''`, `'\u{1F600}'` — escape, so scan to the closing quote.
                    let close = b[i + 2..b.len().min(i + 12)]
                        .iter()
                        .position(|&c| c == b'\'');
                    i += close.map_or(1, |off| off + 3);
                } else if i + 2 < b.len() && b[i + 2] == b'\'' {
                    i += 3; // `'x'`
                } else {
                    i += 1; // a lifetime
                }
            }
            _ => i += 1,
        }
    }
    String::from_utf8(out).expect("blanking ASCII comment bytes preserves UTF-8 boundaries")
}

/// `true` when the byte before a `Capabilities {` occurrence makes it an item header or a function
/// signature rather than a struct literal: `struct`/`impl`/`for`/`enum`/`trait`/`union` and
/// `-> [path::]Capabilities {`.
fn is_not_a_literal(before: &str) -> bool {
    let trimmed = before.trim_end_matches(|c: char| c.is_alphanumeric() || c == '_' || c == ':');
    let head = trimmed.trim_end();
    head.ends_with("->")
        || head.ends_with("struct")
        || head.ends_with("impl")
        || head.ends_with("for")
        || head.ends_with("enum")
        || head.ends_with("trait")
        || head.ends_with("union")
}

/// Every `Capabilities { .. }` struct literal in `src`, and whether each carries a `..base` tail.
fn literal_sites(src: &str) -> Vec<Site> {
    let stripped = strip_comments(src);
    let bytes = stripped.as_bytes();
    let mut sites = Vec::new();
    let mut from = 0;
    while let Some(rel) = stripped[from..].find("Capabilities") {
        let at = from + rel;
        from = at + "Capabilities".len();
        // Not part of a longer identifier (`MemoryStrategyCapabilities`, `LlmCapabilities`).
        if at > 0 {
            let prev = bytes[at - 1];
            if prev.is_ascii_alphanumeric() || prev == b'_' {
                continue;
            }
        }
        let rest = &stripped[from..];
        let ws = rest.len() - rest.trim_start().len();
        if !rest[ws..].starts_with('{') {
            continue;
        }
        if is_not_a_literal(&stripped[..at]) {
            continue;
        }
        let open = from + ws;
        let mut depth = 0usize;
        let mut end = open;
        for (off, c) in stripped[open..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = open + off;
                        break;
                    }
                }
                _ => {}
            }
        }
        let inner = &stripped[open + 1..end];
        sites.push(Site {
            line: stripped[..at].matches('\n').count() + 1,
            additive: has_base_expression(inner),
        });
        from = end;
    }
    sites
}

/// Whether a literal body carries a top-level `..base` tail.
///
/// Two things are checked, and both are load-bearing:
///
/// * **Depth 0.** A nested literal inside a field value (a `GenerationRequest { ..base_req() }` in
///   a test fixture) must not satisfy the outer literal.
/// * **Element position.** The `..` must begin a top-level element — the byte before it, skipping
///   whitespace, is the start of the body or a `,`. Without this, ANY depth-0 `..` counts, so an
///   exhaustive literal whose last field happens to be range-valued (`steps: 1..=50`, or a future
///   `size_range: 256..2048`) would be waved through as additive. There is no such field today, so
///   the hole is latent rather than live — which is exactly when it is cheap to close, and the
///   guard's whole job is to stay correct as fields are ADDED.
fn has_base_expression(inner: &str) -> bool {
    let b = inner.as_bytes();
    let mut depth = 0i32;
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth -= 1,
            b'.' if depth == 0 && i + 1 < b.len() && b[i + 1] == b'.' => {
                let starts_element = b[..i]
                    .iter()
                    .rposition(|c| !c.is_ascii_whitespace())
                    .is_none_or(|p| b[p] == b',');
                if starts_element {
                    return true;
                }
                i += 1; // a range operator; skip its second dot so `..=` is not re-examined
            }
            _ => {}
        }
        i += 1;
    }
    false
}

fn workspace_root() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        if dir.join("crates/contracts/gen-core/Cargo.toml").is_file() {
            return dir;
        }
        assert!(
            dir.pop(),
            "no ancestor of {} contains crates/contracts/gen-core/Cargo.toml",
            env!("CARGO_MANIFEST_DIR")
        );
    }
}

/// Every `.rs` file under `crates/`, skipping build output and the vendored candle kernels.
fn rust_sources(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.join("crates")];
    while let Some(dir) = stack.pop() {
        let entries = std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("read {dir:?}: {e}"));
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if path.is_dir() {
                if name != "target" && name != "vendor" && !name.starts_with('.') {
                    stack.push(path);
                }
            } else if name.ends_with(".rs") {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

/// The workspace's member crate directories, expanded from the root manifest's `members` globs.
///
/// Derived from the manifest rather than listed here, so a new crate is covered the moment it is
/// added to the workspace — there is no number or list to maintain.
fn member_dirs(root: &Path) -> BTreeSet<PathBuf> {
    let manifest = std::fs::read_to_string(root.join("Cargo.toml")).expect("root Cargo.toml");
    let members = manifest
        .split("members = [")
        .nth(1)
        .and_then(|s| s.split(']').next())
        .expect("root manifest has a members list");
    let mut dirs = BTreeSet::new();
    for raw in members.split(',') {
        let pattern = raw.trim().trim_matches('"');
        if pattern.is_empty() {
            continue;
        }
        match pattern.rsplit_once('/') {
            Some((parent, leaf)) if leaf.contains('*') => {
                let prefix = leaf.trim_end_matches('*');
                let parent_dir = root.join(parent);
                let entries = std::fs::read_dir(&parent_dir)
                    .unwrap_or_else(|e| panic!("read {parent_dir:?}: {e}"));
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir()
                        && entry.file_name().to_string_lossy().starts_with(prefix)
                        && path.join("Cargo.toml").is_file()
                    {
                        dirs.insert(path);
                    }
                }
            }
            _ => {
                let path = root.join(pattern);
                if path.join("Cargo.toml").is_file() {
                    dirs.insert(path);
                }
            }
        }
    }
    assert!(!dirs.is_empty(), "expanded zero workspace members");
    dirs
}

// ---------------------------------------------------------------------------------------------
// The predicate discriminates. Without these, a parser that reported "additive" unconditionally —
// or found nothing at all — would keep the sweep below green forever.
// ---------------------------------------------------------------------------------------------

#[test]
fn an_exhaustive_literal_is_flagged() {
    let src = "fn d() -> ModelDescriptor {\n    ModelDescriptor {\n        capabilities: \
               Capabilities {\n            max_count: 1,\n            mac_only: true,\n        \
               },\n    }\n}\n";
    let sites = literal_sites(src);
    assert_eq!(sites.len(), 1, "one literal, got {sites:?}");
    assert!(!sites[0].additive, "an exhaustive literal must be flagged");
}

#[test]
fn a_functional_update_literal_passes() {
    for base in [
        "..Default::default()",
        "..caps()",
        "..other.capabilities.clone()",
    ] {
        let src = format!("let c = Capabilities {{\n    max_count: 1,\n    {base}\n}};\n");
        let sites = literal_sites(&src);
        assert_eq!(sites.len(), 1, "one literal for base {base}, got {sites:?}");
        assert!(sites[0].additive, "`{base}` is a base expression");
    }
}

#[test]
fn a_nested_base_expression_does_not_satisfy_the_outer_literal() {
    // The trap this guard must not fall into: a field value that itself uses functional update.
    // Counting any `..` in the body would call this exhaustive literal additive.
    let src =
        "let c = Capabilities {\n    conditioning: pick(&Req { steps: None, ..base() }),\n    \
               max_count: 1,\n};\n";
    let sites = literal_sites(src);
    assert_eq!(sites.len(), 1, "one literal, got {sites:?}");
    assert!(
        !sites[0].additive,
        "a `..` nested inside a field value is not the literal's own base expression"
    );
}

#[test]
fn a_range_valued_last_field_does_not_pass_as_a_base_expression() {
    // The other half of the same trap, and the one that survives to depth 0: an EXHAUSTIVE literal
    // whose last field is range-valued. `..` alone is not the discriminator — where it sits is. No
    // `Capabilities` field is range-valued today, so this is a latent hole rather than a live one;
    // it is closed here because the guard exists precisely to survive new fields being added.
    for tail in ["steps: 1..=50", "sizes: 256..2048", "steps: (1..=50).len()"] {
        let src = format!("let c = Capabilities {{\n    max_count: 1,\n    {tail},\n}};\n");
        let sites = literal_sites(&src);
        assert_eq!(sites.len(), 1, "one literal for {tail}, got {sites:?}");
        assert!(
            !sites[0].additive,
            "`{tail}` is a range, not a base expression — this literal is exhaustive"
        );
    }
}

#[test]
fn item_headers_and_signatures_are_not_literals() {
    let src =
        "pub struct Capabilities {\n    pub a: bool,\n}\nimpl Capabilities {\n    fn f() {}\n\
               }\nimpl Default for Capabilities {\n    fn default() -> Self { Self { a: false } }\n\
               }\nfn caps() -> Capabilities {\n    todo!()\n}\nfn g() -> gen_core::Capabilities {\n\
               todo!()\n}\n";
    assert!(
        literal_sites(src).is_empty(),
        "item headers and return types are not struct literals: {:?}",
        literal_sites(src)
    );
}

#[test]
fn a_longer_identifier_is_not_a_capabilities_literal() {
    let src = "let m = MemoryStrategyCapabilities {\n    tiling: true,\n};\n";
    assert!(
        literal_sites(src).is_empty(),
        "`MemoryStrategyCapabilities` is a different type"
    );
}

#[test]
fn commented_out_and_quoted_literals_are_ignored() {
    let src = "/// `Capabilities { supports_guidance: true,\n/// ..Default::default() }`\n\
               // let dead = Capabilities { max_count: 1 };\n\
               /* Capabilities { max_count: 1 } */\n\
               let s = \"Capabilities { max_count: 1 }\";\n";
    assert!(
        literal_sites(src).is_empty(),
        "prose and string contents are not construction sites: {:?}",
        literal_sites(src)
    );
}

// ---------------------------------------------------------------------------------------------
// The sweep.
// ---------------------------------------------------------------------------------------------

/// The reach proof. A sweep that walked the wrong directory, or silently read nothing, would
/// report zero violations forever. This asserts the walker visited **every** workspace member —
/// expanded from the root manifest, so it needs no maintained list — and that the literal scanner
/// actually matched something in a crate other than `gen-core`.
#[test]
fn the_sweep_reaches_every_workspace_member() {
    let root = workspace_root();
    let sources = rust_sources(&root);
    for member in member_dirs(&root) {
        assert!(
            sources.iter().any(|p| p.starts_with(&member)),
            "the source walk visited no .rs file under workspace member {}",
            member.display()
        );
    }

    let mut crates_with_literals = BTreeSet::new();
    for path in &sources {
        if !literal_sites(&std::fs::read_to_string(path).unwrap()).is_empty() {
            let rel = path.strip_prefix(&root).unwrap().to_path_buf();
            crates_with_literals.insert(rel);
        }
    }
    assert!(
        crates_with_literals
            .iter()
            .any(|p| !p.starts_with("crates/contracts")),
        "the scanner found no Capabilities literal outside the contract crates — it is not \
         reaching the provider crates this guard exists for"
    );
}

/// The invariant: no `Capabilities` literal anywhere in the workspace constructs the struct
/// exhaustively, so adding a field to it is additive rather than a repo-wide compile break.
#[test]
fn every_capabilities_literal_carries_a_base_expression() {
    let root = workspace_root();
    let mut violations = Vec::new();
    for path in rust_sources(&root) {
        let src = std::fs::read_to_string(&path).unwrap();
        for site in literal_sites(&src) {
            if !site.additive {
                violations.push(format!(
                    "{}:{}",
                    path.strip_prefix(&root).unwrap().display(),
                    site.line
                ));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "these `Capabilities {{ .. }}` literals name every field instead of ending in \
         `..Default::default()`, so the next field added to `Capabilities` breaks each of them \
         (sc-19561):\n  {}",
        violations.join("\n  ")
    );
}
