//! `REG-2` / `TOOL-4`: no production source file under `crates/media/` or `crates/audio/`
//! constructs a `Capabilities { .. }` struct literal directly.
//!
//! Every provider's [`Capabilities`](gen_core::Capabilities) is computed exhaustively from its
//! primary [`RouteRecord`](gen_core::RouteRecord) declarations via
//! [`RouteRecord::compute_capabilities`](gen_core::RouteRecord::compute_capabilities) in
//! `sceneworks-gen-core`. Hand-written `Capabilities` literals outside `gen-core` bypass primary
//! declarations and let advertised capabilities drift from execution.

use std::path::{Path, PathBuf};

/// One violating `Capabilities { .. }` literal site in production code.
#[derive(Debug, PartialEq, Eq)]
struct Violation {
    line: usize,
}

/// Blank comments, strings, raw strings, and char literals with ASCII spaces while preserving
/// newlines so reported line numbers match the original source.
fn strip_comments_and_strings(src: &str) -> String {
    let b = src.as_bytes();
    let mut out = b.to_vec();
    let mut i = 0;
    let blank = |out: &mut [u8], from: usize, to: usize| {
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
            b'r' | b'b' => {
                let mut j = i + 1;
                if b[i] == b'b' && j < b.len() && b[j] == b'r' {
                    j += 1;
                } else if b[i] == b'b' {
                    i += 1;
                    continue;
                }
                let mut hashes = 0usize;
                while j < b.len() && b[j] == b'#' {
                    hashes += 1;
                    j += 1;
                }
                if j < b.len() && b[j] == b'"' {
                    let start = j;
                    j += 1;
                    while j < b.len() {
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
                if i + 1 < b.len() && b[i + 1] == b'\\' {
                    let close = b[i + 2..b.len().min(i + 12)]
                        .iter()
                        .position(|&c| c == b'\'');
                    let end = i + close.map_or(1, |off| off + 3);
                    blank(&mut out, i, end);
                    i = end;
                } else if i + 2 < b.len() && b[i + 2] == b'\'' {
                    blank(&mut out, i, i + 3);
                    i += 3;
                } else {
                    i += 1;
                }
            }
            _ => i += 1,
        }
    }
    String::from_utf8(out).expect("blanking ASCII ranges preserves UTF-8")
}

fn is_test_only_cfg_predicate(predicate: &str) -> bool {
    let trimmed = predicate.trim();
    if trimmed == "test" {
        return true;
    }
    if let Some(inner) = trimmed
        .strip_prefix("all(")
        .and_then(|rest| rest.strip_suffix(')'))
    {
        return inner.split(',').any(|part| part.trim() == "test");
    }
    false
}

/// Blank file-level `#![cfg(test)]` files and item-level `#[cfg(test)]` / `#[cfg(all(test, ..))]`
/// blocks (including nested modules/functions) while preserving newlines.
fn strip_cfg_test_items(code: &str) -> String {
    let b = code.as_bytes();
    let mut out = b.to_vec();
    let blank = |out: &mut [u8], from: usize, to: usize| {
        for byte in out.iter_mut().take(to).skip(from) {
            if *byte != b'\n' {
                *byte = b' ';
            }
        }
    };

    let mut i = 0;
    while i < b.len() {
        if b[i] != b'#' {
            i += 1;
            continue;
        }
        let is_inner = i + 1 < b.len() && b[i + 1] == b'!';
        let bracket = if is_inner { i + 2 } else { i + 1 };
        if bracket >= b.len() || b[bracket] != b'[' {
            i += 1;
            continue;
        }
        let Some(attr_end) = find_matching_delim(b, bracket, b'[', b']') else {
            break;
        };
        let attr_body = code[bracket + 1..attr_end].trim();
        let Some(pred) = attr_body
            .strip_prefix("cfg(")
            .and_then(|rest| rest.strip_suffix(')'))
        else {
            i = attr_end + 1;
            continue;
        };
        if !is_test_only_cfg_predicate(pred) {
            i = attr_end + 1;
            continue;
        }

        if is_inner {
            // File-level `#![cfg(test)]`: the entire file is test-only.
            blank(&mut out, 0, b.len());
            break;
        }

        // Item-level `#[cfg(test)]`: skip any stacked outer attributes and blank through the
        // item's terminating `;` or balanced `{ .. }`.
        let mut cursor = attr_end + 1;
        loop {
            while cursor < b.len() && b[cursor].is_ascii_whitespace() {
                cursor += 1;
            }
            if cursor + 1 < b.len() && b[cursor] == b'#' && b[cursor + 1] == b'[' {
                if let Some(next_end) = find_matching_delim(b, cursor + 1, b'[', b']') {
                    cursor = next_end + 1;
                    continue;
                }
            }
            break;
        }
        let mut end = cursor;
        while end < b.len() {
            match b[end] {
                b';' => {
                    end += 1;
                    break;
                }
                b'{' => {
                    if let Some(close) = find_matching_delim(b, end, b'{', b'}') {
                        end = close + 1;
                    } else {
                        end = b.len();
                    }
                    break;
                }
                _ => end += 1,
            }
        }
        blank(&mut out, i, end);
        i = end;
    }

    String::from_utf8(out).expect("blanking ASCII ranges preserves UTF-8")
}

fn find_matching_delim(b: &[u8], open_idx: usize, open: u8, close: u8) -> Option<usize> {
    let mut depth = 0usize;
    let mut i = open_idx;
    while i < b.len() {
        if b[i] == open {
            depth += 1;
        } else if b[i] == close {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        }
        i += 1;
    }
    None
}

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

/// Find every production `Capabilities { .. }` struct literal in `src`.
fn find_production_capabilities_literals(src: &str) -> Vec<Violation> {
    let code = strip_cfg_test_items(&strip_comments_and_strings(src));
    let bytes = code.as_bytes();
    let mut violations = Vec::new();
    let mut from = 0;
    while let Some(rel) = code[from..].find("Capabilities") {
        let at = from + rel;
        from = at + "Capabilities".len();
        if at > 0 {
            let prev = bytes[at - 1];
            if prev.is_ascii_alphanumeric() || prev == b'_' {
                continue;
            }
        }
        let rest = &code[from..];
        let ws = rest.len() - rest.trim_start().len();
        if !rest[ws..].starts_with('{') {
            continue;
        }
        if is_not_a_literal(&code[..at]) {
            continue;
        }
        violations.push(Violation {
            line: code[..at].matches('\n').count() + 1,
        });
    }
    violations
}

/// Collect production `.rs` files under `dir`, skipping `vendor`, `tests`, `examples`, `benches`,
/// `target`, and out-of-line test modules (`tests.rs`, `*_tests.rs`).
fn collect_production_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut paths: Vec<PathBuf> = entries
        .map(|entry| entry.expect("readable directory entry").path())
        .collect();
    paths.sort();
    for path in paths {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();
        if path.is_dir() {
            if matches!(
                name,
                "vendor" | "tests" | "examples" | "benches" | "target" | ".git"
            ) {
                continue;
            }
            collect_production_sources(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            if name == "tests.rs" || name.ends_with("_tests.rs") {
                continue;
            }
            out.push(path);
        }
    }
}

#[test]
fn rejects_direct_capabilities_literals_in_production_code() {
    let direct = r#"
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        capabilities: Capabilities {
            supports_guidance: true,
            ..Default::default()
        },
        ..other()
    }
}
"#;
    assert_eq!(
        find_production_capabilities_literals(direct),
        vec![Violation { line: 4 }]
    );

    let qualified_and_multiline = r#"
pub fn caps() -> gen_core::Capabilities {
    gen_core::Capabilities
    {
        max_count: 1,
        ..Default::default()
    }
}
"#;
    assert_eq!(
        find_production_capabilities_literals(qualified_and_multiline),
        vec![Violation { line: 3 }]
    );
}

#[test]
fn accepts_test_only_comments_route_records_and_other_capabilities_structs() {
    let clean = r#"
/// Example prose: `Capabilities { supports_guidance: true, ..Default::default() }`
pub fn route() -> RouteRecord {
    let _msg = "Capabilities { max_count: 1 }";
    let _lifecycle = MemoryLifecycleCapabilities { supports_staged: true };
    let _transform = TransformCapabilities { mac_only: false };
    let _caption = CaptionCapabilities { supports_prompt: true };
    let _audio_tx = AudioTransformCapabilities { sample_rate: 24_000 };
    let _transcribe = TranscribeCapabilities { languages: &["en"] };
    RouteRecord::default()
}

pub fn caps() -> Capabilities {
    route().compute_capabilities()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_caps() -> Capabilities {
        Capabilities {
            supports_guidance: true,
            ..Default::default()
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
fn cuda_test_caps() -> Capabilities {
    gen_core::Capabilities {
        max_count: 2,
        ..Default::default()
    }
}
"#;
    assert_eq!(find_production_capabilities_literals(clean), Vec::new());

    let whole_test_file = r#"
#![cfg(test)]
fn fixture() -> Capabilities {
    Capabilities { max_count: 1, ..Default::default() }
}
"#;
    assert_eq!(
        find_production_capabilities_literals(whole_test_file),
        Vec::new()
    );
}

#[test]
fn no_provider_source_under_media_or_audio_constructs_capabilities_directly() {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("crates/contracts/gen-core sits three levels below the workspace root");

    let mut files = Vec::new();
    for subdir in ["crates/media", "crates/audio"] {
        collect_production_sources(&workspace_root.join(subdir), &mut files);
    }
    assert!(
        files.len() > 100,
        "expected > 100 production source files under crates/media and crates/audio, found {}",
        files.len()
    );

    let mut violations = Vec::new();
    for path in &files {
        let src = std::fs::read_to_string(path)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
        for violation in find_production_capabilities_literals(&src) {
            let rel = path
                .strip_prefix(workspace_root)
                .unwrap_or(path)
                .display()
                .to_string();
            violations.push(format!("{rel}:{}", violation.line));
        }
    }

    assert!(
        violations.is_empty(),
        "production sources under crates/media/ or crates/audio/ must compute Capabilities via \
         RouteRecord::compute_capabilities() rather than constructing `Capabilities {{ .. }}` \
         directly (REG-2 / TOOL-4):\n  {}",
        violations.join("\n  ")
    );
}
