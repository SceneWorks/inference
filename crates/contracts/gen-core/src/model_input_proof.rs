//! Opt-in, source-bound observations of inputs used by successful existing readers.
//!
//! This module never opens, stats, scans, hashes or evaluates an input. Content
//! digests must come from an existing validated receipt or the exact bytes an
//! existing reader already owns. A path binding explicitly attests NO content
//! integrity. Callers record only after their original whole operation succeeds.

use std::collections::HashSet;
use std::fmt::Write as _;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

#[derive(Clone, Copy, Debug)]
pub enum InputRole {
    ModelWeights,
    Configuration,
    Tokenizer,
    Reference,
    TensorInput,
    ArtifactInput,
}

impl InputRole {
    fn name(self) -> &'static str {
        match self {
            Self::ModelWeights => "model-weights",
            Self::Configuration => "configuration",
            Self::Tokenizer => "tokenizer",
            Self::Reference => "reference",
            Self::TensorInput => "tensor-input",
            Self::ArtifactInput => "artifact-input",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Provenance<'a> {
    ValidatedContent {
        sha256: [u8; 32],
        verifier: &'a str,
        source_binding: &'a str,
    },
    SameReadBuffer {
        sha256: [u8; 32],
    },
    SuccessfulReaderBinding,
}

/// A diagnostic tag from an existing, actually selected source branch only.
/// Omission means unknown; it must never be inferred from a path or env presence.
#[derive(Clone, Copy, Debug)]
pub enum SourceOrigin {
    SourceDefinedDefault,
    ExternalSelected,
}

impl SourceOrigin {
    fn name(self) -> &'static str {
        match self {
            Self::SourceDefinedDefault => "source-defined-default",
            Self::ExternalSelected => "external-selected",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct InputIdentity<'a> {
    pub path: Option<&'a Path>,
    pub provenance: Provenance<'a>,
    pub source_origin: Option<SourceOrigin>,
}

impl<'a> InputIdentity<'a> {
    pub fn reader_path(path: &'a Path) -> Self {
        Self {
            path: Some(path),
            provenance: Provenance::SuccessfulReaderBinding,
            source_origin: None,
        }
    }

    pub fn same_read_buffer(path: Option<&'a Path>, sha256: [u8; 32]) -> Self {
        Self {
            path,
            provenance: Provenance::SameReadBuffer { sha256 },
            source_origin: None,
        }
    }

    pub fn validated(
        path: &'a Path,
        sha256: [u8; 32],
        verifier: &'a str,
        source_binding: &'a str,
    ) -> Self {
        Self {
            path: Some(path),
            provenance: Provenance::ValidatedContent {
                sha256,
                verifier,
                source_binding,
            },
            source_origin: None,
        }
    }

    pub fn with_source_origin(mut self, source_origin: SourceOrigin) -> Self {
        self.source_origin = Some(source_origin);
        self
    }
}

fn quoted(output: &mut String, value: &str) {
    output.push('"');
    for ch in value.chars() {
        match ch {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            ch if ch <= '\u{1f}' => {
                write!(output, "\\u{:04x}", ch as u32).expect("String formatting");
            }
            ch => output.push(ch),
        }
    }
    output.push('"');
}

fn hex(output: &mut String, bytes: &[u8]) {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    for byte in bytes {
        output.push(DIGITS[(byte >> 4) as usize] as char);
        output.push(DIGITS[(byte & 15) as usize] as char);
    }
}

fn path_identity(output: &mut String, path: &Path) {
    output.push_str("{\"display\":");
    quoted(output, &path.to_string_lossy());
    output.push_str(",\"os_encoded_hex\":\"");
    hex(output, path.as_os_str().as_encoded_bytes());
    output.push_str("\"}");
}

fn envelope(reader: &str, role: InputRole, items: &[InputIdentity<'_>]) -> Option<String> {
    if reader.is_empty() || items.is_empty() {
        return None;
    }
    let mut output = String::from("{\"schema_version\":1,\"reader\":");
    quoted(&mut output, reader);
    output.push_str(",\"role\":");
    quoted(&mut output, role.name());
    output.push_str(",\"complete\":true,\"items\":[");
    for (index, item) in items.iter().enumerate() {
        if index != 0 {
            output.push(',');
        }
        output.push('{');
        if let Some(path) = item.path {
            // Absolute LEXICAL binding only. Never canonicalize or retarget an
            // extension-sensitive/guarded loader path, and never inspect payload.
            let absolute = std::path::absolute(path).ok()?;
            output.push_str("\"literal_path\":");
            path_identity(&mut output, path);
            output.push_str(",\"absolute_lexical_path\":");
            path_identity(&mut output, &absolute);
            output.push_str(",\"os_encoding\":\"rust-osstr-encoded\",");
            if let Some(source_origin) = item.source_origin {
                output.push_str("\"source_origin\":");
                quoted(&mut output, source_origin.name());
                output.push(',');
            }
        } else {
            // A content-addressed buffer is real input without a made-up path.
            if item.source_origin.is_some()
                || !matches!(item.provenance, Provenance::SameReadBuffer { .. })
            {
                return None;
            }
            output.push_str("\"origin\":\"memory-buffer\",");
        }
        output.push_str("\"provenance\":{");
        match item.provenance {
            Provenance::SuccessfulReaderBinding => {
                output.push_str(
                    "\"kind\":\"successful-reader-binding\",\"content_integrity\":\"not_attested\"",
                );
            }
            Provenance::SameReadBuffer { sha256 } => {
                output.push_str("\"kind\":\"same-read-buffer\",\"content_integrity\":\"same-buffer-sha256\",\"sha256\":\"");
                hex(&mut output, &sha256);
                output.push('"');
            }
            Provenance::ValidatedContent {
                sha256,
                verifier,
                source_binding,
            } => {
                if verifier.is_empty() || source_binding.is_empty() {
                    return None;
                }
                output.push_str("\"kind\":\"validated-content\",\"content_integrity\":\"validated-existing-receipt\",\"sha256\":\"");
                hex(&mut output, &sha256);
                output.push_str("\",\"verifier\":");
                quoted(&mut output, verifier);
                output.push_str(",\"source_binding\":");
                quoted(&mut output, source_binding);
            }
        }
        output.push_str("}}");
    }
    output.push_str("]}");
    Some(output)
}

static OBSERVED: OnceLock<Mutex<HashSet<(String, String)>>> = OnceLock::new();

/// Observe the exact WHOLE input collection of an already-successful operation.
///
/// No filesystem input or model operation is performed here. Duplicate records
/// are suppressed per actual protocol scope, never across different child IDs.
pub fn record_inputs(reader: &str, role: InputRole, items: &[InputIdentity<'_>]) {
    let Ok(scope) = std::env::var("MODEL_INFERENCE_CASE_FQN") else {
        return;
    };
    if scope.is_empty() || scope.chars().any(char::is_whitespace) {
        return;
    }
    let Some(payload) = envelope(reader, role, items) else {
        return;
    };
    let inserted = OBSERVED
        .get_or_init(|| Mutex::new(HashSet::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert((scope.clone(), payload.clone()));
    if inserted {
        println!("\nMODEL_INFERENCE_INPUT_USED {scope} {payload}");
    }
}

/// Decode an existing hexadecimal digest without hashing or reading anything.
/// An invalid diagnostic digest suppresses that receipt; it never changes the
/// original reader's result or introduces a new validation gate.
pub fn sha256_from_hex(value: &str) -> Option<[u8; 32]> {
    if value.len() != 64 {
        return None;
    }
    let mut digest = [0u8; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let high = char::from(pair[0]).to_digit(16)?;
        let low = char::from(pair[1]).to_digit(16)?;
        digest[index] = (high * 16 + low) as u8;
    }
    Some(digest)
}

/// Invoke the original path reader once and bind only its successful result.
///
/// The original path expression and its `AsRef` conversion each run once. This
/// adapter does not open or read anything itself, inspect payloads, retain a
/// buffer, evaluate a tensor, or turn a reader binding into byte integrity.
pub fn observe_path_reader<P: AsRef<Path>, T, E>(
    reader: &str,
    role: InputRole,
    path: P,
    read: impl FnOnce(&Path) -> Result<T, E>,
) -> Result<T, E> {
    let path = path.as_ref();
    read(path).inspect(|_| record_reader_paths(reader, role, &[path]))
}

/// Observe a successful read only when the source's existing consumption guard
/// accepts its value. The predicate affects diagnostics only, never the result.
pub fn observe_path_reader_if<P: AsRef<Path>, T, E>(
    reader: &str,
    role: InputRole,
    path: P,
    read: impl FnOnce(&Path) -> Result<T, E>,
    record_if: impl FnOnce(&T) -> bool,
) -> Result<T, E> {
    let path = path.as_ref();
    read(path).inspect(|value| {
        if record_if(value) {
            record_reader_paths(reader, role, &[path]);
        }
    })
}

/// Bind an existing successful reader to the origin selected by its original
/// branch. The tag is supplied by source code, never discovered from the path.
pub fn observe_path_reader_tagged<P: AsRef<Path>, T, E>(
    reader: &str,
    role: InputRole,
    path: P,
    read: impl FnOnce(&Path) -> Result<T, E>,
    source_origin: SourceOrigin,
    record_if: impl FnOnce(&T) -> bool,
) -> Result<T, E> {
    let path = path.as_ref();
    read(path).inspect(|value| {
        if record_if(value) {
            record_inputs(
                reader,
                role,
                &[InputIdentity::reader_path(path).with_source_origin(source_origin)],
            );
        }
    })
}

/// Bind a source-audited helper that returns a decoded value or panics on read
/// failure. It runs exactly once; the original value is moved back unchanged.
pub fn observe_path_value_tagged<P: AsRef<Path>, T>(
    reader: &str,
    role: InputRole,
    path: P,
    read: impl FnOnce(&Path) -> T,
    source_origin: SourceOrigin,
) -> T {
    let path = path.as_ref();
    let value = read(path);
    record_inputs(
        reader,
        role,
        &[InputIdentity::reader_path(path).with_source_origin(source_origin)],
    );
    value
}

/// Observe exact actual paths without inventing a content-integrity claim.
pub fn record_reader_paths<P: AsRef<Path>>(reader: &str, role: InputRole, paths: &[P]) {
    // Avoid allocating diagnostics when the manual proof scope is absent.
    if std::env::var_os("MODEL_INFERENCE_CASE_FQN").is_none() {
        return;
    }
    let identities: Vec<_> = paths
        .iter()
        .map(|path| InputIdentity::reader_path(path.as_ref()))
        .collect();
    record_inputs(reader, role, &identities);
}

/// Observe one actual UTF-8 configuration-file read. This is a raw successful-reader
/// binding only: the caller retains every original parse, validation and fallback.
pub fn read_configuration_to_string(path: impl AsRef<Path>) -> std::io::Result<String> {
    read_configuration_string_impl(path, None)
}

/// Same physical configuration-text read, with a source-fixed purpose selector.
/// Exactly one base-or-qualified identity is emitted, never both for one read.
pub fn read_configuration_to_string_for(
    path: impl AsRef<Path>,
    selector: &str,
) -> std::io::Result<String> {
    read_configuration_string_impl(path, Some(selector))
}

fn read_configuration_string_impl(
    path: impl AsRef<Path>,
    selector: Option<&str>,
) -> std::io::Result<String> {
    let qualified = selector
        .map(|value| format!("gen-core::model_input_proof::read_configuration_to_string#{value}"));
    let reader = qualified
        .as_deref()
        .unwrap_or("gen-core::model_input_proof::read_configuration_to_string");
    observe_path_reader(reader, InputRole::Configuration, path, |actual_path| {
        std::fs::read_to_string(actual_path)
    })
}

/// Observe one actual configuration/index byte read, without UTF-8 conversion.
/// Raw successful-reader binding only; caller parsing and validation stay unchanged.
pub fn read_configuration_bytes(path: impl AsRef<Path>) -> std::io::Result<Vec<u8>> {
    observe_path_reader(
        "gen-core::model_input_proof::read_configuration_bytes",
        InputRole::Configuration,
        path,
        |actual_path| std::fs::read(actual_path),
    )
}

std::thread_local! {
    // Source-owned, proof-only purpose. Scope selection emits nothing itself.
    static MODEL_WEIGHT_PURPOSE: std::cell::Cell<Option<&'static str>> = const {
        std::cell::Cell::new(None)
    };
}

/// Qualify only successful physical weight-reader receipts in this tight source scope.
/// Restores the previous purpose on success, error, early return, panic, and nesting.
/// This never reads paths, queries controls, or emits a receipt at scope entry.
pub fn with_model_weight_purpose<T>(purpose: &'static str, read: impl FnOnce() -> T) -> T {
    struct Restore(Option<&'static str>);
    impl Drop for Restore {
        fn drop(&mut self) {
            MODEL_WEIGHT_PURPOSE.with(|slot| slot.set(self.0));
        }
    }
    let previous = MODEL_WEIGHT_PURPOSE.with(|slot| slot.replace(Some(purpose)));
    let _restore = Restore(previous);
    read()
}

/// Emit exactly the existing base identity OR its source-selected purpose identity.
/// Called only at an existing physical reader's original successful boundary.
pub fn record_model_weight_paths<P: AsRef<Path>>(reader: &str, paths: &[P]) {
    MODEL_WEIGHT_PURPOSE.with(|slot| {
        if let Some(purpose) = slot.get() {
            let qualified = format!("{reader}#{purpose}");
            record_reader_paths(&qualified, InputRole::ModelWeights, paths);
        } else {
            record_reader_paths(reader, InputRole::ModelWeights, paths);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whole_ordered_collection_has_lossless_paths_and_no_fake_content_digest() {
        let first = Path::new("actual/weight-02.safetensors");
        let second = Path::new("actual/weight-01.safetensors");
        let payload = envelope(
            "mock::successful_reader",
            InputRole::ModelWeights,
            &[
                InputIdentity::reader_path(first),
                InputIdentity::reader_path(second),
            ],
        )
        .unwrap();
        assert!(payload.contains("\"complete\":true"));
        assert!(payload.find("weight-02").unwrap() < payload.find("weight-01").unwrap());
        assert!(payload.contains("\"os_encoding\":\"rust-osstr-encoded\""));
        assert!(payload.contains("\"content_integrity\":\"not_attested\""));
        assert!(!payload.contains("sha256"));
        assert!(!payload.contains("mtime"));
        assert!(!payload.contains("canonical"));
    }

    #[test]
    fn same_buffer_digest_does_not_invent_a_filesystem_path() {
        let payload = envelope(
            "mock::owned_buffer_reader",
            InputRole::TensorInput,
            &[InputIdentity::same_read_buffer(None, [0x5a; 32])],
        )
        .unwrap();
        assert!(payload.contains("\"origin\":\"memory-buffer\""));
        assert!(payload.contains(&"5a".repeat(32)));
        assert!(!payload.contains("literal_path"));
    }

    #[test]
    fn existing_validated_receipt_keeps_original_assurance_description() {
        let payload = envelope(
            "mock::retained_pin_reader",
            InputRole::ArtifactInput,
            &[InputIdentity::validated(
                Path::new("actual/config-and-weights"),
                [0x31; 32],
                "existing-verifier",
                "pre/post retained path seal; not an opened-handle or ABA guarantee",
            )],
        )
        .unwrap();
        assert!(payload.contains("validated-existing-receipt"));
        assert!(payload.contains("not an opened-handle or ABA guarantee"));
        assert!(payload.contains("\"role\":\"artifact-input\""));
    }

    #[test]
    fn no_empty_external_collection_or_unbound_weak_buffer_is_a_receipt() {
        assert!(envelope("mock", InputRole::ModelWeights, &[]).is_none());
        assert!(envelope(
            "mock",
            InputRole::TensorInput,
            &[InputIdentity {
                path: None,
                provenance: Provenance::SuccessfulReaderBinding,
                source_origin: None,
            }],
        )
        .is_none());
    }

    #[test]
    fn json_strings_escape_control_characters_without_losing_unicode() {
        let mut output = String::new();
        quoted(&mut output, "quote\" slash\\ newline\n π\u{0001}");
        assert_eq!(output, "\"quote\\\" slash\\\\ newline\\n π\\u0001\"");
    }

    #[cfg(unix)]
    #[test]
    fn non_utf8_literal_paths_are_distinguished_losslessly() {
        use std::os::unix::ffi::OsStrExt;
        let a = Path::new(std::ffi::OsStr::from_bytes(b"actual/\xff"));
        let b = Path::new(std::ffi::OsStr::from_bytes(b"actual/\xfe"));
        let a = envelope(
            "mock",
            InputRole::Reference,
            &[InputIdentity::reader_path(a)],
        )
        .unwrap();
        let b = envelope(
            "mock",
            InputRole::Reference,
            &[InputIdentity::reader_path(b)],
        )
        .unwrap();
        assert_ne!(a, b);
        assert!(a.contains("ff"));
        assert!(b.contains("fe"));
    }
}
