//! Persisted, backend-neutral model snapshot preparation (story 7659).
//!
//! Turning a downloaded model into something the engine can load has two parts: *deciding* what to
//! do (which format is this? does it need converting? quantize or not?) and *doing the tensor work*
//! (dequantize, re-quantize, serialize). This module owns the first part and **delegates the second
//! to the linked backend**, exactly as [`registry`](crate::registry) owns provider routing while the
//! backend owns the decode.
//!
//! A caller hands [`prepare_snapshot`] a [`PrepareSpec`] — an already-downloaded source (an
//! HF-safetensors snapshot directory or a `*.gguf` file; fetching it is the caller's job, this crate
//! has no network), an output directory, and one quantization knob — and gets back a persisted,
//! loadable snapshot that [`load_for_model`](crate::load_for_model) consumes. The backend that does
//! the work is selected from the link-time registry by a cheap, weightless probe, mirroring
//! [`can_load`](crate::registry::TextLlmRegistration::can_load).
//!
//! `core-llm` stays **tensor-free**: it detects the on-disk format, routes to a backend, validates,
//! and reports — it never touches a weight. The quantization invariant it documents (and the backend
//! enforces) is that only attention/MLP **projection** weights are quantized; embeddings, the LM
//! head, and norms stay dense. That is a tensor-level rule, so it is architecture-agnostic.

use crate::error::{Error, Result};
use crate::request::Quantize;
use std::path::{Path, PathBuf};

/// The on-disk format of a model source, detected from bytes and layout only — never the model
/// architecture (`core-llm` does not interpret architectures; the backend does).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelFormat {
    /// A llama.cpp `*.gguf` container (a single file beginning with the `GGUF` magic).
    Gguf,
    /// A Hugging Face-shaped snapshot: a directory with `config.json` (+ `*.safetensors` shards), or
    /// a bare `*.safetensors` file.
    Safetensors,
}

impl ModelFormat {
    /// A short lowercase tag for diagnostics.
    pub fn as_str(self) -> &'static str {
        match self {
            ModelFormat::Gguf => "gguf",
            ModelFormat::Safetensors => "safetensors",
        }
    }
}

/// The first four bytes of every `*.gguf` container.
const GGUF_MAGIC: &[u8; 4] = b"GGUF";

/// Detect the on-disk [`ModelFormat`] of `source` by inspecting bytes and layout only (tensor-free):
///
/// - a **file** is GGUF iff it begins with the `GGUF` magic (falling back to a `.gguf` extension when
///   the bytes can't be read), or a HF snapshot iff it is a `*.safetensors`;
/// - a **directory** is a HF snapshot iff it holds `config.json`, else GGUF iff it holds a `*.gguf`.
///
/// Returns [`Error::Unsupported`] for anything else (a missing path, an empty directory, a stray
/// file) — never a panic.
pub fn detect_format(source: &Path) -> Result<ModelFormat> {
    if source.is_file() {
        detect_file_format(source)
    } else if source.is_dir() {
        detect_dir_format(source)
    } else {
        Err(Error::Unsupported(format!(
            "cannot prepare '{}': path does not exist or is neither a file nor a directory",
            source.display()
        )))
    }
}

fn detect_file_format(path: &Path) -> Result<ModelFormat> {
    if file_has_gguf_magic(path) {
        return Ok(ModelFormat::Gguf);
    }
    match path.extension().and_then(|e| e.to_str()) {
        Some("gguf") => Ok(ModelFormat::Gguf),
        Some("safetensors") => Ok(ModelFormat::Safetensors),
        _ => Err(Error::Unsupported(format!(
            "cannot prepare '{}': not a GGUF (no `GGUF` magic) nor a `*.safetensors` file",
            path.display()
        ))),
    }
}

fn detect_dir_format(dir: &Path) -> Result<ModelFormat> {
    if dir.join("config.json").is_file() {
        return Ok(ModelFormat::Safetensors);
    }
    if dir_contains_gguf(dir) {
        return Ok(ModelFormat::Gguf);
    }
    Err(Error::Unsupported(format!(
        "cannot prepare '{}': directory has no config.json (HF snapshot) and no *.gguf",
        dir.display()
    )))
}

/// Read the first four bytes; `true` iff they are the `GGUF` magic. Any IO error ⇒ `false` (the
/// caller falls back to the file extension).
fn file_has_gguf_magic(path: &Path) -> bool {
    use std::io::Read;
    let mut buf = [0u8; 4];
    std::fs::File::open(path)
        .and_then(|mut f| f.read_exact(&mut buf))
        .map(|()| &buf == GGUF_MAGIC)
        .unwrap_or(false)
}

fn dir_contains_gguf(dir: &Path) -> bool {
    std::fs::read_dir(dir)
        .map(|rd| rd.flatten().any(|e| e.path().extension().and_then(|x| x.to_str()) == Some("gguf")))
        .unwrap_or(false)
}

/// A request to materialize a loadable, persisted snapshot from `source` into `out_dir`.
///
/// `source` is an already-downloaded model directory or file (fetching it is the caller's
/// responsibility — `core-llm` has no network). `out_dir` receives the prepared snapshot
/// (`config.json` + `model.safetensors` + tokenizer files) that
/// [`load_for_model`](crate::load_for_model) consumes. `quantize` is the single knob: `None` ⇒ a
/// dense snapshot, `Some(_)` ⇒ projections re-quantized to that scheme on the way out.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrepareSpec {
    /// The downloaded model: an HF-safetensors snapshot directory or a `*.gguf` file/dir.
    pub source: PathBuf,
    /// Where the prepared, loadable snapshot is written.
    pub out_dir: PathBuf,
    /// Optional load-time-independent quantization baked into the persisted snapshot.
    pub quantize: Option<Quantize>,
}

impl PrepareSpec {
    /// A dense (non-quantized) preparation.
    pub fn dense(source: impl Into<PathBuf>, out_dir: impl Into<PathBuf>) -> Self {
        Self { source: source.into(), out_dir: out_dir.into(), quantize: None }
    }

    /// A preparation that re-quantizes projections to `quantize`.
    pub fn quantized(
        source: impl Into<PathBuf>,
        out_dir: impl Into<PathBuf>,
        quantize: Quantize,
    ) -> Self {
        Self { source: source.into(), out_dir: out_dir.into(), quantize: Some(quantize) }
    }
}

/// What a [`prepare_snapshot`] call produced — reported honestly so a caller (or the conformance
/// suite) sees exactly what was done, including a no-rewrite passthrough.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrepareReport {
    /// The detected format of the input.
    pub input_format: ModelFormat,
    /// The quantization actually baked into the snapshot (`None` ⇒ dense).
    pub quantized: Option<Quantize>,
    /// The directory the loadable snapshot is at (equals `spec.source` on a passthrough).
    pub out_dir: PathBuf,
    /// Number of weight tensors in the snapshot.
    pub num_tensors: usize,
    /// `true` when the source was already a loadable dense snapshot and was returned as-is (the
    /// backend wrote nothing). Never silently true — surfaced so "prepared" never hides "did
    /// nothing".
    pub passthrough: bool,
}

/// A registered backend snapshot preparer: which sources it can handle (a cheap, weightless probe)
/// and how to materialize the snapshot (the tensor work). Backends register one with
/// [`inventory::submit!`], exactly as they register a
/// [`TextLlmRegistration`](crate::registry::TextLlmRegistration). Stored as `fn` pointers (not a
/// trait object) because preparation is a stateless one-shot — there is no instance to hold.
pub struct SnapshotPreparerRegistration {
    /// The backend tag for diagnostics (e.g. `"mlx"`, `"candle"`).
    pub backend: fn() -> &'static str,
    /// Cheap probe: can this backend prepare `spec.source`? Typically a [`detect_format`] call
    /// filtered to the formats the backend implements. MUST NOT read weight shards.
    pub can_prepare: fn(&PrepareSpec) -> bool,
    /// Materialize the persisted snapshot (dequantize / re-quantize / serialize). The tensor work.
    pub prepare: fn(&PrepareSpec) -> Result<PrepareReport>,
}

inventory::collect!(SnapshotPreparerRegistration);

/// Iterate every registered snapshot preparer (link-time collected).
pub fn snapshot_preparers() -> impl Iterator<Item = &'static SnapshotPreparerRegistration> {
    inventory::iter::<SnapshotPreparerRegistration>.into_iter()
}

/// Materialize a loadable, persisted snapshot per `spec`, selecting the linked backend that can
/// prepare the source — naming no backend. The selected backend does all tensor work; this function
/// only routes and reports.
///
/// Returns [`Error::Unsupported`] when no registered preparer accepts the source (naming the detected
/// format and the linked preparers), mirroring [`load_for_model`](crate::load_for_model)'s
/// no-provider path — never a panic, never a silent default.
///
/// ```ignore
/// // The app downloads the repo, then prepares a Q4 snapshot and loads it — backend-agnostic:
/// let report = core_llm::prepare_snapshot(
///     &core_llm::PrepareSpec::quantized("/dl/qwen3-0.6b", "/snap/qwen3-q4", core_llm::Quantize::Q4),
/// )?;
/// let llm = core_llm::load_for_model(&core_llm::LoadSpec::dense(report.out_dir.to_string_lossy()))?;
/// ```
pub fn prepare_snapshot(spec: &PrepareSpec) -> Result<PrepareReport> {
    let reg = select_preparer(snapshot_preparers(), spec)?;
    (reg.prepare)(spec)
}

/// Resolve the preparer to run: the first registered backend whose weightless probe accepts the
/// source. Pure over the supplied registrations so it is unit-testable without the global inventory.
fn select_preparer<'a>(
    regs: impl Iterator<Item = &'a SnapshotPreparerRegistration>,
    spec: &PrepareSpec,
) -> Result<&'a SnapshotPreparerRegistration> {
    let all: Vec<&SnapshotPreparerRegistration> = regs.collect();
    match all.iter().copied().find(|r| (r.can_prepare)(spec)) {
        Some(r) => Ok(r),
        None => Err(Error::Unsupported(no_preparer_msg(spec, &all))),
    }
}

/// `backend` summary of a set of registrations, for diagnostics.
fn preparer_summary(regs: &[&SnapshotPreparerRegistration]) -> String {
    if regs.is_empty() {
        return "(none)".to_string();
    }
    let mut v: Vec<String> = regs.iter().map(|r| (r.backend)().to_string()).collect();
    v.sort();
    v.dedup();
    v.join(", ")
}

fn no_preparer_msg(spec: &PrepareSpec, all: &[&SnapshotPreparerRegistration]) -> String {
    let fmt = match detect_format(&spec.source) {
        Ok(f) => format!("format={}", f.as_str()),
        Err(_) => "unrecognized source format".to_string(),
    };
    format!(
        "no linked backend can prepare a snapshot from '{}' ({fmt}); linked preparers: {}",
        spec.source.display(),
        preparer_summary(all),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    /// A fresh, unique temp directory for a test (no `tempfile` dep — mirrors the testkit's
    /// process-id scheme).
    fn tmp(tag: &str) -> PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("core-llm-prepare-{}-{tag}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write(path: &Path, bytes: &[u8]) {
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn detects_gguf_by_magic_regardless_of_extension() {
        let dir = tmp("magic");
        let f = dir.join("model.bin");
        write(&f, b"GGUF\x00\x00\x00\x03rest");
        assert_eq!(detect_format(&f).unwrap(), ModelFormat::Gguf);
    }

    #[test]
    fn detects_gguf_by_extension_when_unreadable_magic() {
        let dir = tmp("ext");
        let f = dir.join("model.gguf");
        write(&f, b"xx"); // too short for magic, but the extension carries it
        assert_eq!(detect_format(&f).unwrap(), ModelFormat::Gguf);
    }

    #[test]
    fn detects_safetensors_file() {
        let dir = tmp("st-file");
        let f = dir.join("model.safetensors");
        write(&f, b"\x00\x00\x00\x00not-really-but-extension-counts");
        assert_eq!(detect_format(&f).unwrap(), ModelFormat::Safetensors);
    }

    #[test]
    fn detects_hf_snapshot_dir_by_config_json() {
        let dir = tmp("hf-dir");
        write(&dir.join("config.json"), b"{\"model_type\":\"llama\"}");
        write(&dir.join("model.safetensors"), b"\x00");
        assert_eq!(detect_format(&dir).unwrap(), ModelFormat::Safetensors);
    }

    #[test]
    fn detects_gguf_dir_by_contained_file() {
        let dir = tmp("gguf-dir");
        write(&dir.join("weights.gguf"), b"GGUF\x00\x00\x00\x03");
        assert_eq!(detect_format(&dir).unwrap(), ModelFormat::Gguf);
    }

    #[test]
    fn empty_dir_is_unsupported() {
        let dir = tmp("empty");
        match detect_format(&dir) {
            Err(Error::Unsupported(m)) => assert!(m.contains("no config.json"), "{m}"),
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn missing_path_is_unsupported() {
        let p = std::env::temp_dir().join("core-llm-prepare-definitely-not-here-zzz");
        match detect_format(&p) {
            Err(Error::Unsupported(m)) => assert!(m.contains("does not exist"), "{m}"),
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn stray_file_is_unsupported() {
        let dir = tmp("stray");
        let f = dir.join("notes.txt");
        write(&f, b"hello");
        match detect_format(&f) {
            Err(Error::Unsupported(_)) => {}
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    // --- select_preparer over throwaway registrations (no global inventory, no tensor work) ---

    fn mlx_tag() -> &'static str {
        "mlx"
    }
    fn candle_tag() -> &'static str {
        "candle"
    }
    fn yes(_spec: &PrepareSpec) -> bool {
        true
    }
    fn no(_spec: &PrepareSpec) -> bool {
        false
    }
    fn never_prepares(_spec: &PrepareSpec) -> Result<PrepareReport> {
        Ok(PrepareReport {
            input_format: ModelFormat::Safetensors,
            quantized: None,
            out_dir: PathBuf::from("/unused"),
            num_tensors: 1,
            passthrough: false,
        })
    }
    fn reg(
        backend: fn() -> &'static str,
        can_prepare: fn(&PrepareSpec) -> bool,
    ) -> SnapshotPreparerRegistration {
        SnapshotPreparerRegistration { backend, can_prepare, prepare: never_prepares }
    }

    #[test]
    fn select_picks_the_accepting_backend() {
        let mlx = reg(mlx_tag, no);
        let candle = reg(candle_tag, yes);
        let spec = PrepareSpec::dense("/some/model", "/out");
        let chosen = select_preparer([&mlx, &candle].into_iter(), &spec).unwrap();
        assert_eq!((chosen.backend)(), "candle");
    }

    #[test]
    fn no_accepting_backend_is_a_typed_unsupported() {
        let mlx = reg(mlx_tag, no);
        let candle = reg(candle_tag, no);
        let spec = PrepareSpec::dense("/some/model", "/out");
        // Map the Ok side to a printable value (the registration itself isn't `Debug`, matching
        // `TextLlmRegistration`).
        match select_preparer([&mlx, &candle].into_iter(), &spec).map(|r| (r.backend)()) {
            Err(Error::Unsupported(m)) => {
                assert!(m.contains("no linked backend can prepare"), "{m}");
                assert!(m.contains("mlx") && m.contains("candle"), "{m}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn spec_constructors_carry_quantize() {
        assert_eq!(PrepareSpec::dense("a", "b").quantize, None);
        assert_eq!(
            PrepareSpec::quantized("a", "b", Quantize::Q4).quantize,
            Some(Quantize::Q4)
        );
    }
}
