//! Cached **acoustic latents** — the artifact the YuE2 VAE decodes (sc-22993).
//!
//! Upstream's acoustic synthesis returns CPU FP32 latents of shape `[frames, 64]`
//! (`yue2.nar.synthesize`), `YuE2Pipeline.decode` consumes them, and `save_artifacts` persists them
//! as `latent.npy`. Decoding the *same* latents with a different decoder is how upstream compares the
//! standard and legacy VAEs ("decode the same cached latents instead of generating a new song",
//! `docs/generation.md`), and its decode helper refuses to proceed if the latents changed.
//!
//! [`AcousticLatents`] is that artifact with an identity: the SHA-256 of its canonical bytes
//! (row-major `[frames, 64]` little-endian `f32`), its shape, its dtype and its [`LatentSource`].
//! The values are private and the identity is computed at construction, so a value always matches
//! its identity; [`AcousticLatents::verify`] re-derives the hash at the decode boundary, and
//! [`AcousticLatents::load`] / [`AcousticLatents::from_npy_verified`] refuse bytes that disagree
//! with a recorded identity. Nothing about planning, semantic generation or synthesis is needed to
//! decode — which is what lets a decoder switch reuse the latents without re-running those stages.
//!
//! Persistence is upstream-compatible: [`AcousticLatents::to_npy`] writes exactly the bytes
//! `numpy.save` writes for a C-order `float32` `[frames, 64]` array (checked against numpy in the
//! tests), so a `latent.npy` from either side hashes identically.

use std::fs;
use std::path::Path;

use candle_audio::candle_core::{DType, Device, Tensor};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

/// Latent channels per frame (the VAE's `latent_dim`).
pub const LATENT_CHANNELS: usize = 64;
/// The only latent dtype (upstream converts to FP32 before saving and decoding).
pub const LATENT_DTYPE: &str = "F32";
/// The artifact's latent file name, as upstream's `save_artifacts` names it.
pub const LATENT_FILE: &str = "latent.npy";
/// The identity sidecar written beside [`LATENT_FILE`].
pub const IDENTITY_FILE: &str = "latent.json";
/// The identity sidecar schema.
pub const IDENTITY_SCHEMA: u64 = 1;

/// Everything that can be wrong with a latent artifact.
#[derive(Debug, thiserror::Error)]
pub enum LatentError {
    /// Shape, length, dtype or value content is not a valid `[frames, 64]` FP32 latent.
    #[error("invalid acoustic latents: {0}")]
    Invalid(String),
    /// The bytes do not hash to the recorded identity (changed, corrupt or a different latent).
    #[error("acoustic latents changed: identity {expected}, content hashes to {actual}")]
    IdentityMismatch {
        /// The recorded SHA-256.
        expected: String,
        /// The SHA-256 of the content found.
        actual: String,
    },
    /// The content hashes correctly but the recorded shape/source metadata disagrees.
    #[error("acoustic latent metadata mismatch: {0}")]
    MetadataMismatch(String),
    /// A `.npy` file is malformed or not a C-order little-endian `float32` `[frames, 64]` array.
    #[error("malformed latent .npy: {0}")]
    Npy(String),
    /// Reading or writing the artifact failed.
    #[error("latent artifact I/O on {path}: {source}")]
    Io {
        /// The path.
        path: String,
        /// The I/O error.
        source: std::io::Error,
    },
    /// A tensor operation failed.
    #[error(transparent)]
    Candle(#[from] candle_audio::candle_core::Error),
}

/// Where a latent came from. Part of the identity record, so two latents with equal values but
/// different provenance are still distinguishable in artifacts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LatentSource {
    /// Produced by acoustic synthesis (the NAR). `stage_identity` is the identity of the exact
    /// stage inputs that produced it (assigned by the stage cache).
    Synthesis {
        /// Identity of the synthesis stage that produced the latents.
        stage_identity: String,
    },
    /// Produced by a VAE encoder from audio.
    Encoded {
        /// The encoder's component key (`yue2_vae` / `yue2_vae_legacy`).
        vae: String,
        /// SHA-256 of the input audio's `[channels, samples]` little-endian `f32` bytes.
        audio_sha256: String,
        /// `false` for the posterior mean, `true` for a sample drawn with injected noise.
        sampled: bool,
    },
    /// Imported from an external `latent.npy` (for example an upstream `save_artifacts` run).
    Imported {
        /// SHA-256 of the imported file's bytes.
        file_sha256: String,
    },
}

impl LatentSource {
    fn to_json(&self) -> Value {
        match self {
            LatentSource::Synthesis { stage_identity } => {
                json!({"kind": "synthesis", "stage_identity": stage_identity})
            }
            LatentSource::Encoded {
                vae,
                audio_sha256,
                sampled,
            } => json!({"kind": "encoded", "vae": vae, "audio_sha256": audio_sha256,
                        "sampled": sampled}),
            LatentSource::Imported { file_sha256 } => {
                json!({"kind": "imported", "file_sha256": file_sha256})
            }
        }
    }

    fn from_json(v: &Value) -> Result<Self, LatentError> {
        let bad = |what: &str| LatentError::MetadataMismatch(format!("source: {what}"));
        let s = |k: &str| {
            v.get(k)
                .and_then(Value::as_str)
                .map(str::to_string)
                .ok_or_else(|| bad(&format!("missing string `{k}`")))
        };
        match v.get("kind").and_then(Value::as_str) {
            Some("synthesis") => Ok(LatentSource::Synthesis {
                stage_identity: s("stage_identity")?,
            }),
            Some("encoded") => Ok(LatentSource::Encoded {
                vae: s("vae")?,
                audio_sha256: s("audio_sha256")?,
                sampled: v
                    .get("sampled")
                    .and_then(Value::as_bool)
                    .ok_or_else(|| bad("missing bool `sampled`"))?,
            }),
            Some("imported") => Ok(LatentSource::Imported {
                file_sha256: s("file_sha256")?,
            }),
            other => Err(bad(&format!("unknown kind {other:?}"))),
        }
    }
}

/// The identity of a latent artifact: content hash, shape, dtype and provenance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LatentIdentity {
    /// Lower-case hex SHA-256 of the row-major `[frames, 64]` little-endian `f32` bytes.
    pub sha256: String,
    /// Latent frames (`shape[0]`; `shape[1]` is always [`LATENT_CHANNELS`]).
    pub frames: usize,
    /// Provenance.
    pub source: LatentSource,
}

impl LatentIdentity {
    /// `[frames, 64]`.
    pub fn shape(&self) -> [usize; 2] {
        [self.frames, LATENT_CHANNELS]
    }

    /// Always [`LATENT_DTYPE`].
    pub fn dtype(&self) -> &'static str {
        LATENT_DTYPE
    }

    /// The identity record as JSON (the `latent.json` sidecar body).
    pub fn to_json(&self) -> Value {
        json!({
            "schema": IDENTITY_SCHEMA,
            "sha256": self.sha256,
            "shape": self.shape(),
            "dtype": LATENT_DTYPE,
            "source": self.source.to_json(),
        })
    }

    /// Parse an identity record. Unknown schema, dtype or a non-`[frames, 64]` shape is an error.
    pub fn from_json(v: &Value) -> Result<Self, LatentError> {
        let bad = |what: String| LatentError::MetadataMismatch(what);
        if v.get("schema").and_then(Value::as_u64) != Some(IDENTITY_SCHEMA) {
            return Err(bad(format!("unsupported schema {:?}", v.get("schema"))));
        }
        if v.get("dtype").and_then(Value::as_str) != Some(LATENT_DTYPE) {
            return Err(bad(format!("dtype {:?}, expected F32", v.get("dtype"))));
        }
        let shape: Vec<u64> = v
            .get("shape")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_u64).collect())
            .unwrap_or_default();
        let frames = match shape.as_slice() {
            [f, c] if *c == LATENT_CHANNELS as u64 && *f > 0 => *f as usize,
            _ => {
                return Err(bad(format!(
                    "shape {:?}, expected [frames, 64]",
                    v.get("shape")
                )))
            }
        };
        let sha256 = v
            .get("sha256")
            .and_then(Value::as_str)
            .filter(|s| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()))
            .ok_or_else(|| bad("missing or malformed sha256".into()))?
            .to_string();
        let source = LatentSource::from_json(
            v.get("source")
                .ok_or_else(|| bad("missing source".into()))?,
        )?;
        Ok(Self {
            sha256,
            frames,
            source,
        })
    }
}

/// FP32 acoustic latents `[frames, 64]` with their identity. See the [module docs](self).
#[derive(Clone, Debug, PartialEq)]
pub struct AcousticLatents {
    values: Vec<f32>,
    identity: LatentIdentity,
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// SHA-256 of `values` as little-endian `f32` bytes (endianness-independent).
pub(crate) fn sha256_f32(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for chunk in values.chunks(16 * 1024) {
        let bytes: Vec<u8> = chunk.iter().flat_map(|v| v.to_le_bytes()).collect();
        hasher.update(&bytes);
    }
    hex(&hasher.finalize())
}

fn sha256_bytes(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

impl AcousticLatents {
    /// Wrap row-major `[frames, 64]` values. Empty, ragged or non-finite latents are refused
    /// (upstream `YuE2VAE._latent` refuses the same before decoding).
    pub fn new(values: Vec<f32>, frames: usize, source: LatentSource) -> Result<Self, LatentError> {
        if frames == 0 {
            return Err(LatentError::Invalid("zero frames".into()));
        }
        if values.len() != frames * LATENT_CHANNELS {
            return Err(LatentError::Invalid(format!(
                "{} values for {frames} frames of {LATENT_CHANNELS} channels",
                values.len()
            )));
        }
        if let Some(i) = values.iter().position(|v| !v.is_finite()) {
            return Err(LatentError::Invalid(format!(
                "non-finite value at frame {}, channel {}",
                i / LATENT_CHANNELS,
                i % LATENT_CHANNELS
            )));
        }
        let identity = LatentIdentity {
            sha256: sha256_f32(&values),
            frames,
            source,
        };
        Ok(Self { values, identity })
    }

    /// Latents from a tensor laid out `[frames, 64]` or `[1, 64, frames]` (the two layouts
    /// `YuE2Pipeline.decode` accepts). The values are converted to FP32 on the CPU, as upstream's
    /// `synthesize` does (`.float().cpu()`) before latents are cached.
    pub fn from_tensor(t: &Tensor, source: LatentSource) -> Result<Self, LatentError> {
        let t = t.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let rows = match t.dims() {
            [_, c] if *c == LATENT_CHANNELS => t,
            [1, c, _] if *c == LATENT_CHANNELS => t.squeeze(0)?.t()?,
            dims => {
                return Err(LatentError::Invalid(format!(
                    "tensor shape {dims:?}, expected [frames, 64] or [1, 64, frames]"
                )))
            }
        };
        let frames = rows.dim(0)?;
        let values = rows.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
        Self::new(values, frames, source)
    }

    /// The identity.
    pub fn identity(&self) -> &LatentIdentity {
        &self.identity
    }

    /// Latent frames.
    pub fn frames(&self) -> usize {
        self.identity.frames
    }

    /// The row-major `[frames, 64]` values.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Re-derive the content hash and compare it with the identity. Called at the decode
    /// boundary; a mismatch means the values changed after the identity was recorded.
    pub fn verify(&self) -> Result<(), LatentError> {
        let actual = sha256_f32(&self.values);
        if actual != self.identity.sha256 || self.values.len() != self.frames() * LATENT_CHANNELS {
            return Err(LatentError::IdentityMismatch {
                expected: self.identity.sha256.clone(),
                actual,
            });
        }
        Ok(())
    }

    /// [`Self::verify`], and require the identity to equal `expected` (hash, shape and source).
    pub fn verify_against(&self, expected: &LatentIdentity) -> Result<(), LatentError> {
        self.verify()?;
        if self.identity.sha256 != expected.sha256 {
            return Err(LatentError::IdentityMismatch {
                expected: expected.sha256.clone(),
                actual: self.identity.sha256.clone(),
            });
        }
        if self.identity != *expected {
            return Err(LatentError::MetadataMismatch(format!(
                "recorded {:?}, found {:?}",
                expected, self.identity
            )));
        }
        Ok(())
    }

    /// Mutable access that bypasses the identity, so tests can prove tampering is caught.
    #[cfg(test)]
    pub(crate) fn values_mut_for_test(&mut self) -> &mut [f32] {
        &mut self.values
    }

    /// The latents as the decoder's `[1, 64, frames]` FP32 input on `device`.
    pub fn to_decoder_input(&self, device: &Device) -> Result<Tensor, LatentError> {
        Ok(
            Tensor::from_slice(&self.values, (self.frames(), LATENT_CHANNELS), device)?
                .t()?
                .contiguous()?
                .unsqueeze(0)?,
        )
    }

    /// The exact bytes `numpy.save` writes for this `[frames, 64]` C-order `float32` array
    /// (format 1.0, numpy's header growth padding and 64-byte alignment).
    pub fn to_npy(&self) -> Vec<u8> {
        let mut header = format!(
            "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}), }}",
            self.frames(),
            LATENT_CHANNELS
        );
        // numpy leaves room to grow the leading axis in place: GROWTH_AXIS_MAX_DIGITS (21) minus
        // the digits of shape[0].
        let digits = self.frames().to_string().len();
        header.push_str(&" ".repeat(21usize.saturating_sub(digits)));
        // `_wrap_header`: pad so magic (8) + u16 length (2) + header + '\n' is a multiple of 64;
        // numpy's pad is 64 - (len % 64), i.e. a full 64 when already aligned.
        let hlen = header.len() + 1;
        let pad = 64 - ((10 + hlen) % 64);
        let total = hlen + pad;
        let mut out = Vec::with_capacity(10 + total + self.values.len() * 4);
        out.extend_from_slice(b"\x93NUMPY\x01\x00");
        out.extend_from_slice(&(total as u16).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.resize(out.len() + pad, b' ');
        out.push(b'\n');
        for v in &self.values {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    /// Parse a `.npy` holding a C-order little-endian `float32` `[frames, 64]` array (format 1.x
    /// or 2.x), attributing it to `source`.
    pub fn from_npy(bytes: &[u8], source: LatentSource) -> Result<Self, LatentError> {
        let (frames, data) = parse_npy(bytes)?;
        let values = data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        Self::new(values, frames, source)
    }

    /// [`Self::from_npy`] attributed to `expected.source`, refusing content or shape that
    /// disagrees with `expected`.
    pub fn from_npy_verified(bytes: &[u8], expected: &LatentIdentity) -> Result<Self, LatentError> {
        let latents = Self::from_npy(bytes, expected.source.clone())?;
        latents.verify_against(expected)?;
        Ok(latents)
    }

    /// Import an external `latent.npy` (for example an upstream artifact); the source records the
    /// file's SHA-256.
    pub fn import_npy(bytes: &[u8]) -> Result<Self, LatentError> {
        Self::from_npy(
            bytes,
            LatentSource::Imported {
                file_sha256: sha256_bytes(bytes),
            },
        )
    }

    /// Write [`LATENT_FILE`] and its [`IDENTITY_FILE`] sidecar into `dir` (created if absent).
    pub fn save(&self, dir: &Path) -> Result<(), LatentError> {
        let io = |path: &Path| {
            let path = path.display().to_string();
            move |source| LatentError::Io { path, source }
        };
        fs::create_dir_all(dir).map_err(io(dir))?;
        let npy = dir.join(LATENT_FILE);
        fs::write(&npy, self.to_npy()).map_err(io(&npy))?;
        let sidecar = dir.join(IDENTITY_FILE);
        let body = serde_json::to_string_pretty(&self.identity.to_json())
            .expect("identity JSON serializes");
        fs::write(&sidecar, body + "\n").map_err(io(&sidecar))?;
        Ok(())
    }

    /// Read a [`Self::save`]d artifact from `dir`, verifying the latent bytes against the sidecar
    /// identity. A missing, corrupt or changed file is an error, never a fresh identity.
    pub fn load(dir: &Path) -> Result<Self, LatentError> {
        let read = |name: &str| {
            let path = dir.join(name);
            fs::read(&path).map_err(|source| LatentError::Io {
                path: path.display().to_string(),
                source,
            })
        };
        let sidecar: Value = serde_json::from_slice(&read(IDENTITY_FILE)?).map_err(|e| {
            LatentError::MetadataMismatch(format!("{IDENTITY_FILE} is not JSON: {e}"))
        })?;
        let expected = LatentIdentity::from_json(&sidecar)?;
        Self::from_npy_verified(&read(LATENT_FILE)?, &expected)
    }
}

/// Parse a `.npy`, returning `(frames, data bytes)` for a `<f4` C-order `[frames, 64]` array.
fn parse_npy(bytes: &[u8]) -> Result<(usize, &[u8]), LatentError> {
    let bad = |m: String| LatentError::Npy(m);
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(bad("missing \\x93NUMPY magic".into()));
    }
    let (hlen, start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 if bytes.len() >= 12 => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        ),
        v => return Err(bad(format!("unsupported format version {v}"))),
    };
    let header = bytes
        .get(start..start + hlen)
        .ok_or_else(|| bad("truncated header".into()))?;
    let header = std::str::from_utf8(header).map_err(|_| bad("non-UTF-8 header".into()))?;
    let field = |key: &str| -> Result<&str, LatentError> {
        let pat = format!("'{key}':");
        let at = header
            .find(&pat)
            .ok_or_else(|| bad(format!("header lacks `{key}`")))?;
        Ok(header[at + pat.len()..].trim_start())
    };
    if !field("descr")?.starts_with("'<f4'") {
        return Err(bad("dtype is not little-endian float32 ('<f4')".into()));
    }
    if !field("fortran_order")?.starts_with("False") {
        return Err(bad("Fortran-order arrays are not accepted".into()));
    }
    let shape = field("shape")?;
    let close = shape
        .find(')')
        .ok_or_else(|| bad("unterminated shape".into()))?;
    let dims: Vec<usize> = shape
        .get(1..close)
        .ok_or_else(|| bad("malformed shape".into()))?
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>())
        .collect::<Result<_, _>>()
        .map_err(|_| bad(format!("malformed shape {:?}", &shape[..=close])))?;
    let frames = match dims.as_slice() {
        [f, c] if *c == LATENT_CHANNELS => *f,
        _ => return Err(bad(format!("shape {dims:?}, expected [frames, 64]"))),
    };
    let data = &bytes[start + hlen..];
    if data.len() != frames * LATENT_CHANNELS * 4 {
        return Err(bad(format!(
            "{} data bytes for shape [{frames}, 64] (expected {})",
            data.len(),
            frames * LATENT_CHANNELS * 4
        )));
    }
    Ok((frames, data))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_meta() -> Value {
        serde_json::from_str(include_str!("../tests/fixtures/vae_tiny_reference.json")).unwrap()
    }

    pub(crate) fn fixture_latents() -> AcousticLatents {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/vae_tiny_reference.safetensors"
        );
        let map = candle_audio::candle_core::safetensors::load(path, &Device::Cpu).unwrap();
        AcousticLatents::from_tensor(
            &map["latent"],
            LatentSource::Synthesis {
                stage_identity: "fixture".into(),
            },
        )
        .unwrap()
    }

    fn source() -> LatentSource {
        LatentSource::Synthesis {
            stage_identity: "s".into(),
        }
    }

    /// The identity hash is over the canonical bytes numpy holds for the `[frames, 64]` array
    /// (mutation: hash the transposed layout, or big-endian bytes → red).
    #[test]
    fn identity_hash_matches_the_numpy_array_bytes() {
        let latents = fixture_latents();
        let meta = fixture_meta();
        assert_eq!(latents.frames(), meta["frames"].as_u64().unwrap() as usize);
        assert_eq!(
            latents.identity().sha256,
            meta["latent_sha256"].as_str().unwrap()
        );
        assert_eq!(latents.identity().shape(), [latents.frames(), 64]);
    }

    /// `to_npy` reproduces `numpy.save` byte for byte (mutation: drop numpy's growth padding or
    /// alignment → red), and reads back to the same identity.
    #[test]
    fn npy_writer_is_byte_identical_to_numpy_save() {
        let latents = fixture_latents();
        let npy = latents.to_npy();
        assert_eq!(
            sha256_bytes(&npy),
            fixture_meta()["latent_npy_sha256"].as_str().unwrap()
        );
        assert_eq!(npy.len() % 4, 0);
        assert_eq!((npy.len() - latents.values().len() * 4) % 64, 0);
        let back = AcousticLatents::from_npy_verified(&npy, latents.identity()).unwrap();
        assert_eq!(back, latents);
        let imported = AcousticLatents::import_npy(&npy).unwrap();
        assert_eq!(imported.identity().sha256, latents.identity().sha256);
        assert_eq!(
            imported.identity().source,
            LatentSource::Imported {
                file_sha256: sha256_bytes(&npy)
            }
        );
    }

    /// Several frame counts round-trip, including ones whose header pads to a full 64 bytes.
    #[test]
    fn npy_round_trips_across_header_lengths() {
        for frames in [1usize, 9, 10, 99, 1000, 123456] {
            let values: Vec<f32> = (0..frames * 64).map(|i| i as f32 * 0.25 - 3.0).collect();
            let l = AcousticLatents::new(values, frames, source()).unwrap();
            let npy = l.to_npy();
            assert_eq!(&npy[..8], b"\x93NUMPY\x01\x00");
            let hlen = u16::from_le_bytes([npy[8], npy[9]]) as usize;
            assert_eq!((10 + hlen) % 64, 0, "frames {frames}");
            assert_eq!(AcousticLatents::from_npy(&npy, source()).unwrap(), l);
        }
    }

    /// A changed value, a truncated file, a wrong dtype or a non-64 width is refused.
    #[test]
    fn corrupt_or_foreign_npy_is_refused() {
        let latents = fixture_latents();
        let mut npy = latents.to_npy();
        let last = npy.len() - 1;
        npy[last] ^= 0x01;
        assert!(matches!(
            AcousticLatents::from_npy_verified(&npy, latents.identity()),
            Err(LatentError::IdentityMismatch { .. })
        ));
        let good = latents.to_npy();
        assert!(matches!(
            AcousticLatents::from_npy(&good[..good.len() - 4], source()),
            Err(LatentError::Npy(_))
        ));
        let f8 = String::from_utf8_lossy(&good).replacen("'<f4'", "'<f8'", 1);
        assert!(matches!(
            AcousticLatents::from_npy(f8.as_bytes(), source()),
            Err(LatentError::Npy(_))
        ));
        let wide = AcousticLatents::new(vec![0.0; 64], 1, source())
            .unwrap()
            .to_npy();
        let wide = String::from_utf8_lossy(&wide).replacen("(1, 64)", "(2, 32)", 1);
        assert!(AcousticLatents::from_npy(wide.as_bytes(), source()).is_err());
    }

    /// Empty, ragged and non-finite latents are refused at construction.
    #[test]
    fn invalid_latents_are_refused() {
        assert!(AcousticLatents::new(vec![], 0, source()).is_err());
        assert!(AcousticLatents::new(vec![0.0; 63], 1, source()).is_err());
        let mut v = vec![0.0; 128];
        v[70] = f32::NAN;
        let err = AcousticLatents::new(v, 2, source())
            .unwrap_err()
            .to_string();
        assert!(err.contains("frame 1, channel 6"), "{err}");
    }

    /// `[1, 64, T]` and `[T, 64]` tensors give the same latents, and `to_decoder_input` restores
    /// the `[1, 64, T]` layout (mutation: skip the transpose in either direction → red).
    #[test]
    fn tensor_layouts_agree() {
        let latents = fixture_latents();
        let bct = latents.to_decoder_input(&Device::Cpu).unwrap();
        assert_eq!(bct.dims(), &[1, 64, latents.frames()]);
        // Channel 5 of frame 2 sits at [0, 5, 2] in BCT and at row 2, column 5 row-major.
        let v = bct
            .get(0)
            .unwrap()
            .get(5)
            .unwrap()
            .get(2)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(v, latents.values()[2 * 64 + 5]);
        let again = AcousticLatents::from_tensor(&bct, latents.identity().source.clone()).unwrap();
        assert_eq!(again, latents);
    }

    /// Save/load round-trips; an edited sidecar or latent file is refused, and a sidecar whose
    /// source differs is refused even though the content hash matches.
    #[test]
    fn saved_artifact_round_trips_and_detects_tampering() {
        let dir = tempfile::tempdir().unwrap();
        let latents = fixture_latents();
        latents.save(dir.path()).unwrap();
        assert_eq!(AcousticLatents::load(dir.path()).unwrap(), latents);

        let npy_path = dir.path().join(LATENT_FILE);
        let mut npy = fs::read(&npy_path).unwrap();
        let at = npy.len() - 10;
        npy[at] ^= 0x40;
        fs::write(&npy_path, &npy).unwrap();
        assert!(matches!(
            AcousticLatents::load(dir.path()),
            Err(LatentError::IdentityMismatch { .. })
        ));

        latents.save(dir.path()).unwrap();
        // Same content, different recorded provenance: refused by `verify_against`.
        let mut id = latents.identity().clone();
        id.source = LatentSource::Imported {
            file_sha256: "0".repeat(64),
        };
        assert!(matches!(
            latents.verify_against(&id),
            Err(LatentError::MetadataMismatch(_))
        ));
        // An edited sidecar hash is refused against the unchanged latent file.
        let sidecar = dir.path().join(IDENTITY_FILE);
        let text = fs::read_to_string(&sidecar).unwrap();
        let sha = &latents.identity().sha256;
        fs::write(&sidecar, text.replace(sha.as_str(), &"0".repeat(64))).unwrap();
        assert!(matches!(
            AcousticLatents::load(dir.path()),
            Err(LatentError::IdentityMismatch { .. })
        ));
        fs::remove_file(dir.path().join(IDENTITY_FILE)).unwrap();
        assert!(matches!(
            AcousticLatents::load(dir.path()),
            Err(LatentError::Io { .. })
        ));
    }

    /// Every source kind survives the JSON record.
    #[test]
    fn identity_json_round_trips_every_source() {
        for source in [
            LatentSource::Synthesis {
                stage_identity: "abc".into(),
            },
            LatentSource::Encoded {
                vae: "yue2_vae_legacy".into(),
                audio_sha256: "f".repeat(64),
                sampled: true,
            },
            LatentSource::Imported {
                file_sha256: "e".repeat(64),
            },
        ] {
            let id = LatentIdentity {
                sha256: "a".repeat(64),
                frames: 3,
                source,
            };
            assert_eq!(LatentIdentity::from_json(&id.to_json()).unwrap(), id);
        }
        let mut bad = LatentIdentity {
            sha256: "a".repeat(64),
            frames: 3,
            source: source(),
        }
        .to_json();
        bad["dtype"] = json!("BF16");
        assert!(LatentIdentity::from_json(&bad).is_err());
    }
}
