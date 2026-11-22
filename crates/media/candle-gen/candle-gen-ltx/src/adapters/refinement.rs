//! Bounded reader for the shipped LTX 2.5 Dev refinement adapter.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

use candle_gen::candle_core::{DType, Device, Tensor};
#[cfg(feature = "model-inference-proof")]
use candle_gen::gen_core::model_input_proof::{record_inputs, InputIdentity, InputRole};
use candle_gen::gen_core::weightsmeta::{self, Dtype, SafetensorsTensorLocation};
use candle_gen::gen_core::{AdapterKind, AdapterSpec, CancelFlag};
use candle_gen::residency::check_cancel;
use candle_gen::train::merge::MAX_ADAPTER_BYTES;
use candle_gen::{CandleError, Result};
use sha2::{Digest, Sha256};

struct RefinementPin {
    file_bytes: u64,
    header_bytes: u64,
    header_sha256: [u8; 32],
    file_sha256: [u8; 32],
    factors: usize,
    max_factor_bytes: u64,
}

const SHIPPED_FILENAME: &str = "ltx-2.5-22b-distilled-lora-450-bf16.safetensors";
const READ_BYTES: usize = 64 * 1024;

// SceneWorks/ltx-2.5-mlx @ 081658ce6886cacba20817ce0359bbefef706ff2,
// distilled_lora/ltx-2.5-22b-distilled-lora-450-bf16.safetensors.
// Both hashes are bound to the published artifact, independent of the caller's local path.
const SHIPPED_REFINEMENT: RefinementPin = RefinementPin {
    file_bytes: 8_899_889_568,
    header_bytes: 526_744,
    header_sha256: [
        0x71, 0x5a, 0x1c, 0xdc, 0xba, 0x60, 0x78, 0xf4, 0xe7, 0xbd, 0xea, 0xc1, 0xc0, 0x82, 0x09,
        0x54, 0xf3, 0x31, 0xfc, 0x2a, 0x38, 0x1e, 0xc0, 0x4b, 0xf4, 0x1a, 0x58, 0x6e, 0x7a, 0x74,
        0xd4, 0x2f,
    ],
    file_sha256: [
        0x86, 0x37, 0x0b, 0xbf, 0x79, 0xa9, 0xeb, 0x4e, 0xda, 0xa1, 0x58, 0x90, 0x7e, 0x2b, 0x48,
        0xa5, 0x18, 0x8f, 0xe4, 0xc5, 0xdc, 0x8c, 0xe3, 0x0c, 0x7e, 0xb8, 0xf2, 0xf1, 0x31, 0xa9,
        0xbb, 0xf5,
    ],
    factors: 3_320,
    // The largest published BF16 factor has shape [36_864, 450].
    max_factor_bytes: 33_177_600,
};

fn refuse(path: &Path, reason: &str) -> CandleError {
    CandleError::Msg(format!(
        "ltx_2_5: cannot load {} as the shipped Dev refinement LoRA: {reason}",
        path.display()
    ))
}

fn read_error(path: &Path, error: io::Error) -> CandleError {
    CandleError::Msg(format!("read adapter {}: {error}", path.display()))
}

/// Reserve the manifest's exact filename for the shipped refinement. Ordinary
/// adapters keep the buffered path. A renamed copy must meet the same pin.
pub(super) fn read_shipped<'a>(
    spec: &'a AdapterSpec,
    used_refinement: &mut bool,
    cancel: &'a CancelFlag,
) -> Result<Option<RefinementSource<'a, File>>> {
    let path = &spec.path;
    let shipped_name = path
        .file_name()
        .is_some_and(|name| name.eq_ignore_ascii_case(SHIPPED_FILENAME));
    let metadata = std::fs::metadata(path).map_err(|error| read_error(path, error))?;
    if !shipped_name && metadata.len() <= MAX_ADAPTER_BYTES {
        return Ok(None);
    }
    check_cancel(cancel)?;
    if *used_refinement {
        return Err(refuse(
            path,
            "the shipped Dev refinement adapter is already selected",
        ));
    }
    let pin = &SHIPPED_REFINEMENT;
    if !metadata.is_file() || metadata.len() != pin.file_bytes || spec.kind != AdapterKind::Lora {
        return Err(refuse(
            path,
            &format!(
                "expected a regular LoRA file of {} bytes, found {} bytes (buffered adapter cap: {MAX_ADAPTER_BYTES} bytes)",
                pin.file_bytes,
                metadata.len(),
            ),
        ));
    }
    let file = File::open(path).map_err(|error| read_error(path, error))?;
    let opened = file.metadata().map_err(|error| read_error(path, error))?;
    if !opened.is_file() || opened.len() != pin.file_bytes {
        return Err(refuse(
            path,
            "opened file differs from the shipped Dev refinement LoRA",
        ));
    }
    let adapter = read_pinned(file, path, pin, cancel)?;
    *used_refinement = true;
    Ok(Some(adapter))
}

pub(super) struct VerifiedFactor {
    offset: u64,
    bytes: usize,
    shape: Vec<usize>,
    sha256: [u8; 32],
}

type NamedFactors = Vec<(String, VerifiedFactor)>;

/// The same open handle used for whole-file authentication, with no retained tensor payload.
pub(super) struct RefinementSource<'a, R> {
    file: R,
    path: &'a Path,
    pin: &'a RefinementPin,
    cancel: &'a CancelFlag,
    meta: HashMap<String, String>,
    factors: NamedFactors,
    remaining: usize,
    scratch: Vec<u8>,
}

impl<R: Read + Seek> RefinementSource<'_, R> {
    pub(super) fn take_layout(&mut self) -> (HashMap<String, String>, NamedFactors) {
        (
            std::mem::take(&mut self.meta),
            std::mem::take(&mut self.factors),
        )
    }

    pub(super) fn read_factor(&mut self, factor: VerifiedFactor) -> Result<Tensor> {
        let Self {
            file,
            path,
            cancel,
            remaining,
            scratch,
            ..
        } = self;
        check_cancel(cancel)?;
        file.seek(SeekFrom::Start(factor.offset))
            .map_err(|error| read_error(path, error))?;
        let bytes = &mut scratch[..factor.bytes];
        let mut hash = Sha256::new();
        for chunk in bytes.chunks_mut(READ_BYTES) {
            check_cancel(cancel)?;
            file.read_exact(chunk)
                .map_err(|error| read_error(path, error))?;
            hash.update(&*chunk);
        }
        check_cancel(cancel)?;
        if <[u8; 32]>::from(hash.finalize()) != factor.sha256 {
            return Err(refuse(
                path,
                "LoRA factor SHA-256 changed after authentication",
            ));
        }
        let tensor = Tensor::from_raw_buffer(bytes, DType::BF16, &factor.shape, &Device::Cpu)?;
        *remaining -= 1;
        Ok(tensor)
    }

    pub(super) fn finish(mut self) -> Result<()> {
        check_cancel(self.cancel)?;
        if self.remaining != 0 {
            return Err(refuse(
                self.path,
                "not all authenticated LoRA factors were consumed",
            ));
        }
        let length = self
            .file
            .seek(SeekFrom::End(0))
            .map_err(|error| read_error(self.path, error))?;
        if length != self.pin.file_bytes {
            return Err(refuse(self.path, "file size changed after authentication"));
        }
        check_cancel(self.cancel)?;
        #[cfg(feature = "model-inference-proof")]
        record_inputs(
            "candle-gen-ltx::adapters::refinement::read_shipped/verified-weights",
            InputRole::ArtifactInput,
            &[InputIdentity::validated(
                self.path,
                self.pin.file_sha256,
                "candle-gen-ltx::adapters::refinement::read_pinned",
                "pinned whole-file SHA-256 matched and every decoded factor matched its authenticated SHA-256 on the same open file",
            )],
        );
        Ok(())
    }
}

fn finish_hash(file: &mut impl Read, path: &Path, pin: &RefinementPin, hash: Sha256) -> Result<()> {
    let digest: [u8; 32] = hash.finalize().into();
    if file
        .read(&mut [0u8; 1])
        .map_err(|error| read_error(path, error))?
        != 0
        || digest != pin.file_sha256
    {
        return Err(refuse(
            path,
            "file size or SHA-256 differs from the shipped asset",
        ));
    }
    Ok(())
}

fn authenticate_payload(
    file: &mut impl Read,
    path: &Path,
    pin: &RefinementPin,
    locations: Vec<SafetensorsTensorLocation>,
    mut hash: Sha256,
    cancel: &CancelFlag,
) -> Result<NamedFactors> {
    let mut scratch = [0u8; READ_BYTES];
    let mut factors = Vec::with_capacity(locations.len());
    // The validated layout covers the file contiguously and is in byte order.
    for location in locations {
        let header = location.header;
        let bytes = header.data_bytes as usize;
        let mut remaining = bytes;
        let mut factor_hash = Sha256::new();
        while remaining != 0 {
            check_cancel(cancel)?;
            let n = remaining.min(scratch.len());
            let chunk = &mut scratch[..n];
            file.read_exact(chunk)
                .map_err(|error| read_error(path, error))?;
            hash.update(&*chunk);
            factor_hash.update(&*chunk);
            remaining -= n;
        }
        factors.push((
            header.name,
            VerifiedFactor {
                offset: location.file_offset,
                bytes,
                shape: header.shape,
                sha256: factor_hash.finalize().into(),
            },
        ));
    }
    check_cancel(cancel)?;
    finish_hash(file, path, pin, hash)?;
    Ok(factors)
}

fn read_pinned<'a, R: Read + Seek>(
    mut file: R,
    path: &'a Path,
    pin: &'a RefinementPin,
    cancel: &'a CancelFlag,
) -> Result<RefinementSource<'a, R>> {
    check_cancel(cancel)?;
    let mut prefix = [0u8; 8];
    file.read_exact(&mut prefix)
        .map_err(|error| read_error(path, error))?;
    if u64::from_le_bytes(prefix) != pin.header_bytes {
        return Err(refuse(path, "unexpected safetensors header length"));
    }
    let mut layout = weightsmeta::safetensors_reader_tensor_locations(
        prefix.as_slice().chain(&mut file),
        pin.file_bytes,
        path,
    )
    .map_err(|error| CandleError::Msg(error.to_string()))?;
    let header_sha256: [u8; 32] = Sha256::digest(&layout.header_json).into();
    if header_sha256 != pin.header_sha256 {
        return Err(refuse(
            path,
            "safetensors header SHA-256 differs from the shipped asset",
        ));
    }
    if layout.tensors.len() != pin.factors
        || layout.tensors.iter().any(|location| {
            let factor = &location.header;
            factor.dtype != Dtype::BF16
                || factor.shape.len() != 2
                || factor.data_bytes == 0
                || factor.data_bytes > pin.max_factor_bytes
        })
    {
        return Err(refuse(
            path,
            "unexpected BF16 LoRA factor count, geometry, or size",
        ));
    }
    let meta = weightsmeta::safetensors_metadata_from_json(path, &layout.header_json)
        .map_err(|error| CandleError::Msg(error.to_string()))?
        .into_iter()
        .collect();

    // Authenticate before allocating factors. An untrusted file can match the
    // published header and length without containing the published payload.
    let mut hash = Sha256::new();
    hash.update(prefix);
    hash.update(&layout.header_json);
    let factors = authenticate_payload(&mut file, path, pin, layout.tensors, hash, cancel)?;

    // Recheck the header on the same handle. Each factor is checked when the
    // installer seeks to its payload, before the decoded values can be used.
    check_cancel(cancel)?;
    file.seek(SeekFrom::Start(0))
        .map_err(|error| read_error(path, error))?;
    file.read_exact(&mut prefix)
        .map_err(|error| read_error(path, error))?;
    file.read_exact(&mut layout.header_json)
        .map_err(|error| read_error(path, error))?;
    let current_header_sha256: [u8; 32] = Sha256::digest(&layout.header_json).into();
    if u64::from_le_bytes(prefix) != pin.header_bytes || current_header_sha256 != pin.header_sha256
    {
        return Err(refuse(
            path,
            "safetensors header changed after authentication",
        ));
    }
    check_cancel(cancel)?;
    let max_bytes = factors
        .iter()
        .map(|(_, factor)| factor.bytes)
        .max()
        .unwrap_or(0);
    Ok(RefinementSource {
        file,
        path,
        pin,
        cancel,
        meta,
        remaining: factors.len(),
        factors,
        scratch: vec![0; max_bytes],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const A: &str = "diffusion_model.proj.lora_A.weight";
    const B: &str = "diffusion_model.proj.lora_B.weight";

    fn encode(header: &[u8], payload: &[u8]) -> (Vec<u8>, RefinementPin) {
        let header_bytes = header.len() as u64;
        let mut bytes = header_bytes.to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(payload);
        let pin = RefinementPin {
            file_bytes: bytes.len() as u64,
            header_bytes,
            header_sha256: Sha256::digest(header).into(),
            file_sha256: Sha256::digest(&bytes).into(),
            factors: 2,
            max_factor_bytes: 4,
        };
        (bytes, pin)
    }

    fn fixture() -> (Vec<u8>, RefinementPin) {
        let header = serde_json::to_vec(&serde_json::json!({
            "__metadata__": {"lora_rank": "1", "lora_alpha": "1"},
            (A): {"dtype": "BF16", "shape": [1, 2], "data_offsets": [0, 4]},
            (B): {"dtype": "BF16", "shape": [2, 1], "data_offsets": [4, 8]},
        }))
        .unwrap();
        encode(&header, &[0x80, 0x3f, 0x00, 0xc0, 0x40, 0x40, 0x80, 0x40])
    }

    fn failure(bytes: &[u8], pin: &RefinementPin) -> String {
        read_pinned(
            Cursor::new(bytes),
            Path::new("adapter.safetensors"),
            pin,
            &CancelFlag::new(),
        )
        .err()
        .expect("reader must refuse the altered input")
        .to_string()
    }

    #[test]
    fn reads_native_factors_and_metadata_from_the_open_handle() {
        let (bytes, pin) = fixture();
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), &bytes).unwrap();
        let cancel = CancelFlag::new();
        let mut source = read_pinned(
            File::open(file.path()).unwrap(),
            Path::new("display-only.safetensors"),
            &pin,
            &cancel,
        )
        .unwrap();
        let (meta, factors) = source.take_layout();
        assert_eq!(meta["lora_rank"], "1");
        assert_eq!(meta["lora_alpha"], "1");
        assert_eq!(source.scratch.len(), pin.max_factor_bytes as usize);
        assert_eq!(source.remaining, pin.factors);
        let tensors: HashMap<_, _> = factors
            .into_iter()
            .rev()
            .map(|(name, factor)| (name, source.read_factor(factor).unwrap()))
            .collect();
        source.finish().unwrap();
        assert_eq!(tensors[A].dtype(), DType::BF16);
        assert_eq!(
            tensors[A]
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap(),
            vec![vec![1.0, -2.0]]
        );
        assert_eq!(
            tensors[B]
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap(),
            vec![vec![3.0], vec![4.0]]
        );
    }

    #[test]
    fn rejects_unpinned_headers_and_truncated_changed_or_extended_payloads() {
        let (bytes, mut pin) = fixture();
        pin.header_sha256[0] ^= 1;
        assert!(failure(&bytes, &pin).contains("header SHA-256"));
        pin.header_sha256[0] ^= 1;
        assert!(failure(&bytes[..bytes.len() - 1], &pin).contains("read adapter"));
        let mut changed = bytes.clone();
        *changed.last_mut().unwrap() ^= 1;
        assert!(failure(&changed, &pin).contains("file size or SHA-256"));
        changed.push(0);
        assert!(failure(&changed, &pin).contains("file size or SHA-256"));
    }

    #[test]
    fn authenticates_before_rewinding_and_refuses_in_place_changes() {
        struct ChangeOnRewind {
            file: Cursor<Vec<u8>>,
            rewound: bool,
            change_at: usize,
        }

        impl Read for ChangeOnRewind {
            fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
                self.file.read(bytes)
            }
        }

        impl Seek for ChangeOnRewind {
            fn seek(&mut self, offset: SeekFrom) -> io::Result<u64> {
                if offset == SeekFrom::Start(0) {
                    self.rewound = true;
                    self.file.get_mut()[self.change_at] ^= 1;
                }
                self.file.seek(offset)
            }
        }

        let (mut bytes, pin) = fixture();
        let path = Path::new("adapter.safetensors");
        let cancel = CancelFlag::new();
        let change_at = bytes.len() - 1;
        let mut source = ChangeOnRewind {
            file: Cursor::new(bytes.clone()),
            rewound: false,
            change_at,
        };
        let mut adapter = read_pinned(&mut source, path, &pin, &cancel).unwrap();
        let (_, factors) = adapter.take_layout();
        for (name, factor) in factors {
            let result = adapter.read_factor(factor);
            if name == B {
                let error = result.unwrap_err().to_string();
                assert!(error.contains("factor SHA-256 changed after authentication"));
            } else {
                assert_eq!(name, A);
                assert!(result.is_ok());
            }
        }
        assert!(adapter.finish().is_err());
        assert!(source.rewound);

        let mut changed_header = ChangeOnRewind {
            file: Cursor::new(bytes.clone()),
            rewound: false,
            change_at: 8,
        };
        let header_error = read_pinned(&mut changed_header, path, &pin, &cancel)
            .err()
            .unwrap()
            .to_string();
        assert!(header_error.contains("header changed after authentication"));

        bytes[change_at] ^= 1;
        let mut already_changed = ChangeOnRewind {
            file: Cursor::new(bytes),
            rewound: false,
            change_at,
        };
        assert!(read_pinned(&mut already_changed, path, &pin, &cancel).is_err());
        assert!(!already_changed.rewound);
    }

    #[test]
    fn refuses_size_changes_after_checked_factors_were_consumed() {
        let (bytes, pin) = fixture();
        let cancel = CancelFlag::new();
        let mut source = read_pinned(
            Cursor::new(bytes),
            Path::new("adapter.safetensors"),
            &pin,
            &cancel,
        )
        .unwrap();
        let (_, factors) = source.take_layout();
        for (_, factor) in factors {
            source.read_factor(factor).unwrap();
        }
        source.file.get_mut().push(0);
        let error = source.finish().unwrap_err().to_string();
        assert!(error.contains("file size changed after authentication"));
    }

    #[test]
    fn cancels_in_both_payload_passes_before_another_read() {
        struct CancelOnPayload<'a> {
            file: Cursor<Vec<u8>>,
            cancel: &'a CancelFlag,
            data_start: u64,
            cancel_after_rewind: bool,
            rewound: bool,
        }

        impl Read for CancelOnPayload<'_> {
            fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
                assert!(
                    !self.cancel.is_cancelled(),
                    "reader continued after cancellation"
                );
                let offset = self.file.position();
                let read = self.file.read(bytes)?;
                if offset >= self.data_start && self.rewound == self.cancel_after_rewind {
                    self.cancel.cancel();
                }
                Ok(read)
            }
        }

        impl Seek for CancelOnPayload<'_> {
            fn seek(&mut self, offset: SeekFrom) -> io::Result<u64> {
                assert!(
                    !self.cancel.is_cancelled(),
                    "reader rewound after cancellation"
                );
                if offset == SeekFrom::Start(0) {
                    self.rewound = true;
                }
                self.file.seek(offset)
            }
        }

        let (bytes, pin) = fixture();
        for cancel_after_rewind in [false, true] {
            let cancel = CancelFlag::new();
            let mut source = CancelOnPayload {
                file: Cursor::new(bytes.clone()),
                cancel: &cancel,
                data_start: 8 + pin.header_bytes,
                cancel_after_rewind,
                rewound: false,
            };
            let result = read_pinned(&mut source, Path::new("adapter.safetensors"), &pin, &cancel)
                .and_then(|mut source| {
                    let (_, factors) = source.take_layout();
                    let (_, first_factor) = factors.into_iter().next().unwrap();
                    source.read_factor(first_factor).map(|_| ())
                });
            assert!(matches!(result, Err(CandleError::Canceled)));
            assert_eq!(source.rewound, cancel_after_rewind);
        }
    }

    #[test]
    fn validates_layout_and_factor_size_before_reading_payload() {
        let (bytes, mut pin) = fixture();
        pin.max_factor_bytes = 3;
        assert!(failure(&bytes[..8 + pin.header_bytes as usize], &pin)
            .contains("factor count, geometry, or size"));

        let broken_header = serde_json::to_vec(&serde_json::json!({
            (A): {"dtype": "BF16", "shape": [1, 2], "data_offsets": [0, 4]},
            (B): {"dtype": "BF16", "shape": [2, 1], "data_offsets": [2, 6]},
        }))
        .unwrap();
        let (overlap, pin) = encode(&broken_header, &[0; 8]);
        assert!(failure(&overlap, &pin).contains("expected contiguous offset"));
    }

    #[test]
    fn reserves_the_shipped_filename_even_for_a_valid_small_lora() {
        let (bytes, _) = fixture();
        let dir = tempfile::tempdir().unwrap();
        let shipped = dir.path().join(SHIPPED_FILENAME);
        let ordinary = dir.path().join("small.safetensors");
        std::fs::write(&shipped, &bytes).unwrap();
        std::fs::write(&ordinary, &bytes).unwrap();

        let mut used = false;
        let cancel = CancelFlag::new();
        let spec = AdapterSpec::new(shipped, 1.0, AdapterKind::Lora);
        let error = read_shipped(&spec, &mut used, &cancel).err().unwrap();
        assert!(error.to_string().contains("expected a regular LoRA file"));
        assert!(!used);
        let spec = AdapterSpec::new(ordinary, 1.0, AdapterKind::Lora);
        assert!(read_shipped(&spec, &mut used, &cancel).unwrap().is_none());
        assert_eq!(
            candle_gen::train::merge::read_adapter(&spec.path)
                .unwrap()
                .tensors
                .len(),
            2
        );
    }
}
