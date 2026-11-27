//! The Krea 2 **W4A8 tier** (SW-6135, feature SW-F287): its admission gate and its builder.
//!
//! # What the tier is
//!
//! A single native-keyed safetensors file — the same layout as the community INT8-ConvRot export
//! and the Comfy-Org `krea2_turbo_*` single files (bare `blocks.N.…` keys) — whose selected DiT
//! projections are ComfyUI **`asym_w4a8_int8`** projections: ConvRot-rotated weights, 16-level
//! Lloyd-Max int4 codes packed two per byte, per-group E4M3 relative scales and per-channel F32
//! scales, written by SW-6134's packer ([`candle_gen::gen_core::w4a8_int8::quantize_weight`]). Every
//! other tensor is copied byte for byte from the bf16 master. It loads through the ordinary native
//! single-file import route (`load_from_native_dit_file`, `WeightsSource::File` + `base_snapshot`):
//! the loader delegates the W4A8 projections out of the codec plan and serves them through
//! `W4A8Linear` ([`crate::loader::linear_detect`]); the tokenizer, text encoder, VAE and config come
//! from the base snapshot, exactly as for the INT8-ConvRot and NVFP4 single files.
//!
//! # Which projections pack ([`W4A8Policy`])
//!
//! [`W4A8Policy::Mixed`] (the shipped tier) asks Krea's own NVFP4 role table
//! ([`crate::nvfp4_dit::LayerRole::for_krea_layer`]) the same question the NVFP4 lane asks: only a
//! projection whose structural role selects the packed lane is packed. The sensitive classes the
//! role table names stay at their source precision: the four leading and the last single-stream
//! block (caption outliers on the block inputs), every post-nonlinearity leaf (`attn.to_out.0`,
//! `ff.down`), every context reader (`text_fusion.*`, `txt_in.*`) and the trunk head — plus every
//! projection the role table does not classify (`time_embed.*`, `time_mod_proj`,
//! `text_fusion.projector`). A packable role must also clear the format's geometry
//! (`in % 256 == 0` for the ConvRot block, `out % 16 == 0` for the CUDA GEMM); `img_in` (`in = 64`)
//! does not and stays dense. [`W4A8Policy::Blanket`] packs every geometrically eligible projection
//! — a measurement arm, not a shipping tier.
//!
//! # Admission ([`check_w4a8_tier`])
//!
//! W4A8 executes on CUDA `sm_80+` only (int8 `mma.sync`; S9's
//! [`candle_gen::quant::w4a8_int8::W4A8_COMPUTE_CAP_FLOOR`]). There is no floating fallback for the
//! packed codes, so every other backend and every older CUDA part gets a typed, backend-aware
//! refusal before any weight is read (epic E5).

use std::path::{Path, PathBuf};

use candle_gen::gen_core::w4a8_int8::{self, W4A8CheckpointTensor, W4A8Packed};
use candle_gen::gen_core::weightsmeta::Dtype;
use candle_gen::quant::w4a8_int8::{w4a8_tier_refusal, W4A8TierRefusal, W4A8_COMPUTE_CAP_FLOOR};

use crate::loader::convrot_diffusers_to_native;
use crate::nvfp4_dit::{DenseReason, ExecutionRole, LayerRole};

/// Why a backend cannot run the W4A8 tier at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum W4A8TierUnavailable {
    /// The device is not a CUDA device (CPU, Metal), or this build has no CUDA backend.
    NotCuda {
        /// The backend the load was asked to run on (`cpu`, `metal`, or `cuda` for a build without
        /// the `cuda` feature).
        backend: &'static str,
    },
    /// A CUDA device below the `sm_80` int8 tensor-core floor.
    BelowFloor(W4A8TierRefusal),
}

impl std::fmt::Display for W4A8TierUnavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotCuda { backend } => write!(
                f,
                "the Krea 2 W4A8 tier runs only on CUDA sm_{}{}+ (packed int4 weights have no \
                 floating fallback); this load targets the {backend} backend",
                W4A8_COMPUTE_CAP_FLOOR.0, W4A8_COMPUTE_CAP_FLOOR.1
            ),
            Self::BelowFloor(refusal) => {
                write!(f, "the Krea 2 W4A8 tier is not offered: {refusal}")
            }
        }
    }
}

impl std::error::Error for W4A8TierUnavailable {}

/// Pure W4A8 tier admission: `cuda_capability` is `Some((major, minor))` for a CUDA device in a
/// CUDA build, `None` otherwise (`backend` names what the load targets). CPU-testable.
pub fn check_w4a8_tier(
    cuda_capability: Option<(i32, i32)>,
    backend: &'static str,
) -> Result<(), W4A8TierUnavailable> {
    match cuda_capability {
        None => Err(W4A8TierUnavailable::NotCuda { backend }),
        Some(cap) => match w4a8_tier_refusal(cap) {
            Some(refusal) => Err(W4A8TierUnavailable::BelowFloor(refusal)),
            None => Ok(()),
        },
    }
}

/// The backend label [`check_w4a8_tier`] reports for `device`.
pub fn backend_label(device: &candle_gen::candle_core::Device) -> &'static str {
    match device {
        candle_gen::candle_core::Device::Cpu => "cpu",
        candle_gen::candle_core::Device::Metal(_) => "metal",
        candle_gen::candle_core::Device::Cuda(_) => "cuda",
    }
}

/// Which projections a tier build packs (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum W4A8Policy {
    /// Krea's NVFP4 role table decides; sensitive projections stay at source precision. The
    /// shipped tier.
    Mixed,
    /// Every geometrically eligible projection packs. A measurement arm only.
    Blanket,
}

/// Why a projection stays at its source precision in a tier build.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeepReason {
    /// The role table selects a dense role for it (the NVFP4 lane's reason).
    Role(DenseReason),
    /// `in_features` is not a multiple of the 256-column ConvRot block, or `out_features` is not a
    /// multiple of 16 (the CUDA GEMM's output tile).
    Geometry,
}

/// The decision for one projection `diffusers_base` (`transformer_blocks.4.attn.to_q`) of shape
/// `[out, in]` on a trunk of `num_layers` single-stream blocks.
pub fn w4a8_role(
    policy: W4A8Policy,
    diffusers_base: &str,
    out_features: usize,
    in_features: usize,
    num_layers: usize,
) -> Result<(), KeepReason> {
    if !in_features.is_multiple_of(w4a8_int8::CONVROT_GROUP) || !out_features.is_multiple_of(16) {
        return Err(KeepReason::Geometry);
    }
    match policy {
        W4A8Policy::Blanket => Ok(()),
        W4A8Policy::Mixed => {
            match LayerRole::for_krea_layer(diffusers_base, num_layers).execution_role() {
                ExecutionRole::PackedW4A4 => Ok(()),
                ExecutionRole::DenseBf16(reason) => Err(KeepReason::Role(reason)),
            }
        }
    }
}

/// What a tier build wrote.
#[derive(Clone, Debug, Default)]
pub struct W4A8TierReport {
    /// Packed projections, as `(diffusers base, out, in)`, in key order.
    pub packed: Vec<(String, usize, usize)>,
    /// Two-dimensional projection weights kept at source precision, with the reason.
    pub kept: Vec<(String, KeepReason)>,
    /// Tensors written in total.
    pub tensors: usize,
    /// Source payload bytes read.
    pub source_bytes: u64,
    /// Payload bytes written (excluding the safetensors header).
    pub output_bytes: u64,
}

/// Build the W4A8 tier of a Krea 2 DiT.
///
/// `transformer_dir` is a **dense** diffusers `transformer/` component (the bf16 master:
/// `SceneWorks/krea-2-turbo-mlx` `bf16/transformer`); `num_layers` is its single-stream block
/// count. The output is one native-keyed file at `out_file` (see the module docs). Rows pack in
/// parallel over `threads` workers; the codebook is fixed, so the bytes do not depend on the
/// thread count. Refuses a packed or foreign source rather than re-quantizing a quantized tier.
pub fn build_w4a8_tier(
    transformer_dir: &Path,
    num_layers: usize,
    policy: W4A8Policy,
    out_file: &Path,
    threads: usize,
) -> Result<W4A8TierReport, String> {
    let mut shards: Vec<PathBuf> = std::fs::read_dir(transformer_dir)
        .map_err(|e| format!("{}: {e}", transformer_dir.display()))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    shards.sort();
    if shards.is_empty() {
        return Err(format!(
            "{}: no .safetensors shards",
            transformer_dir.display()
        ));
    }
    let mut report = W4A8TierReport::default();
    let mut tensors: Vec<W4A8CheckpointTensor> = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for shard in &shards {
        let layout = candle_gen::gen_core::safetensors_file_tensor_locations(shard)
            .map_err(|e| format!("{}: {e}", shard.display()))?;
        let mut file =
            std::fs::File::open(shard).map_err(|e| format!("{}: {e}", shard.display()))?;
        for location in layout.tensors {
            let header = location.header;
            let name = header.name.clone();
            const QUANTIZED: [&str; 8] = [
                ".scales",
                ".biases",
                ".comfy_quant",
                ".weight_scale",
                ".weight_scale_2",
                ".weight_codebook",
                ".weight_s_rel",
                ".weight_s_channel",
            ];
            if QUANTIZED.iter().any(|suffix| name.ends_with(suffix)) {
                return Err(format!(
                    "{name}: the W4A8 tier packs from the dense bf16 master, not a quantized tier"
                ));
            }
            if !seen.insert(name.clone()) {
                return Err(format!("{name}: duplicate tensor across shards"));
            }
            let native = convrot_diffusers_to_native(&name)
                .ok_or_else(|| format!("{name}: not a Krea 2 DiT tensor"))?;
            let data = read_payload(&mut file, location.file_offset, header.data_bytes)
                .map_err(|e| format!("{name}: {e}"))?;
            report.source_bytes += header.data_bytes;
            let projection = name
                .strip_suffix(".weight")
                .filter(|_| header.shape.len() == 2);
            let decision = projection
                .map(|base| w4a8_role(policy, base, header.shape[0], header.shape[1], num_layers));
            match (projection, decision) {
                (Some(base), Some(Ok(()))) => {
                    let (out, inp) = (header.shape[0], header.shape[1]);
                    let weight = to_f32(&name, header.dtype, &data)?;
                    drop(data);
                    let packed = pack_parallel(&weight, out, inp, threads)
                        .map_err(|e| format!("{name}: {e}"))?;
                    let native_base = native
                        .strip_suffix(".weight")
                        .expect("a projection weight maps to a native weight");
                    tensors.extend(packed.checkpoint_tensors(native_base));
                    report.packed.push((base.to_owned(), out, inp));
                }
                (base, decision) => {
                    if let (Some(base), Some(Err(reason))) = (base, decision) {
                        report.kept.push((base.to_owned(), reason));
                    }
                    // The native per-block modulation table is stored flat (`mod.lin [6·h]`) in
                    // every native Krea export; the bytes are identical row-major.
                    let shape = if name.starts_with("transformer_blocks.")
                        && name.ends_with(".scale_shift_table")
                    {
                        vec![header.shape.iter().product()]
                    } else {
                        header.shape.clone()
                    };
                    tensors.push(W4A8CheckpointTensor::passthrough(
                        native,
                        header.dtype,
                        shape,
                        data,
                    )?);
                }
            }
        }
    }
    if report.packed.is_empty() {
        return Err("the policy packed no projection; refusing to write a dense file".into());
    }
    report.packed.sort();
    report.kept.sort_by(|a, b| a.0.cmp(&b.0));
    report.tensors = tensors.len();
    report.output_bytes = tensors.iter().map(|t| t.data.len() as u64).sum();
    if let Some(parent) = out_file.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    w4a8_int8::write_checkpoint(&tensors, out_file)?;
    Ok(report)
}

fn read_payload(file: &mut std::fs::File, offset: u64, len: u64) -> std::io::Result<Vec<u8>> {
    use std::io::{Read, Seek, SeekFrom};
    let mut data = vec![0_u8; len as usize];
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(&mut data)?;
    Ok(data)
}

/// Widen a stored BF16 / F32 weight to F32 (exact for both).
fn to_f32(name: &str, dtype: Dtype, data: &[u8]) -> Result<Vec<f32>, String> {
    match dtype {
        Dtype::BF16 => Ok(data
            .chunks_exact(2)
            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
            .collect()),
        Dtype::F32 => Ok(data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()),
        other => Err(format!(
            "{name}: W4A8 packs a BF16 or F32 master weight, found {other:?}"
        )),
    }
}

/// [`w4a8_int8::quantize_weight`] over contiguous row blocks on `threads` workers, concatenated in
/// row order. Rows are independent and the codebook is fixed, so this is byte-identical to one
/// call over the whole matrix (pinned by `parallel_pack_is_byte_identical_to_one_call`).
pub fn pack_parallel(
    weight: &[f32],
    out: usize,
    inp: usize,
    threads: usize,
) -> Result<W4A8Packed, String> {
    let threads = threads.clamp(1, out.max(1));
    let rows_per = out.div_ceil(threads);
    let chunks: Vec<Result<W4A8Packed, String>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..out)
            .step_by(rows_per)
            .map(|r0| {
                let r1 = (r0 + rows_per).min(out);
                let rows = &weight[r0 * inp..r1 * inp];
                scope.spawn(move || {
                    w4a8_int8::quantize_weight(rows, r1 - r0, inp).map_err(|e| e.to_string())
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| Err("pack worker panicked".into()))
            })
            .collect()
    });
    let mut packed = W4A8Packed {
        out_features: out,
        in_features: inp,
        packed: Vec::with_capacity(out * inp / 2),
        s_rel: Vec::with_capacity(out * inp / w4a8_int8::GROUP_SIZE),
        s_channel: Vec::with_capacity(out),
        codebook: w4a8_int8::DEFAULT_CODEBOOK,
    };
    for chunk in chunks {
        let chunk = chunk?;
        packed.packed.extend_from_slice(&chunk.packed);
        packed.s_rel.extend_from_slice(&chunk.s_rel);
        packed.s_channel.extend_from_slice(&chunk.s_channel);
    }
    Ok(packed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pseudo(n: usize, seed: u64) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let mut z = (i as u64)
                    .wrapping_add(seed)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15);
                z ^= z >> 29;
                z = z.wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z ^= z >> 32;
                (z as f64 / u64::MAX as f64) as f32 - 0.5
            })
            .collect()
    }

    #[test]
    fn admission_is_cuda_sm80_plus_and_backend_aware() {
        assert_eq!(check_w4a8_tier(Some((8, 0)), "cuda"), Ok(()));
        assert_eq!(check_w4a8_tier(Some((12, 0)), "cuda"), Ok(()));
        let below = check_w4a8_tier(Some((7, 5)), "cuda").unwrap_err();
        assert!(matches!(below, W4A8TierUnavailable::BelowFloor(r) if r.found == (7, 5)));
        assert!(below.to_string().contains("sm_80"), "{below}");
        let cpu = check_w4a8_tier(None, "cpu").unwrap_err();
        assert_eq!(cpu, W4A8TierUnavailable::NotCuda { backend: "cpu" });
        assert!(cpu.to_string().contains("cpu backend"), "{cpu}");
        let metal = check_w4a8_tier(None, "metal").unwrap_err();
        assert!(metal.to_string().contains("metal backend"), "{metal}");
    }

    /// The mixed policy is the NVFP4 role table, verbatim: interior normalized-input leaves pack,
    /// every class the table guards stays dense, and the geometry gate applies to both policies.
    #[test]
    fn mixed_policy_follows_the_role_table_and_geometry() {
        let m = |base: &str, out, inp| w4a8_role(W4A8Policy::Mixed, base, out, inp, 28);
        assert_eq!(m("transformer_blocks.4.attn.to_q", 6144, 6144), Ok(()));
        assert_eq!(m("transformer_blocks.26.ff.up", 16384, 6144), Ok(()));
        assert_eq!(m("transformer_blocks.10.attn.to_k", 1536, 6144), Ok(()));
        for (base, reason) in [
            ("transformer_blocks.3.attn.to_q", DenseReason::EdgeBlock),
            ("transformer_blocks.27.ff.gate", DenseReason::EdgeBlock),
            (
                "transformer_blocks.10.ff.down",
                DenseReason::PostNonlinearity,
            ),
            (
                "transformer_blocks.10.attn.to_out.0",
                DenseReason::PostNonlinearity,
            ),
            (
                "text_fusion.refiner_blocks.0.attn.to_q",
                DenseReason::ContextRead,
            ),
            ("txt_in.linear_2", DenseReason::ContextRead),
            ("final_layer.linear", DenseReason::TrunkHead),
            ("time_embed.linear_2", DenseReason::Unclassified),
            ("time_mod_proj", DenseReason::Unclassified),
        ] {
            assert_eq!(m(base, 6144, 6144), Err(KeepReason::Role(reason)), "{base}");
        }
        assert_eq!(m("img_in", 6144, 64), Err(KeepReason::Geometry));
        let b = |base: &str, out, inp| w4a8_role(W4A8Policy::Blanket, base, out, inp, 28);
        assert_eq!(b("transformer_blocks.0.ff.down", 6144, 16384), Ok(()));
        assert_eq!(b("text_fusion.projector", 1, 12), Err(KeepReason::Geometry));
    }

    #[test]
    fn parallel_pack_is_byte_identical_to_one_call() {
        let (out, inp) = (48, 512);
        let w = pseudo(out * inp, 7);
        let one = w4a8_int8::quantize_weight(&w, out, inp).unwrap();
        for threads in [1, 5, 48, 64] {
            assert_eq!(
                pack_parallel(&w, out, inp, threads).unwrap(),
                one,
                "{threads}"
            );
        }
    }

    /// A tiny dense "transformer/" packs into a native-keyed file whose W4A8 projections decode
    /// through gen-core's reader, whose dense tensors are byte-identical copies, whose block
    /// modulation table is stored flat, and which the delegation splits exactly.
    #[test]
    fn builds_a_native_keyed_tier_file() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("transformer");
        std::fs::create_dir_all(&src).unwrap();
        let bf16 = |v: &[f32]| -> Vec<u8> {
            v.iter()
                .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
                .collect()
        };
        let q = pseudo(32 * 256, 1);
        let down = pseudo(32 * 256, 2);
        let table = pseudo(6 * 4, 3);
        let tensor = |name: &str, dtype: Dtype, shape: Vec<usize>, data: Vec<u8>| {
            W4A8CheckpointTensor::passthrough(name, dtype, shape, data).unwrap()
        };
        let source = vec![
            tensor(
                "transformer_blocks.4.attn.to_q.weight",
                Dtype::BF16,
                vec![32, 256],
                bf16(&q),
            ),
            tensor(
                "transformer_blocks.4.ff.down.weight",
                Dtype::BF16,
                vec![32, 256],
                bf16(&down),
            ),
            tensor(
                "transformer_blocks.4.scale_shift_table",
                Dtype::F32,
                vec![6, 4],
                table.iter().flat_map(|x| x.to_le_bytes()).collect(),
            ),
        ];
        w4a8_int8::write_checkpoint(&source, &src.join("diffusion_pytorch_model.safetensors"))
            .unwrap();
        let out = dir.path().join("out").join("krea2_turbo_w4a8.safetensors");
        let report = build_w4a8_tier(&src, 28, W4A8Policy::Mixed, &out, 3).unwrap();
        assert_eq!(
            report.packed,
            vec![("transformer_blocks.4.attn.to_q".to_owned(), 32, 256)]
        );
        assert_eq!(
            report.kept,
            vec![(
                "transformer_blocks.4.ff.down".to_owned(),
                KeepReason::Role(DenseReason::PostNonlinearity)
            )]
        );

        let payloads =
            candle_gen::gen_core::read_safetensors_tensor_payloads(&out, |_| true, 1 << 20)
                .unwrap();
        let reference_rows: Vec<f32> = q
            .iter()
            .map(|x| f32::from_bits(x.to_bits() & 0xFFFF_0000))
            .collect();
        let expected = w4a8_int8::quantize_weight(&reference_rows, 32, 256).unwrap();
        for t in expected.checkpoint_tensors("blocks.4.attn.wq") {
            assert_eq!(
                payloads[&t.name], t.data,
                "{}: the tier stores exactly the packer's output for the bf16 master",
                t.name
            );
        }
        assert_eq!(payloads["blocks.4.mlp.down.weight"], bf16(&down));
        let headers = candle_gen::gen_core::safetensors_path_tensor_headers(&out).unwrap();
        let descriptors = candle_gen::gen_core::read_safetensors_tensor_payloads(
            &out,
            |h| h.name.ends_with(".comfy_quant"),
            4096,
        )
        .unwrap();
        let delegation = w4a8_int8::delegate_projections(&headers, &descriptors).unwrap();
        assert_eq!(
            delegation.projections.keys().collect::<Vec<_>>(),
            ["blocks.4.attn.wq"]
        );
        assert_eq!(delegation.keys.len(), 5);
        assert_eq!(
            headers.len(),
            7,
            "5 W4A8 tensors + the kept weight + the table"
        );
        let table_header = headers
            .iter()
            .find(|h| h.name == "blocks.4.mod.lin")
            .unwrap();
        assert_eq!(
            table_header.shape,
            [24],
            "the block modulation table is stored flat"
        );
    }

    #[test]
    fn refuses_a_quantized_source() {
        let dir = tempfile::tempdir().unwrap();
        let packed = w4a8_int8::quantize_weight(&pseudo(32 * 256, 4), 32, 256).unwrap();
        w4a8_int8::write_checkpoint(
            &packed.checkpoint_tensors("transformer_blocks.4.attn.to_q"),
            &dir.path().join("t.safetensors"),
        )
        .unwrap();
        let err = build_w4a8_tier(
            dir.path(),
            28,
            W4A8Policy::Mixed,
            &dir.path().join("o.safetensors"),
            1,
        )
        .unwrap_err();
        assert!(err.contains("dense bf16 master"), "{err}");
    }
}
