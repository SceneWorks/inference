//! Native **GGUF k-quant** loader for the Wan DiT (sc-12735, Pillar 2 of epic 12732 — the 24 GB lever).
//!
//! Where the MLX-packed path ([`crate::quant`], sc-10025) reads a `SceneWorks/wan2.2-*-mlx` tier's affine
//! `.scales`/`.biases` triple, and the ComfyUI scaled-fp8 seam ([`crate::comfyui`]) *pre-dequantizes* to a
//! dense bf16 map, this module holds the DiT as **resident Q4_K_M [`QTensor`]s and dequantizes per-block
//! per-matmul** — the ComfyUI-GGUF posture. A `QuantStack/Wan2.2-TI2V-5B-GGUF` `.gguf` is opened with
//! candle's native reader ([`candle_gen::candle_core::quantized::gguf_file`], k-quant CUDA-supported in the
//! SceneWorks candle pin), and **every k-quant weight stays quantized-resident**: the dequant happens on
//! the matmul ([`candle_gen::quant::QLinear::from_qtensor_dequant`], the sc-7702-safe
//! [`candle_gen::quant::MatmulStrategy::DequantDense`] forward), **never at load**. That is the whole win —
//! copying the ComfyUI seam's `from_tensors(dense bf16)` naively would erase it.
//!
//! ## Scope (sub-story 1)
//!
//! The loader mechanism, proven on the **5B** (single dense DiT — the simplest vehicle). The manifest /
//! catalog / tier routing (sub-story 2), the A14B dual-expert GGUF (sub-story 3), and a GGUF text encoder
//! (sub-story 4) are **separate** stories. The 5B GGUF path is selected here by the
//! [`env_gguf_path`] test seam (an env var pointing at a downloaded `.gguf`), NOT by the manifest.
//!
//! ## The two transforms (shared with the ComfyUI seam)
//!
//! 1. **Native-Wan → diffusers key remap** — QuantStack GGUF ships the **native-Wan** tensor names
//!    (`blocks.N.self_attn.q`, `cross_attn`, `ffn.0/2`, `modulation`, `norm3`, `head.head`,
//!    `text_embedding.0/2`, `time_projection.1`); the loader reuses the ONE
//!    [`crate::comfyui::remap_wan_key`] rename to the diffusers schema [`crate::transformer::WanTransformer`]
//!    reads.
//! 2. **Resident k-quant vs dense sidecar split** — the attention/FFN/embedder/`proj_out` Linears are
//!    k-quant `QTensor`s held resident; the dense sidecars (norms, biases, `modulation`, `patch_embedding`,
//!    `scale_shift_table`) are the GGUF's F16/F32 blocks, dequantized on read to the DiT compute dtype.
//!
//! The build routes through the SAME [`WeightSrc`] the dense path uses (so `WanTransformer::new` and
//! `WanTransformer::from_gguf` share every shape rule), and the resulting DiT reports `is_packed()` — it
//! drops into the sc-12757 sequential residency as the (now quantized-resident) denoise component,
//! unchanged staging.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_gen::candle_core::quantized::{gguf_file, QTensor};
use candle_gen::candle_core::{Device, Error as CError, Result as CResult, Shape, Tensor};
use candle_gen::candle_nn::VarBuilder;

use crate::comfyui::remap_wan_key;
use crate::config::TransformerConfig;
use crate::quant::QLinear;
use crate::transformer::WanTransformer;

/// The env var the sub-story-1 test seam reads: an absolute path to a `QuantStack/Wan2.2-TI2V-5B-GGUF`
/// `.gguf` file. When set (and non-empty), [`crate::Pipeline::build_dit`] builds the 5B DiT natively from
/// its k-quant `QTensor`s (this module) instead of the snapshot's `transformer/`. This is the deliberate
/// minimal seam so the loader can be GPU-validated **without** the manifest/catalog wiring (sub-story 2).
pub(crate) const GGUF_ENV: &str = "CANDLE_GEN_WAN_GGUF";

/// The 5B-GGUF path selected by the [`GGUF_ENV`] test seam (`None` ⇒ the normal snapshot path). A
/// present-but-empty value is treated as unset. Clearly marked as the sub-story-1 seam — sub-story 2
/// replaces this env probe with manifest/catalog tier routing.
pub(crate) fn env_gguf_path() -> Option<PathBuf> {
    match std::env::var(GGUF_ENV) {
        Ok(v) if !v.trim().is_empty() => Some(PathBuf::from(v)),
        _ => None,
    }
}

/// Open a Wan DiT `.gguf` and build the [`WanTransformer`] with its k-quant weights held
/// **quantized-resident** (sc-12735). The entry point [`crate::Pipeline::build_dit`] calls on the GGUF seam.
pub(crate) fn load_wan_dit_gguf(
    path: &Path,
    cfg: &TransformerConfig,
    device: &Device,
    dtype: candle_gen::candle_core::DType,
) -> CResult<WanTransformer> {
    let dit = GgufDit::open(path, device, dtype)?;
    WanTransformer::from_gguf(cfg, &dit)
}

/// A Wan DiT `.gguf` parsed into **resident** [`QTensor`]s, diffusers-keyed (post native→diffusers remap).
/// k-quant Linear weights (`Q4_K` etc.) are held quantized — NEVER dequantized to a dense `[out,in]` weight
/// at load. Dense sidecars (norms, biases, `modulation`, `patch_embedding`, `scale_shift_table`) are the
/// GGUF's F16/F32 blocks, dequantized on read.
pub(crate) struct GgufDit {
    /// diffusers-key → resident GGUF tensor (k-quant Linear weight, or an F16/F32 dense sidecar block).
    tensors: HashMap<String, Arc<QTensor>>,
    device: Device,
    /// The DiT compute dtype (bf16) dense sidecars are cast to on read — matching the dense path's
    /// `VarBuilder::get` cast, so the GGUF and snapshot builds agree tensor-for-tensor on the sidecars.
    dtype: candle_gen::candle_core::DType,
}

impl GgufDit {
    /// Open `path`, read **every** tensor resident (`gguf_file::Content::tensor`, no dequant), and remap
    /// each native-Wan name to its diffusers key ([`remap_wan_key`]). The rope `freqs` buffer (which the
    /// DiT recomputes) is dropped. Tensors are read in sorted order so any error message is stable.
    pub(crate) fn open(
        path: &Path,
        device: &Device,
        dtype: candle_gen::candle_core::DType,
    ) -> CResult<Self> {
        let mut file = std::fs::File::open(path)
            .map_err(|e| CError::msg(format!("wan gguf: open {}: {e}", path.display())))?;
        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| CError::msg(format!("wan gguf: parse {}: {e}", path.display())))?;
        let mut names: Vec<String> = content.tensor_infos.keys().cloned().collect();
        names.sort();
        let mut tensors: HashMap<String, Arc<QTensor>> = HashMap::with_capacity(names.len());
        for name in &names {
            // The DiT derives RoPE from theta; a precomputed `freqs` buffer (present on some exports) is
            // dropped rather than mapped, mirroring the ComfyUI seam.
            if name == "freqs" || name.ends_with(".freqs") {
                continue;
            }
            let qt = content
                .tensor(&mut file, name, device)
                .map_err(|e| CError::msg(format!("wan gguf: read tensor {name:?}: {e}")))?;
            let key = remap_wan_key(name);
            if tensors.insert(key.clone(), Arc::new(qt)).is_some() {
                // Two native keys colliding on one diffusers key is a malformed file — surface it.
                return Err(CError::msg(format!(
                    "wan gguf: duplicate diffusers key {key:?} (two native tensors remapped onto it)"
                )));
            }
        }
        Ok(Self {
            tensors,
            device: device.clone(),
            dtype,
        })
    }

    /// The resident tensor at diffusers `key`, or a **loud** error naming the missing key (a renamed /
    /// absent tensor must fail the load, not silently degrade — the sc-12735 "fail loudly" contract).
    fn require(&self, key: &str) -> CResult<&Arc<QTensor>> {
        self.tensors.get(key).ok_or_else(|| {
            CError::msg(format!(
                "wan gguf: missing tensor {key:?} — the 5B DiT expects it (renamed/absent GGUF key, or \
                 an unmapped native-Wan name); a native→diffusers remap gap fails the load here"
            ))
        })
    }

    /// Build a **resident-QTensor** [`QLinear`] for `{base}` from the k-quant `{base}.weight` (held
    /// quantized) plus the optional dense `{base}.bias`. The QTensor is shared by `Arc` (no copy); the
    /// forward dequantizes it per-matmul (the sc-7702-safe [`candle_gen::quant::MatmulStrategy::DequantDense`]
    /// path). A `[out, in]` shape mismatch is a loud error (a wrong-dim GGUF, never silent garbage).
    fn qlinear(&self, base: &str, in_dim: usize, out_dim: usize, bias: bool) -> CResult<QLinear> {
        let wkey = format!("{base}.weight");
        let qt = self.require(&wkey)?.clone();
        let dims = qt.shape().dims();
        if dims != [out_dim, in_dim] {
            return Err(CError::msg(format!(
                "wan gguf: {wkey:?} shape {dims:?} != expected [{out_dim}, {in_dim}]"
            )));
        }
        let bias = if bias {
            Some(self.dense(&format!("{base}.bias"), Shape::from((out_dim,)))?)
        } else {
            None
        };
        // Ingest the resident k-quant QTensor WITHOUT dequantizing to dense (the whole point), then wrap
        // it as the packed base of an `AdaptLinear` so `transformer.rs` keeps calling `QLinear` unchanged.
        let base = candle_gen::quant::QLinear::from_qtensor_dequant(qt, bias);
        Ok(QLinear::from_packed(base, in_dim, out_dim))
    }

    /// Dequantize a dense sidecar `{key}` (an F16/F32 GGUF block) to the DiT compute dtype, verifying its
    /// element count matches `shape` (reshaping when the block's logical shape differs but the count
    /// agrees — e.g. a flattened bias). Mirrors the dense path's `VarBuilder::get(shape, key)` (which also
    /// shape-checks + casts to the builder dtype), so the two builds agree on every sidecar.
    fn dense(&self, key: &str, shape: Shape) -> CResult<Tensor> {
        let qt = self.require(key)?;
        let t = qt.dequantize(&self.device)?.to_dtype(self.dtype)?;
        if t.dims() == shape.dims() {
            Ok(t)
        } else if t.elem_count() == shape.elem_count() {
            t.reshape(shape)
        } else {
            Err(CError::msg(format!(
                "wan gguf: {key:?} element count {} != expected {} (shape {:?} vs {:?})",
                t.elem_count(),
                shape.elem_count(),
                t.dims(),
                shape.dims()
            )))
        }
    }
}

/// The weight source a [`WanTransformer`] build reads from: a **dense** [`VarBuilder`] (a snapshot or an
/// MLX-packed tier, the unchanged path) or a **native-GGUF** [`GgufDit`] (resident k-quant, sc-12735). It
/// unifies the two so `WanTransformer::new` and `WanTransformer::from_gguf` share every shape rule — the
/// dense arm forwards to the exact `VarBuilder`/`QLinear::linear_detect` calls as before (byte-identical),
/// the GGUF arm routes each Linear through [`GgufDit::qlinear`] (resident QTensor) and each sidecar through
/// [`GgufDit::dense`].
pub(crate) enum WeightSrc<'a> {
    /// The dense / MLX-packed path (unchanged) — packed-detects `.scales` per [`QLinear::linear_detect`].
    Dense(VarBuilder<'a>),
    /// The native-GGUF k-quant path (sc-12735) — a resident [`GgufDit`] plus the current dotted key prefix.
    Gguf { dit: &'a GgufDit, prefix: String },
}

impl<'a> WeightSrc<'a> {
    /// The dense / MLX-packed source over `vb` (the unchanged path).
    pub(crate) fn dense(vb: VarBuilder<'a>) -> Self {
        Self::Dense(vb)
    }

    /// The native-GGUF source at the root prefix (sc-12735).
    pub(crate) fn gguf(dit: &'a GgufDit) -> Self {
        Self::Gguf {
            dit,
            prefix: String::new(),
        }
    }

    /// A sub-scope under `seg` (the [`VarBuilder::pp`] analogue) — appends `seg.` to the key prefix so a
    /// `to_out.0`-style nesting survives on both arms.
    pub(crate) fn pp(&self, seg: impl std::fmt::Display) -> WeightSrc<'a> {
        match self {
            Self::Dense(vb) => Self::Dense(vb.pp(seg.to_string())),
            Self::Gguf { dit, prefix } => Self::Gguf {
                dit,
                prefix: format!("{prefix}{seg}."),
            },
        }
    }

    /// The device this source builds on.
    pub(crate) fn device(&self) -> Device {
        match self {
            Self::Dense(vb) => vb.device().clone(),
            Self::Gguf { dit, .. } => dit.device.clone(),
        }
    }

    /// The compute dtype (bf16 for the DiT).
    pub(crate) fn dtype(&self) -> candle_gen::candle_core::DType {
        match self {
            Self::Dense(vb) => vb.dtype(),
            Self::Gguf { dit, .. } => dit.dtype,
        }
    }

    /// A dense tensor `{key}` (relative to this scope) at the source dtype — a norm / bias / modulation /
    /// `patch_embedding` / `scale_shift_table` sidecar. Dense arm: `vb.get`; GGUF arm: [`GgufDit::dense`].
    pub(crate) fn get(&self, shape: impl Into<Shape>, key: &str) -> CResult<Tensor> {
        match self {
            Self::Dense(vb) => vb.get(shape, key),
            Self::Gguf { dit, prefix } => dit.dense(&format!("{prefix}{key}"), shape.into()),
        }
    }

    /// A [`QLinear`] for `{base}` (relative to this scope). Dense arm: [`QLinear::linear_detect`]
    /// (packed-detecting, unchanged); GGUF arm: [`GgufDit::qlinear`] (resident k-quant QTensor).
    pub(crate) fn qlinear(
        &self,
        in_dim: usize,
        out_dim: usize,
        base: &str,
        bias: bool,
    ) -> CResult<QLinear> {
        match self {
            Self::Dense(vb) => QLinear::linear_detect(in_dim, out_dim, vb, base, bias),
            Self::Gguf { dit, prefix } => {
                dit.qlinear(&format!("{prefix}{base}"), in_dim, out_dim, bias)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rope::WanRope;
    use candle_gen::candle_core::quantized::{gguf_file, GgmlDType};
    use candle_gen::candle_core::{DType, Device};
    use candle_gen::quant::MatmulStrategy;

    /// A small config whose every Linear contraction (`in`) is a multiple of the Q4_K block (256), so a
    /// real k-quant fixture can be written: dim 256 (= 2 heads × 128), ffn 512, freq/text 256. 2 blocks.
    fn gguf_cfg() -> TransformerConfig {
        TransformerConfig {
            in_channels: 4,
            out_channels: 4,
            num_layers: 2,
            num_heads: 2,
            head_dim: 128,
            dim: 256,
            ffn_dim: 512,
            freq_dim: 256,
            text_dim: 256,
            patch: (1, 2, 2),
            eps: 1e-6,
            rope_theta: 10000.0,
            rope_max_seq_len: 64,
        }
    }

    /// A k-quant `QTensor` of shape `[out, in]` (`in` must be a multiple of 256) from a deterministic
    /// small-magnitude grid — the resident weight the loader must keep quantized.
    fn q4k(out: usize, inn: usize) -> QTensor {
        let data: Vec<f32> = (0..out * inn)
            .map(|i| ((i % 17) as f32 / 17.0 - 0.5) * 0.1)
            .collect();
        let w = Tensor::from_vec(data, (out, inn), &Device::Cpu).unwrap();
        QTensor::quantize(&w, GgmlDType::Q4K).unwrap()
    }

    /// A dense (F32-block) `QTensor` of `shape` — the sidecar form (norms / biases / modulation /
    /// `patch_embedding` / `scale_shift_table`), stored uncompressed so it dequantizes back exactly.
    fn dense_qt(shape: &[usize]) -> QTensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013).sin() * 0.2).collect();
        let t = Tensor::from_vec(data, shape, &Device::Cpu).unwrap();
        QTensor::quantize(&t, GgmlDType::F32).unwrap()
    }

    /// Emit the FULL **native-Wan** keyed tensor set a `WanTransformer` of `cfg` needs — Q4_K Linear
    /// weights + F32 dense sidecars/biases — so the fixture exercises [`remap_wan_key`] end-to-end (the
    /// same key layout a real `QuantStack/Wan2.2-*-GGUF` ships). Returned as an owned map so the caller
    /// can borrow `(&str, &QTensor)` pairs for [`gguf_file::write`].
    fn native_wan_tensors(cfg: &TransformerConfig) -> HashMap<String, QTensor> {
        let d = cfg.dim;
        let (pt, ph, pw) = cfg.patch;
        let mut m: HashMap<String, QTensor> = HashMap::new();
        // Linear weight (Q4_K) + dense bias, under a native base key.
        let lin = |m: &mut HashMap<String, QTensor>, base: &str, out: usize, inn: usize| {
            m.insert(format!("{base}.weight"), q4k(out, inn));
            m.insert(format!("{base}.bias"), dense_qt(&[out]));
        };
        // Top-level embedders + head (native names).
        lin(&mut m, "text_embedding.0", d, cfg.text_dim);
        lin(&mut m, "text_embedding.2", d, d);
        lin(&mut m, "time_embedding.0", d, cfg.freq_dim);
        lin(&mut m, "time_embedding.2", d, d);
        lin(&mut m, "time_projection.1", 6 * d, d);
        lin(&mut m, "head.head", cfg.out_channels * pt * ph * pw, d);
        m.insert("head.modulation".into(), dense_qt(&[1, 2, d]));
        // patch_embedding (native == diffusers key), a 5-D conv weight + bias, dense.
        m.insert(
            "patch_embedding.weight".into(),
            dense_qt(&[d, cfg.in_channels, pt, ph, pw]),
        );
        m.insert("patch_embedding.bias".into(), dense_qt(&[d]));
        // Per-block (native names): self/cross attn q/k/v/o (+ norm_q/k), ffn.0/2, norm3, modulation.
        for i in 0..cfg.num_layers {
            let b = format!("blocks.{i}");
            for attn in ["self_attn", "cross_attn"] {
                for leaf in ["q", "k", "v", "o"] {
                    lin(&mut m, &format!("{b}.{attn}.{leaf}"), d, d);
                }
                m.insert(format!("{b}.{attn}.norm_q.weight"), dense_qt(&[d]));
                m.insert(format!("{b}.{attn}.norm_k.weight"), dense_qt(&[d]));
            }
            lin(&mut m, &format!("{b}.ffn.0"), cfg.ffn_dim, d);
            lin(&mut m, &format!("{b}.ffn.2"), d, cfg.ffn_dim);
            m.insert(format!("{b}.norm3.weight"), dense_qt(&[d]));
            m.insert(format!("{b}.norm3.bias"), dense_qt(&[d]));
            m.insert(format!("{b}.modulation"), dense_qt(&[1, 6, d]));
        }
        m
    }

    /// Write `tensors` to a fresh `.gguf` at a unique temp path and return it.
    fn write_gguf(tensors: &HashMap<String, QTensor>, tag: &str) -> PathBuf {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let uniq = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "sc12735_{tag}_{}_{}.gguf",
            std::process::id(),
            uniq
        ));
        let refs: Vec<(&str, &QTensor)> = tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();
        let mut f = std::fs::File::create(&path).unwrap();
        // A minimal metadata table (the DiT config comes from `TransformerConfig`, not the GGUF here).
        let arch = gguf_file::Value::String("wan".to_string());
        gguf_file::write(&mut f, &[("general.architecture", &arch)], &refs).unwrap();
        path
    }

    // ── remap coverage: every Linear the 5B DiT expects (native → diffusers) ──────────────────────────

    /// [`remap_wan_key`] covers **every** Linear (and its bias) the 5B DiT reads — attention q/k/v/out,
    /// FFN, the condition embedders, `time_proj`, `proj_out`, `scale_shift_table` — for a native-Wan GGUF.
    /// A gap here would surface as a missing-tensor load error; pinning it as a pure-function test makes a
    /// remap regression fail loudly and locally.
    #[test]
    fn remap_covers_every_5b_linear() {
        // attention (self → attn1, cross → attn2) weights + biases + qk-norms
        assert_eq!(
            remap_wan_key("blocks.7.self_attn.q.weight"),
            "blocks.7.attn1.to_q.weight"
        );
        assert_eq!(
            remap_wan_key("blocks.7.self_attn.k.bias"),
            "blocks.7.attn1.to_k.bias"
        );
        assert_eq!(
            remap_wan_key("blocks.7.self_attn.v.weight"),
            "blocks.7.attn1.to_v.weight"
        );
        assert_eq!(
            remap_wan_key("blocks.7.self_attn.o.weight"),
            "blocks.7.attn1.to_out.0.weight"
        );
        assert_eq!(
            remap_wan_key("blocks.7.self_attn.norm_q.weight"),
            "blocks.7.attn1.norm_q.weight"
        );
        assert_eq!(
            remap_wan_key("blocks.7.cross_attn.q.weight"),
            "blocks.7.attn2.to_q.weight"
        );
        assert_eq!(
            remap_wan_key("blocks.7.cross_attn.o.bias"),
            "blocks.7.attn2.to_out.0.bias"
        );
        assert_eq!(
            remap_wan_key("blocks.7.cross_attn.norm_k.weight"),
            "blocks.7.attn2.norm_k.weight"
        );
        // ffn + norm3 + block modulation
        assert_eq!(
            remap_wan_key("blocks.7.ffn.0.weight"),
            "blocks.7.ffn.net.0.proj.weight"
        );
        assert_eq!(
            remap_wan_key("blocks.7.ffn.2.bias"),
            "blocks.7.ffn.net.2.bias"
        );
        assert_eq!(
            remap_wan_key("blocks.7.norm3.weight"),
            "blocks.7.norm2.weight"
        );
        assert_eq!(
            remap_wan_key("blocks.7.modulation"),
            "blocks.7.scale_shift_table"
        );
        // top-level embedders / head
        assert_eq!(
            remap_wan_key("text_embedding.0.weight"),
            "condition_embedder.text_embedder.linear_1.weight"
        );
        assert_eq!(
            remap_wan_key("text_embedding.2.bias"),
            "condition_embedder.text_embedder.linear_2.bias"
        );
        assert_eq!(
            remap_wan_key("time_embedding.0.weight"),
            "condition_embedder.time_embedder.linear_1.weight"
        );
        assert_eq!(
            remap_wan_key("time_embedding.2.weight"),
            "condition_embedder.time_embedder.linear_2.weight"
        );
        assert_eq!(
            remap_wan_key("time_projection.1.weight"),
            "condition_embedder.time_proj.weight"
        );
        assert_eq!(remap_wan_key("head.head.weight"), "proj_out.weight");
        assert_eq!(remap_wan_key("head.modulation"), "scale_shift_table");
        // dense sidecar that must pass through unchanged
        assert_eq!(
            remap_wan_key("patch_embedding.weight"),
            "patch_embedding.weight"
        );
    }

    // ── loader: resident (NOT dense), remap applied, missing key fails loud ───────────────────────────

    /// A native-keyed `.gguf` opens with each Linear held as a **resident k-quant `QTensor`** (Q4_K, not a
    /// dense `[out,in]` weight), reachable at its **diffusers** key (the remap fired), and the bias is a
    /// dense companion. The resident-not-dense guarantee — a naive `dequantize`-at-load would fail this.
    #[test]
    fn loader_keeps_kquant_resident_and_remaps() {
        let cfg = gguf_cfg();
        let tensors = native_wan_tensors(&cfg);
        let path = write_gguf(&tensors, "resident");
        let dit = GgufDit::open(&path, &Device::Cpu, DType::BF16).unwrap();

        // The native `blocks.0.self_attn.q` is reachable at the diffusers `blocks.0.attn1.to_q`.
        let q = dit
            .qlinear("blocks.0.attn1.to_q", cfg.dim, cfg.dim, true)
            .expect("remapped diffusers key resolves");
        assert!(
            q.is_packed(),
            "the k-quant weight must load quantized-resident, not dense"
        );
        let inner = q.base_qlinear().expect("packed base exposes the QLinear");
        assert_eq!(
            inner.quant_dtype(),
            Some(GgmlDType::Q4K),
            "the resident weight must stay Q4_K (NOT dequantized to a dense [out,in] at load)"
        );
        assert_eq!(
            inner.matmul_strategy(),
            Some(MatmulStrategy::DequantDense),
            "the forward must dequant-on-matmul (sc-7702-safe), not the int8-fast path"
        );
        std::fs::remove_file(&path).ok();
    }

    /// A missing / renamed key fails the load **loudly** (naming the key), never a silent dense fallback —
    /// the sc-12735 fail-loud contract. Drop `blocks.0.self_attn.q.weight` and the diffusers lookup errors.
    #[test]
    fn loader_missing_key_fails_loud() {
        let cfg = gguf_cfg();
        let mut tensors = native_wan_tensors(&cfg);
        tensors.remove("blocks.0.self_attn.q.weight");
        let path = write_gguf(&tensors, "missing");
        let dit = GgufDit::open(&path, &Device::Cpu, DType::BF16).unwrap();
        // `AdaptLinear` isn't `Debug`, so match rather than `expect_err`.
        let err = match dit.qlinear("blocks.0.attn1.to_q", cfg.dim, cfg.dim, true) {
            Ok(_) => panic!("a missing weight must error, not silently degrade"),
            Err(e) => e,
        };
        let msg = format!("{err}");
        assert!(
            msg.contains("blocks.0.attn1.to_q.weight") && msg.contains("missing"),
            "the error must name the missing diffusers key: {msg}"
        );
        std::fs::remove_file(&path).ok();
    }

    // ── integration: the full WanTransformer::from_gguf build is entirely resident + runs a forward ───

    /// `WanTransformer::from_gguf` builds the whole 5B-shaped DiT with **every** adaptable projection
    /// packed-resident at `Q4_K` (walked via `visit_adaptable_mut`) — no projection accidentally
    /// dequantized to dense at load — reports `is_packed()`, and produces a finite velocity on a CPU
    /// forward (the dequant-on-matmul path executes end-to-end).
    #[test]
    fn from_gguf_builds_fully_resident_dit_and_forwards() {
        let cfg = gguf_cfg();
        let tensors = native_wan_tensors(&cfg);
        let path = write_gguf(&tensors, "full");
        // F32 compute dtype so the CPU forward runs (CPU has no bf16 matmul); the k-quant weights stay
        // Q4_K resident regardless of the compute dtype — only the activation/sidecar dtype changes. On
        // CUDA the production path uses bf16 (DIT_DTYPE); this asserts the resident-QTensor mechanism.
        let dit = GgufDit::open(&path, &Device::Cpu, DType::F32).unwrap();
        let mut model = WanTransformer::from_gguf(&cfg, &dit).expect("from_gguf builds");
        assert!(
            model.is_packed(),
            "a GGUF DiT must report packed (is_packed)"
        );

        // Every adaptable projection is a resident Q4_K base — none dequantized to dense at load.
        let mut count = 0usize;
        model
            .visit_adaptable_mut(&mut |path, ql| {
                assert!(ql.is_packed(), "{path} must be packed-resident");
                let inner = ql.base_qlinear().expect("packed base");
                assert_eq!(
                    inner.quant_dtype(),
                    Some(GgmlDType::Q4K),
                    "{path} must stay Q4_K resident (not dense) after load"
                );
                count += 1;
                Ok(())
            })
            .unwrap();
        // 5 condition-embedder + num_layers×(4+4+2) block projections + proj_out.
        assert_eq!(
            count,
            5 + cfg.num_layers * 10 + 1,
            "every DiT Linear walked"
        );

        // A CPU forward exercises the dequant-on-matmul path end-to-end and stays finite.
        let latents =
            Tensor::randn(0f32, 1f32, (1, cfg.in_channels, 2, 4, 4), &Device::Cpu).unwrap();
        let context = Tensor::randn(0f32, 1f32, (1, 3, cfg.dim), &Device::Cpu).unwrap();
        let (cos, sin) = WanRope::new(&cfg).cos_sin(2, 2, 2, &Device::Cpu).unwrap();
        let vel = model
            .forward(&latents, &context, 700.0, &cos, &sin)
            .unwrap();
        assert_eq!(vel.dims(), &[1, cfg.out_channels, 2, 4, 4]);
        let max = vel
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            max.is_finite(),
            "GGUF-resident DiT forward must be finite (got {max})"
        );
        std::fs::remove_file(&path).ok();
    }
}
