//! Residual-capable linear for Anima's packed-tier LoRA path (sc-10640 — the candle twin of MLX
//! sc-10578). A frozen **base** (dense `candle_nn::Linear`, or an MLX-packed
//! [`candle_gen::quant::QLinear`] that dequantizes-on-forward, sc-7702) plus zero or more **forward-time
//! additive LoRA residuals** `scale·((x·A)·B)` (epic 10043, the candle mirror of
//! `candle-gen-wan/src/quant.rs`'s `QLinear` and of mlx-gen's `AdaptableLinear`).
//!
//! **The base weight is NEVER mutated.** On a packed q4/q8 DiT that is the whole point: the packed codes
//! survive load (`is_packed()` stays true, no dense `[out,in]` weight is materialized), and the adapter
//! rides *unmerged* as two small matmuls per forward — so a q4 tier keeps its q4 footprint. On a dense
//! base the fold path ([`crate::adapters::apply_anima_adapters`]) is still used instead, because a merge
//! into a real `.weight` is byte-for-byte what the fork-parity goldens expect. With **no** adapter
//! attached this type's forward is byte-identical to the pre-sc-10640 `Linear` / shared `QLinear` path,
//! so swapping it in for the DiT + conditioner projections leaves the plain-model / dense-fold paths
//! unchanged.

use candle_gen::candle_core::{DType, Tensor};
use candle_gen::candle_nn::{Linear, Module, VarBuilder};
use candle_gen::quant::{QLinear, MLX_GROUP_SIZE};
use candle_gen::Result;

/// The frozen base weight — **dense** (`candle_nn::Linear`) or **MLX-packed** ([`QLinear`], dequant-on-
/// forward). Both compute `x·Wᵀ (+ b)`; neither is ever mutated by an adapter.
enum Base {
    Dense(Linear),
    Packed(QLinear),
}

impl Base {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Base::Dense(l) => Ok(l.forward(x)?),
            Base::Packed(q) => Ok(q.forward(x)?),
        }
    }
}

/// A forward-time additive LoRA residual `scale·((x·a)·b)`: `a` `[in, rank]` (= `downᵀ`), `b`
/// `[rank, out]` (= `upᵀ` with the `alpha/rank` ratio already folded in). Two **small** matmuls — never
/// the `[out,in]` product — so it is memory-free on any quant. Factors are held **f32** (the merge/train
/// dtype) and cast to the activation dtype per forward (they are tiny, so the cast is cheap).
struct Adapter {
    a: Tensor,
    b: Tensor,
    scale: f64,
}

impl Adapter {
    /// The residual this adapter contributes, in the activation dtype of `x`.
    fn residual(&self, x: &Tensor) -> Result<Tensor> {
        let xd = x.dtype();
        let r = x
            .broadcast_matmul(&self.a.to_dtype(xd)?)?
            .broadcast_matmul(&self.b.to_dtype(xd)?)?;
        Ok((r * self.scale)?)
    }
}

/// A projection with a frozen base and stacked forward-time LoRA residuals. Built bias-less packed-
/// detecting ([`Self::detect`], the DiT projections), or dense with/without bias ([`Self::dense`] /
/// [`Self::dense_bias`], the conditioner projections). `forward` = `base(x)` plus every residual, in
/// push order.
pub struct AdaptLinear {
    base: Base,
    /// The projection's logical `(out_features, in_features)` — captured at construction (recoverable
    /// even from a packed base, where the dense weight is never materialized) so the residual installer
    /// can shape-check a factor without reading the quantized weight back.
    out_features: usize,
    in_features: usize,
    /// Forward-time additive residuals, applied in push order (adapters stack). Empty on the plain /
    /// dense-fold path ⇒ forward is byte-identical to the bare base.
    adapters: Vec<Adapter>,
}

impl AdaptLinear {
    /// Bias-less, **packed-detecting** `[out, in]` projection from `{name}` on `vb` — the DiT loader
    /// (port of the old `transformer::lin`). If `{name}.scales` is present, load the MLX-packed triple at
    /// their native dtypes (u32 codes must NOT be cast through the vb's float dtype) and repack at group
    /// 64; otherwise read the dense `{name}.weight` unchanged. The logical dims are recovered from the
    /// packed `scales` shape (`[out, in/group]`) on the packed arm, or the weight shape on the dense arm.
    pub fn detect(vb: &VarBuilder, name: &str) -> Result<Self> {
        let scales_key = format!("{name}.scales");
        if vb.contains_tensor(&scales_key) {
            let device = vb.device().clone();
            let wq = vb.get_unchecked_dtype(&format!("{name}.weight"), DType::U32)?;
            let scales = vb.get_unchecked_dtype(&scales_key, DType::F32)?;
            let biases = vb.get_unchecked_dtype(&format!("{name}.biases"), DType::F32)?;
            // scales is [out, in/group] — recover the logical dims without touching the packed codes.
            let sdims = scales.dims();
            let out_features = sdims[0];
            let in_features = sdims[1] * MLX_GROUP_SIZE;
            let q = QLinear::from_packed_gs(&wq, &scales, &biases, None, MLX_GROUP_SIZE, &device)?;
            Ok(Self {
                base: Base::Packed(q),
                out_features,
                in_features,
                adapters: Vec::new(),
            })
        } else {
            let w = vb.get_unchecked(&format!("{name}.weight"))?;
            let (out_features, in_features) = (w.dims()[0], w.dims()[1]);
            Ok(Self {
                base: Base::Dense(Linear::new(w, None)),
                out_features,
                in_features,
                adapters: Vec::new(),
            })
        }
    }

    /// Bias-less **dense** `[out, in]` projection from `{name}.weight` (the conditioner q/k/v/o — always
    /// dense bf16; Anima packs only the DiT). Port of the old `nn::lin`.
    pub fn dense(vb: &VarBuilder, name: &str) -> Result<Self> {
        let w = vb.get_unchecked(&format!("{name}.weight"))?;
        let (out_features, in_features) = (w.dims()[0], w.dims()[1]);
        Ok(Self {
            base: Base::Dense(Linear::new(w, None)),
            out_features,
            in_features,
            adapters: Vec::new(),
        })
    }

    /// **Dense** `[out, in]` projection with bias from `{name}.weight` + `{name}.bias` (the conditioner
    /// MLP + `out_proj`). Port of the old `nn::lin_bias`.
    pub fn dense_bias(vb: &VarBuilder, name: &str) -> Result<Self> {
        let w = vb.get_unchecked(&format!("{name}.weight"))?;
        let (out_features, in_features) = (w.dims()[0], w.dims()[1]);
        let b = vb.get_unchecked(&format!("{name}.bias"))?;
        Ok(Self {
            base: Base::Dense(Linear::new(w, Some(b))),
            out_features,
            in_features,
            adapters: Vec::new(),
        })
    }

    /// `x·Wᵀ (+ b)` plus every attached additive residual, in push order. With no adapter this is
    /// byte-identical to the bare base forward.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut y = self.base.forward(x)?;
        for ad in &self.adapters {
            y = (y + ad.residual(x)?)?;
        }
        Ok(y)
    }

    /// Whether the base loaded from an MLX-packed tier (its codes are quantized) — used to gate the
    /// residual path and asserted by the tests (packed survives load, no dense weight materialized).
    pub fn is_packed(&self) -> bool {
        matches!(self.base, Base::Packed(_))
    }

    /// The base projection's `(out_features, in_features)` — the shape a resolved LoRA factor is checked
    /// against, recoverable even from a packed base.
    pub fn base_shape(&self) -> (usize, usize) {
        (self.out_features, self.in_features)
    }

    /// Whether any additive residual is attached.
    pub fn is_adapted(&self) -> bool {
        !self.adapters.is_empty()
    }

    /// Attach a forward-time LoRA residual `scale·((x·a)·b)`: `a` `[in, rank]` (= `downᵀ`), `b`
    /// `[rank, out]` (= `upᵀ` with `alpha/rank` folded in), `scale` the caller's per-adapter strength.
    /// Multiple pushes stack. Valid on **any** base — the base weight is untouched, so a packed q4/q8
    /// tier keeps its footprint.
    pub fn push_lora(&mut self, a: Tensor, b: Tensor, scale: f64) {
        self.adapters.push(Adapter { a, b, scale });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::safetensors::MmapedSafetensors;
    use candle_gen::candle_core::{Device, Tensor};
    use std::collections::HashMap;

    /// Test-side MLX Q4 packer (group 64): per-element 4-bit codes → u32 words (LSB-first nibbles).
    /// Returns `(wq [out, in/8] u32, scales [out, in/g], biases [out, in/g], affine grid [out, in])`.
    fn q4_packed(out_dim: usize, in_dim: usize) -> (Tensor, Tensor, Tensor, Vec<f32>) {
        let dev = Device::Cpu;
        let g = MLX_GROUP_SIZE;
        let codes: Vec<u8> = (0..out_dim * in_dim)
            .map(|i| ((i * 7 + i / 13) % 16) as u8)
            .collect();
        let gpr = in_dim / g;
        let groups = out_dim * gpr;
        // Small, BOUNDED scales/biases so the dequantized grid stays ~O(1). A large-magnitude grid makes
        // `base.forward` huge, and the residual-isolation test (adapted − base) then recovers a tiny
        // residual from a catastrophic f32 cancellation. Bounded via `% k` so it holds at any group count.
        let scales: Vec<f32> = (0..groups)
            .map(|gi| 0.01 * ((gi % 5) as f32 + 1.0))
            .collect();
        let biases: Vec<f32> = (0..groups).map(|gi| -0.03 * (gi % 7) as f32).collect();
        let grid: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| {
                let (row, col) = (i / in_dim, i % in_dim);
                let gi = row * gpr + col / g;
                scales[gi] * codes[i] as f32 + biases[gi]
            })
            .collect();
        let words: Vec<u32> = codes
            .chunks_exact(8)
            .map(|c| {
                c.iter()
                    .enumerate()
                    .fold(0u32, |acc, (i, &q)| acc | ((q as u32 & 0xF) << (4 * i)))
            })
            .collect();
        let wq = Tensor::from_vec(words, (out_dim, in_dim / 8), &dev).unwrap();
        let s = Tensor::from_vec(scales, (out_dim, gpr), &dev).unwrap();
        let b = Tensor::from_vec(biases, (out_dim, gpr), &dev).unwrap();
        (wq, s, b, grid)
    }

    /// A per-call unique suffix so parallel tests never share a temp file (`cargo test` runs threads in
    /// ONE process, so a pid-only name would collide — one test truncating/deleting the file another is
    /// mid-mmap on → corrupt reads / flaky failures).
    static TMP_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    /// Build a packed [`AdaptLinear`] via [`AdaptLinear::detect`] on a written `.weight`/`.scales`/
    /// `.biases` triple (the round-trip the DiT loader takes) plus the affine grid it represents.
    fn packed_adapt(out_dim: usize, in_dim: usize) -> (AdaptLinear, Tensor) {
        let dev = Device::Cpu;
        let (wq, s, b, grid) = q4_packed(out_dim, in_dim);
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("p.weight".into(), wq);
        map.insert("p.scales".into(), s);
        map.insert("p.biases".into(), b);
        let uniq = TMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let tmp = std::env::temp_dir().join(format!(
            "anima_adapt_{}_{}.safetensors",
            std::process::id(),
            uniq
        ));
        candle_gen::candle_core::safetensors::save(&map, &tmp).unwrap();
        // SAFETY: freshly written, single-reader for the test.
        let st = unsafe { MmapedSafetensors::new(&tmp).unwrap() };
        let vb = VarBuilder::from_backend(Box::new(st), DType::F32, dev.clone());
        let lin = AdaptLinear::detect(&vb, "p").unwrap();
        std::fs::remove_file(&tmp).ok();
        (
            lin,
            Tensor::from_vec(grid, (out_dim, in_dim), &dev).unwrap(),
        )
    }

    /// `detect` recovers the logical dims from the packed `scales` shape and keeps the base packed.
    #[test]
    fn detect_recovers_dims_and_stays_packed() {
        let (out_dim, in_dim) = (64usize, 128usize); // in divisible by group 64
        let (lin, _grid) = packed_adapt(out_dim, in_dim);
        assert!(lin.is_packed(), "`.scales` ⇒ packed base");
        assert_eq!(
            lin.base_shape(),
            (out_dim, in_dim),
            "dims from scales shape"
        );
        assert!(!lin.is_adapted());
    }

    /// The additive LoRA residual `scale·((x·a)·b)` reproduces the **folded** `x·(W + δ)ᵀ` with
    /// `δ = (alpha/rank)·scale·(up·down)` on a dense f32 base — the additive==folded identity (tight in
    /// f32), the core weight-level property (no GPU needed).
    #[test]
    fn additive_lora_matches_folded_dense() {
        let dev = Device::Cpu;
        let (out_dim, in_dim, rank) = (48usize, 64usize, 4usize);
        let w = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev).unwrap();
        let down = Tensor::randn(0f32, 1f32, (rank, in_dim), &dev).unwrap(); // A [rank, in]
        let up = Tensor::randn(0f32, 1f32, (out_dim, rank), &dev).unwrap(); // B [out, rank]
        let (alpha, user_scale) = (8.0f64, 0.7f64);
        let ratio = alpha / rank as f64;

        // a = downᵀ [in, rank]; b = (upᵀ·ratio) [rank, out].
        let a = down.t().unwrap().contiguous().unwrap();
        let b = (up.t().unwrap().contiguous().unwrap() * ratio).unwrap();
        let mut lin = AdaptLinear {
            base: Base::Dense(Linear::new(w.clone(), None)),
            out_features: out_dim,
            in_features: in_dim,
            adapters: Vec::new(),
        };
        lin.push_lora(a, b, user_scale);
        assert!(lin.is_adapted());

        // Folded reference: δ = ratio·user_scale·(up·down); W_merged = W + δ.
        let delta = ((up.matmul(&down).unwrap()) * (ratio * user_scale)).unwrap();
        let folded = Linear::new((w + delta).unwrap(), None);

        let x = Tensor::randn(0f32, 1f32, (3usize, in_dim), &dev).unwrap();
        let dev_max = (lin.forward(&x).unwrap() - folded.forward(&x).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(dev_max < 1e-4, "additive vs folded deviates by {dev_max}");
    }

    /// A LoRA applied additively onto a **packed** base shifts the output, keeps the base **packed** (no
    /// dense weight materialized), and stays finite — the core acceptance on a quantized tier. A scale-0
    /// residual is an exact no-op (the mutation anchor: break the scale and this equality breaks).
    #[test]
    fn additive_lora_on_packed_shifts_stays_packed_and_scale0_is_noop() {
        let dev = Device::Cpu;
        let (out_dim, in_dim, rank) = (64usize, 128usize, 8usize);
        let (packed_base, _grid) = packed_adapt(out_dim, in_dim);
        assert!(packed_base.is_packed());

        let (mut adapted, _) = packed_adapt(out_dim, in_dim);
        let a = (Tensor::randn(0f32, 1f32, (in_dim, rank), &dev).unwrap() * 0.1).unwrap();
        let b = (Tensor::randn(0f32, 1f32, (rank, out_dim), &dev).unwrap() * 0.1).unwrap();
        adapted.push_lora(a.clone(), b.clone(), 1.0);
        assert!(adapted.is_packed(), "adapter must not un-pack the base");

        let x = Tensor::randn(0f32, 1f32, (4usize, in_dim), &dev).unwrap();
        let base_y = packed_base.forward(&x).unwrap();
        let adapted_y = adapted.forward(&x).unwrap();
        let shift = (adapted_y.sub(&base_y).unwrap())
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            shift > 1e-4,
            "additive LoRA on packed did not shift ({shift})"
        );
        assert!(
            adapted_y
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
                .is_finite(),
            "packed additive output non-finite"
        );

        // scale 0 ⇒ exact no-op vs the un-adapted packed base.
        let (mut zero, _) = packed_adapt(out_dim, in_dim);
        zero.push_lora(a, b, 0.0);
        let zero_dev = (zero.forward(&x).unwrap().sub(&base_y).unwrap())
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(
            zero_dev, 0.0,
            "scale-0 residual must be an exact no-op on packed"
        );
    }

    /// The **acceptance parity** on a packed base, isolated from the base's own quant error: the residual
    /// the adapter contributes equals `scale·((x·a)·b)` **exactly** (f32). `adapted.forward − base.forward`
    /// cancels the (bit-identical) packed base — the dequant-repack quant error included — leaving only the
    /// residual. This proves the LoRA is added **additively over the packed weight** (never folded into a
    /// re-quantized dense weight): the packed base contributes its dequant, the adapter contributes exactly
    /// its two-matmul residual, and they sum. (Comparing the full forward to a raw affine grid would instead
    /// measure the `Q4_1`-repack quant error — ~7e-4 — not the residual; that error cancels here.)
    #[test]
    fn additive_on_packed_adds_exact_residual_over_base() {
        let dev = Device::Cpu;
        let (out_dim, in_dim, rank) = (64usize, 128usize, 4usize);
        let (base, _grid) = packed_adapt(out_dim, in_dim);
        let (mut adapted, _grid) = packed_adapt(out_dim, in_dim); // bit-identical packed base
        let a = (Tensor::randn(0f32, 1f32, (in_dim, rank), &dev).unwrap() * 0.1).unwrap();
        let b = (Tensor::randn(0f32, 1f32, (rank, out_dim), &dev).unwrap() * 0.1).unwrap();
        let scale = 0.7f64;
        adapted.push_lora(a.clone(), b.clone(), scale);
        assert!(adapted.is_packed(), "base stays packed under the residual");

        let x = Tensor::randn(0f32, 1f32, (4usize, in_dim), &dev).unwrap();
        // The residual the adapter contributes = adapted − base (identical packed bases cancel exactly).
        let residual_actual = (adapted.forward(&x).unwrap() - base.forward(&x).unwrap()).unwrap();
        // Expected: scale·((x·a)·b).
        let residual_expected = ((x.matmul(&a).unwrap().matmul(&b).unwrap()) * scale).unwrap();
        let dev_max = (residual_actual - residual_expected)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            dev_max < 1e-5,
            "packed residual != scale·(x·a)·b (max diff {dev_max})"
        );
    }
}
