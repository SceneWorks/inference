//! Joint (dual-stream) attention for the NR-MMDiT — **owned by sc-14040**.
//!
//! Port of `Attention` + `MageDoubleStreamAttnProcessor`
//! (`_vendor/mage_flow/models/modules/mage_layers.py:212-511`):
//!
//! - separate `to_q`/`to_k`/`to_v` per stream with bias ([`crate::config::QKV_BIAS`]), plus
//!   `add_q_proj`/`add_k_proj`/`add_v_proj` for the text stream;
//! - **QK-RMSNorm on both streams** before the rotation (`norm_q`/`norm_k` and
//!   `norm_added_q`/`norm_added_k`, all elementwise-affine, `eps = 1e-6`);
//! - msrope applied to the **image** q/k only (`:421-422`) — the text stream is never rotated,
//!   which is what [`crate::config::APPLY_TEXT_ROTARY_EMB`] pins. The config reader *rejects* a
//!   checkpoint that flips it, rather than running this port against a model that expects rotated
//!   text;
//! - order **`[text, image]`** ([`crate::config::TEXT_STREAM_FIRST`]), `causal=false` (`:490`),
//!   default softmax scale (`softmax_scale=None` ⇒ `dim_head^-0.5`). The reference expresses that
//!   order as scatter offsets rather than a `cat`: text lands at each sample's start and image is
//!   shifted by that sample's text length (`:456-457`), consumed by the scatter at `:470-477`; the
//!   `torch.cat` at `:425-427` is commented out.
//!
//! Rotation convention is adjacent-pair complex (`view_as_complex` over `[..., -1, 2]`, `:15-21`),
//! so the table from [`crate::rope_embedder`] is consumed as `(cos, sin)` over adjacent lanes,
//! **not** the half-split convention FLUX/Qwen use.
//!
//! ## Varlen without a varlen kernel
//!
//! The reference isolates samples with `flash_attn_varlen_func`'s `cu_seqlens`. MLX has no varlen
//! flash kernel, so this port runs **one SDPA per packed segment** over that segment's
//! `[text, image]` joint sequence. Same computation, same isolation — and it is precisely what the
//! reference itself falls back to when flash-attn is unavailable (`_attn_backend.py`'s
//! `_sdpa_wrapper`, which is the backend the CPU parity goldens were dumped with).
//!
//! The alternative — one big SDPA plus a block-diagonal additive mask — was rejected: at 1024²
//! with fused CFG the joint sequence is ~8.2k tokens, so the mask alone would be ~270 MB of f32
//! for no numerical gain, and native-resolution packing (a 50k-token budget upstream) makes that
//! worse quadratically.

use mlx_rs::ops::{concatenate_axis, split_sections};
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::{AdaptableHost, AdaptableLinear};
use mlx_gen::attention::{sdpa_budgeted_bhsd, AttentionPlan};
use mlx_gen::qkv::{
    self, AttnPrepSpec, QkNormSpec, QkvHeads, QkvSource, RopeDtype, RopeSpec, RopeStyle,
    RopeTables, StreamOrder,
};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::rope_embedder::PackContext;
#[cfg(test)]
use crate::rope_embedder::RopeTable;
use crate::transformer::Linear;

/// The two token streams a dual-stream block carries, each `[1, tokens, dim]`.
///
/// A named pair rather than a tuple on purpose: the reference's block returns
/// `(encoder_hidden_states, hidden_states)` — text first — while its attention processor returns
/// `(img_attn_output, txt_attn_output)` — image first. Both orders are live in the same file
/// (`mage_layers.py:511`, `:665`), and swapping them yields a running model with garbage output
/// and no shape error, since both streams are `[1, ·, 3072]`.
#[derive(Debug, Clone)]
pub struct DualStream {
    pub txt: Array,
    pub img: Array,
}

/// `Attention(query_dim=dim, added_kv_proj_dim=dim, …, processor=MageDoubleStreamAttnProcessor())`.
#[derive(Debug, Clone)]
pub struct MageJointAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    add_q_proj: Linear,
    add_k_proj: Linear,
    add_v_proj: Linear,
    to_add_out: Linear,
    norm_q: Array,
    norm_k: Array,
    norm_added_q: Array,
    norm_added_k: Array,
    heads: i32,
    head_dim: i32,
    scale: f32,
    eps: f32,
}

impl MageJointAttention {
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        for linear in [
            &mut self.to_q,
            &mut self.to_k,
            &mut self.to_v,
            &mut self.to_out,
            &mut self.add_q_proj,
            &mut self.add_k_proj,
            &mut self.add_v_proj,
            &mut self.to_add_out,
        ] {
            linear.quantize(bits)?;
        }
        Ok(())
    }

    pub(crate) fn quantized_linear_count(&self) -> usize {
        [
            &self.to_q,
            &self.to_k,
            &self.to_v,
            &self.to_out,
            &self.add_q_proj,
            &self.add_k_proj,
            &self.add_v_proj,
            &self.to_add_out,
        ]
        .into_iter()
        .filter(|linear| linear.is_quantized())
        .count()
    }

    /// Load from `{prefix}.{to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj,to_add_out}`
    /// plus the four QK-norm scales — e.g. `transformer_blocks.0.attn`.
    ///
    /// Every projection carries a bias: the image side because `bias=True` is passed explicitly
    /// (`mage_layers.py:541`), the text side because `added_proj_bias` defaults to `True`
    /// (`:264-268`), and both output projections because `out_bias` defaults to `True`.
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        heads: i32,
        head_dim: i32,
        eps: f32,
    ) -> Result<Self> {
        let lin = |name: &str| Linear::from_weights(w, &format!("{prefix}.{name}"));
        let norm = |name: &str| -> Result<Array> {
            Ok(w.require(&format!("{prefix}.{name}.weight"))?.clone())
        };
        Ok(Self {
            to_q: lin("to_q")?,
            to_k: lin("to_k")?,
            to_v: lin("to_v")?,
            to_out: lin("to_out.0")?,
            add_q_proj: lin("add_q_proj")?,
            add_k_proj: lin("add_k_proj")?,
            add_v_proj: lin("add_v_proj")?,
            to_add_out: lin("to_add_out")?,
            norm_q: norm("norm_q")?,
            norm_k: norm("norm_k")?,
            norm_added_q: norm("norm_added_q")?,
            norm_added_k: norm("norm_added_k")?,
            heads,
            head_dim,
            // `softmax_scale=None` (`mage_layers.py:489`) ⇒ flash-attn's default `dim_head^-0.5`.
            scale: (head_dim as f32).powf(-0.5),
            eps,
        })
    }

    pub fn heads(&self) -> i32 {
        self.heads
    }

    pub fn head_dim(&self) -> i32 {
        self.head_dim
    }

    pub fn cast_weights(&mut self, dtype: Dtype) -> Result<()> {
        for lin in [
            &mut self.to_q,
            &mut self.to_k,
            &mut self.to_v,
            &mut self.to_out,
            &mut self.add_q_proj,
            &mut self.add_k_proj,
            &mut self.add_v_proj,
            &mut self.to_add_out,
        ] {
            lin.cast_weights(dtype)?;
        }
        for norm in [
            &mut self.norm_q,
            &mut self.norm_k,
            &mut self.norm_added_q,
            &mut self.norm_added_k,
        ] {
            if norm.dtype() != dtype {
                *norm = norm.as_dtype(dtype)?;
            }
        }
        Ok(())
    }

    /// Joint attention over the packed pair. Inputs and outputs are `[1, tokens, dim]`; the
    /// returned [`DualStream`] carries the two *attention outputs* (post `to_out` / `to_add_out`),
    /// **not** the residual — the block owns the gated add.
    pub fn forward(&self, stream: &DualStream, ctx: &PackContext) -> Result<DualStream> {
        self.forward_budgeted(stream, ctx, AttentionPlan::UNBOUNDED)
    }

    /// The production joint-attention path under the shared request-selected scratch budget.
    /// An unbounded plan preserves the historical single fused SDPA call byte-for-byte.
    pub fn forward_budgeted(
        &self,
        stream: &DualStream,
        ctx: &PackContext,
        plan: AttentionPlan<'_>,
    ) -> Result<DualStream> {
        let dim = self.heads * self.head_dim;
        let img_tokens = ctx.layout().img_tokens();
        let txt_tokens = ctx.layout().txt_tokens();
        expect_packed(&stream.img, img_tokens, dim, "image")?;
        expect_packed(&stream.txt, txt_tokens, dim, "text")?;

        // SC-18319 — the shared prologue, once per stream. Mage's knob selection: separate q/k/v
        // (knob 9), per-head QK-RMSNorm on BOTH streams (knob 1; `mage_layers.py:407-414`),
        // adjacent-pair complex rotation computed in f32 (knob 2; `:15-21`), and — knob 5 —
        // **msrope on the IMAGE q/k only**, the text stream deliberately unrotated
        // (`apply_text_rotary_emb: false`; `:420-422`), which is a `RopeStyle::None` text spec
        // rather than an identity table.
        let rope = ctx.rope();
        let img_spec = AttnPrepSpec::new(self.heads, self.head_dim)
            .with_qk_norm(QkNormSpec::per_head(&self.norm_q, &self.norm_k, self.eps))
            .with_rope(RopeSpec {
                style: RopeStyle::AdjacentPair,
                q: Some(RopeTables::new(&rope.cos, &rope.sin)),
                k: Some(RopeTables::new(&rope.cos, &rope.sin)),
                // Knob 12 — `x.float() … .type_as(x)` (`mage_layers.py:15-21`): the f32 promotion
                // is undone before SDPA, so a bf16 stream stays bf16.
                dtype: RopeDtype::RestoreInput,
                ..RopeSpec::default()
            });
        let img = qkv::prepare(
            QkvSource::Separate {
                q: &self.to_q.forward(&stream.img)?,
                k: &self.to_k.forward(&stream.img)?,
                v: &self.to_v.forward(&stream.img)?,
            },
            &img_spec,
        )?;
        let txt_spec = AttnPrepSpec::new(self.heads, self.head_dim).with_qk_norm(
            QkNormSpec::per_head(&self.norm_added_q, &self.norm_added_k, self.eps),
        );
        let txt = qkv::prepare(
            QkvSource::Separate {
                q: &self.add_q_proj.forward(&stream.txt)?,
                k: &self.add_k_proj.forward(&stream.txt)?,
                v: &self.add_v_proj.forward(&stream.txt)?,
            },
            &txt_spec,
        )?;

        let (img_out, txt_out) = self.joint_sdpa(&img, &txt, ctx, dim, plan)?;

        Ok(DualStream {
            img: self.to_out.forward(&img_out)?,
            txt: self.to_add_out.forward(&txt_out)?,
        })
    }

    /// One SDPA per packed segment over `[text, image]`, results gathered back into the two
    /// per-stream packed orders. Returns `([1, img_tokens, dim], [1, txt_tokens, dim])`.
    ///
    /// Both streams arrive in **BHSD** from [`qkv::prepare`], so the per-segment split is on the
    /// token axis (2) and the join is [`StreamOrder::TextFirst`] — the scatter offsets at
    /// `mage_layers.py:456-457` expressed as knob 11 rather than as a hand-rolled `cat`.
    fn joint_sdpa(
        &self,
        img: &QkvHeads,
        txt: &QkvHeads,
        ctx: &PackContext,
        dim: i32,
        plan: AttentionPlan<'_>,
    ) -> Result<(Array, Array)> {
        let segments = ctx.segments();
        let img_at = ctx.img_split_points();
        let txt_at = ctx.txt_split_points();
        let img_q = split_sections(&img.q, img_at, 2)?;
        let img_k = split_sections(&img.k, img_at, 2)?;
        let img_v = split_sections(&img.v, img_at, 2)?;
        let txt_q = split_sections(&txt.q, txt_at, 2)?;
        let txt_k = split_sections(&txt.k, txt_at, 2)?;
        let txt_v = split_sections(&txt.v, txt_at, 2)?;

        let mut img_parts = Vec::with_capacity(segments);
        let mut txt_parts = Vec::with_capacity(segments);
        for s in 0..segments {
            let txt_len = ctx.layout().txt_lens()[s];
            let joint = StreamOrder::TextFirst.join(
                &QkvHeads {
                    q: img_q[s].clone(),
                    k: img_k[s].clone(),
                    v: img_v[s].clone(),
                },
                &QkvHeads {
                    q: txt_q[s].clone(),
                    k: txt_k[s].clone(),
                    v: txt_v[s].clone(),
                },
            )?;
            // `causal=False`, no mask: per-sample isolation is the segmentation itself.
            let out = sdpa_budgeted_bhsd(&joint.q, &joint.k, &joint.v, self.scale, None, plan)?;
            // `[1, heads, L, head_dim]` → `[1, L, heads · head_dim]` (`flatten(1, 2)`).
            let out = qkv::merge_heads(&out)?;
            let halves = split_sections(&out, &[txt_len], 1)?;
            txt_parts.push(halves[0].clone());
            img_parts.push(halves[1].clone());
        }

        let repack = |parts: Vec<Array>, tokens: i32| -> Result<Array> {
            let flat = if parts.len() == 1 {
                parts
                    .into_iter()
                    .next()
                    .expect("one segment must produce one attention part")
            } else {
                let refs: Vec<&Array> = parts.iter().collect();
                concatenate_axis(&refs, 1)?
            };
            Ok(flat.reshape(&[1, tokens, dim])?)
        };
        Ok((
            repack(img_parts, ctx.layout().img_tokens())?,
            repack(txt_parts, ctx.layout().txt_tokens())?,
        ))
    }
}

/// LoRA/LoKr targets on the joint attention (sc-14055): both streams' q/k/v and output projections.
/// Diffusers names the image output `to_out.0` (an `nn.Sequential`, Linear at index 0) and the text
/// output `to_add_out`; the six input projections are bare.
impl AdaptableHost for MageJointAttention {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["to_q"] => Some(self.to_q.adaptable_mut()),
            ["to_k"] => Some(self.to_k.adaptable_mut()),
            ["to_v"] => Some(self.to_v.adaptable_mut()),
            ["to_out", "0"] => Some(self.to_out.adaptable_mut()),
            ["add_q_proj"] => Some(self.add_q_proj.adaptable_mut()),
            ["add_k_proj"] => Some(self.add_k_proj.adaptable_mut()),
            ["add_v_proj"] => Some(self.add_v_proj.adaptable_mut()),
            ["to_add_out"] => Some(self.to_add_out.adaptable_mut()),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        [
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_add_out",
        ]
        .into_iter()
        .map(String::from)
        .collect()
    }
}

fn expect_packed(x: &Array, tokens: i32, dim: i32, what: &str) -> Result<()> {
    if x.shape() != [1, tokens, dim] {
        return Err(Error::Msg(format!(
            "mage_flow: packed {what} stream must be [1, {tokens}, {dim}], got {:?}",
            x.shape()
        )));
    }
    Ok(())
}

/// `apply_rotary_emb_mageflow` (`mage_layers.py:15-21`) is now
/// `mlx_gen::qkv::apply_rope(.., RopeStyle::AdjacentPair, RotationAxes::TokenMajor, .., true)`
/// (SC-18319) — the identical adjacent-pair complex rotation through `nn::rope_rotate`, computed in
/// f32 (`x.float()`) and cast back to the input dtype (`.type_as(x)`), with the table broadcast one
/// row per token across the heads (`freqs_cis.unsqueeze(1)`).
///
/// Test-only shim keeping this file's rotation-convention pins pointed at the shared kernel.
#[cfg(test)]
fn apply_rope(x: &Array, rope: &RopeTable) -> Result<Array> {
    qkv::apply_rope(
        x,
        RopeTables::new(&rope.cos, &rope.sin),
        RopeStyle::AdjacentPair,
        mlx_gen::qkv::RotationAxes::TokenMajor,
        None,
        RopeDtype::RestoreInput,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rope_embedder::{ImgShape, MsRope, PackLayout};

    /// An identity table (`cos = 1`, `sin = 0`) leaves q/k untouched — the control that makes the
    /// rotation tests below meaningful.
    #[test]
    fn identity_rotation_is_a_no_op() {
        // `[B=1, tokens=2, heads=3, head_dim=4]` — the token-major layout the shared prologue uses.
        let values: Vec<f32> = (0..2 * 3 * 4).map(|v| v as f32).collect();
        let x = Array::from_slice(&values, &[1, 2, 3, 4]);
        let rope = RopeTable {
            cos: Array::from_slice(&[1.0f32; 4], &[2, 2]),
            sin: Array::from_slice(&[0.0f32; 4], &[2, 2]),
        };
        let out = apply_rope(&x, &rope).unwrap();
        assert_eq!(out.as_slice::<f32>(), x.as_slice::<f32>());
    }

    /// A quarter turn on adjacent pairs: `(a, b) → (−b, a)`. Pins the **adjacent-pair** convention
    /// against the half-split one FLUX/Qwen use — under a half-split reading the same table would
    /// pair lane 0 with lane 2 and yield `[−3, −4, 1, 2]`.
    #[test]
    fn rotation_pairs_adjacent_lanes_not_split_halves() {
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 1, 4]);
        let rope = RopeTable {
            cos: Array::from_slice(&[0.0f32, 0.0], &[1, 2]),
            sin: Array::from_slice(&[1.0f32, 1.0], &[1, 2]),
        };
        let out = apply_rope(&x, &rope).unwrap();
        assert_eq!(out.as_slice::<f32>(), &[-2.0, 1.0, -4.0, 3.0]);
    }

    #[test]
    fn rotation_rejects_a_table_of_the_wrong_length() {
        let x = Array::from_slice(&[0.0f32; 8], &[1, 2, 1, 4]);
        let rope = RopeTable {
            cos: Array::from_slice(&[1.0f32; 2], &[1, 2]),
            sin: Array::from_slice(&[0.0f32; 2], &[1, 2]),
        };
        assert!(apply_rope(&x, &rope).is_err());
    }

    /// A tiny 1-head, head_dim-4 joint attention (dim 4) built from zeroed weights — enough to
    /// exercise the LoRA host routing and adapter save without a real checkpoint.
    fn tiny_attention() -> MageJointAttention {
        let mut w = Weights::empty();
        for name in [
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_add_out",
        ] {
            w.insert(
                format!("a.{name}.weight"),
                Array::from_slice(&[0.0f32; 16], &[4, 4]),
            );
            w.insert(
                format!("a.{name}.bias"),
                Array::from_slice(&[0.0f32; 4], &[4]),
            );
        }
        for name in ["norm_q", "norm_k", "norm_added_q", "norm_added_k"] {
            w.insert(
                format!("a.{name}.weight"),
                Array::from_slice(&[1.0f32; 4], &[4]),
            );
        }
        MageJointAttention::from_weights(&w, "a", 1, 4, 1e-6).unwrap()
    }

    /// sc-14055 — the LoRA host must reach every one of the eight joint-attention projections, and
    /// every enumerated path must resolve (the save/reload round-trip depends on the two agreeing).
    #[test]
    fn adaptable_routing_covers_all_eight_projections() {
        let mut attn = tiny_attention();
        assert_eq!(
            attn.adaptable_paths(),
            [
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
                "add_q_proj",
                "add_k_proj",
                "add_v_proj",
                "to_add_out",
            ]
        );
        for path in attn.adaptable_paths() {
            let segs: Vec<&str> = path.split('.').collect();
            assert!(
                attn.adaptable_mut(&segs).is_some(),
                "enumerated path {path} must resolve via adaptable_mut"
            );
        }
        assert!(
            attn.adaptable_mut(&["to_out"]).is_none(),
            "the bare `to_out` (no `.0`) is not a target"
        );
        assert!(attn.adaptable_mut(&["nope"]).is_none());
    }

    /// sc-14055 — the known past gap: a saved LoRA adapter must carry `alpha` and `rank` in the
    /// safetensors `__metadata__` (and `networkType`), and its PEFT keys must round-trip. Built on
    /// the real Mage host routing so the saved paths are exactly what a reload resolves against.
    #[test]
    fn lora_adapter_persists_alpha_and_rank_in_metadata() {
        use mlx_gen::adapters::AdaptableHost;
        use mlx_gen::train::lora::{build_lora_targets, TrainAdapter};

        let mut host = tiny_attention();
        let paths = AdaptableHost::adaptable_paths(&host);
        let (targets, params) = build_lora_targets(&mut host, &paths, 4, 7).unwrap();
        let adapter = TrainAdapter::Lora { targets };

        // Per-process scratch dir — a fixed `$TMPDIR` name races a second concurrent `cargo test`.
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
        let path = dir.join("m.safetensors");
        adapter.save(&params, 8.0, 4.0, -1, "", &path).unwrap();

        let w = Weights::from_file(&path).unwrap();
        assert_eq!(w.metadata("networkType"), Some("lora"));
        assert_eq!(
            w.metadata("rank"),
            Some("4"),
            "rank must be in __metadata__"
        );
        assert_eq!(
            w.metadata("alpha"),
            Some("8"),
            "alpha must be in __metadata__"
        );
        assert!(
            w.keys().any(|k| k == "to_q.lora_A.weight"),
            "PEFT LoRA-A factor key must be present"
        );
        assert!(
            w.keys().any(|k| k == "to_q.alpha"),
            "per-target alpha tensor must be present"
        );
    }

    #[test]
    fn packed_stream_shape_is_validated() {
        let layout = PackLayout::generation(vec![ImgShape::latent(2, 2)], vec![3]).unwrap();
        let rope = MsRope::new(&[16, 56, 56], 10_000.0, true, 4096).unwrap();
        let ctx = PackContext::new(layout, &rope).unwrap();
        let packed = Array::from_slice(&[0.0f32; 4 * 8], &[1, 4, 8]);
        assert!(expect_packed(&packed, ctx.layout().img_tokens(), 8, "image").is_ok());
        let flat = Array::from_slice(&[0.0f32; 4 * 8], &[4, 8]);
        assert!(expect_packed(&flat, ctx.layout().img_tokens(), 8, "image").is_err());
    }
}
