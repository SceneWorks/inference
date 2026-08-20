//! Boogu's **Qwen3-VL vision tower** — candle (Windows/CUDA) port of `mlx-gen-boogu`'s `vision/`.
//! The ViT that turns a reference image into the merged vision tokens (and 3 deepstack features) the
//! MLLM consumes for image-conditioned editing (the Edit path; sc-7523).
//!
//! Port of `Qwen3VLVisionModel` (transformers `models/qwen3_vl/modeling_qwen3_vl.py`). Structure:
//!   - **Patch embed** — a `Conv3d` with kernel == stride == `[temporal 2, 16, 16]`; the full-window
//!     kernel is folded to a per-patch matmul (`[embed, in·t·ph·pw]`).
//!   - **Learned `pos_embed`** — an `nn.Embedding(num_position_embeddings, hidden)` (a `√n × √n` grid)
//!     **bilinearly interpolated** to the image grid (merge-grouped order) and added.
//!   - **`depth` blocks** — pre-`LayerNorm` (eps 1e-6) → full attention (fused-QKV + bias, 2-D NeoX
//!     half-split rotary, single-image ⇒ full unmasked) → `proj`; pre-LayerNorm → **GELU-tanh** MLP
//!     (`linear_fc1`/`linear_fc2`, bias). No windowing (unlike Qwen2.5-VL).
//!   - **Patch merger** — pre-shuffle `LayerNorm` → concat `merge²` (=4) group → `linear_fc1 →
//!     GELU(exact) → linear_fc2` → `out_hidden`.
//!   - **Deepstack** — at vision layers `deepstack_visual_indexes` (`8,16,24`), a post-shuffle-norm
//!     merger produces a feature the LM later injects into its early layers.
//!
//! The grid-derived host-side math (rope table, bilinear pos-embed indices/weights) mirrors the
//! reference `get_vision_position_ids` / `get_vision_bilinear_indices_and_weights`. Runs in **f32**
//! (parity-grade; image-embeds cosine 0.9998 vs the reference) — the DiT casts the features → bf16.
//!
//! ## Packed (pre-quantized) towers are served (sc-20267)
//!
//! The tower's **projections** now packed-detect, mirroring what the MLX lane already does through
//! `mlx_gen::quant::lin`: when the component carries an MLX `quantization` marker *and* a
//! `{base}.scales` sibling is present, `vision_linear` builds the projection straight from the MLX
//! packed triple ([`crate::quant::QLinear::packed`]) instead of refusing the load. This is what a
//! packed **text-encoder tier** requires: `mlx_gen_minimax_h3::convert::TE_PACK_SUFFIXES` packs the
//! Qwen3-VL vision tower into the TE tier (`.attn.qkv`, `.attn.proj`, `.linear_fc1`, `.linear_fc2`),
//! so without this the candle lane cannot load one at all. Boogu's own hosted tiers keep the whole
//! tower dense bf16, and that path is byte-for-byte unchanged (see below).
//!
//! Three properties of the pack are load-bearing here:
//!
//! - **The group size comes from the tier's own declared marker**, never a constant: h3's TE tier
//!   packs at 64, the boogu/mage tiers at 32, and the packed shapes alone cannot disambiguate the
//!   two (the MLX lane learned this the hard way — see `mlx-gen-boogu/src/vision/mod.rs`).
//! - **A tower can be MIXED.** The vision blocks' `mlp.linear_fc2` has `in_features = 4304`, which is
//!   not a multiple of any published group size, so the MLX converter leaves that tensor **dense by
//!   shape** even though it is a declared pack target. The per-tensor auto-detect handles this with
//!   no special case: some projections land packed, others dense, in the same load.
//! - **`patch_embed.proj`, `pos_embed` and every `norm` stay dense by converter *policy***
//!   (`TE_DENSE_BY_POLICY`), in every tier, and each of the three is fronted by a `.scales` refusal
//!   (`require_dense`) so a snapshot that violates that policy fails loudly instead of reading
//!   codes as floats.
//!
//!   Only `patch_embed.proj` carried such a guard before this change. `pos_embed` and the norms were
//!   bare [`Weights::get_f32`] calls — and `get_f32` is `load_native(..).to_dtype(F32)`, so a u32
//!   code stream was **silently cast** to floats with no shape check able to notice. That is
//!   precisely the sc-14980 Mage `pos_embed` failure, and teaching this tower to load packed tiers is
//!   what turned packed vision input from exotic into ordinary. The guard is now one helper applied
//!   at all three sites rather than one inline check at one of them.
//!
//! The projections are read with [`Weights::get_native`] on the packed path (an f32 cast would
//! reinterpret the bit-packed nibbles) and asserted `U32`; everything on the dense path keeps using
//! [`Weights::get_f32`] for the reason in `vision_linear`'s docs.

pub mod preprocess;

use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::candle_nn::ops::softmax_last_dim;
use candle_gen::candle_nn::rotary_emb::rope;
use candle_gen::candle_nn::{LayerNorm, Linear, Module};

use crate::loader::Weights;
use crate::quant::QLinear;

/// Qwen3-VL vision-tower config (the `vision_config` block of `mllm/config.json`).
#[derive(Clone, Debug)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub depth: usize,
    pub out_hidden_size: usize,
    pub norm_eps: f64,
    pub rope_theta: f32,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    pub in_channels: usize,
    pub num_position_embeddings: usize,
    pub deepstack_visual_indexes: Vec<usize>,
}

impl VisionConfig {
    /// Boogu's Qwen3-VL-8B vision tower (`mllm/config.json::vision_config`).
    pub fn qwen3_vl() -> Self {
        Self {
            hidden_size: 1152,
            num_heads: 16,
            depth: 27,
            out_hidden_size: 4096,
            norm_eps: 1e-6,
            rope_theta: 10_000.0,
            patch_size: 16,
            temporal_patch_size: 2,
            spatial_merge_size: 2,
            in_channels: 3,
            num_position_embeddings: 2304,
            deepstack_visual_indexes: vec![8, 16, 24],
        }
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    /// `spatial_merge_size²` — patches per merged token.
    fn merge_unit(&self) -> usize {
        self.spatial_merge_size * self.spatial_merge_size
    }

    /// `√num_position_embeddings` — the learned pos-embed grid side.
    fn num_grid_per_side(&self) -> usize {
        (self.num_position_embeddings as f64).sqrt() as usize
    }
}

/// Affine LayerNorm over the last dim (eps 1e-6), built from a `Weights` `{prefix}.weight`/`.bias`.
fn layer_norm(w: &Weights, prefix: &str, eps: f64) -> Result<LayerNorm> {
    require_dense(
        w,
        prefix,
        "a LayerNorm is a 1-D affine pair, not a projection",
    )?;
    let weight = w.get_f32(&format!("{prefix}.weight"))?;
    let bias = w.get_f32(&format!("{prefix}.bias"))?;
    Ok(LayerNorm::new(weight, bias, eps))
}

/// Refuse a **packed** tensor at a loader that has no packed path (sc-14980).
///
/// The dense-by-policy keys — `patch_embed.proj`, `pos_embed` and every `norm` — are read with
/// [`Weights::get_f32`], and that is exactly what makes an unguarded one dangerous: `get_f32` is
/// `load_native(..).to_dtype(F32)`, so a u32 code stream is **silently cast** to floats. No shape
/// check downstream can see it, the tower runs, and the output is quietly wrong. That is the
/// sc-14980 Mage `pos_embed` failure, and packed vision input stopped being exotic the moment this
/// tower learned to load a packed text-encoder tier.
///
/// Gated on `w.packed().is_some()` as well as the `.scales` sibling, so a *dense* component that
/// merely happens to carry a similarly-named tensor is untouched — the marker is what makes the
/// sibling mean "packed".
fn require_dense(w: &Weights, base: &str, why: &str) -> Result<()> {
    if w.packed().is_some() && w.contains(&format!("{base}.scales")) {
        return Err(candle_gen::candle_core::Error::Msg(format!(
            "boogu: `{base}` has a `.scales` sibling in a packed component, but it is loaded by a \
             dense-only path — the MLX converter's `TE_DENSE_BY_POLICY` keeps `patch_embed.proj`, \
             `pos_embed` and every norm dense in EVERY tier ({why}). Reading its u32 codes as floats \
             is silent garbage (sc-14980), so this snapshot is refused rather than rendered."
        )));
    }
    Ok(())
}

/// Load one vision projection — **packed** straight from the MLX triple when the component declares
/// a `quantization` marker *and* a `{base}.scales` sibling exists, else **dense** (unchanged).
///
/// # Why the detect lives here rather than reusing [`crate::loader::linear_detect`]
///
/// The dense arm loads every tensor with [`Weights::get_f32`], **not** `Weights::get`, and that is
/// deliberate: Mage shares one Qwen3-VL component between a BF16 language model and this parity-grade
/// f32 tower (`candle-gen-mage/src/text_encoder.rs` builds it from a `DType::BF16` `Weights`), so
/// reading at the store dtype would silently inherit BF16 and reject f32 pixel inputs.
/// `loader::linear_detect`'s dense arm returns the tensor at the **store** dtype, so routing through
/// it would downgrade Mage's dense vision tower from f32 to bf16 — a real numerical regression. The
/// packed arm's detect logic is otherwise identical to `linear_detect`'s.
///
/// # The packed arm
///
/// The codes are read with [`Weights::get_native`] — no dtype cast, because a float cast would
/// reinterpret the bit-packed nibbles — and asserted `U32` so a snapshot whose `.weight` has already
/// been widened fails loudly instead of feeding garbage into the repack. `group_size` comes from the
/// component's declared `quantization.group_size` (h3's TE tier packs at 64, boogu/mage at 32; the
/// packed shapes alone cannot tell the two apart), and the optional dense `{base}.bias` is read f32
/// for the same store-dtype reason as the dense arm.
///
/// A projection the converter left dense — whether by policy or, like `mlp.linear_fc2`
/// (`in_features = 4304`), because its input width is not group-aligned — simply has no `.scales`
/// sibling and falls through to the dense arm. No per-tensor special cases.
fn vision_linear(w: &Weights, base: &str, bias: bool) -> Result<QLinear> {
    let weight_key = format!("{base}.weight");
    let scales_key = format!("{base}.scales");
    if let (Some(cfg), true) = (w.packed(), w.contains(&scales_key)) {
        let wq = w.get_native(&weight_key)?;
        if wq.dtype() != DType::U32 {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "boogu vision `{base}`: `{scales_key}` marks this projection packed, but \
                 `{weight_key}` loaded as {:?} rather than U32 — the packed code stream must reach \
                 the repack at its native width, and a float cast has already destroyed it",
                wq.dtype()
            )));
        }
        let scales = w.get_f32(&scales_key)?;
        let biases = w.get_f32(&format!("{base}.biases"))?;
        let dense_bias = bias
            .then(|| w.get_f32(&format!("{base}.bias")))
            .transpose()?;
        return QLinear::packed(&wq, &scales, &biases, dense_bias, cfg.group_size as usize);
    }
    let weight = w.get_f32(&weight_key)?;
    let bias = bias
        .then(|| w.get_f32(&format!("{base}.bias")))
        .transpose()?;
    Ok(QLinear::dense(Linear::new(weight, bias)))
}

/// One vision block: pre-LayerNorm full attention + pre-LayerNorm GELU-tanh MLP, both residual.
struct Block {
    norm1: LayerNorm,
    norm2: LayerNorm,
    qkv: QLinear,
    proj: QLinear,
    fc1: QLinear,
    fc2: QLinear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Block {
    fn load(w: &Weights, prefix: &str, cfg: &VisionConfig) -> Result<Self> {
        let head_dim = cfg.head_dim();
        Ok(Self {
            norm1: layer_norm(w, &format!("{prefix}.norm1"), cfg.norm_eps)?,
            norm2: layer_norm(w, &format!("{prefix}.norm2"), cfg.norm_eps)?,
            qkv: vision_linear(w, &format!("{prefix}.attn.qkv"), true)?,
            proj: vision_linear(w, &format!("{prefix}.attn.proj"), true)?,
            fc1: vision_linear(w, &format!("{prefix}.mlp.linear_fc1"), true)?,
            fc2: vision_linear(w, &format!("{prefix}.mlp.linear_fc2"), true)?,
            num_heads: cfg.num_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    /// Full attention over `x` `[seq, hidden]` with precomputed `cos`/`sin` `[seq, head_dim/2]` (f32).
    /// Single-image ⇒ unmasked. NeoX half-split rope ([`rope`]) then `matmul → softmax → matmul`.
    fn attention(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let seq = x.dim(0)?;
        let (h, hd) = (self.num_heads, self.head_dim);

        let qkv = self.qkv.forward(x)?.reshape((seq, 3, h, hd))?;
        // Each → [1, h, seq, hd] for candle's [b, h, s, d] rope/attention layout.
        let to_heads = |idx: usize| -> Result<Tensor> {
            qkv.narrow(1, idx, 1)?
                .squeeze(1)? // [seq, h, hd]
                .transpose(0, 1)? // [h, seq, hd]
                .unsqueeze(0)? // [1, h, seq, hd]
                .contiguous()
        };
        let q = rope(&to_heads(0)?, cos, sin)?;
        let k = rope(&to_heads(1)?, cos, sin)?;
        let v = to_heads(2)?;

        // i32-overflow guard (sc-11154 / F-081): the ViT runs full-image self-attention over every
        // patch token BEFORE any downstream token cap, so a single ~3.0 MP reference already gives a
        // `[1, h, seq, seq]` scores tensor of `16·11585² ≈ 2.15e9 > i32::MAX` — candle's CUDA kernels
        // index scores with i32 and silently corrupt the tail (this tower feeds boogu edit AND the new
        // `krea_2_edit` grounding). Chunk over the query rows via the shared helper; single un-chunked
        // pass (byte-identical) for in-budget sizes, exact fused `softmax_last_dim` preserved.
        let o = candle_gen::sdpa_budgeted_bhsd(
            &q,
            &k,
            &v,
            self.scale,
            None,
            softmax_last_dim,
            candle_gen::ATTN_SCORES_BUDGET,
        )?; // [1, h, seq, hd]
        let o = o
            .squeeze(0)? // [h, seq, hd]
            .transpose(0, 1)? // [seq, h, hd]
            .contiguous()?
            .reshape((seq, h * hd))?;
        self.proj.forward(&o)
    }

    fn mlp(&self, x: &Tensor) -> Result<Tensor> {
        self.fc2.forward(&self.fc1.forward(x)?.gelu()?)
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let x = (x + self.attention(&self.norm1.forward(x)?, cos, sin)?)?;
        &x + self.mlp(&self.norm2.forward(&x)?)?
    }
}

/// Patch merger: `LayerNorm` → concat `merge²` group → `linear_fc1 → GELU(exact) → linear_fc2`.
/// The main merger norms **pre-shuffle** (over `hidden`); the deepstack mergers norm **post-shuffle**
/// (over `hidden·merge²`).
struct Merger {
    norm: LayerNorm,
    fc1: QLinear,
    fc2: QLinear,
    postshuffle: bool,
    merged_dim: usize, // hidden · merge²
}

impl Merger {
    fn load(
        w: &Weights,
        prefix: &str,
        postshuffle: bool,
        merged_dim: usize,
        norm_eps: f64,
    ) -> Result<Self> {
        Ok(Self {
            norm: layer_norm(w, &format!("{prefix}.norm"), norm_eps)?,
            fc1: vision_linear(w, &format!("{prefix}.linear_fc1"), true)?,
            fc2: vision_linear(w, &format!("{prefix}.linear_fc2"), true)?,
            postshuffle,
            merged_dim,
        })
    }

    /// `x` `[seq, hidden]` → `[merged, out_hidden]` (`merged = seq / merge²`).
    fn forward(&self, x: &Tensor, merged: usize) -> Result<Tensor> {
        let x = if self.postshuffle {
            // group merge-units first, then norm over hidden·merge².
            let g = x.reshape((merged, self.merged_dim))?;
            self.norm.forward(&g)?
        } else {
            // norm over hidden per-patch, then group merge-units.
            let n = self.norm.forward(x)?;
            n.reshape((merged, self.merged_dim))?
        };
        self.fc2.forward(&self.fc1.forward(&x)?.gelu_erf()?)
    }
}

/// Host-side `grid_thw`-derived plan: the rope `cos`/`sin` (f32 `[seq, head_dim/2]`, merge-grouped
/// order) and the 4 bilinear corner indices + weights for the learned pos-embed interpolation.
struct Plan {
    merged: usize,
    cos: Tensor,               // f32 [seq, head_dim/2]
    sin: Tensor,               // f32 [seq, head_dim/2]
    bilinear_idx: [Tensor; 4], // u32 [seq]
    bilinear_w: [Tensor; 4],   // f32 [seq, 1]
}

/// The native Qwen3-VL vision tower.
pub struct VisionTower {
    patch_embed: Linear,
    pos_embed: Tensor, // [num_position_embeddings, hidden]
    blocks: Vec<Block>,
    merger: Merger,
    deepstack_mergers: Vec<Merger>,
    cfg: VisionConfig,
    device: Device,
}

impl VisionTower {
    /// Build from the mllm weight set (`{prefix}.*`, e.g. `"model.visual"`), loaded f32.
    ///
    /// The **projections** packed-detect per tensor (`vision_linear`); `patch_embed.proj`,
    /// `pos_embed` and the norms load dense unconditionally because the MLX converter's
    /// `TE_DENSE_BY_POLICY` keeps them dense in *every* tier, and all three are fronted by
    /// `require_dense` so a snapshot that violates that policy fails loudly rather than having its
    /// codes silently cast to floats by [`Weights::get_f32`].
    pub fn load(w: &Weights, cfg: VisionConfig, prefix: &str) -> Result<Self> {
        // `patch_embed.proj` is dense in every tier by converter policy (`TE_DENSE_BY_POLICY`), and
        // this 5-D conv fold cannot read a packed code stream anyway — so a `.scales` sibling here is
        // a snapshot this loader must refuse, not a shape to support (sc-9410, Issue 1; sc-14980).
        require_dense(
            w,
            &format!("{prefix}.patch_embed.proj"),
            "this 5-D conv fold cannot read a packed code stream anyway, sc-9410",
        )?;
        // Fold the Conv3d patch-embed weight `[embed, in, t, ph, pw]` → `[embed, in·t·ph·pw]` so the
        // full-kernel conv runs as a per-patch matmul; keep its bias.
        let conv = w.get_f32(&format!("{prefix}.patch_embed.proj.weight"))?;
        let dims = conv.dims();
        let embed = dims[0];
        let in_dim: usize = dims[1..].iter().product();
        let bias = w.get_f32(&format!("{prefix}.patch_embed.proj.bias"))?;
        let patch_embed = Linear::new(conv.reshape((embed, in_dim))?, Some(bias));

        let blocks = (0..cfg.depth)
            .map(|i| Block::load(w, &format!("{prefix}.blocks.{i}"), &cfg))
            .collect::<Result<Vec<_>>>()?;

        let merged_dim = cfg.hidden_size * cfg.merge_unit();
        let merger = Merger::load(
            w,
            &format!("{prefix}.merger"),
            false,
            merged_dim,
            cfg.norm_eps,
        )?;
        let deepstack_mergers = (0..cfg.deepstack_visual_indexes.len())
            .map(|i| {
                Merger::load(
                    w,
                    &format!("{prefix}.deepstack_merger_list.{i}"),
                    true,
                    merged_dim,
                    cfg.norm_eps,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        // The sc-14980 key itself. `get_f32` would cast a u32 code stream to floats without a word,
        // and nothing downstream of here can tell the difference.
        require_dense(
            w,
            &format!("{prefix}.pos_embed"),
            "it is an interpolated position table, read densely and never projected",
        )?;

        Ok(Self {
            patch_embed,
            pos_embed: w.get_f32(&format!("{prefix}.pos_embed.weight"))?,
            blocks,
            merger,
            deepstack_mergers,
            cfg,
            device: w.device().clone(),
        })
    }

    pub fn config(&self) -> &VisionConfig {
        &self.cfg
    }

    /// Host-side plan from `grid_thw` (rows `[t, h, w]` in patches), merge-grouped order — mirrors
    /// `get_vision_position_ids` (rope) + `get_vision_bilinear_indices_and_weights` (pos-embed).
    fn build_plan(&self, grid: &[[i32; 3]]) -> Result<Plan> {
        let c = &self.cfg;
        let m = c.spatial_merge_size as i32;
        let hd = c.head_dim();
        let rd = hd / 2; // rope width per token (= head_dim/2)
        let nfreq = rd / 2; // inv_freq length (= head_dim/4)
        let side = c.num_grid_per_side() as i32;
        let inv: Vec<f32> = (0..nfreq)
            .map(|j| c.rope_theta.powf(-((2 * j) as f32) / rd as f32))
            .collect();

        let mut seq = 0usize;
        let mut merged = 0usize;
        let mut rope_rows: Vec<f32> = Vec::new(); // [seq, rd]
        let mut idx: [Vec<u32>; 4] = [vec![], vec![], vec![], vec![]];
        let mut wts: [Vec<f32>; 4] = [vec![], vec![], vec![], vec![]];

        for g in grid {
            let (t, h, w) = (g[0], g[1], g[2]);
            seq += (t * h * w) as usize;
            merged += (t * (h / m) * (w / m)) as usize;

            // linspace(0, side-1, n): value at index i.
            let lin = |i: i32, n: i32| -> f64 {
                if n <= 1 {
                    0.0
                } else {
                    (side - 1) as f64 * i as f64 / (n - 1) as f64
                }
            };

            for _f in 0..t {
                for bh in 0..(h / m) {
                    for bw in 0..(w / m) {
                        for ih in 0..m {
                            for iw in 0..m {
                                let hpos = bh * m + ih;
                                let wpos = bw * m + iw;
                                // rope row: [hpos·inv(nfreq), wpos·inv(nfreq)] → rd.
                                for &fq in &inv {
                                    rope_rows.push(hpos as f32 * fq);
                                }
                                for &fq in &inv {
                                    rope_rows.push(wpos as f32 * fq);
                                }
                                // bilinear pos-embed interpolation corners (into the side×side grid).
                                let hc = lin(hpos, h);
                                let wc = lin(wpos, w);
                                let hf = hc.floor();
                                let wf = wc.floor();
                                let h0 = hf as i32;
                                let w0 = wf as i32;
                                let h1 = (h0 + 1).min(side - 1);
                                let w1 = (w0 + 1).min(side - 1);
                                let hfr = (hc - hf) as f32;
                                let wfr = (wc - wf) as f32;
                                idx[0].push((h0 * side + w0) as u32);
                                idx[1].push((h0 * side + w1) as u32);
                                idx[2].push((h1 * side + w0) as u32);
                                idx[3].push((h1 * side + w1) as u32);
                                wts[0].push((1.0 - hfr) * (1.0 - wfr));
                                wts[1].push((1.0 - hfr) * wfr);
                                wts[2].push(hfr * (1.0 - wfr));
                                wts[3].push(hfr * wfr);
                            }
                        }
                    }
                }
            }
        }

        let rope = Tensor::from_vec(rope_rows, (seq, rd), &self.device)?;
        let cos = rope.cos()?;
        let sin = rope.sin()?;
        let mk_i = |v: &[u32]| Tensor::from_vec(v.to_vec(), (seq,), &self.device);
        let mk_w = |v: &[f32]| Tensor::from_vec(v.to_vec(), (seq, 1), &self.device);
        Ok(Plan {
            merged,
            cos,
            sin,
            bilinear_idx: [
                mk_i(&idx[0])?,
                mk_i(&idx[1])?,
                mk_i(&idx[2])?,
                mk_i(&idx[3])?,
            ],
            bilinear_w: [
                mk_w(&wts[0])?,
                mk_w(&wts[1])?,
                mk_w(&wts[2])?,
                mk_w(&wts[3])?,
            ],
        })
    }

    /// Bilinearly-interpolated learned pos-embed `[seq, hidden]` (f32) for the plan.
    fn pos_embeds(&self, plan: &Plan) -> Result<Tensor> {
        let pe = self.pos_embed.to_dtype(DType::F32)?;
        let mut acc: Option<Tensor> = None;
        for k in 0..4 {
            let gathered = pe.index_select(&plan.bilinear_idx[k], 0)?; // [seq, hidden]
            let term = gathered.broadcast_mul(&plan.bilinear_w[k])?;
            acc = Some(match acc {
                Some(a) => (a + term)?,
                None => term,
            });
        }
        Ok(acc.unwrap())
    }

    /// Encode packed patches → (merged image embeds `[merged, out_hidden]`, deepstack features —
    /// one `[merged, out_hidden]` per `deepstack_visual_indexes` entry).
    ///
    /// `pixel_values` is `[seq, in·t·ph·pw]` (f32); `grid_thw` rows are `[t, h, w]` (patches).
    pub fn forward(
        &self,
        pixel_values: &Tensor,
        grid_thw: &[[i32; 3]],
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let plan = self.build_plan(grid_thw)?;
        let merged = plan.merged;

        // Patch embed + learned (interpolated) position embedding (all f32).
        let h = self.patch_embed.forward(pixel_values)?;
        let pos = self.pos_embeds(&plan)?;
        let mut h = (h + pos)?;

        let mut deepstack = Vec::with_capacity(self.cfg.deepstack_visual_indexes.len());
        for (i, blk) in self.blocks.iter().enumerate() {
            h = blk.forward(&h, &plan.cos, &plan.sin)?;
            if let Some(di) = self
                .cfg
                .deepstack_visual_indexes
                .iter()
                .position(|&x| x == i)
            {
                deepstack.push(self.deepstack_mergers[di].forward(&h, merged)?);
            }
        }

        let embeds = self.merger.forward(&h, merged)?;
        Ok((embeds, deepstack))
    }
}

#[cfg(test)]
mod tests {
    //! CPU-only coverage of the packed-vision seam (sc-20267). Every fixture is a synthetic MLX
    //! packed triple written into a temp component dir, so no real tier or device is involved.

    use super::*;
    use candle_gen::candle_core::safetensors;
    use std::collections::HashMap;
    use std::path::Path;

    /// The tier group size the fixtures pack at. Deliberately *not* the codebase default 64, so a
    /// loader that ignored the declared marker and assumed 64 would mis-derive the geometry.
    const G: usize = 32;

    /// A deterministic MLX **Q4** pack at [`G`] of an `[out, in]` weight: per-element 4-bit codes →
    /// u32 words (LSB-first nibbles), one `(scale, bias)` per group. Returns
    /// `(wq [out, in/8] u32, scales [out, in/G], biases [out, in/G], grid [out, in])`.
    ///
    /// The grid is the **dense reference** the packed forward is measured against: it is exactly what
    /// the tier's producer quantized, so agreement with it is the property that matters (a dense
    /// checkpoint of the *unquantized* weight is a different tensor and would only measure
    /// quantization error). Scales/biases are chosen f16-exact so the `Q4_1` repack is lossless.
    fn q4_pack(out_dim: usize, in_dim: usize, phase: usize) -> (Tensor, Tensor, Tensor, Tensor) {
        let dev = Device::Cpu;
        assert_eq!(in_dim % G, 0, "fixture in_dim must be a multiple of {G}");
        let codes: Vec<u8> = (0..out_dim * in_dim)
            .map(|i| (((i + phase) * 7 + i / 13) % 16) as u8)
            .collect();
        let groups_per_row = in_dim / G;
        let groups = out_dim * groups_per_row;
        let scales: Vec<f32> = (0..groups)
            .map(|k| 0.0625 * (((k + phase) % 7) as f32 + 1.0))
            .collect();
        let biases: Vec<f32> = (0..groups)
            .map(|k| -0.5 - 0.125 * ((k + phase) % 5) as f32)
            .collect();
        let grid: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| {
                let (row, col) = (i / in_dim, i % in_dim);
                let k = row * groups_per_row + col / G;
                scales[k] * codes[i] as f32 + biases[k]
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
        (
            Tensor::from_vec(words, (out_dim, in_dim / 8), &dev).unwrap(),
            Tensor::from_vec(scales, (out_dim, groups_per_row), &dev).unwrap(),
            Tensor::from_vec(biases, (out_dim, groups_per_row), &dev).unwrap(),
            Tensor::from_vec(grid, (out_dim, in_dim), &dev).unwrap(),
        )
    }

    /// **Relative max-abs deviation** — `max|a-b| / max|b|`. Never cosine: cosine is scale-invariant
    /// and therefore structurally blind to a mis-decoded group scale, which is exactly the defect
    /// class the packed path can produce.
    fn rel_max_abs(a: &Tensor, b: &Tensor) -> f32 {
        assert_eq!(a.dims(), b.dims(), "shape");
        let max_abs = |t: &Tensor| -> f32 {
            t.abs()
                .unwrap()
                .flatten_all()
                .unwrap()
                .max(0)
                .unwrap()
                .to_vec0::<f32>()
                .unwrap()
        };
        let d = max_abs(&(a - b).unwrap());
        let scale = max_abs(b);
        if scale == 0.0 {
            d
        } else {
            d / scale
        }
    }

    /// Insert the packed triple for `{base}` into a key map, returning the affine grid it decodes to.
    fn insert_packed(
        map: &mut HashMap<String, Tensor>,
        base: &str,
        out_dim: usize,
        in_dim: usize,
        phase: usize,
    ) -> Tensor {
        let (wq, scales, biases, grid) = q4_pack(out_dim, in_dim, phase);
        map.insert(format!("{base}.weight"), wq);
        map.insert(format!("{base}.scales"), scales);
        map.insert(format!("{base}.biases"), biases);
        grid
    }

    /// Write a component dir: the tensors plus a `config.json` that either declares a `quantization`
    /// marker at `quant_group` (a packed tier) or does not (a dense tier).
    fn write_component(dir: &Path, tensors: &HashMap<String, Tensor>, quant_group: Option<usize>) {
        std::fs::create_dir_all(dir).unwrap();
        safetensors::save(tensors, dir.join("model.safetensors")).unwrap();
        let cfg = match quant_group {
            Some(g) => serde_json::json!({ "quantization": { "bits": 4, "group_size": g } }),
            None => serde_json::json!({ "hidden_size": 64 }),
        };
        std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
    }

    /// A `Weights` over `map` at `dtype`. The `TempDir` is returned because the store is mmaped.
    fn weights_from(
        map: &HashMap<String, Tensor>,
        quant_group: Option<usize>,
        dtype: DType,
    ) -> (tempfile::TempDir, Weights) {
        let dir = tempfile::tempdir().unwrap();
        write_component(dir.path(), map, quant_group);
        let w = Weights::from_dir(dir.path(), &Device::Cpu, dtype).unwrap();
        (dir, w)
    }

    // ---- The tiny tower fixture -------------------------------------------------------------
    //
    // A depth-1 Qwen3-VL tower small enough to load and run on the CPU, with every projection's
    // `in_features` a multiple of `G` so any of them may be packed.

    const TINY_PREFIX: &str = "model.visual";
    const TINY_HIDDEN: usize = 64;
    const TINY_FFN: usize = 128;
    const TINY_OUT: usize = 32;

    fn tiny_cfg() -> VisionConfig {
        VisionConfig {
            hidden_size: TINY_HIDDEN,
            num_heads: 2,
            depth: 1,
            out_hidden_size: TINY_OUT,
            norm_eps: 1e-6,
            rope_theta: 10_000.0,
            patch_size: 2,
            temporal_patch_size: 2,
            spatial_merge_size: 2,
            in_channels: 3,
            num_position_embeddings: 16,
            deepstack_visual_indexes: vec![],
        }
    }

    /// Every tensor a [`tiny_cfg`] tower reads, all dense.
    fn tiny_dense_map() -> HashMap<String, Tensor> {
        let dev = Device::Cpu;
        let p = TINY_PREFIX;
        let (h, f, o) = (TINY_HIDDEN, TINY_FFN, TINY_OUT);
        let merged_dim = h * tiny_cfg().merge_unit(); // 256
        let patch_in = 3 * 2 * 2 * 2; // in_channels · temporal · ph · pw
        let mut map = HashMap::new();
        let rnd = |shape: (usize, usize)| Tensor::randn(0f32, 0.2f32, shape, &dev).unwrap();

        map.insert(
            format!("{p}.patch_embed.proj.weight"),
            Tensor::randn(0f32, 0.2f32, (h, 3, 2, 2, 2), &dev).unwrap(),
        );
        map.insert(
            format!("{p}.patch_embed.proj.bias"),
            Tensor::zeros(h, DType::F32, &dev).unwrap(),
        );
        map.insert(format!("{p}.pos_embed.weight"), rnd((16, h)));
        for norm in [
            format!("{p}.blocks.0.norm1"),
            format!("{p}.blocks.0.norm2"),
            format!("{p}.merger.norm"),
        ] {
            map.insert(
                format!("{norm}.weight"),
                Tensor::ones(h, DType::F32, &dev).unwrap(),
            );
            map.insert(
                format!("{norm}.bias"),
                Tensor::zeros(h, DType::F32, &dev).unwrap(),
            );
        }
        for (base, out_dim, in_dim) in [
            (format!("{p}.blocks.0.attn.qkv"), 3 * h, h),
            (format!("{p}.blocks.0.attn.proj"), h, h),
            (format!("{p}.blocks.0.mlp.linear_fc1"), f, h),
            (format!("{p}.blocks.0.mlp.linear_fc2"), h, f),
            (format!("{p}.merger.linear_fc1"), merged_dim, merged_dim),
            (format!("{p}.merger.linear_fc2"), o, merged_dim),
        ] {
            map.insert(format!("{base}.weight"), rnd((out_dim, in_dim)));
            map.insert(
                format!("{base}.bias"),
                Tensor::zeros(out_dim, DType::F32, &dev).unwrap(),
            );
        }
        assert_eq!(patch_in, 24);
        map
    }

    /// `[seq, in·t·ph·pw]` pixels for the single `[t=1, h=2, w=2]` grid the tiny tower is run on.
    fn tiny_pixels() -> Tensor {
        Tensor::randn(0f32, 1f32, (4usize, 24usize), &Device::Cpu).unwrap()
    }

    /// Swap `{base}`'s dense weight for a packed triple in place, returning the affine grid.
    fn repack_in_place(
        map: &mut HashMap<String, Tensor>,
        base: &str,
        out_dim: usize,
        in_dim: usize,
        phase: usize,
    ) -> Tensor {
        map.remove(&format!("{base}.weight"));
        insert_packed(map, base, out_dim, in_dim, phase)
    }

    // ---- (a) a packed projection really loads packed -----------------------------------------

    /// A declared-packed component with a `.scales` sibling loads the projection **packed**, and the
    /// packed weight recovers the full logical `[out, in]` geometry.
    ///
    /// The shape recovery is the non-default evidence: `forward(I)` returns `Wᵀ`, so a `[in, out]`
    /// result can only come from the repack having decoded `in = scales_cols · group_size` correctly.
    /// A silent dense fallback would instead have read the `[out, in/8]` code stream as the weight and
    /// failed the matmul outright.
    #[test]
    fn packed_projection_loads_packed_and_recovers_its_geometry() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (192usize, 64usize);
        let mut map = HashMap::new();
        insert_packed(&mut map, "blocks.0.attn.qkv", out_dim, in_dim, 3);
        let (_dir, w) = weights_from(&map, Some(G), DType::F32);

        let lin = vision_linear(&w, "blocks.0.attn.qkv", false)?;
        assert!(
            lin.is_packed(),
            "`.scales` + a quantization marker ⇒ packed load, not a silent dense fallback"
        );

        let eye = Tensor::eye(in_dim, DType::F32, &dev)?;
        let recovered = lin.forward(&eye)?; // [in, out] == Wᵀ
        assert_eq!(
            recovered.dims(),
            &[in_dim, out_dim],
            "the packed weight must recover the logical [out, in] geometry"
        );
        Ok(())
    }

    /// The group size is taken from the **tier's declared marker**, not a constant: the same group-32
    /// triple that loads under a `group_size: 32` marker is rejected under a `group_size: 64` one,
    /// because the derived bit-width comes out at 2. h3's TE tier packs at 64 and boogu/mage at 32, so
    /// a hardcoded constant would mis-decode one of them.
    #[test]
    fn packed_group_size_comes_from_the_declared_marker() -> Result<()> {
        let mut map = HashMap::new();
        insert_packed(&mut map, "blocks.0.attn.qkv", 192, 64, 1);

        let (_d32, w32) = weights_from(&map, Some(G), DType::F32);
        assert!(vision_linear(&w32, "blocks.0.attn.qkv", false)?.is_packed());

        let (_d64, w64) = weights_from(&map, Some(64), DType::F32);
        let err = vision_linear(&w64, "blocks.0.attn.qkv", false)
            .err()
            .expect("a group-32 pack read at group 64 must not load silently");
        assert!(
            err.to_string().contains("bit-width"),
            "unexpected error: {err}"
        );
        Ok(())
    }

    // ---- (b) the dense path is unchanged ------------------------------------------------------

    /// The dense arm is byte-for-byte what it always was. Neither half of the packed condition alone
    /// diverts a dense projection: a `.scales` sibling with **no** marker stays dense (the marker is
    /// the tier authority), and a marker with **no** `.scales` stays dense (the converter left this
    /// tensor dense — by policy or, like `mlp.linear_fc2`, by shape).
    ///
    /// And the dense weight is loaded **f32 under a BF16 store**: Mage builds this tower from a
    /// `DType::BF16` `Weights`, and the f32 read is the reason the packed detect could not simply
    /// reuse `loader::linear_detect` (whose dense arm returns the store dtype).
    #[test]
    fn dense_path_is_unchanged_and_stays_f32_under_a_bf16_store() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (64usize, 128usize);
        let dense_w = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev)?;
        let (_wq, scales, biases, _grid) = q4_pack(out_dim, in_dim, 0);

        // Marker present, no `.scales` ⇒ dense (the `linear_fc2`-by-shape case).
        let mut marked = HashMap::new();
        marked.insert("mlp.linear_fc2.weight".to_string(), dense_w.clone());
        let (_d, w) = weights_from(&marked, Some(G), DType::BF16);
        let lin = vision_linear(&w, "mlp.linear_fc2", false)?;
        assert!(!lin.is_packed(), "no `.scales` ⇒ dense path unchanged");
        match &lin {
            QLinear::Dense(l) => assert_eq!(
                l.weight().dtype(),
                DType::F32,
                "the dense vision arm must read f32 even under a BF16 store (Mage)"
            ),
            QLinear::Packed { .. } => unreachable!(),
        }

        // `.scales` present, no marker ⇒ dense (a dense tier's config carries no quantization block).
        let mut unmarked = HashMap::new();
        unmarked.insert("mlp.linear_fc2.weight".to_string(), dense_w);
        unmarked.insert("mlp.linear_fc2.scales".to_string(), scales);
        unmarked.insert("mlp.linear_fc2.biases".to_string(), biases);
        let (_d, w) = weights_from(&unmarked, None, DType::BF16);
        assert!(w.packed().is_none(), "no quantization block ⇒ dense tier");
        assert!(
            !vision_linear(&w, "mlp.linear_fc2", false)?.is_packed(),
            "no marker ⇒ dense path unchanged"
        );
        Ok(())
    }

    // ---- (c) the packed forward matches the affine grid --------------------------------------

    /// The packed forward reproduces a dense projection built from the **same dequantized affine
    /// grid**, measured by relative max-abs deviation (never cosine — see [`rel_max_abs`]). Bias
    /// included, so the dense `{base}.bias` is proven to survive the packed arm.
    #[test]
    fn packed_forward_matches_the_affine_grid() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (128usize, 256usize);
        let mut map = HashMap::new();
        let grid = insert_packed(&mut map, "attn.proj", out_dim, in_dim, 5);
        let bias = Tensor::randn(0f32, 0.5f32, out_dim, &dev)?;
        map.insert("attn.proj.bias".to_string(), bias.clone());
        let (_dir, w) = weights_from(&map, Some(G), DType::F32);

        let packed = vision_linear(&w, "attn.proj", true)?;
        assert!(packed.is_packed());
        let reference = QLinear::dense(Linear::new(grid, Some(bias)));

        let x = Tensor::randn(0f32, 1f32, (6usize, in_dim), &dev)?;
        let drift = rel_max_abs(&packed.forward(&x)?, &reference.forward(&x)?);
        assert!(
            drift < 1e-6,
            "packed vs affine-grid relative max-abs drift {drift:e} exceeds 1e-6"
        );
        Ok(())
    }

    // ---- (d) a float `.weight` under a `.scales` sibling is refused --------------------------

    /// A `.scales` sibling marks the weight packed, so a `{base}.weight` that arrives as a **float**
    /// means the code stream was already destroyed by a cast (or the snapshot is malformed). That is
    /// refused with a typed error naming the key, never fed into the repack.
    #[test]
    fn packed_marker_over_a_float_weight_is_refused() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (64usize, 128usize);
        let (_wq, scales, biases, grid) = q4_pack(out_dim, in_dim, 2);
        let mut map = HashMap::new();
        // A float `.weight` where the triple demands u32 codes.
        map.insert("blocks.0.attn.qkv.weight".to_string(), grid);
        map.insert("blocks.0.attn.qkv.scales".to_string(), scales);
        map.insert("blocks.0.attn.qkv.biases".to_string(), biases);
        let (_dir, w) = weights_from(&map, Some(G), DType::F32);
        assert_eq!(
            w.get_native("blocks.0.attn.qkv.weight")?.dtype(),
            DType::F32,
            "fixture precondition: the stored weight really is a float"
        );

        let err = vision_linear(&w, "blocks.0.attn.qkv", false)
            .err()
            .expect("a float `.weight` under a `.scales` sibling must be refused");
        let msg = err.to_string();
        assert!(
            msg.contains("blocks.0.attn.qkv.weight") && msg.contains("U32"),
            "the error must name the offending key and the required dtype, got: {msg}"
        );
        let _ = dev;
        Ok(())
    }

    // ---- (e) a MIXED tower loads and runs ----------------------------------------------------

    /// A **mixed** tower — `attn.qkv`, `attn.proj`, `mlp.linear_fc1` and both merger projections
    /// packed, `mlp.linear_fc2` dense — loads with the right per-tensor decision and runs end to end.
    ///
    /// This is the real published shape: `TE_PACK_SUFFIXES` names `.linear_fc2`, but the vision
    /// blocks' `mlp.linear_fc2` has `in_features = 4304`, which is not group-aligned, so the MLX
    /// converter leaves that one tensor dense **by shape**. The per-tensor auto-detect covers it with
    /// no special case.
    #[test]
    fn mixed_tower_loads_packed_and_dense_projections_together() -> Result<()> {
        let p = TINY_PREFIX;
        let (h, f, o) = (TINY_HIDDEN, TINY_FFN, TINY_OUT);
        let merged_dim = h * tiny_cfg().merge_unit();
        let mut map = tiny_dense_map();

        repack_in_place(&mut map, &format!("{p}.blocks.0.attn.qkv"), 3 * h, h, 1);
        repack_in_place(&mut map, &format!("{p}.blocks.0.attn.proj"), h, h, 2);
        repack_in_place(&mut map, &format!("{p}.blocks.0.mlp.linear_fc1"), f, h, 3);
        // `mlp.linear_fc2` deliberately stays dense — the not-group-aligned case.
        repack_in_place(
            &mut map,
            &format!("{p}.merger.linear_fc1"),
            merged_dim,
            merged_dim,
            4,
        );
        repack_in_place(
            &mut map,
            &format!("{p}.merger.linear_fc2"),
            o,
            merged_dim,
            5,
        );

        let (_dir, w) = weights_from(&map, Some(G), DType::BF16);
        let tower = VisionTower::load(&w, tiny_cfg(), p)?;

        let blk = &tower.blocks[0];
        assert!(blk.qkv.is_packed(), "attn.qkv packed");
        assert!(blk.proj.is_packed(), "attn.proj packed");
        assert!(blk.fc1.is_packed(), "mlp.linear_fc1 packed");
        assert!(
            !blk.fc2.is_packed(),
            "mlp.linear_fc2 has no `.scales` ⇒ dense by shape, in the same load"
        );
        assert!(tower.merger.fc1.is_packed(), "merger.linear_fc1 packed");
        assert!(tower.merger.fc2.is_packed(), "merger.linear_fc2 packed");

        let (embeds, deepstack) = tower.forward(&tiny_pixels(), &[[1, 2, 2]])?;
        assert_eq!(
            embeds.dims(),
            &[1, o],
            "a mixed packed/dense tower must still run"
        );
        assert_eq!(embeds.dtype(), DType::F32, "the tower computes in f32");
        assert!(deepstack.is_empty());
        assert!(
            embeds.abs()?.sum_all()?.to_vec0::<f32>()?.is_finite(),
            "mixed-tower output must be finite"
        );
        Ok(())
    }

    /// The all-dense tiny tower still loads with every projection dense and runs — the zero-diff
    /// baseline for the existing consumers (boogu's own tiers, krea, mage).
    #[test]
    fn fully_dense_tower_is_unchanged() -> Result<()> {
        let map = tiny_dense_map();
        let (_dir, w) = weights_from(&map, None, DType::BF16);
        let tower = VisionTower::load(&w, tiny_cfg(), TINY_PREFIX)?;
        let blk = &tower.blocks[0];
        for (name, lin) in [
            ("attn.qkv", &blk.qkv),
            ("attn.proj", &blk.proj),
            ("mlp.linear_fc1", &blk.fc1),
            ("mlp.linear_fc2", &blk.fc2),
            ("merger.linear_fc1", &tower.merger.fc1),
            ("merger.linear_fc2", &tower.merger.fc2),
        ] {
            assert!(!lin.is_packed(), "{name} must stay dense on a dense tier");
        }
        let (embeds, _) = tower.forward(&tiny_pixels(), &[[1, 2, 2]])?;
        assert_eq!(embeds.dims(), &[1, TINY_OUT]);
        Ok(())
    }

    // ---- (f) the dense-by-policy guards still refuse -----------------------------------------

    /// `patch_embed.proj` is dense in every tier by converter policy (`TE_DENSE_BY_POLICY`) and this
    /// 5-D conv fold cannot read a code stream, so a packed `patch_embed.proj` is **still rejected**
    /// (sc-9410 / sc-14980). Serving packed projections did not weaken this.
    #[test]
    fn packed_patch_embed_is_still_rejected() -> Result<()> {
        let p = TINY_PREFIX;
        let mut map = tiny_dense_map();
        // Only the `.scales` marker sibling is needed: the guard fires before any tensor is read.
        let (_wq, scales, biases, _grid) = q4_pack(TINY_HIDDEN, 32, 0);
        map.insert(format!("{p}.patch_embed.proj.scales"), scales);
        map.insert(format!("{p}.patch_embed.proj.biases"), biases);

        let (_dir, w) = weights_from(&map, Some(G), DType::BF16);
        let err = VisionTower::load(&w, tiny_cfg(), p)
            .err()
            .expect("a packed patch_embed.proj must still be refused");
        let msg = err.to_string();
        assert!(
            msg.contains("patch_embed.proj"),
            "the refusal must name the offending key, got: {msg}"
        );
        Ok(())
    }

    /// **A packed `pos_embed` is refused — the sc-14980 key itself.**
    ///
    /// This is the guard that did NOT exist until the packed-tier work landed, and its absence was
    /// the dangerous kind: `pos_embed` is read with [`Weights::get_f32`], which is
    /// `load_native(..).to_dtype(F32)`, so a u32 code stream was **silently cast** to floats. Nothing
    /// downstream could notice — the shape is unchanged, the tower runs, the output is quietly wrong.
    ///
    /// Driven at the TOWER, i.e. the production path `VisionTower::load` really takes. A test against
    /// a standalone guard helper would prove the helper works while production never called it, which
    /// is exactly the gap this closes.
    #[test]
    fn packed_pos_embed_is_refused_rather_than_silently_cast_to_floats() -> Result<()> {
        let p = TINY_PREFIX;
        let mut map = tiny_dense_map();
        let (wq, scales, biases, _grid) = q4_pack(TINY_HIDDEN, 32, 0);
        // The FULL triple, including u32 codes over the dense `pos_embed.weight`, so this fixture is
        // the real silent-cast shape rather than a bare marker.
        map.insert(format!("{p}.pos_embed.weight"), wq);
        map.insert(format!("{p}.pos_embed.scales"), scales);
        map.insert(format!("{p}.pos_embed.biases"), biases);

        let (_dir, w) = weights_from(&map, Some(G), DType::BF16);
        let err = VisionTower::load(&w, tiny_cfg(), p)
            .err()
            .expect("a packed pos_embed must be refused, not read as floats");
        let msg = err.to_string();
        assert!(
            msg.contains("pos_embed"),
            "the refusal must name the offending key, got: {msg}"
        );
        assert!(
            msg.contains("sc-14980"),
            "the refusal must cite the silent-garbage class it exists to prevent, got: {msg}"
        );

        // **The hazard is real, not hypothetical**: the same codes under `get_f32` come back as a
        // plausible float tensor of the RIGHT SHAPE, with no error at all. That is what the guard
        // above is standing in front of.
        let silently_cast = w.get_f32(&format!("{p}.pos_embed.weight"))?;
        assert_eq!(
            silently_cast.dtype(),
            DType::F32,
            "get_f32 casts u32 codes to floats without complaint — the whole reason for the guard"
        );
        Ok(())
    }

    /// **A packed vision `norm` is refused**, at the tower.
    ///
    /// `layer_norm` is the shared loader behind every `norm1` / `norm2` / merger norm, so guarding it
    /// there covers all of them at once. Same silent-cast exposure as `pos_embed`: a LayerNorm is a
    /// 1-D affine pair read with `get_f32`, and `TE_DENSE_BY_POLICY` keeps norms dense in every tier.
    #[test]
    fn packed_vision_norm_is_refused() -> Result<()> {
        let p = TINY_PREFIX;
        let mut map = tiny_dense_map();
        let (_wq, scales, biases, _grid) = q4_pack(TINY_HIDDEN, 32, 0);
        map.insert(format!("{p}.blocks.0.norm1.scales"), scales);
        map.insert(format!("{p}.blocks.0.norm1.biases"), biases);

        let (_dir, w) = weights_from(&map, Some(G), DType::BF16);
        let err = VisionTower::load(&w, tiny_cfg(), p)
            .err()
            .expect("a packed vision norm must be refused");
        assert!(
            err.to_string().contains("norm1"),
            "the refusal must name the offending key, got: {err}"
        );
        Ok(())
    }

    /// The guards key off the component's **`quantization` marker**, not off the tensor name alone —
    /// so an unmarked (genuinely dense) component is loaded even if it happens to carry a
    /// `.scales`-suffixed tensor. Without this the three refusals above could pass by refusing
    /// everything, and `require_dense` would disagree with `vision_linear`, whose packed detect is
    /// gated on the same marker: one would refuse the component while the other read it dense.
    ///
    /// The fixture is the discriminating one — a `.scales` sibling on a dense-by-policy key in a
    /// component with **no marker**. Dropping the `w.packed().is_some()` half of the condition reds
    /// this and nothing else.
    #[test]
    fn the_dense_by_policy_guards_are_gated_on_the_marker_not_the_tensor_name() -> Result<()> {
        let p = TINY_PREFIX;
        let mut map = tiny_dense_map();
        let (_wq, scales, biases, _grid) = q4_pack(TINY_HIDDEN, 32, 0);
        map.insert(format!("{p}.pos_embed.scales"), scales);
        map.insert(format!("{p}.pos_embed.biases"), biases);

        // `None` ⇒ no `quantization` block is written, so the component is dense however its tensors
        // happen to be named.
        let (_dir, w) = weights_from(&map, None, DType::F32);
        VisionTower::load(&w, tiny_cfg(), p).expect(
            "an UNMARKED component is dense by definition — the guards must not fire on a stray \
             `.scales` name, or they would disagree with vision_linear's own marker-gated detect",
        );
        Ok(())
    }

    /// A fully dense, unmarked component still loads with every guard in place — the control for the
    /// three refusals above.
    #[test]
    fn the_dense_by_policy_guards_do_not_fire_on_a_dense_component() -> Result<()> {
        let p = TINY_PREFIX;
        let (_dir, w) = weights_from(&tiny_dense_map(), None, DType::F32);
        VisionTower::load(&w, tiny_cfg(), p)
            .expect("a dense component must load with every dense-by-policy guard in place");
        Ok(())
    }
}
