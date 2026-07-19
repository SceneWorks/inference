//! Boogu's **Qwen3-VL-8B-Instruct** condition encoder (text path; the vision tower is unused for
//! text-to-image). A 36-layer decoder-only LM whose **last_hidden_state** (all layers + final norm)
//! is the per-token `[1, L, 4096]` instruction features the DiT's caption embedder consumes. Port of
//! `mlx-gen-boogu`'s `text_encoder/`.
//!
//! GQA (32 query / 8 kv heads), bias-less q/k/v/o, **per-head q/k RMSNorm**, HF half-split RoPE
//! (θ = 5e6), SwiGLU MLP, pre-norm causal decoder blocks. The text-only path uses plain 1-D RoPE
//! (Qwen3-VL's MRoPE sections all index the same sequential text position with no image tokens).
//! **Computes in f32** — the proven parity-grade precision for this exact encoder in the sibling
//! ideogram port; the DiT casts the features down to bf16. The weights are **stored bf16** (sc-12828):
//! the Qwen3-VL-8B weights ship bf16 on disk, so an f32 store only widens them (~16 GB resident to
//! carry no extra precision). The embedding is upcast to f32 (unlike krea/ideogram, this encoder did
//! not upcast before — its f32 compute was implicit in the f32 store, so the upcast is now explicit)
//! and each projection runs [`QLinear::forward_upcast`] (bf16 weight → f32 per matmul), with the
//! RMSNorm weights (including the final norm) loaded f32, so the forward is bit-identical to an f32
//! store at half the resident footprint.

use candle_gen::candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_gen::candle_nn::ops::softmax_last_dim;
use candle_gen::candle_nn::rotary_emb::rope;
// Shared Qwen3-VL grounding helpers (sc-11205 / F-118) — the MRoPE / vision-splice machinery this
// encoder previously defined inline, byte-identical to `candle-gen-krea`'s copy. Now one shared home.
use candle_gen::grounding::{
    causal_mask, image_blocks, mrope_cos_sin, mrope_positions, repeat_kv, replace_seq, slice_seq,
    Rotary,
};

use crate::loader::{embedding_detect, linear_detect, rmsnorm, Weights};
use crate::quant::{QEmbedding, QLinear};

/// Qwen3-VL-8B text-tower architecture (from `mllm/config.json` `text_config`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BooguTextEncoderConfig {
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
}

impl BooguTextEncoderConfig {
    pub fn qwen3_vl_8b() -> Self {
        Self {
            num_layers: 36,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 5_000_000.0,
        }
    }
}

struct Attention {
    q_proj: QLinear,
    k_proj: QLinear,
    v_proj: QLinear,
    o_proj: QLinear,
    q_norm: Tensor,
    k_norm: Tensor,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    eps: f64,
}

impl Attention {
    fn load(w: &Weights, prefix: &str, cfg: &BooguTextEncoderConfig) -> Result<Self> {
        Ok(Self {
            q_proj: linear_detect(w, &format!("{prefix}.q_proj"), false)?,
            k_proj: linear_detect(w, &format!("{prefix}.k_proj"), false)?,
            v_proj: linear_detect(w, &format!("{prefix}.v_proj"), false)?,
            o_proj: linear_detect(w, &format!("{prefix}.o_proj"), false)?,
            // f32 norm weights (sc-12828): tiny, kept f32 so `rmsnorm` runs f32-on-f32 regardless of the
            // encoder's bf16 weight store — bit-identical to the old all-f32 load (disk weights are bf16).
            q_norm: w.get_f32(&format!("{prefix}.q_norm.weight"))?,
            k_norm: w.get_f32(&format!("{prefix}.k_norm.weight"))?,
            n_heads: cfg.num_heads,
            n_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            eps: cfg.rms_norm_eps,
        })
    }

    /// `x`: `[b, s, hidden]`; `cos`/`sin`: `[s, head_dim/2]` (the text 1-D or image 3-D MRoPE table);
    /// `mask`: additive causal `[b, 1, s, s]`.
    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let (nh, nkv, hd) = (self.n_heads, self.n_kv_heads, self.head_dim);

        // `forward_upcast` (sc-12828): bf16-stored projections upcast to the f32 hidden per matmul —
        // bit-identical to an f32 store, inert when `x` already matches the weight dtype.
        let q = self.q_proj.forward_upcast(x)?.reshape((b, s, nh, hd))?;
        let k = self.k_proj.forward_upcast(x)?.reshape((b, s, nkv, hd))?;
        let v = self.v_proj.forward_upcast(x)?.reshape((b, s, nkv, hd))?;
        // Per-head q/k RMSNorm over the head dim, then transpose to [B, H, S, D].
        let q = rmsnorm(&q, &self.q_norm, self.eps)?
            .transpose(1, 2)?
            .contiguous()?;
        let k = rmsnorm(&k, &self.k_norm, self.eps)?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let q = rope(&q, cos, sin)?;
        let k = rope(&k, cos, sin)?;
        let k = repeat_kv(&k, nh / nkv)?;
        let v = repeat_kv(&v, nh / nkv)?;

        let scale = (hd as f64).powf(-0.5);
        // i32-overflow guard (sc-11193 / F-087, completing the sc-11154 / F-081 sweep that budgeted the
        // boogu ViT + krea grounded TE but missed THIS grounded MLLM path): the image-grounded edit
        // encode runs right up to the inclusive `MAX_EDIT_TOKENS` cap, so the `[B, nh, S, S]` scores
        // tensor reaches `32·8192² = 2^31 > i32::MAX` — candle's CUDA kernels index scores with i32 and
        // silently corrupt the tail (subtly wrong vision grounding). Chunk over the query rows via the
        // shared helper (the additive causal mask is `[B,1,S,S]`, narrowed per chunk); a single
        // un-chunked pass (byte-identical, fused `softmax_last_dim` preserved) runs below budget, so the
        // t2i / Base / Turbo paths are unaffected.
        let o = candle_gen::sdpa_budgeted_bhsd(
            &q,
            &k,
            &v,
            scale,
            Some(mask),
            softmax_last_dim,
            candle_gen::ATTN_SCORES_BUDGET,
        )?; // [B, nh, S, D]
        let o = o.transpose(1, 2)?.contiguous()?.reshape((b, s, nh * hd))?;
        self.o_proj.forward_upcast(&o)
    }
}

struct Mlp {
    gate: QLinear,
    up: QLinear,
    down: QLinear,
}

impl Mlp {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            gate: linear_detect(w, &format!("{prefix}.gate_proj"), false)?,
            up: linear_detect(w, &format!("{prefix}.up_proj"), false)?,
            down: linear_detect(w, &format!("{prefix}.down_proj"), false)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // `forward_upcast` (sc-12828): bf16-stored projections, f32 hidden — see `Attention::forward`.
        let gated = (self.gate.forward_upcast(x)?.silu()? * self.up.forward_upcast(x)?)?;
        self.down.forward_upcast(&gated)
    }
}

struct DecoderLayer {
    input_ln: Tensor,
    post_ln: Tensor,
    attn: Attention,
    mlp: Mlp,
    eps: f64,
}

impl DecoderLayer {
    fn load(w: &Weights, prefix: &str, cfg: &BooguTextEncoderConfig) -> Result<Self> {
        Ok(Self {
            // f32 norm weights (sc-12828): kept f32 so `rmsnorm` runs f32-on-f32 under the bf16 store.
            input_ln: w.get_f32(&format!("{prefix}.input_layernorm.weight"))?,
            post_ln: w.get_f32(&format!("{prefix}.post_attention_layernorm.weight"))?,
            attn: Attention::load(w, &format!("{prefix}.self_attn"), cfg)?,
            mlp: Mlp::load(w, &format!("{prefix}.mlp"))?,
            eps: cfg.rms_norm_eps,
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let h = (x + self
            .attn
            .forward(&rmsnorm(x, &self.input_ln, self.eps)?, cos, sin, mask)?)?;
        &h + self.mlp.forward(&rmsnorm(&h, &self.post_ln, self.eps)?)?
    }
}

/// Qwen3-VL `text_config.rope_parameters.mrope_section` — the per-axis (T/H/W) frequency counts over
/// `head_dim/2 = 64`. The image path interleaves these across the rotary freqs (the Qwen3-VL form).
const MROPE_SECTION: [usize; 3] = [24, 20, 20];

/// The Boogu Qwen3-VL text-path condition encoder.
pub struct BooguTextEncoder {
    embed_tokens: QEmbedding,
    layers: Vec<DecoderLayer>,
    rotary: Rotary,
    final_norm: Tensor,
    eps: f64,
    head_dim: usize,
    rope_theta: f32,
    device: Device,
}

impl BooguTextEncoder {
    /// Load from the `mllm` weights under `prefix` (`"model.language_model"`).
    pub fn load(
        w: &Weights,
        prefix: &str,
        cfg: &BooguTextEncoderConfig,
        max_seq: usize,
    ) -> Result<Self> {
        let embed_tokens = embedding_detect(w, &format!("{prefix}.embed_tokens"))?;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(DecoderLayer::load(w, &format!("{prefix}.layers.{i}"), cfg)?);
        }
        Ok(Self {
            embed_tokens,
            layers,
            rotary: Rotary::new(cfg.head_dim, cfg.rope_theta, max_seq.max(1), w.device())?,
            // f32 final-norm weight (sc-12828): applied to the f32 hidden, kept f32 under the bf16 store.
            final_norm: w.get_f32(&format!("{prefix}.norm.weight"))?,
            eps: cfg.rms_norm_eps,
            head_dim: cfg.head_dim,
            rope_theta: cfg.rope_theta,
            device: w.device().clone(),
        })
    }

    /// `input_ids`: `[1, S]` u32. Returns `last_hidden_state` `[1, S, 4096]` (f32) — all layers run,
    /// final norm applied. Causal (decoder-only); no padding (the candle tokenizer emits none).
    pub fn last_hidden(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (b, s) = input_ids.dims2()?;
        let (cos, sin) = self.rotary.text(s)?;
        let mask = causal_mask(b, s, &self.device)?;
        // Upcast the embedding to f32 (sc-12828) so the decoder stack computes f32 regardless of the
        // bf16 weight store. Before sc-12828 this was implicit (the f32 store made the table f32); now
        // the projections upcast per matmul via `forward_upcast`, so the embed must be widened here.
        let mut hidden = self.embed_tokens.forward(input_ids)?.to_dtype(DType::F32)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &cos, &sin, &mask)?;
        }
        rmsnorm(&hidden, &self.final_norm, self.eps)
    }

    /// Single-reference image-conditioned forward (Edit) — a thin wrapper over
    /// [`Self::last_hidden_with_images`] for one reference image (`grid_thw`, `image_embeds`
    /// `[n, 4096]`, and its 3 `deepstack` features). Kept for the single-reference call sites and the
    /// component-parity harness.
    pub fn last_hidden_with_image(
        &self,
        input_ids: &Tensor,
        image_embeds: &Tensor,
        deepstack: &[Tensor],
        grid_thw: [i32; 3],
        image_token_id: u32,
    ) -> Result<Tensor> {
        self.last_hidden_with_images(
            input_ids,
            std::slice::from_ref(image_embeds),
            std::slice::from_ref(&deepstack.to_vec()),
            std::slice::from_ref(&grid_thw),
            image_token_id,
        )
    }

    /// Multi-reference image-conditioned forward (Edit). Splices each reference's `image_embeds[k]`
    /// (`[n_k, 4096]`, the vision tower's merged output) into the token embeddings at the k-th
    /// contiguous `image_token_id` block (in input-id order), runs the 36 decoder layers under the
    /// 3-D **interleaved MRoPE** (each image's grid advancing the shared position counter), and injects
    /// each reference's `deepstack[k]` features at its image block after layers 0/1/2 — mirroring
    /// `Qwen3VLTextModel` with multiple `<|image_pad|>` blocks. `grids[k]` is image k's patch grid
    /// `[t, h, w]`. `b = 1`. The block order must match the reference order (the chat template emits
    /// the references' vision blocks before the instruction, in order).
    pub fn last_hidden_with_images(
        &self,
        input_ids: &Tensor,
        image_embeds: &[Tensor],
        deepstack: &[Vec<Tensor>],
        grids: &[[i32; 3]],
        image_token_id: u32,
    ) -> Result<Tensor> {
        let (b, s) = input_ids.dims2()?;
        let ids: Vec<u32> = input_ids.i(0)?.to_vec1::<u32>()?;

        // Contiguous `<|image_pad|>` blocks, in order; block k carries reference k.
        let blocks = image_blocks(&ids, image_token_id);
        if blocks.len() != image_embeds.len() {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "boogu edit: {} image-token blocks in input_ids but {} reference embeds",
                blocks.len(),
                image_embeds.len()
            )));
        }

        // Token embeddings (upcast to f32 — sc-12828, see `last_hidden`), then splice each reference's
        // vision embeds at its block. Each replacement is the same length as the block it replaces, so
        // earlier splices don't shift later indices.
        let mut hidden = self.embed_tokens.forward(input_ids)?.to_dtype(DType::F32)?; // [1, s, 4096], f32
        for (k, &(start, len)) in blocks.iter().enumerate() {
            if image_embeds[k].dim(0)? != len {
                return Err(candle_gen::candle_core::Error::Msg(format!(
                    "boogu edit: reference {k} has {} vision tokens but its image block is {len}",
                    image_embeds[k].dim(0)?
                )));
            }
            let img = image_embeds[k].unsqueeze(0)?.to_dtype(hidden.dtype())?; // [1, n_k, 4096]
            hidden = replace_seq(&hidden, &img, start, start + len, s)?;
        }

        // 3-D interleaved MRoPE (per-image grids) + causal mask (shared grounding helpers, sc-11205).
        let (pt, ph, pw) = mrope_positions(&ids, image_token_id, grids);
        let (cos, sin) = mrope_cos_sin(
            self.head_dim,
            MROPE_SECTION,
            self.rope_theta,
            &pt,
            &ph,
            &pw,
            &self.device,
        )?;
        let mask = causal_mask(b, s, &self.device)?;

        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, &cos, &sin, &mask)?;
            // Deepstack: after LM layers 0/1/2, add each reference's layer-i feature at its block.
            for (k, &(start, len)) in blocks.iter().enumerate() {
                if i < deepstack[k].len() {
                    let ds = deepstack[k][i].unsqueeze(0)?.to_dtype(hidden.dtype())?; // [1, n_k, 4096]
                    let mid = (slice_seq(&hidden, start, start + len)? + ds)?;
                    hidden = replace_seq(&hidden, &mid, start, start + len, s)?;
                }
            }
        }
        rmsnorm(&hidden, &self.final_norm, self.eps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const IMG: u32 = 151655;

    #[test]
    fn image_blocks_finds_runs_in_order() {
        // text, text, [4 image], text, [2 image], text.
        let ids = [9u32, 9, IMG, IMG, IMG, IMG, 9, IMG, IMG, 9];
        assert_eq!(image_blocks(&ids, IMG), vec![(2, 4), (7, 2)]);
    }

    #[test]
    fn mrope_positions_advance_across_two_images() {
        // Block 0 ↔ grid [1,4,4] (merged 2×2 = 4 tokens, t-step max(4,4)/2 = 2);
        // block 1 ↔ grid [1,4,2] (merged 2×1 = 2 tokens, t-step max(4,2)/2 = 2).
        let ids = [9u32, 9, IMG, IMG, IMG, IMG, 9, IMG, IMG, 9];
        let grids = [[1, 4, 4], [1, 4, 2]];
        let (pt, ph, pw) = mrope_positions(&ids, IMG, &grids);
        assert_eq!(pt.len(), ids.len());

        // Leading text advances 0,1.
        assert_eq!((pt[0], pt[1]), (0, 1));
        // Image 0 sits at t-axis = 2 (the running offset); spatial in h/w.
        assert_eq!(&pt[2..6], &[2, 2, 2, 2]);
        assert_eq!(&ph[2..6], &[2, 2, 3, 3]); // rows 0,0,1,1 + offset 2
        assert_eq!(&pw[2..6], &[2, 3, 2, 3]); // cols 0,1,0,1 + offset 2
                                              // Text after image 0: offset advanced by max(4,4)/2 = 2 → 4.
        assert_eq!(pt[6], 4);
        // Image 1 sits at t-axis = 5 (one past the text), 2 tokens (2×1 grid).
        assert_eq!(&pt[7..9], &[5, 5]);
        assert_eq!(&ph[7..9], &[5, 6]); // rows 0,1 + offset 5
        assert_eq!(&pw[7..9], &[5, 5]); // single column
                                        // Trailing text: offset advanced by max(4,2)/2 = 2 → 7.
        assert_eq!(pt[9], 7);
    }

    // ── sc-12828: bf16 weight store, f32 compute — bit-identical to an f32 store ──────────────────

    /// A tiny valid Qwen3-VL text-encoder weight map (2 layers, hidden 6, GQA 2/1, head_dim 4) drawn
    /// as **bf16** — modelling the hosted TE, whose weights ship bf16 on disk.
    fn tiny_boogu_te_map() -> (
        std::collections::HashMap<String, Tensor>,
        BooguTextEncoderConfig,
    ) {
        let cfg = BooguTextEncoderConfig {
            num_layers: 2,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 4,
            rms_norm_eps: 1e-6,
            rope_theta: 5_000_000.0,
        };
        let (hidden, inter, vocab) = (6usize, 8usize, 12usize);
        let (nh, nkv, hd) = (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
        let bf16 = |shape: &[usize]| {
            Tensor::randn(0f32, 0.5f32, shape, &Device::Cpu)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
        };
        let mut t = std::collections::HashMap::new();
        t.insert("lm.embed_tokens.weight".to_string(), bf16(&[vocab, hidden]));
        t.insert("lm.norm.weight".to_string(), bf16(&[hidden]));
        for i in 0..cfg.num_layers {
            let p = format!("lm.layers.{i}");
            t.insert(format!("{p}.input_layernorm.weight"), bf16(&[hidden]));
            t.insert(
                format!("{p}.post_attention_layernorm.weight"),
                bf16(&[hidden]),
            );
            t.insert(
                format!("{p}.self_attn.q_proj.weight"),
                bf16(&[nh * hd, hidden]),
            );
            t.insert(
                format!("{p}.self_attn.k_proj.weight"),
                bf16(&[nkv * hd, hidden]),
            );
            t.insert(
                format!("{p}.self_attn.v_proj.weight"),
                bf16(&[nkv * hd, hidden]),
            );
            t.insert(
                format!("{p}.self_attn.o_proj.weight"),
                bf16(&[hidden, nh * hd]),
            );
            t.insert(format!("{p}.self_attn.q_norm.weight"), bf16(&[hd]));
            t.insert(format!("{p}.self_attn.k_norm.weight"), bf16(&[hd]));
            t.insert(format!("{p}.mlp.gate_proj.weight"), bf16(&[inter, hidden]));
            t.insert(format!("{p}.mlp.up_proj.weight"), bf16(&[inter, hidden]));
            t.insert(format!("{p}.mlp.down_proj.weight"), bf16(&[hidden, inter]));
        }
        (t, cfg)
    }

    /// Serialize a tiny TE weight map to a dense component dir (`Weights::from_dir`-loadable at either
    /// store dtype). Caller removes it.
    fn write_boogu_te_dir(
        map: &std::collections::HashMap<String, Tensor>,
        tag: &str,
    ) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "boogu_te_{tag}_{}_{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        candle_gen::candle_core::safetensors::save(map, dir.join("model.safetensors")).unwrap();
        // A dense component config (no `quantization` block ⇒ `Weights::packed()` is None).
        std::fs::write(dir.join("config.json"), br#"{"hidden_size": 6}"#).unwrap();
        dir
    }

    /// The parity gate (sc-12828): a bf16 weight **store** with f32 **compute** is bit-identical to an
    /// f32 store — the disk weights are bf16, so an f32 store only widens them and every matmul still
    /// runs f32 (projections upcast via `forward_upcast`, norms + final norm load f32, and the embedding
    /// is upcast to f32). Reverting any of those makes the bf16 path a dtype-mismatch error, so this
    /// goes RED — it is not a tautology that passes with the win ripped out. CPU-runnable precisely
    /// because the compute never leaves f32.
    #[test]
    fn bf16_store_last_hidden_is_bit_identical_to_f32_store() {
        let (map, cfg) = tiny_boogu_te_map();
        let dir = write_boogu_te_dir(&map, "biteq");
        let ids = Tensor::from_vec(vec![1u32, 5, 3, 9], (1, 4), &Device::Cpu).unwrap();

        let w_f32 = Weights::from_dir(&dir, &Device::Cpu, DType::F32).unwrap();
        let out_f32 = BooguTextEncoder::load(&w_f32, "lm", &cfg, 64)
            .unwrap()
            .last_hidden(&ids)
            .unwrap();

        let w_bf16 = Weights::from_dir(&dir, &Device::Cpu, DType::BF16).unwrap();
        let out_bf16 = BooguTextEncoder::load(&w_bf16, "lm", &cfg, 64)
            .unwrap()
            .last_hidden(&ids)
            .unwrap();

        assert_eq!(out_f32.dtype(), DType::F32);
        assert_eq!(out_bf16.dtype(), DType::F32);
        assert_eq!(out_f32.dims(), &[1, 4, 6]);
        let a = out_f32.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = out_bf16.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(
            a.iter().all(|x| x.is_finite()),
            "last_hidden must be finite"
        );
        assert_eq!(
            a, b,
            "bf16-store last_hidden must be bit-identical to the f32-store forward"
        );

        // The footprint win is real: the bulk projection weight is bf16 at the bf16 store.
        let q = "lm.layers.0.self_attn.q_proj.weight";
        assert_eq!(w_bf16.get(q).unwrap().dtype(), DType::BF16);
        assert_eq!(w_f32.get(q).unwrap().dtype(), DType::F32);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The **image-grounded edit** path (`last_hidden_with_images`) is bit-identical under the bf16
    /// store too (sc-12828). This path is where Boogu **newly** gained an explicit `.to_dtype(F32)` on
    /// the embedding — reverting just that would leave the edit encode computing in bf16 (a parity
    /// regression the text-path test can't see), so this pins it: same bf16 disk, f32 vs bf16 store,
    /// bit-identical grounded context.
    #[test]
    fn bf16_store_grounded_is_bit_identical_to_f32_store() {
        let (map, cfg) = tiny_boogu_te_map();
        let dir = write_boogu_te_dir(&map, "grounded");
        let dev = Device::Cpu;

        // 2 text, a 4-token image block (id 0, a 2×2 merged grid → grid [1,4,4]), 1 text.
        let ids = Tensor::from_vec(vec![1u32, 1, 0, 0, 0, 0, 1], (1, 7), &dev).unwrap();
        let (n, hidden) = (4usize, 6usize);
        let embeds = vec![(Tensor::ones((n, hidden), DType::F32, &dev).unwrap() * 0.5).unwrap()];
        let deepstack = vec![(0..3)
            .map(|k| {
                (Tensor::ones((n, hidden), DType::F32, &dev).unwrap() * (0.01 * (k + 1) as f64))
                    .unwrap()
            })
            .collect::<Vec<_>>()];
        let grids = [[1i32, 4, 4]];

        let run = |store: DType| -> Tensor {
            let w = Weights::from_dir(&dir, &dev, store).unwrap();
            BooguTextEncoder::load(&w, "lm", &cfg, 64)
                .unwrap()
                .last_hidden_with_images(&ids, &embeds, &deepstack, &grids, 0)
                .unwrap()
        };
        let out_f32 = run(DType::F32);
        let out_bf16 = run(DType::BF16);

        assert_eq!(out_bf16.dtype(), DType::F32);
        assert_eq!(out_f32.dims(), &[1, 7, 6]);
        let a = out_f32.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = out_bf16.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(a.iter().all(|x| x.is_finite()));
        assert_eq!(
            a, b,
            "bf16-store grounded context must be bit-identical to the f32-store forward"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
