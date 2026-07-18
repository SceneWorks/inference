//! UMT5-XXL text encoder — port of `models/wan/text_encoder.py` (`T5Encoder` and friends) plus the
//! `_clean_text` / `encode_text` orchestration from `models/wan/loading.py`.
//!
//! Wan conditions on Google's **UMT5-XXL** (24 layers, dim 4096, 64 heads, dim_ffn 10240). It
//! differs from the standard HF T5 in two ways the port must honor:
//!   1. **Per-layer relative-position bias** (`shared_pos=False`): every block owns its own
//!      `[num_buckets, num_heads]` bucket-embedding table (24 distinct tables), rather than sharing
//!      block-0's. The bucket *grid* is identical across layers, so it is computed once and only the
//!      per-layer table lookup differs.
//!   2. **Gated-GELU FFN** named `gate_proj` / `fc1` / `fc2`: `fc2(fc1(x) · gelu_tanh(gate_proj(x)))`.
//!
//! The whole encoder runs **f32** (the reference upcasts every weight to f32 and computes the
//! attention softmax in f32 — unscaled T5 logits can be large, so bf16 softmax loses precision).
//! The default (bf16-tier) build keeps the large Linear weights as loaded (bf16) and runs f32
//! activations: `matmul(f32, bf16)` promotes to an f32 GEMM, which is bit-identical to the
//! reference's explicit-f32 weights (bf16→f32 is lossless) and is the same proven pattern the FLUX
//! T5 path uses. The tiny norm / position-bias tables are upcast to f32 so `fast::rms_norm` and the
//! bias add see f32 operands.
//!
//! Bit-exactness target = the `mlx_video` reference (itself MLX), so `T5LayerNorm` maps to
//! `mlx_rs::fast::rms_norm` (the reference's `mx.fast.rms_norm`) and the FFN gate to the hand-rolled
//! [`gelu_tanh`] (NOT `mlx_rs::nn::gelu_approximate`, whose `√(2/π)` constant is 1 ULP off — see its
//! doc). The encoder is verified **bit-exact** (max|Δ| = 0.0) to the reference on every test prompt
//! via [`Umt5Encoder::from_weights`].
//!
//! ## Quantized tiers (sc-12831)
//! On a **quantized DiT tier** (q4/q8) the encoder loads via [`Umt5Encoder::from_weights_quantized`],
//! which packs the seven projection/FFN linears per block to the caller's resolved width — **Q8** in
//! production, the near-lossless floor for this drift-sensitive encoder (cosine 0.9998 vs the bf16
//! baseline at Q8 vs ~0.976 at Q4; see `model::effective_te_quant`). The predicate mirrors the DiT
//! `_quantize_predicate` — attn `{q,k,v,o}` + `ffn.{gate_proj,fc1,fc2}`; the token embedding, norms and
//! position-bias tables stay dense. Activations remain **f32** (`quantized_matmul` accumulates fp32, so
//! the numerically-sensitive unscaled-logit softmax is unaffected — only the weights are lower
//! precision). This retires the residual ~12 GiB TE-encode **active** peak that sc-12796 identified as
//! the 5B's binding constraint: the ~11 GB bf16 f32-TE was the largest staged component, and packing it
//! Q8 drops the encode peak to ~7.7 GiB (measured 11.83 → 7.72). It is a **numerics change** vs the bf16
//! baseline — "no quality regression," not bit-exact (the parity-reframe posture). The consuming build
//! **drains** each block's bf16 source as it packs it (sc-11030): without the drain the sources
//! accumulate to ~the full 11 GB (a no-drain probe peaked at 13.6 GiB); with it the LOAD-phase transient
//! stays bounded to ~one block's bf16 plus the accumulated packs — a distinct, earlier phase from the
//! 7.72 GiB encode-stage peak above (which additionally holds the dense bf16 embedding, materialized at
//! gather-time).

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::array::scalar;
use mlx_gen::nn::TokenEmbedding;
use mlx_gen::quant::{self, DEFAULT_GROUP_SIZE};
use mlx_gen::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_rs::fast::rms_norm;
use mlx_rs::ops::{add, matmul, multiply, softmax_axis, subtract};
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};

use crate::config::{WanModelConfig, WanQuant};

/// The dtype-preserving, golden-bit-exact tanh-GELU now lives in shared core `nn` (sc-2779) —
/// re-exported here so `text_encoder::gelu_tanh` (used by both the UMT5 FFN and the DiT FFN via
/// `transformer.rs`) keeps resolving. The UMT5 TE feeds it f32 (bit-exact); the DiT FFN feeds it
/// bf16 (preserved as bf16) — both matching `nn.GELU(approx="tanh")`.
pub(crate) use mlx_gen::nn::gelu_tanh;

/// The additive value the reference uses to mask padding keys (`F.softmax`'s `dtype.min` analogue;
/// `mx.where(mask == 0, -3.389e38, 0.0)`). Large enough that `exp` underflows to exactly 0.
const MASK_FILL: f32 = -3.389e38;

/// Build the tokenizer policy for the UMT5-XXL encoder: encode the (cleaned) prompt verbatim,
/// right-truncate + pad to `text_len` with pad id 0, emit the attention mask. The HF
/// `google/umt5-xxl` `tokenizer.json` post-processor appends the `</s>` (id 1) EOS and adds no BOS.
pub fn umt5_tokenizer_config(text_len: usize) -> TokenizerConfig {
    TokenizerConfig {
        max_length: text_len,
        pad_token_id: 0,
        chat_template: ChatTemplate::None,
        pad_to_max_length: true,
    }
}

/// UMT5-XXL encoder (Wan text conditioning).
pub struct Umt5Encoder {
    /// `[vocab, dim]` token table. Dense (bf16, gathered rows cast to f32) on every tier — the
    /// embedding is never in the DiT `_quantize_predicate`, and its ~2.1 GB is not the binding term
    /// once the projections are packed. Held as [`TokenEmbedding`] so a future pre-quantized-on-disk
    /// TE (`.scales` present) loads packed transparently.
    token_embedding: TokenEmbedding,
    blocks: Vec<Umt5Block>,
    final_norm_w: Array, // [dim], f32
    num_heads: usize,
    head_dim: usize,
    num_buckets: usize,
    eps: f32,
}

struct Umt5Block {
    norm1_w: Array, // f32
    // Bias-less projections — dense (`matmul(x, wᵀ)`, bit-identical to the reference) or packed Q4/Q8
    // (`quantized_matmul`, fp32-accumulate on the f32 activations). `AdaptableLinear` picks per its base.
    q: AdaptableLinear,
    k: AdaptableLinear,
    v: AdaptableLinear,
    o: AdaptableLinear,
    norm2_w: Array, // f32
    gate_proj: AdaptableLinear,
    fc1: AdaptableLinear,
    fc2: AdaptableLinear,
    pos_embedding: Array, // [num_buckets, num_heads], f32
}

impl Umt5Block {
    /// Load block `i` from the converted `t5_encoder.safetensors` keys. Projections load **dense**
    /// (bf16) or **packed** when `{key}.scales` is present (a pre-quantized-on-disk TE), via the shared
    /// [`quant::lin`] auto-detector (bias-less). Norms and the per-layer position-bias table upcast to
    /// f32 so `fast::rms_norm` and the bias add see f32 operands.
    fn from_weights(w: &Weights, i: usize) -> Result<Self> {
        let f32 = |a: &Array| -> Result<Array> { Ok(a.as_dtype(Dtype::Float32)?) };
        let p = format!("blocks.{i}");
        let lin = |name: &str| quant::lin(w, &format!("{p}.{name}"), false, DEFAULT_GROUP_SIZE);
        Ok(Self {
            norm1_w: f32(w.require(&format!("{p}.norm1.weight"))?)?,
            q: lin("attn.q")?,
            k: lin("attn.k")?,
            v: lin("attn.v")?,
            o: lin("attn.o")?,
            norm2_w: f32(w.require(&format!("{p}.norm2.weight"))?)?,
            gate_proj: lin("ffn.gate_proj")?,
            fc1: lin("ffn.fc1")?,
            fc2: lin("ffn.fc2")?,
            pos_embedding: f32(w.require(&format!("{p}.pos_embedding.embedding.weight"))?)?,
        })
    }

    /// Pack the seven projection/FFN linears to Q4/Q8 in place (the UMT5 analogue of the DiT
    /// `_quantize_predicate`). No-op on any linear already packed from disk.
    fn quantize(&mut self, bits: i32, group: i32) -> Result<()> {
        for lin in [
            &mut self.q,
            &mut self.k,
            &mut self.v,
            &mut self.o,
            &mut self.gate_proj,
            &mut self.fc1,
            &mut self.fc2,
        ] {
            lin.quantize(bits, Some(group))?;
        }
        Ok(())
    }

    /// Push this block's quantized packs (`wq`/`scales`/`biases`) for the sc-5360 eval-to-free pass; a
    /// linear still dense contributes nothing.
    fn push_quant_arrays(&self, out: &mut Vec<Array>) {
        for lin in [
            &self.q,
            &self.k,
            &self.v,
            &self.o,
            &self.gate_proj,
            &self.fc1,
            &self.fc2,
        ] {
            if let Some((wq, scales, biases, _, _, _)) = lin.quantized_params() {
                out.push(wq.clone());
                out.push(scales.clone());
                out.push(biases.clone());
            }
        }
    }
}

impl Umt5Encoder {
    /// **Dense** build from the converted `t5_encoder.safetensors` (the MLX-layout keys
    /// `token_embedding.weight`, `blocks.{i}.{norm1,norm2}.weight`, `blocks.{i}.attn.{q,k,v,o}.weight`,
    /// `blocks.{i}.ffn.{gate_proj,fc1,fc2}.weight`, `blocks.{i}.pos_embedding.embedding.weight`,
    /// `norm.weight`). The bf16-tier path — **bit-exact** to the reference (a bias-less dense
    /// [`AdaptableLinear`] forward is `matmul(x, wᵀ)`, byte-for-byte the prior raw-`matmul` path).
    pub fn from_weights(w: &Weights, cfg: &WanModelConfig) -> Result<Self> {
        let mut blocks = Vec::with_capacity(cfg.t5_num_layers);
        for i in 0..cfg.t5_num_layers {
            blocks.push(Umt5Block::from_weights(w, i)?);
        }
        let token_embedding = quant::embedding(w, "token_embedding", DEFAULT_GROUP_SIZE)?;
        Self::assemble(token_embedding, blocks, w, cfg)
    }

    /// **Quantized** build (sc-12831): pack the projection/FFN linears to `q.bits` while **consuming**
    /// `w` block-by-block — each block's dense bf16 source is dropped from `w` as soon as it has been
    /// packed (sc-11030 drain) + force-eval'd (sc-5360 eval-to-free), so the LOAD-phase transient stays
    /// bounded to ~one block instead of accumulating the full ~11 GB bf16 (a no-drain probe peaked at
    /// 13.6 GiB). That LOAD phase is distinct from the subsequent **encode** peak — packs + the dense
    /// bf16 embedding materialized at gather-time — which is the 11.83 → 7.72 GiB drop the story targets
    /// (at the production Q8 floor; `q.bits` is the caller's resolved width — see
    /// `model::effective_te_quant`). A **numerics change** vs the bf16 baseline — no quality regression,
    /// not bit-exact. The token embedding + norms stay dense (never in the DiT `_quantize_predicate`).
    pub fn from_weights_quantized(
        w: &mut Weights,
        cfg: &WanModelConfig,
        q: WanQuant,
    ) -> Result<Self> {
        let mut blocks = Vec::with_capacity(cfg.t5_num_layers);
        for i in 0..cfg.t5_num_layers {
            let mut block = Umt5Block::from_weights(w, i)?;
            block.quantize(q.bits, q.group_size)?;
            // Force the packs (+ the tiny f32 norm/pos tables, so they stop referencing their bf16
            // source) now, THEN drop w's dense refs for this block — releasing the block's bf16 before
            // the next one loads. Without the drain the sources accumulate to the full ~11 GB (measured
            // 13.6 GiB peak vs 4.1 with it).
            let mut arrays = vec![
                block.norm1_w.clone(),
                block.norm2_w.clone(),
                block.pos_embedding.clone(),
            ];
            block.push_quant_arrays(&mut arrays);
            eval(arrays.iter())?;
            w.remove_prefix(&format!("blocks.{i}."));
            blocks.push(block);
        }
        // Token embedding stays dense (its ~2.1 GB bf16 clone materializes at gather-time during encode).
        let token_embedding = quant::embedding(w, "token_embedding", q.group_size)?;
        Self::assemble(token_embedding, blocks, w, cfg)
    }

    /// Finish assembling the encoder from its built parts + the (still-dense) final norm. Shared by the
    /// dense and quantized constructors.
    fn assemble(
        token_embedding: TokenEmbedding,
        blocks: Vec<Umt5Block>,
        w: &Weights,
        cfg: &WanModelConfig,
    ) -> Result<Self> {
        Ok(Self {
            token_embedding,
            blocks,
            final_norm_w: w.require("norm.weight")?.as_dtype(Dtype::Float32)?,
            num_heads: cfg.t5_num_heads,
            head_dim: cfg.t5_dim_attn / cfg.t5_num_heads,
            num_buckets: cfg.t5_num_buckets,
            eps: 1e-6,
        })
    }

    /// Run the encoder over a `[1, L]` int32 id row + `[1, L]` mask → `[1, L, dim]` f32 hidden
    /// states. `L` is the padded length; callers slice to the non-pad prefix.
    pub fn forward(&self, ids: &Array, mask: &Array) -> Result<Array> {
        let seq = ids.shape()[1];
        // Token embedding: gather rows (dequantizing if packed), start the f32 activation stream.
        let mut x = self.token_embedding.forward(ids)?; // [1, L, dim] f32

        // Bucket grid (shared across layers) + additive padding mask (shared across layers).
        let buckets = self.bucket_grid(seq);
        let add_mask = additive_mask(mask)?; // [1, 1, 1, L] f32

        for block in &self.blocks {
            x = block.forward(
                &x,
                &buckets,
                &add_mask,
                self.num_heads,
                self.head_dim,
                self.eps,
            )?;
        }
        Ok(rms_norm(&x, &self.final_norm_w, self.eps)?)
    }

    /// Per-stage capture for parity bisection (S3 DiT reuses this template): returns the hidden
    /// state after the token embedding, after each block, and after the final norm.
    pub fn forward_capture(&self, ids: &Array, mask: &Array) -> Result<Vec<Array>> {
        let seq = ids.shape()[1];
        let mut x = self.token_embedding.forward(ids)?;
        let buckets = self.bucket_grid(seq);
        let add_mask = additive_mask(mask)?;
        let mut stages = vec![x.clone()];
        for block in &self.blocks {
            x = block.forward(
                &x,
                &buckets,
                &add_mask,
                self.num_heads,
                self.head_dim,
                self.eps,
            )?;
            stages.push(x.clone());
        }
        stages.push(rms_norm(&x, &self.final_norm_w, self.eps)?);
        Ok(stages)
    }

    /// Clean → tokenize → encode → drop padding, exactly as the reference `encode_text`. Returns the
    /// `[seq_len, dim]` non-pad prompt embedding the DiT consumes.
    pub fn encode(&self, tok: &TextTokenizer, prompt: &str) -> Result<Array> {
        let cleaned = clean_text(prompt);
        let out = tok.tokenize_preformatted(&cleaned)?;
        let (input_ids, attention_mask) = mlx_gen::tokenizer::to_arrays(&out);
        // Cast the mask to Int32 before summing: a bf16/float mask (depending on the tokenizer's
        // dtype) summed then cast to i32 could round at large seq lengths (F-046).
        let seq_len: i32 = attention_mask.as_dtype(Dtype::Int32)?.sum(None)?.item();
        let embeds = self.forward(&input_ids, &attention_mask)?;
        let dim = embeds.shape()[2];
        let flat = embeds.reshape(&[embeds.shape()[1], dim])?;
        let idx = Array::from_slice(&(0..seq_len).collect::<Vec<i32>>(), &[seq_len]);
        Ok(flat.take_axis(&idx, 0)?)
    }

    /// The `[seq, seq]` int32 bucket-index grid (`bucket[q][k] = bucket(k − q)`), built host-side.
    fn bucket_grid(&self, seq: i32) -> Array {
        let n = seq as usize;
        let mut data = Vec::with_capacity(n * n);
        for q in 0..seq {
            for k in 0..seq {
                data.push(relative_position_bucket(k - q, self.num_buckets as i32));
            }
        }
        Array::from_slice(&data, &[seq, seq])
    }
}

impl Umt5Block {
    fn forward(
        &self,
        x: &Array,
        buckets: &Array,
        add_mask: &Array,
        num_heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<Array> {
        // Self-attention (pre-norm, residual outside).
        let normed = rms_norm(x, &self.norm1_w, eps)?;
        let attn = self.self_attention(&normed, buckets, add_mask, num_heads, head_dim)?;
        let x = add(x, &attn)?;
        // Gated-GELU FFN (pre-norm, residual outside): fc2(fc1(h) · gelu_tanh(gate_proj(h))).
        let normed = rms_norm(&x, &self.norm2_w, eps)?;
        let gate = gelu_tanh(&self.gate_proj.forward(&normed)?)?;
        let up = self.fc1.forward(&normed)?;
        let ff = self.fc2.forward(&multiply(&up, &gate)?)?;
        Ok(add(&x, &ff)?)
    }

    fn self_attention(
        &self,
        x: &Array,
        buckets: &Array,
        add_mask: &Array,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<Array> {
        let seq = x.shape()[1];
        let h = num_heads as i32;
        let c = head_dim as i32;
        // [1, L, dim] → [1, heads, L, head_dim]
        let shape = |lin: &AdaptableLinear| -> Result<Array> {
            Ok(lin
                .forward(x)?
                .reshape(&[1, seq, h, c])?
                .transpose_axes(&[0, 2, 1, 3])?)
        };
        // T5 uses NO 1/sqrt(d) scaling — compute QKᵀ and softmax in f32 (acts are already f32; a packed
        // projection's `quantized_matmul` also returns f32, so the softmax precision is unaffected).
        let q = shape(&self.q)?;
        let k = shape(&self.k)?;
        let v = shape(&self.v)?;
        let scores = matmul(&q, &k.transpose_axes(&[0, 1, 3, 2])?)?; // [1, heads, L, L]
        let bias = self.position_bias(buckets)?; // [1, heads, L, L]
        let scores = add(&add(&scores, &bias)?, add_mask)?;
        let weights = softmax_axis(&scores, -1, true)?;
        let out = matmul(&weights, &v)? // [1, heads, L, head_dim]
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[1, seq, h * c])?;
        self.o.forward(&out)
    }

    /// Per-layer relative-position bias: `embedding[buckets]` → `[1, heads, L, L]`.
    fn position_bias(&self, buckets: &Array) -> Result<Array> {
        let seq = buckets.shape()[0];
        let flat = buckets.reshape(&[seq * seq])?;
        let embeds = self.pos_embedding.take_axis(&flat, 0)?; // [L*L, heads]
        let heads = embeds.shape()[1];
        Ok(embeds
            .reshape(&[seq, seq, heads])?
            .transpose_axes(&[2, 0, 1])? // [heads, L, L]
            .expand_dims(0)?) // [1, heads, L, L]
    }
}

/// `[1, L]` int/float mask → `[1, 1, 1, L]` f32 additive mask (`0` where kept, `MASK_FILL` where
/// padded). `(mask − 1) · |MASK_FILL|` gives `0`/`MASK_FILL` for mask ∈ {1, 0}.
fn additive_mask(mask: &Array) -> Result<Array> {
    let m = mask.as_dtype(Dtype::Float32)?;
    let seq = mask.shape()[1];
    let add = multiply(&subtract(&m, scalar(1.0))?, scalar(-MASK_FILL))?;
    Ok(add.reshape(&[1, 1, 1, seq])?)
}

/// T5 bucketing for `bidirectional=True` (`num_buckets`, `max_distance=128`). Port of
/// `T5RelativeEmbedding._relative_position_bucket`; matches the FLUX T5 bucket logic.
fn relative_position_bucket(relative_position: i32, num_buckets: i32) -> i32 {
    let max_distance = 128.0_f32;
    let half = num_buckets / 2;
    let mut bucket = 0;
    let mut n = relative_position;
    if n > 0 {
        bucket += half;
    }
    n = n.abs();
    let max_exact = half / 2;
    let val = if n < max_exact {
        n
    } else {
        let log_ratio = (n as f32 / max_exact as f32).ln() / (max_distance / max_exact as f32).ln();
        let large = max_exact + (log_ratio * (half - max_exact) as f32) as i32; // trunc == floor (≥0)
        large.min(half - 1)
    };
    bucket + val
}

// ── `_clean_text` (loading.py) ─────────────────────────────────────────────────────────────────
//
// Port of the reference `ftfy.fix_text(text)` → `html.unescape(html.unescape(text))` →
// `re.sub(r"\s+", " ", text).strip()`. ftfy is reproduced for the transforms a clean UTF-8 prompt
// actually exercises (verified bit-for-bit against `ftfy.fix_text` on a 19-case battery incl. the
// full Chinese negative prompt): block-scoped fullwidth/halfwidth fold (`fix_character_width`),
// latin-ligature expansion (`fix_latin_ligatures`), quote uncurling (`uncurl_quotes`), and final
// NFC normalization. ftfy's mojibake/encoding-repair fixes only trigger on *corrupted* input and
// are out of scope (a prompt is natural-language text, not mis-decoded bytes).

/// ftfy `LIGATURES` table (FB00–FB06). FB05 is the long-s + t ligature (`ſt`), not `st`.
const LIGATURES: &[(char, &str)] = &[
    ('\u{FB00}', "ff"),
    ('\u{FB01}', "fi"),
    ('\u{FB02}', "fl"),
    ('\u{FB03}', "ffi"),
    ('\u{FB04}', "ffl"),
    ('\u{FB05}', "\u{017F}t"),
    ('\u{FB06}', "st"),
];

/// ftfy `uncurl_quotes`: curly single/double quotes and primes → straight ASCII.
fn uncurl(ch: char) -> Option<char> {
    match ch {
        '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' | '\u{2032}' => Some('\''),
        '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}' | '\u{2033}' => Some('"'),
        _ => None,
    }
}

/// Clean a prompt exactly as the reference `_clean_text` does (see module note above).
pub fn clean_text(text: &str) -> String {
    use unicode_normalization::UnicodeNormalization;

    // 1. ftfy-equivalent: fullwidth fold + latin ligatures + uncurl quotes.
    let mut folded = String::with_capacity(text.len());
    for ch in text.chars() {
        if let Some((_, rep)) = LIGATURES.iter().find(|(c, _)| *c == ch) {
            folded.push_str(rep);
        } else if ch == '\u{3000}' || ('\u{FF00}'..='\u{FFEF}').contains(&ch) {
            // `fix_character_width`: NFKC scoped to the Halfwidth/Fullwidth Forms block + the
            // ideographic space (e.g. fullwidth comma `，` U+FF0C → ASCII `,`).
            folded.extend(ch.nfkc());
        } else if let Some(rep) = uncurl(ch) {
            folded.push(rep);
        } else {
            folded.push(ch);
        }
    }
    // 2. ftfy final `normalization='NFC'`.
    let normalized: String = folded.nfc().collect();
    // 3. Double HTML unescape.
    let unescaped = html_unescape(&html_unescape(&normalized));
    // 4. Collapse whitespace runs to a single space + strip.
    unescaped.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Minimal HTML entity decoder covering the entities a prompt realistically contains: the five
/// predefined XML entities, a handful of common named entities, and the full numeric forms
/// (`&#DDD;` / `&#xHHH;`). Exotic named entities (the full HTML5 table) are out of scope.
fn html_unescape(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'&' {
            // Copy this whole UTF-8 char.
            let ch_len = utf8_len(bytes[i]);
            out.push_str(&s[i..i + ch_len]);
            i += ch_len;
            continue;
        }
        // Find the terminating ';' within a small window.
        if let Some(semi) = s[i..].find(';').filter(|&p| p <= 32) {
            let entity = &s[i + 1..i + semi];
            if let Some(decoded) = decode_entity(entity) {
                out.push(decoded);
                i += semi + 1;
                continue;
            }
        }
        out.push('&');
        i += 1;
    }
    out
}

fn decode_entity(entity: &str) -> Option<char> {
    if let Some(num) = entity.strip_prefix('#') {
        let code = if let Some(hex) = num.strip_prefix(['x', 'X']) {
            u32::from_str_radix(hex, 16).ok()?
        } else {
            num.parse::<u32>().ok()?
        };
        return char::from_u32(code);
    }
    Some(match entity {
        "amp" => '&',
        "lt" => '<',
        "gt" => '>',
        "quot" => '"',
        "apos" => '\'',
        "nbsp" => '\u{00A0}',
        "copy" => '©',
        "reg" => '®',
        "trade" => '™',
        "hellip" => '…',
        "mdash" => '—',
        "ndash" => '–',
        _ => return None,
    })
}

fn utf8_len(first: u8) -> usize {
    match first {
        b if b < 0x80 => 1,
        b if b >> 5 == 0b110 => 2,
        b if b >> 4 == 0b1110 => 3,
        _ => 4,
    }
}

/// Load the UMT5-XXL tokenizer from a `tokenizer.json` (HF `google/umt5-xxl`).
pub fn load_tokenizer(path: impl AsRef<std::path::Path>, text_len: usize) -> Result<TextTokenizer> {
    TextTokenizer::from_file(path, umt5_tokenizer_config(text_len))
        .map_err(|e| Error::Msg(format!("wan umt5 tokenizer: {e}")))
}

/// Stage-1 UMT5 text encode shared by every Wan `generate_impl` (dense TI2V-5B, MoE A14B, VACE,
/// VACE-Fun): load the tokenizer + `t5_encoder.safetensors` from the snapshot `root`, build the
/// [`Umt5Encoder`], encode `prompt` (and, unless `skip_neg`, `neg_prompt`), then `eval` so the
/// encoder loads → is used → frees before the DiT loads. Returns `(context, context_null)`, with
/// `context_null = None` when `skip_neg` (CFG disabled). This was copied verbatim into all four
/// bodies (and the A14B copy had drifted to always-encode-both); centralizing it removes the drift —
/// the A14B caller passes `skip_neg = false` to keep its always-both behavior (F-010).
///
/// `te_quant` (sc-12831) packs the UMT5 projections to the DiT tier's bits — `Some` on a quantized
/// tier, `None` on the bf16 tier (the encoder stays dense / bit-exact). Private on purpose
/// (sc-12914): every caller must come through [`encode_text_staged_for_tier`], which resolves the
/// effective quant (the pre-quantized snapshot's `config.quantization`, else the load-time
/// `spec.quantize`) — so a provider cannot compile a TE stage that skips the tier resolution.
/// This is what retires the residual ~12 GiB f32-TE-encode active peak (sc-12796) on the quantized 5B.
fn encode_text_staged(
    root: &std::path::Path,
    cfg: &WanModelConfig,
    prompt: &str,
    neg_prompt: &str,
    skip_neg: bool,
    te_quant: Option<WanQuant>,
) -> Result<(Array, Option<Array>)> {
    let tokenizer = load_tokenizer(root.join("tokenizer.json"), cfg.text_len)?;
    let mut w = Weights::from_file(root.join("t5_encoder.safetensors"))?;
    // Dense (bf16 tier) stays the original all-clone build. Quantized (sc-12831) CONSUMES `w`,
    // draining each block's bf16 source as it packs it so the load transient stays ~the packed size —
    // the encode-stage bound lives entirely in `from_weights_quantized`, not in when `w` is dropped
    // (the encoder holds its own refs to the buffers it needs; `w` is a lazy alias of the same bytes).
    let enc = match te_quant {
        Some(q) => Umt5Encoder::from_weights_quantized(&mut w, cfg, q)?,
        None => Umt5Encoder::from_weights(&w, cfg)?,
    };
    let context = enc.encode(&tokenizer, prompt)?;
    let context_null = if skip_neg {
        None
    } else {
        Some(enc.encode(&tokenizer, neg_prompt)?)
    };
    match &context_null {
        Some(cn) => mlx_rs::transforms::eval([&context, cn])?,
        None => mlx_rs::transforms::eval([&context])?,
    }
    Ok((context, context_null))
}

/// Production tier-aware UMT5 staging shared by the dense, A14B, VACE, and VACE-Fun providers.
///
/// Exposed only so the real-weight acceptance gate can measure the exact production selection path;
/// callers should normally use a Wan provider rather than this probe seam.
#[doc(hidden)]
pub fn encode_text_staged_for_tier(
    root: &std::path::Path,
    cfg: &WanModelConfig,
    prompt: &str,
    neg_prompt: &str,
    skip_neg: bool,
    load_quant: Option<mlx_gen::Quant>,
) -> Result<(Array, Option<Array>)> {
    encode_text_staged(
        root,
        cfg,
        prompt,
        neg_prompt,
        skip_neg,
        crate::model::effective_te_quant(cfg, load_quant),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_match_reference_edges() {
        // Same bidirectional T5 bucketing as the FLUX T5 (num_buckets=32, max_dist=128).
        assert_eq!(relative_position_bucket(0, 32), 0);
        assert_eq!(relative_position_bucket(1, 32), 17);
        assert_eq!(relative_position_bucket(-1, 32), 1);
        assert_eq!(relative_position_bucket(128, 32), 31);
        assert_eq!(relative_position_bucket(-128, 32), 15);
    }

    #[test]
    fn clean_text_collapses_whitespace_and_unescapes() {
        assert_eq!(
            clean_text("  a  cat\tplaying\n piano "),
            "a cat playing piano"
        );
        assert_eq!(
            clean_text("fox &amp; hound &lt;tag&gt;"),
            "fox & hound <tag>"
        );
    }

    #[test]
    fn clean_text_folds_fullwidth_punctuation() {
        // The load-bearing case: the Chinese negative prompt's fullwidth commas → ASCII commas.
        assert_eq!(clean_text("色调艳丽，过曝"), "色调艳丽,过曝");
        assert_eq!(clean_text("ＡＢＣ１２３"), "ABC123");
        assert_eq!(clean_text("100％ ＃tag"), "100% #tag");
    }

    #[test]
    fn clean_text_uncurls_quotes_and_expands_ligatures() {
        assert_eq!(clean_text("“curly” ‘quotes’"), "\"curly\" 'quotes'");
        assert_eq!(clean_text("ﬁle ﬂag office"), "file flag office");
        // Em-dash and ellipsis are preserved (ftfy does not touch them).
        assert_eq!(clean_text("a — b…"), "a — b…");
    }

    #[test]
    fn clean_text_handles_numeric_entities() {
        assert_eq!(clean_text("A&#38;B"), "A&B");
        assert_eq!(clean_text("&#x41;&#x42;"), "AB");
    }
}
