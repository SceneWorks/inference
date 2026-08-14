#![allow(dead_code)]
//! Shared helpers. `mod common;` is compiled into every test binary, so only a subset is used by
//! any one of them.

use std::path::PathBuf;

use mlx_rs::Array;

use mlx_gen::weights::Weights;
use mlx_gen_minimax_h3::{BigVganConfig, MiniMaxH3AudioVaeConfig, MiniMaxH3VaeConfig};

/// The committed video-VAE parity fixture, produced by `tools/dump_minimax_h3_video_vae.py`
/// running the **official diffusers** `AutoencoderKLMiniMaxH3` — the converted-checkpoint layout
/// production loads. It carries the pre-conversion `src.` tensors alongside, so
/// `video_vae_parity.rs` can assert the committed weights really are in the converted layout
/// (sc-18740: a fixture dumped from the reference modules through a pure rename made the whole
/// suite a false green).
pub const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/video_vae_decode.safetensors"
);

/// The committed video-VAE **encode** fixture (sc-17148), produced by
/// `tools/dump_minimax_h3_video_vae_encode.py`.
///
/// # Why the encode half has its own file
///
/// Its geometry cannot be the decode fixture's. The encoder must not **crop**: a level that
/// strides time without also striding space convolves a 3-wide kernel with no spatial padding and
/// loses two columns instead of halving, which breaks the tiled encode outright (the stitch
/// assumes `latent = pixel / ratio`). The original MiniMax module asserts `time_stride in [1, 2]`,
/// so `patch_size_t` 4 needs TWO time-strided levels — each of which must therefore also be
/// spatial-strided, making the spatial cumprod 4 rather than [`FIXTURE`]'s 2.
///
/// Regenerating [`FIXTURE`] at that geometry was the obvious move and is the wrong one. Its bytes
/// are **shared verbatim with `candle-gen-minimax-h3`** (sc-17154), whose `cross_backend.rs`
/// digests them, and re-randomizing it perturbs every decoder weight — which measurably eroded
/// that crate's `a_gated_ffn_half_swap_is_loud_against_the_mlx_record` mutation gate from ~1.2e-1
/// to 8.4e-2 against its 1e-1 floor. A second file keeps this slice's change inside this slice.
pub const ENCODE_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/video_vae_encode.safetensors"
);

/// The committed audio parity fixture, produced by `tools/dump_minimax_h3_audio_vae.py` running
/// the same snapshot's `FL2VA/audio_vae` bundle (`DacAudioVAE` / `BigVGAN` / `SnakeBeta` /
/// `kaiser_sinc_filter1d`).
pub const AUDIO_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/audio_vae_decode.safetensors"
);

/// The committed text-encoder parity fixture, produced by `tools/dump_minimax_h3_te.py` running
/// the **transformers** `Qwen3VLTextModel` — an independent reference graph — at tiny dims.
///
/// Carries the select-layer context AND both neighbouring layers' contexts, so an off-by-one in
/// either direction is a failure rather than a plausible-looking tensor. Its safetensors metadata
/// additionally pins the **presentation** — the official conditioner's own
/// `tokenizer(prompt, add_special_tokens=False)` ids for a probe prompt, alongside the ids
/// sc-17143's chat-template render produced as an explicit negative control — plus the
/// special-token id map read from the shipped `tokenizer_config.json`.
pub const TE_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/te_context.safetensors"
);

/// The committed DiT parity fixture, produced by `tools/dump_minimax_h3_dit.py` running the
/// **official diffusers** `MiniMaxH3Transformer3DModel` — the converted-checkpoint layout
/// production loads (sc-18740's Rule 3).
///
/// Carries the pre-conversion `src.` tensors alongside the published ones — the raw per-head
/// interleaved fused QKV, its reordered `[q_all; k_all; v_all]` form, and the gate-first
/// `mlp.fc1` — each round-tripped through the OFFICIAL conversion functions in the generator, so
/// `dit_parity.rs` can assert which transform produced the published bytes rather than trusting a
/// comment. Its `layout.*` tensors are the reference's own `build_packed_sequence` output, which
/// is what pins the audio-token position convention.
pub const DIT_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/dit_block.safetensors"
);

/// The tiny DiT geometry the fixture was dumped at. Mirrors `dump_minimax_h3_dit.py`.
///
/// Two structural properties of the shipped model are preserved rather than shrunk away:
/// `heads · head_dim` (96) stays **wider** than `hidden_size` (64), as 7168 > 5376; and
/// `2 · 3 · rope_freq_dim` (12) stays **narrower** than `attention_head_dim` (24), as 96 < 128, so
/// the partial-rotary path is the one under test.
pub fn dit_fixture_config() -> mlx_gen_minimax_h3::MiniMaxH3DitConfig {
    mlx_gen_minimax_h3::MiniMaxH3DitConfig {
        num_attention_heads: 4,
        attention_head_dim: 24,
        hidden_size: 64,
        num_layers: 2,
        num_refiner_layers: 2,
        ffn_dim: 32,
        text_dim: 40,
        freq_dim: 16,
        time_embed_hidden_dim: 64,
        time_embed_dim: 48,
        rope_freq_dim: 2,
        ..mlx_gen_minimax_h3::MiniMaxH3DitConfig::default()
    }
}

/// The packed layout `dump_minimax_h3_dit.py` built the goldens over.
pub struct DitLayout {
    /// Text rows.
    pub num_text_tokens: i32,
    /// Audio latents **per channel**.
    pub num_audio_latents: i32,
    /// Channels the soundtrack is packed channel-major over.
    pub audio_channels: i32,
    /// Target video latent frames.
    pub num_latent_frames: i32,
    /// Target latent height.
    pub latent_height: i32,
    /// Target latent width.
    pub latent_width: i32,
}

/// 5 text rows, 3 audio latents over 2 channels, and 3 latent frames of a 4×6 latent at patch
/// `[1, 2, 2]` — 29 packed rows. Kept in one place so a fixture regenerated at different dims
/// fails loudly rather than drifting.
pub const DIT_LAYOUT: DitLayout = DitLayout {
    num_text_tokens: 5,
    num_audio_latents: 3,
    audio_channels: 2,
    num_latent_frames: 3,
    latent_height: 4,
    latent_width: 6,
};

/// The tiny text-encoder geometry the fixture was dumped at. Mirrors `dump_minimax_h3_te.py`:
/// `head_dim` deliberately != `hidden_size / num_heads` (32 vs 16), exactly as the real model has
/// 128 != 5120/64, and `select_hidden` < `num_layers` so the unused-tail trim is exercised.
pub fn te_fixture_config() -> mlx_gen_minimax_h3::MiniMaxH3TeConfig {
    mlx_gen_minimax_h3::MiniMaxH3TeConfig {
        hidden_size: 64,
        num_layers: 6,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 32,
        intermediate_size: 128,
        rms_norm_eps: 1e-6,
        rope_theta: 5_000_000.0,
        vocab_size: 256,
        select_hidden: 4,
        ..mlx_gen_minimax_h3::MiniMaxH3TeConfig::qwen3_vl_32b()
    }
}

/// The tiny audio geometry the fixture was dumped at.
///
/// Only the *width* is shrunk — `decoder_dim` 1024 → 128 (the smallest that survives all seven
/// halvings) and `latent_dim` 2048 → 64. Everything that drives the arithmetic is the shipped
/// value: `sample_rate` 32000, which is what selects the production BigVGAN branch, so all seven
/// upsample stages, the `[3,7,11]` × `[1,3,5]` AMP table, the 800× hop and the 12-tap Kaiser-sinc
/// filters are the real ones — as are the 32 latent channels and their de-normalization
/// statistics.
///
/// Built by hand rather than through `from_source_files`, because the dump passes `latent_dim`
/// explicitly instead of letting it derive from `encoder_dim · 2^len(encoder_rates)`; the
/// derivation itself is covered by the config unit tests and by the real-weight smoke.
pub fn audio_fixture_config() -> MiniMaxH3AudioVaeConfig {
    let shipped = MiniMaxH3AudioVaeConfig::default();
    MiniMaxH3AudioVaeConfig {
        sample_rate: 32_000,
        output_channels: 2,
        latent_channels: 32,
        encoder_dim: 8,
        encoder_rates: vec![2, 2],
        decoder_rates: vec![5, 5, 2, 2, 2, 2, 2],
        decoder_dim: 128,
        attn_proj: true,
        decoder_type: "bigvgan".into(),
        // The REAL 32-entry de-normalization statistics, not placeholders.
        latents_mean: shipped.latents_mean.clone(),
        latents_std: shipped.latents_std.clone(),
        bigvgan: BigVganConfig::for_sample_rate(32_000, 64, 128).expect("32 kHz is supported"),
    }
}

/// NCL `[B, C, T]` → MLX-native NLC `[B, T, C]`. The audio fixture stores the reference's NCL
/// tensors; the port's public API is NCL too, so this is only for the internal NLC modules.
pub fn to_nlc(x: &Array) -> Array {
    x.transpose_axes(&[0, 2, 1]).unwrap()
}

/// The tiny geometry the fixture was dumped at. Structurally identical to the shipped model —
/// same partial-rotary ratio, same 24 latent channels, and the **production** temporal knobs
/// (`clip_length 17`, `patch_size_t 4`) so the chunk plan is the real one; only the width and
/// depth are shrunk so the weights are committable.
pub fn fixture_config(token_drop: i32) -> MiniMaxH3VaeConfig {
    let shipped = MiniMaxH3VaeConfig::default();
    MiniMaxH3VaeConfig {
        latent_channels: 24,
        out_channels: 3,
        num_layers: 2,
        num_heads: 2,
        head_dim: 16,
        num_register_tokens: 4,
        ffn_mult: 4,
        rope_theta: 100.0,
        rope_dim_ratio: 0.75,
        norm_eps: 1e-5,
        patch_size: 2,
        patch_size_t: 4,
        clip_length: 17,
        token_drop,
        // The REAL 24-entry de-normalization statistics, not placeholders.
        latents_mean: shipped.latents_mean.clone(),
        latents_std: shipped.latents_std.clone(),
        // The encoder fields the struct gained in sc-17148. This fixture carries NO `encoder.*`
        // tensors — it is the decode golden and its bytes are shared with
        // `candle-gen-minimax-h3` — so these only have to satisfy `validate_encoder`'s
        // "the cumprods are the patch sizes" rule. `MiniMaxH3VideoVae::from_weights` loads the
        // encode half only when it is present; see [`ENCODE_FIXTURE`] for the one that has it.
        in_channels: 3,
        block_out_channels: vec![32, 32, 32, 32],
        layers_per_block: 1,
        spatial_downsample_factors: vec![2, 1, 1, 1],
        temporal_downsample_factors: vec![1, 2, 2, 1],
        norm_num_groups: 32,
        encoder_norm_eps: 1e-6,
    }
}

/// The geometry [`ENCODE_FIXTURE`] was dumped at, mirroring
/// `tools/dump_minimax_h3_video_vae_encode.py`.
///
/// Differs from [`fixture_config`] in exactly the ways the encode half forces (see
/// [`ENCODE_FIXTURE`]): every downsampling level is spatial-strided, so nothing crops and the
/// spatial cumprod — and therefore `patch_size` — is 4. The width also changes at the last level,
/// so `conv_shortcut` is actually built rather than left unexercised.
pub fn encode_fixture_config(token_drop: i32) -> MiniMaxH3VaeConfig {
    MiniMaxH3VaeConfig {
        patch_size: 4,
        block_out_channels: vec![32, 32, 32, 64],
        spatial_downsample_factors: vec![1, 2, 2, 1],
        temporal_downsample_factors: vec![1, 2, 2, 1],
        ..fixture_config(token_drop)
    }
}

/// The tile geometry [`ENCODE_FIXTURE`]'s tiled golden was dumped at — deliberately smaller than
/// the shipped 256/64 so a committable canvas spans more than one tile.
pub fn encode_fixture_tiles(f: &Weights) -> (i32, i32) {
    let t = f
        .require("const.encode_tile")
        .expect("the encode fixture carries const.encode_tile")
        .as_slice::<i32>()
        .to_vec();
    assert_eq!(t.len(), 2, "const.encode_tile is [tile_size, min_overlap]");
    (t[0], t[1])
}

/// `(max|a-b| / peak|b|, mean|a-b| / mean|b|)` over the full tensors — the peak- and
/// mean-relative error the parity gates use.
pub fn rel(a: &Array, b: &Array) -> (f32, f32) {
    let a = a
        .as_dtype(mlx_rs::Dtype::Float32)
        .unwrap()
        .flatten(None, None)
        .unwrap();
    let b = b
        .as_dtype(mlx_rs::Dtype::Float32)
        .unwrap()
        .flatten(None, None)
        .unwrap();
    let diff = mlx_rs::ops::subtract(&a, &b).unwrap().abs().unwrap();
    let bb = b.abs().unwrap();
    let peak: f32 = bb.max(None).unwrap().item();
    let mean: f32 = bb.mean(None).unwrap().item();
    let max_d: f32 = diff.max(None).unwrap().item();
    let mean_d: f32 = diff.mean(None).unwrap().item();
    (max_d / peak.max(1e-12), mean_d / mean.max(1e-12))
}

/// Cosine similarity between two tensors, flattened.
///
/// Reported alongside [`rel`] wherever a **layout** error is under test rather than a numeric one.
/// sc-18740's gate/value half-swap leaves the output norm essentially unchanged (89 vs 85 on real
/// weights) so magnitude, std and checksum assertions are all blind to it — but it also keeps
/// cosine at 0.73-0.78 rather than ~0, because `silu(a)·b` and `silu(b)·a` share sign structure.
/// **Neither metric alone is sufficient**: cosine is scale-invariant and the relative max-abs-diff
/// is the one that actually exposes the swap, so the tests print both and gate on both.
pub fn cosine(a: &Array, b: &Array) -> f32 {
    let a = a
        .as_dtype(mlx_rs::Dtype::Float32)
        .unwrap()
        .flatten(None, None)
        .unwrap();
    let b = b
        .as_dtype(mlx_rs::Dtype::Float32)
        .unwrap()
        .flatten(None, None)
        .unwrap();
    let dot: f32 = mlx_rs::ops::multiply(&a, &b)
        .unwrap()
        .sum(None)
        .unwrap()
        .item();
    let na: f32 = a.square().unwrap().sum(None).unwrap().item::<f32>().sqrt();
    let nb: f32 = b.square().unwrap().sum(None).unwrap().item::<f32>().sqrt();
    dot / (na * nb).max(1e-12)
}

/// L2 norm of a tensor — reported (never asserted on) in the half-swap tests, as the direct
/// demonstration that magnitude cannot see a layout error.
pub fn l2_norm(x: &Array) -> f32 {
    x.as_dtype(mlx_rs::Dtype::Float32)
        .unwrap()
        .square()
        .unwrap()
        .sum(None)
        .unwrap()
        .item::<f32>()
        .sqrt()
}

/// Assert `got` matches `want` within the parity tolerance, reporting the actual error.
pub fn assert_parity(got: &Array, want: &Array, tol: f32, what: &str) {
    assert_eq!(got.shape(), want.shape(), "{what}: shape");
    let (peak, mean) = rel(got, want);
    assert!(
        peak < tol,
        "{what}: peak-relative error {peak:.3e} (mean {mean:.3e}) exceeds {tol:.1e}"
    );
}

/// Standard deviation of a tensor — a golden whose expected output is ~constant proves nothing.
pub fn std_dev(x: &Array) -> f32 {
    let x = x.as_dtype(mlx_rs::Dtype::Float32).unwrap();
    let mean: f32 = x.mean(None).unwrap().item();
    let centered = mlx_rs::ops::subtract(&x, Array::from_f32(mean)).unwrap();
    let var: f32 = centered.square().unwrap().mean(None).unwrap().item();
    var.sqrt()
}

// --- sc-19449: the AdaLN precompute / denoise phase pair ---------------------------------------

/// Fraction of the evicted bytes the precompute→denoise drop has to account for, as `num/den`.
///
/// Integer arithmetic on purpose: these are byte counts in the tens of gigabytes, where an `f64`
/// round-trip is avoidable noise in a message a reviewer has to trust.
pub const ADALN_GAP_NUM: usize = 3;
/// Denominator of [`ADALN_GAP_NUM`].
pub const ADALN_GAP_DEN: usize = 4;

/// Fraction of the evicted bytes the precompute's own transient has to stay under, as `num/den`.
pub const ADALN_TRANSIENT_NUM: usize = 1;
/// Denominator of [`ADALN_TRANSIENT_NUM`].
pub const ADALN_TRANSIENT_DEN: usize = 2;

/// One AdaLN precompute-and-evict measurement: both phase peaks, and the residency either side of
/// the evict. Every field is MLX **active** bytes.
///
/// # Why this is a shared type rather than assertions written twice
///
/// `tests/adaln_evict_memory.rs` and `tests/adaln_evict_real_weights.rs` each computed
/// `peak_precompute`, printed it, and asserted **nothing** against it (sc-19449). Both existing
/// assertion sets pin the post-evict drop against the *pre-evict resident* — a residency, not a
/// phase peak — so the two tests positioned to measure the precompute/denoise relationship
/// measured it and let it go. That is how `convert.rs` came to carry the claim that AdaLN
/// precompute is free because "the precompute peak stays under the denoise peak … it holds,
/// comfortably", which sc-18659 retracted: the one in-repo measurement of the precompute side
/// (`memory_strategy::ADALN_EVICTED_BYTES`, 64.56 → 38.70 GB active) says the precompute side sits
/// *above* the residency that follows, i.e. the inequality ran the other way.
///
/// Keeping the relationship in one place means the synthetic and the real-weight measurement
/// cannot drift into pinning different things.
pub struct AdaLnPhases<'a> {
    /// Tier and denoise geometry this pair was taken at — both are load-bearing, see
    /// [`assert_adaln_phase_envelope`]. Printed on the record line.
    pub scale: &'a str,
    /// Active bytes with every `adaln_proj` still resident, before `precompute_and_evict`.
    pub active_before: usize,
    /// Active bytes after the evict and the allocator drain.
    pub active_after: usize,
    /// High-water of the precompute+evict phase — peak reset immediately before it, read
    /// immediately after.
    pub peak_precompute: usize,
    /// High-water of the phase that runs *after* the evict — peak reset again, then a real block
    /// forward against the cached table.
    pub peak_denoise: usize,
    /// Projection bytes `precompute_and_evict` reported releasing.
    pub released: usize,
    /// Bytes of the modulation table the precompute retains in their place.
    pub table: usize,
}

impl AdaLnPhases<'_> {
    /// `peak_precompute − peak_denoise`, saturating. The quantity sc-19449 is about.
    pub fn gap(&self) -> usize {
        self.peak_precompute.saturating_sub(self.peak_denoise)
    }

    /// What the precompute phase adds on top of the residency it starts from.
    pub fn precompute_transient(&self) -> usize {
        self.peak_precompute.saturating_sub(self.active_before)
    }

    /// What the denoise phase adds on top of the residency it starts from — activations, the rope
    /// tables, the gathered modulation. This is the term that moves with **geometry**.
    pub fn denoise_working_set(&self) -> usize {
        self.peak_denoise.saturating_sub(self.active_after)
    }

    fn x(&self, bytes: usize) -> f64 {
        bytes as f64 / (self.released.max(1)) as f64
    }

    /// The single greppable record line. Both the tier and the denoise geometry are on it because
    /// the bound below is only meaningful with them.
    pub fn record(&self) -> String {
        format!(
            "ADALN PHASE PAIR [{}]: peak_precompute {} B, peak_denoise {} B, gap {} B ({:.3}x \
             released); precompute transient {} B ({:.3}x), denoise working set {} B ({:.3}x); \
             active {} -> {} B, released {} B, table {} B",
            self.scale,
            self.peak_precompute,
            self.peak_denoise,
            self.gap(),
            self.x(self.gap()),
            self.precompute_transient(),
            self.x(self.precompute_transient()),
            self.denoise_working_set(),
            self.x(self.denoise_working_set()),
            self.active_before,
            self.active_after,
            self.released,
            self.table,
        )
    }
}

/// Pin the AdaLN precompute/denoise phase relationship (sc-19449).
///
/// # The relationship, and why it has the direction it has
///
/// `mlx_reset_peak_memory` sets the counter to **zero** — `MetalAllocator::reset_peak_memory` in
/// `mlx/backend/metal/allocator.h` is `peak_memory_ = 0`, not `= active_memory_` — and the
/// allocator raises it back to the then-current active on the very next `malloc`
/// (`peak_memory_ = std::max(peak_memory_, active_memory_)` in `allocator.cpp`). Both read at the
/// MLX revision this workspace pins. The precompute's first allocation therefore happens while
/// every `adaln_proj` is still resident, so the precompute phase's high-water cannot be below the
/// pre-evict residency — assertion (e). The denoise phase's window opens *after* the evict, so its
/// high-water is the post-evict residency plus its own working set. The gap between the two is
/// therefore the eviction, less whatever the denoise phase adds back:
///
/// ```text
/// gap  =  peak_precompute − peak_denoise  ≈  released − denoise_working_set
/// ```
///
/// # This is an envelope, not a universal inequality (sc-19449 AC6)
///
/// The identity above makes both dependences explicit, and neither is slack:
///
/// * **Geometry.** `denoise_working_set` grows with the packed sequence length. MLX's fused SDPA
///   working set tracks `4·B·H·S·D` and was measured at **5.966 GB** at `S = 104_030`, the render
///   geometry pinned in `mlx_gen_minimax_h3::memory_strategy` (rung 3 / sc-18661). Against a bf16
///   `released` of 26.02 GB that leaves `gap ≈ 0.77x` — barely above the `3/4` bound below. The
///   callers here measure at their own much smaller `SEQ`, where the term is negligible; the
///   geometry is on the record line for exactly that reason.
/// * **Tier.** `released` is the projections' bytes *at the tier they were loaded at*, so it falls
///   roughly fourfold at q4 while the denoise working set does not. Deriving from the same two
///   pinned numbers, a q4 render at full geometry would land near `gap ≈ 0.08x` — the bound below
///   would **not** hold there. It is not a tier-independent law and must not be copied into a q4
///   path without re-measuring.
///
/// The crossing point is `denoise_working_set = released`, and
/// `adaln_evict_memory.rs`'s third arm pins that shape by construction rather than by argument.
///
/// # Scale-free by construction
///
/// Every bound is a fraction of `released` and every message reports the same, so one function
/// serves the 144 MiB synthetic stack and the 26.02 GB real one without a per-scale constant.
pub fn assert_adaln_phase_envelope(p: &AdaLnPhases<'_>) {
    println!("  {}", p.record());

    // (e) STAGE WINDOW. The precompute phase's high-water is at least the residency it began
    //     with. This is what makes `peak_precompute` a *phase* reading at all: if the peak were
    //     reset after the evict, or read from the wrong side of it, this is the assertion that
    //     notices. Attributing a peak to the wrong stage is the error that produced two wrong
    //     handoffs on this epic, and nothing here could see it before.
    assert!(
        p.peak_precompute >= p.active_before,
        "the precompute phase peaked at {} B, BELOW the {} B that was resident when the phase \
         opened — the peak was not read over the precompute window, so nothing here is a \
         per-stage measurement",
        p.peak_precompute,
        p.active_before
    );

    // (f) DIRECTION AND MAGNITUDE. The precompute side is the high-water, not the denoise side —
    //     the negation of the claim sc-18659 retracted — and the gap is the eviction rather than
    //     noise. Entails `peak_precompute > peak_denoise`; stated as one assertion because a
    //     separate bare-direction assertion would be strictly implied by this one.
    let want_gap = p.released / ADALN_GAP_DEN * ADALN_GAP_NUM;
    assert!(
        p.gap() >= want_gap,
        "precompute peak {} B is only {} B above the denoise peak {} B ({:.3}x the {} B evicted, \
         wanted >= {}/{}). Either the precompute no longer spans the pre-evict residency, or the \
         denoise phase's own working set ({} B) has grown to the size of the eviction — at which \
         point this inequality is at its crossing point and the envelope, not the bound, is what \
         holds",
        p.peak_precompute,
        p.gap(),
        p.peak_denoise,
        p.x(p.gap()),
        p.released,
        ADALN_GAP_NUM,
        ADALN_GAP_DEN,
        p.denoise_working_set()
    );

    // (g) TRANSIENT CEILING. The corrected form of the retracted claim: the precompute is NOT
    //     free — it pays for the full pre-evict residency, which (e) pins — but its own transient
    //     stays well under what it releases, so the lever is net-positive at its own worst
    //     instant and the precompute stage is not a new process ceiling.
    //
    //     Deliberately loose. The transient is the retained table plus the projection-output
    //     intermediates, which at both callers' geometries is a small multiple of `table`; the
    //     real-weight value has NOT been measured (sc-19449's 62 GB run is outstanding) and a
    //     tight constant guessed here would cost a 62 GB run to a false red. The record line
    //     reports the measured ratio so that run can tighten it.
    let ceiling = p.released / ADALN_TRANSIENT_DEN * ADALN_TRANSIENT_NUM;
    assert!(
        p.precompute_transient() <= ceiling,
        "the precompute added {} B on top of the {} B it started from ({:.3}x the {} B it \
         releases, wanted <= {}/{}) — its transient is approaching the size of what it buys, so \
         the evict is no longer clearly net-positive at the precompute instant. Retained table is \
         {} B of it",
        p.precompute_transient(),
        p.active_before,
        p.x(p.precompute_transient()),
        p.released,
        ADALN_TRANSIENT_NUM,
        ADALN_TRANSIENT_DEN,
        p.table
    );
}

/// The MiniMax-H3 snapshot root, from `MINIMAX_H3_SNAPSHOT`.
///
/// Deliberately **panics** when unset. These are `#[ignore]`d tests: if this returned `None` and
/// the test quietly returned, `cargo test -- --ignored` would print `ok` in 0.00s and a reviewer
/// would read a skipped test as a passing one. Inference never derives a cache location or
/// self-fetches (epic 13657), so the path must be supplied explicitly.
pub fn snapshot() -> PathBuf {
    let raw = std::env::var("MINIMAX_H3_SNAPSHOT").unwrap_or_default();
    assert!(
        !raw.is_empty(),
        "MINIMAX_H3_SNAPSHOT must point at a MiniMaxAI/MiniMax-H3 snapshot root (the dir holding \
         `vae/`). This test is #[ignore]d and asserts rather than skips so that a missing \
         snapshot cannot be mistaken for a pass."
    );
    let path = PathBuf::from(raw);
    assert!(
        path.join("vae").join("config.json").is_file(),
        "MINIMAX_H3_SNAPSHOT={} has no vae/config.json",
        path.display()
    );
    path
}
