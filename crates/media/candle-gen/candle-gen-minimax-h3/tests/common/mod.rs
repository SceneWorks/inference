#![allow(dead_code)]
//! Shared helpers. `mod common;` is compiled into every test binary, so only a subset is used by
//! any one of them.

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::Weights;

use candle_gen_minimax_h3::{BigVganConfig, MiniMaxH3AudioVaeConfig, MiniMaxH3VaeConfig};

/// The committed video-VAE parity fixture, produced by the MLX lane's
/// `tools/dump_minimax_h3_video_vae.py` running the **official diffusers**
/// `AutoencoderKLMiniMaxH3` — the converted-checkpoint layout production loads. It carries the
/// pre-conversion `src.` tensors alongside, so `video_vae_parity.rs` can assert the committed
/// weights really are in the converted layout (sc-18740: a fixture dumped from the reference
/// modules through a pure rename made the whole MLX suite a false green).
pub const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/video_vae_decode.safetensors"
);

/// The committed video-VAE **encode** fixture (sc-19008), produced by the MLX lane's
/// `tools/dump_minimax_h3_video_vae_encode.py` running the official diffusers
/// `AutoencoderKLMiniMaxH3` — the converted-checkpoint layout, and copied byte-for-byte from the
/// MLX lane (`cross_backend.rs` asserts that).
///
/// # Why the encode half has its own file
///
/// Its geometry cannot be [`FIXTURE`]'s. The encoder must not **crop**: a level that strides time
/// without also striding space convolves a 3-wide kernel with no spatial padding and loses two
/// columns instead of halving, which breaks the tiled encode outright (the stitch assumes
/// `latent = pixel / ratio`). The original MiniMax module asserts `time_stride in [1, 2]`, so
/// `patch_size_t` 4 needs TWO time-strided levels — each of which must therefore also be
/// spatial-strided, making the spatial cumprod 4 rather than [`FIXTURE`]'s 2.
///
/// Regenerating [`FIXTURE`] at that geometry was the obvious move and is the wrong one: its bytes
/// are shared across both lanes and `cross_backend.rs` digests them, and re-randomizing it
/// perturbs every decoder weight — which measurably eroded this crate's
/// `a_gated_ffn_half_swap_is_loud_against_the_mlx_record` mutation gate from ~1.2e-1 to 8.4e-2
/// against its 1e-1 floor the last time it happened.
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

/// The committed audio-VAE **encode** fixture (sc-17157), produced by the MLX lane's
/// `tools/dump_minimax_h3_audio_vae_encode.py` running the official
/// `diffusers.AutoencoderKLMiniMaxH3Audio` — the only executable reference for this half, because
/// the snapshot's own `FL2VA/audio_vae` bundle is inference-only and defines no `encode`.
///
/// **Copied byte-for-byte from the MLX lane**, which `cross_backend.rs` asserts: that shared-golden
/// identity is the cross-backend agreement argument for the Ref2VA soundtrack path, since MLX and
/// candle cannot coexist in one process.
///
/// Its geometry is deliberately *harder* than the shipped model's: `encoder_rates = (2, 5)` keeps
/// an ODD stride (whose `padding = ceil(stride / 2)` is what makes the shipped chain land on
/// exactly `samples / 800`), and `num_attention_heads = 2` puts the adaptive pool at a ragged
/// 48 → 32 with **overlapping** windows rather than the shipped exact 256 → 32.
pub const AUDIO_ENCODE_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/audio_vae_encode.safetensors"
);

/// The MLX lane's copy of the same two goldens. `cross_backend.rs` asserts this crate's copies are
/// **byte-identical** to them: the cross-backend parity argument is that both ports are held to the
/// same reference tensors, and two fixtures that had silently drifted apart would break that
/// without breaking either suite on its own.
pub const MLX_FIXTURE_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../mlx-gen/mlx-gen-minimax-h3/tests/fixtures"
);

/// The committed record of the **MLX lane's own decode** of those goldens, written by
/// `mlx-gen-minimax-h3`'s `tests/cross_backend_record.rs` (an `#[ignore]`d generator that runs on
/// Metal). `cross_backend.rs` compares this port's tensors against it directly, so the cross-backend
/// claim is a measurement rather than a triangle bound through the reference.
pub const MLX_RECORD: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/mlx_cross_backend.safetensors"
);

/// The committed **DiT** parity fixture (sc-17155), produced by
/// `crates/media/mlx-gen/tools/dump_minimax_h3_dit.py` running the official diffusers
/// `MiniMaxH3Transformer3DModel` — the converted-checkpoint layout production loads.
///
/// Carries the pre-conversion `src.` tensors alongside the published ones — the raw per-head
/// interleaved fused QKV, its reordered `[q_all; k_all; v_all]` form, and the gate-first
/// `mlp.fc1` — each round-tripped through the OFFICIAL conversion functions in the generator, so
/// `dit_parity.rs` can assert which transform produced the published bytes rather than trusting a
/// comment. Its `layout.*` tensors are the reference's own `build_packed_sequence` output, which is
/// what pins the audio-token position convention.
///
/// **Committed byte-identical to the MLX lane's copy**, which `cross_backend.rs` asserts.
pub const DIT_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/dit_block.safetensors"
);

/// The committed **joint-denoise** fixture (sc-17155), produced by
/// `crates/media/mlx-gen/tools/dump_minimax_h3_av_denoise.py` running the official diffusers
/// `MiniMaxH3Scheduler` — two instances loaded from the *published* `scheduler/` and
/// `audio_scheduler/` configs, so the 12.0 / 3.0 pair is read from the same bytes production reads
/// rather than typed twice.
pub const DENOISE_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/av_denoise.safetensors"
);

/// The tiny DiT geometry the fixture was dumped at. Mirrors `dump_minimax_h3_dit.py`, and is the
/// same set of numbers the MLX lane's `dit_fixture_config` uses — the two lanes must agree about
/// the fixture's shape before either compares a tensor.
///
/// Structurally identical to the shipped model where it matters: `rope_freq_dim` 2 still makes the
/// rotary **partial** (12 of 24 head channels), `inner_dim` (96) is still **wider** than
/// `hidden_size` (64), and the patch, modality count and modulation-parameter count are the real
/// ones. Only the widths and depths are shrunk so the weights are committable.
pub fn dit_fixture_config() -> candle_gen_minimax_h3::MiniMaxH3DitConfig {
    candle_gen_minimax_h3::MiniMaxH3DitConfig {
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
        ..candle_gen_minimax_h3::MiniMaxH3DitConfig::default()
    }
}

/// The packed-sequence shape the DiT fixture's `layout.*` tensors were built at.
pub struct DitLayout {
    pub num_text_tokens: usize,
    pub num_audio_latents: usize,
    pub audio_channels: usize,
    pub num_latent_frames: usize,
    pub latent_height: usize,
    pub latent_width: usize,
}

/// The DiT fixture's layout, mirroring `dump_minimax_h3_dit.py`.
pub const DIT_LAYOUT: DitLayout = DitLayout {
    num_text_tokens: 5,
    num_audio_latents: 3,
    audio_channels: 2,
    num_latent_frames: 3,
    latent_height: 4,
    latent_width: 6,
};

// -------------------------------------------------------------------------------------------
// A hand-rolled safetensors reader
// -------------------------------------------------------------------------------------------

struct Entry {
    dtype: String,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

/// A committed golden, read by parsing the safetensors container directly.
///
/// `candle_core::safetensors` **drops `__metadata__`**, and the video fixture's provenance record
/// (`provenance`, `reference`, `gated_ffn_layout`, …) is the sc-18740 Rule-3 gate — a fixture that
/// cannot be shown to come from the converted-checkpoint path is one this suite must refuse. So the
/// header is parsed here, exactly as `candle-gen-bernini`'s goldens are.
pub struct Golden {
    data: Vec<u8>,
    data_start: usize,
    entries: HashMap<String, Entry>,
    meta: HashMap<String, String>,
}

impl Golden {
    /// Load a fixture from an absolute path.
    pub fn load(path: &str) -> Golden {
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        assert!(bytes.len() > 8, "{path}: truncated safetensors container");
        let hlen = u64::from_le_bytes(bytes[0..8].try_into().expect("8 bytes")) as usize;
        let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + hlen])
            .unwrap_or_else(|e| panic!("{path}: parse safetensors header: {e}"));
        let obj = header.as_object().expect("header object");

        let mut entries = HashMap::new();
        let mut meta = HashMap::new();
        for (k, v) in obj {
            if k == "__metadata__" {
                for (mk, mv) in v.as_object().expect("metadata object") {
                    meta.insert(mk.clone(), mv.as_str().unwrap_or_default().to_string());
                }
                continue;
            }
            let dtype = v["dtype"].as_str().expect("dtype").to_string();
            let shape: Vec<usize> = v["shape"]
                .as_array()
                .expect("shape")
                .iter()
                .map(|x| x.as_u64().expect("dim") as usize)
                .collect();
            let offs = v["data_offsets"].as_array().expect("data_offsets");
            entries.insert(
                k.clone(),
                Entry {
                    dtype,
                    shape,
                    start: offs[0].as_u64().expect("start") as usize,
                    end: offs[1].as_u64().expect("end") as usize,
                },
            );
        }
        Golden {
            data_start: 8 + hlen,
            data: bytes,
            entries,
            meta,
        }
    }

    /// Whether the fixture carries `key`.
    pub fn has(&self, key: &str) -> bool {
        self.entries.contains_key(key)
    }

    /// Every tensor key, sorted.
    pub fn keys(&self) -> BTreeSet<String> {
        self.entries.keys().cloned().collect()
    }

    /// A safetensors `__metadata__` entry, or `None`.
    pub fn meta(&self, key: &str) -> Option<&str> {
        self.meta.get(key).map(|s| s.as_str())
    }

    /// Declared shape of `key`.
    pub fn shape(&self, key: &str) -> Vec<usize> {
        self.entries
            .get(key)
            .unwrap_or_else(|| panic!("missing tensor {key}"))
            .shape
            .clone()
    }

    fn raw(&self, key: &str) -> &[u8] {
        let e = self
            .entries
            .get(key)
            .unwrap_or_else(|| panic!("missing tensor {key}"));
        &self.data[self.data_start + e.start..self.data_start + e.end]
    }

    /// The f32 values of `key`, in logical (row-major) order.
    pub fn f32(&self, key: &str) -> Vec<f32> {
        let e = self
            .entries
            .get(key)
            .unwrap_or_else(|| panic!("missing tensor {key}"));
        assert_eq!(e.dtype, "F32", "{key} is {}, not F32", e.dtype);
        self.raw(key)
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
            .collect()
    }

    /// `key` as a CPU f32 tensor.
    pub fn tensor(&self, key: &str) -> Tensor {
        Tensor::from_vec(self.f32(key), self.shape(key), &Device::Cpu)
            .unwrap_or_else(|e| panic!("tensor {key}: {e}"))
    }

    /// `key` as a `[rows, cols]` f32 tensor reshaped to the `[1, rows, cols]` the packed-sequence
    /// paths take.
    pub fn batched(&self, key: &str) -> Tensor {
        let s = self.shape(key);
        assert_eq!(s.len(), 2, "{key}: expected a [rows, features] tensor");
        Tensor::from_vec(self.f32(key), (1, s[0], s[1]), &Device::Cpu)
            .unwrap_or_else(|e| panic!("tensor {key}: {e}"))
    }

    /// An index tensor the generator stored as float32, as the `u32` candle's `index_select` takes.
    ///
    /// The DiT fixture's whole `layout.*` namespace is written as f32 (safetensors from a numpy
    /// dump), so every index the port consumes has to be narrowed here rather than in the port —
    /// which is deliberate: the crate's own index tensors are `u32` end to end, and a test that
    /// silently accepted floats would be exercising a conversion production never performs.
    pub fn indices(&self, key: &str) -> Tensor {
        Tensor::from_vec(self.u32_vec(key), (self.shape(key)[0],), &Device::Cpu)
            .unwrap_or_else(|e| panic!("indices {key}: {e}"))
    }

    /// An `int32` tensor's values as a host slice. The encode fixture stores its tile geometry
    /// this way (`const.encode_tile`), so [`Self::f32`]'s dtype assertion would reject it.
    pub fn i32_vec(&self, key: &str) -> Vec<i32> {
        let e = self
            .entries
            .get(key)
            .unwrap_or_else(|| panic!("missing tensor {key}"));
        assert_eq!(e.dtype, "I32", "{key} is {}, not I32", e.dtype);
        self.raw(key)
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().expect("4 bytes")))
            .collect()
    }

    /// The same values as a host slice.
    pub fn u32_vec(&self, key: &str) -> Vec<u32> {
        self.f32(key)
            .into_iter()
            .map(|v| {
                assert!(
                    v >= 0.0 && v.fract() == 0.0,
                    "{key}: {v} is not a non-negative integer index"
                );
                v as u32
            })
            .collect()
    }

    /// Every tensor whose key does NOT start with one of `drop_prefixes`, as a weight map.
    ///
    /// For the video fixture that is exactly the model weights in the published root naming; the
    /// `src.` / `in.` / `out.` / `const.` namespaces are reference-side extras.
    pub fn model_map(&self, drop_prefixes: &[&str]) -> HashMap<String, Tensor> {
        self.entries
            .keys()
            .filter(|k| !drop_prefixes.iter().any(|p| k.starts_with(p)))
            .map(|k| (k.clone(), self.tensor(k)))
            .collect()
    }

    /// Every tensor whose key starts with `prefix`, keys unchanged.
    pub fn prefixed_map(&self, prefix: &str) -> HashMap<String, Tensor> {
        self.entries
            .keys()
            .filter(|k| k.starts_with(prefix))
            .map(|k| (k.clone(), self.tensor(k)))
            .collect()
    }
}

/// Wrap a weight map as a `candle_gen::Weights`.
pub fn weights(map: HashMap<String, Tensor>) -> Weights {
    Weights::from_map(map)
}

// -------------------------------------------------------------------------------------------
// Fixture geometry — mirrors the dump scripts
// -------------------------------------------------------------------------------------------

/// The tiny geometry the video fixture was dumped at. Structurally identical to the shipped model
/// — same partial-rotary ratio, same 24 latent channels, and the **production** temporal knobs
/// (`clip_length 17`, `patch_size_t 4`) so the chunk plan is the real one; only the width and depth
/// are shrunk so the weights are committable.
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
        // The encoder half of this fixture's geometry. `video_vae_decode.safetensors` carries no
        // `encoder.*` tensors at all, so nothing loads against these — they are here because
        // `MiniMaxH3VaeConfig::validate` covers both halves and the shape must be self-consistent.
        // See [`ENCODE_FIXTURE`] for why the encode goldens needed a different one.
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
/// `tools/dump_minimax_h3_video_vae_encode.py`. Identical to the MLX lane's
/// `encode_fixture_config` — the two backends must agree about the fixture's shape before either
/// compares a tensor.
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
/// the shipped 256/64 so a committable canvas spans more than one tile. Read from the fixture
/// rather than typed here, so a regenerated fixture cannot silently disagree with the test.
pub fn encode_fixture_tiles(g: &Golden) -> (usize, usize) {
    let t = g.i32_vec("const.encode_tile");
    assert_eq!(t.len(), 2, "const.encode_tile is [tile_size, min_overlap]");
    assert!(t[0] > 0 && t[1] >= 0, "nonsensical tile geometry {t:?}");
    (t[0] as usize, t[1] as usize)
}

/// The tiny audio geometry the fixture was dumped at.
///
/// Only the *width* is shrunk — `decoder_dim` 1024 → 128 (the smallest that survives all seven
/// halvings) and `latent_dim` 2048 → 64. Everything that drives the arithmetic is the shipped
/// value: `sample_rate` 32000, which is what selects the production BigVGAN branch, so all seven
/// upsample stages, the `[3,7,11]` × `[1,3,5]` AMP table, the 800× hop and the 12-tap Kaiser-sinc
/// filters are the real ones — as are the 32 latent channels and their de-normalization statistics.
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
        latents_mean: shipped.latents_mean.clone(),
        latents_std: shipped.latents_std.clone(),
        bigvgan: BigVganConfig::for_sample_rate(32_000, 64, 128).expect("32 kHz is supported"),
    }
}

// -------------------------------------------------------------------------------------------
// Metrics
// -------------------------------------------------------------------------------------------

/// Flatten a tensor to f32 in logical order.
pub fn flat(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .expect("f32")
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("vec")
}

/// `(max|a-b| / peak|b|, mean|a-b| / mean|b|)` over the full tensors — the peak- and mean-relative
/// error the parity gates use.
///
/// **Gate on the peak-relative form.** Five separate defects in this epic had cosine ≥ 0.98 or an
/// unchanged norm; the relative max-abs-diff is the metric that actually moves.
pub fn rel(a: &Tensor, b: &Tensor) -> (f32, f32) {
    let a = flat(a);
    let b = flat(b);
    assert_eq!(a.len(), b.len(), "length mismatch");
    let peak = b.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-12);
    let mean = (b.iter().map(|v| v.abs()).sum::<f32>() / b.len() as f32).max(1e-12);
    let max_d = a
        .iter()
        .zip(&b)
        .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
    let mean_d = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len() as f32;
    (max_d / peak, mean_d / mean)
}

/// Cosine similarity between two tensors, flattened.
///
/// Reported alongside [`rel`] wherever a **layout** error is under test rather than a numeric one.
/// sc-18740's gate/value half-swap leaves the output norm essentially unchanged (89 vs 85 on real
/// weights) so magnitude, std and checksum assertions are all blind to it — but it also keeps
/// cosine at 0.73-0.78 rather than ~0, because `silu(a)·b` and `silu(b)·a` share sign structure.
/// **Neither metric alone is sufficient**: cosine is scale-invariant and the relative max-abs-diff
/// is the one that actually exposes the swap, so the tests print both and gate on the latter.
pub fn cosine(a: &Tensor, b: &Tensor) -> f32 {
    let a = flat(a);
    let b = flat(b);
    let dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb).max(1e-12)
}

/// L2 norm — reported (never asserted on) in the half-swap tests, as the direct demonstration that
/// magnitude cannot see a layout error.
pub fn l2_norm(x: &Tensor) -> f32 {
    flat(x).iter().map(|v| v * v).sum::<f32>().sqrt()
}

/// Standard deviation — a golden whose expected output is ~constant proves nothing.
pub fn std_dev(x: &Tensor) -> f32 {
    let v = flat(x);
    let mean = v.iter().sum::<f32>() / v.len() as f32;
    (v.iter().map(|a| (a - mean) * (a - mean)).sum::<f32>() / v.len() as f32).sqrt()
}

/// Assert `got` matches `want` within the parity tolerance, printing the actual error first so a
/// failing run reports the real numbers rather than only the bound.
pub fn assert_parity(got: &Tensor, want: &Tensor, tol: f32, what: &str) {
    assert_eq!(got.dims(), want.dims(), "{what}: shape");
    let (peak, mean) = rel(got, want);
    println!("  {what}: peak rel {peak:.3e} (mean {mean:.3e}, tol {tol:.1e})");
    assert!(
        peak < tol,
        "{what}: peak-relative error {peak:.3e} (mean {mean:.3e}) exceeds {tol:.1e}"
    );
}

// -------------------------------------------------------------------------------------------
// Real weights
// -------------------------------------------------------------------------------------------

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

/// Read a file from the snapshot, asserting rather than returning an `Option`.
pub fn read_snapshot_file(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
}

/// The keys of a safetensors file, read from its header only (no weight I/O).
pub fn safetensors_keys(path: &Path) -> BTreeSet<String> {
    let mut file =
        std::fs::File::open(path).unwrap_or_else(|e| panic!("opening {}: {e}", path.display()));
    use std::io::Read;
    let mut len = [0u8; 8];
    file.read_exact(&mut len)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let hlen = u64::from_le_bytes(len) as usize;
    let mut header = vec![0u8; hlen];
    file.read_exact(&mut header)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let json: serde_json::Value = serde_json::from_slice(&header)
        .unwrap_or_else(|e| panic!("{}: header: {e}", path.display()));
    json.as_object()
        .expect("header object")
        .keys()
        .filter(|k| *k != "__metadata__")
        .cloned()
        .collect()
}
