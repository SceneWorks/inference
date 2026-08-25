//! sc-17154: record **this lane's own decode** of the committed VAE goldens, so the candle twin can
//! be held against MLX numerically rather than only against the same reference.
//!
//! `candle-gen-minimax-h3` and `mlx-gen-minimax-h3` cannot coexist in one build — MLX is
//! macOS/Metal, candle-cuda is Windows/Linux — so a cross-backend assertion has to go through a
//! committed artifact. The repo's existing pattern for that is a *shared reference golden*: both
//! crates assert against the same `.safetensors` and the cross-backend claim is the sum of two
//! independent residuals. That is a real argument, but it is a bound, not a measurement.
//!
//! This generator closes the gap. It runs the MLX decoders on the fixture inputs and writes their
//! **outputs**; `candle-gen-minimax-h3/tests/cross_backend.rs` then compares candle's tensors to
//! MLX's directly and reports the residual between the two backends.
//!
//! It is `#[ignore]`d and **asserts rather than skips** on its output path, so a run that produced
//! nothing cannot print `ok` in 0.00s and read as a pass.
//!
//! ```sh
//! MINIMAX_H3_CROSS_BACKEND_OUT=/tmp/mlx_cross_backend.safetensors \
//!   cargo test -p mlx-gen-minimax-h3 --test integration cross_backend_record:: -- --ignored --nocapture
//! cp /tmp/mlx_cross_backend.safetensors \
//!   crates/media/candle-gen/candle-gen-minimax-h3/tests/fixtures/
//! ```
//!
//! The record's metadata carries an FNV-1a digest of each source fixture, and the candle side
//! asserts those match the fixtures it holds — so a regenerated golden cannot silently leave a
//! stale MLX record in place, which would turn the cross-backend gate into a comparison against
//! history.

use crate::common;

use std::collections::HashMap;

use common::{
    audio_fixture_config, dit_fixture_config, encode_fixture_config, encode_fixture_tiles,
    fixture_config, to_nlc, AUDIO_FIXTURE, DIT_FIXTURE, ENCODE_FIXTURE, FIXTURE,
};

use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, Result};
use mlx_gen_minimax_h3::audio_vae::{AmpBlock1, BigVgan};
use mlx_gen_minimax_h3::blocks::TransformerBlock;
use mlx_gen_minimax_h3::denoise::{
    adaln_schedule, denoise_av, JointGeometry, JointSchedule, JointStep, JointVelocity,
    PackedLayout, TEXT_TAG,
};
use mlx_gen_minimax_h3::dit::layers::DitAttention;
use mlx_gen_minimax_h3::dit::model::{BlockModulation, PackedForward};
use mlx_gen_minimax_h3::dit::positions::KeyframeAnchor;
use mlx_gen_minimax_h3::{
    DitBlock, MiniMaxH3AudioVae, MiniMaxH3Dit, MiniMaxH3VideoVae, MmRope, Rope3d, SnakeBeta,
    TokenRefiner, ViT3dDecoder,
};

/// The committed joint-denoise fixture (sc-17146), whose loop output the candle lane is held
/// against (sc-17155).
const DENOISE_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/av_denoise.safetensors"
);

/// Replays the denoise fixture's per-step velocities, so the recorded loop output is a property of
/// the scheduler and the row bookkeeping rather than of a model.
struct Replay {
    video: Vec<Array>,
    audio: Vec<Array>,
}

impl JointVelocity for Replay {
    fn forward(&mut self, step: &JointStep<'_>) -> Result<(Array, Array)> {
        Ok((
            self.video[step.index].clone(),
            self.audio[step.index].clone(),
        ))
    }
}

/// `[rows, F]` fixture tensor → the `[1, rows, F]` the loop takes.
fn batched(f: &Weights, key: &str) -> Array {
    let t = f.require(key).unwrap();
    let s = t.shape();
    t.reshape(&[1, s[0], s[1]]).unwrap()
}

/// FNV-1a over a file's bytes — a dependency-free binding between this record and the exact fixture
/// bytes it was produced from.
fn digest(path: &str) -> String {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{h:016x}")
}

fn model_weights(path: &str, drop: &[&str]) -> Weights {
    let mut w = Weights::from_file(path).unwrap();
    for prefix in drop {
        w.remove_prefix(prefix);
    }
    w
}

/// NLC `[B, T, C]` → NCL `[B, C, T]`. The audio record is written in the reference's NCL so the
/// candle side, which is NCL throughout, compares without reinterpreting anything.
fn to_ncl(x: &Array) -> Array {
    x.transpose_axes(&[0, 2, 1]).unwrap()
}

#[test]
#[ignore = "generator: writes the MLX cross-backend record to MINIMAX_H3_CROSS_BACKEND_OUT; needs Metal"]
fn record_mlx_decode_for_the_candle_cross_backend_gate() {
    let out_path = std::env::var("MINIMAX_H3_CROSS_BACKEND_OUT").unwrap_or_default();
    assert!(
        !out_path.is_empty(),
        "MINIMAX_H3_CROSS_BACKEND_OUT must name the .safetensors to write. This test is \
         #[ignore]d and asserts rather than skips so an unproductive run cannot read as a pass."
    );

    let mut record: Vec<(String, Array)> = Vec::new();

    // ---- video ------------------------------------------------------------------------------
    let f = Weights::from_file(FIXTURE).unwrap();
    let cfg = fixture_config(3);

    let mut w = model_weights(FIXTURE, &["src.", "in.", "out.", "const."]);
    let block = TransformerBlock::from_weights(
        &mut w,
        "decoder.transformer_blocks.0",
        &cfg,
        Dtype::Float32,
    )
    .unwrap();
    let rope = Rope3d::new(cfg.rope_apply_dim(), cfg.rope_theta).unwrap();
    let tables = rope.tables(f.require("in.block.ids").unwrap()).unwrap();
    record.push((
        "video.block.hidden".into(),
        block
            .forward(f.require("in.block.hidden").unwrap(), &rope, &tables)
            .unwrap(),
    ));
    record.push(("video.block.rope_cos".into(), tables.cos.clone()));

    let mut w = model_weights(FIXTURE, &["src.", "in.", "out.", "const."]);
    let decoder = ViT3dDecoder::from_weights(&mut w, "decoder", &cfg, Dtype::Float32).unwrap();
    record.push((
        "video.vit.video".into(),
        decoder
            .forward(f.require("in.vit.latent").unwrap())
            .unwrap(),
    ));

    let mut w = model_weights(FIXTURE, &["src.", "in.", "out.", "const."]);
    let vae = MiniMaxH3VideoVae::from_weights(&mut w, &cfg, Dtype::Float32).unwrap();
    for tokens in [7, 12] {
        record.push((
            format!("video.temporal{tokens}.video"),
            vae.decode(f.require(&format!("in.temporal{tokens}.latent")).unwrap())
                .unwrap(),
        ));
    }

    // ---- video ENCODE (sc-19008) --------------------------------------------------------------
    // A separate fixture with its own geometry — see `common::ENCODE_FIXTURE` for why the encode
    // half cannot share the decode goldens'. All five records go through `MiniMaxH3VideoVae`, so
    // the candle twin is held against this lane's own tiling, causal padding, frame-isolated
    // GroupNorm and posterior clamp rather than only against the shared reference.
    let ef = Weights::from_file(ENCODE_FIXTURE).unwrap();
    let ecfg = encode_fixture_config(3);
    let (etile, eoverlap) = encode_fixture_tiles(&ef);
    let mut ew = model_weights(ENCODE_FIXTURE, &["src.", "in.", "out.", "const."]);
    let evae = MiniMaxH3VideoVae::from_weights(&mut ew, &ecfg, Dtype::Float32).unwrap();
    assert!(
        evae.can_encode(),
        "the encode fixture must carry `encoder.*` / `quant_conv.*`"
    );
    let eclip = ef.require("in.encode_clip.pixels").unwrap();
    // A tile wider than the canvas degenerates to a single span — that IS the untiled path.
    record.push((
        "video.encode.params".into(),
        evae.encode_clip_tiled(eclip, 4096, 64).unwrap(),
    ));
    record.push((
        "video.encode_tiled.params".into(),
        evae.encode_clip_tiled(eclip, etile, eoverlap).unwrap(),
    ));
    let keyframe = evae
        .encode(ef.require("in.encode_single.pixels").unwrap())
        .unwrap();
    record.push(("video.encode_single.mean".into(), keyframe.mean().clone()));
    record.push(("video.encode_single.std".into(), keyframe.std().clone()));
    record.push((
        "video.encode_chunked.mean".into(),
        evae.encode(ef.require("in.encode_chunked.pixels").unwrap())
            .unwrap()
            .mean()
            .clone(),
    ));

    // ---- audio ------------------------------------------------------------------------------
    let af = Weights::from_file(AUDIO_FIXTURE).unwrap();
    let acfg = audio_fixture_config();

    let snake = SnakeBeta::new(
        af.require("in.snake.alpha").unwrap().clone(),
        af.require("in.snake.beta").unwrap().clone(),
        true,
    )
    .unwrap();
    record.push((
        "audio.snake.log".into(),
        to_ncl(
            &snake
                .forward(&to_nlc(af.require("in.snake.x").unwrap()))
                .unwrap(),
        ),
    ));

    let mut amp = Weights::from_map(
        af.keys()
            .filter(|k| k.starts_with("amp."))
            .map(|k| (k.to_string(), af.require(k).unwrap().clone()))
            .collect::<HashMap<_, _>>(),
    );
    let block =
        AmpBlock1::from_weights(&mut amp, "amp", 7, &[1, 3, 5], true, Dtype::Float32).unwrap();
    record.push((
        "audio.amp.y".into(),
        to_ncl(
            &block
                .forward(&to_nlc(af.require("in.amp.x").unwrap()))
                .unwrap(),
        ),
    ));

    let mut w = model_weights(AUDIO_FIXTURE, &["in.", "out.", "const.", "amp."]);
    let vocoder = BigVgan::from_weights(&mut w, "decoder", &acfg, Dtype::Float32).unwrap();
    record.push((
        "audio.bigvgan.y".into(),
        to_ncl(
            &vocoder
                .forward(&to_nlc(af.require("in.bigvgan.x").unwrap()))
                .unwrap(),
        ),
    ));

    let mut w = model_weights(AUDIO_FIXTURE, &["in.", "out.", "const.", "amp."]);
    let avae = MiniMaxH3AudioVae::from_weights(&mut w, &acfg, Dtype::Float32).unwrap();
    record.push((
        "audio.decode.audio".into(),
        avae.decode(af.require("in.decode.z").unwrap()).unwrap(),
    ));
    let z = af.require("in.stereo.z").unwrap();
    record.push((
        "audio.stereo.audio".into(),
        avae.decode_stereo(&avae.denormalize(z).unwrap()).unwrap(),
    ));

    // ---- DiT + joint denoise (sc-17155) -------------------------------------------------------
    // The candle lane implements the DiT, the AdaLN precompute/evict and the joint loop, and is
    // held against THIS lane's numbers rather than only against the shared reference. Everything
    // below runs at the committed fixtures' tiny geometry.
    let df = Weights::from_file(DIT_FIXTURE).unwrap();
    let dcfg = dit_fixture_config();
    let dit_weights = || {
        let mut w = Weights::from_file(DIT_FIXTURE).unwrap();
        for prefix in ["src.", "in.", "out.", "layout."] {
            w.remove_prefix(prefix);
        }
        w
    };
    let position_ids = df.require("layout.position_ids").unwrap();
    let rope = MmRope::new(dcfg.rope_freq_dim, dcfg.rope_theta).unwrap();
    let tables = rope.tables(position_ids).unwrap();
    record.push(("dit.rope_cos".into(), tables.cos.clone()));
    record.push(("dit.rope_sin".into(), tables.sin.clone()));

    let adaln_idx = df
        .require("layout.adaln_indices")
        .unwrap()
        .as_dtype(Dtype::Int32)
        .unwrap();
    let temb = df.require("in.temb").unwrap().clone();

    let mut w = dit_weights();
    let attn =
        DitAttention::from_weights(&mut w, "transformer_blocks.0.attn", &dcfg, Dtype::Float32)
            .unwrap();
    record.push((
        "dit.attn.hidden".into(),
        attn.forward(
            df.require("in.attn.hidden").unwrap(),
            Some((&rope, &tables)),
        )
        .unwrap(),
    ));

    let mut w = dit_weights();
    let block =
        DitBlock::from_weights(&mut w, "transformer_blocks.0", &dcfg, Dtype::Float32).unwrap();
    record.push((
        "dit.block.hidden".into(),
        block
            .forward_with_temb(
                df.require("in.block.hidden").unwrap(),
                &temb,
                &adaln_idx,
                &rope,
                &tables,
            )
            .unwrap(),
    ));

    let mut w = dit_weights();
    let refiner =
        TokenRefiner::from_weights(&mut w, "token_refiner", &dcfg, Dtype::Float32).unwrap();
    record.push((
        "dit.refiner.hidden".into(),
        refiner
            .forward(df.require("in.refiner.hidden").unwrap())
            .unwrap(),
    ));

    // The whole model — the gate that covers the 17 projections and their composition.
    let mut w = dit_weights();
    let dit = MiniMaxH3Dit::from_weights(&mut w, &dcfg, Dtype::Float32).unwrap();
    let text_rows = dit
        .embed_context(df.require("in.refiner.context").unwrap())
        .unwrap();
    record.push(("dit.refined_context".into(), text_rows.clone()));
    let norm_out = dit.projections().norm_out.modulation(&temb).unwrap();
    let idx = |k: &str| -> Vec<i32> {
        df.require(k)
            .unwrap()
            .as_dtype(Dtype::Int32)
            .unwrap()
            .as_slice::<i32>()
            .to_vec()
    };
    let (text_indices, video_indices, audio_indices) = (
        idx("layout.text_indices"),
        idx("layout.video_indices"),
        idx("layout.audio_indices"),
    );
    let timestep_indices = df
        .require("layout.timestep_indices")
        .unwrap()
        .as_dtype(Dtype::Int32)
        .unwrap();
    let video_rows = df.require("in.model.video_rows").unwrap().clone();
    let audio_rows = df.require("in.model.audio_rows").unwrap().clone();
    let packed = PackedForward {
        video_rows: &video_rows,
        audio_rows: &audio_rows,
        text_rows: &text_rows,
        adaln_indices: &adaln_idx,
        timestep_indices: &timestep_indices,
        tables: &tables,
        text_indices: &text_indices,
        video_indices: &video_indices,
        audio_indices: &audio_indices,
    };
    let (video_velocity, audio_velocity) = dit
        .forward_packed(&packed, BlockModulation::Temb(&temb), &norm_out)
        .unwrap();
    record.push(("dit.model.video_velocity".into(), video_velocity));
    record.push(("dit.model.audio_velocity".into(), audio_velocity));

    // The joint denoise loop, replayed over the reference's own velocities.
    let nf = Weights::from_file(DENOISE_FIXTURE).unwrap();
    let layout = PackedLayout::build(
        JointGeometry::new(124, 4, 6).unwrap(),
        [1, 2, 2],
        &[TEXT_TAG; 5],
        2,
        &[KeyframeAnchor::First, KeyframeAnchor::Last],
    )
    .unwrap();
    let joint = JointSchedule::new(3).unwrap();
    let adaln = adaln_schedule(&joint).unwrap();
    let mut model = Replay {
        video: (0..2)
            .map(|i| batched(&nf, &format!("in.video_velocity.{i}")))
            .collect(),
        audio: (0..2)
            .map(|i| batched(&nf, &format!("in.audio_velocity.{i}")))
            .collect(),
    };
    let (dv, da) = denoise_av(
        &mut model,
        &layout,
        &joint,
        &adaln,
        &batched(&nf, "in.video_latents"),
        &batched(&nf, "in.audio_latents"),
        &CancelFlag::default(),
        &mut |_| {},
    )
    .unwrap();
    record.push(("denoise.video_latents".into(), dv));
    record.push(("denoise.audio_latents".into(), da));

    // ---- write ------------------------------------------------------------------------------
    for (name, array) in &record {
        assert!(
            array.size() > 0,
            "{name} is empty; the record would gate nothing"
        );
        let finite: bool = array.is_finite().unwrap().all(None).unwrap().item();
        assert!(finite, "{name} contains non-finite values");
    }

    let mut metadata: HashMap<String, String> = HashMap::new();
    metadata.insert("backend".into(), "mlx".into());
    metadata.insert("dtype".into(), "float32".into());
    metadata.insert("story".into(), "sc-17154, sc-17155, sc-19008".into());
    metadata.insert("video_fixture_fnv1a64".into(), digest(FIXTURE));
    metadata.insert(
        "video_encode_fixture_fnv1a64".into(),
        digest(ENCODE_FIXTURE),
    );
    metadata.insert("audio_fixture_fnv1a64".into(), digest(AUDIO_FIXTURE));
    metadata.insert("dit_fixture_fnv1a64".into(), digest(DIT_FIXTURE));
    metadata.insert("denoise_fixture_fnv1a64".into(), digest(DENOISE_FIXTURE));

    Array::save_safetensors(
        record.iter().map(|(k, v)| (k.as_str(), v)),
        &metadata,
        &out_path,
    )
    .unwrap();

    let written = std::fs::metadata(&out_path).unwrap().len();
    println!(
        "wrote {} MLX cross-backend tensors ({written} bytes) to {out_path}\n  video fixture \
         fnv1a64 {}\n  audio fixture fnv1a64 {}",
        record.len(),
        metadata["video_fixture_fnv1a64"],
        metadata["audio_fixture_fnv1a64"],
    );
    assert!(written > 1024, "the record is implausibly small");
}
