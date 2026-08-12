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
//!   cargo test -p mlx-gen-minimax-h3 --test cross_backend_record -- --ignored --nocapture
//! cp /tmp/mlx_cross_backend.safetensors \
//!   crates/media/candle-gen/candle-gen-minimax-h3/tests/fixtures/
//! ```
//!
//! The record's metadata carries an FNV-1a digest of each source fixture, and the candle side
//! asserts those match the fixtures it holds — so a regenerated golden cannot silently leave a
//! stale MLX record in place, which would turn the cross-backend gate into a comparison against
//! history.

mod common;

use std::collections::HashMap;

use common::{audio_fixture_config, fixture_config, to_nlc, AUDIO_FIXTURE, FIXTURE};

use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_minimax_h3::audio_vae::{AmpBlock1, BigVgan};
use mlx_gen_minimax_h3::blocks::TransformerBlock;
use mlx_gen_minimax_h3::{MiniMaxH3AudioVae, MiniMaxH3VideoVae, Rope3d, SnakeBeta, ViT3dDecoder};

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
    metadata.insert("story".into(), "sc-17154".into());
    metadata.insert("video_fixture_fnv1a64".into(), digest(FIXTURE));
    metadata.insert("audio_fixture_fnv1a64".into(), digest(AUDIO_FIXTURE));

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
