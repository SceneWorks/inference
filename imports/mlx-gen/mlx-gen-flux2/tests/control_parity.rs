//! sc-2292: exact-correctness gates for the FLUX.2-dev Fun-Controlnet-Union (VACE) control branch,
//! on the committed tiny synthetic base transformer (`tests/fixtures/transformer_golden.safetensors`,
//! the sc-2346 S3 fixture) + an in-memory synthetic control branch. No real weights — these prove the
//! *mechanism* (hint injection + the `control_context_scale` knob) is wired correctly via two
//! invariants:
//!
//!   (a) **`scale = 0` is the base forward.** With `control_context_scale = 0` every control hint is
//!       multiplied by 0 and added (`+0`), so `forward_with_control(scale = 0)` is byte-identical to
//!       the plain `forward` — regardless of the (here random) control weights. This is the parity
//!       self-check the Z-Image control port uses (sc-2257).
//!
//!   (b) **`scale > 0` actually injects.** With non-zero control weights + `scale = 0.8` the output
//!       *differs* from the base forward (the hints flow into the base image stream) and stays finite
//!       — proving the control branch is a real contribution, not a silent no-op.

use mlx_gen::weights::Weights;
use mlx_gen_flux2::{
    Flux2Config, Flux2ControlBranch, Flux2ControlTransformer, Flux2Transformer, CONTROL_IN_DIM,
};
use mlx_rs::ops::{abs, array_eq, max, subtract};
use mlx_rs::Array;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/transformer_golden.safetensors"
);

const TS: f32 = 500.0;

/// The tiny config the base fixture was dumped with (inner = 2·8 = 16), shared with
/// `kv_cache_parity.rs` / `transformer_parity.rs`. `num_double_layers = 1` → `control_layers = [0]`
/// → one control block injecting at base double block 0.
fn tiny_config() -> Flux2Config {
    Flux2Config {
        num_double_layers: 1,
        num_single_layers: 1,
        num_heads: 2,
        head_dim: 8,
        in_channels: 4,
        out_channels: 4,
        joint_attention_dim: 12,
        mlp_ratio: 3.0,
        timestep_channels: 16,
        axes_dim: [2, 2, 2, 2],
        rope_theta: 2000.0,
        te_hidden_size: 4,
        te_intermediate_size: 12,
        te_out_layers: [0, 1, 2],
        max_sequence_length: 512,
        num_latent_channels: 1,
        vae_scale_factor: 8,
    }
}

/// Deterministic pseudo-random fill — a bounded sine pattern so the synthetic control branch is
/// non-degenerate (every hint is non-zero) but small enough to keep the forward finite.
fn filled(shape: &[i32], scale: f32, phase: f32) -> Array {
    let n: i32 = shape.iter().product();
    let v: Vec<f32> = (0..n)
        .map(|i| ((i as f32) * 0.0137 + phase).sin() * scale)
        .collect();
    Array::from_slice(&v, shape)
}

/// Build a synthetic control checkpoint (un-prefixed keys) for `tiny_config`: `control_img_in` +
/// one control block (a full double block + `before_proj` + `after_proj`). `inner = 16`,
/// `mlp_hidden = 3·16 = 48` (SwiGLU `linear_in` is `2·mlp_hidden`), `head_dim = 8`. The control
/// context width is `control_latents (in) + mask (in/latent) + inpaint (in)` — for the tiny config
/// `4 + 4 + 4 = 12`.
fn synthetic_control_weights() -> (Weights, i32) {
    let inner = 16;
    let head_dim = 8;
    let mlp_hidden = 48; // mlp_ratio 3.0 × inner 16
    let control_in = tiny_config().in_channels as i32
        + tiny_config().in_channels as i32 / tiny_config().num_latent_channels as i32
        + tiny_config().in_channels as i32; // 4 + 4 + 4 = 12

    let mut w = Weights::empty();
    w.insert(
        "control_img_in.weight",
        filled(&[inner, control_in], 0.05, 0.1),
    );
    w.insert("control_img_in.bias", filled(&[inner], 0.02, 0.2));

    let p = "control_transformer_blocks.0";
    for n in [
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out",
        "attn.add_q_proj",
        "attn.add_k_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
    ] {
        w.insert(
            format!("{p}.{n}.weight"),
            filled(&[inner, inner], 0.05, 0.3),
        );
    }
    for n in [
        "attn.norm_q",
        "attn.norm_k",
        "attn.norm_added_q",
        "attn.norm_added_k",
    ] {
        // RMSNorm weights near 1.0 so the norm is meaningful.
        let v: Vec<f32> = (0..head_dim)
            .map(|i| 1.0 + ((i as f32) * 0.0137 + 0.4).sin() * 0.05)
            .collect();
        w.insert(
            format!("{p}.{n}.weight"),
            Array::from_slice(&v, &[head_dim]),
        );
    }
    for ff in ["ff", "ff_context"] {
        w.insert(
            format!("{p}.{ff}.linear_in.weight"),
            filled(&[2 * mlp_hidden, inner], 0.05, 0.5),
        );
        w.insert(
            format!("{p}.{ff}.linear_out.weight"),
            filled(&[inner, mlp_hidden], 0.05, 0.6),
        );
    }
    for proj in ["before_proj", "after_proj"] {
        w.insert(
            format!("{p}.{proj}.weight"),
            filled(&[inner, inner], 0.05, 0.7),
        );
        w.insert(format!("{p}.{proj}.bias"), filled(&[inner], 0.02, 0.8));
    }
    (w, control_in)
}

struct Fixture {
    control: Flux2ControlTransformer,
    base: Flux2Transformer,
    hidden: Array,          // [1, target_seq, in_channels]
    encoder: Array,         // [1, txt_seq, joint]
    img_ids: Array,         // [1, target_seq, 4]
    txt_ids: Array,         // [1, txt_seq, 4]
    control_context: Array, // [1, target_seq, control_in]
}

impl Fixture {
    fn load() -> Self {
        let cfg = tiny_config();
        let w = Weights::from_file(FIXTURE).unwrap();
        let base = Flux2Transformer::from_weights(&w, &cfg).unwrap();
        let base2 = Flux2Transformer::from_weights(&w, &cfg).unwrap();
        let (control_w, control_in) = synthetic_control_weights();
        let branch = Flux2ControlBranch::from_weights(&control_w, "", &cfg).unwrap();
        let control = Flux2ControlTransformer::new(base2, branch);

        let hidden = w.require("hidden").unwrap().clone();
        let encoder = w.require("encoder").unwrap().clone();
        let target_seq = hidden.shape()[1];
        let txt_seq = encoder.shape()[1];
        // The control context shares the target image grid (same seq), here a deterministic fill.
        let control_context = filled(&[1, target_seq, control_in], 0.3, 1.1);
        Self {
            control,
            base,
            hidden,
            encoder,
            img_ids: mlx_gen_flux2::prepare_grid_ids(1, target_seq as usize, 0),
            txt_ids: mlx_gen_flux2::prepare_text_ids(txt_seq as usize),
            control_context,
        }
    }

    fn base_forward(&self) -> Array {
        self.base
            .forward(
                &self.hidden,
                &self.encoder,
                &self.img_ids,
                &self.txt_ids,
                TS,
            )
            .unwrap()
    }

    fn control_forward(&self, scale: f32) -> Array {
        self.control
            .forward(
                &self.hidden,
                &self.encoder,
                &self.img_ids,
                &self.txt_ids,
                TS,
                None, // tiny base fixture has no guidance embedder
                &self.control_context,
                scale,
            )
            .unwrap()
    }
}

fn exact_eq(a: &Array, b: &Array) -> bool {
    a.shape() == b.shape() && array_eq(a, b, false).unwrap().item::<bool>()
}

fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    let diff = subtract(a, b).unwrap();
    max(abs(diff).unwrap(), None).unwrap().item::<f32>()
}

#[test]
fn control_in_dim_is_260_for_dev() {
    // The dev control context width: control latent (128) + mask (4) + inpaint latent (128).
    let c = Flux2Config::dev();
    let in_ch = c.in_channels as i32;
    let mask_ch = in_ch / c.num_latent_channels as i32;
    assert_eq!(in_ch + mask_ch + in_ch, CONTROL_IN_DIM);
    assert_eq!(CONTROL_IN_DIM, 260);
}

#[test]
fn scale_zero_equals_base_forward() {
    let f = Fixture::load();
    let base = f.base_forward();
    let control0 = f.control_forward(0.0);
    assert_eq!(control0.shape(), base.shape());
    assert!(
        exact_eq(&base, &control0),
        "control_context_scale = 0 must be byte-identical to the base forward (hints ×0); \
         max|Δ| = {}",
        max_abs_diff(&base, &control0)
    );
}

#[test]
fn scale_nonzero_injects_and_stays_finite() {
    let f = Fixture::load();
    let base = f.base_forward();
    let controlled = f.control_forward(0.8);
    assert_eq!(controlled.shape(), base.shape());
    // The hints flow → the output differs from the base forward.
    let d = max_abs_diff(&base, &controlled);
    assert!(
        d > 1e-6,
        "scale = 0.8 must change the output (hints injected); max|Δ| = {d}"
    );
    // …and stays finite (no NaN/inf from the control branch).
    let peak = max(abs(controlled).unwrap(), None).unwrap().item::<f32>();
    assert!(
        peak.is_finite(),
        "controlled output must be finite, got peak {peak}"
    );
}
