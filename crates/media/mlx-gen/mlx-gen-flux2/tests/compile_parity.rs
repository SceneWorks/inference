//! sc-2963 invariant (rollout of the Wan sc-2957 template): the **compiled elementwise glue**
//! ([`set_compile_glue(true)`]) produces a transformer forward within the pinned MLX numerical
//! contract. `mx.compile` fuses the adaLN affine, SwiGLU activation, gated residual, and complex RoPE
//! rotation into single kernels. Under MLX 0.32, bf16 fused glue remains exact while f32 may round
//! 1–2 ULP differently from eager. This gates the whole double+single-block composition on the
//! committed tiny synthetic f32 config — in CI, no real checkpoint.

use mlx_gen::weights::Weights;
use mlx_gen_flux2::transformer::set_compile_glue;
use mlx_gen_flux2::{Flux2Config, Flux2Transformer};
use mlx_rs::{Array, Dtype};

/// Peak-relative f32 allowance for the tiny two-block whole-forward composition. The shared
/// primitive contract is four ULP; the repository's established shallow whole-forward budget is
/// four times that while remaining orders of magnitude below a wrong fusion.
const COMPILED_GLUE_F32_WHOLE_FWD_ULP_TOL: f32 = 4.0 * mlx_gen::nn::COMPILED_GLUE_F32_ULP_TOL;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/transformer_golden.safetensors"
);

/// The tiny config the golden dump used (mirrors `tests/transformer_parity.rs`).
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

fn max_abs(got: &Array, exp: &Array) -> f32 {
    let d = mlx_rs::ops::abs(mlx_rs::ops::subtract(got, exp).unwrap()).unwrap();
    mlx_rs::ops::max(&d, None).unwrap().item::<f32>()
}

#[test]
fn compiled_glue_matches_eager_within_f32_contract() {
    let w = Weights::from_file(FIXTURE).unwrap();
    let t = Flux2Transformer::from_weights(&w, &tiny_config()).unwrap();

    let run = || {
        t.forward(
            w.require("hidden").unwrap(),
            w.require("encoder").unwrap(),
            w.require("img_ids").unwrap(),
            w.require("txt_ids").unwrap(),
            500.0,
        )
        .unwrap()
    };

    set_compile_glue(false);
    let eager = run();
    set_compile_glue(true);
    let compiled = run();
    set_compile_glue(false);

    assert_eq!(compiled.shape(), eager.shape(), "shape");
    assert_eq!(eager.dtype(), Dtype::Float32, "fixture output dtype");
    assert_eq!(compiled.dtype(), eager.dtype(), "compiled output dtype");
    let d = max_abs(&compiled, &eager);
    let rel = mlx_gen::nn::max_rel_diff(&compiled, &eager);
    println!("[flux2 compiled vs eager] max|Δ|={d:.3e} rel={rel:.3e}");
    assert!(
        rel <= COMPILED_GLUE_F32_WHOLE_FWD_ULP_TOL,
        "FLUX.2 compiled glue rel|Δ|={rel:e} exceeds the f32 whole-forward bound {:e}",
        COMPILED_GLUE_F32_WHOLE_FWD_ULP_TOL
    );
}
