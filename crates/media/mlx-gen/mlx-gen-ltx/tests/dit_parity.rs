//! S3b full-DiT (velocity) parity vs the reference video-only `LTXModel` (sc-2679 S3b).
//!
//! `#[ignore]`d: needs the real `ltx_2_3_base_q8` `transformer.safetensors` (~20 GB). The committed
//! golden (`tests/fixtures/ltx_dit_golden.safetensors`, from `tools/dump_ltx_dit_golden.py`) holds
//! the reference **f32-activation × Q8** velocity over synthetic inputs; this test loads the SAME Q8
//! weights (kept quantized → `quantized_matmul`) and checks the Rust `LtxDiT` reproduces it.
//!
//! **The golden MUST match the Rust build's MLX — now 0.32.0** (epic 12742, `pmetal-mlx-rs`
//! 932beb4e): `quantized_matmul` changed 0.31.0→0.31.2 (a 0.31.0 golden mismatches by ~5e-4/op) and
//! again on 0.31.2→0.32.0 (sc-12744/sc-12747; the whole-forward velocity moved peak_rel ≈ 1.1e-1 —
//! per-op ULP drift chaos-amplified over 48 layers). The committed golden was re-dumped on the
//! **0.32.0 non-NAX env** (sc-12896: mlx 0.32.0 built from source at `MACOSX_DEPLOYMENT_TARGET=15.0`
//! into `~/Repos/mflux/.venv-0320`; see `tools/dump_ltx_dit_golden.py`).
//!
//! **Contract on 0.32.0 (sc-12896):** the **bf16-activation path stays bit-exact** (peak_rel =
//! mean_rel = 0.0, asserted). The **f32-activation path is no longer cross-stack bit-exact**: 0.32.0's
//! rewritten steel/quant kernels select shape-dependent variants, so the Python reference and the Rust
//! port (which fuse/batch some Linears differently) accumulate ~1-ULP per-op differences over the
//! 48-layer residual — measured peak_rel 6.228e-5 / mean_rel 4.383e-5 at matched 0.32.0 (both stacks
//! provably non-NAX, byte-identical weights + inputs; the output head alone is still 0.0, isolating
//! the drift to in-block accumulation). The f32 gate therefore uses the tight measured bounds below.
//! History (sc-2842): a host-f64 timestep table once seeded a real ~0.9% divergence — that class
//! (named, fixable op bugs) lands orders of magnitude above these bounds and is still caught.
//!
//! Run: `LTX_BASE_DIR=… cargo test -p mlx-gen-ltx --test dit_parity -- --ignored --nocapture`

use mlx_rs::ops::{abs, max as max_op, subtract, sum};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_ltx::config::{LtxConfig, SplitModel};
use mlx_gen_ltx::transformer::{LtxDiT, Precision};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/ltx_dit_golden.safetensors"
);

/// sc-12896: cross-stack (Python-reference vs Rust-port) bound for the **f32-activation × Q8**
/// whole-DiT forward at matched MLX 0.32.0. Measured peak_rel 6.228e-5 / mean_rel 4.383e-5
/// (2026-07-18, both stacks non-NAX dt15.0, byte-identical weights/inputs, fresh 0.32.0 golden);
/// bounds are ~4× the measurement. A real regression (wrong op/scale/routing — e.g. the sc-2842
/// timestep-table bug at ~0.9%, or a stale/requantized checkpoint at ~1e-1) lands far above.
/// bf16 stays exact — do NOT reuse these for the bf16 gate.
const DIT_F32_XSTACK_PEAK_REL: f32 = 2.5e-4;
const DIT_F32_XSTACK_MEAN_REL: f32 = 1.5e-4;
/// The reference's **native bf16+Q8** velocity golden (`LTX_BF16=1 dump_ltx_dit_golden.py`) — the
/// production-precision target for [`Precision::quant_bf16`].
const GOLDEN_BF16: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/ltx_dit_golden_bf16.safetensors"
);

fn base_dir() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("LTX_BASE_DIR") {
        return d.into();
    }
    let home = std::env::var("HOME").unwrap();
    std::path::PathBuf::from(home)
        .join("Library/Application Support/SceneWorks/data/models/mlx/ltx_2_3_base_q8")
}

fn f32(x: &Array) -> Array {
    x.as_dtype(Dtype::Float32).unwrap()
}

/// `max|Δ| / max|ref|`.
fn peak_rel(got: &Array, want: &Array) -> f32 {
    let diff = abs(subtract(f32(got), want).unwrap()).unwrap();
    let denom = max_op(abs(want).unwrap(), None).unwrap().item::<f32>();
    max_op(&diff, None).unwrap().item::<f32>() / denom.max(1e-12)
}

/// `Σ|Δ| / Σ|ref|` — robust to the output-LayerNorm-amplified massive-activation channels.
fn mean_rel(got: &Array, want: &Array) -> f32 {
    let num = sum(abs(subtract(f32(got), want).unwrap()).unwrap(), None).unwrap();
    let den = sum(abs(want).unwrap(), None).unwrap();
    num.item::<f32>() / den.item::<f32>().max(1e-12)
}

/// Build the DiT at the checkpoint's quant geometry (`split_model.json` — base_q8 ⇒ Q8). `bf16`
/// selects the activation dtype: the f32 quality path (`false`) or the native bf16 path (`true`).
fn build_prec(bf16: bool, golden: &str) -> (LtxDiT, Weights) {
    let dir = base_dir();
    let cfg = LtxConfig::from_model_dir(&dir).expect("embedded_config.json");
    let split = SplitModel::from_model_dir(&dir).expect("split_model.json");
    let prec = if bf16 {
        Precision::quant_bf16(split.bits, split.group)
    } else {
        Precision::quant_f32(split.bits, split.group)
    };
    let w =
        Weights::from_file(dir.join("transformer.safetensors")).expect("transformer.safetensors");
    let dit = LtxDiT::from_weights(&w, &cfg, prec).expect("build LtxDiT");
    let g = Weights::from_file(golden).expect("golden (run tools/dump_ltx_dit_golden.py)");
    (dit, g)
}

fn build() -> (LtxDiT, Weights) {
    build_prec(false, GOLDEN)
}

#[test]
#[ignore = "needs ltx_2_3_base_q8 transformer.safetensors (~20 GB)"]
fn dit_velocity_matches_reference() {
    let (dit, g) = build();
    let got = dit
        .forward(
            g.require("latent").unwrap(),
            g.require("timestep").unwrap(),
            g.require("context").unwrap(),
            None,
            g.require("positions").unwrap(),
            None,
        )
        .expect("dit forward");
    let want = g.require("velocity").unwrap();
    assert_eq!(got.shape(), want.shape(), "velocity shape");
    let (pr, mr) = (peak_rel(&got, want), mean_rel(&got, want));
    eprintln!("dit velocity peak_rel = {pr:.3e} mean_rel = {mr:.3e}");
    // The per-forward DiT is bit-exact at matched mlx 0.31.2 (sc-2842 fixed the last seed, the
    // host-f64 timestep table). A non-zero residual here means a per-op divergence has crept back.
    // sc-7141: the per-stage RoPE epoch fast path must produce velocity byte-identical to the content
    // path (the memo's computed tables don't depend on the cache key). Re-run with `Some(epoch)` on the
    // SAME inputs and assert exact equality with `got` — transitively gating the epoch path against the
    // reference golden on real weights.
    let got_epoch = dit
        .forward(
            g.require("latent").unwrap(),
            g.require("timestep").unwrap(),
            g.require("context").unwrap(),
            None,
            g.require("positions").unwrap(),
            Some(dit.next_rope_epoch()),
        )
        .expect("dit forward (epoch path)");
    mlx_rs::transforms::eval([&got, &got_epoch]).unwrap();
    assert_eq!(
        f32(&got_epoch).as_slice::<f32>(),
        f32(&got).as_slice::<f32>(),
        "sc-7141: epoch-path velocity must be byte-identical to the content path"
    );
    assert!(
        pr <= DIT_F32_XSTACK_PEAK_REL,
        "dit velocity peak_rel {pr:.3e} exceeds the 0.32.0 cross-stack f32 bound {DIT_F32_XSTACK_PEAK_REL:.1e} (sc-12896)"
    );
    assert!(
        mr <= DIT_F32_XSTACK_MEAN_REL,
        "dit velocity mean_rel {mr:.3e} exceeds the 0.32.0 cross-stack f32 bound {DIT_F32_XSTACK_MEAN_REL:.1e} (sc-12896)"
    );
}

/// The reference's **native bf16+Q8** per-forward — the production-speed path ([`Precision::quant_bf16`]).
/// Bit-exact at matched MLX — verified 0.0 on 0.32.0 non-NAX (sc-12896; the distilled stage-1 sampler
/// is chaos-sensitive, so the bf16 per-forward stays asserted exact). The same sc-2842 timestep-table
/// fix applies, plus
/// the `timestep × 1000` scaling must run in the **input (bf16) dtype** — `denoise_av` feeds a bf16
/// timestep, so a pre-upcast-to-f32 would round differently (`f32(σ)·1000` ≠ `bf16(σ·1000)`).
#[test]
#[ignore = "needs ltx_2_3_base_q8 transformer.safetensors (~20 GB)"]
fn dit_velocity_matches_reference_bf16() {
    let (dit, g) = build_prec(true, GOLDEN_BF16);
    let got = dit
        .forward(
            g.require("latent").unwrap(),
            g.require("timestep").unwrap(),
            g.require("context").unwrap(),
            None,
            g.require("positions").unwrap(),
            None,
        )
        .expect("dit forward");
    let want = g.require("velocity").unwrap();
    assert_eq!(got.shape(), want.shape(), "velocity shape");
    let (pr, mr) = (peak_rel(&got, want), mean_rel(&got, want));
    eprintln!("dit velocity (bf16) peak_rel = {pr:.3e} mean_rel = {mr:.3e}");
    assert!(
        pr == 0.0,
        "dit velocity (bf16) peak_rel {pr:.3e} must be bit-exact"
    );
    assert!(
        mr == 0.0,
        "dit velocity (bf16) mean_rel {mr:.3e} must be bit-exact"
    );
}

/// Sanity that the output head is exact: feed the reference post-block hidden through the Rust head
/// and compare the velocity — isolates the head from the 48-layer accumulation (was bit-exact at
/// bring-up).
#[test]
#[ignore = "needs ltx_2_3_base_q8 transformer.safetensors (~20 GB)"]
fn dit_output_head_exact() {
    let (dit, g) = build();
    let head = dit
        .output_head(
            g.require("tap_h").unwrap(),
            g.require("tap_emb_ts").unwrap(),
        )
        .expect("output_head");
    let pr = peak_rel(&head, g.require("velocity").unwrap());
    eprintln!("output_head(golden h) peak_rel = {pr:.3e}");
    assert!(pr < 5e-3, "output head peak_rel {pr:.3e} too high");
}
