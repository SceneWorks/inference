//! SC-15754 — DiT-level bounded transformer residency (ladder rung 4) on the MLX Z-Image DiT.
//!
//! Weight-free: reuses the tiny synthetic fixtures the parity tests already ship
//! (`z_transformer.safetensors`, `z_control_transformer.safetensors`) copied to a temp file, so this
//! runs in ordinary CI rather than only on a Mac with a 35 GB snapshot. The *memory* half is
//! necessarily a real-weights measurement and lives in `block_residency_real_weights.rs`; what runs
//! here is the half that can silently rot without anyone noticing — **correctness**.
//!
//! ## Why "windowed == resident" alone would be a false green
//!
//! SC-15750 built the primitive with a mutation check because its failure mode is silent: drop a
//! window before forcing evaluation and you free nothing while still producing correct images. The
//! family integration has a *mirror-image* silent failure — run the resident blocks and never touch
//! the stream at all. Output is identical, no error, no saving. An equivalence assertion is green in
//! both the working and the deleted implementation.
//!
//! So the equivalence tests here are paired with a **source mutation**: after the model is built, the
//! `.safetensors` on disk is rewritten with perturbed `layers.*` weights. A windowed forward re-reads
//! its blocks from that file, so its output must CHANGE; the resident forward holds the originals in
//! memory, so its output must NOT. A windowed path that quietly used `self.layers` fails both halves.
//!
//! That also pins the **captured adapters**, which are what make a streamed block a faithful twin
//! rather than an approximation on an adapted load — with a mutation arm that skips the capture and
//! requires the outputs to diverge.
//!
//! The other half of that fidelity, the per-block **quantization replay**, is deliberately NOT here:
//! the whole-transformer `quantize()` also touches the timestep embedder, whose fixture in-features
//! are not group-64 divisible, so a DiT-level test would either fail or silently stay dense. It lives
//! where the mechanism does — `block_stream::tests::a_materialized_block_replays_the_recorded_quantization`
//! compares a streamed block's packed codes/scales/biases against a resident block quantized the same
//! way. (Nothing reached it before: the real-weights suite runs against a pre-quantized tier dir,
//! where `quantize()` is a documented no-op.)

use mlx_gen::adapters::{AdaptableHost, Adapter};
use mlx_gen::attention::AttentionPlan;
use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, WeightsSource};
use mlx_gen_z_image::{ZImageControlTransformer, ZImageTransformer, ZImageTransformerConfig};
use mlx_rs::{Array, Dtype};

const BASE_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/z_transformer.safetensors"
);
const CONTROL_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/z_control_transformer.safetensors"
);

fn base_cfg() -> ZImageTransformerConfig {
    ZImageTransformerConfig {
        patch_size: 2,
        f_patch_size: 1,
        in_channels: 4,
        dim: 96,
        n_layers: 2,
        n_refiner_layers: 1,
        n_heads: 4,
        norm_eps: 1e-5,
        cap_feat_dim: 32,
        rope_theta: 256.0,
        t_scale: 1000.0,
        axes_dims: vec![8, 8, 8],
        axes_lens: vec![64, 64, 64],
        frequency_embedding_size: 256,
    }
}

fn control_cfg() -> ZImageTransformerConfig {
    ZImageTransformerConfig {
        patch_size: 2,
        f_patch_size: 1,
        in_channels: 4,
        dim: 64,
        n_layers: 4,
        n_refiner_layers: 2,
        n_heads: 4,
        norm_eps: 1e-5,
        cap_feat_dim: 32,
        rope_theta: 256.0,
        t_scale: 1000.0,
        axes_dims: vec![8, 4, 4],
        axes_lens: vec![64, 64, 64],
        frequency_embedding_size: 256,
    }
}

/// A private, writable copy of a fixture, so a test can perturb the *source* under a built model.
struct Scratch {
    dir: std::path::PathBuf,
    file: std::path::PathBuf,
}

impl Scratch {
    fn new(tag: &str, fixture: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "zimage-block-window-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("model.safetensors");
        std::fs::copy(fixture, &file).unwrap();
        Self { dir, file }
    }

    fn source(&self) -> WeightsSource {
        WeightsSource::File(self.file.clone())
    }

    /// Rewrite the file with every `{prefix}.layers.*` (or `{prefix}.control_layers.*`) tensor scaled,
    /// leaving everything else byte-identical. A model already holding the originals in memory is
    /// unaffected; anything that re-reads the file sees the change.
    fn perturb_blocks(&self, stack: &str) {
        let w = Weights::from_file(&self.file).unwrap();
        let keys: Vec<String> = w.keys().map(str::to_owned).collect();
        let mut named: Vec<(String, Array)> = Vec::new();
        for key in keys {
            let value = w.require(&key).unwrap().clone();
            let value = if key.contains(stack) && key.ends_with(".weight") {
                value
                    .multiply(
                        Array::from_slice(&[1.25f32], &[1])
                            .as_dtype(value.dtype())
                            .unwrap(),
                    )
                    .unwrap()
            } else {
                value
            };
            named.push((key, value));
        }
        let refs: Vec<(&str, &Array)> = named.iter().map(|(k, v)| (k.as_str(), v)).collect();
        Array::save_safetensors(refs, None, &self.file).unwrap();
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.dir).ok();
    }
}

fn peak_relative_delta(a: &Array, b: &Array) -> f32 {
    let n: i32 = b.shape().iter().product();
    let a = a.as_dtype(Dtype::Float32).unwrap().reshape(&[n]).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap().reshape(&[n]).unwrap();
    let (xs, ys) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let peak = ys.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-12);
    xs.iter()
        .zip(ys)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()))
        / peak
}

/// A streamed block runs the *same* ops on the *same* weights as its resident twin, so the two agree
/// far below the reduced-precision Metal matmul floor this crate already tolerates. Kept as a bound
/// rather than an exact compare because the windowed path forces an `eval` at each window boundary,
/// which can change kernel fusion (not arithmetic).
const TOLERANCE: f32 = 1e-5;

fn base_model(scratch: &Scratch) -> (ZImageTransformer, Array, Array) {
    let w = Weights::from_file(&scratch.file).unwrap();
    let model = ZImageTransformer::from_weights(&w, "w", base_cfg())
        .unwrap()
        .with_block_stream(scratch.source(), "w");
    let x = w.require("in.x").unwrap().clone();
    let cap = w.require("in.cap_feats").unwrap().clone();
    (model, x, cap)
}

#[test]
fn windowed_base_dit_matches_the_resident_forward_and_really_streams() {
    let scratch = Scratch::new("base", BASE_FIXTURE);
    let (model, x, cap) = base_model(&scratch);
    let cancel = CancelFlag::default();

    let resident = model.forward(&x, 0.7, &cap).unwrap();
    // Every advertised window over a 2-block fixture stack, plus the degenerate all-covering one.
    for window in [1usize, 2, 8] {
        let windowed = model
            .forward_windowed(&x, 0.7, &cap, AttentionPlan::UNBOUNDED, window, &cancel)
            .unwrap();
        assert_eq!(windowed.shape(), resident.shape(), "window {window} shape");
        let delta = peak_relative_delta(&windowed, &resident);
        assert!(
            delta < TOLERANCE,
            "window {window} changed the DiT velocity: peak-relative delta {delta:e}"
        );
    }

    // ── The anti-false-green half. Perturb the trunk weights ON DISK.
    scratch.perturb_blocks(".layers.");

    // The resident forward still holds the originals, so it is unchanged...
    let resident_again = model.forward(&x, 0.7, &cap).unwrap();
    assert_eq!(
        peak_relative_delta(&resident_again, &resident),
        0.0,
        "the resident forward must not re-read the file"
    );

    // ...and the windowed forward re-reads them, so it MUST now differ. If this assertion fails, the
    // windowed path is running `self.layers` and every equivalence check above is vacuous.
    let windowed = model
        .forward_windowed(&x, 0.7, &cap, AttentionPlan::UNBOUNDED, 1, &cancel)
        .unwrap();
    let delta = peak_relative_delta(&windowed, &resident);
    assert!(
        delta > 1e-3,
        "the windowed forward did not re-materialize its blocks from the source (delta {delta:e}) — \
         it is silently running the resident stack, which would report a rung-4 saving of zero"
    );
}

/// Cancellation is checked at every window boundary by the shared primitive; the family must not
/// swallow it. A tripped flag under a *bounded* plan aborts; the resident forward has no window
/// boundary and is unaffected.
#[test]
fn a_tripped_cancel_aborts_a_windowed_forward_but_not_a_resident_one() {
    let scratch = Scratch::new("cancel", BASE_FIXTURE);
    let (model, x, cap) = base_model(&scratch);

    let cancel = CancelFlag::default();
    cancel.cancel();
    // The TYPED `Error::Canceled`, not merely "an error". The worker distinguishes a cancelled job
    // from a failed one by this variant (sc-4481), so a generic error here would report the user's own
    // cancel as a backend failure — and `is_err()` alone would never notice.
    match model.forward_windowed(&x, 0.7, &cap, AttentionPlan::UNBOUNDED, 1, &cancel) {
        Err(mlx_gen::Error::Canceled) => {}
        other => panic!(
            "a tripped cancel must abort at the first window boundary with the typed \
             Error::Canceled, got {:?}",
            other.map(|v| v.shape().to_vec())
        ),
    }
    assert!(
        model.forward(&x, 0.7, &cap).is_ok(),
        "the resident forward has no window boundary and must be unaffected"
    );
}

/// Rung 4 must be unavailable — loudly — when the transformer has no re-openable source, rather than
/// silently falling back to the resident stack.
#[test]
fn a_sourceless_transformer_refuses_a_window_instead_of_running_resident() {
    let w = Weights::from_file(BASE_FIXTURE).unwrap();
    let model = ZImageTransformer::from_weights(&w, "w", base_cfg()).unwrap();
    let x = w.require("in.x").unwrap();
    let cap = w.require("in.cap_feats").unwrap();
    let err = model
        .forward_windowed(
            x,
            0.7,
            cap,
            AttentionPlan::UNBOUNDED,
            1,
            &CancelFlag::default(),
        )
        .map(|v| v.shape().to_vec())
        .unwrap_err()
        .to_string();
    assert!(err.contains("re-openable"), "{err}");
    // ...while the ordinary forward is entirely unaffected.
    assert!(model.forward(x, 0.7, cap).is_ok());
}

/// **The correctness risk SC-15750 explicitly did not cover.** A LoRA installed on the resident
/// blocks must reach the streamed ones, byte-for-byte — and the capture that makes that happen must
/// be load-bearing, so the second half asserts that WITHOUT it the outputs diverge.
#[test]
fn a_lora_on_the_trunk_survives_the_window_and_the_capture_is_load_bearing() {
    let scratch = Scratch::new("lora", BASE_FIXTURE);
    let cancel = CancelFlag::default();
    let dim = base_cfg().dim;
    let rank = 4;
    // Deterministic, non-trivial factors: a constant LoRA would be a weak discriminator.
    let a = Array::from_slice(
        &(0..dim * rank)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.02)
            .collect::<Vec<f32>>(),
        &[dim, rank],
    );
    let b = Array::from_slice(
        &(0..rank * dim)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.03)
            .collect::<Vec<f32>>(),
        &[rank, dim],
    );
    let lora = || Adapter::Lora {
        a: a.clone(),
        b: b.clone(),
        scale: 0.8,
    };

    // Adapted resident model, adapters captured for the stream (the production order: install, then
    // capture — see `model::load_heavy`).
    let (mut adapted, x, cap) = base_model(&scratch);
    for i in 0..base_cfg().n_layers {
        adapted
            .adaptable_mut(&["layers", &i.to_string(), "attention", "to_q"])
            .expect("layer target")
            .push(lora());
    }
    adapted.capture_block_adapters();

    let resident = adapted.forward(&x, 0.7, &cap).unwrap();
    let windowed = adapted
        .forward_windowed(&x, 0.7, &cap, AttentionPlan::UNBOUNDED, 1, &cancel)
        .unwrap();
    let delta = peak_relative_delta(&windowed, &resident);
    assert!(
        delta < TOLERANCE,
        "the streamed blocks lost the LoRA residual: peak-relative delta {delta:e}"
    );

    // The adapter genuinely changes the output — otherwise the agreement above proves nothing.
    let (plain, _, _) = base_model(&scratch);
    let unadapted = plain.forward(&x, 0.7, &cap).unwrap();
    assert!(
        peak_relative_delta(&resident, &unadapted) > 1e-3,
        "the test LoRA must actually move the output"
    );

    // Mutation: install the adapters but SKIP the capture. The streamed blocks are then unadapted and
    // the windowed forward must diverge from the resident one — which is what makes
    // `capture_block_adapters` load-bearing rather than decorative.
    let (mut uncaptured, _, _) = base_model(&scratch);
    for i in 0..base_cfg().n_layers {
        uncaptured
            .adaptable_mut(&["layers", &i.to_string(), "attention", "to_q"])
            .expect("layer target")
            .push(lora());
    }
    let windowed_uncaptured = uncaptured
        .forward_windowed(&x, 0.7, &cap, AttentionPlan::UNBOUNDED, 1, &cancel)
        .unwrap();
    assert!(
        peak_relative_delta(&windowed_uncaptured, &resident) > 1e-3,
        "without `capture_block_adapters` the streamed blocks must be unadapted — if they match \
         anyway, the capture is not what is carrying the LoRA and the identity test is vacuous"
    );
}

/// Rung 3 and rung 4 compose: the ladder is cumulative, so a rung-4 selection also carries the
/// attention budget, and both must be active in the same forward.
#[test]
fn the_window_composes_with_the_attention_budget() {
    let scratch = Scratch::new("compose", BASE_FIXTURE);
    let (model, x, cap) = base_model(&scratch);
    let cancel = CancelFlag::default();
    let budget = mlx_gen::attention::AttentionBudget::from_score_elements(2048, true);

    let resident_bounded = model
        .forward_budgeted(&x, 0.7, &cap, AttentionPlan::budgeted(budget))
        .unwrap();
    let windowed_bounded = model
        .forward_windowed(&x, 0.7, &cap, AttentionPlan::budgeted(budget), 1, &cancel)
        .unwrap();
    let delta = peak_relative_delta(&windowed_bounded, &resident_bounded);
    assert!(
        delta < TOLERANCE,
        "windowing changed a bounded-attention forward: peak-relative delta {delta:e}"
    );

    // Both rungs really engage: a tripped cancel aborts a windowed+chunked forward, and the chunked
    // attention is what the rung-3 suite already pins.
    let tripped = CancelFlag::default();
    tripped.cancel();
    assert!(model
        .forward_windowed(
            &x,
            0.7,
            &cap,
            AttentionPlan::budgeted(budget).with_cancel(&tripped),
            1,
            &tripped
        )
        .is_err());
}

/// The ControlNet route: BOTH stacks are windowed, and the dual-injection schedule — `before_proj`
/// seed, per-block `after_proj` hints, `control_context_scale`, and the injection places — is
/// unchanged. This is the second half of the correctness risk SC-15750 left open.
#[test]
fn windowed_control_dit_preserves_the_hint_injection() {
    let scratch = Scratch::new("control", CONTROL_FIXTURE);
    let w = Weights::from_file(&scratch.file).unwrap();
    let base = ZImageTransformer::from_weights(&w, "w", control_cfg())
        .unwrap()
        .with_block_stream(scratch.source(), "w");
    let model = ZImageControlTransformer::from_weights(base, &w, "w")
        .unwrap()
        .with_control_block_stream(scratch.source(), "w");
    let x = w.require("in.x").unwrap().clone();
    let cap = w.require("in.cap_feats").unwrap().clone();
    let cc = w.require("in.control_context").unwrap().clone();
    let scale = 1.0;
    let cancel = CancelFlag::default();

    let resident = model.forward(&x, 0.7, &cap, Some(&cc), scale).unwrap();
    for window in [1usize, 2, 4] {
        let windowed = model
            .forward_windowed(
                &x,
                0.7,
                &cap,
                Some(&cc),
                scale,
                AttentionPlan::UNBOUNDED,
                window,
                &cancel,
            )
            .unwrap();
        let delta = peak_relative_delta(&windowed, &resident);
        assert!(
            delta < TOLERANCE,
            "control window {window} changed the velocity: peak-relative delta {delta:e}"
        );
    }

    // The no-control short-circuit delegates to the base DiT and must carry the window with it.
    let base_resident = model.forward(&x, 0.7, &cap, None, scale).unwrap();
    let base_windowed = model
        .forward_windowed(
            &x,
            0.7,
            &cap,
            None,
            scale,
            AttentionPlan::UNBOUNDED,
            1,
            &cancel,
        )
        .unwrap();
    assert!(peak_relative_delta(&base_windowed, &base_resident) < TOLERANCE);

    // Anti-false-green: perturbing the CONTROL stack on disk must move the windowed control forward.
    // Only `w.control_layers.*` is rewritten, so this discriminates the CONTROL stream specifically —
    // the base stream is discriminated by its own test above.
    scratch.perturb_blocks("control_layers.");
    let windowed = model
        .forward_windowed(
            &x,
            0.7,
            &cap,
            Some(&cc),
            scale,
            AttentionPlan::UNBOUNDED,
            1,
            &cancel,
        )
        .unwrap();
    assert!(
        peak_relative_delta(&windowed, &resident) > 1e-3,
        "the windowed control forward did not re-materialize its control blocks from the source"
    );
    // The resident control forward is untouched by the on-disk change.
    let resident_again = model.forward(&x, 0.7, &cap, Some(&cc), scale).unwrap();
    assert_eq!(peak_relative_delta(&resident_again, &resident), 0.0);
}

/// A control transformer whose ControlNet checkpoint is not re-openable refuses a window rather than
/// running a half-bounded forward with 15 resident control blocks.
#[test]
fn a_half_streamable_control_transformer_refuses_the_window() {
    let scratch = Scratch::new("half", CONTROL_FIXTURE);
    let w = Weights::from_file(&scratch.file).unwrap();
    // Base armed, control NOT armed.
    let base = ZImageTransformer::from_weights(&w, "w", control_cfg())
        .unwrap()
        .with_block_stream(scratch.source(), "w");
    let model = ZImageControlTransformer::from_weights(base, &w, "w").unwrap();
    let x = w.require("in.x").unwrap();
    let cap = w.require("in.cap_feats").unwrap();
    let cc = w.require("in.control_context").unwrap();
    let err = model
        .forward_windowed(
            x,
            0.7,
            cap,
            Some(cc),
            1.0,
            AttentionPlan::UNBOUNDED,
            1,
            &CancelFlag::default(),
        )
        .map(|v| v.shape().to_vec())
        .unwrap_err()
        .to_string();
    assert!(err.contains("re-openable"), "{err}");
}

/// The two training-only mutators disarm rung 4 (SC-15754). Neither is replayable by a
/// re-materialized block — the bf16 cast lives in the live arrays and the SDPA-checkpoint flag has no
/// on-disk representation — so a windowed forward after either would silently run different weights
/// or a different backward. Disarming makes that a typed refusal instead.
#[test]
fn a_training_mutation_disarms_the_window_rather_than_streaming_stale_weights() {
    for mutate in [
        &(|m: &mut ZImageTransformer| m.cast_weights(Dtype::Bfloat16).unwrap())
            as &dyn Fn(&mut ZImageTransformer),
        &|m: &mut ZImageTransformer| m.set_sdpa_checkpoint(true, true),
    ] {
        let scratch = Scratch::new("training", BASE_FIXTURE);
        let (mut model, x, cap) = base_model(&scratch);
        // Armed before the mutation...
        assert!(model
            .forward_windowed(
                &x,
                0.7,
                &cap,
                AttentionPlan::UNBOUNDED,
                1,
                &CancelFlag::default()
            )
            .is_ok());
        mutate(&mut model);
        // ...and refused after it, rather than streaming blocks that no longer match.
        let err = model
            .forward_windowed(
                &x,
                0.7,
                &cap,
                AttentionPlan::UNBOUNDED,
                1,
                &CancelFlag::default(),
            )
            .map(|v| v.shape().to_vec())
            .unwrap_err()
            .to_string();
        assert!(err.contains("re-openable"), "{err}");
        // The ordinary forward still works — disarming the rung is not disabling the model.
        assert!(model.forward(&x, 0.7, &cap).is_ok());
    }
}

/// **The request-plumbing hole.** Everything above drives the DiT directly; the contract tests drive
/// the declaration. Neither notices if `block_window` were dropped at the four `generate` call sites —
/// the rung would be unreachable in production while every test stayed green.
///
/// This closes it at the seam a weight-free test can reach: `pipeline::denoise_with_progress`, which
/// is what all four generators call. It drives the loop with a window, then perturbs the source on
/// disk — same discriminator as the DiT tests, one layer up. A `block_window: None` regression makes
/// the second half fail, because the resident stack does not re-read the file.
#[test]
fn the_denoise_loop_threads_the_window_through_to_the_blocks() {
    use mlx_gen::attention::AttentionBudget;
    use mlx_gen::{FlowMatchEuler, Progress};

    let scratch = Scratch::new("plumbing", BASE_FIXTURE);
    let (model, x, cap) = base_model(&scratch);
    let cancel = CancelFlag::default();
    let scheduler = FlowMatchEuler {
        sigmas: vec![1.0, 0.5, 0.0],
    };
    let mut noop = |_: Progress| {};

    let run = |window: Option<usize>, on: &mut dyn FnMut(Progress)| {
        mlx_gen_z_image::denoise_with_progress(
            &model,
            &scheduler,
            None,
            7,
            x.clone(),
            &cap,
            0,
            AttentionBudget::UNBOUNDED,
            window,
            &cancel,
            on,
        )
    };

    let resident = run(None, &mut noop).expect("resident denoise");
    let windowed = run(Some(1), &mut noop).expect("windowed denoise");
    assert!(
        peak_relative_delta(&windowed, &resident) < TOLERANCE,
        "the windowed denoise loop must match the resident one"
    );

    // Perturb the trunk on disk: only a path that re-materializes from the source can see it.
    scratch.perturb_blocks(".layers.");
    let windowed_after = run(Some(1), &mut noop).expect("windowed denoise after perturb");
    assert!(
        peak_relative_delta(&windowed_after, &resident) > 1e-3,
        "`denoise_with_progress` did not thread its `block_window` down to the blocks — the rung is \
         unreachable from the generators even though the DiT-level entry works"
    );
    let resident_after = run(None, &mut noop).expect("resident denoise after perturb");
    assert_eq!(
        peak_relative_delta(&resident_after, &resident),
        0.0,
        "the resident loop must not re-read the file"
    );
}

/// The adapter-fidelity argument rests on Z-Image installing every adapter as a forward-time
/// **residual** (`apply_adapters_strict` only ever `push`es an `Adapter`), which is what makes
/// capture-and-replay a faithful reproduction. The sibling entry point
/// `apply_adapters_strict_with_diff_patch` instead FOLDS `.diff`/`.diff_b` deltas into the dense base
/// — a mutation a re-materialized block would not carry, so every streamed block would silently
/// differ from its resident twin.
///
/// That coupling was only a doc comment. This pins it the same way the `remap_transformer_keys`
/// guard does: a diff-patch-keyed file must be REJECTED by this family's applier, not folded.
#[test]
fn the_z_image_adapter_path_never_folds_a_diff_patch_into_the_base() {
    use mlx_gen::runtime::AdapterSpec;

    let dir = std::env::temp_dir().join(format!("zimage-diffpatch-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("diffpatch.safetensors");
    // A ComfyUI-style diff-patch file: `.diff` / `.diff_b` keys and nothing low-rank.
    let delta = Array::from_slice(
        &vec![0.01f32; (base_cfg().dim * base_cfg().dim) as usize],
        &[base_cfg().dim, base_cfg().dim],
    );
    Array::save_safetensors(vec![("layers.0.attention.to_q.diff", &delta)], None, &file).unwrap();

    let w = Weights::from_file(BASE_FIXTURE).unwrap();
    let mut model = ZImageTransformer::from_weights(&w, "w", base_cfg()).unwrap();
    let spec = AdapterSpec {
        path: file.clone(),
        scale: 1.0,
        kind: mlx_gen::runtime::AdapterKind::Lora,
        pass_scales: None,
        moe_expert: None,
    };
    let result = mlx_gen_z_image::apply_z_image_adapters(&mut model, std::slice::from_ref(&spec));
    assert!(
        result.is_err(),
        "the Z-Image adapter path must not fold a diff patch into the base — if it starts to, every \
         streamed block silently diverges from its resident twin and `block_stream`'s \
         capture-and-replay is no longer faithful. Teach `ZImageBlockStream` about it first."
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// SC-15754's request plumbing reaches the DiT through **four** denoise entry points, one per
/// registered variant: plain t2i, base CFG, turbo control, and base control. The first version of this
/// coverage drove only `denoise_with_progress`, so dropping `block_window` in any of the other three
/// left CI green.
///
/// Each is driven here with a window and then discriminated by the same on-disk perturbation. The CFG
/// arms use `guidance != 1.0` so the uncond forward actually runs — rung 4 has to cover BOTH forwards
/// or the bound is half-applied for the whole of a base render.
#[test]
fn every_denoise_entry_point_threads_the_window_through_to_the_blocks() {
    use mlx_gen::attention::AttentionBudget;
    use mlx_gen::{FlowMatchEuler, Progress};

    let scratch = Scratch::new("entrypoints", CONTROL_FIXTURE);
    let w = Weights::from_file(&scratch.file).unwrap();
    let base = ZImageTransformer::from_weights(&w, "w", control_cfg())
        .unwrap()
        .with_block_stream(scratch.source(), "w");
    let control = ZImageControlTransformer::from_weights(
        ZImageTransformer::from_weights(&w, "w", control_cfg())
            .unwrap()
            .with_block_stream(scratch.source(), "w"),
        &w,
        "w",
    )
    .unwrap()
    .with_control_block_stream(scratch.source(), "w");

    let x = w.require("in.x").unwrap().clone();
    let cap = w.require("in.cap_feats").unwrap().clone();
    let cc = w.require("in.control_context").unwrap().clone();
    let cancel = CancelFlag::default();
    let scheduler = FlowMatchEuler {
        sigmas: vec![1.0, 0.5, 0.0],
    };
    let mut noop = |_: Progress| {};

    // (label, closure) over every entry point the four generators call.
    type Run<'a> = Box<dyn Fn(Option<usize>) -> mlx_gen::Result<Array> + 'a>;
    let entries: Vec<(&str, Run)> = vec![
        (
            "denoise_with_progress",
            Box::new(|window| {
                mlx_gen_z_image::denoise_with_progress(
                    &base,
                    &scheduler,
                    None,
                    7,
                    x.clone(),
                    &cap,
                    0,
                    AttentionBudget::UNBOUNDED,
                    window,
                    &cancel,
                    &mut |_| {},
                )
            }),
        ),
        (
            "denoise_cfg_with_progress",
            Box::new(|window| {
                mlx_gen_z_image::denoise_cfg_with_progress(
                    &base,
                    &scheduler,
                    None,
                    7,
                    x.clone(),
                    &cap,
                    Some(&cap),
                    3.5, // != 1.0, so the uncond forward really runs
                    0,
                    AttentionBudget::UNBOUNDED,
                    window,
                    &cancel,
                    &mut |_| {},
                )
            }),
        ),
        (
            "denoise_control_with_progress",
            Box::new(|window| {
                mlx_gen_z_image::denoise_control_with_progress(
                    &control,
                    &scheduler,
                    None,
                    7,
                    x.clone(),
                    &cap,
                    &cc,
                    1.0,
                    0,
                    AttentionBudget::UNBOUNDED,
                    window,
                    &cancel,
                    &mut |_| {},
                )
            }),
        ),
        (
            "denoise_control_cfg_with_progress",
            Box::new(|window| {
                mlx_gen_z_image::denoise_control_cfg_with_progress(
                    &control,
                    &scheduler,
                    None,
                    7,
                    x.clone(),
                    &cap,
                    Some(&cap),
                    3.5,
                    &cc,
                    1.0,
                    0,
                    AttentionBudget::UNBOUNDED,
                    window,
                    &cancel,
                    &mut |_| {},
                )
            }),
        ),
    ];

    let _ = &mut noop;
    // Baselines before the source is touched.
    let baselines: Vec<(&str, Array, Array)> = entries
        .iter()
        .map(|(label, run)| {
            let resident = run(None).unwrap_or_else(|e| panic!("{label} resident: {e}"));
            let windowed = run(Some(1)).unwrap_or_else(|e| panic!("{label} windowed: {e}"));
            let delta = peak_relative_delta(&windowed, &resident);
            assert!(
                delta < TOLERANCE,
                "{label}: the windowed loop must match the resident one ({delta:e})"
            );
            (*label, resident, windowed)
        })
        .collect();

    // Perturb BOTH stacks: only a path that re-materializes from the source can see it.
    scratch.perturb_blocks(".layers.");

    for ((label, run), (_, resident, _)) in entries.iter().zip(&baselines) {
        let windowed_after = run(Some(1)).unwrap_or_else(|e| panic!("{label} windowed after: {e}"));
        assert!(
            peak_relative_delta(&windowed_after, resident) > 1e-3,
            "{label} did not thread its `block_window` down to the blocks — rung 4 is unreachable \
             through this entry point even though the DiT-level entry works"
        );
        let resident_after = run(None).unwrap_or_else(|e| panic!("{label} resident after: {e}"));
        assert_eq!(
            peak_relative_delta(&resident_after, resident),
            0.0,
            "{label}: the resident loop must not re-read the file"
        );
    }
}

/// sc-16631: every public legacy denoise entry remains an exact inert-preview wrapper. This covers
/// all four registered routes independently; a future signature/body drift in any one cannot hide
/// behind the other three.
#[test]
fn every_legacy_denoise_entry_is_exactly_inert() {
    use mlx_gen::attention::AttentionBudget;
    use mlx_gen::{FlowMatchEuler, PreviewSink};

    let w = Weights::from_file(CONTROL_FIXTURE).unwrap();
    let base = ZImageTransformer::from_weights(&w, "w", control_cfg()).unwrap();
    let control = ZImageControlTransformer::from_weights(
        ZImageTransformer::from_weights(&w, "w", control_cfg()).unwrap(),
        &w,
        "w",
    )
    .unwrap();
    let x = w.require("in.x").unwrap().clone();
    let cap = w.require("in.cap_feats").unwrap().clone();
    let cc = w.require("in.control_context").unwrap().clone();
    let scheduler = FlowMatchEuler {
        sigmas: vec![1.0, 0.5, 0.0],
    };
    let cancel = CancelFlag::default();
    let inert = PreviewSink::default();

    let assert_exact = |label: &str, legacy: Array, explicit: Array| {
        let legacy = legacy.reshape(&[-1]).unwrap();
        let explicit = explicit.reshape(&[-1]).unwrap();
        assert_eq!(legacy.shape(), explicit.shape(), "{label} shape");
        assert_eq!(
            legacy.as_slice::<f32>(),
            explicit.as_slice::<f32>(),
            "{label}: inert preview changed latent bytes"
        );
    };

    assert_exact(
        "z_image_turbo",
        mlx_gen_z_image::denoise_with_progress(
            &base,
            &scheduler,
            None,
            7,
            x.clone(),
            &cap,
            0,
            AttentionBudget::UNBOUNDED,
            None,
            &cancel,
            &mut |_| {},
        )
        .unwrap(),
        mlx_gen_z_image::denoise_with_progress_and_preview(
            &base,
            &scheduler,
            None,
            7,
            x.clone(),
            &cap,
            0,
            AttentionBudget::UNBOUNDED,
            None,
            &cancel,
            &inert,
            &mut |_| {},
        )
        .unwrap(),
    );
    assert_exact(
        "z_image",
        mlx_gen_z_image::denoise_cfg_with_progress(
            &base,
            &scheduler,
            None,
            7,
            x.clone(),
            &cap,
            Some(&cap),
            3.5,
            0,
            AttentionBudget::UNBOUNDED,
            None,
            &cancel,
            &mut |_| {},
        )
        .unwrap(),
        mlx_gen_z_image::denoise_cfg_with_progress_and_preview(
            &base,
            &scheduler,
            None,
            7,
            x.clone(),
            &cap,
            Some(&cap),
            3.5,
            0,
            AttentionBudget::UNBOUNDED,
            None,
            &cancel,
            &inert,
            &mut |_| {},
        )
        .unwrap(),
    );
    assert_exact(
        "z_image_turbo_control",
        mlx_gen_z_image::denoise_control_with_progress(
            &control,
            &scheduler,
            None,
            7,
            x.clone(),
            &cap,
            &cc,
            1.0,
            0,
            AttentionBudget::UNBOUNDED,
            None,
            &cancel,
            &mut |_| {},
        )
        .unwrap(),
        mlx_gen_z_image::denoise_control_with_progress_and_preview(
            &control,
            &scheduler,
            None,
            7,
            x.clone(),
            &cap,
            &cc,
            1.0,
            0,
            AttentionBudget::UNBOUNDED,
            None,
            &cancel,
            &inert,
            &mut |_| {},
        )
        .unwrap(),
    );
    assert_exact(
        "z_image_control",
        mlx_gen_z_image::denoise_control_cfg_with_progress(
            &control,
            &scheduler,
            None,
            7,
            x.clone(),
            &cap,
            Some(&cap),
            3.5,
            &cc,
            1.0,
            0,
            AttentionBudget::UNBOUNDED,
            None,
            &cancel,
            &mut |_| {},
        )
        .unwrap(),
        mlx_gen_z_image::denoise_control_cfg_with_progress_and_preview(
            &control,
            &scheduler,
            None,
            7,
            x,
            &cap,
            Some(&cap),
            3.5,
            &cc,
            1.0,
            0,
            AttentionBudget::UNBOUNDED,
            None,
            &cancel,
            &inert,
            &mut |_| {},
        )
        .unwrap(),
    );
}
