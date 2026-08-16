//! sc-10840 (epic 10834): Bernini's staged-residency peak scaffold on real weights.
//!
//! `#[ignore]`d — assembles + loads the full ~56 GB Bernini snapshot (see `bernini_e2e.rs`). Run:
//!   cargo test -p mlx-gen-bernini --release --test sequential_residency_real_weights -- --ignored --nocapture
//!
//! **Why no Resident-vs-Sequential A/B.** Unlike the image engines wired onto the two-phase
//! [`mlx_gen::Residency`] seam (SD3 / Qwen-Image / Boogu), Bernini is **structurally always-staged**:
//! its generator holds NO component weights, and `generate_impl` loads per generate in phase order —
//! planner (Qwen2.5-VL-7B) → drop → UMT5-XXL T5 → drop → the two co-resident MoE experts + z16 VAE —
//! dropping BOTH encoders (+ `clear_cache()`, sc-10840) before the experts load. There is no
//! Resident-warm mode to toggle, so there is no A/B baseline to compare against. What sc-10840 added is
//! the `clear_cache()` discipline at the two encoder-drop boundaries, which is **output-neutral** (it
//! only returns freed buffer-cache pages to the OS) — the coherence smokes in `bernini_e2e.rs` already
//! guard the output. This scaffold measures the staged peak and asserts it stays well below the naive
//! whole-model resident sum (planner + T5 + both experts + VAE), i.e. the encoders really did free
//! before the experts.

use std::path::PathBuf;

use mlx_gen::gen_core::{
    MemoryOptimizationAuthority, MemoryRunOutcome, MemorySafetyDecision, MemoryStrategy,
};
use mlx_gen::media::Image;
use mlx_gen::{GenerationOutput, GenerationRequest, LoadShape, LoadSpec, WeightsSource};
use mlx_gen_bernini::convert::assemble_bernini_snapshot;
use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// The converted-snapshot store these tests assemble into.
///
/// `MLX_GEN_CONVERTED_ROOT` overrides it. The `$HOME` default stays because this is a **derived
/// cache** the tests build themselves from a caller-provisioned HF snapshot — not a provided input,
/// which is why it takes a fallback rather than the hard epic-13657 requirement `MLX_GEN_MODELS_ROOT`
/// and the `*_SRC` variables carry. Without the override, pointing the suite at a real store did
/// nothing: resolution read `$HOME` unconditionally and the rows skipped or mis-resolved while still
/// reporting green.
fn converted_root() -> PathBuf {
    std::env::var("MLX_GEN_CONVERTED_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").expect("HOME")).join(".cache/mlx-gen-models")
        })
}

fn hf_snapshot(repo: &str) -> Option<PathBuf> {
    let home = std::env::var("MLX_GEN_MODELS_ROOT").ok()?;
    let snaps = PathBuf::from(home)
        .join(format!("models--{}", repo.replace('/', "--")))
        .join("snapshots");
    std::fs::read_dir(snaps)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.is_dir())
}

/// Assemble the combined full-Bernini snapshot once (reused across reruns), returning its dir.
fn ensure_snapshot() -> PathBuf {
    let home = converted_root();
    let snapshot = home.join("bernini_full_mlx_bf16");
    let complete = snapshot.join("qwen2_5_vl.safetensors").is_file()
        && snapshot.join("high_noise_model.safetensors").is_file();
    if !complete {
        let pkg = hf_snapshot("ByteDance/Bernini-Diffusers")
            .expect("ByteDance/Bernini-Diffusers snapshot in the HF cache");
        let base = home.join("wan2_2_t2v_a14b_mlx_bf16");
        assert!(
            base.join("high_noise_model.safetensors").is_file(),
            "converted base Wan2.2-T2V-A14B snapshot required at {}",
            base.display()
        );
        assemble_bernini_snapshot(&snapshot, &pkg, &base, true).expect("assemble full snapshot");
    }
    snapshot
}

#[test]
#[ignore = "real weights: assembles + loads the ~56 GB full Bernini snapshot, runs a staged denoise"]
fn staged_peak_bounds_below_whole_model_sum() {
    let snapshot = ensure_snapshot();
    let model =
        mlx_gen_bernini::bernini::load(&LoadSpec::new(WeightsSource::Dir(snapshot.clone())))
            .expect("load bernini");
    // Tiny t2i (1 frame, 256², 4 steps) — the whole staged stack: planner load + MAR loop + drop +
    // clear_cache → T5 encode + drop + clear_cache → two experts + APG denoise → VAE decode.
    let req = GenerationRequest {
        prompt: "a red apple on a wooden table, studio lighting".into(),
        width: 256,
        height: 256,
        frames: Some(1),
        steps: Some(4),
        seed: Some(0),
        video_mode: Some("t2i".into()),
        ..Default::default()
    };
    reset_peak_memory();
    let out = model.generate(&req, &mut |_| {}).expect("generate");
    let peak = get_peak_memory();
    let img = match out {
        GenerationOutput::Images(mut v) => {
            assert_eq!(v.len(), 1, "1-frame t2i yields one image");
            v.pop().unwrap()
        }
        _ => panic!("expected Images for a 1-frame request"),
    };
    // Output stays coherent (the sc-10840 clear_cache calls are memory-only, not compute).
    let Image {
        width,
        height,
        pixels,
    } = &img;
    assert_eq!((*width, *height), (256, 256));
    assert!(
        pixels.iter().any(|&p| p != 0) && pixels.iter().any(|&p| p != 255),
        "decoded image must not be uniformly black/white"
    );

    // Self-calibrating tripwire (sc-10840). The old fixed 72 GiB ceiling sat BETWEEN the ~56 GiB clean
    // staged peak and the ~80 GiB whole-model sum, so losing ONE of the two `clear_cache()` flushes —
    // which re-admits the ~11 GiB T5 into the expert phase (~67 GiB) — still passed (false-green). Derive
    // the bound from the real on-disk expert bytes instead: a clean run peaks at the two co-resident bf16
    // experts (+ z16 VAE) because BOTH encoders (planner Qwen2.5-VL ~15 GiB, UMT5-XXL T5 ~11 GiB) are
    // dropped + `clear_cache()`d before the experts load. A lost flush lingers ~11-15 GiB of encoder into
    // that phase and blows past `experts + VAE + HEADROOM`, which sits well below a single-flush loss.
    let file_gib = |name: &str| {
        std::fs::metadata(snapshot.join(name))
            .map(|m| m.len() as f64 / GIB)
            .unwrap_or(0.0)
    };
    let expert_phase_gib = file_gib("low_noise_model.safetensors")
        + file_gib("high_noise_model.safetensors")
        + file_gib("vae.safetensors");
    // Denoise activations for the tiny 256² × 1-frame × 4-step run are well under this headroom; the
    // point is to sit below `expert_phase + smaller_encoder (T5 ~11 GiB)` so a single lost flush trips.
    const HEADROOM_GIB: f64 = 6.0;
    let ceiling = expert_phase_gib + HEADROOM_GIB;
    println!(
        "Bernini full t2i 256² @ 4 steps: staged peak = {:.3} GiB (ceiling {:.3} GiB = experts+VAE \
         {:.3} + {:.1} headroom)",
        peak as f64 / GIB,
        ceiling,
        expert_phase_gib,
        HEADROOM_GIB,
    );
    assert!(
        (peak as f64 / GIB) < ceiling,
        "staged peak {:.3} GiB exceeded experts+VAE + {:.0} GiB headroom ({:.3} GiB) — an encoder drop \
         / clear_cache regressed and a freed encoder lingered into the expert phase",
        peak as f64 / GIB,
        HEADROOM_GIB,
        ceiling,
    );
    drop(model);
    clear_cache();
}

/// sc-18609 — the resident-versus-selected rung-4 A/B, driven through the **loaded generator's**
/// memory-strategy hooks rather than by hand-writing `req.memory`.
///
/// The module header above explains why this family has no Resident-vs-Sequential *load* A/B: it is
/// structurally always-staged. Rung 4 is a different axis and does have one — the baseline is the
/// same load with no optimized selection, and the arm is the same load with the windowed selection
/// the contract admits. Both render the same request, so a peak drop that comes with a changed image
/// is a regression, not a saving.
///
/// Driving it through `begin_memory_strategy_request` is the point: that is the seam the worker uses,
/// and until sc-18609 the loaded generator did not implement it at all, so an evidence run that set
/// `req.memory` directly would have measured a mechanism no production route could select.
///
/// **Not executed by CI, and not executed by the change that added it** — it needs the ~56 GB
/// snapshot and a multi-minute Apple-Silicon render. It is the harness the epic's evidence campaign
/// runs; treat a green `0.00s` here as the skip it is.
#[test]
#[ignore = "real weights: loads the ~56 GB full Bernini snapshot twice and renders both A/B arms"]
fn windowed_trunk_lowers_the_request_peak_and_preserves_the_render() {
    let snapshot = ensure_snapshot();
    let spec = LoadSpec::new(WeightsSource::Dir(snapshot.clone()))
        .with_load_shape(LoadShape::DeferredMaterialization);
    let request = || GenerationRequest {
        prompt: "a red apple on a wooden table, studio lighting".into(),
        width: 256,
        height: 256,
        frames: Some(1),
        steps: Some(4),
        seed: Some(0),
        video_mode: Some("t2i".into()),
        ..Default::default()
    };
    let pixels = |output: GenerationOutput| match output {
        GenerationOutput::Images(mut images) => {
            assert_eq!(images.len(), 1);
            images.pop().unwrap()
        }
        _ => panic!("expected Images for a 1-frame request"),
    };

    // Baseline: the identical load and request with no optimized selection.
    let model = mlx_gen_bernini::bernini::load(&spec).expect("load bernini");
    reset_peak_memory();
    let resident = pixels(model.generate(&request(), &mut |_| {}).expect("generate"));
    let resident_peak = get_peak_memory();
    drop(model);
    clear_cache();

    // Arm: the same load, with the window the loaded generator's own admission accepts.
    let model = mlx_gen_bernini::bernini::load(&spec).expect("load bernini");
    assert_eq!(
        model
            .memory_strategy_contract()
            .expect("sc-18609: the loaded generator must publish its declared contract")
            .provider_id,
        mlx_gen_bernini::bernini::MODEL_ID
    );
    // The run context is built from the DECLARATION contract because it is the only one carrying a
    // calibration identity until this family's first cell is measured; admission below still runs
    // against the LOADED generator, which is the seam under test.
    let declaration = mlx_gen_bernini::memory_strategy::weights_free_memory_strategy_contract(
        mlx_gen_bernini::bernini::MODEL_ID,
        &spec,
    )
    .expect("declaration contract");
    let mut context = mlx_gen::gen_core::standard_memory_behavior_context(
        &declaration,
        MemoryStrategy::BoundedTransformerResidency,
        mlx_gen::gen_core::MemoryNumericTier {
            precision: spec.precision,
            quant: spec.quantize,
            component_precision_floors: &[],
        },
        mlx_gen::gen_core::MemoryBehaviorRoute {
            mode: mlx_gen::gen_core::MemoryMode::TextToImage,
            reference_count: 0,
            use_pid: false,
            has_phases: false,
            overlay: None,
        },
    )
    .expect("windowed run context");
    context.optimization_authority = MemoryOptimizationAuthority::Estimated;
    context.geometry.width = 256;
    context.geometry.height = 256;
    assert_eq!(
        model.memory_strategy_safety_check(&context),
        MemorySafetyDecision::Accept,
        "the declared rung-4 route must be admitted by the loaded generator"
    );
    let mut windowed = request();
    let mut scope = model
        .begin_memory_strategy_request(&context)
        .expect("rung 4 must open a request scope")
        .expect("rung 4 must open a request scope");
    scope.configure_request(&mut windowed).unwrap();
    assert!(
        windowed
            .memory
            .expect("the scope configures request memory")
            .stream_transformer_blocks,
        "the admitted selection must reach the block-plan lever"
    );
    reset_peak_memory();
    let selected = pixels(model.generate(&windowed, &mut |_| {}).expect("generate"));
    let selected_peak = get_peak_memory();
    scope.finish(MemoryRunOutcome::Complete).unwrap();
    // The scope borrows the generator, so it must be released before the generator is dropped.
    drop(scope);

    println!(
        "Bernini rung-4 A/B (t2i 256^2 @ 4 steps): resident peak = {:.3} GiB, windowed peak = {:.3} GiB",
        resident_peak as f64 / GIB,
        selected_peak as f64 / GIB,
    );
    let Image {
        width,
        height,
        pixels: selected_pixels,
    } = &selected;
    assert_eq!((*width, *height), (256, 256));
    assert_eq!(
        selected_pixels.len(),
        resident.pixels.len(),
        "the window is a residency lever, not a geometry one"
    );
    assert!(
        selected_peak < resident_peak,
        "windowing the 80-block dual-expert trunk must lower the request peak: \
         resident {resident_peak}, windowed {selected_peak}"
    );
    drop(model);
    clear_cache();
}
