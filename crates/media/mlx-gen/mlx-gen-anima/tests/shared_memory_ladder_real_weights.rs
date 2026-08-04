//! SC-15524 hardware-gated Anima shared memory-ladder evidence (MLX/Metal).
//!
//! `#[ignore]`d — needs the licensed `circlestone-labs/Anima` snapshot and an Apple GPU. Run:
//!   ANIMA_SNAPSHOT=<…>/models--circlestone-labs--Anima/snapshots/<rev> \
//!   cargo test -p mlx-gen-anima --release --test shared_memory_ladder_real_weights -- --ignored --nocapture
//!
//! Every number this suite prints is measured **on this backend, against this family's own exact
//! reference**. Nothing here is inherited from Z-Image or Qwen-Image because they share the VAE and
//! the tiling machinery: what transfers between families (and between backends) is the mechanism, not
//! the magnitude and never the candidate set.
//!
//! `ANIMA_LADDER_SIZE` / `ANIMA_LADDER_STEPS` tune the probe; the defaults are the shipped 1024²
//! bucket at 2 steps (quality is irrelevant — the assertions are about peak memory and about the
//! ladder not changing the image).

use std::path::PathBuf;

use mlx_gen::gen_core::{GenerationMemory, TransformerComponent};
use mlx_gen::tiling::TilingConfig;
use mlx_gen::{
    GenerationOutput, GenerationRequest, Image, LoadShape, LoadSpec, OffloadPolicy, Progress,
    WeightsSource,
};
use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};

use mlx_gen_anima::memory_strategy as ladder;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
const PROVIDER: &str = "anima_base";

fn gib(bytes: usize) -> f64 {
    bytes as f64 / GIB
}

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

/// The snapshot is a caller-provisioned path. Inference never derives a cache location (epic 13657).
fn snapshot() -> PathBuf {
    PathBuf::from(std::env::var("ANIMA_SNAPSHOT").expect(
        "set ANIMA_SNAPSHOT to the Anima snapshot dir; inference never self-fetches or derives a \
         cache location (epic 13657)",
    ))
}

fn size() -> u32 {
    env_u32("ANIMA_LADDER_SIZE", 1024)
}

fn steps() -> u32 {
    env_u32("ANIMA_LADDER_STEPS", 2)
}

fn spec(shape: LoadShape) -> LoadSpec {
    LoadSpec::new(WeightsSource::Dir(snapshot()))
        .with_offload_policy(OffloadPolicy::Sequential)
        .with_load_shape(shape)
}

fn request(memory: GenerationMemory) -> GenerationRequest {
    GenerationRequest {
        prompt: "an anime girl with silver hair on a rooftop at dusk, detailed, masterpiece".into(),
        negative_prompt: Some("blurry, low quality".into()),
        guidance: Some(4.5),
        width: size(),
        height: size(),
        steps: Some(steps()),
        seed: Some(1234),
        memory: Some(memory),
        ..Default::default()
    }
}

struct Run {
    conditioning: usize,
    denoise: usize,
    decode: usize,
    image: Image,
}

impl Run {
    fn peak(&self) -> usize {
        self.conditioning.max(self.denoise).max(self.decode)
    }
}

/// One measured render. The phase peaks are read at the phase boundaries the generator already
/// announces: the first sampler `Step` closes conditioning, `Decoding` closes the denoise.
fn run(shape: LoadShape, memory: GenerationMemory) -> Run {
    let generator = mlx_gen_anima::provider_registry()
        .expect("anima registry")
        .load(PROVIDER, &spec(shape))
        .expect("load anima_base");
    clear_cache();
    reset_peak_memory();
    let mut conditioning = 0;
    let mut denoise = 0;
    let mut progress = |event| match event {
        Progress::Step { current: 1, .. } if conditioning == 0 => {
            conditioning = get_peak_memory();
            reset_peak_memory();
        }
        Progress::Decoding if denoise == 0 => {
            denoise = get_peak_memory();
            reset_peak_memory();
        }
        _ => {}
    };
    let output = generator
        .generate(&request(memory), &mut progress)
        .expect("generate anima image");
    let decode = get_peak_memory();
    let image = match output {
        GenerationOutput::Images(mut images) => images.pop().expect("one image"),
        other => panic!("expected image output, got {other:?}"),
    };
    drop(generator);
    clear_cache();
    assert!(
        conditioning > 0 && denoise > 0 && decode > 0,
        "phase boundaries were not observed: {conditioning}/{denoise}/{decode}"
    );
    Run {
        conditioning,
        denoise,
        decode,
        image,
    }
}

fn full_ladder() -> GenerationMemory {
    GenerationMemory {
        stage_residency: true,
        tile_vae_decode: true,
        chunk_attention: true,
        stream_transformer_blocks: true,
        decode_tile_edge: Some(ladder::DECODE_TILE_EDGE),
        decode_overlap: Some(ladder::DECODE_OVERLAP),
        attention_chunk_size: Some(ladder::ATTENTION_CHUNK_SIZE),
        transformer_window_size: Some(ladder::TRANSFORMER_WINDOW_SIZE),
        transformer_window_component: Some(TransformerComponent::Dit),
        ..Default::default()
    }
}

fn delta(a: &Image, b: &Image) -> (u8, f64) {
    assert_eq!((a.width, a.height), (b.width, b.height));
    let mut max = 0;
    let mut sum = 0_u64;
    for (left, right) in a.pixels.iter().zip(&b.pixels) {
        let d = left.abs_diff(*right);
        max = max.max(d);
        sum += u64::from(d);
    }
    (max, sum as f64 / a.pixels.len() as f64)
}

// =================================================================================================
// 1. The cumulative ladder: each rung bounds its phase, and none of them changes the image.
// =================================================================================================

#[test]
#[ignore = "needs the licensed circlestone-labs/Anima snapshot and Apple/Metal"]
fn full_shared_ladder_reduces_peak_and_preserves_output() {
    // The contract this render admits must be the full ladder on a deferred load, and the resident
    // baseline must be the SAME provider — a rung is only meaningful against rung 0.
    let contract = mlx_gen_anima::provider_registry()
        .expect("anima registry")
        .memory_strategy_contract(PROVIDER, &spec(LoadShape::DeferredMaterialization))
        .expect("query contract")
        .expect("anima memory contract");
    assert_eq!(
        contract.calibration.as_ref().unwrap().fingerprint,
        ladder::MEMORY_CALIBRATION_FINGERPRINT
    );
    for strategy in mlx_gen::gen_core::MemoryStrategy::ALL {
        assert_eq!(
            contract.capability(strategy).unwrap().support,
            mlx_gen::gen_core::MemoryStrategySupport::Implemented,
            "a deferred Anima load must implement {strategy:?}"
        );
    }

    let arms: Vec<(&str, LoadShape, GenerationMemory)> = vec![
        (
            "rung0-resident",
            LoadShape::EagerMaterialization,
            GenerationMemory::default(),
        ),
        (
            "rung1-staged",
            LoadShape::EagerMaterialization,
            GenerationMemory {
                stage_residency: true,
                ..Default::default()
            },
        ),
        (
            "rung2-bounded-decode",
            LoadShape::EagerMaterialization,
            GenerationMemory {
                stage_residency: true,
                tile_vae_decode: true,
                decode_tile_edge: Some(ladder::DECODE_TILE_EDGE),
                decode_overlap: Some(ladder::DECODE_OVERLAP),
                ..Default::default()
            },
        ),
        (
            "rung3-bounded-attention",
            LoadShape::EagerMaterialization,
            GenerationMemory {
                stage_residency: true,
                tile_vae_decode: true,
                chunk_attention: true,
                decode_tile_edge: Some(ladder::DECODE_TILE_EDGE),
                decode_overlap: Some(ladder::DECODE_OVERLAP),
                attention_chunk_size: Some(ladder::ATTENTION_CHUNK_SIZE),
                ..Default::default()
            },
        ),
        (
            "rung4-bounded-transformer",
            LoadShape::DeferredMaterialization,
            full_ladder(),
        ),
    ];

    let mut runs = Vec::new();
    for (name, shape, memory) in arms {
        let run = run(shape, memory);
        println!(
            "ARM name={name} conditioning_gib={:.3} denoise_gib={:.3} decode_gib={:.3} request_gib={:.3}",
            gib(run.conditioning),
            gib(run.denoise),
            gib(run.decode),
            gib(run.peak())
        );
        runs.push((name, run));
    }

    let resident = &runs[0].1;
    let staged = &runs[1].1;
    let tiled = &runs[2].1;
    let chunked = &runs[3].1;
    let full = &runs[4].1;

    // Rung 1 is BYTE-identical: it only changes when components exist, never what they compute.
    assert_eq!(
        delta(&resident.image, &staged.image),
        (0, 0.0),
        "staged residency must not change the image"
    );
    // Rung 2 re-associates the decode across tiles, so it is close-but-not-bitwise. The bound is the
    // one the tile sweep below measures the admitted domain against.
    // **Mean is the gate, max is a corruption guard**, and the split is deliberate. A per-pixel
    // maximum over 3 M pixels is one seam pixel and moves image to image (measured 2 on this
    // prompt/seed, 4 and 16 on the two in the per-entry test); the mean absolute drift is stable to
    // the third decimal across all three checkpoints and is what "the tiling is visible" actually
    // means. The mean bound is the worst the ADMITTED domain measured (0.0589 at the tightest edge,
    // 192) with ~4x headroom, and every REJECTED edge sits above it. The max bound is a quarter of
    // the range — loose enough to tolerate seam pixels, tight enough that a render which silently
    // dropped its weights or its adapters cannot pass.
    let (decode_max, decode_mean) = delta(&staged.image, &tiled.image);
    assert!(
        decode_max <= 64 && decode_mean < 0.25,
        "bounded decode drifted beyond the admitted seam bound: max={decode_max} mean={decode_mean}"
    );
    // Rungs 3 and 4 touch the DENOISE, and both are exact there: chunking only changes when score
    // rows materialize, and a window only changes when weights exist.
    for (name, run) in [("rung3", chunked), ("rung4", full)] {
        let (max, mean) = delta(&tiled.image, &run.image);
        assert!(
            max <= 1,
            "{name} changed the denoise: max={max} mean={mean} (expected exact)"
        );
    }

    assert!(
        staged.peak() < resident.peak(),
        "rung 1 did not reduce the request peak"
    );
    assert!(
        full.peak() <= staged.peak(),
        "the full ladder must not exceed the staged baseline"
    );
    assert!(
        full.denoise < chunked.denoise,
        "rung 4 must bound the denoise phase: {:.3} vs {:.3} GiB",
        gib(full.denoise),
        gib(chunked.denoise)
    );
    println!(
        "RESULT status=pass provider={PROVIDER} backend=mlx size={} steps={} fingerprint={} \
         resident_peak_gib={:.3} staged_peak_gib={:.3} full_peak_gib={:.3} \
         chunked_denoise_gib={:.3} windowed_denoise_gib={:.3} decode_max_delta={decode_max} \
         decode_mean_delta={decode_mean:.6}",
        size(),
        steps(),
        ladder::MEMORY_CALIBRATION_FINGERPRINT,
        gib(resident.peak()),
        gib(staged.peak()),
        gib(full.peak()),
        gib(chunked.denoise),
        gib(full.denoise),
    );
}

// =================================================================================================
// 2. The decode ladder, swept against the EXACT untiled decode of the SAME latent.
// =================================================================================================

/// One denoise, then every candidate decode of that one latent — so the sweep isolates the decode
/// instead of comparing two independently sampled renders.
#[test]
#[ignore = "needs the licensed circlestone-labs/Anima snapshot and Apple/Metal"]
fn decode_tile_sweep_bounds_the_admitted_ladder() {
    use mlx_gen_anima::config::{Variant, VAE_CHANNELS, VAE_COMPRESSION};

    let edge = size();
    let pipeline =
        mlx_gen_anima::AnimaPipeline::from_source(&WeightsSource::Dir(snapshot()), Variant::Base)
            .expect("load anima pipeline");
    let key = mlx_rs::random::key(1234).unwrap();
    let init = mlx_rs::random::normal::<f32>(
        &[
            1,
            VAE_CHANNELS as i32,
            1,
            (edge / VAE_COMPRESSION) as i32,
            (edge / VAE_COMPRESSION) as i32,
        ][..],
        None,
        None,
        Some(&key),
    )
    .unwrap();
    let latent = pipeline
        .denoise_from_latent(
            &init,
            "an anime girl with silver hair on a rooftop at dusk, detailed, masterpiece",
            "blurry, low quality",
            Variant::Base,
            steps() as usize,
            4.5,
            mlx_gen_anima::DEFAULT_SAMPLER,
            mlx_rs::Dtype::Bfloat16,
        )
        .expect("denoise");
    mlx_rs::transforms::eval([&latent]).unwrap();
    let vae = &pipeline.components().vae;

    let decode = |tiling: Option<TilingConfig>| {
        clear_cache();
        reset_peak_memory();
        let decoded = match &tiling {
            Some(config) => vae
                .decode_tiled(&latent, config, None)
                .expect("tiled decode"),
            None => vae.decode(&latent).expect("decode"),
        };
        let image = mlx_gen::image::decoded_to_image(&decoded).expect("to image");
        let peak = get_peak_memory();
        clear_cache();
        (image, peak)
    };

    let (reference, untiled_peak) = decode(None);
    println!("TILE edge=untiled decode_gib={:.3}", gib(untiled_peak));

    // Every candidate this family ever considered, measured in one pass. `DECODE_TILE_EDGES` and
    // `DECODE_TILE_EDGES_REJECTED` partition exactly this list, so the split published in the
    // contract is a conclusion drawn from these rows rather than a claim sitting beside them.
    let candidates: Vec<u32> = ladder::DECODE_TILE_EDGES
        .iter()
        .chain(ladder::DECODE_TILE_EDGES_REJECTED)
        .copied()
        .collect();
    let mut rows = Vec::new();
    for tile in &candidates {
        let admitted = ladder::DECODE_TILE_EDGES.contains(tile);
        let (image, peak) = decode(Some(TilingConfig::spatial_only(
            *tile as i32,
            ladder::DECODE_OVERLAP as i32,
        )));
        let (max, mean) = delta(&reference, &image);
        println!(
            "TILE edge={tile} decode_gib={:.3} max_delta={max} mean_delta={mean:.6} \
             admitted={admitted}",
            gib(peak)
        );
        assert!(
            peak < untiled_peak,
            "tile {tile} did not bound the decode against the exact untiled reference"
        );
        rows.push((*tile, admitted, peak, max, mean));
    }
    drop(pipeline);
    clear_cache();

    // Part B — the same candidates inside the PRODUCTION envelope. A tile that bounds the decode
    // phase but does not move the REQUEST peak is not a saving (SC-15998), and the request peak is
    // the only number a selector can admit against. Run the full ladder so the decode is measured
    // where it actually binds.
    let registry = mlx_gen_anima::provider_registry().expect("anima registry");
    let mut request_peaks = Vec::new();
    for (tile, admitted, ..) in &rows {
        if !admitted {
            // A rejected edge is not merely undocumented — the production path REFUSES it, which is
            // what keeps the rejection evidence from decaying into a comment nobody enforces.
            let generator = registry
                .load(PROVIDER, &spec(LoadShape::DeferredMaterialization))
                .expect("load anima_base deferred");
            let error = generator
                .generate(
                    &request(GenerationMemory {
                        decode_tile_edge: Some(*tile),
                        ..full_ladder()
                    }),
                    &mut |_| {},
                )
                .expect_err("a rejected tile edge must be refused by the production path");
            println!("TILE-REQUEST edge={tile} admitted=false refused=true");
            assert!(
                error.to_string().contains("calibrated domain"),
                "expected the decode-domain rejection for {tile}, got: {error}"
            );
            drop(generator);
            clear_cache();
            continue;
        }
        let run = run(
            LoadShape::DeferredMaterialization,
            GenerationMemory {
                decode_tile_edge: Some(*tile),
                ..full_ladder()
            },
        );
        println!(
            "TILE-REQUEST edge={tile} admitted=true decode_gib={:.3} request_gib={:.3}",
            gib(run.decode),
            gib(run.peak())
        );
        request_peaks.push((*tile, run.peak()));
    }

    // The published default must be a candidate, and it must be one that actually bounds.
    assert!(ladder::DECODE_TILE_EDGES.contains(&ladder::DECODE_TILE_EDGE));

    // The separation is what the domain boundary is SET from — re-asserted here rather than left as
    // prose, so a VAE change that closed the gap fails loudly instead of drifting. `mean` is the
    // signal rather than `max`: a single worst pixel is noise on a 3 MB image, while the mean
    // absolute drift over every channel is what "the tiling is visible" actually means.
    let admitted_worst_mean = rows
        .iter()
        .filter(|(_, admitted, ..)| *admitted)
        .map(|(.., mean)| *mean)
        .fold(0.0_f64, f64::max);
    let rejected_best_mean = rows
        .iter()
        .filter(|(_, admitted, ..)| !*admitted)
        .map(|(.., mean)| *mean)
        .fold(f64::INFINITY, f64::min);
    assert!(
        admitted_worst_mean < rejected_best_mean,
        "the admitted ladder must be separable from the rejected one by measured drift: worst \
         admitted mean_delta={admitted_worst_mean:.6}, best rejected mean_delta={rejected_best_mean:.6}"
    );
    println!(
        "RESULT status=pass sweep=decode provider={PROVIDER} untiled_gib={:.3} \
         admitted={:?} admitted_worst_mean_delta={admitted_worst_mean:.6} rejected={:?} \
         rejected_best_mean_delta={rejected_best_mean:.6} request_peaks={:?}",
        gib(untiled_peak),
        ladder::DECODE_TILE_EDGES,
        ladder::DECODE_TILE_EDGES_REJECTED,
        request_peaks
            .iter()
            .map(|(tile, peak)| (*tile, format!("{:.3}", gib(*peak))))
            .collect::<Vec<_>>()
    );
}

// =================================================================================================
// 3. The window domain, with the attribution control that decides whether it has more than one row.
// =================================================================================================

/// A window of 28 is ONE all-covering window that bounds nothing. It is the control that decides
/// whether `TRANSFORMER_WINDOW_SIZES` is a ladder or a single exact value: if the peak does not move
/// between 1 and 28, the bound comes from the per-block materialize/drop, not from the window.
#[test]
#[ignore = "needs the licensed circlestone-labs/Anima snapshot and Apple/Metal"]
fn transformer_window_sweep_shows_the_published_domain_is_exact() {
    let resident = run(
        LoadShape::DeferredMaterialization,
        GenerationMemory {
            stage_residency: true,
            tile_vae_decode: true,
            chunk_attention: true,
            decode_tile_edge: Some(ladder::DECODE_TILE_EDGE),
            decode_overlap: Some(ladder::DECODE_OVERLAP),
            attention_chunk_size: Some(ladder::ATTENTION_CHUNK_SIZE),
            ..Default::default()
        },
    );
    println!(
        "WINDOW size=resident denoise_gib={:.3} request_gib={:.3}",
        gib(resident.denoise),
        gib(resident.peak())
    );

    let mut measured = Vec::new();
    for window in [1_u32, 2, 4, 14, 28] {
        let run = run(
            LoadShape::DeferredMaterialization,
            GenerationMemory {
                transformer_window_size: Some(window),
                ..full_ladder()
            },
        );
        println!(
            "WINDOW size={window} denoise_gib={:.3} request_gib={:.3} max_delta={}",
            gib(run.denoise),
            gib(run.peak()),
            delta(&resident.image, &run.image).0
        );
        assert!(
            run.denoise < resident.denoise,
            "window {window} did not bound the denoise phase"
        );
        assert!(
            delta(&resident.image, &run.image).0 <= 1,
            "window {window} changed the output"
        );
        measured.push((window, run.denoise));
    }

    let at_one = measured[0].1;
    assert_eq!(measured[0].0, 1);
    for (window, denoise) in &measured[1..] {
        assert!(
            *denoise >= at_one,
            "window {window} bounded tighter than the published window 1 ({:.3} vs {:.3} GiB) — \
             the published domain would then under-predict peak",
            gib(*denoise),
            gib(at_one)
        );
    }
    println!(
        "RESULT status=pass sweep=window provider={PROVIDER} published={:?} \
         resident_denoise_gib={:.3} measured={:?}",
        ladder::TRANSFORMER_WINDOW_SIZES,
        gib(resident.denoise),
        measured
            .iter()
            .map(|(window, denoise)| (*window, format!("{:.3}", gib(*denoise))))
            .collect::<Vec<_>>()
    );
}

// =================================================================================================
// 4. Fail closed: the rung-4 preconditions are refused at admission, not silently degraded.
// =================================================================================================

#[test]
#[ignore = "needs the licensed circlestone-labs/Anima snapshot and Apple/Metal"]
fn rung_four_preconditions_fail_closed_on_real_weights() {
    let registry = mlx_gen_anima::provider_registry().expect("anima registry");

    // An eager load cannot window: the contract says Missing and the generator refuses.
    let eager = registry
        .load(PROVIDER, &spec(LoadShape::EagerMaterialization))
        .expect("load anima_base eagerly");
    let error = eager
        .generate(&request(full_ladder()), &mut |_| {})
        .expect_err("an eager load must refuse a transformer window");
    assert!(
        error.to_string().contains("DeferredMaterialization"),
        "expected the load-shape rejection, got: {error}"
    );
    drop(eager);
    clear_cache();

    // Rung 4 without rung 1 engaged in the same request is refused rather than executed as a phase
    // win that does not move the request peak.
    let deferred = registry
        .load(PROVIDER, &spec(LoadShape::DeferredMaterialization))
        .expect("load anima_base deferred");
    let error = deferred
        .generate(
            &request(GenerationMemory {
                stage_residency: false,
                ..full_ladder()
            }),
            &mut |_| {},
        )
        .expect_err("rung 4 without rung 1 must be refused");
    assert!(
        error.to_string().contains("staged residency"),
        "expected the prerequisite rejection, got: {error}"
    );

    // An unimplemented window scope is rejected, never narrowed to `Dit`.
    let error = deferred
        .generate(
            &request(GenerationMemory {
                transformer_window_component: Some(TransformerComponent::TextEncoder),
                ..full_ladder()
            }),
            &mut |_| {},
        )
        .expect_err("an unimplemented window component must be refused");
    assert!(
        error.to_string().contains("transformer window component"),
        "expected the scope rejection, got: {error}"
    );

    // An out-of-domain decode geometry is rejected, never silently re-planned onto the default.
    let error = deferred
        .generate(
            &request(GenerationMemory {
                decode_tile_edge: Some(ladder::DECODE_TILE_EDGES_REJECTED[0]),
                ..full_ladder()
            }),
            &mut |_| {},
        )
        .expect_err("an out-of-domain tile edge must be refused");
    assert!(
        error.to_string().contains("calibrated domain"),
        "expected the decode-domain rejection, got: {error}"
    );
    drop(deferred);
    clear_cache();
    println!("RESULT status=pass check=fail-closed provider={PROVIDER}");
}

// =================================================================================================
// 5. Every catalog entry, on its own weights. Sharing code is not what makes an entry work.
// =================================================================================================

/// `anima_base`, `anima_aesthetic` and `anima_turbo` are three real providers over three different
/// DiT checkpoints, not one provider with aliases. They share this file's implementation — so the
/// interesting question is not whether the code runs (it is the same code) but whether each entry's
/// **own checkpoint** loads under the ladder and produces a bounded, unchanged render. Turbo in
/// particular takes a different denoise route: it is the merged CFG-free student, so its rung-3 and
/// rung-4 coverage is the single-forward path rather than the cond/uncond pair the other two run.
///
/// This is loadability + route coverage for the provider story. It is deliberately NOT per-entry
/// calibration: the per-tier envelopes and Verified signoff belong to SC-15492 / SC-15493 / SC-15494,
/// each of which must supply its own evidence. Nothing here promotes a sibling.
#[test]
#[ignore = "needs the licensed circlestone-labs/Anima snapshot and Apple/Metal"]
fn every_catalog_entry_executes_the_full_ladder_on_its_own_checkpoint() {
    let registry = mlx_gen_anima::provider_registry().expect("anima registry");
    let edge = size();
    for (provider, cfg_free) in [
        ("anima_base", false),
        ("anima_aesthetic", false),
        ("anima_turbo", true),
    ] {
        // Every entry must publish the same full ladder on a deferred load — the declaration is
        // shared by construction, and this pins that it did not silently diverge per id.
        let contract = registry
            .memory_strategy_contract(provider, &spec(LoadShape::DeferredMaterialization))
            .expect("query contract")
            .unwrap_or_else(|| panic!("{provider} memory contract"));
        for strategy in mlx_gen::gen_core::MemoryStrategy::ALL {
            assert_eq!(
                contract.capability(strategy).unwrap().support,
                mlx_gen::gen_core::MemoryStrategySupport::Implemented,
                "{provider} must implement {strategy:?}"
            );
        }
        assert!(contract.conformance_errors().is_empty(), "{provider}");

        let render = |shape: LoadShape, memory: GenerationMemory| {
            let generator = registry
                .load(provider, &spec(shape))
                .unwrap_or_else(|error| panic!("load {provider}: {error}"));
            clear_cache();
            reset_peak_memory();
            let request = GenerationRequest {
                prompt: "an anime girl with silver hair on a rooftop at dusk, detailed".into(),
                // Turbo is the merged CFG-free student: it advertises neither guidance nor a
                // negative prompt, so asking for either is a validation error.
                negative_prompt: (!cfg_free).then(|| "blurry, low quality".to_owned()),
                guidance: (!cfg_free).then_some(4.5),
                width: edge,
                height: edge,
                steps: Some(steps()),
                seed: Some(4321),
                memory: Some(memory),
                ..Default::default()
            };
            let GenerationOutput::Images(mut images) = generator
                .generate(&request, &mut |_| {})
                .unwrap_or_else(|error| panic!("{provider} generate: {error}"))
            else {
                panic!("{provider}: expected image output");
            };
            let peak = get_peak_memory();
            let image = images.pop().expect("one image");
            drop(generator);
            clear_cache();
            (image, peak)
        };

        let (baseline, baseline_peak) =
            render(LoadShape::EagerMaterialization, GenerationMemory::default());
        let (bounded, bounded_peak) = render(LoadShape::DeferredMaterialization, full_ladder());

        assert_eq!((bounded.width, bounded.height), (edge, edge));
        let (max, mean) = delta(&baseline, &bounded);
        let output_mean =
            bounded.pixels.iter().map(|v| f64::from(*v)).sum::<f64>() / bounded.pixels.len() as f64;
        println!(
            "ENTRY provider={provider} cfg_free={cfg_free} resident_peak_gib={:.3} \
             full_ladder_peak_gib={:.3} saved_pct={:.1} max_delta={max} mean_delta={mean:.6} \
             output_mean={output_mean:.3}",
            gib(baseline_peak),
            gib(bounded_peak),
            100.0 * (baseline_peak - bounded_peak) as f64 / baseline_peak as f64,
        );
        assert!(
            bounded_peak < baseline_peak,
            "{provider}: the full ladder did not reduce the request peak"
        );
        // Same split as the cumulative-ladder arm: the mean is the admitted domain's own worst
        // (0.0589) with headroom, the max is the quarter-range corruption guard.
        assert!(
            max <= 64 && mean < 0.25,
            "{provider}: the ladder changed the image beyond the admitted decode bound              (max={max} mean={mean})"
        );
        assert!(
            output_mean > 2.0 && output_mean < 253.0,
            "{provider}: degenerate output — a windowed render that dropped its weights would look \
             like this rather than error"
        );
    }
}
