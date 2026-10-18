//! Reachable-sizes PREDICTION for the iOS phone (epic 16774): per route, mode, tier and edge, every
//! phase's counted peak — resident bytes + its largest quantum + the engineering reserve — against
//! the entitled launch headroom, priced by the crate's own iOS pricing
//! ([`crate::ios_pricing`], the formulas the device guards check) from the real snapshot's configs
//! and tensor HEADERS (no payload is read, nothing runs). These are predictions, never evidence.
//!
//! `cargo test -p mlx-gen-iris --lib reachable_sizes -- --ignored --nocapture`, with
//! `IRIS_WEIGHTS_DIR` / `IRIS_TEXT_ENCODER_DIR` pointing at the snapshots.
//!
//! The `q8` / `q4` cells price the packed exports an iOS load reads (`crate::packed`) from the dense
//! release's headers. Columns, each the phase's peak in MiB:
//!
//! * **A — before**: the fully optimized configuration before the epic-16774 levers: the eager
//!   load's deepest implemented rung (generation: rung 3, staged; the encoder-free routes: rung 0)
//!   at its cheapest declared selection, with the PiT stage batched;
//! * **B — after**: the deferred load's deepest rung (rung 4) at its cheapest declared selection
//!   (staged, rung 3 with the per-row PiT stage, both components streamed at window 1), with its
//!   binding quantum and the lever(s) that made a cell fit that A refused;
//! * **C — rung 0**: the eager, warm (unstaged) resident configuration.
//!
//! Levers: **(a)** the Qwen3-VL tower streamed by layer window, **(b)** the backbone's block stacks
//! streamed by window, **(c)** rung 3's per-row PiT stage. A lever is credited when it alone makes
//! the phase fit (`a+b+c` when only their composition does). Outside the conditioning and denoise
//! phases (load, encode, decode) the saving is the deferred load itself — both components hold only
//! their skeletons, whichever scope streams — and is credited as `deferred load (a/b skeletons)`.

use std::path::PathBuf;

use mlx_gen::gen_core::iris::restoration::{plan, Dims, RestorationOptions, TileGeometry};
use mlx_gen::gen_core::{
    GenerationMemory, LoadShape, MemoryGeometry, MemoryMode, MemoryNumericTier, MemoryStrategy,
    TransformerComponent,
};
use mlx_gen::ios_pricing::{IosPhase, IosPricing, IosPricingRequest};
use mlx_gen::{LoadSpec, Quant, WeightsSource};

use crate::ios_pricing::{price, Pricing};

/// The entitled launch headroom of the target phone (MiB).
const ENTITLED_MIB: u64 = 6142;
const MIB: u64 = 1024 * 1024;

/// The caller-provisioned snapshot (inference never derives a cache path).
fn snapshot(env: &str) -> Option<PathBuf> {
    std::env::var_os(env).map(PathBuf::from)
}

fn fits(peak: u64) -> bool {
    peak.saturating_add(mlx_gen::ios_quantum::PROCESS_RESERVE_BYTES) <= ENTITLED_MIB * MIB
}

fn mib(bytes: u64) -> u64 {
    (bytes + MIB / 2) / MIB
}

/// One route of the prediction: its provider, request surface, load spec builder and geometry.
struct Route {
    name: String,
    provider: &'static str,
    mode: MemoryMode,
    spec: Box<dyn Fn(Option<Quant>, LoadShape) -> LoadSpec>,
    geometry: Box<dyn Fn(u64) -> MemoryGeometry>,
}

/// The selection a configuration runs and the `GenerationMemory` it writes: the cheapest declared
/// selection of `strategy` whose component scope is `component` (any when `None`).
fn configure(
    provider: &str,
    spec: &LoadSpec,
    strategy: MemoryStrategy,
    component: Option<TransformerComponent>,
) -> (mlx_gen::gen_core::MemorySelection, GenerationMemory) {
    // The declaration (identical on every surface of a load shape); the packed export the cell
    // prices is emulated, so the dense release's own artifact checks do not apply.
    let contract =
        crate::memory_strategy::weights_free_memory_strategy_contract(provider, spec).unwrap();
    let tier = MemoryNumericTier {
        precision: spec.precision,
        quant: spec.quantize,
        component_precision_floors: &[],
    };
    let space = contract.selection_space(strategy, tier, false).unwrap();
    let chosen = *space
        .iter()
        .rev()
        .find(|s| component.is_none() || s.parameters.transformer_window_component == component)
        .unwrap();
    let memory = contract.generation_memory(&chosen).unwrap_or_default();
    (chosen, memory)
}

/// The deepest rung `spec`'s contract implements.
fn deepest(provider: &str, spec: &LoadSpec) -> MemoryStrategy {
    let contract =
        crate::memory_strategy::weights_free_memory_strategy_contract(provider, spec).unwrap();
    MemoryStrategy::ALL
        .into_iter()
        .rev()
        .find(|s| {
            matches!(
                contract.capability(*s).map(|c| &c.support),
                Some(mlx_gen::gen_core::MemoryStrategySupport::Implemented)
            )
        })
        .unwrap()
}

fn priced(
    route: &Route,
    tier: Option<Quant>,
    shape: LoadShape,
    strategy: Option<MemoryStrategy>,
    component: Option<TransformerComponent>,
    pixel_rows: bool,
    edge: u64,
) -> IosPricing {
    let spec = (route.spec)(tier, shape);
    let strategy = strategy.unwrap_or_else(|| deepest(route.provider, &spec));
    let (selection, memory) = configure(route.provider, &spec, strategy, component);
    price(
        &IosPricingRequest {
            provider: route.provider,
            spec: &spec,
            mode: &route.mode,
            geometry: (route.geometry)(edge),
            selection: &selection,
            execution: &memory,
            token_window: 0,
        },
        Pricing {
            assume_packed: true,
            pixel_rows,
        },
    )
    .unwrap_or_else(|e| panic!("{} {tier:?} {edge}: {e}", route.name))
    .pricing
}

fn phase_peak(pricing: &IosPricing, phase: IosPhase) -> Option<u64> {
    pricing.phase(phase).map(|p| p.peak_bytes())
}

#[test]
#[ignore = "prints the iOS reachable-sizes prediction from the real snapshot's headers"]
fn print_the_reachable_sizes_prediction() {
    let (Some(root), Some(te_dir)) = (
        snapshot("IRIS_WEIGHTS_DIR"),
        snapshot("IRIS_TEXT_ENCODER_DIR"),
    ) else {
        println!("no Iris snapshot on disk: nothing to predict");
        return;
    };
    let generation = |prompts: u32| {
        let (root, te_dir) = (root.clone(), te_dir.clone());
        Route {
            name: if prompts == 1 {
                "iris_3b text".into()
            } else {
                format!("iris_3b prompt_batch {prompts}")
            },
            provider: crate::model::MODEL_ID,
            mode: MemoryMode::TextToImage,
            spec: Box::new(move |tier, shape| {
                let mut spec =
                    LoadSpec::new(WeightsSource::Dir(root.clone())).with_load_shape(shape);
                spec.quantize = tier;
                spec.components.insert(
                    mlx_gen::gen_core::iris::TEXT_ENCODER_COMPONENT.into(),
                    WeightsSource::Dir(te_dir.clone()),
                );
                spec
            }),
            geometry: Box::new(move |edge| MemoryGeometry {
                width: edge as u32,
                height: edge as u32,
                batch: prompts,
                frames: 1,
                reference_count: 0,
            }),
        }
    };
    let export = |name: &str, provider: &'static str, task, dir: PathBuf| Route {
        name: name.into(),
        provider,
        mode: crate::memory_strategy::memory_mode(task),
        spec: Box::new(move |tier, shape| {
            let mut spec = LoadSpec::new(WeightsSource::Dir(dir.clone())).with_load_shape(shape);
            spec.quantize = tier;
            spec
        }),
        geometry: Box::new(move |edge| match task {
            // Restoration: `edge` is the OUTPUT edge at the default 4x; admitted at the size its
            // plan processes.
            mlx_gen::gen_core::iris::IrisTask::Restoration => {
                let p = plan(
                    Dims::new((edge / 4) as u32, (edge / 4) as u32),
                    &RestorationOptions {
                        scale: 4.0,
                        ..Default::default()
                    },
                    TileGeometry::RELEASE,
                    crate::restoration::MODEL_ID,
                )
                .unwrap();
                crate::memory_strategy::restoration_memory_geometry(&p)
            }
            _ => MemoryGeometry {
                width: edge as u32,
                height: edge as u32,
                batch: 1,
                frames: 1,
                reference_count: 0,
            },
        }),
    };
    use mlx_gen::gen_core::iris::IrisTask;
    let routes = [
        generation(1),
        generation(2),
        export(
            "iris_3b_restore",
            crate::restoration::MODEL_ID,
            IrisTask::Restoration,
            root.join("upscaler"),
        ),
        export(
            "iris_3b_depth",
            crate::depth::DEPTH_MODEL_ID,
            IrisTask::Depth,
            root.join("depth"),
        ),
    ];
    println!(
        "| route | tier | edge | phase | A before (resident) MiB | A | B after (resident) MiB | B binding | B | lever | C rung 0 (resident) MiB | C |"
    );
    println!("|---|---|---:|---|---:|---|---:|---|---|---|---:|---|");
    let mark = |peak: u64| if fits(peak) { "fits" } else { "NO" };
    let mut refused_after = Vec::new();
    for (tier, label) in [
        (Some(Quant::Q4), "q4"),
        (Some(Quant::Q8), "q8"),
        (None, "bf16"),
    ] {
        for route in &routes {
            let generation = route.provider == crate::model::MODEL_ID;
            for edge in [512u64, 768, 1024] {
                let eager = LoadShape::EagerMaterialization;
                let deferred = LoadShape::DeferredMaterialization;
                let rung4 = Some(MemoryStrategy::BoundedTransformerResidency);
                let a = priced(route, tier, eager, None, None, false, edge);
                let b = priced(route, tier, deferred, None, None, true, edge);
                let c = priced(
                    route,
                    tier,
                    eager,
                    Some(MemoryStrategy::Resident),
                    None,
                    true,
                    edge,
                );
                // Each lever alone: (a) the tower streamed, (b) the backbone streamed (each without
                // the per-row stage), (c) the per-row stage on the eager route.
                let alone: Vec<(&str, IosPricing)> = if generation {
                    vec![
                        (
                            "(a) tower stream",
                            priced(
                                route,
                                tier,
                                deferred,
                                rung4,
                                Some(TransformerComponent::TextEncoder),
                                false,
                                edge,
                            ),
                        ),
                        (
                            "(b) backbone stream",
                            priced(
                                route,
                                tier,
                                deferred,
                                rung4,
                                Some(TransformerComponent::Dit),
                                false,
                                edge,
                            ),
                        ),
                        (
                            "(c) per-row PiT",
                            priced(route, tier, eager, None, None, true, edge),
                        ),
                    ]
                } else {
                    vec![("(b) backbone stream", b.clone())]
                };
                for phase in IosPhase::ALL {
                    let (Some(pa), Some(pb)) = (phase_peak(&a, phase), phase_peak(&b, phase))
                    else {
                        continue;
                    };
                    let pc = phase_peak(&c, phase).unwrap_or(0);
                    let lever = if fits(pa) {
                        "—".to_string()
                    } else if !fits(pb) {
                        "NONE".to_string()
                    } else if !matches!(phase, IosPhase::Conditioning | IosPhase::Denoise) {
                        "deferred load (a/b skeletons)".to_string()
                    } else {
                        let single: Vec<&str> = alone
                            .iter()
                            .filter(|(_, p)| phase_peak(p, phase).is_some_and(fits))
                            .map(|(name, _)| *name)
                            .collect();
                        if single.is_empty() {
                            "a+b+c".to_string()
                        } else {
                            single.join(" or ")
                        }
                    };
                    let bp = b.phase(phase).unwrap();
                    if !fits(pb) {
                        refused_after.push(format!("{} {label} {edge} {phase:?}", route.name));
                    }
                    println!(
                        "| {} | {label} | {edge} | {} | {} ({}) | {} | {} ({}) | {} {} | {} | {lever} | {} ({}) | {} |",
                        route.name,
                        phase.as_key(),
                        mib(pa + mlx_gen::ios_quantum::PROCESS_RESERVE_BYTES),
                        mib(a.phase(phase).unwrap().resident_bytes),
                        mark(pa),
                        mib(pb + mlx_gen::ios_quantum::PROCESS_RESERVE_BYTES),
                        mib(bp.resident_bytes),
                        bp.binding,
                        mib(bp.scratch_bytes),
                        mark(pb),
                        mib(pc + mlx_gen::ios_quantum::PROCESS_RESERVE_BYTES),
                        mib(c.phase(phase).map_or(0, |p| p.resident_bytes)),
                        mark(pc),
                    );
                }
            }
        }
    }
    println!(
        "\nfully optimized (B) cells refused: {}",
        if refused_after.is_empty() {
            "none".to_string()
        } else {
            refused_after.join(", ")
        }
    );
}
