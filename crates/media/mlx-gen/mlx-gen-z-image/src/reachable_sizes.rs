//! Reachable-sizes PREDICTION for the iOS phone (epic 16774): per provider, mode, tier and square
//! edge, the per-phase counted peak of a Z-Image Turbo request — `resident + largest quantum + the
//! 512 MiB reserve` against the entitled launch headroom — priced by [`crate::ios_pricing`] (the
//! header-only pricing the device admits with; one source of truth for the arithmetic) on the real
//! snapshot's headers. Predictions, never evidence: the device campaign measures.
//!
//! `ZIMAGE_TIER_ROOT=<SceneWorks/z-image-turbo-mlx snapshot> cargo test -p mlx-gen-z-image --lib
//! reachable_sizes -- --ignored --nocapture`.
//!
//! Two tables:
//!
//! - **(A) fully optimized** — the deepest rung the iOS load spec declares `Implemented` at the LAST
//!   (cheapest) selection of `MemoryProviderContract::selection_space(strategy, tier, false)`, with
//!   `execution = contract.generation_memory(&selection)` (exactly `ios_admission::build_candidate`).
//!   Every cell must fit: a cell that does not is a defect.
//! - **(B) rung 0 Resident** — the `Resident` strategy's only selection (everything resident, the
//!   whole-image decode, unbounded attention). Expected to refuse in places.
//!
//! The load spec mirrors `mlx_gen_ios_catalog::zimage::z_image_turbo_load_spec` (the iOS offload
//! policy and a deferred materialization, the snapshot's stored tier requested): the catalog crate
//! depends on this one, so it cannot be a dependency here.
//!
//! The Fun-Controlnet-Union checkpoint is not provisioned on this host. Its headers are derived from
//! the published architecture into a SPARSE (header-only, never read) safetensors file: each control
//! block is a DiT block (the dense bf16 tier's `layers.0.` tensors — the checkpoint is published
//! dense) plus its `[dim, dim]` `after_proj` (and, on block 0 of each stack, `before_proj`), 15 main
//! blocks and a 2-block refiner, and the `[dim, 132]` patch embedder.
//!
//! The control KIND (pose / canny / depth) is not a pricing input: the Fun-Controlnet-Union is a
//! union ControlNet whose kinds share one VAE-encoded control path with no kind index
//! (`model_base_control.rs`: `impl ControlBranch for ZImageControl`; `model_control.rs`'s
//! `generate_impl` reads only the control image and scale) — the kind's preprocessing (an edge map,
//! a depth map, a pose render) happens before the request reaches the provider, and the iOS work
//! key's mode (`ios_admission::work_mode`) is `Other("control")` for all three. Each kind's rows are
//! printed (they are one route's price).

use std::path::{Path, PathBuf};

use mlx_gen::gen_core::{
    GenerationMemory, LoadShape, LoadSpec, MemoryGeometry, MemoryMode, MemoryStrategy,
    MemoryStrategySupport, OffloadPolicy, Quant, WeightsSource,
};
use mlx_gen::ios_pricing::{IosPhase, IosPricing, IosPricingRequest};
use mlx_gen::ios_quantum::PROCESS_RESERVE_BYTES;

/// The entitled launch headroom of the target phone (MiB): iPhone 16 Pro Max, three launches.
const ENTITLED_MIB: u64 = 6142;
const MIB: u64 = 1024 * 1024;

fn mib(bytes: u64) -> u64 {
    bytes.div_ceil(MIB)
}

/// A sparse, header-only Fun-Controlnet-Union checkpoint at `path` derived from the dense tier's
/// DiT block headers (see the module docs). The payload is never written: the file is extended to
/// its full length as a hole.
fn write_sparse_control(dense_transformer: &Path, path: &Path) {
    let headers =
        mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(dense_transformer).unwrap();
    let block: Vec<_> = headers
        .iter()
        .filter_map(|h| h.name.strip_prefix("layers.0.").map(|s| (s.to_string(), h)))
        .collect();
    let dim = headers
        .iter()
        .find(|h| h.name == "x_pad_token")
        .unwrap()
        .shape[1];
    let mut tensors: Vec<(String, String, Vec<usize>)> = vec![
        (
            "control_all_x_embedder.2-1.weight".into(),
            "BF16".into(),
            vec![dim, 132],
        ),
        (
            "control_all_x_embedder.2-1.bias".into(),
            "BF16".into(),
            vec![dim],
        ),
    ];
    for (stack, count) in [("control_layers", 15), ("control_noise_refiner", 2)] {
        for i in 0..count {
            for (suffix, h) in &block {
                tensors.push((
                    format!("{stack}.{i}.{suffix}"),
                    format!("{:?}", h.dtype),
                    h.shape.clone(),
                ));
            }
            let projections: &[&str] = if i == 0 {
                &["after_proj", "before_proj"]
            } else {
                &["after_proj"]
            };
            for p in projections {
                tensors.push((
                    format!("{stack}.{i}.{p}.weight"),
                    "BF16".into(),
                    vec![dim, dim],
                ));
                tensors.push((format!("{stack}.{i}.{p}.bias"), "BF16".into(), vec![dim]));
            }
        }
    }
    let mut header = serde_json::Map::new();
    let mut offset = 0usize;
    for (name, dtype, shape) in &tensors {
        let width = match dtype.as_str() {
            "F32" | "U32" => 4,
            _ => 2,
        };
        let bytes = width * shape.iter().product::<usize>();
        header.insert(
            name.clone(),
            serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + bytes]}),
        );
        offset += bytes;
    }
    let mut json = serde_json::Value::Object(header).to_string().into_bytes();
    while !json.len().is_multiple_of(8) {
        json.push(b' ');
    }
    let mut file = std::fs::File::create(path).unwrap();
    use std::io::Write;
    file.write_all(&(json.len() as u64).to_le_bytes()).unwrap();
    file.write_all(&json).unwrap();
    file.set_len((8 + json.len() + offset) as u64).unwrap();
}

/// One provider × mode × tier × edge × rung cell's pricing.
struct Cell {
    selection: String,
    pricing: IosPricing,
}

/// The iOS load spec of `tier` (mirrors the catalog's `z_image_turbo_load_spec` + the stored tier).
fn load_spec(root: &Path, tier: &str, control: Option<&Path>) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(root.join(tier)))
        .with_offload_policy(OffloadPolicy::Sequential)
        .with_load_shape(LoadShape::DeferredMaterialization);
    spec.control = control.map(|p| WeightsSource::File(p.to_path_buf()));
    spec.quantize = match tier {
        "q4" => Some(Quant::Q4),
        "q8" => Some(Quant::Q8),
        _ => None,
    };
    spec
}

/// Price one request at `strategy`'s LAST (cheapest) selection, built from the provider's real
/// contract on `spec` exactly as `ios_admission::build_candidate` builds it.
fn price(
    provider: &str,
    spec: &LoadSpec,
    mode: &MemoryMode,
    edge: u32,
    reference_count: u32,
    strategy: MemoryStrategy,
) -> Cell {
    let contract = crate::memory_strategy::memory_strategy_contract(provider, spec).unwrap();
    let tier = crate::memory_strategy::loaded_tier(spec, provider).unwrap();
    let space = contract.selection_space(strategy, tier, false).unwrap();
    let selection = *space.last().unwrap();
    let execution: GenerationMemory = contract.generation_memory(&selection).unwrap_or_default();
    let req = IosPricingRequest {
        provider,
        spec,
        mode,
        geometry: MemoryGeometry {
            width: edge,
            height: edge,
            batch: 1,
            frames: 1,
            reference_count,
        },
        selection: &selection,
        execution: &execution,
        token_window: 0,
    };
    let p = selection.parameters;
    Cell {
        selection: format!(
            "{}{}_o{}_c{}_w{}_{}",
            if execution.stage_residency { "s_" } else { "" },
            p.decode_tile_edge
                .map_or("untiled".into(), |t| format!("t{t}")),
            p.decode_overlap.map_or("-".into(), |o| o.to_string()),
            p.attention_chunk_size.map_or("-".into(), |c| c.to_string()),
            p.transformer_window_size
                .map_or("-".into(), |w| w.to_string()),
            match p.transformer_window_component {
                Some(c) => format!("{c:?}").to_ascii_lowercase(),
                None => "-".into(),
            }
        ),
        pricing: crate::ios_pricing::ios_pricing(&req).unwrap(),
    }
}

/// The deepest rung `spec`'s contract declares `Implemented`.
fn deepest(provider: &str, spec: &LoadSpec) -> MemoryStrategy {
    let contract = crate::memory_strategy::memory_strategy_contract(provider, spec).unwrap();
    MemoryStrategy::ALL
        .into_iter()
        .rev()
        .find(|s| {
            contract
                .strategies
                .iter()
                .any(|c| c.strategy == *s && c.support == MemoryStrategySupport::Implemented)
        })
        .unwrap()
}

/// One table row: each phase's `resident + scratch + reserve` (MiB) with its binding quantum.
fn row(cell: &Cell) -> (String, u64) {
    let mut cells = Vec::new();
    for phase in IosPhase::ALL {
        cells.push(match cell.pricing.phase(phase) {
            Some(p) => format!(
                "{} ({})",
                mib(p.peak_bytes() + PROCESS_RESERVE_BYTES),
                p.binding.trim_start_matches("Z-Image ")
            ),
            None => "-".into(),
        });
    }
    let peak = mib(cell.pricing.peak_bytes() + PROCESS_RESERVE_BYTES);
    (cells.join(" | "), peak)
}

#[test]
#[ignore = "reads a provisioned Z-Image Turbo snapshot's headers (ZIMAGE_TIER_ROOT)"]
fn reachable_sizes() {
    let root = PathBuf::from(std::env::var_os("ZIMAGE_TIER_ROOT").expect("ZIMAGE_TIER_ROOT"));
    // Nothing is evaluated (the pricing builds payload-free shape carriers); CPU regardless.
    mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
    let tmp = tempfile::tempdir().unwrap();
    let control = tmp.path().join("fun_controlnet_union.safetensors");
    write_sparse_control(&root.join("bf16/transformer/model.safetensors"), &control);

    let turbo = crate::model::MODEL_ID;
    let control_id = crate::model_control::MODEL_ID;
    let control_mode = MemoryMode::Other("control".into());
    // (provider, mode name, MemoryMode, reference count): `ios_admission::work_mode` + the request's
    // `memory_reference_count` (a control image is one reference, an img2img init another).
    let mut modes: Vec<(&str, String, MemoryMode, u32)> = vec![
        (turbo, "text".into(), MemoryMode::TextToImage, 0),
        (turbo, "reference".into(), MemoryMode::ImageToImage, 1),
    ];
    for kind in ["canny", "depth", "pose"] {
        modes.push((
            control_id,
            format!("control_{kind}"),
            control_mode.clone(),
            1,
        ));
    }
    for kind in ["canny", "depth", "pose"] {
        modes.push((
            control_id,
            format!("reference_{kind}"),
            control_mode.clone(),
            2,
        ));
    }
    let limit = ENTITLED_MIB;
    let header = "| provider | mode | tier | edge | selection | load | conditioning | encode | denoise | decode | peak | fits |";
    let rule = "|---|---|---|---:|---|---|---|---|---|---|---:|---|";
    let mut failures = Vec::new();
    for (table, rung0) in [
        ("(A) fully optimized", false),
        ("(B) rung 0 Resident", true),
    ] {
        println!("\n{table} — MiB per phase = resident + largest counted quantum + {} MiB reserve (binding quantum); entitled {ENTITLED_MIB} MiB\n", PROCESS_RESERVE_BYTES / MIB);
        println!("{header}\n{rule}");
        for (provider, name, mode, refs) in &modes {
            for tier in ["q4", "q8", "bf16"] {
                let spec = load_spec(
                    &root,
                    tier,
                    (*provider == control_id).then_some(control.as_path()),
                );
                let strategy = if rung0 {
                    MemoryStrategy::Resident
                } else {
                    deepest(provider, &spec)
                };
                for edge in [512u32, 768, 1024] {
                    let cell = price(provider, &spec, mode, edge, *refs, strategy);
                    let (cells, peak) = row(&cell);
                    let fits = peak <= limit;
                    if !rung0 && !fits {
                        failures.push(format!("{provider} {name} {tier} {edge}: {peak} MiB"));
                    }
                    println!(
                        "| {provider} | {name} | {tier} | {edge} | {:?} {} | {cells} | {peak} | {} |",
                        strategy,
                        cell.selection,
                        if fits { "fits" } else { "REFUSED" }
                    );
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "fully optimized cells over the entitled headroom: {failures:#?}"
    );
}
