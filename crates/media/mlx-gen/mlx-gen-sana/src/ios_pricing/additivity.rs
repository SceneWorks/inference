//! The iOS anchor law's ADDITIVITY proofs for SANA (epic 16774): iOS admission anchors on the TEXT
//! mode of the base architecture (`sana_1600m`), and the other requests this crate serves are
//! DERIVED from that anchor by [`mlx_gen::ios_anchor_law::derive`] over the counted pricing
//! ([`super::price`]):
//!
//! * `sana_1600m` reference (img2img): the reference DC-AE encoder re-added through the request's
//!   counted residency and its encode phase priced by its counted formula plus the anchor's largest
//!   unpriced overhead;
//! * `sana_sprint_1600m` text and reference: Sprint shares the Linear-DiT architecture and differs by
//!   its steps and its CFG-free SCM sampler (one conditioning instead of the CFG pair, the SCM step
//!   state instead of the flow-match step), all of which the counted pricing carries.
//!
//! Each case runs the production modules on a tiny, structurally real snapshot (a 6-stage x32 DC-AE,
//! a two-block trunk, a two-layer Gemma-2, at widths the production converter can pack), dense and
//! converted by [`crate::convert::prequantize_turnkey`] to q4 / q8, at two selections of the iOS
//! load spec (`Sequential`): the FULLY OPTIMIZED one (the deepest rung the contract declares
//! `Implemented`, the last selection of its `selection_space`) and rung 0 `Resident`. The text
//! anchor and the mode are each MEASURED ([`mlx_gen::ios_anchor_law::measure_phases`]: MLX
//! active-memory growth per phase over a fresh load + generate in the generator's `Sequential`
//! phase order, the worst of [`RUNS`]); the mode's demand derived from the anchor must cover the
//! mode's measurement, overall and in every phase both ran. CPU child, Metal fatal.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use mlx_gen::gen_core::{
    GenerationMemory, LoadSpec, MemoryNumericTier, MemoryStrategy, MemoryStrategySupport,
    OffloadPolicy, Quant,
};
use mlx_gen::ios_anchor_law::{area, derive, measure_phases, MeasuredRun};
use mlx_gen::ios_quantum::with_cpu_diagnostic_headroom;
use mlx_gen::weights::Weights;
use mlx_gen::StagedHeavy;
use mlx_rs::Array;

use super::*;
use crate::dc_ae::{DcAeDecoder, DcAeEncoder};
use crate::pipeline::{SanaConditioning, SanaGenerateRequest, SanaHeavy};
use crate::text_encoder::Gemma2;
use crate::transformer::SanaTransformer;

const CHILD: &str = "SANA_IOS_ADDITIVITY_CPU_CHILD";
/// Measured runs per configuration (each a fresh load); the worst per phase is the measurement.
const RUNS: usize = 3;
/// Far above every tiny quantum's scratch + the reserve: every guard passes.
const AMPLE: u64 = 1 << 36;
/// The fixture edge: above the cheapest published decode tile (192), so the bounded decode and
/// the bounded encode tile at the optimized selection.
const EDGE: u32 = 256;
const VOCAB: i32 = 256;
/// Gemma's padded window and the caption tokens the trunk attends.
const WINDOW: i32 = 24;
const CAPTION: i32 = 16;

fn in_cpu_child(name: &str) -> bool {
    if std::env::var_os(CHILD).is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                &format!("ios_pricing::additivity::{name}"),
                "--nocapture",
                "--test-threads=1",
            ])
            .env(CHILD, "1")
            .status()
            .unwrap();
        assert!(status.success(), "CPU child for {name} failed");
        return false;
    }
    mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
    assert!(crate::test_metal_guard::forbid_metal());
    true
}

fn dcae() -> DcAeConfig {
    use crate::config::BlockType::{EfficientVit as E, Res as R};
    DcAeConfig {
        in_channels: 3,
        latent_channels: 32,
        attention_head_dim: 8,
        block_out_channels: vec![8, 8, 16, 16, 32, 32],
        layers_per_block: vec![1; 6],
        encoder_layers_per_block: vec![1; 6],
        block_types: vec![R, R, R, E, E, E],
        qkv_multiscales: vec![5],
        upsample_interpolate: true,
        norm_eps: 1e-5,
        attn_eps: 1e-15,
        scaling_factor: 0.41407,
    }
}

/// A two-layer Gemma-2 whose every projection the converter packs at group 64.
fn gemma() -> Gemma2Config {
    Gemma2Config {
        hidden_size: 64,
        num_layers: 2,
        num_heads: 2,
        num_kv_heads: 1,
        head_dim: 32,
        intermediate_size: 128,
        ..Gemma2Config::gemma_2_2b()
    }
}

/// The trunk of a variant (inner width 64: every trunk Linear packs at group 64).
fn trunk_cfg(sprint: bool) -> SanaTransformerConfig {
    SanaTransformerConfig {
        in_channels: 32,
        out_channels: 32,
        num_layers: 2,
        caption_channels: gemma().hidden_size,
        ..crate::transformer::quantum_proofs::cfg(4, 16, 2.5, sprint)
    }
}

fn rand(shape: &[i32], seed: u64, dtype: mlx_rs::Dtype) -> Array {
    let a =
        mlx_rs::random::normal::<f32>(shape, None, None, Some(&mlx_rs::random::key(seed).unwrap()))
            .unwrap()
            .multiply(Array::from_f32(0.05))
            .unwrap()
            .as_dtype(dtype)
            .unwrap();
    a.eval().unwrap();
    a
}

fn save(dir: &Path, file: &str, map: &HashMap<String, Array>) {
    std::fs::create_dir_all(dir).unwrap();
    Array::save_safetensors(map.iter(), None, dir.join(file)).unwrap();
}

/// Write the dense snapshot of a variant at `root`.
fn write_snapshot(root: &Path, sprint: bool) {
    let g = gemma();
    let mut m = HashMap::new();
    let mut seed = 7000u64;
    let mut put = |k: String, shape: &[i32]| {
        seed += 1;
        m.insert(k, rand(shape, seed, mlx_rs::Dtype::Bfloat16));
    };
    let (h, hd) = (g.hidden_size, g.head_dim);
    put("model.embed_tokens.weight".into(), &[VOCAB, h]);
    put("model.norm.weight".into(), &[h]);
    for i in 0..g.num_layers {
        let p = format!("model.layers.{i}");
        for n in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            put(format!("{p}.{n}.weight"), &[h]);
        }
        put(
            format!("{p}.self_attn.q_proj.weight"),
            &[g.num_heads * hd, h],
        );
        put(
            format!("{p}.self_attn.k_proj.weight"),
            &[g.num_kv_heads * hd, h],
        );
        put(
            format!("{p}.self_attn.v_proj.weight"),
            &[g.num_kv_heads * hd, h],
        );
        put(
            format!("{p}.self_attn.o_proj.weight"),
            &[h, g.num_heads * hd],
        );
        put(
            format!("{p}.mlp.gate_proj.weight"),
            &[g.intermediate_size, h],
        );
        put(format!("{p}.mlp.up_proj.weight"), &[g.intermediate_size, h]);
        put(
            format!("{p}.mlp.down_proj.weight"),
            &[h, g.intermediate_size],
        );
    }
    save(&root.join("text_encoder"), "gemma-2-2b-it.safetensors", &m);
    let cfg = trunk_cfg(sprint);
    let mut b = crate::transformer::quantum_proofs::Builder {
        map: HashMap::new(),
        seed: 100,
        dtype: mlx_rs::Dtype::Float16,
    };
    b.trunk(&cfg);
    for i in 0..cfg.num_layers {
        b.block(&format!("transformer_blocks.{i}"), &cfg);
    }
    save(
        &root.join("transformer"),
        "diffusion_pytorch_model.safetensors",
        &b.map,
    );
    save(
        &root.join("vae"),
        "diffusion_pytorch_model.safetensors",
        &crate::dc_ae::quantum_proofs::weight_map(&dcae()),
    );
}

/// The fixture snapshots: per (variant, tier) a root, the dense one written, the packed ones
/// converted from it by the production converter.
struct Fixture {
    _dir: tempfile::TempDir,
    roots: BTreeMap<(bool, &'static str), PathBuf>,
}

const TIERS: [(&str, Option<i32>); 3] = [("dense", None), ("q4", Some(4)), ("q8", Some(8))];

fn fixture() -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let mut roots = BTreeMap::new();
    for sprint in [false, true] {
        let dense = dir.path().join(format!("sprint{sprint}-dense"));
        write_snapshot(&dense, sprint);
        for (tier, bits) in TIERS {
            let root = match bits {
                None => dense.clone(),
                Some(bits) => {
                    let root = dir.path().join(format!("sprint{sprint}-{tier}"));
                    crate::convert::prequantize_turnkey(&dense, &root, bits).unwrap();
                    root
                }
            };
            roots.insert((sprint, tier), root);
        }
    }
    Fixture { _dir: dir, roots }
}

/// The two selections of `provider`'s iOS load spec (mirrors `mlx_gen_ios_catalog::sana_load_spec`:
/// a `Sequential` load of the tier directory), as `ios_admission::build_candidate` builds each:
/// (fully optimized, rung 0).
fn selections(provider: &str, root: &Path, bits: Option<i32>) -> [(String, GenerationMemory); 2] {
    let mut spec = LoadSpec::new(WeightsSource::Dir(root.to_path_buf()))
        .with_offload_policy(OffloadPolicy::Sequential);
    spec.quantize = bits.map(|b| if b == 4 { Quant::Q4 } else { Quant::Q8 });
    let contract =
        crate::memory_strategy::weights_free_memory_strategy_contract(provider, &spec).unwrap();
    let tier = MemoryNumericTier {
        precision: spec.precision,
        quant: spec.quantize,
        component_precision_floors: &[],
    };
    let deepest = MemoryStrategy::ALL
        .into_iter()
        .rev()
        .find(|s| {
            contract
                .capability(*s)
                .is_some_and(|c| c.support == MemoryStrategySupport::Implemented)
        })
        .unwrap();
    [deepest, MemoryStrategy::Resident].map(|strategy| {
        let selection = *contract
            .selection_space(strategy, tier, false)
            .unwrap()
            .last()
            .unwrap();
        let execution = contract.generation_memory(&selection).unwrap_or_default();
        crate::memory_strategy::validate_request_memory(provider, &spec, &execution).unwrap();
        (format!("{strategy:?}"), execution)
    })
}

/// One request: variant, mode, snapshot root and its resolved execution.
#[derive(Clone)]
struct Request {
    sprint: bool,
    img2img: bool,
    root: PathBuf,
    execution: GenerationMemory,
}

impl Request {
    /// The plan exactly as [`super::ios_pricing`] resolves it on device (a `Sequential` load).
    fn plan(&self) -> SanaPricingPlan {
        let memory = Some(self.execution);
        SanaPricingPlan {
            sprint: self.sprint,
            sequential: true,
            img2img: self.img2img,
            width: EDGE,
            height: EDGE,
            count: 1,
            decode_tiling: crate::pipeline::resolved_decode_plan(memory, true)
                .map(|p| (p.edge, p.overlap)),
            encode_tile: crate::pipeline::resolved_encode_tile_on(memory, true),
            attention: crate::pipeline::resolved_rung_plan(memory, 2)
                .unwrap()
                .attention,
        }
    }

    fn pricing(&self) -> IosPricing {
        let facts = SanaPricingFacts {
            root: self.root.clone(),
            transformer: trunk_cfg(self.sprint),
            dc_ae: dcae(),
            gemma: gemma(),
            gemma_tokens: WINDOW as u64,
            caption_tokens: CAPTION as u64,
            text_encoder: Inventory::read(&self.root.join("text_encoder")).unwrap(),
            trunk: Inventory::read(&self.root.join("transformer")).unwrap(),
            vae: Inventory::read(&self.root.join("vae")).unwrap(),
        };
        price(&facts, &self.plan()).unwrap()
    }

    /// One fresh load + generate in the generator's `Sequential` phase order
    /// (`model::Sana::generate_impl`: `Residency::run_staged`, or `run_staged_img2img` for a
    /// reference), with the production loaders' guards and materialization at the fixture's
    /// geometry.
    fn run(&self) -> Result<()> {
        let root = &self.root;
        let (sprint, img2img) = (self.sprint, self.img2img);
        let memory = Some(self.execution);
        let plan = self.plan();
        let cancel = mlx_gen::CancelFlag::default();
        let g = gemma();

        // Phase A: the Gemma-2 encoder (`load_text_encoder_component`), the conditioning (the CFG
        // pair for base SANA), materialized, then the encoder dropped.
        let conditioning = {
            mlx_gen::ios_quantum::before_load(labels::TEXT_LOAD, &[root.join("text_encoder")])?;
            let w = Weights::from_file(root.join("text_encoder/gemma-2-2b-it.safetensors"))?;
            let encoder = Gemma2::from_weights(&w, "model.", &g)?;
            w.materialize_accessed()?;
            drop(w);
            let encode = |offset: i32| -> Result<(Array, Array)> {
                let ids: Vec<i32> = (0..WINDOW).map(|i| (i * 7 + offset) % VOCAB).collect();
                let pad = vec![1i32; WINDOW as usize];
                let hidden = encoder.forward(
                    &Array::from_slice(&ids, &[1, WINDOW]),
                    Some(&Array::from_slice(&pad, &[1, WINDOW])),
                )?;
                let cond = mlx_gen::attention::slice_axis(&hidden, 1, WINDOW - CAPTION, CAPTION)?;
                let mask = Array::from_slice(&vec![1i32; CAPTION as usize], &[1, CAPTION]);
                Ok((cond, mask))
            };
            let (cond, cond_mask) = encode(0)?;
            let uncond = if sprint { None } else { Some(encode(3)?) };
            let mut arrays = vec![&cond, &cond_mask];
            if let Some((u, um)) = &uncond {
                arrays.push(u);
                arrays.push(um);
            }
            mlx_rs::transforms::eval(arrays)?;
            SanaConditioning {
                cond,
                cond_mask,
                uncond,
            }
        };
        mlx_gen::residency::drain_allocator_cache();

        let image = mlx_gen::Image {
            width: EDGE,
            height: EDGE,
            pixels: (0..EDGE * EDGE * 3).map(|i| (i % 251) as u8).collect(),
        };
        let req = SanaGenerateRequest {
            width: EDGE,
            height: EDGE,
            steps: Some(2),
            seed: Some(3),
            init_image: img2img.then_some(&image),
            strength: img2img.then_some(0.9),
            ..SanaGenerateRequest::new("a fox")
        };

        // Phase A': the reference encode with only the DC-AE encoder resident
        // (`load_reference_encoder`), then the encoder dropped.
        let reference = if img2img {
            let vae = root.join("vae");
            mlx_gen::ios_quantum::before_load_where(
                labels::REFERENCE_ENCODER_LOAD,
                std::slice::from_ref(&vae),
                |name| name.starts_with("encoder."),
            )?;
            let vae_w = Weights::from_dir(vae)?;
            let encoder = DcAeEncoder::from_weights(&vae_w, dcae())?;
            vae_w.materialize_accessed()?;
            drop(vae_w);
            let r = crate::pipeline::prepare_reference_latent(
                Some(&encoder),
                &dcae(),
                sprint,
                &req,
                &cancel,
                plan.encode_tile,
            )?;
            assert!(r.is_some(), "the img2img case must encode");
            r
        } else {
            None
        };
        mlx_gen::residency::drain_allocator_cache();

        // Phase B: the heavy bundle (`load_heavy`, no encoder), the denoise, the latent batch.
        let heavy = {
            mlx_gen::ios_quantum::before_load_where(
                labels::HEAVY_LOAD,
                &[root.join("transformer"), root.join("vae")],
                |name| !name.starts_with("encoder."),
            )?;
            let trunk_w = Weights::from_dir(root.join("transformer"))?;
            let vae_w = Weights::from_dir(root.join("vae"))?;
            let decoder = DcAeDecoder::from_weights(&vae_w, dcae())?;
            vae_w.materialize_accessed()?;
            let cfg = trunk_cfg(sprint);
            let trunk = SanaTransformer::from_weights(&trunk_w, cfg.clone())?;
            trunk_w.materialize_accessed()?;
            if sprint {
                SanaHeavy::new_sprint_text_to_image(
                    trunk,
                    decoder,
                    dcae(),
                    cfg.guidance_embeds_scale,
                )
            } else {
                SanaHeavy::new_text_to_image(trunk, decoder, dcae())
            }
        };
        let forward = crate::pipeline::resolved_rung_plan(memory, 2)?;
        let latent = heavy.denoise_one_with_prepared_reference(
            &conditioning,
            &req,
            if sprint {
                crate::pipeline::SPRINT_DEFAULT_GUIDANCE
            } else {
                crate::pipeline::DEFAULT_GUIDANCE
            },
            reference.as_ref(),
            &cancel,
            &mut |_| {},
            &mlx_gen::PreviewSink::default(),
            forward,
        )?;
        crate::model::materialize_latent_batch(std::slice::from_ref(&latent))?;
        let light = heavy.shed_dit();
        drop((conditioning, reference));
        mlx_gen::residency::drain_allocator_cache();

        // Phase C: the decode with only the decoder resident.
        let tiling = crate::pipeline::resolve_decode_tiling(memory, true);
        SanaHeavy::light_view(&light).decode_one(&latent, &cancel, tiling.as_ref())?;
        Ok(())
    }

    /// [`RUNS`] measured runs (each a fresh load + generate).
    fn measure(&self) -> Measured {
        let mut worst = MeasuredRun {
            demand: 0,
            phases: BTreeMap::new(),
        };
        let mut least: BTreeMap<IosPhase, u64> = BTreeMap::new();
        for _ in 0..RUNS {
            let ((), run) = with_cpu_diagnostic_headroom(AMPLE, || {
                measure_phases(ios_quantum_phase, || self.run())
            })
            .unwrap();
            worst.demand = worst.demand.max(run.demand);
            for (phase, bytes) in run.phases {
                let slot = worst.phases.entry(phase).or_insert(0);
                *slot = (*slot).max(bytes);
                let low = least.entry(phase).or_insert(u64::MAX);
                *low = (*low).min(bytes);
            }
        }
        let spread = worst
            .phases
            .iter()
            .map(|(p, &hi)| (*p, hi - least[p]))
            .collect();
        Measured { worst, spread }
    }
}

/// One configuration's measurement: the worst of [`RUNS`] fresh runs per phase, and how far the
/// runs disagreed per phase (the capture's own spread).
struct Measured {
    worst: MeasuredRun,
    spread: BTreeMap<IosPhase, u64>,
}

/// The bound a mode's measured phase must stay under, and how it was formed.
///
/// * A phase the mode prices IDENTICALLY to the anchor (same resident bytes, scratch and binding
///   quantum) runs the same work: the law returns the anchor's own measurement for it, so the
///   mode's capture is a RECAPTURE of the anchor and may differ from it only by capture noise —
///   bounded by the law's measured same-cell recapture spread
///   ([`mlx_gen::ios_anchor_law::CONSISTENCY_TOLERANCE`]).
/// * A phase the mode CHANGES is held to the derivation, up to the larger of the two captures' own
///   run-to-run spread in that phase (MLX's CPU executor releases buffers asynchronously, so
///   identical runs differ by up to ~3% of a small phase; a derivation cannot be asked to beat the
///   noise of the measurement it is derived from).
fn bound(
    at_anchor: &IosPricing,
    request: &IosPricing,
    anchor: &Measured,
    mode: &Measured,
    d: &mlx_gen::ios_anchor_law::DerivedPhase,
) -> (u64, &'static str) {
    if at_anchor.phase(d.phase).is_some() && at_anchor.phase(d.phase) == request.phase(d.phase) {
        let spread = d.bytes as f64 * mlx_gen::ios_anchor_law::CONSISTENCY_TOLERANCE;
        (d.bytes + spread.ceil() as u64, " (recapture)")
    } else {
        let noise = |m: &Measured| m.spread.get(&d.phase).copied().unwrap_or(0);
        (d.bytes + noise(anchor).max(noise(mode)), "")
    }
}

/// Derive `request` from the measured `anchor` and compare it with `measured`: the table row and
/// any failures.
fn verdict(
    case: &str,
    selection: &str,
    tier: &str,
    anchor: &Measured,
    at_anchor: &IosPricing,
    request: &IosPricing,
    measured: &Measured,
) -> (String, Vec<String>) {
    let a = area(EDGE, EDGE, 1);
    let derived = derive(&anchor.worst.anchor(a), at_anchor, request, a)
        .unwrap_or_else(|e| panic!("{case} {selection} {tier}: not derivable: {e}"));
    let mut failures = Vec::new();
    let mut phases = Vec::new();
    // The whole run is bounded by its worst phase's bound (phases run in sequence).
    let demand_bound = derived
        .phases
        .iter()
        .map(|d| bound(at_anchor, request, anchor, measured, d).0)
        .max()
        .unwrap_or(0);
    for d in &derived.phases {
        let Some(&m) = measured.worst.phases.get(&d.phase) else {
            continue;
        };
        let (limit, tag) = bound(at_anchor, request, anchor, measured, d);
        let ok = m <= limit;
        phases.push(format!(
            "{} {m}/{}{tag}{}",
            d.phase.as_key(),
            d.bytes,
            if ok { "" } else { " FAIL" }
        ));
        if !ok {
            failures.push(format!(
                "{case} {selection} {tier}: phase {} measured {m} B > {limit} B (derived {} B{tag})",
                d.phase.as_key(),
                d.bytes
            ));
        }
    }
    if measured.worst.demand > demand_bound {
        failures.push(format!(
            "{case} {selection} {tier}: demand measured {} B > {demand_bound} B (derived {} B)",
            measured.worst.demand, derived.demand
        ));
    }
    (
        format!(
            "| {case} | {selection} | {tier} | {} | {} | {} | {} | {} |",
            anchor.worst.demand,
            measured.worst.demand,
            derived.demand,
            if failures.is_empty() { "PASS" } else { "FAIL" },
            phases.join(", ")
        ),
        failures,
    )
}

/// Every derived SANA case at both selections and every tier, from the `sana_1600m` text anchor.
/// `tamper` edits each derived request's pricing (the mutation check). Returns the failures.
fn prove_all(tamper: impl Fn(&Request, &mut IosPricing), tiers: usize) -> Vec<String> {
    let fx = fixture();
    println!(
        "| case | selection | tier | text anchor B | mode measured B | mode derived B | verdict | per phase measured/derived |"
    );
    println!("|---|---|---|---:|---:|---:|---|---|");
    let mut failures = Vec::new();
    for (tier, bits) in TIERS.into_iter().take(tiers) {
        let base_root = &fx.roots[&(false, tier)];
        let sprint_root = &fx.roots[&(true, tier)];
        let base = selections(crate::MODEL_ID, base_root, bits);
        let sprint = selections(crate::SPRINT_MODEL_ID, sprint_root, bits);
        for ((name, base_execution), (_, sprint_execution)) in base.iter().zip(&sprint) {
            let anchor = Request {
                sprint: false,
                img2img: false,
                root: base_root.clone(),
                execution: *base_execution,
            };
            let measured_anchor = anchor.measure();
            let at_anchor = anchor.pricing();
            let cases = [
                (
                    "sana_1600m reference",
                    Request {
                        img2img: true,
                        ..anchor.clone()
                    },
                ),
                (
                    "sana_sprint_1600m text",
                    Request {
                        sprint: true,
                        img2img: false,
                        root: sprint_root.clone(),
                        execution: *sprint_execution,
                    },
                ),
                (
                    "sana_sprint_1600m reference",
                    Request {
                        sprint: true,
                        img2img: true,
                        root: sprint_root.clone(),
                        execution: *sprint_execution,
                    },
                ),
            ];
            for (case, request) in cases {
                let mut priced = request.pricing();
                tamper(&request, &mut priced);
                let (line, f) = verdict(
                    case,
                    name,
                    tier,
                    &measured_anchor,
                    &at_anchor,
                    &priced,
                    &request.measure(),
                );
                println!("{line}");
                failures.extend(f);
            }
        }
    }
    failures
}

/// THE PROOF: base SANA's reference mode and Sprint's text and reference modes are covered by their
/// derivation from the `sana_1600m` text anchor, overall and per phase, at the fully optimized
/// selection and rung 0, dense and packed. No mode is listed in [`super::IOS_NON_ADDITIVE_MODES`].
#[test]
fn every_mode_is_derivable_from_the_text_anchor() {
    if !in_cpu_child("every_mode_is_derivable_from_the_text_anchor") {
        return;
    }
    assert!(super::IOS_NON_ADDITIVE_MODES.is_empty());
    let failures = prove_all(|_, _| {}, TIERS.len());
    assert!(failures.is_empty(), "not additive: {failures:#?}");
}

/// The proof has teeth: a derivation whose requests forget the sampler state their denoise carries
/// beside the latents (the drawn noise an img2img or Sprint denoise keeps alive, Sprint's separate
/// model input — the components the anchor's text denoise does not hold, re-added through `C_req`)
/// goes red for every such mode. Dense tier.
#[test]
fn dropping_the_carried_sampler_state_breaks_the_proof() {
    if !in_cpu_child("dropping_the_carried_sampler_state_breaks_the_proof") {
        return;
    }
    let latent = tensor_bytes(4, &[32 * u64::from(EDGE / 32) * u64::from(EDGE / 32)]);
    let failures = prove_all(
        |request, pricing| {
            let state =
                latent * (u64::from(request.img2img || request.sprint) + u64::from(request.sprint));
            for phase in &mut pricing.phases {
                if phase.phase == IosPhase::Denoise {
                    phase.resident_bytes -= state;
                }
            }
        },
        1,
    );
    println!("{failures:#?}");
    for case in [
        "sana_1600m reference",
        "sana_sprint_1600m text",
        "sana_sprint_1600m reference",
    ] {
        assert!(
            failures
                .iter()
                .any(|f| f.starts_with(case) && f.contains("denoise")),
            "{case}: dropping its carried sampler state must break the derivation"
        );
    }
}
