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
//! load forms: FULLY OPTIMIZED (`Sequential` + `DeferredMaterialization`, deepest legal bounded
//! attention/decode; transformer windows intentionally withheld) and literal FULLY RESIDENT
//! (`Resident` + `EagerMaterialization`, actual eager warm-pair ownership through decode). The text
//! anchor and the mode are each MEASURED ([`mlx_gen::ios_anchor_law::measure_phases`]: MLX
//! active-memory growth per phase over a fresh load + generate, the worst of [`RUNS`]); the mode's
//! demand derived from the anchor must cover the
//! mode's measurement, overall and in every phase both ran. CPU child, Metal fatal.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use mlx_gen::gen_core::{
    GenerationMemory, LoadShape, LoadSpec, MemoryNumericTier, MemoryStrategy,
    MemoryStrategySupport, OffloadPolicy, Quant,
};
use mlx_gen::ios_anchor_law::{area, derive, measure_phases, MeasuredRun};
use mlx_gen::ios_quantum::with_cpu_diagnostic_headroom;
use mlx_gen::weights::Weights;
use mlx_gen::StagedHeavy;
use mlx_rs::Array;

use super::*;
use crate::dc_ae::DcAeEncoder;
use crate::pipeline::{SanaConditioning, SanaGenerateRequest, SanaHeavy};
use crate::text_encoder::Gemma2;

const CHILD: &str = "SANA_IOS_ADDITIVITY_CPU_CHILD";

const DONE: &str = "SANA_IOS_ADDITIVITY_CPU_CHILD_DONE";

// Existing CPU-child isolation, with a normal-return marker after every fixture has dropped.
struct CpuChildCompletion;
impl Drop for CpuChildCompletion {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            std::fs::write(
                std::env::var(DONE).expect("CPU child completion path"),
                b"ok",
            )
            .expect("write CPU child completion marker");
        }
    }
}

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
        let done = tempfile::tempdir().unwrap();
        let marker = done.path().join("done");
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "--include-ignored",
                &format!("ios_pricing::additivity::{name}"),
                "--nocapture",
                "--test-threads=1",
            ])
            .env(CHILD, "1")
            .env("IOS_CAPTURE_IN_CHILD", "1")
            .env(DONE, &marker)
            .status()
            .unwrap();
        assert!(status.success(), "CPU child for {name} failed");
        assert!(
            marker.is_file(),
            "CPU child for {name} ran no completed body"
        );
        return false;
    }
    mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
    assert_eq!(
        mlx_rs::Device::try_default().expect("CPU default-device query failed"),
        mlx_rs::Device::cpu()
    );
    assert_eq!(
        unsafe { mlx_sys::mlx_set_default_stream(mlx_rs::Stream::cpu().as_ptr()) },
        0
    );
    assert_eq!(
        mlx_rs::Stream::new().get_index().unwrap(),
        mlx_rs::Stream::cpu().get_index().unwrap()
    );
    assert!(crate::test_metal_guard::forbid_metal());
    assert!(crate::test_metal_guard::metal_forbidden());
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

/// Each literal capture endpoint's own load spec and contract: (fully optimized, fully resident).
/// Prior resident-labelled runs staged components and do not validate the corrected eager endpoint.
fn selections(
    provider: &str,
    root: &Path,
    bits: Option<i32>,
) -> [(String, GenerationMemory, LoadSpec); 2] {
    [false, true].map(|resident| {
        let mut spec = LoadSpec::new(WeightsSource::Dir(root.to_path_buf()))
            .with_offload_policy(if resident {
                OffloadPolicy::Resident
            } else {
                OffloadPolicy::Sequential
            })
            .with_load_shape(if resident {
                LoadShape::EagerMaterialization
            } else {
                LoadShape::DeferredMaterialization
            });
        spec.quantize = bits.map(|b| match b {
            4 => Quant::Q4,
            8 => Quant::Q8,
            _ => panic!("unsupported fixture tier {b}"),
        });
        let contract =
            crate::memory_strategy::weights_free_memory_strategy_contract(provider, &spec).unwrap();
        let tier = MemoryNumericTier {
            precision: spec.precision,
            quant: spec.quantize,
            component_precision_floors: &[],
        };
        let strategy = if resident {
            MemoryStrategy::Resident
        } else {
            MemoryStrategy::ALL
                .into_iter()
                .rev()
                .find(|s| {
                    contract
                        .capability(*s)
                        .is_some_and(|c| c.support == MemoryStrategySupport::Implemented)
                })
                .unwrap()
        };
        let selection = *contract
            .selection_space(strategy, tier, false)
            .unwrap()
            .last()
            .unwrap();
        let execution = contract
            .generation_memory(&selection)
            .expect("endpoint execution must be defined");
        crate::memory_strategy::validate_request_memory(provider, &spec, &execution).unwrap();
        assert_eq!(execution.stage_residency, !resident);
        assert!(
            !execution.stream_transformer_blocks,
            "SANA transformer windows are intentionally withheld"
        );
        if !resident {
            assert!(execution.tile_vae_decode && execution.chunk_attention);
            assert_eq!(
                execution.decode_tile_edge,
                crate::memory_strategy::DECODE_TILE_EDGES
                    .iter()
                    .copied()
                    .min()
            );
            assert_eq!(
                execution.attention_chunk_size,
                crate::memory_strategy::ATTENTION_CHUNK_SIZES
                    .iter()
                    .copied()
                    .min()
            );
        }
        (format!("{strategy:?}"), execution, spec)
    })
}

/// One request: variant, mode, snapshot root and its resolved execution.
#[derive(Clone)]
struct Request {
    sprint: bool,
    img2img: bool,
    root: PathBuf,
    execution: GenerationMemory,
    spec: LoadSpec,
}

fn storage_witness(array: &Array) -> usize {
    let mut available = false;
    assert_eq!(
        unsafe { mlx_sys::_mlx_array_is_available(&mut available, array.as_ptr()) },
        0
    );
    assert!(
        available,
        "endpoint owns a lazy stored parameter, not resident storage"
    );
    let address = unsafe { mlx_sys::mlx_array_data_uint8(array.as_ptr()) } as usize;
    assert_ne!(address, 0);
    address
}

fn load_fixture_encoder(root: &Path) -> Result<Gemma2> {
    mlx_gen::ios_quantum::before_load(labels::TEXT_LOAD, &[root.join("text_encoder")])?;
    let w = Weights::from_file(root.join("text_encoder/gemma-2-2b-it.safetensors"))?;
    let encoder = Gemma2::from_weights(&w, "model.", &gemma())?;
    // Exactly SanaTextEncoder::from_snapshot's eager materialization and alias ownership.
    w.materialize_accessed()?;
    // Observe the actual materialized source aliases without retaining copies. Derived Gemma
    // normalization expressions may legitimately stay lazy; do not force them to evaluate here.
    let accessed = w.accessed_entries();
    assert!(!accessed.is_empty());
    for (_, array) in &accessed {
        storage_witness(array);
    }
    Ok(encoder)
}

fn fixture_conditioning(encoder: &Gemma2, sprint: bool) -> Result<SanaConditioning> {
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
    Ok(SanaConditioning {
        cond,
        cond_mask,
        uncond,
    })
}

impl Request {
    /// The plan exactly as [`super::ios_pricing`] resolves it for this actual endpoint load.
    fn plan(&self) -> SanaPricingPlan {
        let memory = Some(self.execution);
        assert_eq!(
            self.execution.stage_residency,
            self.spec.offload_policy == OffloadPolicy::Sequential
        );
        assert_eq!(
            self.spec.load_shape,
            if self.execution.stage_residency {
                LoadShape::DeferredMaterialization
            } else {
                LoadShape::EagerMaterialization
            }
        );
        SanaPricingPlan {
            sprint: self.sprint,
            sequential: self.spec.offload_policy == OffloadPolicy::Sequential,
            img2img: self.img2img,
            width: EDGE,
            height: EDGE,
            count: 1,
            decode_tiling: crate::pipeline::resolved_decode_plan(
                memory,
                self.spec.offload_policy == OffloadPolicy::Sequential,
            )
            .map(|p| (p.edge, p.overlap)),
            encode_tile: crate::pipeline::resolved_encode_tile_on(memory, true),
            attention: crate::pipeline::resolved_rung_plan(memory, 2)
                .unwrap()
                .attention,
        }
    }

    fn facts(&self) -> SanaPricingFacts {
        SanaPricingFacts {
            root: self.root.clone(),
            transformer: trunk_cfg(self.sprint),
            dc_ae: dcae(),
            gemma: gemma(),
            gemma_tokens: WINDOW as u64,
            caption_tokens: CAPTION as u64,
            text_encoder: Inventory::read(&self.root.join("text_encoder")).unwrap(),
            trunk: Inventory::read(&self.root.join("transformer")).unwrap(),
            vae: Inventory::read(&self.root.join("vae")).unwrap(),
        }
    }

    fn pricing(&self) -> IosPricing {
        price(&self.facts(), &self.plan()).unwrap()
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
        if !plan.sequential {
            return self.run_resident();
        }

        // Optimized: the actual encoder is dropped before either heavy component loads.
        let conditioning = {
            let encoder = load_fixture_encoder(root)?;
            fixture_conditioning(&encoder, sprint)?
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
        let streamable = crate::memory_strategy::is_streamable(&self.spec);
        let heavy = crate::model::load_heavy_with_config(
            root,
            sprint,
            false,
            streamable,
            dcae(),
            trunk_cfg(sprint),
        )?;
        assert_eq!(
            heavy.endpoint_inventory(),
            (2, false, streamable),
            "optimized heavy has no reference encoder and no missing trunk blocks"
        );
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

    /// The production warm-pair state machine owns the literal resident endpoint. In particular,
    /// its heavy callback receives true even for TEXT and retains one DC-AE encoder alongside the
    /// decoder from the same source. Nothing here invents extra component copies or retention.
    fn run_resident(&self) -> Result<()> {
        self.run_resident_observing(|_| {})
    }

    // Test-only scalar observation of the SAME borrowed conditioning owners. It neither clones
    // arrays nor evaluates extra graphs and does not change the resident warm-pair lifecycle.
    fn run_resident_observing(&self, mut observe: impl FnMut(&SanaConditioning)) -> Result<()> {
        let text_root = self.root.clone();
        let heavy_root = self.root.clone();
        let sprint = self.sprint;
        let streamable = crate::memory_strategy::is_streamable(&self.spec);
        let callbacks = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let text_calls = callbacks.clone();
        let heavy_calls = callbacks.clone();
        let residency = mlx_gen::Residency::<Gemma2, SanaHeavy>::from_policy(
            self.spec.offload_policy,
            move || {
                text_calls.lock().unwrap().push("text");
                load_fixture_encoder(&text_root)
            },
            move |needs_encoder| {
                heavy_calls.lock().unwrap().push("heavy");
                assert!(
                    needs_encoder,
                    "production resident warm callback must request the reusable encoder superset"
                );
                crate::model::load_heavy_with_config(
                    &heavy_root,
                    sprint,
                    needs_encoder,
                    streamable,
                    dcae(),
                    trunk_cfg(sprint),
                )
            },
        )?;
        assert_eq!(
            *callbacks.lock().unwrap(),
            ["text", "heavy"],
            "all resident components load eagerly, before conditioning"
        );
        assert!(!residency.is_sequential());
        let memory = Some(self.execution);
        let plan = self.plan();
        let cancel = mlx_gen::CancelFlag::default();
        residency
            .with_resident_parts(|encoder, heavy| -> Result<()> {
                assert_eq!(
                    heavy.endpoint_inventory(),
                    (2, true, false),
                    "literal resident warm inventory (including TEXT encoder superset)"
                );
                let storage: Vec<_> = heavy
                    .endpoint_storage_witnesses()
                    .into_iter()
                    .map(storage_witness)
                    .collect();
                assert_eq!(
                    storage.len(),
                    3,
                    "caption norm and every resident trunk block have real materialized storage"
                );
                let conditioning = fixture_conditioning(encoder, sprint)?;
                observe(&conditioning);
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
                    init_image: self.img2img.then_some(&image),
                    strength: self.img2img.then_some(0.9),
                    ..SanaGenerateRequest::new("a fox")
                };
                let reference =
                    heavy.prepare_reference_with_tiling(&req, &cancel, plan.encode_tile)?;
                assert_eq!(reference.is_some(), self.img2img);
                let forward =
                    crate::pipeline::resolved_rung_plan(memory, heavy.transformer_blocks())?;
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
                drop((conditioning, reference));
                assert_eq!(
                    heavy.endpoint_inventory(),
                    (2, true, false),
                    "no resident shedding before decode"
                );
                let tiling = crate::pipeline::resolve_decode_tiling(memory, false);
                heavy
                    .decode_view()
                    .decode_one(&latent, &cancel, tiling.as_ref())?;
                // Both actual owners are still borrowed from the SAME warm pair through decode.
                assert_eq!(heavy.endpoint_inventory(), (2, true, false));
                assert_eq!(
                    heavy
                        .endpoint_storage_witnesses()
                        .into_iter()
                        .map(storage_witness)
                        .collect::<Vec<_>>(),
                    storage,
                    "same materialized trunk parameters survive conditioning, denoise and decode"
                );
                let _ = encoder;
                Ok(())
            })?
            .expect("Resident must own the eager warm pair")?;
        assert_eq!(
            *callbacks.lock().unwrap(),
            ["text", "heavy"],
            "no per-phase reload or rematerialization"
        );
        assert!(residency
            .with_resident_parts(|_, heavy| heavy.endpoint_inventory())?
            .is_some());
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
            failures.push(format!(
                "{case} {selection} {tier}: missing measured phase {}",
                d.phase.as_key()
            ));
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
        for ((name, base_execution, base_spec), (_, sprint_execution, sprint_spec)) in
            base.iter().zip(&sprint)
        {
            let anchor = Request {
                sprint: false,
                img2img: false,
                root: base_root.clone(),
                execution: *base_execution,
                spec: base_spec.clone(),
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
                        spec: sprint_spec.clone(),
                    },
                ),
                (
                    "sana_sprint_1600m reference",
                    Request {
                        sprint: true,
                        img2img: true,
                        root: sprint_root.clone(),
                        execution: *sprint_execution,
                        spec: sprint_spec.clone(),
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

/// Literal resident TEXT uses the production warm callback's encoder superset, independent of
/// img2img. Price each actual header inventory once (packed weights stay packed).
#[test]
#[ignore = "expensive fixture ownership capture; run through the selected iOS capture workflow"]
fn resident_text_keeps_and_prices_the_production_encoder_superset() {
    if !in_cpu_child("resident_text_keeps_and_prices_the_production_encoder_superset") {
        return;
    }
    let _completion = CpuChildCompletion;
    let fx = fixture();
    for (tier, bits) in TIERS {
        for sprint in [false, true] {
            let root = &fx.roots[&(sprint, tier)];
            let endpoints = selections(
                if sprint {
                    crate::SPRINT_MODEL_ID
                } else {
                    crate::MODEL_ID
                },
                root,
                bits,
            );
            let (_, execution, spec) = &endpoints[1];
            let request = Request {
                sprint,
                img2img: false,
                root: root.clone(),
                execution: *execution,
                spec: spec.clone(),
            };
            let facts = request.facts();
            let expected = facts.text_encoder.stored_where(|_| true)
                + facts.trunk.stored_where(|_| true)
                + facts
                    .vae
                    .f32_where(|n| n.starts_with("encoder.") || n.starts_with("decoder."));
            assert!(facts.vae.f32_where(|n| n.starts_with("encoder.")) > 0);
            let mut actual_owners = None;
            with_cpu_diagnostic_headroom(AMPLE, || {
                request.run_resident_observing(|conditioning| {
                    // Logical payload metadata of the ACTUAL borrowed outputs. A sliced caption
                    // may retain a larger hidden-state parent; these are NOT parent allocation
                    // bytes, physical lower bounds or a process-footprint certificate.
                    let owners = (
                        mlx_gen::ios_quantum::bytes(&conditioning.cond),
                        mlx_gen::ios_quantum::bytes(&conditioning.cond_mask),
                        conditioning.uncond.is_some(),
                    );
                    assert!(
                        actual_owners.replace(owners).is_none(),
                        "one actual conditioning owner set per warm run"
                    );
                })
            })
            .unwrap();
            let (caption_bytes, mask_bytes, has_uncond) =
                actual_owners.expect("actual resident conditioning producer must be observed");
            assert_eq!(
                has_uncond, !sprint,
                "Sprint is CFG-free; base fixture actually encodes the negative prompt"
            );
            let one_prompt = caption_bytes.checked_add(mask_bytes).unwrap();
            assert!(caption_bytes > 0 && mask_bytes > 0);
            let quanta = priced_quanta(&facts, &request.plan()).unwrap();
            let conditioning: Vec<_> = quanta
                .iter()
                .filter(|q| q.phase == IosPhase::Conditioning)
                .collect();
            let per_pass = usize::try_from(facts.gemma.num_layers)
                .unwrap()
                .checked_add(2)
                .unwrap();
            assert_eq!(
                conditioning.len(),
                per_pass * if has_uncond { 2 } else { 1 },
                "every actual Gemma pass is present, including CFG-free absence"
            );
            for (index, q) in conditioning.into_iter().enumerate() {
                let held = if has_uncond && index >= per_pass {
                    one_prompt
                } else {
                    0
                };
                assert_eq!(
                    q.window.resident_bytes, expected,
                    "{tier} sprint={sprint} {} structural warm encoder/decoder inventory exactly once",
                    q.label
                );
                assert_eq!(q.window.held_bytes, held,
                    "{tier} sprint={sprint} {} actual first-prompt logical owner during negative encode",
                    q.label);
                assert_eq!(
                    q.resident_bytes,
                    expected.checked_add(held).unwrap(),
                    "{tier} sprint={sprint} {} aggregate structural plus actual held payload",
                    q.label
                );
            }
        }
    }
    if std::env::var_os("IOS_CAPTURE_IN_CHILD").is_some() {
        eprintln!("\nIOS_CAPTURE_BODY_COMPLETED ios_pricing::additivity::resident_text_keeps_and_prices_the_production_encoder_superset");
    }
}

/// Legacy positive additivity regression, still requiring a source-correlated verdict. Logical
/// counts and the legacy mode inventory do not prove text-anchor inheritance. Preserve the real
/// positive assertions; Uncertified must not become passing refusal or a capture-family collapse.
#[test]
#[ignore = "expensive fixture capture; run through the selected iOS capture workflow"]
fn every_mode_is_derivable_from_the_text_anchor() {
    if !in_cpu_child("every_mode_is_derivable_from_the_text_anchor") {
        return;
    }
    let _completion = CpuChildCompletion;
    // No additivity certificate is inferred from the legacy inventory.
    let failures = prove_all(|_, _| {}, TIERS.len());
    assert!(failures.is_empty(), "not additive: {failures:#?}");
    if std::env::var_os("IOS_CAPTURE_IN_CHILD").is_some() {
        eprintln!("\nIOS_CAPTURE_BODY_COMPLETED ios_pricing::additivity::every_mode_is_derivable_from_the_text_anchor");
    }
}

/// The proof has teeth: a derivation whose requests forget the sampler state their denoise carries
/// beside the latents (the drawn noise an img2img or Sprint denoise keeps alive, Sprint's separate
/// model input — the components the anchor's text denoise does not hold, re-added through `C_req`)
/// goes red for every such mode. Dense tier.
#[test]
#[ignore = "expensive fixture capture; run through the selected iOS capture workflow"]
fn dropping_the_carried_sampler_state_breaks_the_proof() {
    if !in_cpu_child("dropping_the_carried_sampler_state_breaks_the_proof") {
        return;
    }
    let _completion = CpuChildCompletion;
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
    if std::env::var_os("IOS_CAPTURE_IN_CHILD").is_some() {
        eprintln!("\nIOS_CAPTURE_BODY_COMPLETED ios_pricing::additivity::dropping_the_carried_sampler_state_breaks_the_proof");
    }
}
