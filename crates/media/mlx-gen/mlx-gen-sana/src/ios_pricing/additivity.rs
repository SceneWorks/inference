//! SANA source-native bounds exercised through the iOS anchor law's
//! [`mlx_gen::ios_anchor_law::derive_certified`] consumer (epic 16774).
//! Base img2img and Sprint text/reference requests retain their actual source
//! paths, parameter/tier/geometry domains and endpoint ownership. Raw counted
//! prices never supply native proof authority or a subtractable anchor lower.
//!
//! The safe certificate here is a complete cumulative native backing upper,
//! including every source operation and actual solver evaluation, with NO
//! retirement credit. It is deliberately not a claim of useful phone fit,
//! physical-process transfer or text-anchor cancellation. Such credit requires
//! a verified native owner-retirement contract and persistent-owner proof.
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
use mlx_gen::ios_anchor_law::{area, derive_certified, MeasuredRun};
use mlx_gen::ios_pricing::{IosAnchorObservation, IosWindowBound};

#[path = "capture_observations.rs"]
mod capture_observations;
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
    let conditioning = SanaConditioning {
        cond,
        cond_mask,
        uncond,
    };
    if crate::quantum::retired_lifecycle()? {
        crate::pipeline::materialize_conditioning_if_retired(&conditioning)?;
    } else {
        let mut arrays = vec![&conditioning.cond, &conditioning.cond_mask];
        if let Some((u, um)) = &conditioning.uncond {
            arrays.push(u);
            arrays.push(um);
        }
        mlx_rs::transforms::eval(arrays)?;
    }
    Ok(conditioning)
}

impl Request {
    /// The fixture's REAL input carrier. The pipeline below borrows this same
    /// image/control object; no metadata-only request or pixel clone is used.
    fn actual_generation(&self) -> mlx_gen::GenerationRequest {
        mlx_gen::GenerationRequest {
            prompt: "a fox".into(),
            width: EDGE,
            height: EDGE,
            count: 1,
            steps: Some(2),
            seed: Some(3),
            strength: self.img2img.then_some(0.9),
            memory: Some(self.execution),
            conditioning: if self.img2img {
                vec![mlx_gen::Conditioning::Reference {
                    image: mlx_gen::Image {
                        width: EDGE,
                        height: EDGE,
                        pixels: (0..EDGE * EDGE * 3).map(|i| (i % 251) as u8).collect(),
                    },
                    strength: Some(0.9),
                }]
            } else {
                Vec::new()
            },
            ..Default::default()
        }
    }
    fn pipeline_request<'a>(
        &self,
        actual: &'a mlx_gen::GenerationRequest,
    ) -> SanaGenerateRequest<'a> {
        SanaGenerateRequest {
            width: actual.width,
            height: actual.height,
            steps: actual.steps.map(|n| n as usize),
            seed: actual.seed,
            init_image: actual.conditioning.iter().find_map(|role| match role {
                mlx_gen::Conditioning::Reference { image, .. } => Some(image),
                _ => None,
            }),
            strength: actual.strength,
            ..SanaGenerateRequest::new(&actual.prompt)
        }
    }
    /// Actual provider begin/configure, then the SAME shared native invocation
    /// envelope used by BoundGenerator. Diagnostic budget is the ALREADY active
    /// CPU fixture reading, explicitly not phone RAM/process admission evidence.
    fn with_native_lifecycle<T>(
        &self,
        actual: &mlx_gen::GenerationRequest,
        body: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        use mlx_gen::gen_core::{
            MemoryBehaviorRoute, MemoryBudget, MemoryGeometry, MemoryMode, MemoryRunOutcome,
        };
        use sha2::{Digest, Sha256};
        use std::fmt::Write as _;
        let provider = if self.sprint {
            crate::SPRINT_MODEL_ID
        } else {
            crate::MODEL_ID
        };
        let contract =
            crate::memory_strategy::weights_free_memory_strategy_contract(provider, &self.spec)?;
        let tier = MemoryNumericTier {
            precision: self.spec.precision,
            quant: self.spec.quantize,
            component_precision_floors: &[],
        };
        let selection = MemoryStrategy::ALL
            .into_iter()
            .find_map(|strategy| {
                contract
                    .selection_space(strategy, tier, false)
                    .ok()?
                    .into_iter()
                    .find(|selection| contract.generation_memory(selection) == Some(self.execution))
            })
            .ok_or_else(|| Error::Unsupported("fixture has no actual provider selection".into()))?;
        let mut context = mlx_gen::gen_core::standard_memory_behavior_context(
            &contract,
            selection.strategy,
            tier,
            MemoryBehaviorRoute {
                mode: if self.img2img {
                    MemoryMode::ImageToImage
                } else {
                    MemoryMode::TextToImage
                },
                reference_count: u32::from(self.img2img),
                use_pid: false,
                has_phases: true,
                overlay: None,
            },
        )?;
        context.selection = selection;
        context.geometry = MemoryGeometry {
            width: actual.width,
            height: actual.height,
            batch: actual.count,
            frames: 1,
            reference_count: actual.memory_reference_count(),
        };
        context.budget = MemoryBudget {
            total_bytes: mlx_gen::ios_quantum::available_bytes().ok_or_else(|| {
                Error::Unsupported("actual CPU diagnostic allowance missing".into())
            })?,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        };
        let page = mlx_gen::ios_quantum::native_metal_page_bytes()
            .ok_or_else(|| Error::Unsupported("actual native page metadata missing".into()))?;
        let sampling = super::native_certificate::fixture_sampling(
            actual,
            self.sprint,
            self.img2img,
            1 << 20,
        )?;
        context.predicted_peak_bytes = super::native_graph::cumulative_upper(
            &self.facts(),
            &self.plan(),
            &sampling,
            page,
            super::native_alloc::NativeBackend::Cpu,
        )?;
        let streams = mlx_gen::ios_quantum::NativeRetirementStreams::prepare()?;
        streams.drain()?;
        let mut scope =
            crate::memory_strategy::begin_request(provider, &self.spec, &contract, &context)?
                .ok_or_else(|| {
                    Error::Unsupported("actual fixture provider has no lifecycle".into())
                })?;
        if let Err(primary) = scope.configure_admitted_request(actual) {
            if let Err(cleanup) = streams.drain() {
                eprintln!("fixture configure drain failed: {cleanup}");
            }
            if let Err(cleanup) = scope.finish(MemoryRunOutcome::Error {
                message: primary.to_string(),
            }) {
                eprintln!("fixture provider cleanup failed: {cleanup}");
            }
            if let Err(cleanup) = streams.drain() {
                eprintln!("fixture final configure drain failed: {cleanup}");
            }
            return Err(primary.into());
        }
        // Exact fixture-only identity streams the ORIGINAL request's derived
        // Debug into SHA, never allocates a whole Debug String or clones media.
        // It is not presented as the catalog's reusable admission binding.
        struct Writer<'a>(&'a mut Sha256);
        impl std::fmt::Write for Writer<'_> {
            fn write_str(&mut self, text: &str) -> std::fmt::Result {
                self.0.update(text.as_bytes());
                Ok(())
            }
        }
        let mut digest = Sha256::new();
        digest.update(b"sana-actual-capture-input-v1\0");
        write!(Writer(&mut digest), "{actual:?}").expect("infallible digest writer");
        let registration = mlx_gen::ios_quantum::ConfiguredNativeRetirement::register(
            mlx_gen::ios_quantum::NativeRetirementBinding {
                owner_token: actual as *const mlx_gen::GenerationRequest as usize,
                work: digest.finalize().into(),
                provider,
            },
        );
        mlx_gen::ios_quantum::invoke_native_retirement(
            &registration.lease(),
            &streams,
            actual,
            body,
            |outcome| {
                let result = scope.finish(outcome);
                registration.invalidate();
                result.map_err(Error::from)
            },
            |error| matches!(error, Error::Canceled),
        )
    }

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
        let mut pricing = IosPricing::default();
        for quantum in priced_quanta_in_regime(
            &self.facts(),
            &self.plan(),
            !self.sprint,
            SanaPricingRegime::RetiredLifecycle,
        )
        .unwrap()
        {
            pricing.take_window(quantum.window);
        }
        if !self.img2img {
            pricing.phase_not_run(IosPhase::Encode, "text mode has no reference encode");
        }
        pricing
    }

    /// One fresh load + generate in the generator's `Sequential` phase order
    /// (`model::Sana::generate_impl`: `Residency::run_staged`, or `run_staged_img2img` for a
    /// reference), with the production loaders' guards and materialization at the fixture's
    /// geometry.
    fn run(&self, actual: &mlx_gen::GenerationRequest) -> Result<[u8; 32]> {
        self.with_native_lifecycle(actual, || self.run_body(actual))
    }
    fn ordinary_pixels(&self, actual: &mlx_gen::GenerationRequest) -> [u8; 32] {
        assert_eq!(mlx_gen::ios_quantum::native_retirement_binding(), None);
        with_cpu_diagnostic_headroom(AMPLE, || self.run_body(actual)).unwrap()
    }
    fn run_body(&self, actual: &mlx_gen::GenerationRequest) -> Result<[u8; 32]> {
        let root = &self.root;
        let (sprint, img2img) = (self.sprint, self.img2img);
        let memory = Some(self.execution);
        let plan = self.plan();
        let cancel = actual.cancel.clone();
        if !plan.sequential {
            return self.run_resident_body(actual, |_| {});
        }

        // Optimized: the actual encoder is dropped before either heavy component loads.
        let conditioning = {
            let encoder = load_fixture_encoder(root)?;
            fixture_conditioning(&encoder, sprint)?
        };
        mlx_gen::residency::drain_allocator_cache();

        let req = self.pipeline_request(actual);

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
            crate::pipeline::retire_staged_reference_encoder()?;
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
        let image = SanaHeavy::light_view(&light).decode_one(&latent, &cancel, tiling.as_ref())?;
        Ok(pixel_digest(&image))
    }

    /// The production warm-pair state machine owns the literal resident endpoint. In particular,
    /// its heavy callback receives true even for TEXT and retains one DC-AE encoder alongside the
    /// decoder from the same source. Nothing here invents extra component copies or retention.
    // Test-only scalar observation of the SAME borrowed conditioning owners. It neither clones
    // arrays nor evaluates extra graphs and does not change the resident warm-pair lifecycle.
    fn run_resident_observing(&self, observe: impl FnMut(&SanaConditioning)) -> Result<()> {
        let actual = self.actual_generation();
        self.with_native_lifecycle(&actual, || self.run_resident_body(&actual, observe))
            .map(|_| ())
    }
    fn run_resident_body(
        &self,
        actual: &mlx_gen::GenerationRequest,
        mut observe: impl FnMut(&SanaConditioning),
    ) -> Result<[u8; 32]> {
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
        let cancel = actual.cancel.clone();
        let output = residency
            .with_resident_parts(|encoder, heavy| -> Result<[u8; 32]> {
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
                let req = self.pipeline_request(actual);
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
                let image = heavy
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
                Ok(pixel_digest(&image))
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
        Ok(output)
    }

    /// [`RUNS`] measured runs (each a fresh load + generate).
    fn measure(&self) -> Measured {
        // Same saved weights, seed, controls and ORIGINAL carrier, ordinary
        // graph first. Joined scheduling must preserve the actual RGB bytes.
        // This comparison lives only in already selected heavy CPU captures.
        let actual = self.actual_generation();
        let expected_pixels = self.ordinary_pixels(&actual);
        let mut worst = MeasuredRun {
            demand: 0,
            phases: BTreeMap::new(),
        };
        let mut least: BTreeMap<IosPhase, u64> = BTreeMap::new();
        let mut observations = Vec::new();
        for _ in 0..RUNS {
            let (result, observation) = capture_observations::measure(|| self.run(&actual));
            let (pixels, run) = result.unwrap();
            assert_eq!(
                pixels, expected_pixels,
                "joined SANA changed actual pixels/seed/controls"
            );
            observations.push(observation);
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
        Measured {
            worst,
            spread,
            observations,
        }
    }
}

/// One configuration's measurement: the worst of [`RUNS`] fresh runs per phase, and how far the
/// runs disagreed per phase (the capture's own spread).
fn pixel_digest(image: &mlx_gen::Image) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut digest = Sha256::new();
    digest.update(image.width.to_le_bytes());
    digest.update(image.height.to_le_bytes());
    digest.update(&image.pixels);
    digest.finalize().into()
}

struct Measured {
    worst: MeasuredRun,
    spread: BTreeMap<IosPhase, u64>,
    observations: Vec<IosAnchorObservation>,
}

/// A source-certified cumulative upper is an absolute native ceiling, not a
/// calibrated recapture. Preserve measured spread as a diagnostic, NEVER as an
/// allowance that could turn an undercounted native certificate green.
fn bound(
    _at_anchor: &IosPricing,
    _request: &IosPricing,
    _anchor: &Measured,
    _mode: &Measured,
    d: &mlx_gen::ios_anchor_law::DerivedPhase,
) -> (u64, &'static str) {
    (d.bytes, " (source-native cumulative)")
}

/// Diagnostic identity of one case; no pricing/proof inputs are bundled or changed.
struct VerdictCase<'a> {
    case: &'a str,
    selection: &'a str,
    tier: &'a str,
}

/// Derive `request` from the measured `anchor` and compare it with `measured`: the table row and
/// any failures.
fn verdict(
    label: VerdictCase<'_>,
    anchor: &Measured,
    at_anchor: &IosPricing,
    request: &IosPricing,
    bounds: &[IosWindowBound],
    measured: &Measured,
) -> (String, Vec<String>) {
    let VerdictCase {
        case,
        selection,
        tier,
    } = label;
    let a = area(EDGE, EDGE, 1);
    let derived = derive_certified(&anchor.worst.anchor(a), at_anchor, request, bounds)
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
        if request.absent_phases.contains_key(&d.phase) {
            assert_eq!(
                d.bytes, 0,
                "an actually absent phase has zero native demand"
            );
            assert!(
                !measured.worst.phases.contains_key(&d.phase),
                "an absent phase must not execute"
            );
            continue;
        }
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
            "{} {m}/{}{tag} spread={}{}",
            d.phase.as_key(),
            d.bytes,
            measured.spread.get(&d.phase).copied().unwrap_or(0),
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

/// Remove one REAL recorded source epoch, keeping scalar chronology contiguous.
/// The independent exact source multiplicity check must still reject the trace.
fn remove_actual_block_epoch(observation: &mut IosAnchorObservation) {
    let index = observation
        .checks
        .iter()
        .position(|check| check.label == labels::BLOCK)
        .expect("actual trunk block must execute");
    let check = observation.checks.remove(index);
    let epoch = check
        .context
        .into_iter()
        .find_map(|index| {
            let mark = &observation.metadata[index as usize];
            (mark.provider == "mlx" && mark.kind == "native-quantum").then(|| mark.fields["epoch"])
        })
        .expect("actual block epoch context");
    for mark in &mut observation.metadata {
        if mark.provider == "mlx"
            && matches!(mark.kind.as_str(), "native-quantum" | "native-joined")
        {
            let actual = mark.fields.get_mut("epoch").expect("actual epoch scalar");
            if *actual == epoch {
                mark.kind = "actual-test-removed-epoch".into();
            } else if *actual > epoch {
                *actual -= 1;
            }
        }
        if mark.kind == "native-lifecycle-completed" {
            *mark
                .fields
                .get_mut("epochs")
                .expect("actual endpoint epoch total") -= 1;
        }
    }
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
            let expected_anchor = anchor.pricing();
            let mut at_anchor = expected_anchor.clone();
            at_anchor.observations = measured_anchor.observations.clone();
            let anchor_facts = anchor.facts();
            let anchor_plan = anchor.plan();
            let anchor_generation = anchor.actual_generation();
            let anchor_sampling = super::native_certificate::fixture_sampling(
                &anchor_generation,
                false,
                false,
                1 << 20,
            )
            .unwrap();
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
                let measured = request.measure();
                let expected = request.pricing();
                let mut priced = expected.clone();
                priced.observations = measured.observations.clone();
                let facts = request.facts();
                let plan = request.plan();
                let generation = request.actual_generation();
                let sampling = super::native_certificate::fixture_sampling(
                    &generation,
                    request.sprint,
                    request.img2img,
                    1 << 20,
                )
                .unwrap();
                let certify = |observed_anchor: &IosPricing, supplied: &IosPricing| {
                    use super::native_certificate::CertificateProfile;
                    super::native_certificate::certify_profiles(
                        CertificateProfile {
                            facts: &anchor_facts,
                            plan: &anchor_plan,
                            expected: &expected_anchor,
                            observed: observed_anchor,
                            sampling: Some(&anchor_sampling),
                        },
                        CertificateProfile {
                            facts: &facts,
                            plan: &plan,
                            expected: &expected,
                            observed: supplied,
                            sampling: Some(&sampling),
                        },
                        SanaPricingRegime::RetiredLifecycle,
                        1 << 20,
                    )
                };
                // BOTH positive and mutation captures require the intact source
                // certificate and successful derivation before any negative edit.
                let baseline = certify(&at_anchor, &priced).unwrap_or_else(|e| {
                    panic!("{case} {name} {tier}: intact source certificate refused: {e}")
                });
                let (_, baseline_failures) = verdict(
                    VerdictCase {
                        case,
                        selection: name,
                        tier,
                    },
                    &measured_anchor,
                    &at_anchor,
                    &priced,
                    &baseline,
                    &measured,
                );
                assert!(
                    baseline_failures.is_empty(),
                    "intact certificate underbounds: {baseline_failures:#?}"
                );
                // Mutate ACTUAL marks, not made-up observation fixtures. Changed
                // graph/lifetime context, missing geometry and invalid allocator
                // metadata must each invalidate the source correspondence.
                for (provider, kind, field) in [
                    ("gemma2", "native-forward", "tokens"),
                    ("sana", "native-trunk", "streamed"),
                    ("sana", "native-decoder", "tail_stages"),
                    ("sana", "native-pixels", "nchw_view_of_decoder"),
                    ("sana", "native-pixels", "held_unscaled_elements"),
                    ("mlx", "allocation-scope", "page_bytes"),
                    (crate::MODEL_ID, "native-lifecycle", "configured"),
                    (crate::MODEL_ID, "native-lifecycle-completed", "status"),
                    ("mlx", "native-joined", "all_source_streams"),
                    ("mlx", "native-joined", "range_captured_prefix"),
                    ("sana", "native-dcae-glu", "original_elements"),
                    ("sana", "native-selected", "caption_elements"),
                ] {
                    let mut changed = at_anchor.clone();
                    let mut edits = 0;
                    for observation in &mut changed.observations {
                        for mark in &mut observation.metadata {
                            if mark.provider == provider && mark.kind == kind {
                                if field == "page_bytes" {
                                    mark.fields.insert(field.into(), 3);
                                } else if field == "streamed" {
                                    mark.fields.insert(field.into(), 1);
                                } else if field == "held_unscaled_elements" {
                                    mark.fields.insert(field.into(), 0);
                                } else {
                                    mark.fields.remove(field);
                                }
                                edits += 1;
                            }
                        }
                    }
                    assert!(edits > 0, "actual {provider}/{kind} mark was not exercised");
                    assert!(
                        certify(&changed, &priced).is_err(),
                        "changed {provider}/{kind}/{field} certified"
                    );
                }
                let mut truncated = at_anchor.clone();
                for observation in &mut truncated.observations {
                    observation.telemetry_complete = false;
                }
                assert_eq!(
                    super::observed_pricing_regime(&truncated),
                    SanaPricingRegime::Ordinary
                );
                assert!(
                    certify(&truncated, &priced).is_err(),
                    "truncated receipts granted additional tight credit"
                );
                let mut missing_epoch = at_anchor.clone();
                for observation in &mut missing_epoch.observations {
                    remove_actual_block_epoch(observation);
                }
                assert_eq!(
                    super::observed_pricing_regime(&missing_epoch),
                    SanaPricingRegime::RetiredLifecycle,
                    "mutation must preserve a structurally complete chronology"
                );
                assert!(
                    certify(&missing_epoch, &priced).is_err(),
                    "missing whole source block epoch certified"
                );
                tamper(&request, &mut priced);
                let bounds = match certify(&at_anchor, &priced) {
                    Ok(bounds) => bounds,
                    Err(error) => {
                        failures.push(format!(
                            "{case} {name} {tier}: denoise source certificate invalid: {error}"
                        ));
                        continue;
                    }
                };
                let (line, f) = verdict(
                    VerdictCase {
                        case,
                        selection: name,
                        tier,
                    },
                    &measured_anchor,
                    &at_anchor,
                    &priced,
                    &bounds,
                    &measured,
                );
                println!("{line}");
                failures.extend(f);
            }
        }
    }
    failures
}

/// Ordinary macOS CI walking skeleton: real tiny model execution, NOT a peak
/// measurement or physical/process certificate. One dense Sprint text request,
/// same seed/weights/carrier, with exact ordinary-versus-joined output parity.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn joined_native_sprint_matches_ordinary_pixels_and_completes_receipts() {
    if !in_cpu_child("joined_native_sprint_matches_ordinary_pixels_and_completes_receipts") {
        return;
    }
    let _completion = CpuChildCompletion;
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("sprint-dense");
    write_snapshot(&root, true);
    let (_, execution, spec) = selections(crate::SPRINT_MODEL_ID, &root, None)
        .into_iter()
        .next()
        .unwrap();
    let case = Request {
        sprint: true,
        img2img: false,
        root,
        execution,
        spec,
    };
    let mut actual = case.actual_generation();
    actual.steps = Some(1);
    let ordinary = case.ordinary_pixels(&actual);
    let (joined, observation) = capture_observations::record(|| case.run(&actual));
    assert_eq!(
        joined.expect("real joined tiny SANA execution failed"),
        ordinary,
        "retirement boundaries must not change actual RGB bytes"
    );
    assert!(observation.telemetry_complete);
    assert_eq!(
        observation.domain,
        mlx_gen::ios_pricing::IosMemoryDomain::Unspecified,
        "the non-measuring recorder must not declare native allocation evidence"
    );
    assert_eq!(observation.baseline_bytes, 0);
    for label in [
        labels::SCM_STEP,
        labels::DECODER_GLU,
        labels::DECODED_PIXELS,
    ] {
        assert!(
            observation.checks.iter().any(|check| check.label == label),
            "real source execution did not reach {label}"
        );
    }
    super::native_certificate::validate_retirement_receipts(
        observation.metadata.len(),
        |index| {
            observation.metadata.get(index).map(|mark| {
                super::native_certificate::RetirementMetadataView {
                    provider: &mark.provider,
                    kind: &mark.kind,
                    fields: &mark.fields,
                }
            })
        },
        observation
            .checks
            .iter()
            .map(|check| super::native_certificate::RetirementCheckView {
                phase: check.phase,
                label: &check.label,
                context: &check.context,
                count: 1,
            }),
        observation.telemetry_complete,
    )
    .expect("real configured invocation must complete every captured-prefix receipt");

    eprintln!("\nMODEL_INFERENCE_BODY_COMPLETED ios_pricing::additivity::joined_native_sprint_matches_ordinary_pixels_and_completes_receipts");
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
