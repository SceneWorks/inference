//! `iris_3b_restore` — the Iris-3B restoration / upscaling [`Transform`] on MLX (story sc-25683).
//!
//! The restoration task is the `upscaler/` export: its own fine-tuned backbone (`config.yaml` with a
//! `task: restoration` section + `model.safetensors`) and the shipped empty-prompt states
//! (`empty_prompt.safetensors`). It runs **without the text encoder** (E4): the conditioning is the
//! empty-prompt stack, read once at load.
//!
//! Everything around the forward — the input budget, the bicubic upsample, small-image enlargement,
//! padding, 50 %-overlap tiling with upstream's Gaussian fusion, the wavelet colour fix and the 8-bit
//! quantization — is the backend-neutral [`gen_core::iris::restoration`] driver, shared with the
//! Candle twin. This crate supplies only the one-step velocity of each tile.
//!
//! [`gen_core::iris::restoration`]: mlx_gen::gen_core::iris::restoration

use gen_core::iris::restoration::{
    self as contract, plan_request, restore, RestorationPlan, RestorationResources,
    RestorationSettings, TileGeometry, DEFAULT_SCALE, INPUT_BUDGET,
};
use gen_core::iris::{IrisConfig, FAMILY};
use mlx_gen::{
    gen_core, Error, Image, LoadSpec, Progress, Result, Transform, TransformCapabilities,
    TransformDescriptor, TransformRegistration, TransformRequest,
};
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};

use crate::dit::{IrisDiT, TextBatch};
use crate::model::{compute_dtype, load_backbone};

/// Registry id of the restoration transform.
pub const MODEL_ID: &str = contract::MODEL_ID;

/// Identity + capabilities, constructible without weights.
pub fn descriptor() -> TransformDescriptor {
    TransformDescriptor {
        id: MODEL_ID,
        family: FAMILY,
        backend: "mlx",
        capabilities: capabilities(true),
    }
}

/// The restorer's request surface (shared with the Candle twin but for `mac_only`).
pub(crate) fn capabilities(mac_only: bool) -> TransformCapabilities {
    TransformCapabilities {
        // `TargetSize::Scale` — any positive finite factor; the output is `floor(input · scale)`.
        scale: true,
        min_edge: false,
        resolution: false,
        // Upstream has no ceiling: compute grows with the output area (the planner refuses only a
        // zero-pixel or overflowing output).
        max_scale: f32::MAX,
        // One deterministic forward per tile: no seed, no strength.
        is_diffusion: false,
        supports_strength: false,
        mac_only,
        default_scale: DEFAULT_SCALE as f32,
        input_budget: Some(INPUT_BUDGET),
        supports_color_fix: true,
    }
}

/// Labels of the restoration task's own iOS execution quanta (the backbone's are
/// [`crate::dit`]'s).
pub(crate) mod quanta {
    pub const LOAD: &str = "Iris restoration load";
    pub const TILE: &str = "Iris restoration tile";
    /// The driver's host planes, checked before the first tile.
    pub const HOST_PLANES: &str = "Iris restoration host planes";
    /// The post-tile phase's new host planes, checked when the last tile returns.
    pub const POST_TILE: &str = "Iris restoration post-tile planes";
}

/// A loaded Iris-3B restorer.
pub struct IrisRestorer {
    descriptor: TransformDescriptor,
    config: IrisConfig,
    settings: RestorationSettings,
    dit: IrisDiT,
    /// `[1, T, L, D]` f32 empty-prompt states.
    states: Array,
    mask: Vec<Vec<i32>>,
    /// The loaded memory contract ([`crate::memory_strategy::export_contract`] over the export on
    /// disk) — the one [`crate::memory_strategy::RESTORE_MEMORY_REGISTRATION`] builds for this
    /// load.
    memory_strategy: gen_core::MemoryProviderContract,
    /// The precision this restorer computes in — the numeric tier its safety check binds to.
    precision: mlx_gen::Precision,
    /// The packed tier of the loaded export (`None`: the dense export).
    tier: Option<mlx_gen::Quant>,
}

/// Refuse every load knob the restoration route does not implement, by name.
fn refuse_unsupported_spec(spec: &LoadSpec) -> Result<()> {
    crate::memory_strategy::refuse_deferred_load_shape(MODEL_ID, spec)?;
    crate::memory_strategy::refuse_unloadable_quant(MODEL_ID, spec)?;
    let refusals: [(&str, bool); 7] = [
        ("adapters", !spec.adapters.is_empty()),
        ("text_encoder", spec.text_encoder.is_some()),
        ("control", spec.control.is_some()),
        ("extra_controls", !spec.extra_controls.is_empty()),
        ("ip_adapter", spec.ip_adapter.is_some()),
        ("pid", spec.pid.is_some()),
        ("identity", spec.identity.is_some()),
    ];
    for (field, set) in refusals {
        if set {
            return Err(Error::Unsupported(format!(
                "{MODEL_ID}: LoadSpec::{field} is not supported by the Iris-3B restoration route"
            )));
        }
    }
    Ok(())
}

impl IrisRestorer {
    /// Load the restoration export (`spec.weights` = the `upscaler/` directory). The release
    /// computes under bf16 autocast (`Precision::Bf16`, default); `Precision::Fp32` is upstream's
    /// CPU path. `OffloadPolicy` is advisory and this route has a single heavy component, so it is
    /// always resident.
    pub fn load(spec: &LoadSpec) -> Result<Self> {
        refuse_unsupported_spec(spec)?;
        let resources = RestorationResources::from_spec(spec, MODEL_ID)?;
        let weights = resources.export.weights_path();
        crate::packed::refuse_dense_requested_as_packed(MODEL_ID, &weights, spec.quantize)?;
        let tier = crate::packed::artifact_tier(&weights)?;
        mlx_gen::ios_quantum::before_load(
            quanta::LOAD,
            &[
                resources.export.weights_path(),
                resources.export.empty_prompt_path(),
            ],
        )?;
        let prompt = resources.empty_prompt()?;
        let dit = load_backbone(
            &resources.export.dir,
            &resources.export.config,
            compute_dtype(spec),
        )?;
        let shape: Vec<i32> = prompt.shape.iter().map(|&d| d as i32).collect();
        let states = Array::from_slice(&prompt.embeddings, &shape);
        eval([&states])?;
        let memory_strategy = crate::memory_strategy::export_contract(
            gen_core::iris::IrisTask::Restoration,
            spec.precision,
            tier,
            &resources.export,
        )?;
        Ok(Self {
            memory_strategy,
            precision: spec.precision,
            tier,
            descriptor: descriptor(),
            config: resources.export.config,
            settings: resources.settings,
            dit,
            states,
            mask: vec![prompt.mask],
        })
    }

    /// The loaded memory-strategy contract of this restorer (sc-25690): the resident baseline —
    /// which is already one backbone forward per export tile — priced from the export's tensor
    /// headers. [`Transform`] has no memory seam, so this is its `memory_strategy_contract`.
    pub fn memory_strategy_contract(&self) -> &gen_core::MemoryProviderContract {
        &self.memory_strategy
    }

    /// The registered restoration safety check against this restorer's own load: the shared
    /// handshake bound to the precision it computes in, then the restoration route gate. Admit a
    /// request at [`crate::memory_strategy::restoration_memory_geometry`] of its plan.
    pub fn memory_strategy_safety_check(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::MemorySafetyDecision {
        crate::memory_strategy::loaded_safety_check(
            self.precision,
            self.tier,
            &self.memory_strategy,
            context,
        )
    }

    /// The export's `task` settings (`sigma`, `tile`) the restorer runs with.
    pub fn settings(&self) -> &RestorationSettings {
        &self.settings
    }

    /// The export's tile geometry (the release: 1024-px tiles, 16-px patches).
    pub fn geometry(&self) -> TileGeometry {
        self.settings.geometry(&self.config)
    }

    /// Plan a request against the loaded export.
    pub fn plan(&self, req: &TransformRequest) -> Result<RestorationPlan> {
        Ok(plan_request(req, self.geometry(), MODEL_ID)?)
    }

    /// One tile's velocity: `[3, h, w]` CHW f32 in `[-1, 1]` → `v` (same layout, f32), the backbone
    /// run once at `sigma · num_train_timesteps` on the empty-prompt states.
    pub fn velocity(&self, tile: &[f32], height: usize, width: usize) -> Result<Vec<f32>> {
        mlx_gen::ios_quantum::before(quanta::TILE, tile_scratch(tile.len() as u64))?;
        let v = self.velocity_body(tile, height, width)?;
        eval([&v])?;
        mlx_gen::ios_quantum::finish(quanta::TILE, &[&v])?;
        let n = v.shape().iter().product::<i32>();
        Ok(v.reshape(&[n])?.as_slice::<f32>().to_vec())
    }

    /// The body of one tile quantum: the tile's upload and the backbone forward (whose own blocks
    /// are checked inside it) to the f32 velocity.
    pub(crate) fn velocity_body(&self, tile: &[f32], height: usize, width: usize) -> Result<Array> {
        let x = Array::from_slice(tile, &[1, 3, height as i32, width as i32]);
        let t = Array::from_slice(&[self.settings.model_time(&self.config)], &[1]);
        let text = TextBatch {
            states: &self.states,
            mask: &self.mask,
        };
        Ok(self.dit.forward(&x, &t, &text)?.as_dtype(Dtype::Float32)?)
    }

    fn apply_impl(
        &self,
        req: &TransformRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        let plan = self.plan(req)?;
        let _quantum_scope = mlx_gen::ios_quantum::Scope::new();
        // The driver's host planes (f32 canvases on the CPU heap, which count toward the iOS
        // footprint like any MLX buffer): the whole route's peak is checked before anything is
        // built, and the post-tile phase's new planes again — against live headroom — once the
        // last tile has returned, before the driver allocates them.
        let host = host_plane_bytes(&plan);
        mlx_gen::ios_quantum::before(quanta::HOST_PLANES, host.peak())?;
        let tiles = plan.forward_count();
        let mut done = 0usize;
        let image = restore(
            &req.image,
            &plan,
            self.settings.sigma,
            &req.cancel,
            on_progress,
            &mut |tile, h, w| {
                let v = self.velocity(tile, h, w)?;
                done += 1;
                if done == tiles {
                    mlx_gen::ios_quantum::before(quanta::POST_TILE, host.post_tile_new())?;
                }
                Ok(v)
            },
        );
        mlx_rs::memory::clear_cache();
        Ok(image?)
    }
}

/// The tile quantum over a `[3, h, w]` tile of `elements` values: its f32 upload and the f32
/// velocity the forward returns (the forward's blocks are checked inside it), plus the host
/// readback `Vec<f32>` made right after the quantum synchronizes.
pub(crate) fn tile_scratch(elements: u64) -> u64 {
    mlx_gen::ios_quantum::tensor_bytes(4, &[3, elements])
}

/// The restoration driver's host planes ([`gen_core::iris::restoration`]'s `run_restoration`),
/// counted from its allocations. Every plane is f32 (4 B) over `O` output, `R` processing and `P`
/// padded pixels (three channels: `3·O` …); a tile is `t²` pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HostPlanes {
    /// The tile phase. Tiled: the bicubic reference (3·O), the padded input (3·P), the fused
    /// canvas (3·P), its weight plane (P), the Gaussian window (t²) and, per tile, the crop, the
    /// velocity readback and the restored crop (3·t² each). Single pass: the reference, the padded
    /// input, its velocity readback and the restored canvas built from it (3·P each).
    pub tile_phase: u64,
    /// What is still allocated when the last tile's velocity returns — the reference, the padded
    /// input and either the fused canvas and its weight plane (tiled) or the readback (single pass).
    pub live_at_last_tile: u64,
    /// The post-tile peak, the largest of: the single pass's restored canvas beside its readback
    /// (3·O + 9·P); the `(y + 1) / 2` crop (3·R) beside the reference and the fused canvas;
    /// when enlarged, the resize's working copy and first-pass output (6·R) beside the reference
    /// and the crop; under the colour fix, its three concurrent channel workers' planes (the kept
    /// high plane, the second split's high / low / blur: 4·O each, 12·O) beside the reference and
    /// the content (3·O each); and the final RGB8 image (3·O bytes) beside the restored planes.
    pub post_tile_peak: u64,
}

impl HostPlanes {
    /// The host peak of the whole route (checked before the first tile).
    pub fn peak(&self) -> u64 {
        self.tile_phase.max(self.post_tile_peak)
    }

    /// What the post-tile phase allocates beyond what is live when the last tile returns
    /// (checked then, against live headroom).
    pub fn post_tile_new(&self) -> u64 {
        self.post_tile_peak.saturating_sub(self.live_at_last_tile)
    }
}

pub(crate) fn host_plane_bytes(plan: &RestorationPlan) -> HostPlanes {
    let px = |d: gen_core::iris::restoration::Dims| d.width as u64 * d.height as u64;
    let (o, r, p) = (px(plan.output), px(plan.processing), px(plan.padded));
    let t2 = plan.tile as u64 * plan.tile as u64;
    let f = 4u64;
    let (tile_phase, live_at_last_tile, single) = if plan.single_pass() {
        let single = f * (3 * o + 9 * p);
        (single, f * (3 * o + 6 * p), single)
    } else {
        (f * (3 * o + 7 * p + t2 + 9 * t2), f * (3 * o + 7 * p), 0)
    };
    let crop = f * (3 * o + 3 * p + 3 * r);
    let resize = if plan.enlarged {
        f * (3 * o + 3 * r + 6 * r)
    } else {
        0
    };
    let colour_fix = if plan.color_fix {
        f * (3 * o + 3 * o + 12 * o)
    } else {
        0
    };
    let rgb8 = f * 3 * o + 3 * o;
    HostPlanes {
        tile_phase,
        live_at_last_tile,
        post_tile_peak: single.max(crop).max(resize).max(colour_fix).max(rgb8),
    }
}

impl Transform for IrisRestorer {
    fn descriptor(&self) -> &TransformDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TransformRequest) -> gen_core::Result<()> {
        plan_request(req, self.geometry(), MODEL_ID).map(|_| ())
    }

    fn apply(
        &self,
        req: &TransformRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<Image> {
        self.apply_impl(req, on_progress).map_err(Into::into)
    }
}

/// Registry loader.
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Transform>> {
    Ok(Box::new(IrisRestorer::load(spec)?))
}

pub(crate) const REGISTRATION: TransformRegistration = TransformRegistration { descriptor, load };

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::WeightsSource;

    #[test]
    fn descriptor_declares_the_release_surface() {
        let d = descriptor();
        assert_eq!(d.id, "iris_3b_restore");
        assert_eq!(d.family, "iris");
        let c = &d.capabilities;
        assert!(c.scale && !c.min_edge && !c.resolution);
        assert_eq!(c.default_scale, 4.0);
        assert!(c.supports_color_fix && !c.is_diffusion && !c.supports_strength);
        assert_eq!(c.input_budget, Some(INPUT_BUDGET));
    }

    /// The host planes are counted from the driver's allocations, per plan shape.
    ///
    /// *Mutation that reds this:* dropping any term of `host_plane_bytes` (e.g. the colour fix's
    /// worker planes, the padded input, or the single pass's restored canvas).
    #[test]
    fn host_planes_are_counted_from_the_drivers_allocations() {
        use gen_core::iris::restoration::{plan, Dims, RestorationOptions};
        use mlx_gen::InputSizing;
        let options = |scale: f64, color_fix: bool| RestorationOptions {
            scale,
            input_sizing: InputSizing::Original,
            color_fix,
        };
        // 4x of 512² → 2048² output, tiled (9 release tiles), not enlarged, colour fix on.
        let tiled = plan(
            Dims::new(512, 512),
            &options(4.0, true),
            TileGeometry::RELEASE,
            MODEL_ID,
        )
        .unwrap();
        assert!(!tiled.single_pass() && !tiled.enlarged);
        let (o, p, t2) = (2048u64 * 2048, 2048u64 * 2048, 1024u64 * 1024);
        let h = host_plane_bytes(&tiled);
        assert_eq!(h.tile_phase, 4 * (3 * o + 7 * p + 10 * t2));
        assert_eq!(h.live_at_last_tile, 4 * (3 * o + 7 * p));
        // The colour fix's three workers (4 planes each) beside the reference and the content.
        assert_eq!(h.post_tile_peak, 4 * (3 * o + 3 * o + 12 * o));
        assert_eq!(h.peak(), h.post_tile_peak.max(h.tile_phase));
        assert_eq!(
            h.post_tile_new(),
            h.post_tile_peak.saturating_sub(h.live_at_last_tile)
        );
        // Without the colour fix the crop (beside the reference and the fused canvas) is the
        // post-tile peak.
        let plain = host_plane_bytes(
            &plan(
                Dims::new(512, 512),
                &options(4.0, false),
                TileGeometry::RELEASE,
                MODEL_ID,
            )
            .unwrap(),
        );
        assert_eq!(plain.post_tile_peak, 4 * (3 * o + 3 * p + 3 * o));
        // 1x of 300×200: enlarged to a one-tile short side, padded, single pass.
        let small = plan(
            Dims::new(300, 200),
            &options(1.0, false),
            TileGeometry::RELEASE,
            MODEL_ID,
        )
        .unwrap();
        assert!(small.enlarged);
        let px = |d: Dims| d.width as u64 * d.height as u64;
        let (o, r, p) = (px(small.output), px(small.processing), px(small.padded));
        let h = host_plane_bytes(&small);
        if small.single_pass() {
            assert_eq!(h.tile_phase, 4 * (3 * o + 9 * p));
        }
        // The resize's working copy and first pass (6·R) beside the reference and the crop.
        assert!(h.post_tile_peak >= 4 * (3 * o + 3 * r + 6 * r));
    }

    /// A tier this task cannot load is refused by name before any resource is read (NVFP4; a
    /// packed tier under the f32 reference compute), and the dense export asked for at a packed
    /// tier is refused rather than requantized at load.
    ///
    /// *Mutation that reds this:* dropping `refuse_unloadable_quant` or
    /// `refuse_dense_requested_as_packed` from the load.
    #[test]
    fn unsupported_spec_fields_are_typed_refusals() {
        let refused = |spec: &LoadSpec, field: &str| match IrisRestorer::load(spec) {
            Err(Error::Unsupported(m)) => assert!(m.contains(field), "{m}"),
            other => panic!("expected a typed refusal, got {:?}", other.err()),
        };
        let mut spec = LoadSpec::new(WeightsSource::Dir("/iris-not-on-disk".into()));
        spec.quantize = Some(mlx_gen::Quant::Nvfp4);
        refused(&spec, "quantize");
        spec.quantize = Some(mlx_gen::Quant::Q4);
        spec.precision = mlx_gen::Precision::Fp32;
        refused(&spec, "precision");
        let mut dense = LoadSpec::new(WeightsSource::Dir(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/tiny-snapshot/upscaler"),
        ));
        dense.quantize = Some(mlx_gen::Quant::Q4);
        refused(&dense, "quantize");
    }
}
