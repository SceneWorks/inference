//! Iris-3B **monocular depth** on MLX (`iris3b/downstream/depth.py` @ [`crate::UPSTREAM_CODE_REVISION`],
//! story sc-25682) — [`DEPTH_MODEL_ID`] (`iris_3b_depth`).
//!
//! `IrisDepth` is the backbone with its input projections widened by one channel, a zero channel
//! concatenated after RGB, one forward at model time `flow.num_train_timesteps` conditioned on the
//! export's shipped empty-prompt states, and a 1×1 convolution reducing the 3-channel output to one.
//! **No text encoder is loaded** (E4): the task's resource closure is its `depth/` export only
//! ([`TaskExport`]), and a text encoder, the generation checkpoint or the restoration export is a
//! typed refusal.
//!
//! The source preprocessing, the resize back and the result/metadata types are the backend-neutral
//! [`gen_core::iris::depth`]; this module is the tensor forward.

use gen_core::iris::depth::{
    estimate_with, DepthMetadata, DepthOutput, DepthRequest, IrisDepthEstimator,
    DEPTH_EXTRA_INPUT_CHANNELS,
};
use gen_core::iris::downstream::{EmptyPrompt, TaskExport};
use gen_core::iris::IrisTask;
use mlx_gen::weights::Weights;
use mlx_gen::{gen_core, Error, LoadSpec, Progress, Result};
use mlx_rs::ops::{addmm, concatenate_axis, zeros};
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};

use crate::dit::{ForwardHooks, IrisDiT, TextBatch};
use crate::model::compute_dtype;

/// Model id of the depth task.
pub const DEPTH_MODEL_ID: &str = gen_core::iris::DEPTH_MODEL_ID;

/// The export's key prefix of the widened backbone (`IrisDepth.pixel`).
const BACKBONE_PREFIX: &str = "pixel.";
/// `IrisDepth.depth_reducer` (`nn.Conv2d(in_channels, 1, 1)`).
const REDUCER_WEIGHT: &str = "depth_reducer.weight";
const REDUCER_BIAS: &str = "depth_reducer.bias";

/// Labels of the depth task's own iOS execution quanta (the backbone's are [`crate::dit`]'s).
pub(crate) mod quanta {
    pub const LOAD: &str = "Iris depth load";
    pub const INPUT: &str = "Iris depth input";
    pub const REDUCER: &str = "Iris depth reducer";
    pub const HOST_PREP: &str = "Iris depth host preprocessing";
    pub const RGB: &str = "Iris depth RGB input copy";
    pub const HOST_MAP: &str = "Iris depth host map copy";
    pub const HOST_POST: &str = "Iris depth source resize";
    /// Every label above, for the phase attribution's coverage test.
    pub const ALL: [&str; 7] = [LOAD, INPUT, REDUCER, HOST_PREP, RGB, HOST_MAP, HOST_POST];
}

/// New internal whole-estimate payloads, not caller RGB ownership or phone evidence. A source
/// bounder must account for caller storage according to the ACTUAL observation coordinate; a
/// BeforeRequestConstruction process capture is not a BeforeModelLoad native measurement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DepthHostBytes {
    pub preprocessing_peak: u64,
    pub input_live: u64,
    pub rgb_copy: u64,
    pub map_copy: u64,
    pub postprocess_new: u64,
}

pub fn depth_host_bytes(plan: &gen_core::iris::depth::DepthInputPlan) -> Result<DepthHostBytes> {
    let overflow = || Error::Msg("Iris depth host-payload accounting overflow".into());
    let model = u64::from(plan.model_width)
        .checked_mul(u64::from(plan.model_height))
        .ok_or_else(overflow)?;
    let source = u64::from(plan.source_width)
        .checked_mul(u64::from(plan.source_height))
        .ok_or_else(overflow)?;
    let input = model
        .checked_mul(3 * std::mem::size_of::<f32>() as u64)
        .ok_or_else(overflow)?;
    let map_copy = model
        .checked_mul(std::mem::size_of::<f32>() as u64)
        .ok_or_else(overflow)?;
    let mut preprocessing_peak = input.checked_mul(2).ok_or_else(overflow)?;
    let postprocess_new = if plan.resized() {
        preprocessing_peak = preprocessing_peak.max(
            gen_core::imageops::resize_lanczos_u8_host_bytes(
                plan.source_height,
                plan.source_width,
                plan.model_height,
                plan.model_width,
            )
            .map_err(Error::from)?,
        );
        let axes = (u64::from(plan.source_width) + u64::from(plan.source_height))
            .checked_mul(std::mem::size_of::<(usize, usize, f32, f32)>() as u64)
            .ok_or_else(overflow)?;
        source
            .checked_mul(std::mem::size_of::<f32>() as u64)
            .and_then(|pixels| pixels.checked_add(axes))
            .ok_or_else(overflow)?
    } else {
        0
    };
    Ok(DepthHostBytes {
        preprocessing_peak,
        input_live: input,
        rgb_copy: crate::retention::allocated(input),
        map_copy,
        postprocess_new,
    })
}

/// A loaded Iris-3B depth model.
pub struct IrisDepth {
    export: TaskExport,
    dit: IrisDiT,
    /// `[1, C]` (the 1×1 conv kernel) and `[1]`, in the compute dtype.
    reducer_weight: Array,
    reducer_bias: Array,
    /// `[1, T, L, D]` f32 empty-prompt states and their `[T]` mask.
    embeddings: Array,
    mask: Vec<i32>,
    /// The loaded memory contract ([`crate::memory_strategy::export_contract`] over the export on
    /// disk) — the one [`crate::memory_strategy::DEPTH_MEMORY_REGISTRATION`] builds for this load.
    memory_strategy: gen_core::MemoryProviderContract,
    /// The precision this model computes in — the numeric tier its safety check binds to.
    precision: mlx_gen::Precision,
    /// The packed tier of the loaded export (`None`: the dense export).
    tier: Option<mlx_gen::Quant>,
}

fn refuse_unsupported_spec(spec: &LoadSpec) -> Result<()> {
    crate::memory_strategy::refuse_unloadable_quant(DEPTH_MODEL_ID, spec)?;
    let refusals: [(&str, bool); 6] = [
        ("adapters", !spec.adapters.is_empty()),
        ("control", spec.control.is_some()),
        ("extra_controls", !spec.extra_controls.is_empty()),
        ("ip_adapter", spec.ip_adapter.is_some()),
        ("pid", spec.pid.is_some()),
        ("identity", spec.identity.is_some()),
    ];
    for (field, set) in refusals {
        if set {
            return Err(Error::Unsupported(format!(
                "{DEPTH_MODEL_ID}: LoadSpec::{field} is not supported by the Iris-3B depth task"
            )));
        }
    }
    Ok(())
}

/// Load the depth task: `spec.weights` is the `depth/` export directory (`config.yaml` with
/// `task.name: depth`, `model.safetensors`, `empty_prompt.safetensors`); nothing else may be staged.
/// `spec.precision` selects upstream's CUDA bf16 autocast (default) or its FP32 CPU path. Every
/// parameter is evaluated before this returns.
pub fn load_depth(spec: &LoadSpec) -> Result<IrisDepth> {
    refuse_unsupported_spec(spec)?;
    let export = TaskExport::from_spec(spec, IrisTask::Depth, DEPTH_MODEL_ID)?;
    crate::packed::refuse_dense_requested_as_packed(
        DEPTH_MODEL_ID,
        &export.weights_path(),
        spec.quantize,
    )?;
    if spec.load_shape == gen_core::LoadShape::DeferredMaterialization {
        return load_depth_export_streamed(export, compute_dtype(spec));
    }
    load_depth_export(export, compute_dtype(spec))
}

/// [`load_depth`] from an already-resolved export at an explicit compute dtype.
pub fn load_depth_export(export: TaskExport, compute: Dtype) -> Result<IrisDepth> {
    load_depth_export_as(export, compute, false)
}

/// [`load_depth_export`] as a deferred-materialization load (rung 4): the backbone's block stacks
/// are read per window of [`crate::stream::TRANSFORMER_WINDOW_SIZE`] blocks at forward time
/// (`BlockStream`); everything else is read now. The depth task has no request
/// channel for a selection, so the load shape is its rung-4 engagement.
pub fn load_depth_export_streamed(export: TaskExport, compute: Dtype) -> Result<IrisDepth> {
    load_depth_export_as(export, compute, true)
}

/// Whether a depth export key (prefix included) belongs to one of the backbone's block stacks.
fn is_stack_key(key: &str) -> bool {
    key.strip_prefix(BACKBONE_PREFIX)
        .is_some_and(|name| crate::dit::Stack::of(name).is_some())
}

fn load_depth_export_as(export: TaskExport, compute: Dtype, streamed: bool) -> Result<IrisDepth> {
    // Fresh-buffer accounting requires cache-zero BEFORE materialization, not only later
    // generation. Keep this existing enabled-only policy alive through all load evaluations.
    let _load_quantum_scope = mlx_gen::ios_quantum::Scope::try_new()?;

    let model = &export.config.model;
    let path = export.weights_path();
    // A packed export (`crate::packed`) is read packed, under the bf16 compute its scales are
    // stored at; the dense export loads as before.
    let artifact = crate::packed::artifact_tier(&path)?;
    if artifact.is_some() && compute != Dtype::Bfloat16 {
        return Err(Error::Unsupported(format!(
            "{DEPTH_MODEL_ID}: {} is a packed {artifact:?} export, which runs under the bf16 \
             compute policy",
            path.display()
        )));
    }
    crate::stream::before_materialized_load(
        quanta::LOAD,
        &[path.clone(), export.empty_prompt_path()],
        |name| !streamed || !is_stack_key(name),
    )?;
    let mut source = Weights::from_file(&path)
        .map_err(|e| Error::Msg(format!("iris: loading {}: {e}", path.display())))?;
    let mut keys: Vec<String> = source.keys().map(str::to_owned).collect();
    keys.sort();
    // Split the export into the backbone (`pixel.*`, prefix stripped) and the reducer, casting each
    // backbone matrix to the compute dtype one tensor at a time (the generation loader's policy).
    let mut backbone = Weights::empty();
    let (mut reducer_weight, mut reducer_bias) = (None, None);
    for key in keys {
        let Some(array) = source.remove(&key) else {
            continue;
        };
        // A streamed stack tensor's lazy handle is dropped unread: its window reads it.
        if streamed && is_stack_key(&key) {
            continue;
        }
        if let Some(name) = key.strip_prefix(BACKBONE_PREFIX) {
            let array = crate::model::read_backbone_tensor(name, array, compute)?;
            backbone.insert(name, array);
        } else if key == REDUCER_WEIGHT {
            mlx_rs::transforms::eval_pending_loads([&array])?;
            reducer_weight = Some(array);
        } else if key == REDUCER_BIAS {
            mlx_rs::transforms::eval_pending_loads([&array])?;
            reducer_bias = Some(array);
        } else {
            return Err(Error::Msg(format!(
                "iris: the depth export {} carries `{key}`, which is neither a `{BACKBONE_PREFIX}*` \
                 backbone key nor the depth reducer",
                path.display()
            )));
        }
    }
    let missing = |key: &str| Error::Msg(format!("iris: the depth export is missing `{key}`"));
    let reducer_weight = reducer_weight.ok_or_else(|| missing(REDUCER_WEIGHT))?;
    let reducer_bias = reducer_bias.ok_or_else(|| missing(REDUCER_BIAS))?;
    crate::nn::expect_shape(
        REDUCER_WEIGHT,
        &reducer_weight,
        &[1, model.in_channels as i32, 1, 1],
    )?;
    crate::nn::expect_shape(REDUCER_BIAS, &reducer_bias, &[1])?;
    let reducer_weight = reducer_weight
        .reshape(&[1, model.in_channels as i32])?
        .as_dtype(compute)?;
    let reducer_bias = reducer_bias.as_dtype(compute)?;

    let input_channels = model.in_channels + DEPTH_EXTRA_INPUT_CHANNELS;
    let dit = if streamed {
        let stream = crate::dit::BlockStream::open(
            &path,
            BACKBONE_PREFIX,
            model,
            compute,
            Some(crate::stream::TRANSFORMER_WINDOW_SIZE as usize),
        )?;
        IrisDiT::from_weights_streamed(&backbone, model, compute, input_channels, stream)?
    } else {
        IrisDiT::from_weights_widened(&backbone, model, compute, input_channels)?
    };
    drop(backbone);

    let prompt: EmptyPrompt = export.empty_prompt()?;
    let shape: Vec<i32> = prompt.shape.iter().map(|&d| d as i32).collect();
    let embeddings = Array::from_slice(&prompt.embeddings, &shape);

    let mut arrays = dit.arrays();
    arrays.extend([&reducer_weight, &reducer_bias, &embeddings]);
    for group in arrays.chunks(64) {
        eval(group.iter().copied())?;
    }
    let precision = match compute {
        Dtype::Float32 => mlx_gen::Precision::Fp32,
        _ => mlx_gen::Precision::Bf16,
    };
    let load_shape = if streamed {
        gen_core::LoadShape::DeferredMaterialization
    } else {
        gen_core::LoadShape::EagerMaterialization
    };
    let memory_strategy = crate::memory_strategy::export_contract(
        IrisTask::Depth,
        precision,
        artifact,
        &export,
        load_shape,
    )?;
    Ok(IrisDepth {
        memory_strategy,
        precision,
        tier: artifact,
        dit,
        reducer_weight,
        reducer_bias,
        embeddings,
        mask: prompt.mask,
        export,
    })
}

/// The pieces of one `IrisDepth.forward` — the loaded estimator's, or a training step's (built from
/// the traced trainable tensors, `crate::train::depth`), so training and inference share one graph.
pub struct DepthParts<'a> {
    /// The widened backbone.
    pub dit: &'a IrisDiT,
    /// `[1, C]` (the 1×1 conv kernel) and `[1]`, in the compute dtype.
    pub reducer_weight: &'a Array,
    pub reducer_bias: &'a Array,
    /// `[1, T, L, D]` f32 empty-prompt states and their `[T]` mask.
    pub embeddings: &'a Array,
    pub mask: &'a [i32],
    pub num_train_timesteps: usize,
}

/// `IrisDepth.forward` over `parts`: RGB ‖ zero channel, one backbone forward at model time
/// `num_train_timesteps` on the empty prompt, the 1×1 reducer. `[B, 3, H, W]` → `[B, 1, H, W]` f32.
pub fn depth_forward(parts: &DepthParts, rgb: &Array) -> Result<Array> {
    depth_forward_with(parts, rgb, None)
}

/// [`depth_forward`] with `hooks` around every trunk and pixel block of the backbone forward — the
/// depth trainer's seam for activation-checkpointing regions (sc-25690). With [`crate::dit::NoHooks`] it is
/// exactly the estimator's forward.
pub fn depth_forward_hooked(
    parts: &DepthParts,
    rgb: &Array,
    hooks: &mut dyn ForwardHooks,
) -> Result<Array> {
    depth_forward_with(parts, rgb, Some(hooks))
}

fn depth_forward_with(
    parts: &DepthParts,
    rgb: &Array,
    hooks: Option<&mut dyn ForwardHooks>,
) -> Result<Array> {
    let sh = rgb.shape();
    let (b, h, w) = (sh[0], sh[2], sh[3]);
    mlx_gen::ios_quantum::before(quanta::INPUT, depth_input_scratch(b, h, w))?;
    let (x, t, states) = depth_input_body(parts, rgb)?;
    mlx_gen::ios_quantum::finish(quanta::INPUT, &[&x, &states])?;
    let mask = vec![parts.mask.to_vec(); b as usize];
    let text = TextBatch {
        states: &states,
        mask: &mask,
    };
    let out = match hooks {
        Some(hooks) => parts.dit.forward_hooked(&x, &t, &text, hooks)?,
        None => parts.dit.forward(&x, &t, &text)?,
    };
    mlx_gen::ios_quantum::run(
        quanta::REDUCER,
        depth_reducer_scratch(&out, parts.dit.compute_dtype()),
        || depth_reducer_body(parts, &out),
    )
}

/// The depth input quantum over a `[B, 3, H, W]` input (each allocation page-rounded): the f32
/// RGB ‖ zero-channel
/// concatenation (`[B, 4, H, W]`), the zero channel itself and its f32 fill scalar, and the `[B]`
/// model times. The prompt states are broadcast to the batch as a view, so they allocate nothing.
pub(crate) fn depth_input_scratch(b: i32, h: i32, w: i32) -> u64 {
    use crate::retention::allocated;
    use mlx_gen::ios_quantum::tensor_bytes;
    let d = |v: i32| v.max(0) as u64;
    let channels = 3 + DEPTH_EXTRA_INPUT_CHANNELS as u64;
    allocated(tensor_bytes(4, &[d(b), channels, d(h), d(w)]))
        .saturating_add(allocated(tensor_bytes(
            4,
            &[d(b), DEPTH_EXTRA_INPUT_CHANNELS as u64, d(h), d(w)],
        )))
        .saturating_add(allocated(tensor_bytes(4, &[d(b)])))
        .saturating_add(4)
}

/// The body of the depth input quantum: RGB ‖ zero channel, the model time, and the broadcast
/// empty-prompt states.
pub(crate) fn depth_input_body(parts: &DepthParts, rgb: &Array) -> Result<(Array, Array, Array)> {
    let sh = rgb.shape();
    let (b, h, w) = (sh[0], sh[2], sh[3]);
    let e = parts.embeddings.shape();
    let x = concatenate_axis(
        &[
            &rgb.as_dtype(Dtype::Float32)?,
            &zeros::<f32>(&[b, DEPTH_EXTRA_INPUT_CHANNELS as i32, h, w])?,
        ],
        1,
    )?;
    let t_model = parts.num_train_timesteps as f32;
    let t = Array::from_slice(&vec![t_model; b as usize], &[b]);
    let states = mlx_rs::ops::broadcast_to(parts.embeddings, &[b, e[1], e[2], e[3]])?;
    Ok((x, t, states))
}

/// The depth reducer quantum over the backbone's `[B, C, H, W]` f32 output (each allocation
/// page-rounded): the `[B·H·W, C]` re-layout (f32), and the reduced `[B·H·W, 1]` map — under bf16 also the re-layout's
/// compute-width cast and the map's f32 copy.
pub(crate) fn depth_reducer_scratch(out: &Array, compute: Dtype) -> u64 {
    depth_reducer_bytes(
        out.size() as u64,
        out.shape().get(1).copied().unwrap_or(1).max(1) as u64,
        crate::nn::dtype_bytes(compute),
    )
}

/// [`depth_reducer_scratch`] over `n` output elements of `c` channels at compute width `cw`.
pub(crate) fn depth_reducer_bytes(n: u64, c: u64, cw: u64) -> u64 {
    use crate::retention::allocated;
    use mlx_gen::ios_quantum::tensor_bytes;
    let c = c.max(1);
    allocated(tensor_bytes(4, &[n]))
        .saturating_add(allocated(tensor_bytes(cw, &[n / c])))
        .saturating_add(if cw < 4 {
            allocated(tensor_bytes(cw, &[n])).saturating_add(allocated(tensor_bytes(4, &[n / c])))
        } else {
            0
        })
}

/// The body of the depth reducer quantum: a 1×1 conv over the channels — under upstream's
/// autocast a bf16 conv, so it runs in the compute dtype like every other projection.
pub(crate) fn depth_reducer_body(parts: &DepthParts, out: &Array) -> Result<Array> {
    let sh = out.shape();
    let (b, c, h, w) = (sh[0], sh[1], sh[2], sh[3]);
    let pixels = out
        .transpose_axes(&[0, 2, 3, 1])?
        .reshape(&[-1, c])?
        .as_dtype(parts.dit.compute_dtype())?;
    let reduced = addmm(
        parts.reducer_bias,
        &pixels,
        parts.reducer_weight.t(),
        1.0,
        1.0,
    )?;
    Ok(reduced.reshape(&[b, 1, h, w])?.as_dtype(Dtype::Float32)?)
}

impl IrisDepth {
    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn assert_endpoint_ownership(&self, streamed: bool) {
        self.dit.assert_endpoint_ownership(streamed);
        assert!(self.embeddings.nbytes() > 0, "depth owns no conditioning");
        assert!(
            self.reducer_weight.nbytes() > 0 && self.reducer_bias.nbytes() > 0,
            "depth owns no reducer parameters"
        );
        let _addresses = self.endpoint_parameter_addresses();
    }

    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn endpoint_parameter_addresses(&self) -> Vec<usize> {
        let mut addresses = self.dit.endpoint_parameter_addresses();
        addresses.extend(crate::anchor_tests::materialized_addresses([
            &self.embeddings,
            &self.reducer_weight,
            &self.reducer_bias,
        ]));
        addresses
    }

    pub fn export(&self) -> &TaskExport {
        &self.export
    }

    pub fn compute_dtype(&self) -> Dtype {
        self.dit.compute_dtype()
    }

    /// The loaded memory-strategy contract of this depth model (sc-25690): the resident baseline,
    /// priced from the export's tensor headers. The depth task is a provider-specific API rather
    /// than a registry [`mlx_gen::Generator`], so this is its `memory_strategy_contract`.
    pub fn memory_strategy_contract(&self) -> &gen_core::MemoryProviderContract {
        &self.memory_strategy
    }

    /// The registered depth safety check against this model's own load: the shared handshake
    /// bound to the precision it computes in, then the depth route gate. Admit a request at
    /// [`crate::memory_strategy::depth_memory_geometry`] of its input plan.
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

    /// The pieces of this model's forward.
    pub(crate) fn parts(&self) -> DepthParts<'_> {
        DepthParts {
            dit: &self.dit,
            reducer_weight: &self.reducer_weight,
            reducer_bias: &self.reducer_bias,
            embeddings: &self.embeddings,
            mask: &self.mask,
            num_train_timesteps: self.export.config.flow.num_train_timesteps,
        }
    }

    /// `IrisDepth.forward`: `[B, 3, H, W]` RGB in `[-1, 1]` (f32, H and W on the patch grid) →
    /// `[B, 1, H, W]` relative log depth (f32).
    pub fn forward(&self, rgb: &Array) -> Result<Array> {
        depth_forward(&self.parts(), rgb)
    }

    fn estimate_impl(
        &self,
        req: &DepthRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<DepthOutput> {
        let compute = match self.compute_dtype() {
            Dtype::Float32 => "float32",
            _ => "bfloat16",
        };
        let _quantum_scope =
            mlx_gen::ios_quantum::Scope::try_new().map_err(gen_core::Error::from)?;
        gen_core::iris::depth::validate_depth_request(req)?;
        let input_plan = gen_core::iris::depth::plan_depth_input(
            req.image.width,
            req.image.height,
            req.resolution,
            self.export.config.model.patch_size,
        )?;
        let host = depth_host_bytes(&input_plan).map_err(gen_core::Error::from)?;
        mlx_gen::ios_quantum::diagnostic_metadata(
            DEPTH_MODEL_ID,
            "depth_request",
            &[
                ("source_width", u64::from(req.image.width)),
                ("source_height", u64::from(req.image.height)),
                ("model_width", u64::from(input_plan.model_width)),
                ("model_height", u64::from(input_plan.model_height)),
                ("max_side", u64::from(req.resolution.max_side())),
                ("caller_rgb_bytes", req.image.pixels.len() as u64),
                (
                    "caller_rgb_capacity_bytes",
                    req.image.pixels.capacity() as u64,
                ),
                ("backbone_streamed", u64::from(self.dit.is_streamed())),
            ],
        );
        mlx_gen::ios_quantum::before(quanta::HOST_PREP, host.preprocessing_peak)
            .map_err(gen_core::Error::from)?;
        let output = estimate_with(
            req,
            self.export.config.model.patch_size,
            |plan| {
                DepthMetadata::new(
                    "mlx",
                    self.export.config_sha256.clone(),
                    compute,
                    req.resolution,
                    plan,
                )
            },
            on_progress,
            |input, plan| {
                mlx_gen::ios_quantum::finish(quanta::HOST_PREP, &[])
                    .map_err(gen_core::Error::from)?;
                let prediction = self
                    .predict(input, plan.model_width, plan.model_height)
                    .map_err(gen_core::Error::from)?;
                if plan.resized() {
                    mlx_gen::ios_quantum::before(quanta::HOST_POST, host.postprocess_new)
                        .map_err(gen_core::Error::from)?;
                }
                Ok(prediction)
            },
        )?;
        if input_plan.resized() {
            mlx_gen::ios_quantum::finish(quanta::HOST_POST, &[]).map_err(gen_core::Error::from)?;
        }
        Ok(output)
    }

    /// One `[3, h, w]` host input → the `h × w` host prediction.
    fn predict(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>> {
        let (h, w) = (height as i32, width as i32);
        let rgb_bytes = crate::retention::allocated(mlx_gen::ios_quantum::tensor_bytes(
            4,
            &[3, u64::from(height), u64::from(width)],
        ));
        mlx_gen::ios_quantum::before(quanta::RGB, rgb_bytes)?;
        let rgb = Array::from_slice(input, &[1, 3, h, w]);
        mlx_gen::ios_quantum::finish(quanta::RGB, &[&rgb])?;
        let depth = self.forward(&rgb)?;
        eval([&depth])?;
        mlx_gen::ios_quantum::before(
            quanta::HOST_MAP,
            mlx_gen::ios_quantum::tensor_bytes(4, &[u64::from(height), u64::from(width)]),
        )?;
        let values = depth.reshape(&[h * w])?.as_slice::<f32>().to_vec();
        mlx_gen::ios_quantum::finish(quanta::HOST_MAP, &[])?;
        drop(depth);
        mlx_rs::memory::clear_cache();
        Ok(values)
    }
}

impl IrisDepthEstimator for IrisDepth {
    fn backend(&self) -> &'static str {
        "mlx"
    }

    fn estimate(
        &self,
        req: &DepthRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<DepthOutput> {
        self.estimate_impl(req, on_progress)
    }
}

/// Load the depth task as a boxed backend-neutral estimator (see [`load_depth`]).
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn IrisDepthEstimator>> {
    Ok(Box::new(load_depth(spec)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::WeightsSource;

    /// A tier this task cannot load is refused by name before any resource is read (NVFP4; a
    /// packed tier under the f32 reference compute), and the dense export asked for at a packed
    /// tier is refused rather than requantized at load.
    ///
    /// *Mutation that reds this:* dropping `refuse_unloadable_quant` or
    /// `refuse_dense_requested_as_packed` from the load.
    #[test]
    fn unsupported_spec_fields_are_typed_refusals() {
        let refused = |spec: &LoadSpec, field: &str| match load_depth(spec) {
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
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-depth"),
        ));
        dense.quantize = Some(mlx_gen::Quant::Q4);
        refused(&dense, "quantize");
    }
}
