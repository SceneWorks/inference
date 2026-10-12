//! **Representation alignment** (REPA / iREPA) of the Iris generation trainer (story sc-25689):
//! the backend-neutral half of `iris3b/repa.py` @ [`super::super::UPSTREAM_CODE_REVISION`] — the
//! resolved plan, the frozen teacher's architecture and resource layout, and the exact resampling
//! operators (torch `F.interpolate` bicubic / bilinear, with and without antialias) as dense
//! per-axis matrices the MLX and Candle trainers apply with two matmuls.
//!
//! Upstream semantics kept exactly:
//! * the student tokens are the patch tokens **after** trunk block `repa_layer` (1-based);
//! * `repa`: a 3-layer SiLU MLP projector `D → proj_hidden_dim → proj_hidden_dim → teacher_dim`;
//!   `irepa`: a `3×3` conv (`padding=1`) on the student token grid, and the teacher tokens are
//!   spatially normalized (`(t − γ·mean_N) / √(var_N + 1e−6)`, biased variance);
//! * the teacher sees the clean image `(x + 1)/2`, bicubic-resized (no antialias) to its own grid
//!   `· patch`, ImageNet-normalized, and returns its **final-norm patch tokens**
//!   (`forward_features()["x_norm_patchtokens"]`), all in fp32 under `no_grad`;
//! * the projector runs in fp32 and the cosine is taken in fp32; on a grid mismatch the two token
//!   maps meet on whichever grid has **fewer** tokens (bilinear, `align_corners=False`), the student
//!   normalized before and after the resize;
//! * `loss = −mean(cos(student, teacher))`, `total = flow + weight · loss`; the projector is an
//!   AdamW-routed auxiliary parameter group, outside the EMA and the export.
//!
//! The teacher is the source's **DINOv2** family from `facebookresearch/dinov2` (torch-hub
//! entrypoints `dinov2_vit{s,b,l,g}14[_reg]`, the release's being `dinov2_vitb14`): a pinned training
//! resource provisioned by the caller (never downloaded here), read under either the hub's
//! `state_dict` key names or the Hugging Face `Dinov2Model` names (the same tensors).
//! Divergence (deliberate, typed): upstream silently disables the loss when the teacher fails to
//! load (`available = False`); natively a missing or mismatched teacher resource is a refusal.

use crate::{Error, Result};

/// `LoadSpec::components` key of the REPA teacher resource directory.
pub const TEACHER_COMPONENT: &str = "repa_teacher";
/// The teacher's weights file inside the resource directory.
pub const TEACHER_WEIGHTS_FILE: &str = "model.safetensors";
/// The source's default teacher (`RepaConfig.teacher`).
pub const DEFAULT_TEACHER: &str = "dinov2_vitb14";
/// The source's teacher hub repo (`RepaConfig.teacher_hub`).
pub const DEFAULT_TEACHER_HUB: &str = "facebookresearch/dinov2";
/// The pinned `facebookresearch/dinov2` commit the reference forward was checked against.
pub const DINOV2_CODE_REVISION: &str = "7764ea0f912e53c92e82eb78a2a1631e92725fc8";
/// The Hugging Face mirror of the release teacher's weights (the hub checkpoint's tensors under
/// `Dinov2Model` names) and its pinned revision — the resource SceneWorks provisions.
pub const DINOV2_BASE_HF_REPO: &str = "facebook/dinov2-base";
pub const DINOV2_BASE_HF_REVISION: &str = "f9e44c814b77203eaa57a6bdbbd535f21ede1415";
/// `repa.py` `_IMAGENET_MEAN` / `_IMAGENET_STD`.
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];
/// `_spatial_normalize`'s `eps`.
pub const SPATIAL_NORM_EPS: f64 = 1e-6;
/// DINOv2's LayerNorm `eps`.
pub const TEACHER_LN_EPS: f64 = 1e-6;

/// `repa.variant`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RepaVariant {
    /// 3-layer MLP projector on the raw teacher tokens.
    Repa,
    /// Conv3x3 projector on the student grid + spatially normalized teacher tokens.
    Irepa,
}

impl RepaVariant {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "repa" => Ok(Self::Repa),
            "irepa" => Ok(Self::Irepa),
            other => Err(Error::Msg(format!(
                "iris training: unknown repa variant '{other}' (repa | irepa)"
            ))),
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Repa => "repa",
            Self::Irepa => "irepa",
        }
    }
}

/// The teacher's feed-forward block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TeacherFfn {
    /// `fc1 → GELU(erf) → fc2`, hidden `4·D`.
    Mlp,
    /// `SwiGLUFFNFused`: `w12 → silu(x1)·x2 → w3`, hidden `(int(4D·2/3) + 7) // 8 · 8`.
    SwiGlu,
}

/// One DINOv2 hub entrypoint's architecture (`_make_dinov2_model` + `vit_*`).
#[derive(Clone, Debug, PartialEq)]
pub struct TeacherArch {
    pub entrypoint: String,
    pub embed_dim: usize,
    pub depth: usize,
    pub num_heads: usize,
    pub ffn: TeacherFfn,
    pub ffn_hidden: usize,
    pub patch_size: usize,
    /// `img_size = 518` → a `37 × 37` learned position grid (+ the class token).
    pub pos_grid: usize,
    pub num_register_tokens: usize,
    pub interpolate_antialias: bool,
    pub interpolate_offset: f64,
}

impl TeacherArch {
    /// The architecture of a `facebookresearch/dinov2` hub entrypoint.
    pub fn from_entrypoint(name: &str) -> Result<Self> {
        let (base, reg) = match name.strip_suffix("_reg") {
            Some(b) => (b, true),
            None => (name, false),
        };
        let (dim, depth, heads, ffn): (usize, usize, usize, TeacherFfn) = match base {
            "dinov2_vits14" => (384, 12, 6, TeacherFfn::Mlp),
            "dinov2_vitb14" => (768, 12, 12, TeacherFfn::Mlp),
            "dinov2_vitl14" => (1024, 24, 16, TeacherFfn::Mlp),
            "dinov2_vitg14" => (1536, 40, 24, TeacherFfn::SwiGlu),
            _ => {
                return Err(Error::Unsupported(format!(
                    "iris training: repa.teacher '{name}' is not a facebookresearch/dinov2 hub \
                     entrypoint the native teacher implements (dinov2_vit{{s,b,l,g}}14[_reg])"
                )))
            }
        };
        let ffn_hidden = match ffn {
            TeacherFfn::Mlp => 4 * dim,
            // `(int(hidden · 2 / 3) + 7) // 8 · 8` (the truncation comes first).
            TeacherFfn::SwiGlu => (4 * dim * 2 / 3).div_ceil(8) * 8,
        };
        Ok(Self {
            entrypoint: name.to_string(),
            embed_dim: dim,
            depth,
            num_heads: heads,
            ffn,
            ffn_hidden,
            patch_size: 14,
            pos_grid: 37,
            num_register_tokens: if reg { 4 } else { 0 },
            interpolate_antialias: reg,
            interpolate_offset: if reg { 0.0 } else { 0.1 },
        })
    }

    /// Every tensor of the teacher under the hub's `state_dict` names, with its shape — what a
    /// teacher resource must carry (either naming).
    pub fn tensor_shapes(&self) -> Vec<(String, Vec<usize>)> {
        let d = self.embed_dim;
        let p = self.patch_size;
        let mut out = vec![
            ("cls_token".to_string(), vec![1, 1, d]),
            (
                "pos_embed".to_string(),
                vec![1, self.pos_grid * self.pos_grid + 1, d],
            ),
            ("patch_embed.proj.weight".to_string(), vec![d, 3, p, p]),
            ("patch_embed.proj.bias".to_string(), vec![d]),
        ];
        if self.num_register_tokens > 0 {
            out.push((
                "register_tokens".into(),
                vec![1, self.num_register_tokens, d],
            ));
        }
        for i in 0..self.depth {
            let b = format!("blocks.{i}");
            out.push((format!("{b}.norm1.weight"), vec![d]));
            out.push((format!("{b}.norm1.bias"), vec![d]));
            out.push((format!("{b}.attn.qkv.weight"), vec![3 * d, d]));
            out.push((format!("{b}.attn.qkv.bias"), vec![3 * d]));
            out.push((format!("{b}.attn.proj.weight"), vec![d, d]));
            out.push((format!("{b}.attn.proj.bias"), vec![d]));
            out.push((format!("{b}.ls1.gamma"), vec![d]));
            out.push((format!("{b}.norm2.weight"), vec![d]));
            out.push((format!("{b}.norm2.bias"), vec![d]));
            match self.ffn {
                TeacherFfn::Mlp => {
                    out.push((format!("{b}.mlp.fc1.weight"), vec![self.ffn_hidden, d]));
                    out.push((format!("{b}.mlp.fc1.bias"), vec![self.ffn_hidden]));
                    out.push((format!("{b}.mlp.fc2.weight"), vec![d, self.ffn_hidden]));
                    out.push((format!("{b}.mlp.fc2.bias"), vec![d]));
                }
                TeacherFfn::SwiGlu => {
                    out.push((format!("{b}.mlp.w12.weight"), vec![2 * self.ffn_hidden, d]));
                    out.push((format!("{b}.mlp.w12.bias"), vec![2 * self.ffn_hidden]));
                    out.push((format!("{b}.mlp.w3.weight"), vec![d, self.ffn_hidden]));
                    out.push((format!("{b}.mlp.w3.bias"), vec![d]));
                }
            }
            out.push((format!("{b}.ls2.gamma"), vec![d]));
        }
        out.push(("norm.weight".into(), vec![d]));
        out.push(("norm.bias".into(), vec![d]));
        out
    }
}

/// How a teacher resource names its tensors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TeacherNaming {
    /// The torch-hub `state_dict` (`cls_token`, `blocks.N.attn.qkv`, …).
    Hub,
    /// Hugging Face `Dinov2Model` (`embeddings.cls_token`, `encoder.layer.N.attention.attention.query`, …).
    HuggingFace,
}

impl TeacherNaming {
    pub fn detect<'a>(mut keys: impl Iterator<Item = &'a str>) -> Result<Self> {
        let mut hub = false;
        let mut hf = false;
        for k in keys.by_ref() {
            hub |= k == "cls_token";
            hf |= k == "embeddings.cls_token";
        }
        match (hub, hf) {
            (true, false) => Ok(Self::Hub),
            (false, true) => Ok(Self::HuggingFace),
            _ => Err(Error::Msg(
                "iris training: the REPA teacher resource is neither a DINOv2 hub state_dict nor \
                 a Hugging Face Dinov2Model checkpoint"
                    .into(),
            )),
        }
    }
}

/// Where a hub tensor lives in a Hugging Face `Dinov2Model` checkpoint. A hub `attn.qkv` is the
/// row-concatenation of the three HF projections, returned in order.
pub fn hf_sources(hub_key: &str) -> Vec<String> {
    if let Some(rest) = hub_key.strip_prefix("blocks.") {
        let (i, leaf) = rest.split_once('.').unwrap_or((rest, ""));
        let l = format!("encoder.layer.{i}");
        return match leaf {
            "attn.qkv.weight" | "attn.qkv.bias" => {
                let s = leaf.rsplit('.').next().unwrap_or("weight");
                ["query", "key", "value"]
                    .iter()
                    .map(|p| format!("{l}.attention.attention.{p}.{s}"))
                    .collect()
            }
            "attn.proj.weight" => vec![format!("{l}.attention.output.dense.weight")],
            "attn.proj.bias" => vec![format!("{l}.attention.output.dense.bias")],
            "ls1.gamma" => vec![format!("{l}.layer_scale1.lambda1")],
            "ls2.gamma" => vec![format!("{l}.layer_scale2.lambda1")],
            "mlp.w12.weight" => vec![format!("{l}.mlp.weights_in.weight")],
            "mlp.w12.bias" => vec![format!("{l}.mlp.weights_in.bias")],
            "mlp.w3.weight" => vec![format!("{l}.mlp.weights_out.weight")],
            "mlp.w3.bias" => vec![format!("{l}.mlp.weights_out.bias")],
            other => vec![format!("{l}.{other}")],
        };
    }
    vec![match hub_key {
        "cls_token" => "embeddings.cls_token".into(),
        "pos_embed" => "embeddings.position_embeddings".into(),
        "register_tokens" => "embeddings.register_tokens".into(),
        "patch_embed.proj.weight" => "embeddings.patch_embeddings.projection.weight".into(),
        "patch_embed.proj.bias" => "embeddings.patch_embeddings.projection.bias".into(),
        "norm.weight" => "layernorm.weight".into(),
        "norm.bias" => "layernorm.bias".into(),
        other => other.to_string(),
    }]
}

/// The resolved REPA objective of a run.
#[derive(Clone, Debug, PartialEq)]
pub struct RepaPlan {
    pub weight: f64,
    pub variant: RepaVariant,
    pub spatial_norm_gamma: f64,
    pub teacher: TeacherArch,
    pub teacher_dim: usize,
    pub proj_hidden_dim: usize,
    pub teacher_image_size: usize,
    pub teacher_match_student: bool,
    /// 1-based trunk block whose output patch tokens are aligned.
    pub layer: usize,
    pub student_dim: usize,
    /// `repa.teacher_weights`: an explicit teacher resource path (a directory holding
    /// [`TEACHER_WEIGHTS_FILE`] or the safetensors file itself), used instead of the
    /// [`TEACHER_COMPONENT`] load component. Never a URL: nothing is downloaded.
    pub teacher_weights: Option<String>,
}

impl RepaPlan {
    /// Read the `repa` object of the Iris options bag (`RepaConfig` keys plus `layer`, the
    /// `model.repa_layer` override). `None` when the weight is 0 (upstream builds no teacher and no
    /// projector then). `default_layer` is the backbone config's `model.repa_layer`,
    /// `student_dim` / `depth` its `hidden_size` / trunk depth.
    pub fn parse(
        raw: Option<&serde_json::Value>,
        default_layer: usize,
        student_dim: usize,
        depth: usize,
    ) -> Result<Option<Self>> {
        use serde_json::Value as J;
        let empty = serde_json::Map::new();
        let obj = match raw {
            None | Some(J::Null) => &empty,
            Some(J::Object(m)) => m,
            Some(v) => {
                return Err(Error::Msg(format!(
                    "iris training: repa must be an object, got {v}"
                )))
            }
        };
        const KEYS: [&str; 13] = [
            "weight",
            "variant",
            "spatial_norm_gamma",
            "teacher_source",
            "teacher_hub",
            "teacher",
            "teacher_weights",
            "teacher_dim",
            "proj_hidden_dim",
            "teacher_image_size",
            "teacher_patch_size",
            "teacher_match_student",
            "layer",
        ];
        for k in obj.keys() {
            if !KEYS.contains(&k.as_str()) {
                return Err(Error::Unsupported(format!(
                    "iris training: repa.{k} is not a RepaConfig option (known: {})",
                    KEYS.join(", ")
                )));
            }
        }
        let get = |k: &str| obj.get(k).filter(|v| !v.is_null());
        let num = |k: &str, d: f64| -> Result<f64> {
            match get(k) {
                None => Ok(d),
                Some(v) => v.as_f64().filter(|x| x.is_finite()).ok_or_else(|| {
                    Error::Msg(format!("iris training: repa.{k} must be a number, got {v}"))
                }),
            }
        };
        let uint = |k: &str, d: usize| -> Result<usize> {
            match get(k) {
                None => Ok(d),
                Some(v) => v.as_u64().map(|x| x as usize).ok_or_else(|| {
                    Error::Msg(format!(
                        "iris training: repa.{k} must be a non-negative integer, got {v}"
                    ))
                }),
            }
        };
        let text = |k: &str, d: &str| -> Result<String> {
            match get(k) {
                None => Ok(d.to_string()),
                Some(J::String(s)) => Ok(s.clone()),
                Some(v) => Err(Error::Msg(format!(
                    "iris training: repa.{k} must be a string, got {v}"
                ))),
            }
        };
        let weight = num("weight", 0.0)?;
        if weight < 0.0 {
            return Err(Error::Msg("iris training: repa.weight must be >= 0".into()));
        }
        let variant = RepaVariant::parse(&text("variant", "repa")?)?;
        let gamma = num("spatial_norm_gamma", 1.0)?;
        let source = text("teacher_source", "hub")?;
        if !matches!(source.as_str(), "hub" | "hf") {
            return Err(Error::Msg(format!(
                "iris training: unknown repa teacher_source '{source}' (hub | hf)"
            )));
        }
        let hub = text("teacher_hub", DEFAULT_TEACHER_HUB)?;
        let entry = text("teacher", DEFAULT_TEACHER)?;
        let teacher_dim = uint("teacher_dim", 768)?;
        let proj_hidden_dim = uint("proj_hidden_dim", 2048)?;
        let teacher_image_size = uint("teacher_image_size", 224)?;
        let teacher_patch = uint("teacher_patch_size", 14)?;
        let match_student = match get("teacher_match_student") {
            None => false,
            Some(J::Bool(b)) => *b,
            Some(v) => {
                return Err(Error::Msg(format!(
                    "iris training: repa.teacher_match_student must be a bool, got {v}"
                )))
            }
        };
        let layer = uint("layer", default_layer)?;
        let teacher_weights = text("teacher_weights", "")?;
        if weight == 0.0 {
            return Ok(None);
        }
        if source == "hf" {
            return Err(Error::Unsupported(format!(
                "iris training: repa.teacher_source = hf loads {entry:?} through transformers \
                 AutoModel (DINOv3); the native teacher implements the facebookresearch/dinov2 hub \
                 family (teacher_source = hub)"
            )));
        }
        if hub != DEFAULT_TEACHER_HUB {
            return Err(Error::Unsupported(format!(
                "iris training: repa.teacher_hub {hub:?}: the native teacher implements the \
                 {DEFAULT_TEACHER_HUB} entrypoints"
            )));
        }
        let teacher = TeacherArch::from_entrypoint(&entry)?;
        if teacher_dim != teacher.embed_dim {
            return Err(Error::Msg(format!(
                "iris training: repa.teacher_dim {teacher_dim} != {entry}'s embed dim {}",
                teacher.embed_dim
            )));
        }
        if teacher_patch != teacher.patch_size {
            return Err(Error::Msg(format!(
                "iris training: repa.teacher_patch_size {teacher_patch} != {entry}'s patch stride \
                 {} (the teacher would return a different token grid)",
                teacher.patch_size
            )));
        }
        if teacher_image_size < teacher_patch {
            return Err(Error::Msg(format!(
                "iris training: repa.teacher_image_size {teacher_image_size} is below one teacher \
                 patch"
            )));
        }
        if proj_hidden_dim == 0 {
            return Err(Error::Msg(
                "iris training: repa.proj_hidden_dim must be >= 1".into(),
            ));
        }
        if layer == 0 || layer > depth {
            return Err(Error::Msg(format!(
                "iris training: repa needs a capture block in 1..={depth} (model.repa_layer / \
                 repa.layer), got {layer}"
            )));
        }
        Ok(Some(Self {
            weight,
            variant,
            spatial_norm_gamma: gamma,
            teacher,
            teacher_dim,
            proj_hidden_dim,
            teacher_image_size,
            teacher_match_student: match_student,
            layer,
            student_dim,
            teacher_weights: (!teacher_weights.is_empty()).then_some(teacher_weights),
        }))
    }

    /// The identity a checkpoint records for the projector (variant + shapes): a resume carries
    /// the projector over only when it matches.
    pub fn projector_identity(&self) -> String {
        format!(
            "{}:{}:{}:{}",
            self.variant.as_str(),
            self.student_dim,
            self.proj_hidden_dim,
            self.teacher_dim
        )
    }

    /// `_teacher_grid`: the teacher's patch grid for a student grid.
    pub fn teacher_grid(&self, student_grid: (usize, usize)) -> (usize, usize) {
        if self.teacher_match_student {
            student_grid
        } else {
            let side = self.teacher_image_size / self.teacher.patch_size;
            (side, side)
        }
    }

    /// The trainable projector tensors (`repa.projector.*` under upstream's module names) and their
    /// shapes, in a stable order.
    pub fn projector_shapes(&self) -> Vec<(String, Vec<usize>)> {
        let (d, h, t) = (self.student_dim, self.proj_hidden_dim, self.teacher_dim);
        match self.variant {
            RepaVariant::Repa => vec![
                (format!("{PROJECTOR_PREFIX}0.weight"), vec![h, d]),
                (format!("{PROJECTOR_PREFIX}0.bias"), vec![h]),
                (format!("{PROJECTOR_PREFIX}2.weight"), vec![h, h]),
                (format!("{PROJECTOR_PREFIX}2.bias"), vec![h]),
                (format!("{PROJECTOR_PREFIX}4.weight"), vec![t, h]),
                (format!("{PROJECTOR_PREFIX}4.bias"), vec![t]),
            ],
            RepaVariant::Irepa => vec![
                (format!("{PROJECTOR_PREFIX}weight"), vec![t, d, 3, 3]),
                (format!("{PROJECTOR_PREFIX}bias"), vec![t]),
            ],
        }
    }

    /// PyTorch's default init bound of a projector tensor (`kaiming_uniform_(a=√5)` weights and the
    /// `1/√fan_in` bias bound of `nn.Linear` / `nn.Conv2d`): `U(±bound)`.
    pub fn projector_init_bound(&self, key: &str) -> f64 {
        let shapes = self.projector_shapes();
        let module = key.rsplit_once('.').map(|(m, _)| m).unwrap_or(key);
        let weight = shapes
            .iter()
            .find(|(k, _)| *k == format!("{module}.weight"))
            .map(|(_, s)| s.clone())
            .unwrap_or_default();
        let fan_in: usize = weight.iter().skip(1).product();
        1.0 / (fan_in.max(1) as f64).sqrt()
    }
}

/// Key prefix of the projector tensors in a checkpoint (`repa.projector.` upstream).
pub const PROJECTOR_PREFIX: &str = "repa.projector.";

// =============================================================================================
// Resampling operators (torch `F.interpolate`, `aten/src/ATen/native/UpSample.h`)
// =============================================================================================

/// `area_pixel_compute_scale` without `align_corners`: `1 / scale_factor` when one is given, else
/// `in / out` (f32, as torch's `opmath_t`).
fn pixel_scale(input: usize, output: usize, scale_factor: Option<f64>) -> f32 {
    match scale_factor {
        Some(s) if s > 0.0 => (1.0 / s) as f32,
        _ => input as f32 / output as f32,
    }
}

/// The `[output, input]` row-major matrix of torch's `bicubic` interpolation along one axis
/// (`align_corners=False`, no antialias, `A = −0.75`, border-clamped taps). `scale_factor` is the
/// factor passed to `F.interpolate` (it replaces `in / out` in the source-index law).
pub fn bicubic_matrix(input: usize, output: usize, scale_factor: Option<f64>) -> Vec<f32> {
    let a = -0.75f32;
    let conv1 = |x: f32| ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    let conv2 = |x: f32| ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
    let scale = pixel_scale(input, output, scale_factor);
    let mut m = vec![0f32; output * input];
    for o in 0..output {
        let real = scale * (o as f32 + 0.5) - 0.5;
        let idx = real.floor();
        let t = real - idx;
        let coeffs = [conv2(t + 1.0), conv1(t), conv1(1.0 - t), conv2(2.0 - t)];
        for (k, c) in coeffs.iter().enumerate() {
            let src = (idx as i64 - 1 + k as i64).clamp(0, input as i64 - 1) as usize;
            m[o * input + src] += c;
        }
    }
    m
}

/// The `[output, input]` matrix of torch's antialiased `bicubic` (`antialias=True`,
/// `_upsample_bicubic2d_aa`: the PIL-style separable filter with `a = −0.5`, support `2·scale` when
/// downsampling, weights normalized per output).
pub fn bicubic_aa_matrix(input: usize, output: usize, scale_factor: Option<f64>) -> Vec<f32> {
    let a = -0.5f64;
    let filter = |x: f64| -> f64 {
        let x = x.abs();
        if x < 1.0 {
            ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
        } else if x < 2.0 {
            (((x - 5.0) * x + 8.0) * x - 4.0) * a
        } else {
            0.0
        }
    };
    let scale = pixel_scale(input, output, scale_factor) as f64;
    let support = if scale >= 1.0 { 2.0 * scale } else { 2.0 };
    let invscale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    let mut m = vec![0f32; output * input];
    for o in 0..output {
        let center = scale * (o as f64 + 0.5);
        let xmin = ((center - support + 0.5) as i64).max(0) as usize;
        let xmax = ((center + support + 0.5) as i64).min(input as i64) as usize;
        let ws: Vec<f64> = (xmin..xmax)
            .map(|j| filter((j as f64 - center + 0.5) * invscale))
            .collect();
        let total: f64 = ws.iter().sum();
        for (j, w) in (xmin..xmax).zip(ws) {
            m[o * input + j] = (if total != 0.0 { w / total } else { 0.0 }) as f32;
        }
    }
    m
}

/// The `[output, input]` matrix of torch's `bilinear` interpolation along one axis
/// (`align_corners=False`, no antialias; negative source indices clamp to 0; an unchanged size is
/// the identity).
pub fn bilinear_matrix(input: usize, output: usize) -> Vec<f32> {
    let mut m = vec![0f32; output * input];
    if input == output {
        for i in 0..input {
            m[i * input + i] = 1.0;
        }
        return m;
    }
    let scale = input as f32 / output as f32;
    for o in 0..output {
        let real = (scale * (o as f32 + 0.5) - 0.5).max(0.0);
        let i0 = real as usize;
        let i1 = if i0 < input - 1 { i0 + 1 } else { i0 };
        let l1 = (real - i0 as f32).clamp(0.0, 1.0);
        let l0 = 1.0 - l1;
        m[o * input + i0] += l0;
        m[o * input + i1] += l1;
    }
    m
}

/// The two matrices of the teacher's input resize: the clean image to `grid · patch` pixels.
pub fn teacher_image_resize(
    height: usize,
    width: usize,
    grid: (usize, usize),
    patch: usize,
) -> (Vec<f32>, Vec<f32>) {
    (
        bicubic_matrix(height, grid.0 * patch, None),
        bicubic_matrix(width, grid.1 * patch, None),
    )
}

/// The two matrices of `interpolate_pos_encoding` (`M × M` learned grid → `h0 × w0`), or `None`
/// when the hub returns the table unchanged (`npatch == N and w == h`).
pub fn pos_embed_resize(arch: &TeacherArch, grid: (usize, usize)) -> Option<(Vec<f32>, Vec<f32>)> {
    let m = arch.pos_grid;
    if grid.0 * grid.1 == m * m && grid.0 == grid.1 {
        return None;
    }
    let (h0, w0) = grid;
    let build = |out: usize| -> Vec<f32> {
        if arch.interpolate_offset != 0.0 {
            let sf = (out as f64 + arch.interpolate_offset) / m as f64;
            if arch.interpolate_antialias {
                bicubic_aa_matrix(m, out, Some(sf))
            } else {
                bicubic_matrix(m, out, Some(sf))
            }
        } else if arch.interpolate_antialias {
            bicubic_aa_matrix(m, out, None)
        } else {
            bicubic_matrix(m, out, None)
        }
    };
    Some((build(h0), build(w0)))
}

/// Apply `[out_h, in_h]` and `[out_w, in_w]` matrices to a host `[C, in_h, in_w]` map (reference
/// helper for tests and host-side callers): `out[c] = Mh · x[c] · Mwᵀ`.
pub fn resize_host(
    x: &[f32],
    channels: usize,
    (in_h, in_w): (usize, usize),
    (out_h, out_w): (usize, usize),
    mh: &[f32],
    mw: &[f32],
) -> Vec<f32> {
    let mut out = vec![0f32; channels * out_h * out_w];
    let mut tmp = vec![0f32; in_h * out_w];
    for c in 0..channels {
        let src = &x[c * in_h * in_w..(c + 1) * in_h * in_w];
        for r in 0..in_h {
            for o in 0..out_w {
                let mut s = 0f32;
                for i in 0..in_w {
                    s += src[r * in_w + i] * mw[o * in_w + i];
                }
                tmp[r * out_w + o] = s;
            }
        }
        for o in 0..out_h {
            for col in 0..out_w {
                let mut s = 0f32;
                for r in 0..in_h {
                    s += mh[o * in_h + r] * tmp[r * out_w + col];
                }
                out[c * out_h * out_w + o * out_w + col] = s;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entrypoints_resolve_their_architecture() {
        let b = TeacherArch::from_entrypoint("dinov2_vitb14").unwrap();
        assert_eq!(
            (b.embed_dim, b.depth, b.num_heads, b.ffn_hidden),
            (768, 12, 12, 3072)
        );
        assert_eq!((b.num_register_tokens, b.interpolate_offset), (0, 0.1));
        let g = TeacherArch::from_entrypoint("dinov2_vitg14_reg").unwrap();
        assert_eq!(
            (g.ffn, g.ffn_hidden, g.num_register_tokens),
            (TeacherFfn::SwiGlu, 4096, 4)
        );
        assert!(g.interpolate_antialias && g.interpolate_offset == 0.0);
        assert!(matches!(
            TeacherArch::from_entrypoint("dinov3_vitb16"),
            Err(Error::Unsupported(_))
        ));
        let shapes = b.tensor_shapes();
        assert_eq!(shapes.len(), 4 + 12 * 14 + 2);
    }

    #[test]
    fn hf_names_map_onto_the_hub_layout() {
        assert_eq!(
            hf_sources("blocks.3.attn.qkv.weight"),
            vec![
                "encoder.layer.3.attention.attention.query.weight",
                "encoder.layer.3.attention.attention.key.weight",
                "encoder.layer.3.attention.attention.value.weight"
            ]
        );
        assert_eq!(
            hf_sources("blocks.0.ls2.gamma"),
            vec!["encoder.layer.0.layer_scale2.lambda1"]
        );
        assert_eq!(
            hf_sources("pos_embed"),
            vec!["embeddings.position_embeddings"]
        );
        assert_eq!(
            hf_sources("blocks.1.mlp.fc2.bias"),
            vec!["encoder.layer.1.mlp.fc2.bias"]
        );
    }

    #[test]
    fn interpolation_matrices_are_partitions_of_unity() {
        for (i, o) in [(37, 16), (16, 37), (256, 224), (5, 5)] {
            for m in [
                bicubic_matrix(i, o, None),
                bilinear_matrix(i, o),
                bicubic_aa_matrix(i, o, None),
            ] {
                for r in 0..o {
                    let s: f32 = m[r * i..(r + 1) * i].iter().sum();
                    assert!((s - 1.0).abs() < 1e-5, "{i}->{o} row {r}: {s}");
                }
            }
        }
        // identity sizes
        let id = bilinear_matrix(4, 4);
        assert_eq!(id[5], 1.0);
        // the pos table is untouched only at the native square grid
        let b = TeacherArch::from_entrypoint("dinov2_vitb14").unwrap();
        assert!(pos_embed_resize(&b, (37, 37)).is_none());
        assert!(pos_embed_resize(&b, (16, 16)).is_some());
    }

    fn apply(m: &[f32], x: &[f32], out: usize) -> Vec<f32> {
        let n = x.len();
        (0..out)
            .map(|o| (0..n).map(|i| m[o * n + i] * x[i]).sum())
            .collect()
    }

    /// Values printed by torch 2.8 `F.interpolate` on ramps (`arange(n)` as a `1 × n` image).
    #[test]
    // The expected values are torch's printed f32 outputs, kept digit for digit.
    #[allow(clippy::excessive_precision)]
    fn interpolation_matches_torch_on_ramps() {
        let ramp = |n: usize| (0..n).map(|v| v as f32).collect::<Vec<_>>();
        // size=(1,8), bicubic
        let y = apply(&bicubic_matrix(4, 8, None), &ramp(4), 8);
        let want = [
            -0.10546875f32,
            0.19140625,
            0.66796875,
            1.296875,
            1.703125,
            2.33203125,
            2.80859375,
            3.10546875,
        ];
        for (a, b) in y.iter().zip(want) {
            assert!((a - b).abs() < 1e-5, "bicubic {y:?}");
        }
        // size=(1,16), bicubic, antialias=True (the _reg teachers' pos-embed resize)
        let y = apply(&bicubic_aa_matrix(37, 16, None), &ramp(37), 16);
        for (a, b) in y
            .iter()
            .zip([0.70842701f32, 2.91932178, 5.28336239, 7.59269524])
        {
            assert!((a - b).abs() < 1e-4, "bicubic aa {y:?}");
        }
        // scale_factor=16.1/37 (the hub's interpolate_offset kludge)
        let y = apply(&bicubic_matrix(37, 16, Some(16.1 / 37.0)), &ramp(37), 16);
        for (a, b) in y
            .iter()
            .zip([0.55516231f32, 2.92484164, 5.29248953, 7.53269148])
        {
            assert!((a - b).abs() < 1e-4, "bicubic sf {y:?}");
        }
    }
}
