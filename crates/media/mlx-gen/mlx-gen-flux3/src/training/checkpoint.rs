//! Strict single-file MLX training state and S6-compatible plain-LoRA export.
//! Host readback occurs only for saved-state I/O, never inside the autograd closure.
use super::{
    optim::{AdamMoment, SourceAdamW},
    TrainingGraph,
};
use mlx_gen::gen_core::{
    flux3::{
        training::{
            checkpoint::{publish_bytes, ResumePolicy, TrainingCursor},
            ResumeIdentity,
        },
        BASE_WEIGHTS_REPO, BASE_WEIGHTS_REVISION,
    },
    CancelFlag,
};
use mlx_gen::{train::lora::LoraParams, Error, Result};
use mlx_rs::{transforms::eval, Array, Dtype};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    path::{Path, PathBuf},
};

const FORMAT: &str = "flux3-mlx-training-resume-v1";
const METADATA_KEYS: [&str; 9] = [
    "policy",
    "format",
    "identity",
    "cursor",
    "gain_bits",
    "optimizer_config",
    "optimizer_steps",
    "tensor_manifest",
    "pending_keys",
];

pub struct Snapshot<'a> {
    pub params: &'a LoraParams,
    pub optimizer: &'a SourceAdamW,
    pub identity: &'a ResumeIdentity,
    pub cursor: &'a TrainingCursor,
    pub policy: ResumePolicy,
    pub pending: &'a LoraParams,
}
pub struct RestoredState {
    pub cursor: TrainingCursor,
    pub pending: LoraParams,
}
fn invalid(detail: impl std::fmt::Display) -> Error {
    Error::Unsupported(format!("flux3 checkpoint: {detail}"))
}
fn cancel(flag: &CancelFlag) -> Result<()> {
    if flag.is_cancelled() {
        Err(Error::Canceled)
    } else {
        Ok(())
    }
}
fn validate_identity(
    graph: &TrainingGraph,
    params: &LoraParams,
    identity: &ResumeIdentity,
) -> Result<()> {
    identity.validate().map_err(|e| invalid(e.to_string()))?;
    graph.validate_factors(params)?;
    let shapes = params
        .iter()
        .map(|(name, array)| {
            (
                name.to_string(),
                array
                    .shape()
                    .iter()
                    .map(|d| *d as usize)
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    if shapes != identity.factors
        || identity.configuration.get("optimizer").map(String::as_str) != Some("adamw")
    {
        return Err(invalid(
            "actual factor shapes/master precision or source optimizer differs from identity",
        ));
    }
    Ok(())
}
fn validate_pending(
    pending: &LoraParams,
    cursor: &TrainingCursor,
    factors: &BTreeMap<String, Vec<usize>>,
) -> Result<()> {
    if (cursor.accumulation_count == 0) != pending.is_empty() {
        return Err(invalid(
            "pending accumulation state is missing or unexpectedly present",
        ));
    }
    for (name, array) in pending {
        let shape = array
            .shape()
            .iter()
            .map(|d| *d as usize)
            .collect::<Vec<_>>();
        if array.dtype() != Dtype::Float32 || factors.get(name.as_ref()) != Some(&shape) {
            return Err(invalid(format!(
                "undeclared pending gradient shape/precision {name}"
            )));
        }
    }
    Ok(())
}
struct Buffer {
    shape: Vec<usize>,
    bytes: Vec<u8>,
}
fn buffer(array: &Array) -> Result<Buffer> {
    if array.dtype() != Dtype::Float32 {
        return Err(invalid("snapshot tensors must retain master f32 precision"));
    }
    let flat = array.reshape(&[-1])?;
    eval([&flat])?;
    let values = flat.as_slice::<f32>();
    if values.iter().any(|v| !v.is_finite()) {
        return Err(invalid("nonfinite factor/optimizer/accumulation tensor"));
    }
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(Buffer {
        shape: array.shape().iter().map(|d| *d as usize).collect(),
        bytes,
    })
}
fn record(shape: &[usize], bytes: &[u8]) -> serde_json::Value {
    serde_json::json!({"shape":shape,"dtype":"F32","sha256":format!("{:x}",Sha256::digest(bytes))})
}
fn encode(
    buffers: &BTreeMap<String, Buffer>,
    metadata: HashMap<String, String>,
) -> Result<Vec<u8>> {
    let views = buffers
        .iter()
        .map(|(name, buffer)| {
            let view = safetensors::tensor::TensorView::new(
                safetensors::Dtype::F32,
                buffer.shape.clone(),
                &buffer.bytes,
            )
            .map_err(|e| invalid(format!("serialize tensor {name}: {e}")))?;
            Ok((name.clone(), view))
        })
        .collect::<Result<Vec<_>>>()?;
    safetensors::serialize(views, &Some(metadata))
        .map_err(|e| invalid(format!("serialize state: {e}")))
}
fn array(view: &safetensors::tensor::TensorView<'_>) -> Result<Array> {
    if view.dtype() != safetensors::Dtype::F32 {
        return Err(invalid("snapshot master precision is not F32"));
    }
    let shape = view
        .shape()
        .iter()
        .map(|d| i32::try_from(*d).map_err(|_| invalid("snapshot dimension exceeds MLX i32")))
        .collect::<Result<Vec<_>>>()?;
    let values = view
        .data()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("validated f32 bytes")))
        .collect::<Vec<_>>();
    if values.iter().any(|v| !v.is_finite()) {
        return Err(invalid("nonfinite saved-state tensor"));
    }
    Ok(Array::from_slice(&values, &shape))
}

pub fn save(
    graph: &TrainingGraph,
    state: Snapshot<'_>,
    root: &Path,
    name: &str,
    flag: &CancelFlag,
) -> Result<PathBuf> {
    cancel(flag)?;
    validate_identity(graph, state.params, state.identity)?;
    state
        .cursor
        .validate(
            state.policy.total_micro_steps,
            state.policy.accumulation,
            state.policy.sampler_len,
        )
        .map_err(|e| invalid(e.to_string()))?;
    validate_pending(state.pending, state.cursor, &state.identity.factors)?;
    let mut buffers = BTreeMap::new();
    for (key, value) in state.params {
        cancel(flag)?;
        buffers.insert(format!("factor/{key}"), buffer(value)?);
    }
    for (key, value) in state.pending {
        cancel(flag)?;
        buffers.insert(format!("pending/{key}"), buffer(value)?);
    }
    let mut steps = BTreeMap::new();
    for (key, moment) in &state.optimizer.moments {
        cancel(flag)?;
        let expected = state
            .params
            .get(key)
            .ok_or_else(|| invalid(format!("optimizer has undeclared factor {key}")))?;
        if moment.step == 0
            || moment.step as u64 > state.cursor.update_step
            || moment.first.shape() != expected.shape()
            || moment.second.shape() != expected.shape()
        {
            return Err(invalid(format!(
                "optimizer moment shape/update counter mismatch {key}"
            )));
        }
        buffers.insert(format!("moment_first/{key}"), buffer(&moment.first)?);
        buffers.insert(format!("moment_second/{key}"), buffer(&moment.second)?);
        steps.insert(key.to_string(), moment.step);
    }
    let manifest = buffers
        .iter()
        .map(|(key, b)| (key.clone(), record(&b.shape, &b.bytes)))
        .collect::<BTreeMap<_, _>>();
    let metadata = HashMap::from([
        ("format".into(), FORMAT.into()),
        ("policy".into(), state.policy.fingerprint()),
        (
            "identity".into(),
            state
                .identity
                .to_metadata()
                .map_err(|e| invalid(e.to_string()))?,
        ),
        (
            "cursor".into(),
            state
                .cursor
                .to_metadata()
                .map_err(|e| invalid(e.to_string()))?,
        ),
        (
            "gain_bits".into(),
            format!("{:016x}", graph.scale.to_bits()),
        ),
        (
            "optimizer_config".into(),
            state.optimizer.configuration_fingerprint(),
        ),
        (
            "optimizer_steps".into(),
            serde_json::to_string(&steps).map_err(|e| invalid(e.to_string()))?,
        ),
        (
            "tensor_manifest".into(),
            serde_json::to_string(&manifest).map_err(|e| invalid(e.to_string()))?,
        ),
        (
            "pending_keys".into(),
            serde_json::to_string(
                &state
                    .pending
                    .keys()
                    .map(|key| key.to_string())
                    .collect::<BTreeSet<_>>(),
            )
            .map_err(|e| invalid(e.to_string()))?,
        ),
    ]);
    let bytes = encode(&buffers, metadata)?;
    cancel(flag)?;
    publish_bytes(root, name, &bytes, flag).map_err(Error::from)
}

pub fn restore(
    graph: &TrainingGraph,
    params: &mut LoraParams,
    optimizer: &mut SourceAdamW,
    identity: &ResumeIdentity,
    policy: ResumePolicy,
    path: &Path,
    flag: &CancelFlag,
) -> Result<RestoredState> {
    cancel(flag)?;
    validate_identity(graph, params, identity)?;
    let factor_bytes = identity
        .factors
        .values()
        .try_fold(0usize, |sum, shape| {
            shape
                .iter()
                .try_fold(4usize, |n, d| n.checked_mul(*d))
                .and_then(|n| sum.checked_add(n))
        })
        .ok_or_else(|| invalid("factor byte count overflow"))?;
    let maximum = factor_bytes
        .checked_mul(6)
        .and_then(|n| n.checked_add(16 * 1024 * 1024))
        .ok_or_else(|| invalid("snapshot byte bound overflow"))?;
    if std::fs::metadata(path)
        .map_err(|e| invalid(format!("stat snapshot: {e}")))?
        .len()
        > maximum as u64
    {
        return Err(invalid("snapshot exceeds declared factor/state bound"));
    }
    let bytes = std::fs::read(path).map_err(|e| invalid(format!("read snapshot: {e}")))?;
    let (_, header) = safetensors::SafeTensors::read_metadata(&bytes)
        .map_err(|e| invalid(format!("snapshot header: {e}")))?;
    let metadata = header
        .metadata()
        .clone()
        .ok_or_else(|| invalid("snapshot metadata is missing"))?;
    if metadata.keys().map(String::as_str).collect::<BTreeSet<_>>()
        != METADATA_KEYS.into_iter().collect::<BTreeSet<_>>()
    {
        return Err(invalid("unknown/incomplete snapshot metadata"));
    }
    let field = |key: &str| {
        metadata
            .get(key)
            .map(String::as_str)
            .ok_or_else(|| invalid(format!("missing snapshot {key}")))
    };
    if field("format")? != FORMAT {
        return Err(invalid("unsupported snapshot format"));
    }
    if field("policy")? != policy.fingerprint() {
        return Err(invalid(
            "effective run/sampler/accumulation policy mismatch",
        ));
    }
    let saved =
        ResumeIdentity::from_metadata(field("identity")?).map_err(|e| invalid(e.to_string()))?;
    identity
        .ensure_matches(&saved)
        .map_err(|e| invalid(e.to_string()))?;
    if field("gain_bits")? != format!("{:016x}", graph.scale.to_bits()) {
        return Err(invalid(
            "actual graph alpha/rank gain differs from saved state",
        ));
    }
    if field("optimizer_config")? != optimizer.configuration_fingerprint() {
        return Err(invalid("optimizer configuration mismatch; refusing reset"));
    }
    let cursor =
        TrainingCursor::from_metadata(field("cursor")?).map_err(|e| invalid(e.to_string()))?;
    cursor
        .validate(
            policy.total_micro_steps,
            policy.accumulation,
            policy.sampler_len,
        )
        .map_err(|e| invalid(e.to_string()))?;
    let steps: BTreeMap<String, u32> = serde_json::from_str(field("optimizer_steps")?)
        .map_err(|e| invalid(format!("optimizer counters: {e}")))?;
    let pending_keys: BTreeSet<String> = serde_json::from_str(field("pending_keys")?)
        .map_err(|e| invalid(format!("pending keys: {e}")))?;
    let manifest: BTreeMap<String, serde_json::Value> =
        serde_json::from_str(field("tensor_manifest")?)
            .map_err(|e| invalid(format!("tensor manifest: {e}")))?;
    let container = safetensors::SafeTensors::deserialize(&bytes)
        .map_err(|e| invalid(format!("snapshot payload: {e}")))?;
    let tensors = container.tensors().into_iter().collect::<BTreeMap<_, _>>();
    let mut expected = identity
        .factors
        .keys()
        .map(|name| format!("factor/{name}"))
        .collect::<BTreeSet<_>>();
    for (name, step) in &steps {
        if !identity.factors.contains_key(name) || *step == 0 || *step as u64 > cursor.update_step {
            return Err(invalid(format!(
                "invalid optimizer factor/update counter {name}"
            )));
        }
        expected.insert(format!("moment_first/{name}"));
        expected.insert(format!("moment_second/{name}"));
    }
    for name in &pending_keys {
        if !identity.factors.contains_key(name) {
            return Err(invalid(format!("undeclared accumulation factor {name}")));
        }
        expected.insert(format!("pending/{name}"));
    }
    if tensors.keys().cloned().collect::<BTreeSet<_>>() != expected
        || manifest.keys().cloned().collect::<BTreeSet<_>>() != expected
    {
        return Err(invalid(
            "saved factor/moment/accumulation tensor set is incomplete or contains extra keys",
        ));
    }
    // Check all bytes/shapes BEFORE creating arrays or mutating trainable/optimizer state.
    for (key, view) in &tensors {
        cancel(flag)?;
        let (_, name) = key
            .split_once('/')
            .ok_or_else(|| invalid("malformed tensor key"))?;
        if view.dtype() != safetensors::Dtype::F32
            || view.shape() != identity.factors[name].as_slice()
            || manifest[key] != record(view.shape(), view.data())
        {
            return Err(invalid(format!(
                "tensor {key} shape/precision/content digest mismatch"
            )));
        }
    }
    let restored = identity
        .factors
        .keys()
        .map(|name| {
            Ok((
                name.clone().into(),
                array(&tensors[&format!("factor/{name}")])?,
            ))
        })
        .collect::<Result<LoraParams>>()?;
    let moments = steps
        .iter()
        .map(|(name, step)| {
            Ok((
                name.clone().into(),
                AdamMoment {
                    first: array(&tensors[&format!("moment_first/{name}")])?,
                    second: array(&tensors[&format!("moment_second/{name}")])?,
                    step: *step,
                },
            ))
        })
        .collect::<Result<HashMap<_, _>>>()?;
    let pending = pending_keys
        .iter()
        .map(|name| {
            Ok((
                name.clone().into(),
                array(&tensors[&format!("pending/{name}")])?,
            ))
        })
        .collect::<Result<LoraParams>>()?;
    validate_pending(&pending, &cursor, &identity.factors)?;
    cancel(flag)?;
    *params = restored;
    optimizer.moments = moments;
    cancel(flag)?;
    Ok(RestoredState { cursor, pending })
}

pub fn export_adapter(
    graph: &TrainingGraph,
    params: &LoraParams,
    identity: &ResumeIdentity,
    root: &Path,
    name: &str,
    flag: &CancelFlag,
) -> Result<PathBuf> {
    cancel(flag)?;
    validate_identity(graph, params, identity)?;
    if identity.base_repo != BASE_WEIGHTS_REPO || identity.base_revision != BASE_WEIGHTS_REVISION {
        return Err(invalid(
            "adapter export requires canonical pinned base compatibility identity",
        ));
    }
    let alpha = graph.alpha;
    if !alpha.is_finite() || alpha <= 0.0 {
        return Err(invalid("invalid global LoRA alpha"));
    }
    let mut buffers = BTreeMap::new();
    for target in &graph.targets {
        for suffix in ["lora_A.weight", "lora_B.weight"] {
            cancel(flag)?;
            buffers.insert(
                format!("{}.{}", target.diffusers, suffix),
                buffer(&params[format!("{}.{}", target.native, suffix).as_str()])?,
            );
        }
    }
    let mut metadata = HashMap::from([
        ("base_model".into(),identity.base_repo.clone()),("base_revision".into(),identity.base_revision.clone()),
        ("network_type".into(),"lora".into()),("rank".into(),graph.rank.to_string()),("lora_alpha".into(),alpha.to_string()),("scaling_convention".into(),"alpha/rank".into()),
        ("training_identity".into(),identity.to_metadata().map_err(|e| invalid(e.to_string()))?),
        ("lora_adapter_metadata".into(),serde_json::json!({"peft_type":"LORA","r":graph.rank,"lora_alpha":alpha,"use_rslora":false,"use_dora":false,"rank_pattern":{},"alpha_pattern":{}}).to_string()),
    ]);
    if let Some(fixture) = identity.configuration.get("fixture") {
        metadata.insert("fixture".into(), fixture.clone());
    }
    let bytes = encode(&buffers, metadata)?;
    cancel(flag)?;
    publish_bytes(root, name, &bytes, flag).map_err(Error::from)
}
