//! Strict single-file native training snapshots and compatible adapter export.
//! All host readback here is checkpoint I/O AFTER backward, never a forward/autograd seam.
use super::TrainingGraph;
use candle_gen::candle_core::{DType, Device, Tensor, Var};
use candle_gen::gen_core::{
    flux3::{
        training::{
            checkpoint::{publish_bytes, ResumePolicy, TrainingCursor},
            ResumeIdentity,
        },
        BASE_WEIGHTS_REPO, BASE_WEIGHTS_REVISION,
    },
    CancelFlag,
};
use candle_gen::train::optim::TrainOptimizer;
use candle_gen::{CandleError as Error, Result};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    path::{Path, PathBuf},
};

const FORMAT: &str = "flux3-native-training-resume-v1";
const OWN_METADATA: [&str; 8] = [
    "policy",
    "format",
    "identity",
    "cursor",
    "gain_bits",
    "tensor_manifest",
    "optimizer_order",
    "pending_keys",
];

pub struct Snapshot<'a> {
    pub identity: &'a ResumeIdentity,
    pub cursor: &'a TrainingCursor,
    pub policy: ResumePolicy,
    pub pending: &'a BTreeMap<String, Tensor>,
}
pub struct RestoredState {
    pub cursor: TrainingCursor,
    pub pending: BTreeMap<String, Tensor>,
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
fn names(graph: &TrainingGraph) -> Vec<(String, Var)> {
    graph
        .factors
        .iter()
        .flat_map(|factor| {
            [
                (
                    format!("{}.lora_A.weight", factor.target.native),
                    factor.down.clone(),
                ),
                (
                    format!("{}.lora_B.weight", factor.target.native),
                    factor.up.clone(),
                ),
            ]
        })
        .collect()
}
fn validate_identity(
    graph: &TrainingGraph,
    identity: &ResumeIdentity,
) -> Result<Vec<(String, Var)>> {
    identity.validate().map_err(|e| invalid(e.to_string()))?;
    let named = names(graph);
    let shapes = named
        .iter()
        .map(|(name, var)| (name.clone(), var.dims().to_vec()))
        .collect::<BTreeMap<_, _>>();
    if identity.factors != shapes || named.iter().any(|(_, var)| var.dtype() != DType::F32) {
        return Err(invalid(
            "declared factor keys/shapes/master precision differ from the actual graph",
        ));
    }
    Ok(named)
}
fn validate_pending(
    pending: &BTreeMap<String, Tensor>,
    cursor: &TrainingCursor,
    factors: &BTreeMap<String, Vec<usize>>,
) -> Result<()> {
    if (cursor.accumulation_count == 0) != pending.is_empty() {
        return Err(invalid(
            "pending accumulation state is missing or unexpectedly present",
        ));
    }
    for (name, tensor) in pending {
        if tensor.dtype() != DType::F32
            || factors.get(name).is_none_or(|shape| tensor.dims() != shape)
        {
            return Err(invalid(format!(
                "pending gradient {name} has undeclared shape/precision"
            )));
        }
    }
    Ok(())
}
fn record(tensor: &Tensor) -> Result<serde_json::Value> {
    if tensor.dtype() != DType::F32 {
        return Err(invalid("snapshot tensors must retain master f32 precision"));
    }
    let values = tensor
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    if values.iter().any(|v| !v.is_finite()) {
        return Err(invalid("nonfinite factor/optimizer/accumulation tensor"));
    }
    let mut hash = Sha256::new();
    for value in values {
        hash.update(value.to_le_bytes());
    }
    Ok(
        serde_json::json!({"shape": tensor.dims(), "dtype": "F32", "sha256": format!("{:x}", hash.finalize())}),
    )
}
fn optimizer_layout(
    named: &[(String, Var)],
    tensors: &HashMap<String, Tensor>,
    kind: &str,
) -> Result<()> {
    let suffixes: &[&str] = match kind {
        "adam" | "adamw" => &["m", "v"],
        "rose" => &[],
        "prodigy" => &["exp_avg", "exp_avg_sq", "s", "p0"],
        _ => return Err(invalid("unknown snapshot optimizer")),
    };
    let mut present = BTreeMap::<usize, BTreeSet<String>>::new();
    for (key, tensor) in tensors.iter().filter(|(key, _)| key.starts_with("optim.")) {
        let rest = key.strip_prefix("optim.").expect("filtered");
        let (index, suffix) = rest
            .split_once('.')
            .ok_or_else(|| invalid(format!("malformed optimizer tensor {key}")))?;
        let index: usize = index
            .parse()
            .map_err(|_| invalid(format!("malformed optimizer index {key}")))?;
        let (_, var) = named
            .get(index)
            .ok_or_else(|| invalid(format!("undeclared optimizer factor {key}")))?;
        if !suffixes.contains(&suffix)
            || tensor.dtype() != DType::F32
            || tensor.dims() != var.dims()
        {
            return Err(invalid(format!(
                "optimizer tensor {key} has unexpected key/shape/precision"
            )));
        }
        present.entry(index).or_default().insert(suffix.to_owned());
    }
    let required: BTreeSet<String> = suffixes.iter().map(|s| s.to_string()).collect();
    if present.values().any(|group| *group != required)
        || ((kind == "adamw" || kind == "adam") && present.len() != named.len())
    {
        return Err(invalid(
            "optimizer state is missing one or more moment tensors",
        ));
    }
    Ok(())
}

pub fn save(
    graph: &TrainingGraph,
    optimizer: &TrainOptimizer,
    state: Snapshot<'_>,
    root: &Path,
    name: &str,
    flag: &CancelFlag,
) -> Result<PathBuf> {
    cancel(flag)?;
    let named = validate_identity(graph, state.identity)?;
    state
        .cursor
        .validate(
            state.policy.total_micro_steps,
            state.policy.accumulation,
            state.policy.sampler_len,
        )
        .map_err(|e| invalid(e.to_string()))?;
    validate_pending(state.pending, state.cursor, &state.identity.factors)?;
    let (mut tensors, mut metadata) = optimizer.snapshot()?;
    optimizer_layout(&named, &tensors, optimizer.kind_tag())?;
    for (name, var) in &named {
        tensors.insert(
            format!("factor/{name}"),
            var.as_tensor().to_device(&Device::Cpu)?,
        );
    }
    for (name, tensor) in state.pending {
        tensors.insert(format!("pending/{name}"), tensor.to_device(&Device::Cpu)?);
    }
    let mut manifest = BTreeMap::new();
    for (name, tensor) in &tensors {
        cancel(flag)?;
        manifest.insert(name.clone(), record(tensor)?);
    }
    metadata.insert("format".into(), FORMAT.into());
    metadata.insert("policy".into(), state.policy.fingerprint());
    metadata.insert(
        "identity".into(),
        state
            .identity
            .to_metadata()
            .map_err(|e| invalid(e.to_string()))?,
    );
    metadata.insert(
        "cursor".into(),
        state
            .cursor
            .to_metadata()
            .map_err(|e| invalid(e.to_string()))?,
    );
    metadata.insert(
        "gain_bits".into(),
        format!("{:016x}", graph.scale.to_bits()),
    );
    metadata.insert(
        "optimizer_order".into(),
        serde_json::to_string(&named.iter().map(|(name, _)| name).collect::<Vec<_>>())
            .map_err(|e| invalid(e.to_string()))?,
    );
    metadata.insert(
        "pending_keys".into(),
        serde_json::to_string(&state.pending.keys().collect::<Vec<_>>())
            .map_err(|e| invalid(e.to_string()))?,
    );
    metadata.insert(
        "tensor_manifest".into(),
        serde_json::to_string(&manifest).map_err(|e| invalid(e.to_string()))?,
    );
    let bytes = safetensors::serialize(tensors.iter(), Some(metadata))
        .map_err(|e| invalid(format!("serialize snapshot: {e}")))?;
    cancel(flag)?;
    publish_bytes(root, name, &bytes, flag).map_err(Error::from)
}

pub fn restore(
    graph: &TrainingGraph,
    optimizer: &mut TrainOptimizer,
    identity: &ResumeIdentity,
    policy: ResumePolicy,
    path: &Path,
    flag: &CancelFlag,
) -> Result<RestoredState> {
    cancel(flag)?;
    let named = validate_identity(graph, identity)?;
    let factor_bytes = identity
        .factors
        .values()
        .try_fold(0usize, |sum, shape| {
            shape
                .iter()
                .try_fold(4usize, |bytes, n| bytes.checked_mul(*n))
                .and_then(|bytes| sum.checked_add(bytes))
        })
        .ok_or_else(|| invalid("factor byte count overflows"))?;
    let maximum = factor_bytes
        .checked_mul(6)
        .and_then(|n| n.checked_add(16 * 1024 * 1024))
        .ok_or_else(|| invalid("snapshot byte bound overflows"))?;
    if std::fs::metadata(path)
        .map_err(|e| invalid(format!("stat snapshot: {e}")))?
        .len()
        > maximum as u64
    {
        return Err(invalid(
            "snapshot exceeds the declared factor/optimizer state bound",
        ));
    }
    let bytes = std::fs::read(path).map_err(|e| invalid(format!("read snapshot: {e}")))?;
    let (_, header) = safetensors::SafeTensors::read_metadata(&bytes)
        .map_err(|e| invalid(format!("snapshot header: {e}")))?;
    let metadata = header
        .metadata()
        .clone()
        .ok_or_else(|| invalid("snapshot metadata is missing"))?;
    let field = |key: &str| {
        metadata
            .get(key)
            .map(String::as_str)
            .ok_or_else(|| invalid(format!("missing snapshot metadata {key}")))
    };
    if field("format")? != FORMAT {
        return Err(invalid("unsupported snapshot format"));
    }
    if field("policy")? != policy.fingerprint() {
        return Err(invalid(
            "effective run/sampler/accumulation policy mismatch",
        ));
    }
    let saved_identity =
        ResumeIdentity::from_metadata(field("identity")?).map_err(|e| invalid(e.to_string()))?;
    identity
        .ensure_matches(&saved_identity)
        .map_err(|e| invalid(e.to_string()))?;
    if field("gain_bits")? != format!("{:016x}", graph.scale.to_bits()) {
        return Err(invalid(
            "actual graph alpha/rank gain differs from saved state",
        ));
    }
    let order: Vec<String> = serde_json::from_str(field("optimizer_order")?)
        .map_err(|e| invalid(format!("optimizer order: {e}")))?;
    if order
        != named
            .iter()
            .map(|(name, _)| name.clone())
            .collect::<Vec<_>>()
    {
        return Err(invalid("optimizer factor order mismatch"));
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
    let (_, expected_metadata) = optimizer.snapshot()?;
    let expected_keys = expected_metadata
        .keys()
        .map(String::as_str)
        .chain(OWN_METADATA)
        .collect::<BTreeSet<_>>();
    if metadata.keys().map(String::as_str).collect::<BTreeSet<_>>() != expected_keys {
        return Err(invalid("unknown/incomplete optimizer snapshot metadata"));
    }
    for key in ["optimizer", "optimizer_config"] {
        if metadata.get(key) != expected_metadata.get(key) {
            return Err(invalid(format!(
                "{key} mismatch; optimizer configuration must not reset"
            )));
        }
    }
    if optimizer.kind_tag() == "adamw" || optimizer.kind_tag() == "adam" {
        let step: u64 = field("optim.step")?
            .parse()
            .map_err(|_| invalid("invalid optimizer step counter"))?;
        if step != cursor.update_step {
            return Err(invalid("optimizer step differs from saved update cursor"));
        }
    } else if optimizer.kind_tag() == "prodigy" {
        for key in ["optim.d", "optim.d_max", "optim.d_numerator"] {
            let value: f64 = field(key)?
                .parse()
                .map_err(|_| invalid(format!("invalid {key}")))?;
            if !value.is_finite() {
                return Err(invalid(format!("nonfinite {key}")));
            }
        }
    }
    let manifest: BTreeMap<String, serde_json::Value> =
        serde_json::from_str(field("tensor_manifest")?)
            .map_err(|e| invalid(format!("tensor manifest: {e}")))?;
    // Parse THIS byte buffer, not a second path read that could race with replacement.
    let tensors = candle_gen::candle_core::safetensors::load_buffer(&bytes, &Device::Cpu)?;
    if manifest.keys().collect::<BTreeSet<_>>() != tensors.keys().collect::<BTreeSet<_>>() {
        return Err(invalid("snapshot tensor set differs from its manifest"));
    }
    for (name, tensor) in &tensors {
        cancel(flag)?;
        if manifest[name] != record(tensor)? {
            return Err(invalid(format!(
                "tensor {name} shape/precision/content digest mismatch"
            )));
        }
        if !name.starts_with("factor/")
            && !name.starts_with("pending/")
            && !name.starts_with("optim.")
        {
            return Err(invalid(format!("undeclared snapshot tensor {name}")));
        }
    }
    let factors = tensors
        .iter()
        .filter_map(|(name, t)| {
            name.strip_prefix("factor/")
                .map(|name| (name.to_owned(), t))
        })
        .collect::<BTreeMap<_, _>>();
    if factors.len() != named.len()
        || named.iter().any(|(name, var)| {
            factors
                .get(name)
                .is_none_or(|t| t.dtype() != DType::F32 || t.dims() != var.dims())
        })
    {
        return Err(invalid("snapshot trainable keys/shapes/precision mismatch"));
    }
    optimizer_layout(&named, &tensors, optimizer.kind_tag())?;
    let pending = tensors
        .iter()
        .filter_map(|(name, t)| {
            name.strip_prefix("pending/")
                .map(|name| (name.to_owned(), t.clone()))
        })
        .collect::<BTreeMap<_, _>>();
    let pending_keys: Vec<String> = serde_json::from_str(field("pending_keys")?)
        .map_err(|e| invalid(format!("pending keys: {e}")))?;
    if pending_keys != pending.keys().cloned().collect::<Vec<_>>() {
        return Err(invalid("saved accumulation gradient set is incomplete"));
    }
    validate_pending(&pending, &cursor, &identity.factors)?;
    let prepared = named
        .iter()
        .map(|(name, var)| Ok((var, factors[name].to_device(var.device())?)))
        .collect::<Result<Vec<_>>>()?;
    cancel(flag)?;
    // Every metadata field, tensor key, shape, dtype and digest is validated first.
    optimizer.restore(&tensors, &metadata)?;
    for (var, tensor) in prepared {
        var.set(&tensor)?;
    }
    let pending = pending
        .into_iter()
        .map(|(name, t)| Ok((name, t.to_device(graph.dit.device())?)))
        .collect::<Result<_>>()?;
    cancel(flag)?;
    Ok(RestoredState { cursor, pending })
}

pub fn export_adapter(
    graph: &TrainingGraph,
    identity: &ResumeIdentity,
    root: &Path,
    name: &str,
    flag: &CancelFlag,
) -> Result<PathBuf> {
    cancel(flag)?;
    validate_identity(graph, identity)?;
    if identity.base_repo != BASE_WEIGHTS_REPO || identity.base_revision != BASE_WEIGHTS_REVISION {
        return Err(invalid(
            "adapter export requires the canonical pinned base compatibility identity",
        ));
    }
    let rank = graph
        .factors
        .first()
        .ok_or_else(|| invalid("no trainable factors"))?
        .down
        .dim(0)?;
    if graph
        .factors
        .iter()
        .any(|factor| factor.down.dim(0).ok() != Some(rank))
    {
        return Err(invalid(
            "mixed per-target ranks cannot be exported as plain global LoRA",
        ));
    }
    let alpha = graph.scale * rank as f64;
    if !alpha.is_finite() || alpha <= 0.0 {
        return Err(invalid("invalid global LoRA alpha"));
    }
    let mut tensors = BTreeMap::new();
    for factor in &graph.factors {
        cancel(flag)?;
        record(factor.down.as_tensor())?;
        record(factor.up.as_tensor())?;
        tensors.insert(
            format!("{}.lora_A.weight", factor.target.diffusers),
            factor.down.as_tensor().to_device(&Device::Cpu)?,
        );
        tensors.insert(
            format!("{}.lora_B.weight", factor.target.diffusers),
            factor.up.as_tensor().to_device(&Device::Cpu)?,
        );
    }
    let mut metadata = HashMap::from([
        ("base_model".into(), identity.base_repo.clone()), ("base_revision".into(), identity.base_revision.clone()),
        ("network_type".into(), "lora".into()), ("rank".into(), rank.to_string()), ("lora_alpha".into(), alpha.to_string()),
        ("scaling_convention".into(), "alpha/rank".into()), ("training_identity".into(), identity.to_metadata().map_err(|e| invalid(e.to_string()))?),
        ("lora_adapter_metadata".into(), serde_json::json!({"peft_type":"LORA", "r":rank, "lora_alpha":alpha, "use_rslora":false, "use_dora":false, "rank_pattern":{}, "alpha_pattern":{}}).to_string()),
    ]);
    if let Some(fixture) = identity.configuration.get("fixture") {
        metadata.insert("fixture".into(), fixture.clone());
    }
    let bytes = safetensors::serialize(tensors.iter(), Some(metadata))
        .map_err(|e| invalid(format!("serialize adapter: {e}")))?;
    cancel(flag)?;
    publish_bytes(root, name, &bytes, flag).map_err(Error::from)
}
