//! Observed local-byte identity, not a Hub metadata/download/authentication claim.
use super::super::{
    Flux3Config, GenerationResources, BASE_AUTOENCODER_SHA256, BASE_DIT_SHA256, BASE_WEIGHTS_REPO,
    BASE_WEIGHTS_REVISION, DIT_WEIGHTS_FILE, UPSTREAM_CODE_REVISION,
};
use super::{
    options::TrainingOptions, preprocess::Geometry, rng::RNG_ALGORITHM, LoraTarget, ResumeIdentity,
};
use gen_core::train::TrainingRequest;
use gen_core::{CancelFlag, Error, Result};
use gen_core_train::resume::request_fingerprint;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Read,
    path::Path,
};
fn invalid(detail: impl std::fmt::Display) -> Error {
    Error::Unsupported(format!("flux3 training identity: {detail}"))
}
/// Public training binds the released base graph, not a user-edited descriptive
/// config that silently omits blocks from the same immutable weight file. Tiny
/// tests construct private prepared sessions and never bypass this public guard.
pub fn validate_release_config(cfg: &Flux3Config) -> Result<()> {
    let d = &cfg.dit;
    if cfg.quantization.is_some()
        || d.flow_map
        || d.hidden != 3072
        || d.heads != 24
        || d.depth != 5
        || d.depth_single != 28
        || d.mlp_ratio != 3.0
        || d.axes_dim != [32, 32, 32, 32]
        || d.theta != 10000.0
        || d.image_in_channels != 128
        || d.image_cond_in_channels != 128
        || d.text_in_channels != 20480
        || d.vec_in_dim.is_some()
        || d.image_rope_scale != 0.5
        || d.out_modes != ["image"]
    {
        return Err(invalid("requires the canonical dense undistilled base graph; distilled/quantized/edited geometry cannot be substituted"));
    }
    if cfg.text_encoder.layers != [4, 8, 12, 16, 20, 24, 28, 32]
        || cfg.text_encoder.pad_multiple != 80
    {
        return Err(invalid(
            "teacher selected layers/padding do not match the frozen base recipe",
        ));
    }
    Ok(())
}

pub const TEXT_SHARD_SHA256: [&str; 2] = [
    "30a01a0556622645a3cce87b655bbbbbc1f170c196099f1b666c93202c3339a9",
    "046296a2a387efb43b0c997d5833c789604d168834f6e0d3064bf7bb13d002a6",
];
pub fn sha256_file(path: &Path, cancel: &CancelFlag) -> Result<String> {
    if cancel.is_cancelled() {
        return Err(Error::Canceled);
    }
    let mut file = std::fs::File::open(path)?;
    let before = file.metadata()?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    let after = file.metadata()?;
    if before.len() != after.len() || before.modified().ok() != after.modified().ok() {
        return Err(invalid(
            "resource changed during identity read; finish installation/copy before training",
        ));
    }
    Ok(format!("{:x}", hash.finalize()))
}
/// Verify actual dense base/AE/teacher weight bytes against the pinned public LFS
/// identities. Other config/tokenizer bytes are observed and bound to continuation.
/// No remote request or credential inspection occurs here.
pub fn observe_pinned_resources(
    resources: &GenerationResources,
    cancel: &CancelFlag,
) -> Result<BTreeMap<String, String>> {
    let mut artifacts = BTreeMap::new();
    for (key, path, expected) in [
        (
            "dit/model.safetensors",
            resources.package_dir.join(DIT_WEIGHTS_FILE),
            BASE_DIT_SHA256,
        ),
        (
            "autoencoder/ae.safetensors",
            resources.autoencoder_file.clone(),
            BASE_AUTOENCODER_SHA256,
        ),
    ] {
        let observed = sha256_file(&path, cancel)?;
        if observed != expected {
            return Err(invalid(format!("{key} bytes do not match the canonical pinned dense base; use the installed revision, not an unverified/quantized checkpoint")));
        }
        artifacts.insert(key.to_owned(), observed);
    }
    artifacts.insert(
        "package/config.json".into(),
        sha256_file(&resources.package_dir.join("config.json"), cancel)?,
    );
    let roots = gen_core::safetensors_shards::snapshot_shard_roots(&resources.text_encoder_dir)?;
    let shards = gen_core::safetensors_shards::resolve_indexed_safetensors_shards(
        &resources.text_encoder_dir,
        "model.safetensors.index.json",
        &roots,
    )?
    .map(|shards| {
        shards
            .into_iter()
            .map(|s| s.loader_path)
            .collect::<Vec<_>>()
    })
    .unwrap_or_else(|| vec![resources.text_encoder_dir.join("model.safetensors")]);
    let mut observed_shards = BTreeSet::new();
    for (index, path) in shards.iter().enumerate() {
        let observed = sha256_file(path, cancel)?;
        observed_shards.insert(observed.clone());
        artifacts.insert(format!("text_encoder/shard{index}"), observed);
    }
    if observed_shards != TEXT_SHARD_SHA256.into_iter().map(str::to_owned).collect() {
        return Err(invalid("teacher shard bytes do not match the pinned Qwen3-VL component; packed/merged/unverified teachers are not this source profile"));
    }
    for name in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
        "chat_template.json",
        "generation_config.json",
    ] {
        let path = resources.text_encoder_dir.join(name);
        if path.is_file() {
            artifacts.insert(format!("text_encoder/{name}"), sha256_file(&path, cancel)?);
        }
    }
    Ok(artifacts)
}

/// Full normalized config is HASHED, not published verbatim (captions/private bags
/// must not leak through adapter metadata). Public scalar controls remain explicit.
pub fn build(
    req: &TrainingRequest,
    cfg: &Flux3Config,
    opts: &TrainingOptions,
    artifacts: BTreeMap<String, String>,
    targets: &[LoraTarget],
    geometry: &[Geometry],
    backend: &str,
) -> Result<ResumeIdentity> {
    let mut normalized = req.config.clone();
    normalized.resume = false;
    let config_digest = format!(
        "{:x}",
        Sha256::digest(format!("{normalized:?}|trigger_words={:?}", req.trigger_words).as_bytes())
    );
    let geometry=geometry.iter().map(|g|serde_json::json!({"resize":[g.resized_height,g.resized_width],"crop":[g.crop_y,g.crop_x],"flip":g.flip,"reference_prepass":g.reference_prepass})).collect::<Vec<_>>();
    let recipe = serde_json::json!({
        "native_algorithm":"flux3-source-training-v1", "training_reference":super::TRAINING_REFERENCE_REVISION,
        "forward_reference":UPSTREAM_CODE_REVISION,
        "intake":"S3 dataset-oriented RGB/RGBA boundary; datasets4.4.1 source audit, not publisher dependency pin/universal raw PIL",
        "reference_prepass":"source >1MP Lanczos; effective16 pure center crop; conditional inverse and truncating byte roundtrip",
        "pair_transform":"target-derived ties-even cover resize, PIL bilinear, shared crop/flip",
        "geometry":geometry,"precision":opts.precision.name(),
        "vae":"selected precision frozen encoder; deterministic mean; patch2; ORIGINAL F32 BN stats exactly once; normalized F32 cache",
        "teacher":"selected compute/output; Module.to frequency-round then F32 phase/trig; text-only",
        "max_sequence_length":opts.max_sequence_length,"pad_multiple":cfg.text_encoder.pad_multiple,"layers":cfg.text_encoder.layers,
        "batch_window":"dataset-wide largest window for batch>1, per-caption for batch1; empty prompt matches dropped window",
        "sampler":"within-bucket membership shuffle once, per-bucket drop-last; epoch batch-order shuffle; configured-N loss scale before backward incl epoch/final tails",
        "noise":"native standard normal F32, NOT inference bf16-rounded noise",
        "rng":RNG_ALGORITHM,"source_options":format!("{opts:?}"),
    });
    let preprocessing = format!("{:x}", Sha256::digest(recipe.to_string().as_bytes()));
    let mut factors = BTreeMap::new();
    for target in targets {
        factors.insert(
            format!("{}.lora_A.weight", target.native),
            vec![req.config.rank as usize, target.input],
        );
        factors.insert(
            format!("{}.lora_B.weight", target.native),
            vec![target.output, req.config.rank as usize],
        );
    }
    let identity = ResumeIdentity {
        base_repo: BASE_WEIGHTS_REPO.into(),
        base_revision: BASE_WEIGHTS_REVISION.into(),
        artifacts,
        dataset: request_fingerprint(req)?,
        preprocessing: preprocessing.clone(),
        factors,
        configuration: BTreeMap::from([
            ("optimizer".into(), "adamw".into()),
            (
                "learning_rate".into(),
                format!("{:?}", req.config.learning_rate),
            ),
            ("rank".into(), req.config.rank.to_string()),
            ("alpha".into(), format!("{:?}", req.config.alpha as f64)),
            ("scaling".into(), "alpha/rank".into()),
            ("precision".into(), opts.precision.name().into()),
            (
                "backend".into(),
                format!(
                    "{backend};os={};arch={}",
                    std::env::consts::OS,
                    std::env::consts::ARCH
                ),
            ),
            (
                "scheduler".into(),
                format!(
                    "{:?};warmup={};cycles={};power={}",
                    opts.scheduler, req.config.lr_warmup_steps, opts.cycles, opts.power
                ),
            ),
            (
                "accumulation".into(),
                req.config.gradient_accumulation.to_string(),
            ),
            (
                "accumulation_scaling".into(),
                "loss/configured_N_before_backward; epoch-tail flush".into(),
            ),
            ("rng".into(), RNG_ALGORITHM.into()),
            ("config_sha256".into(), config_digest),
            ("recipe_sha256".into(), preprocessing),
        ]),
    };
    identity.validate()?;
    Ok(identity)
}
