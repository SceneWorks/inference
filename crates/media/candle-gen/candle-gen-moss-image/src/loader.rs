//! Exact prepared asset identities, authenticated before any backend tensor allocation.
use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::{
    self, moss_image::*, weightsmeta, LoadSpec, PinnedWeightsFile, WeightsSource,
};
use candle_gen::weights::Weights;
use candle_gen::{CandleError, Result};
use candle_gen_flux2::vae::{Flux2Vae, Flux2VaeDecoderConfig};
use candle_llm::models::qwen35::{Qwen35Config, Qwen35Model};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

#[derive(Clone)]
pub(crate) struct Asset {
    pub pin: PinnedWeightsFile,
    pub role: MossImageAssetRole,
    pub repo: &'static str,
    pub revision: &'static str,
    pub relative: String,
}
impl Asset {
    pub(crate) fn new(
        spec: &LoadSpec,
        path: PathBuf,
        role: MossImageAssetRole,
        repo: &'static str,
        revision: &'static str,
        relative: &str,
    ) -> gen_core::Result<Self> {
        let path = std::path::absolute(path)?;
        let pin = if spec.prepared_file_pins().is_prepared() {
            spec.validate_prepared_file_pins()?;
            spec.prepared_file_pins()
                .get(&path)
                .cloned()
                .ok_or_else(|| {
                    gen_core::Error::Unsupported(format!(
                        "MOSS prepared source missing {}",
                        path.display()
                    ))
                })?
        } else {
            PinnedWeightsFile::pin(&path)?
        };
        if pin.loader_path() != path {
            return Err(gen_core::Error::Unsupported(
                "MOSS prepared path mismatch".into(),
            ));
        }
        verify_moss_image_asset(&pin, repo, revision, relative)?;
        Ok(Self {
            pin,
            role,
            repo,
            revision,
            relative: relative.into(),
        })
    }
    pub fn receipt(&self) -> gen_core::Result<MossImageAssetReceipt> {
        MossImageAssetReceipt::from_pin(
            self.role,
            self.repo,
            self.revision,
            &self.relative,
            &self.pin,
        )
    }
    pub fn json(&self) -> gen_core::Result<serde_json::Value> {
        serde_json::from_slice(&self.authenticated_bytes()?)
            .map_err(|e| gen_core::Error::Msg(e.to_string()))
    }
    /// Authenticate the exact owned bytes subsequently parsed/materialized. A filesystem swap
    /// wholly within the pin callback cannot substitute different tensor/config/tokenizer bytes.
    pub fn authenticated_bytes(&self) -> gen_core::Result<Vec<u8>> {
        self.pin.read_unchanged(|p| {
            let bytes = std::fs::read(p)?;
            self.verify_owned_bytes(&bytes)?;
            Ok(bytes)
        })
    }
    pub(crate) fn verify_owned_bytes(&self, bytes: &[u8]) -> gen_core::Result<()> {
        let expected =
            moss_image_asset(self.repo, self.revision, &self.relative).ok_or_else(|| {
                gen_core::Error::Unsupported("MOSS file outside frozen manifest".into())
            })?;
        let digest = Sha256::digest(bytes);
        if bytes.len() as u64 != expected.bytes
            || format!("{digest:x}") != expected.sha256
            || digest[..] != self.pin.content_sha256()[..]
        {
            return Err(gen_core::Error::Unsupported(format!(
                "MOSS actual consumed bytes differ from authenticated asset {}",
                self.relative
            )));
        }
        Ok(())
    }
}
#[derive(Clone)]
pub(crate) struct Sources {
    pub ordinary: Vec<Asset>,
    pub mixed: Option<Vec<Asset>>,
    pub full: Option<Vec<Asset>>,
    pub config: MossImageConfig,
    pub text_config: Qwen35Config,
    pub tokenizer: Asset,
    pub spec: LoadSpec,
    pub headers: BTreeMap<[u8; 32], Vec<weightsmeta::SafetensorsTensorHeader>>,
}
fn root(source: &WeightsSource) -> gen_core::Result<&Path> {
    match source {
        WeightsSource::Dir(p) => Ok(p),
        WeightsSource::File(_) => Err(gen_core::Error::Unsupported(
            "MOSS requires a frozen snapshot directory".into(),
        )),
    }
}
fn optional_file(
    spec: &LoadSpec,
    key: &str,
    file: &str,
    role: MossImageAssetRole,
) -> gen_core::Result<Option<Asset>> {
    spec.components
        .get(key)
        .map(|s| {
            Asset::new(
                spec,
                match s {
                    WeightsSource::File(p) => p.clone(),
                    WeightsSource::Dir(p) => p.join(file),
                },
                role,
                DMD_REPO,
                DMD_REVISION,
                file,
            )
        })
        .transpose()
}
impl Sources {
    pub fn new(variant: MossImageVariant, spec: &LoadSpec) -> gen_core::Result<Self> {
        if spec.precision != gen_core::Precision::Bf16
            || spec.quantize.is_some()
            || !spec.adapters.is_empty()
            || spec.control.is_some()
            || !spec.extra_controls.is_empty()
            || spec.ip_adapter.is_some()
            || spec.pid.is_some()
            || spec.identity.is_some()
            || spec.text_encoder.is_some()
        {
            return Err(gen_core::Error::Unsupported(
                "MOSS supports frozen dense checkpoints and only explicit published DMD components"
                    .into(),
            ));
        }
        let allowed = [
            MOSS_SHARED_SNAPSHOT_COMPONENT,
            DMD_LORA_COMPONENT,
            DMD_TIME_COMPONENT,
            DMD_DATASPACE_COMPONENT,
            DMD_TRANSFORMER_COMPONENT,
        ];
        if spec
            .components
            .keys()
            .any(|k| !allowed.contains(&k.as_str()))
        {
            return Err(gen_core::Error::Unsupported(
                "unknown MOSS component".into(),
            ));
        }
        if variant == MossImageVariant::Base
            && spec.components.keys().any(|k| k.starts_with("moss-dmd-"))
        {
            return Err(gen_core::Error::Unsupported(
                "MOSS DMD components require the exact SFT parent, not Base".into(),
            ));
        }
        let base = root(&spec.weights)?;
        let shared = match spec.components.get(MOSS_SHARED_SNAPSHOT_COMPONENT) {
            Some(s) => root(s)?,
            None if variant == MossImageVariant::Sft => base,
            None => {
                return Err(gen_core::Error::Unsupported(
                    "MOSS Base requires moss-shared-snapshot pinned to frozen SFT".into(),
                ))
            }
        };
        let mut ordinary = Vec::new();
        for &relative in SNAPSHOT_FILES {
            if relative == "LICENSE" {
                continue;
            }
            let tr = relative.starts_with("transformer/");
            let role = if tr {
                MossImageAssetRole::Transformer
            } else if relative.starts_with("text_encoder/") {
                MossImageAssetRole::TextEncoder
            } else if relative.starts_with("tokenizer/") {
                MossImageAssetRole::Tokenizer
            } else if relative.starts_with("scheduler/") {
                MossImageAssetRole::Scheduler
            } else {
                MossImageAssetRole::Vae
            };
            ordinary.push(Asset::new(
                spec,
                if tr {
                    base.join(relative)
                } else {
                    shared.join(relative)
                },
                role,
                if tr { variant.repository() } else { SFT_REPO },
                if tr { variant.revision() } else { SFT_REVISION },
                relative,
            )?);
        }
        let config = MossImageConfig::from_json(
            &ordinary
                .iter()
                .find(|a| a.relative == "transformer/config.json")
                .expect("static path")
                .json()?,
        )?;
        let text_config = Qwen35Config::from_json(
            &ordinary
                .iter()
                .find(|a| a.relative == "text_encoder/config.json")
                .expect("static path")
                .json()?,
        )
        .map_err(|e| gen_core::Error::Msg(e.to_string()))?;
        if text_config.hidden_size != 4096 || text_config.moe.is_some() {
            return Err(gen_core::Error::Unsupported(
                "MOSS requires frozen dense Qwen3.5-9B text tower".into(),
            ));
        }
        MossImageSchedulerConfig::from_json(
            &ordinary
                .iter()
                .find(|a| a.role == MossImageAssetRole::Scheduler)
                .expect("static path")
                .json()?,
        )?;
        let vae = ordinary
            .iter()
            .find(|a| a.relative == "vae/config.json")
            .expect("static path")
            .json()?;
        if vae["decoder_block_out_channels"] != serde_json::json!([96, 192, 384, 384])
            || vae["latent_channels"] != 32
            || vae["batch_norm_eps"].as_f64() != Some(0.0001)
        {
            return Err(gen_core::Error::Unsupported(
                "MOSS VAE configuration mismatch".into(),
            ));
        }
        // Validate each index-selected shard set under the caller's supplied repository roots.
        for (dir, index, role) in [
            (
                base.join("transformer"),
                "diffusion_pytorch_model.safetensors.index.json",
                MossImageAssetRole::Transformer,
            ),
            (
                shared.join("text_encoder"),
                "model.safetensors.index.json",
                MossImageAssetRole::TextEncoder,
            ),
        ] {
            validate_shards(&dir, index, &ordinary, role)?;
        }
        let mut headers = validate_source_dtypes(&ordinary)?;
        let lora = optional_file(
            spec,
            DMD_LORA_COMPONENT,
            "lora.safetensors",
            MossImageAssetRole::DmdLora,
        )?;
        let time = optional_file(
            spec,
            DMD_TIME_COMPONENT,
            "time_embedding.safetensors",
            MossImageAssetRole::DmdTime,
        )?;
        let data = optional_file(
            spec,
            DMD_DATASPACE_COMPONENT,
            "dataspace.safetensors",
            MossImageAssetRole::DmdDataspace,
        )?;
        if let Some(data) = &data {
            validate_dataspace(data)?;
        }
        let mixed = match (lora, time, data.clone()) {
            (None, None, _) => None,
            (Some(l), Some(t), Some(d)) => {
                validate_fp32(&l, &dmd_lora_shapes())?;
                validate_fp32(&t, &dmd_replacement_shapes())?;
                validate_dataspace(&d)?;
                Some(vec![l, t, d])
            }
            _ => {
                return Err(gen_core::Error::Unsupported(
                    "MOSS mixed DMD requires lora, time replacements and dataspace together".into(),
                ))
            }
        };
        let full = if let Some(source) = spec.components.get(DMD_TRANSFORMER_COMPONENT) {
            let p = root(source)?;
            let p = if p.file_name().is_some_and(|n| n == "transformer") {
                p.to_path_buf()
            } else {
                p.join("transformer")
            };
            let mut assets = Vec::new();
            for name in [
                "config.json",
                "diffusion_pytorch_model.safetensors.index.json",
                "diffusion_pytorch_model-00001-of-00004.safetensors",
                "diffusion_pytorch_model-00002-of-00004.safetensors",
                "diffusion_pytorch_model-00003-of-00004.safetensors",
                "diffusion_pytorch_model-00004-of-00004.safetensors",
            ] {
                assets.push(Asset::new(
                    spec,
                    p.join(name),
                    MossImageAssetRole::Transformer,
                    DMD_REPO,
                    DMD_REVISION,
                    &format!("transformer/{name}"),
                )?);
            }
            MossImageConfig::from_json(&assets[0].json()?)?;
            validate_shards(
                &p,
                "diffusion_pytorch_model.safetensors.index.json",
                &assets,
                MossImageAssetRole::Transformer,
            )?;
            assets.push(data.ok_or_else(|| {
                gen_core::Error::Unsupported("MOSS full FP32 DMD requires dataspace".into())
            })?);
            headers.extend(validate_source_dtypes(&assets)?);
            Some(assets)
        } else {
            None
        };
        if let Some(assets) = &mixed {
            headers.extend(validate_source_dtypes(assets)?);
        }
        let tokenizer = ordinary
            .iter()
            .find(|a| a.relative == "tokenizer/tokenizer.json")
            .expect("static path")
            .clone();
        Ok(Self {
            ordinary,
            mixed,
            full,
            config,
            text_config,
            tokenizer,
            spec: spec.clone(),
            headers,
        })
    }
    pub fn assets(&self, mode: MossImageAcceleration) -> gen_core::Result<Vec<Asset>> {
        let mut a = self.ordinary.clone();
        match mode {
            MossImageAcceleration::Off=>(),
            MossImageAcceleration::Dmd(MossDmdPrecision::UnmergedBf16)=>a.extend(self.mixed.clone().ok_or_else(||gen_core::Error::Unsupported("MOSS unmerged_bf16 needs prepared moss-dmd-lora/time/dataspace; provision frozen DMD assets".into()))?),
            MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32)=>{a.retain(|a|a.role!=MossImageAssetRole::Transformer);a.extend(self.full.clone().ok_or_else(||gen_core::Error::Unsupported("MOSS full_fp32 needs prepared published moss-dmd-transformer/dataspace; provision frozen full FP32 DMD assets".into()))?);}
        }
        for asset in &a {
            asset.pin.verify_unchanged()?;
        }
        Ok(a)
    }
    pub fn text(&self, device: &Device) -> Result<Qwen35Model> {
        let dtype = if device.is_cpu() {
            DType::F32
        } else {
            DType::BF16
        };
        let w = load_weights(
            &self.ordinary,
            MossImageAssetRole::TextEncoder,
            device,
            dtype,
            &["language_model."],
        )?;
        let map = w
            .keys()
            .map(|k| Ok((k.clone(), w.require(k)?)))
            .collect::<Result<HashMap<_, _>>>()?;
        Qwen35Model::from_weights_headless(
            &candle_llm::primitives::Weights::from_map(map, device.clone()),
            MOSS_TEXT_ENCODER_PREFIX,
            self.text_config.clone(),
        )
        .map_err(|e| CandleError::Msg(e.to_string()))
    }
    pub fn vae(&self, device: &Device) -> Result<Flux2Vae> {
        let weights = load_weights(
            &self.ordinary,
            MossImageAssetRole::Vae,
            device,
            DType::F32,
            &[
                "decoder.",
                "post_quant_conv.",
                "bn.running_mean",
                "bn.running_var",
            ],
        )?;
        let map = weights
            .keys()
            .map(|k| Ok((k.clone(), weights.require(k)?)))
            .collect::<Result<HashMap<_, _>>>()?;
        Ok(Flux2Vae::new_with_decoder_config(
            VarBuilder::from_tensors(map, DType::F32, device),
            &Flux2VaeDecoderConfig {
                block_out_channels: MOSS_VAE_DECODER_CHANNELS,
            },
        )?)
    }
}
fn validate_shards(
    dir: &Path,
    index: &str,
    assets: &[Asset],
    role: MossImageAssetRole,
) -> gen_core::Result<()> {
    let paths = gen_core::safetensors_shards::resolve_indexed_safetensors_shards(
        dir,
        index,
        &gen_core::safetensors_shards::snapshot_shard_roots(dir)?,
    )?
    .ok_or_else(|| gen_core::Error::Unsupported("MOSS missing frozen shard index".into()))?;
    let expected: BTreeSet<_> = assets
        .iter()
        .filter(|a| a.role == role && a.relative.ends_with(".safetensors"))
        .map(|a| a.pin.canonical_target_path().to_path_buf())
        .collect();
    let actual: BTreeSet<_> = paths.into_iter().map(|p| p.canonical_path).collect();
    if expected != actual {
        return Err(gen_core::Error::Unsupported(
            "MOSS index-selected shard set differs from frozen package".into(),
        ));
    }
    Ok(())
}
fn validate_source_dtypes(
    assets: &[Asset],
) -> gen_core::Result<BTreeMap<[u8; 32], Vec<weightsmeta::SafetensorsTensorHeader>>> {
    let mut result = BTreeMap::new();
    for a in assets
        .iter()
        .filter(|a| a.relative.ends_with(".safetensors"))
    {
        let expected = if a.repo == DMD_REPO || a.role == MossImageAssetRole::Vae {
            weightsmeta::Dtype::F32
        } else {
            weightsmeta::Dtype::BF16
        };
        let bytes = a.authenticated_bytes()?;
        let file = candle_gen::candle_core::safetensors::SliceSafetensors::new(&bytes)
            .map_err(|e| gen_core::Error::Msg(e.to_string()))?;
        let headers: Vec<_> = file
            .tensors()
            .into_iter()
            .map(|(name, t)| {
                Ok(weightsmeta::SafetensorsTensorHeader {
                    name,
                    dtype: weightsmeta::Dtype::parse(&format!("{:?}", t.dtype())).ok_or_else(
                        || gen_core::Error::Unsupported("MOSS unknown source tensor dtype".into()),
                    )?,
                    shape: t.shape().to_vec(),
                    data_bytes: t.data().len() as u64,
                })
            })
            .collect::<gen_core::Result<_>>()?;
        if headers.iter().any(|h| {
            h.dtype != expected
                && !h.name.ends_with("num_batches_tracked")
                && h.name != "rope.bands"
        }) {
            return Err(gen_core::Error::Unsupported(format!(
                "MOSS source dtype differs from frozen policy {}",
                a.relative
            )));
        }
        result.insert(*a.pin.content_sha256(), headers);
    }
    Ok(result)
}
fn validate_fp32(a: &Asset, expected: &BTreeMap<String, Vec<usize>>) -> gen_core::Result<()> {
    validate_fp32_bytes(&a.authenticated_bytes()?, expected)
}

pub(crate) fn validate_fp32_bytes(
    bytes: &[u8],
    expected: &BTreeMap<String, Vec<usize>>,
) -> gen_core::Result<()> {
    let file = candle_gen::candle_core::safetensors::SliceSafetensors::new(bytes)
        .map_err(|e| gen_core::Error::Msg(e.to_string()))?;
    let tensors = file.tensors();
    if tensors.len() != expected.len() {
        return Err(gen_core::Error::Unsupported(
            "MOSS DMD exact tensor coverage mismatch".into(),
        ));
    }
    for (name, tensor) in tensors {
        if format!("{:?}", tensor.dtype()) != "F32"
            || expected.get(&name).map(Vec::as_slice) != Some(tensor.shape())
        {
            return Err(gen_core::Error::Unsupported(format!(
                "MOSS DMD dtype/shape/coverage mismatch {name}"
            )));
        }
        for bytes in tensor.data().chunks_exact(4) {
            let value = f32::from_le_bytes(bytes.try_into().expect("four bytes"));
            if !value.is_finite() || (name.ends_with(".alpha") && value != 128.0) {
                return Err(gen_core::Error::Unsupported(format!(
                    "MOSS nonfinite tensor or invalid rank-128 alpha {name}"
                )));
            }
        }
    }
    Ok(())
}

fn validate_dataspace(a: &Asset) -> gen_core::Result<()> {
    validate_fp32(
        a,
        &[
            ("latent_mix.Q".into(), vec![128, 128]),
            ("step_codes".into(), vec![8]),
        ]
        .into_iter()
        .collect(),
    )
}
pub(crate) fn load_weights(
    assets: &[Asset],
    role: MossImageAssetRole,
    device: &Device,
    dtype: DType,
    prefixes: &[&str],
) -> Result<Weights> {
    let mut map = HashMap::new();
    for a in assets
        .iter()
        .filter(|a| a.role == role && a.relative.ends_with(".safetensors"))
    {
        let bytes = a.authenticated_bytes()?;
        let safetensors = candle_gen::candle_core::safetensors::SliceSafetensors::new(&bytes)?;
        for (k, _) in safetensors.tensors() {
            if !prefixes.iter().any(|p| k.starts_with(p)) {
                continue;
            }
            let tensor = safetensors.load(&k, device)?;
            let tensor = if matches!(
                tensor.dtype(),
                DType::F16 | DType::BF16 | DType::F32 | DType::F64
            ) {
                tensor.to_dtype(dtype)?
            } else {
                tensor
            };
            if map.insert(k.clone(), tensor).is_some() {
                return Err(CandleError::Msg(format!("MOSS duplicate tensor {k}")));
            }
        }
    }
    Ok(Weights::from_map(map))
}
pub(crate) fn tensor_f32(t: &Tensor) -> Result<Vec<f32>> {
    Ok(t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?)
}
