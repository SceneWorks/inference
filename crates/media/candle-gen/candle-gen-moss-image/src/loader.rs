//! Exact prepared asset identities, authenticated before any backend tensor allocation.
use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::{
    self, moss_image::*, weightsmeta, CancelFlag, LoadSpec, PinnedWeightsFile, WeightsSource,
};
use candle_gen::weights::Weights;
use candle_gen::{CandleError, Result};
use candle_gen_flux2::vae::{Flux2Vae, Flux2VaeDecoderConfig};
use candle_llm::models::qwen35::{Qwen35Config, Qwen35Model};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

#[derive(Clone)]
pub(crate) struct Asset {
    pub pin: PinnedWeightsFile,
    pub role: MossImageAssetRole,
    pub repo: &'static str,
    pub revision: &'static str,
    pub relative: String,
    #[cfg(feature = "test-support")]
    pub fixture_digest: Option<[u8; 32]>,
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
            if !spec.prepared_file_pins().is_finalized() {
                return Err(gen_core::Error::Unsupported(
                    "MOSS unfinalized prepared source".into(),
                ));
            }
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
            #[cfg(feature = "test-support")]
            fixture_digest: None,
            pin,
            role,
            repo,
            revision,
            relative: relative.into(),
        })
    }
    pub fn receipt(&self) -> gen_core::Result<MossImageAssetReceipt> {
        #[cfg(feature = "test-support")]
        if self.fixture_digest.is_some() {
            self.pin.verify_unchanged()?;
            return Ok(MossImageAssetReceipt {
                role: self.role,
                repository: self.repo.into(),
                revision: self.revision.into(),
                relative_path: self.relative.clone(),
                sha256: format!("{:x}", Sha256::digest(self.authenticated_bytes()?)),
            });
        }
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
        #[cfg(feature = "test-support")]
        if let Some(expected) = self.fixture_digest {
            let actual = Sha256::digest(bytes);
            if actual[..] != expected[..] || actual[..] != self.pin.content_sha256()[..] {
                return Err(gen_core::Error::Unsupported(
                    "synthetic fixture consumed bytes changed".into(),
                ));
            }
            return Ok(());
        }
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
    pub variant: MossImageVariant,
    pub adapter_pins: Vec<PinnedWeightsFile>,
    pub selection: Option<MossImageAcceleration>,
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
fn transformer_config(
    selection: Option<MossImageAcceleration>,
    spec: &LoadSpec,
    ordinary: &[Asset],
) -> gen_core::Result<MossImageConfig> {
    Ok(if selection.is_none() {
        // No DiT is allocated at construction. Every explicit request authenticates its
        // selected published config before renderer construction (sc-26325 / F-279).
        MossImageConfig::default()
    } else if selection == Some(MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32)) {
        let source = spec.components.get(DMD_TRANSFORMER_COMPONENT).ok_or_else(||gen_core::Error::Unsupported("MOSS full_fp32 needs prepared published moss-dmd-transformer/dataspace; provision frozen full FP32 DMD assets".into()))?;
        let dir = root(source)?;
        let path = if dir.file_name().is_some_and(|n| n == "transformer") {
            dir.join("config.json")
        } else {
            dir.join("transformer/config.json")
        };
        MossImageConfig::from_json(
            &Asset::new(
                spec,
                path,
                MossImageAssetRole::Transformer,
                DMD_REPO,
                DMD_REVISION,
                "transformer/config.json",
            )?
            .json()?,
        )?
    } else {
        MossImageConfig::from_json(
            &ordinary
                .iter()
                .find(|a| a.relative == "transformer/config.json")
                .expect("static path")
                .json()?,
        )?
    })
}
fn snapshot_files(selection: Option<MossImageAcceleration>) -> impl Iterator<Item = &'static str> {
    SNAPSHOT_FILES.iter().copied().filter(move |relative| {
        *relative != "LICENSE"
            && (!relative.starts_with("transformer/")
                || selection
                    .is_some_and(|m| m != MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32)))
    })
}
impl Sources {
    pub(crate) fn is_fixture(&self) -> bool {
        #[cfg(feature = "test-support")]
        {
            return !self.ordinary.is_empty()
                && self
                    .ordinary
                    .iter()
                    .all(|asset| asset.fixture_digest.is_some());
        }
        #[cfg(not(feature = "test-support"))]
        {
            false
        }
    }
    pub(crate) fn source_identity(&self, variant: MossImageVariant) -> String {
        if self.is_fixture() {
            let mut hash = Sha256::new();
            for asset in &self.ordinary {
                hash.update(asset.relative.as_bytes());
                hash.update(asset.pin.content_sha256());
            }
            format!(
                "synthetic-moss-full-pipeline-v1:{}:{:x}",
                variant.provider_id(),
                hash.finalize()
            )
        } else {
            format!(
                "{}@{};shared@{}",
                variant.repository(),
                variant.revision(),
                SFT_REVISION
            )
        }
    }
    pub fn new(variant: MossImageVariant, spec: &LoadSpec) -> gen_core::Result<Self> {
        Self::prepare(variant, spec, None, None)
    }
    pub fn selected(
        variant: MossImageVariant,
        spec: &LoadSpec,
        mode: MossImageAcceleration,
    ) -> gen_core::Result<Self> {
        Self::prepare(variant, spec, Some(mode), None)
    }
    pub fn selected_with_cancel(
        variant: MossImageVariant,
        spec: &LoadSpec,
        mode: MossImageAcceleration,
        cancel: &CancelFlag,
    ) -> gen_core::Result<Self> {
        Self::prepare(variant, spec, Some(mode), Some(cancel))
    }
    fn prepare(
        variant: MossImageVariant,
        spec: &LoadSpec,
        selection: Option<MossImageAcceleration>,
        cancel: Option<&CancelFlag>,
    ) -> gen_core::Result<Self> {
        preparation_check(cancel)?;
        let shared_only = selection.is_none();
        let mode = selection.unwrap_or(MossImageAcceleration::Off);
        let full_fp32 = mode == MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32);
        let unmerged = mode == MossImageAcceleration::Dmd(MossDmdPrecision::UnmergedBf16);
        let required: &[&str] = match mode {
            MossImageAcceleration::Off => &[],
            MossImageAcceleration::Dmd(MossDmdPrecision::UnmergedBf16) => &[
                DMD_LORA_COMPONENT,
                DMD_TIME_COMPONENT,
                DMD_DATASPACE_COMPONENT,
            ],
            MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32) => {
                &[DMD_TRANSFORMER_COMPONENT, DMD_DATASPACE_COMPONENT]
            }
        };
        if required.iter().any(|k| !spec.components.contains_key(*k)) {
            return Err(gen_core::Error::Unsupported("MOSS selected DMD closure missing; provision frozen DMD assets for explicit request".into()));
        }
        if variant == MossImageVariant::Base && mode != MossImageAcceleration::Off {
            return Err(gen_core::Error::Unsupported("MOSS DMD requires SFT".into()));
        }
        if spec.precision != gen_core::Precision::Bf16
            || spec.quantize.is_some()
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
        for relative in snapshot_files(selection) {
            preparation_check(cancel)?;
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
        let config = transformer_config(selection, spec, &ordinary)?;
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
            if !((shared_only || full_fp32) && role == MossImageAssetRole::Transformer) {
                validate_shards(&dir, index, &ordinary, role)?;
            }
        }
        let mut headers = validate_source_dtypes(&ordinary, cancel)?;
        let lora = if unmerged {
            optional_file(
                spec,
                DMD_LORA_COMPONENT,
                "lora.safetensors",
                MossImageAssetRole::DmdLora,
            )?
        } else {
            None
        };
        let time = if unmerged {
            optional_file(
                spec,
                DMD_TIME_COMPONENT,
                "time_embedding.safetensors",
                MossImageAssetRole::DmdTime,
            )?
        } else {
            None
        };
        let data = if mode != MossImageAcceleration::Off {
            optional_file(
                spec,
                DMD_DATASPACE_COMPONENT,
                "dataspace.safetensors",
                MossImageAssetRole::DmdDataspace,
            )?
        } else {
            None
        };
        if let Some(data) = &data {
            validate_dataspace(data, cancel)?;
        }
        let mixed = match (lora, time, data.clone()) {
            (None, None, _) => None,
            (Some(l), Some(t), Some(d)) => {
                validate_fp32(&l, &dmd_lora_shapes(), cancel)?;
                validate_fp32(&t, &dmd_replacement_shapes(), cancel)?;
                validate_dataspace(&d, cancel)?;
                Some(vec![l, t, d])
            }
            _ => {
                return Err(gen_core::Error::Unsupported(
                    "MOSS mixed DMD requires lora, time replacements and dataspace together".into(),
                ))
            }
        };
        let full = if let Some(source) = spec
            .components
            .get(DMD_TRANSFORMER_COMPONENT)
            .filter(|_| full_fp32)
        {
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
            headers.extend(validate_source_dtypes(&assets, cancel)?);
            Some(assets)
        } else {
            None
        };
        if let Some(assets) = &mixed {
            headers.extend(validate_source_dtypes(assets, cancel)?);
        }
        let tokenizer = ordinary
            .iter()
            .find(|a| a.relative == "tokenizer/tokenizer.json")
            .expect("static path")
            .clone();
        let adapter_pins = spec
            .adapters
            .iter()
            .map(|a| spec.file_pin_for(&a.path))
            .collect::<gen_core::Result<Vec<_>>>()?;
        Ok(Self {
            variant,
            adapter_pins,
            selection,
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
    pub fn ensure_unchanged(&self) -> gen_core::Result<()> {
        // Optional DMD closures are authenticated only when selected by assets(mode).
        for a in &self.ordinary {
            a.pin.verify_unchanged()?;
        }
        for pin in &self.adapter_pins {
            pin.verify_unchanged()?;
        }
        Ok(())
    }
    pub fn training_vae_cancellable(
        &self,
        device: &Device,
        cancel: &CancelFlag,
    ) -> Result<Flux2Vae> {
        let weights = load_weights_cancellable(
            &self.ordinary,
            MossImageAssetRole::Vae,
            device,
            DType::F32,
            &[
                "encoder.",
                "decoder.",
                "quant_conv.",
                "post_quant_conv.",
                "bn.running_mean",
                "bn.running_var",
            ],
            cancel,
        )?;
        let map = weights
            .keys()
            .map(|k| Ok((k.clone(), weights.require(k)?)))
            .collect::<Result<HashMap<_, _>>>()?;
        #[cfg(feature = "test-support")]
        if self.is_fixture() {
            return Ok(Flux2Vae::from_fixture_widths(
                VarBuilder::from_tensors(map, DType::F32, device),
                [32; 4],
                &Flux2VaeDecoderConfig {
                    block_out_channels: [32; 4],
                },
            )?);
        }
        Ok(Flux2Vae::new_with_encoder_and_decoder_config(
            VarBuilder::from_tensors(map, DType::F32, device),
            &Flux2VaeDecoderConfig {
                block_out_channels: if self.is_fixture() {
                    [32; 4]
                } else {
                    MOSS_VAE_DECODER_CHANNELS
                },
            },
        )?)
    }
    pub fn assets(&self, mode: MossImageAcceleration) -> gen_core::Result<Vec<Asset>> {
        self.ensure_unchanged()?;
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
        self.text_cancellable(device, &CancelFlag::new())
    }
    pub fn text_cancellable(&self, device: &Device, cancel: &CancelFlag) -> Result<Qwen35Model> {
        let dtype = if device.is_cpu() {
            DType::F32
        } else {
            DType::BF16
        };
        let w = load_weights_cancellable(
            &self.ordinary,
            MossImageAssetRole::TextEncoder,
            device,
            dtype,
            &["language_model."],
            cancel,
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
                block_out_channels: if self.is_fixture() {
                    [32; 4]
                } else {
                    MOSS_VAE_DECODER_CHANNELS
                },
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
type HeaderKey = (&'static str, &'static str, &'static str, u64, [u8; 32]);
type HeaderFacts = Vec<weightsmeta::SafetensorsTensorHeader>;
type HeaderCache = Mutex<BTreeMap<HeaderKey, HeaderFacts>>;
type Fp32Proofs = BTreeMap<HeaderKey, BTreeMap<String, Vec<usize>>>;
fn preparation_check(cancel: Option<&CancelFlag>) -> gen_core::Result<()> {
    if cancel.is_some_and(|c| c.is_cancelled()) {
        Err(gen_core::Error::Canceled)
    } else {
        Ok(())
    }
}
fn cached_headers(
    cache: &HeaderCache,
    key: HeaderKey,
    pin: &PinnedWeightsFile,
    cancel: Option<&CancelFlag>,
    read: impl FnOnce() -> gen_core::Result<HeaderFacts>,
) -> gen_core::Result<HeaderFacts> {
    preparation_check(cancel)?;
    pin.verify_unchanged()?;
    let cached = cache
        .lock()
        .map_err(|_| gen_core::Error::Msg("MOSS header cache poisoned".into()))?
        .get(&key)
        .cloned();
    if let Some(headers) = cached {
        pin.verify_unchanged()?;
        preparation_check(cancel)?;
        return Ok(headers);
    }
    let headers = read()?;
    pin.verify_unchanged()?;
    preparation_check(cancel)?;
    cache
        .lock()
        .map_err(|_| gen_core::Error::Msg("MOSS header cache poisoned".into()))?
        .insert(key, headers.clone());
    Ok(headers)
}
struct HashReader<'a, R> {
    inner: R,
    hash: Sha256,
    bytes: u64,
    cancel: Option<&'a CancelFlag>,
}
impl<R: Read> Read for HashReader<'_, R> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        if self.cancel.is_some_and(|c| c.is_cancelled()) {
            return Err(std::io::Error::other("MOSS preparation canceled"));
        }
        let limit = buffer.len().min(64 * 1024);
        let n = self.inner.read(&mut buffer[..limit])?;
        self.hash.update(&buffer[..n]);
        self.bytes += n as u64;
        Ok(n)
    }
}
fn stream_headers(
    reader: impl Read,
    bytes: u64,
    digest: &[u8; 32],
    cancel: Option<&CancelFlag>,
) -> gen_core::Result<HeaderFacts> {
    let mut reader = HashReader {
        inner: reader,
        hash: Sha256::new(),
        bytes: 0,
        cancel,
    };
    let result = (|| {
        let layout = weightsmeta::safetensors_reader_tensor_locations(
            &mut reader,
            bytes,
            "MOSS authenticated stream",
        )?;
        let mut chunk = [0u8; 64 * 1024];
        while reader.read(&mut chunk)? != 0 {
            if reader.bytes > bytes {
                return Err(gen_core::Error::Unsupported(
                    "MOSS stream size mismatch".into(),
                ));
            }
        }
        if reader.bytes != bytes || reader.hash.finalize()[..] != digest[..] {
            return Err(gen_core::Error::Unsupported(
                "MOSS streamed source SHA/size mismatch".into(),
            ));
        }
        Ok(layout.tensors.into_iter().map(|t| t.header).collect())
    })();
    preparation_check(cancel)?;
    result
}
fn asset_headers(a: &Asset, cancel: Option<&CancelFlag>) -> gen_core::Result<HeaderFacts> {
    static CACHE: OnceLock<HeaderCache> = OnceLock::new();
    verify_moss_image_asset(&a.pin, a.repo, a.revision, &a.relative)?;
    let identity = moss_image_asset(a.repo, a.revision, &a.relative)
        .ok_or_else(|| gen_core::Error::Unsupported("unknown MOSS source identity".into()))?;
    let key = (
        identity.repository,
        identity.revision,
        identity.relative_path,
        identity.bytes,
        *a.pin.content_sha256(),
    );
    cached_headers(
        CACHE.get_or_init(Mutex::default),
        key,
        &a.pin,
        cancel,
        || {
            a.pin.read_unchanged(|path| {
                let file = std::fs::File::open(path)?;
                if file.metadata()?.len() != identity.bytes {
                    return Err(gen_core::Error::Unsupported(
                        "MOSS opened source size mismatch".into(),
                    ));
                }
                stream_headers(file, identity.bytes, a.pin.content_sha256(), cancel)
            })
        },
    )
}
fn validate_source_dtypes(
    assets: &[Asset],
    cancel: Option<&CancelFlag>,
) -> gen_core::Result<BTreeMap<[u8; 32], HeaderFacts>> {
    let mut result = BTreeMap::new();
    for a in assets
        .iter()
        .filter(|a| a.relative.ends_with(".safetensors"))
    {
        preparation_check(cancel)?;
        let expected = if a.repo == DMD_REPO || a.role == MossImageAssetRole::Vae {
            weightsmeta::Dtype::F32
        } else {
            weightsmeta::Dtype::BF16
        };
        let headers = asset_headers(a, cancel)?;
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
fn validate_fp32(
    a: &Asset,
    expected: &BTreeMap<String, Vec<usize>>,
    cancel: Option<&CancelFlag>,
) -> gen_core::Result<()> {
    static VALIDATED: OnceLock<Mutex<Fp32Proofs>> = OnceLock::new();
    preparation_check(cancel)?;
    verify_moss_image_asset(&a.pin, a.repo, a.revision, &a.relative)?;
    let identity = moss_image_asset(a.repo, a.revision, &a.relative)
        .ok_or_else(|| gen_core::Error::Unsupported("unknown MOSS asset".into()))?;
    let key = (
        identity.repository,
        identity.revision,
        identity.relative_path,
        identity.bytes,
        *a.pin.content_sha256(),
    );
    let cache = VALIDATED.get_or_init(Mutex::default);
    if cache
        .lock()
        .map_err(|_| gen_core::Error::Msg("MOSS validation cache poisoned".into()))?
        .get(&key)
        == Some(expected)
    {
        a.pin.verify_unchanged()?;
        preparation_check(cancel)?;
        return Ok(());
    }
    let result = a.pin.read_unchanged(|path| {
        let file = std::fs::File::open(path)?;
        if file.metadata()?.len() != identity.bytes {
            return Err(gen_core::Error::Unsupported(
                "MOSS opened source size mismatch".into(),
            ));
        }
        let capacity = usize::try_from(identity.bytes)
            .ok()
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| gen_core::Error::Unsupported("MOSS source too large".into()))?;
        let mut reader = HashReader {
            inner: file.take(identity.bytes + 1),
            hash: Sha256::new(),
            bytes: 0,
            cancel,
        };
        let mut bytes = Vec::with_capacity(capacity);
        reader.read_to_end(&mut bytes)?;
        if reader.bytes != identity.bytes
            || reader.hash.finalize()[..] != a.pin.content_sha256()[..]
        {
            return Err(gen_core::Error::Unsupported(
                "MOSS actual validation bytes differ from approved identity".into(),
            ));
        }
        validate_fp32_bytes_cancellable(&bytes, expected, cancel)
    });
    preparation_check(cancel)?;
    result?;
    cache
        .lock()
        .map_err(|_| gen_core::Error::Msg("MOSS validation cache poisoned".into()))?
        .insert(key, expected.clone());
    Ok(())
}

#[cfg(test)]
pub(crate) fn validate_fp32_bytes(
    bytes: &[u8],
    expected: &BTreeMap<String, Vec<usize>>,
) -> gen_core::Result<()> {
    validate_fp32_bytes_cancellable(bytes, expected, None)
}
fn validate_fp32_bytes_cancellable(
    bytes: &[u8],
    expected: &BTreeMap<String, Vec<usize>>,
    cancel: Option<&CancelFlag>,
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
        for (index, bytes) in tensor.data().chunks_exact(4).enumerate() {
            if index.is_multiple_of(16384) {
                preparation_check(cancel)?;
            }
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

fn validate_dataspace(a: &Asset, cancel: Option<&CancelFlag>) -> gen_core::Result<()> {
    validate_fp32(
        a,
        &[
            ("latent_mix.Q".into(), vec![128, 128]),
            ("step_codes".into(), vec![8]),
        ]
        .into_iter()
        .collect(),
        cancel,
    )
}
pub(crate) fn load_weights(
    assets: &[Asset],
    role: MossImageAssetRole,
    device: &Device,
    dtype: DType,
    prefixes: &[&str],
) -> Result<Weights> {
    load_weights_cancellable(assets, role, device, dtype, prefixes, &CancelFlag::new())
}
pub(crate) fn load_weights_cancellable(
    assets: &[Asset],
    role: MossImageAssetRole,
    device: &Device,
    dtype: DType,
    prefixes: &[&str],
    cancel: &CancelFlag,
) -> Result<Weights> {
    let mut map = HashMap::new();
    for a in assets
        .iter()
        .filter(|a| a.role == role && a.relative.ends_with(".safetensors"))
    {
        crate::transformer::check_cancel(cancel)?;
        let bytes = a.authenticated_bytes()?;
        crate::transformer::check_cancel(cancel)?;
        let safetensors = candle_gen::candle_core::safetensors::SliceSafetensors::new(&bytes)?;
        for (k, _) in safetensors.tensors() {
            crate::transformer::check_cancel(cancel)?;
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
    crate::transformer::check_cancel(cancel)?;
    Ok(Weights::from_map(map))
}
pub(crate) fn tensor_f32(t: &Tensor) -> Result<Vec<f32>> {
    Ok(t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?)
}

#[cfg(test)]
mod header_cpu_tests {
    use super::*;
    use std::{cell::Cell, io::Cursor};
    struct Counted<'a> {
        bytes: Cursor<Vec<u8>>,
        read: &'a Cell<usize>,
        largest: &'a Cell<usize>,
        cancel: Option<&'a CancelFlag>,
    }
    impl Read for Counted<'_> {
        fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
            self.largest.set(self.largest.get().max(out.len()));
            let n = self.bytes.read(out)?;
            self.read.set(self.read.get() + n);
            if self.read.get() > 64 * 1024 {
                if let Some(cancel) = self.cancel {
                    cancel.cancel();
                }
            }
            Ok(n)
        }
    }
    fn fixture() -> Vec<u8> {
        let mut h = serde_json::to_vec(
            &serde_json::json!({"t":{"dtype":"BF16","shape":[65536],"data_offsets":[0,131072]}}),
        )
        .unwrap();
        while !h.len().is_multiple_of(8) {
            h.push(b' ');
        }
        let mut out = (h.len() as u64).to_le_bytes().to_vec();
        out.extend(h);
        out.resize(out.len() + 131072, 0);
        out
    }
    #[test]
    fn authenticated_header_cache_counts_cold_reads_and_zero_warm_payload_io_sc26325() {
        let data = fixture();
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), &data).unwrap();
        let pin = PinnedWeightsFile::pin(tmp.path()).unwrap();
        let digest = *pin.content_sha256();
        let key = (
            "tiny-approved-test",
            "rev",
            "tensor.safetensors",
            data.len() as u64,
            digest,
        );
        let cache = HeaderCache::default();
        let read = Cell::new(0);
        let largest = Cell::new(0);
        let auth = Cell::new(0);
        let headers = cached_headers(&cache, key, &pin, None, || {
            auth.set(auth.get() + 1);
            pin.read_unchanged(|_| {
                stream_headers(
                    Counted {
                        bytes: Cursor::new(data.clone()),
                        read: &read,
                        largest: &largest,
                        cancel: None,
                    },
                    data.len() as u64,
                    &digest,
                    None,
                )
            })
        })
        .unwrap();
        assert_eq!(read.get(), data.len());
        assert_eq!(auth.get(), 1);
        assert!(largest.get() <= 64 * 1024);
        assert_eq!(headers[0].shape, vec![65536]);
        assert_eq!(headers[0].data_bytes, 131072);
        let before = read.get();
        let warm = cached_headers(&cache, key, &pin, None, || {
            panic!("warm header facts reopened payload")
        })
        .unwrap();
        assert_eq!(warm, headers);
        assert_eq!(read.get(), before);
        assert_eq!(auth.get(), 1);
        std::fs::write(tmp.path(), b"changed retained target").unwrap();
        assert!(cached_headers(&cache, key, &pin, None, || panic!("mutated pin read")).is_err());
    }
    #[test]
    fn corrupt_and_canceled_streams_never_cache_and_same_identity_can_retry_sc26325() {
        let data = fixture();
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), &data).unwrap();
        let pin = PinnedWeightsFile::pin(tmp.path()).unwrap();
        let digest = *pin.content_sha256();
        let key = (
            "tiny-approved-test",
            "rev",
            "tensor.safetensors",
            data.len() as u64,
            digest,
        );
        for canceled in [false, true] {
            let cache = HeaderCache::default();
            let token = CancelFlag::default();
            let read = Cell::new(0);
            let largest = Cell::new(0);
            let mut bad = data.clone();
            if !canceled {
                *bad.last_mut().unwrap() = 1;
            }
            let result = cached_headers(&cache, key, &pin, Some(&token), || {
                pin.read_unchanged(|_| {
                    stream_headers(
                        Counted {
                            bytes: Cursor::new(bad),
                            read: &read,
                            largest: &largest,
                            cancel: canceled.then_some(&token),
                        },
                        data.len() as u64,
                        &digest,
                        Some(&token),
                    )
                })
            });
            assert!(result.is_err());
            if canceled {
                assert!(matches!(result, Err(gen_core::Error::Canceled)));
                assert!(read.get() < data.len());
            }
            assert!(candle_gen::lock_recover(&cache).is_empty());
            let fresh = CancelFlag::default();
            let facts = cached_headers(&cache, key, &pin, Some(&fresh), || {
                pin.read_unchanged(|_| {
                    stream_headers(
                        Cursor::new(data.clone()),
                        data.len() as u64,
                        &digest,
                        Some(&fresh),
                    )
                })
            })
            .unwrap();
            assert_eq!(facts[0].data_bytes, 131072);
            assert_eq!(candle_gen::lock_recover(&cache).len(), 1);
        }
    }
    #[test]
    fn truncated_and_oversized_prefixes_refuse_before_large_allocation_sc26325() {
        for bytes in [
            vec![0, 1, 2],
            u64::MAX.to_le_bytes().to_vec(),
            100_000_001u64.to_le_bytes().to_vec(),
            32u64.to_le_bytes().to_vec(),
        ] {
            let read = Cell::new(0);
            let largest = Cell::new(0);
            let digest: [u8; 32] = Sha256::digest(&bytes).into();
            assert!(stream_headers(
                Counted {
                    bytes: Cursor::new(bytes.clone()),
                    read: &read,
                    largest: &largest,
                    cancel: None
                },
                bytes.len() as u64,
                &digest,
                None
            )
            .is_err());
            assert!(read.get() <= 8);
            assert!(largest.get() <= 8);
        }
    }
}

#[cfg(test)]
mod constructor_tests {
    use super::*;
    #[test]
    fn full_prepared_closure_and_generic_constructor_skip_ordinary_dit_sc26325() {
        let tmp = tempfile::tempdir().unwrap();
        let full_root = tmp.path().join("full");
        std::fs::create_dir_all(full_root.join("transformer")).unwrap();
        let config = full_root.join("transformer/config.json");
        std::fs::write(
            &config,
            include_bytes!(
                "../../../../contracts/gen-core/tests/fixtures/moss_image/transformer_config.json"
            ),
        )
        .unwrap();
        let mut spec = LoadSpec::new(WeightsSource::Dir(tmp.path().join("ordinary-unpinned")));
        spec.components.insert(
            DMD_TRANSFORMER_COMPONENT.into(),
            WeightsSource::Dir(full_root),
        );
        spec.components.insert(
            DMD_DATASPACE_COMPONENT.into(),
            WeightsSource::File(tmp.path().join("selected-data-not-yet-opened")),
        );
        let data = tmp.path().join("selected-data-not-yet-opened");
        std::fs::write(
            &data,
            b"unselected at construction; selected consumption must authenticate",
        )
        .unwrap();
        spec.prepare_with_file_pins([
            PinnedWeightsFile::pin(&config).unwrap(),
            PinnedWeightsFile::pin(data).unwrap(),
        ])
        .unwrap();
        let bad = tmp.path().join("ordinary-config");
        std::fs::write(&bad, b"unselected ordinary config must not be consumed").unwrap();
        let ordinary = vec![Asset {
            #[cfg(feature = "test-support")]
            fixture_digest: None,
            pin: PinnedWeightsFile::pin(bad).unwrap(),
            role: MossImageAssetRole::Transformer,
            repo: SFT_REPO,
            revision: SFT_REVISION,
            relative: "transformer/config.json".into(),
        }];
        let full = Some(MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32));
        assert_eq!(
            transformer_config(full, &spec, &ordinary).unwrap().dim,
            3072
        );
        for mode in [
            MossImageAcceleration::Off,
            MossImageAcceleration::Dmd(MossDmdPrecision::UnmergedBf16),
        ] {
            assert!(transformer_config(Some(mode), &spec, &ordinary).is_err());
            assert_eq!(
                snapshot_files(Some(mode))
                    .filter(|p| p.starts_with("transformer/"))
                    .count(),
                5
            );
        }
        assert_eq!(
            snapshot_files(None).collect::<Vec<_>>(),
            snapshot_files(full).collect::<Vec<_>>()
        );
        assert_eq!(snapshot_files(None).count(), 12);
        assert!(snapshot_files(None).all(|p| !p.starts_with("transformer/")));
        // The production constructor reaches required shared preparation without any ordinary
        // DiT pin. Full construction/execution needs real shared weights and remains terminal-only.
        let error =
            crate::MossImageGenerator::load_on_device(MossImageVariant::Sft, &spec, Device::Cpu)
                .err()
                .unwrap()
                .to_string();
        assert!(error.contains("text_encoder/config.json"), "{error}");
        assert!(!error.contains("ordinary-unpinned/transformer"), "{error}");
        std::fs::write(config, b"corrupt selected published config").unwrap();
        assert!(transformer_config(full, &spec, &ordinary).is_err());
    }
}
