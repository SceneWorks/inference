//! Exact caller-provisioned assets, authenticated before graph construction.
use crate::transformer::{Geometry, Transformer};
use mlx_gen::gen_core::moss_image::*;
use mlx_gen::gen_core::weightsmeta::{CheckpointMeta, SafetensorsTensorHeader};
use mlx_gen::gen_core::PinnedWeightsFile;
use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, Error, LoadSpec, Result, WeightsSource};
use mlx_rs::{Array, Dtype};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashMap},
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
};

#[derive(Clone)]
pub(crate) struct Asset {
    pub role: MossImageAssetRole,
    pub repo: &'static str,
    pub revision: &'static str,
    pub relative: String,
    pub pin: PinnedWeightsFile,
}
impl Asset {
    fn capture(
        spec: &LoadSpec,
        path: PathBuf,
        repo: &'static str,
        revision: &'static str,
        relative: &str,
        role: MossImageAssetRole,
    ) -> Result<Self> {
        let absolute = std::path::absolute(path)?;
        let pin = if spec.prepared_file_pins().is_prepared() {
            spec.prepared_file_pins()
                .get(&absolute)
                .cloned()
                .ok_or_else(|| {
                    Error::Unsupported(format!(
                        "prepared MOSS identity missing {}",
                        absolute.display()
                    ))
                })?
        } else {
            PinnedWeightsFile::pin(&absolute)?
        };
        if pin.loader_path() != absolute {
            return Err(Error::Unsupported(
                "MOSS prepared token path mismatch".into(),
            ));
        }
        verify_moss_image_asset(&pin, repo, revision, relative)?;
        Ok(Self {
            role,
            repo,
            revision,
            relative: relative.into(),
            pin,
        })
    }
    pub(crate) fn bytes(&self) -> Result<Vec<u8>> {
        self.pin.read_unchanged(|path| {
            let bytes = std::fs::read(path)?;
            // Authenticate the same owned buffer used by parsers/tensor copies, closing reopen
            // swap/restore ambiguity. No parser silently reopens the pathname after this check.
            let expected = moss_image_asset(self.repo, self.revision, &self.relative)
                .ok_or_else(|| Error::Unsupported("unknown MOSS byte identity".into()))?;
            if bytes.len() as u64 != expected.bytes
                || format!("{:x}", Sha256::digest(&bytes)) != expected.sha256
            {
                return Err(Error::Unsupported(format!(
                    "consumed MOSS bytes mismatch: {}",
                    self.relative
                )));
            }
            Ok(bytes)
        })
    }
    /// Header facts come from the same authenticated owned bytes as materialization. Only the
    /// small immutable headers are cached, keyed by the complete consumed-content digest.
    pub(crate) fn headers(&self) -> Result<Vec<SafetensorsTensorHeader>> {
        static CACHE: OnceLock<Mutex<HashMap<[u8; 32], Vec<SafetensorsTensorHeader>>>> =
            OnceLock::new();
        self.pin.verify_unchanged()?;
        let key = *self.pin.content_sha256();
        let cache = CACHE.get_or_init(Mutex::default);
        if let Some(headers) = cache
            .lock()
            .map_err(|_| Error::Msg("MOSS header cache poisoned".into()))?
            .get(&key)
            .cloned()
        {
            return Ok(headers);
        }
        let meta = CheckpointMeta::from_bytes(self.bytes()?)?;
        let headers = meta
            .keys()
            .map(|name| {
                let tensor = meta.tensor(name).expect("indexed checkpoint tensor");
                SafetensorsTensorHeader {
                    name: name.into(),
                    dtype: tensor.dtype,
                    shape: tensor.shape.to_vec(),
                    data_bytes: tensor.data.len() as u64,
                }
            })
            .collect::<Vec<_>>();
        cache
            .lock()
            .map_err(|_| Error::Msg("MOSS header cache poisoned".into()))?
            .insert(key, headers.clone());
        Ok(headers)
    }
    pub(crate) fn json(&self) -> Result<serde_json::Value> {
        serde_json::from_slice(&self.bytes()?)
            .map_err(|e| Error::Msg(format!("{}: {e}", self.relative)))
    }
    pub(crate) fn receipt(&self) -> Result<MossImageAssetReceipt> {
        self.pin.verify_unchanged()?;
        // File components may have caller-specific lexical names; the verified frozen relative
        // identity describes consumed bytes, while the retained token still binds the actual file.
        Ok(MossImageAssetReceipt {
            role: self.role,
            repository: self.repo.into(),
            revision: self.revision.into(),
            relative_path: self.relative.clone(),
            sha256: self
                .pin
                .content_sha256()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect(),
        })
    }
}
#[derive(Clone)]
pub(crate) struct Sources {
    pub selection: Option<MossImageAcceleration>,
    pub ordinary: Vec<Asset>,
    pub optional: Vec<Asset>,
    pub config: MossImageConfig,
}
fn dir(source: &WeightsSource) -> Result<&Path> {
    match source {
        WeightsSource::Dir(p) => Ok(p),
        _ => Err(Error::Unsupported(
            "MOSS primary/shared components require an exact snapshot directory".into(),
        )),
    }
}
fn role(relative: &str) -> MossImageAssetRole {
    if relative.starts_with("transformer/") {
        MossImageAssetRole::Transformer
    } else if relative.starts_with("text_encoder/") {
        MossImageAssetRole::TextEncoder
    } else if relative.starts_with("tokenizer/") {
        MossImageAssetRole::Tokenizer
    } else if relative.starts_with("scheduler/") {
        MossImageAssetRole::Scheduler
    } else {
        MossImageAssetRole::Vae
    }
}
fn snapshot_files(selection: Option<MossImageAcceleration>) -> impl Iterator<Item = &'static str> {
    SNAPSHOT_FILES.iter().copied().filter(move |relative| {
        *relative != "LICENSE"
            && (!relative.starts_with("transformer/")
                || selection
                    .is_some_and(|m| m != MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32)))
    })
}
fn transformer_config(
    selection: Option<MossImageAcceleration>,
    ordinary: &[Asset],
    optional: &[Asset],
) -> Result<MossImageConfig> {
    if selection.is_none() {
        return Ok(MossImageConfig::default());
    }
    let assets = if selection == Some(MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32)) {
        optional
    } else {
        ordinary
    };
    MossImageConfig::from_json(
        &assets
            .iter()
            .find(|a| a.relative == "transformer/config.json")
            .ok_or_else(|| Error::Msg("missing selected MOSS transformer config".into()))?
            .json()?,
    )
    .map_err(Into::into)
}
impl Sources {
    pub(crate) fn prepare(variant: MossImageVariant, spec: &LoadSpec) -> Result<Self> {
        Self::prepare_inner(variant, spec, None)
    }
    pub(crate) fn selected(
        variant: MossImageVariant,
        spec: &LoadSpec,
        mode: MossImageAcceleration,
    ) -> Result<Self> {
        Self::prepare_inner(variant, spec, Some(mode))
    }
    fn prepare_inner(
        variant: MossImageVariant,
        spec: &LoadSpec,
        selection: Option<MossImageAcceleration>,
    ) -> Result<Self> {
        let mode = selection.unwrap_or(MossImageAcceleration::Off);
        let full = mode == MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32);
        if variant == MossImageVariant::Base && mode != MossImageAcceleration::Off {
            return Err(Error::Unsupported("MOSS DMD requires SFT".into()));
        }
        if moss_dmd_components(mode)
            .iter()
            .any(|id| !spec.components.contains_key(*id))
        {
            return Err(Error::Unsupported(
                "MOSS selected DMD closure missing; provision the exact selected frozen components"
                    .into(),
            ));
        }
        if spec.prepared_file_pins().is_prepared() && !spec.prepared_file_pins().is_finalized() {
            return Err(Error::Unsupported(
                "MOSS unfinalized prepared source".into(),
            ));
        }
        if spec.precision != mlx_gen::Precision::Bf16
            || spec.offload_policy != mlx_gen::OffloadPolicy::Resident
            || spec.load_shape != mlx_gen::gen_core::LoadShape::EagerMaterialization
        {
            return Err(Error::Unsupported("MOSS supports the resident/eager official BF16 load surface; full-FP32 DMD is selected explicitly through MOSS request controls".into()));
        }
        if spec.quantize.is_some()
            || !spec.adapters.is_empty()
            || spec.control.is_some()
            || !spec.extra_controls.is_empty()
            || spec.ip_adapter.is_some()
            || spec.pid.is_some()
            || spec.identity.is_some()
            || spec.text_encoder.is_some()
        {
            return Err(Error::Unsupported("MOSS supports its frozen components and explicit DMD policies; arbitrary quantization/control/adapter/encoder overrides are unavailable".into()));
        }
        for key in spec.components.keys() {
            if ![
                MOSS_SHARED_SNAPSHOT_COMPONENT,
                DMD_LORA_COMPONENT,
                DMD_TIME_COMPONENT,
                DMD_DATASPACE_COMPONENT,
                DMD_TRANSFORMER_COMPONENT,
            ]
            .contains(&key.as_str())
            {
                return Err(Error::Unsupported(format!("unknown MOSS component {key}")));
            }
        }
        if variant == MossImageVariant::Base
            && spec
                .components
                .keys()
                .any(|id| id != MOSS_SHARED_SNAPSHOT_COMPONENT)
        {
            return Err(Error::Unsupported("Base cannot load SFT-only DMD components; use the exact SFT parent for optional acceleration".into()));
        }
        let root = dir(&spec.weights)?;
        let shared = match spec.components.get(MOSS_SHARED_SNAPSHOT_COMPONENT) {
            Some(s) => dir(s)?,
            None if variant == MossImageVariant::Sft => root,
            None => {
                return Err(Error::Unsupported(
                    "Base requires exact SFT moss-shared-snapshot components".into(),
                ))
            }
        };
        let mut ordinary = Vec::new();
        for relative in snapshot_files(selection) {
            let transformer = relative.starts_with("transformer/");
            let (base, repo, rev) = if transformer {
                (root, variant.repository(), variant.revision())
            } else {
                (shared, SFT_REPO, SFT_REVISION)
            };
            ordinary.push(Asset::capture(
                spec,
                base.join(relative),
                repo,
                rev,
                relative,
                role(relative),
            )?);
        }
        // Licence identity is independently verified without inventing a receipt tensor role.
        Asset::capture(
            spec,
            root.join("LICENSE"),
            variant.repository(),
            variant.revision(),
            "LICENSE",
            MossImageAssetRole::Transformer,
        )?;
        if variant == MossImageVariant::Base {
            Asset::capture(
                spec,
                shared.join("LICENSE"),
                SFT_REPO,
                SFT_REVISION,
                "LICENSE",
                MossImageAssetRole::Vae,
            )?;
        }
        let mut optional = Vec::new();
        for (id, file, role) in [
            (
                DMD_LORA_COMPONENT,
                "lora.safetensors",
                MossImageAssetRole::DmdLora,
            ),
            (
                DMD_TIME_COMPONENT,
                "time_embedding.safetensors",
                MossImageAssetRole::DmdTime,
            ),
            (
                DMD_DATASPACE_COMPONENT,
                "dataspace.safetensors",
                MossImageAssetRole::DmdDataspace,
            ),
        ] {
            if let Some(source) = spec
                .components
                .get(id)
                .filter(|_| moss_dmd_components(mode).contains(&id))
            {
                let path = match source {
                    WeightsSource::File(p) => p.clone(),
                    WeightsSource::Dir(p) => p.join(file),
                };
                optional.push(Asset::capture(
                    spec,
                    path,
                    DMD_REPO,
                    DMD_REVISION,
                    file,
                    role,
                )?);
            }
        }
        if let Some(source) = spec
            .components
            .get(DMD_TRANSFORMER_COMPONENT)
            .filter(|_| full)
        {
            let path = dir(source)?;
            let root = if path.file_name().is_some_and(|p| p == "transformer") {
                path.parent()
                    .ok_or_else(|| Error::Msg("invalid DMD transformer directory".into()))?
            } else {
                path
            };
            for asset in MOSS_IMAGE_ASSETS
                .iter()
                .filter(|a| a.repository == DMD_REPO && a.relative_path.starts_with("transformer/"))
            {
                optional.push(Asset::capture(
                    spec,
                    root.join(asset.relative_path),
                    DMD_REPO,
                    DMD_REVISION,
                    asset.relative_path,
                    MossImageAssetRole::Transformer,
                )?);
            }
        }
        let sources = Self {
            selection,
            config: transformer_config(selection, &ordinary, &optional)?,
            ordinary,
            optional,
        };
        if selection.is_some() && !full {
            sources.verify_index(
                false,
                "transformer/diffusion_pytorch_model.safetensors.index.json",
            )?;
        }
        sources.verify_index(false, "text_encoder/model.safetensors.index.json")?;
        if sources
            .optional
            .iter()
            .any(|a| a.relative == "transformer/config.json")
        {
            sources.verify_index(
                true,
                "transformer/diffusion_pytorch_model.safetensors.index.json",
            )?;
        }
        MossImageSchedulerConfig::from_json(
            &sources.find("scheduler/scheduler_config.json")?.json()?,
        )?;
        let vae = sources.find("vae/config.json")?.json()?;
        if vae["decoder_block_out_channels"] != serde_json::json!([96, 192, 384, 384])
            || vae["block_out_channels"] != serde_json::json!([128, 256, 512, 512])
            || vae["latent_channels"] != 32
            || vae["batch_norm_eps"] != 0.0001
        {
            return Err(Error::Unsupported(
                "MOSS small VAE configuration mismatch".into(),
            ));
        }
        Ok(sources)
    }
    fn verify_index(&self, optional: bool, relative: &str) -> Result<()> {
        let assets = if optional {
            &self.optional
        } else {
            &self.ordinary
        };
        let index = assets
            .iter()
            .find(|a| a.relative == relative)
            .ok_or_else(|| Error::Msg("missing MOSS shard index".into()))?;
        let value = index.json()?;
        let map = value["weight_map"]
            .as_object()
            .ok_or_else(|| Error::Msg("MOSS empty/malformed weight map".into()))?;
        if map.is_empty() {
            return Err(Error::Msg("MOSS empty weight map".into()));
        }
        let parent = Path::new(relative)
            .parent()
            .ok_or_else(|| Error::Msg("invalid MOSS index".into()))?;
        for name in map.values() {
            let name = name
                .as_str()
                .ok_or_else(|| Error::Msg("invalid MOSS shard value".into()))?;
            let rel = parent.join(name);
            if Path::new(name)
                .components()
                .any(|c| !matches!(c, std::path::Component::Normal(_)))
                || !assets.iter().any(|a| Path::new(&a.relative) == rel)
            {
                return Err(Error::Unsupported(
                    "MOSS shard index selects a foreign/missing file".into(),
                ));
            }
        }
        // Also use the shared complete-set confinement validator before any tensor source opens.
        let directory = index
            .pin
            .loader_path()
            .parent()
            .ok_or_else(|| Error::Msg("missing index parent".into()))?;
        let roots = mlx_gen::gen_core::safetensors_shards::snapshot_shard_roots(directory)?;
        let _ = mlx_gen::gen_core::safetensors_shards::resolve_indexed_safetensors_shards(
            directory,
            index
                .pin
                .loader_path()
                .file_name()
                .and_then(|p| p.to_str())
                .ok_or_else(|| Error::Msg("invalid shard index filename".into()))?,
            &roots,
        )?;
        Ok(())
    }
    pub(crate) fn find(&self, relative: &str) -> Result<&Asset> {
        self.ordinary
            .iter()
            .find(|a| a.relative == relative)
            .ok_or_else(|| Error::Msg(format!("missing MOSS {relative}")))
    }
    pub(crate) fn mode_assets(&self, mode: MossImageAcceleration) -> Result<Vec<&Asset>> {
        if self.selection.is_some_and(|selected| selected != mode) {
            return Err(Error::Unsupported(
                "MOSS prepared source mode mismatch".into(),
            ));
        }
        let full = mode == MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32);
        let mut assets: Vec<_> = self
            .ordinary
            .iter()
            .filter(|a| !full || a.role != MossImageAssetRole::Transformer)
            .collect();
        let needed = moss_dmd_components(mode);
        for id in needed {
            let found: Vec<_> = self
                .optional
                .iter()
                .filter(|a| match *id {
                    DMD_LORA_COMPONENT => a.role == MossImageAssetRole::DmdLora,
                    DMD_TIME_COMPONENT => a.role == MossImageAssetRole::DmdTime,
                    DMD_DATASPACE_COMPONENT => a.role == MossImageAssetRole::DmdDataspace,
                    DMD_TRANSFORMER_COMPONENT => a.role == MossImageAssetRole::Transformer,
                    _ => false,
                })
                .collect();
            if found.is_empty() {
                return Err(Error::Unsupported(format!("MOSS requested mode needs {id}; provision exact {DMD_REPO}@{DMD_REVISION} assets")));
            }
            assets.extend(found);
        }
        Ok(assets)
    }
    pub(crate) fn ensure_unchanged(&self) -> Result<()> {
        for a in self.ordinary.iter().chain(&self.optional) {
            a.pin.verify_unchanged()?;
        }
        Ok(())
    }
    pub(crate) fn transformer(
        &self,
        mode: MossImageAcceleration,
        cancel: &CancelFlag,
    ) -> Result<Transformer> {
        let assets = self.mode_assets(mode)?;
        let mut weights = read_weights(
            assets.iter().copied().filter(|a| {
                a.role == MossImageAssetRole::Transformer && a.relative.ends_with(".safetensors")
            }),
            |_| true,
            cancel,
        )?;
        let dtype = if mode == MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32) {
            Dtype::Float32
        } else {
            Dtype::Bfloat16
        };
        let checks = weights.clone();
        if checks.keys().any(|k| {
            checks.get(k).is_some_and(|a| {
                a.dtype()
                    != if k == "rope.bands" {
                        Dtype::Float32
                    } else {
                        dtype
                    }
            })
        }) {
            return Err(Error::Unsupported(
                "MOSS transformer source dtype mismatch".into(),
            ));
        }
        let mut model = Transformer::load(&weights, Geometry::from(&self.config))?;
        if !weights.unused_keys().is_empty() {
            return Err(Error::Unsupported(format!(
                "unconsumed MOSS transformer tensors: {:?}",
                weights.unused_keys()
            )));
        }
        weights.materialize_accessed()?;
        weights.remove_accessed();
        if mode == MossImageAcceleration::Dmd(MossDmdPrecision::UnmergedBf16) {
            let lora = read_weights(
                assets
                    .iter()
                    .copied()
                    .filter(|a| a.role == MossImageAssetRole::DmdLora),
                |_| true,
                cancel,
            )?;
            let replacements = read_weights(
                assets
                    .iter()
                    .copied()
                    .filter(|a| a.role == MossImageAssetRole::DmdTime),
                |_| true,
                cancel,
            )?;
            if lora.len() != 756 || replacements.len() != 6 {
                return Err(Error::Unsupported(
                    "incomplete/extra MOSS adapter/replacement tensors".into(),
                ));
            }
            model.install_dmd(&lora, &replacements, 128)?;
            lora.materialize_accessed()?;
            replacements.materialize_accessed()?;
            if !lora.unused_keys().is_empty() || !replacements.unused_keys().is_empty() {
                return Err(Error::Unsupported("unconsumed MOSS DMD tensors".into()));
            }
        }
        model.materialize_owned(cancel)?;
        self.ensure_unchanged()?;
        Ok(model)
    }
}
pub(crate) fn read_weights<'a>(
    assets: impl Iterator<Item = &'a Asset>,
    include: impl Fn(&str) -> bool,
    cancel: &CancelFlag,
) -> Result<Weights> {
    let mut tensors = HashMap::new();
    for asset in assets {
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let bytes = asset.bytes()?;
        let meta = CheckpointMeta::from_bytes(bytes)?;
        let expected = match asset.role {
            MossImageAssetRole::DmdLora => Some(dmd_lora_shapes()),
            MossImageAssetRole::DmdTime => Some(dmd_replacement_shapes()),
            MossImageAssetRole::DmdDataspace => Some(
                [
                    ("latent_mix.Q".into(), vec![128, 128]),
                    ("step_codes".into(), vec![8]),
                ]
                .into_iter()
                .collect(),
            ),
            _ => None,
        };
        if let Some(expected) = expected {
            let views = meta
                .keys()
                .map(|key| {
                    Ok((
                        key.to_string(),
                        meta.tensor(key)
                            .ok_or_else(|| Error::MissingTensor(key.into()))?,
                    ))
                })
                .collect::<Result<BTreeMap<_, _>>>()?;
            validate_dmd_tensors(&views, &expected)?;
        }
        for key in meta.keys().filter(|k| include(k)) {
            let t = meta
                .tensor(key)
                .ok_or_else(|| Error::MissingTensor(key.into()))?;
            let dtype = match t.dtype {
                mlx_gen::gen_core::weightsmeta::Dtype::BF16 => Dtype::Bfloat16,
                mlx_gen::gen_core::weightsmeta::Dtype::F32 => Dtype::Float32,
                _ => {
                    return Err(Error::Unsupported(format!(
                        "MOSS tensor {key} source must be BF16 or FP32"
                    )))
                }
            };
            let shape = t
                .shape
                .iter()
                .map(|d| {
                    i32::try_from(*d)
                        .map_err(|_| Error::Msg("MOSS shape exceeds backend range".into()))
                })
                .collect::<Result<Vec<_>>>()?;
            // SAFETY: CheckpointMeta validates shape, dtype, range and owned payload length; MLX
            // copies bytes into its own storage before the immutable owned source is released.
            let array = unsafe { Array::from_raw_data(t.data.as_ptr().cast(), &shape, dtype) };
            if tensors.insert(key.to_string(), array).is_some() {
                return Err(Error::Unsupported(format!("duplicate MOSS tensor {key}")));
            }
        }
    }
    Ok(Weights::from_map(tensors))
}
pub(crate) fn fp32_map(w: &Weights) -> Result<BTreeMap<String, Vec<f32>>> {
    let mut result = BTreeMap::new();
    for key in w.keys() {
        let a = w.require(key)?;
        if a.dtype() != Dtype::Float32 {
            return Err(Error::Unsupported(format!("MOSS {key} must be FP32")));
        }
        mlx_rs::transforms::eval([a])?;
        let v = a
            .try_as_slice::<f32>()
            .map_err(|e| Error::Msg(e.to_string()))?
            .to_vec();
        if v.iter().any(|x| !x.is_finite()) {
            return Err(Error::Unsupported(format!(
                "MOSS {key} has nonfinite values"
            )));
        }
        result.insert(key.into(), v);
    }
    Ok(result)
}

#[cfg(test)]
mod cpu_tests {
    #[test]
    fn full_prepared_closure_and_default_constructor_skip_ordinary_dit_sc26325() {
        let tmp = tempfile::tempdir().unwrap();
        let config = tmp.path().join("full/transformer/config.json");
        std::fs::create_dir_all(config.parent().unwrap()).unwrap();
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
            WeightsSource::Dir(tmp.path().join("full")),
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
        let selected = Asset::capture(
            &spec,
            config.clone(),
            DMD_REPO,
            DMD_REVISION,
            "transformer/config.json",
            MossImageAssetRole::Transformer,
        )
        .unwrap();
        let bad = tmp.path().join("ordinary-config");
        std::fs::write(&bad, b"unselected ordinary config must not be consumed").unwrap();
        let ordinary = vec![Asset {
            pin: PinnedWeightsFile::pin(bad).unwrap(),
            role: MossImageAssetRole::Transformer,
            repo: SFT_REPO,
            revision: SFT_REVISION,
            relative: "transformer/config.json".into(),
        }];
        let full = Some(MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32));
        assert_eq!(
            transformer_config(full, &ordinary, std::slice::from_ref(&selected))
                .unwrap()
                .dim,
            3072
        );
        for mode in [
            MossImageAcceleration::Off,
            MossImageAcceleration::Dmd(MossDmdPrecision::UnmergedBf16),
        ] {
            assert!(
                transformer_config(Some(mode), &ordinary, std::slice::from_ref(&selected)).is_err()
            );
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
        // Exact normal factory call: refusal occurs in shared-file preparation before any MLX
        // allocation/device lookup. No real-weight constructor success is claimed by this fixture.
        let error = crate::model::load_sft(&spec).err().unwrap().to_string();
        assert!(error.contains("text_encoder/config.json"), "{error}");
        assert!(!error.contains("ordinary-unpinned/transformer"), "{error}");
        std::fs::write(config, b"corrupt selected published config").unwrap();
        assert!(transformer_config(full, &ordinary, &[selected]).is_err());
    }

    use super::*;
    #[test]
    fn prepared_child_pin_authenticates_consumed_bytes_and_refuses_mutation_or_missing_pin() {
        let tmp = tempfile::tempdir().unwrap();
        let folder = tmp.path().join("transformer");
        std::fs::create_dir(&folder).unwrap();
        let path = folder.join("config.json");
        std::fs::write(
            &path,
            include_bytes!(
                "../../../../contracts/gen-core/tests/fixtures/moss_image/transformer_config.json"
            ),
        )
        .unwrap();
        let mut spec = LoadSpec::new(WeightsSource::Dir(tmp.path().into()));
        let pin = PinnedWeightsFile::pin(&path).unwrap();
        spec.prepare_with_file_pins([pin]).unwrap();
        let asset = Asset::capture(
            &spec,
            path.clone(),
            SFT_REPO,
            SFT_REVISION,
            "transformer/config.json",
            MossImageAssetRole::Transformer,
        )
        .unwrap();
        assert_eq!(asset.bytes().unwrap().len(), 240);
        assert_eq!(
            asset.receipt().unwrap().sha256,
            moss_image_asset(SFT_REPO, SFT_REVISION, "transformer/config.json")
                .unwrap()
                .sha256
        );
        std::fs::write(&path, vec![b' '; 240]).unwrap();
        assert!(asset.bytes().is_err());
        let mut missing = LoadSpec::new(WeightsSource::Dir(tmp.path().into()));
        missing.prepare_with_file_pins([]).unwrap();
        assert!(Asset::capture(
            &missing,
            path,
            SFT_REPO,
            SFT_REVISION,
            "transformer/config.json",
            MossImageAssetRole::Transformer
        )
        .is_err());
    }
    #[test]
    fn missing_companions_and_unknown_components_fail_before_allocation() {
        let tmp = tempfile::tempdir().unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(tmp.path().into()));
        assert!(Sources::prepare(MossImageVariant::Base, &spec)
            .err()
            .unwrap()
            .to_string()
            .contains("moss-shared-snapshot"));
        let mut foreign = spec.clone();
        foreign
            .components
            .insert("unrecognized".into(), WeightsSource::Dir(tmp.path().into()));
        assert!(Sources::prepare(MossImageVariant::Sft, &foreign)
            .err()
            .unwrap()
            .to_string()
            .contains("unknown MOSS component"));
        let s = Sources {
            selection: None,
            ordinary: vec![],
            optional: vec![],
            config: MossImageConfig::default(),
        };
        for mode in [
            MossImageAcceleration::Dmd(MossDmdPrecision::UnmergedBf16),
            MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32),
        ] {
            assert!(s.mode_assets(mode).is_err());
        }
    }
}
