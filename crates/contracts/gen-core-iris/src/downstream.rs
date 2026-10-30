//! The Iris-3B **downstream task exports** (`iris3b/downstream/__init__.py` `load_export` @
//! [`super::UPSTREAM_CODE_REVISION`]): the `depth/` and `upscaler/` folders of the weights repo.
//!
//! Both tasks run the full backbone in a single forward conditioned on the shipped embedding of the
//! empty prompt, so **no text encoder is loaded** (E4). An export directory holds:
//!
//! ```text
//! config.yaml               model / text_encoder / flow sections of the parent + a `task` section
//! model.safetensors         FP32 weights of the fine-tuned model
//! empty_prompt.safetensors  `embeddings` [1, T, L, D] F32 and `mask` [1, T] BOOL
//! ```
//!
//! [`TaskExport::from_spec`] resolves exactly that closure for one task and refuses any artifact of
//! another task: a text-encoder component, the generation checkpoint (whose `config.yaml` has no
//! `task` section) or the other downstream task's export.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use super::{IrisConfig, IrisTask, BACKBONE_CONFIG_FILE, BACKBONE_WEIGHTS_FILE};
use gen_core::runtime::{LoadSpec, WeightsSource};
use gen_core::{Error, Result};

/// The shipped empty-prompt conditioning of a downstream export.
pub const EMPTY_PROMPT_FILE: &str = "empty_prompt.safetensors";

/// The `task` section of an export's `config.yaml`: its `name` and every other scalar setting
/// (restoration's `sigma` / `tile`), as written.
#[derive(Clone, Debug, PartialEq)]
pub struct TaskSettings {
    pub name: String,
    pub values: Vec<(String, String)>,
}

impl TaskSettings {
    /// A scalar setting by key.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.values
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }
}

/// Read the `task` section of a `config.yaml` text — `None` for a checkpoint without one (the
/// generation backbone, `scripts/export_checkpoint.py`).
pub fn parse_task_settings(text: &str) -> Result<Option<TaskSettings>> {
    let root = super::yaml::parse(text)?;
    let Some(task) = root.get("task") else {
        return Ok(None);
    };
    let super::yaml::Value::Map(entries) = task else {
        return Err(Error::Msg("task: expected a mapping".into()));
    };
    let mut name = None;
    let mut values = Vec::new();
    for (key, value) in entries {
        let scalar = value.as_str(&format!("task.{key}"))?.to_owned();
        if key == "name" {
            name = Some(scalar);
        } else {
            values.push((key.clone(), scalar));
        }
    }
    let name = name.ok_or_else(|| Error::Msg("task: the section has no `name`".into()))?;
    Ok(Some(TaskSettings { name, values }))
}

/// The resolved, identity-checked resources of one downstream task.
#[derive(Clone, Debug, PartialEq)]
pub struct TaskExport {
    pub task: IrisTask,
    /// The export directory (`config.yaml` + `model.safetensors` + `empty_prompt.safetensors`).
    pub dir: PathBuf,
    /// Its parsed and validated `model` / `text_encoder` / `flow` sections.
    pub config: IrisConfig,
    pub settings: TaskSettings,
    /// sha256 (lowercase hex) of the export's `config.yaml` bytes — the configuration revision a
    /// consumer records next to the pinned weights revision.
    pub config_sha256: String,
}

impl TaskExport {
    /// Resolve `task`'s export from a load spec: `spec.weights` is the export directory and nothing
    /// else may be staged. A text encoder, any component, the generation checkpoint or another
    /// task's export is a typed [`Error::Unsupported`] naming the wrong-task artifact.
    pub fn from_spec(spec: &LoadSpec, task: IrisTask, model_id: &str) -> Result<Self> {
        Self::from_spec_impl(spec, task, model_id, None)
    }

    pub fn from_spec_observed(
        spec: &LoadSpec,
        task: IrisTask,
        model_id: &str,
        observer: &mut gen_core::weightsmeta::SourceMetadataObserver<'_>,
    ) -> Result<Self> {
        Self::from_spec_impl(spec, task, model_id, Some(observer))
    }

    fn from_spec_impl(
        spec: &LoadSpec,
        task: IrisTask,
        model_id: &str,
        observer: Option<&mut gen_core::weightsmeta::SourceMetadataObserver<'_>>,
    ) -> Result<Self> {
        if task == IrisTask::Generation {
            return Err(Error::Msg(format!(
                "{model_id}: the generation task is not a downstream export"
            )));
        }
        let name = task.name();
        let subdir = task.backbone_subdir();
        if let Some(key) = spec.components.keys().next() {
            return Err(Error::Unsupported(format!(
                "{model_id}: the {name} task loads only its `{subdir}/` export — drop the '{key}' \
                 component (the {name} task runs on the shipped empty-prompt conditioning and \
                 never loads a text encoder)"
            )));
        }
        if spec.text_encoder.is_some() {
            return Err(Error::Unsupported(format!(
                "{model_id}: LoadSpec::text_encoder is set, but the {name} task never loads a text \
                 encoder (it runs on the shipped {EMPTY_PROMPT_FILE})"
            )));
        }
        let dir = match &spec.weights {
            WeightsSource::Dir(dir) => dir.clone(),
            WeightsSource::File(file) => {
                return Err(Error::Msg(format!(
                    "{model_id}: the {name} resource must be the `{subdir}/` export directory \
                     ({BACKBONE_CONFIG_FILE} + {BACKBONE_WEIGHTS_FILE} + {EMPTY_PROMPT_FILE}), not \
                     the single file {}",
                    file.display()
                )))
            }
        };
        Self::from_dir_impl(&dir, task, model_id, observer)
    }

    /// [`Self::from_spec`] on a directory.
    pub fn from_dir(dir: &Path, task: IrisTask, model_id: &str) -> Result<Self> {
        Self::from_dir_impl(dir, task, model_id, None)
    }

    fn from_dir_impl(
        dir: &Path,
        task: IrisTask,
        model_id: &str,
        mut observer: Option<&mut gen_core::weightsmeta::SourceMetadataObserver<'_>>,
    ) -> Result<Self> {
        let name = task.name();
        let subdir = task.backbone_subdir();
        let resource = format!("{name} export");
        super::require_dir(dir, model_id, &resource)?;
        let config_path = dir.join(BACKBONE_CONFIG_FILE);
        super::require_file(&config_path, model_id, &resource)?;
        let bytes = if let Some(observe) = observer.as_deref_mut() {
            use std::io::Read as _;
            let io = |e| Error::Msg(format!("iris: read {}: {e}", config_path.display()));
            let mut file = std::fs::File::open(&config_path).map_err(io)?;
            let length = file.metadata().map_err(io)?.len();
            observe(gen_core::weightsmeta::SourceMetadataRead::Input(length))?;
            let length = usize::try_from(length).map_err(|_| {
                Error::Unsupported("Iris guarded config input length overflow".into())
            })?;
            // Exact original length, not read_to_end: concurrent growth cannot expand an
            // admitted Vec before the retained identity check gets to reject the replacement.
            let mut bytes = vec![0; length];
            file.read_exact(&mut bytes).map_err(io)?;
            let mut extra = [0u8; 1];
            if file.read(&mut extra).map_err(io)? != 0 {
                return Err(Error::Unsupported(
                    "Iris guarded config changed during read".into(),
                ));
            }
            bytes
        } else {
            std::fs::read(&config_path)
                .map_err(|e| Error::Msg(format!("iris: read {}: {e}", config_path.display())))?
        };
        if let Some(observe) = observer.as_deref_mut() {
            observe(gen_core::weightsmeta::SourceMetadataRead::Yaml(&bytes))?;
        }
        let text = String::from_utf8(bytes.clone()).map_err(|_| {
            Error::Msg(format!("iris: {} is not UTF-8 text", config_path.display()))
        })?;
        if let Some(observe) = observer.as_deref_mut() {
            observe(gen_core::weightsmeta::SourceMetadataRead::YamlParse(&bytes))?;
        }
        let settings = parse_task_settings(&text)
            .map_err(|e| Error::Msg(format!("iris: {}: {e}", config_path.display())))?;
        let settings = match settings {
            Some(s) if s.name == name => s,
            Some(s) => {
                return Err(Error::Unsupported(format!(
                    "{model_id}: {} holds a '{}' export, not '{name}' — stage the `{subdir}/` \
                     folder of {}",
                    dir.display(),
                    s.name,
                    super::UPSTREAM_WEIGHTS_REPO
                )))
            }
            None => {
                return Err(Error::Unsupported(format!(
                    "{model_id}: {} is a generation checkpoint (its {BACKBONE_CONFIG_FILE} has no \
                     `task` section), not the {name} export — stage the `{subdir}/` folder of {}",
                    dir.display(),
                    super::UPSTREAM_WEIGHTS_REPO
                )))
            }
        };
        for file in [BACKBONE_WEIGHTS_FILE, EMPTY_PROMPT_FILE] {
            super::require_file(&dir.join(file), model_id, &resource)?;
        }
        check_weights_task_stamp_observed(
            &dir.join(BACKBONE_WEIGHTS_FILE),
            task,
            model_id,
            observer.as_deref_mut(),
        )?;
        if let Some(observe) = observer {
            observe(gen_core::weightsmeta::SourceMetadataRead::YamlParse(&bytes))?;
        }
        let config = IrisConfig::parse(&text)
            .map_err(|e| Error::Msg(format!("iris: {}: {e}", config_path.display())))?;
        config.validate_supported()?;
        let config_sha256 = Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        gen_core::__model_input_choose!(
            Ok(Self {
                task,
                dir: dir.to_path_buf(),
                config,
                settings,
                config_sha256,
            }),
            (Ok(Self {
                task,
                dir: dir.to_path_buf(),
                config,
                settings,
                config_sha256,
            }))
            .inspect(|observed: &Self| {
                if let Some(sha256) =
                    gen_core::model_input_proof::sha256_from_hex(&observed.config_sha256)
                {
                    gen_core::model_input_proof::record_inputs(
                        "gen-core::iris::downstream::TaskExport::from_dir_impl",
                        gen_core::model_input_proof::InputRole::Configuration,
                        &[
                            gen_core::model_input_proof::InputIdentity::same_read_buffer(
                                Some(&config_path),
                                sha256,
                            ),
                        ],
                    );
                }
            })
        )
    }

    pub fn weights_path(&self) -> PathBuf {
        self.dir.join(BACKBONE_WEIGHTS_FILE)
    }

    pub fn empty_prompt_path(&self) -> PathBuf {
        self.dir.join(EMPTY_PROMPT_FILE)
    }

    /// Read and shape-check the shipped empty-prompt conditioning.
    pub fn empty_prompt(&self) -> Result<EmptyPrompt> {
        EmptyPrompt::read(&self.empty_prompt_path(), &self.config)
    }
}

/// Refuse a downstream task export staged where a **generation** backbone belongs: a
/// `config.yaml` with a `task` section (`depth/`, `upscaler/`, or a native depth- or
/// restoration-training export) is a typed [`Error::Unsupported`] naming the task and its provider.
/// A directory without a readable `config.yaml` passes (the caller's own existence checks name what is missing).
pub fn refuse_task_export(dir: &Path, model_id: &str) -> Result<()> {
    let Ok(text) = std::fs::read_to_string(dir.join(BACKBONE_CONFIG_FILE)) else {
        return Ok(());
    };
    match parse_task_settings(&text) {
        Ok(Some(settings)) => Err(Error::Unsupported(format!(
            "{model_id}: {} is the '{}' task export (its {BACKBONE_CONFIG_FILE} has a `task` \
             section), not a generation backbone — load it with {}",
            dir.display(),
            settings.name,
            match settings.name.as_str() {
                "restoration" => super::restoration::MODEL_ID,
                "depth" => super::DEPTH_MODEL_ID,
                _ => "the matching task's provider",
            }
        ))),
        _ => Ok(()),
    }
}

/// A trained export stamps its task in the `model.safetensors` metadata (`irisTask`, see
/// `train::task_model_metadata`); the shipped release exports carry no stamp. A stamp naming another
/// task is a typed refusal, so a mislabelled directory never loads as this task.
fn check_weights_task_stamp_observed(
    path: &Path,
    task: IrisTask,
    model_id: &str,
    observer: Option<&mut gen_core::weightsmeta::SourceMetadataObserver<'_>>,
) -> Result<()> {
    let Some(stamp) = weights_metadata_impl(path, observer)?
        .and_then(|m| m.get(super::ADAPTER_TASK_KEY).cloned())
    else {
        return Ok(());
    };
    if stamp == task.name() {
        Ok(())
    } else {
        Err(Error::Unsupported(format!(
            "{model_id}: {} is stamped `{}={stamp}`, not the {} task",
            path.display(),
            super::ADAPTER_TASK_KEY,
            task.name()
        )))
    }
}

/// The `__metadata__` map of a safetensors file, read from its header only (`None` when the file is
/// too short to hold a header or carries no metadata).
pub fn weights_metadata(path: &Path) -> Result<Option<std::collections::HashMap<String, String>>> {
    weights_metadata_impl(path, None)
}

fn weights_metadata_impl(
    path: &Path,
    mut observer: Option<&mut gen_core::weightsmeta::SourceMetadataObserver<'_>>,
) -> Result<Option<std::collections::HashMap<String, String>>> {
    use std::io::Read as _;
    let io = |e: std::io::Error| Error::Msg(format!("iris: read {}: {e}", path.display()));
    let mut f = std::fs::File::open(path).map_err(io)?;
    let file_bytes = f.metadata().map_err(io)?.len();
    if file_bytes < 8 {
        return Ok(None);
    }
    // The crate-wide bounded header read (sc-27314): one ceiling, and the header must end inside
    // the file.
    let (n, _data_start) = gen_core::weightsmeta::read_safetensors_header_len(
        &mut f, file_bytes, path,
    )
    .map_err(|error| match error {
        Error::Io(e) => io(e),
        Error::Msg(message) => Error::Msg(format!("iris: {message}")),
        other => other,
    })?;
    if let Some(observe) = observer.as_deref_mut() {
        observe(gen_core::weightsmeta::SourceMetadataRead::Input(n as u64))?;
    }
    let mut header = vec![0u8; n];
    f.read_exact(&mut header).map_err(io)?;
    if let Some(observe) = observer {
        observe(gen_core::weightsmeta::SourceMetadataRead::Json {
            file_bytes,
            bytes: &header,
        })?;
    }
    let v: serde_json::Value = serde_json::from_slice(&header)
        .map_err(|e| Error::Msg(format!("iris: {} header: {e}", path.display())))?;
    Ok(v.get("__metadata__").and_then(|m| m.as_object()).map(|m| {
        m.iter()
            .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_owned())))
            .collect()
    }))
}

/// The empty-prompt conditioning an export ships (`empty_prompt.safetensors`), host-side: the
/// `[1, T, L, D]` selected-layer states and the `[T]` 0/1 mask the backbone reads.
#[derive(Clone, Debug, PartialEq)]
pub struct EmptyPrompt {
    /// `[1, T, L, D]` row-major.
    pub embeddings: Vec<f32>,
    pub shape: [usize; 4],
    /// `[T]` 0/1 flags (upstream's BOOL mask).
    pub mask: Vec<i32>,
}

impl EmptyPrompt {
    /// Read `path` and check it against the backbone it conditions: `T = model.text_len`,
    /// `L = model.text_lap_num_layers`, `D = model.text_dim`.
    pub fn read(path: &Path, config: &IrisConfig) -> Result<Self> {
        let bytes = std::fs::read(path)
            .map_err(|e| Error::Msg(format!("iris: read {}: {e}", path.display())))?;
        let tensors = safetensors::SafeTensors::deserialize(&bytes)
            .map_err(|e| Error::Msg(format!("iris: {}: {e}", path.display())))?;
        let get = |name: &str| {
            tensors.tensor(name).map_err(|_| {
                Error::Msg(format!(
                    "iris: {} has no `{name}` tensor (an export's empty prompt carries \
                     `embeddings` and `mask`)",
                    path.display()
                ))
            })
        };
        let m = &config.model;
        let want = [1, m.text_len, m.text_lap_num_layers, m.text_dim];
        let emb = get("embeddings")?;
        if emb.dtype() != safetensors::Dtype::F32 || emb.shape() != want {
            return Err(Error::Msg(format!(
                "iris: {} `embeddings` is {:?} {:?}; the backbone reads F32 {want:?}",
                path.display(),
                emb.dtype(),
                emb.shape()
            )));
        }
        let mask = get("mask")?;
        if mask.dtype() != safetensors::Dtype::BOOL || mask.shape() != [1, m.text_len] {
            return Err(Error::Msg(format!(
                "iris: {} `mask` is {:?} {:?}; the backbone reads BOOL [1, {}]",
                path.display(),
                mask.dtype(),
                mask.shape(),
                m.text_len
            )));
        }
        let embeddings = emb
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let mask: Vec<i32> = mask.data().iter().map(|&b| i32::from(b != 0)).collect();
        if !mask.contains(&1) {
            return Err(Error::Msg(format!(
                "iris: {} `mask` has no real token",
                path.display()
            )));
        }
        gen_core::__model_input_result!(
            Ok(Self {
                embeddings,
                shape: want,
                mask,
            }),
            "gen-core::iris::downstream::EmptyPrompt::read",
            TensorInput,
            &[path]
        )
    }

    /// Write the export layout (`embeddings` F32 `[1, T, L, D]`, `mask` BOOL `[1, T]`) to `path`
    /// atomically (temp file, fsync, rename) — `scripts/export_downstream.py`'s pair.
    pub fn write(&self, path: &Path) -> Result<()> {
        let n: usize = self.shape.iter().product();
        if self.embeddings.len() != n || self.mask.len() != self.shape[1] {
            return Err(Error::Msg(format!(
                "iris: the empty prompt holds {} values / {} mask flags for {:?}",
                self.embeddings.len(),
                self.mask.len(),
                self.shape
            )));
        }
        let emb: Vec<u8> = self
            .embeddings
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let mask: Vec<u8> = self.mask.iter().map(|&m| u8::from(m != 0)).collect();
        let bad = |e: safetensors::SafeTensorError| Error::Msg(format!("iris: {e}"));
        let views = [
            (
                "embeddings",
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F32,
                    self.shape.to_vec(),
                    &emb,
                )
                .map_err(bad)?,
            ),
            (
                "mask",
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::BOOL,
                    vec![1, self.shape[1]],
                    &mask,
                )
                .map_err(bad)?,
            ),
        ];
        let bytes = safetensors::serialize(views, &None).map_err(bad)?;
        super::train::atomic_write(path, &bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEPTH_CONFIG: &str =
        "model:\n  patch_size: 16\nflow:\n  shift: 4.0\ntask:\n  name: depth\n";

    fn export(dir: &Path, config: &str) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join(BACKBONE_CONFIG_FILE), config).unwrap();
        std::fs::write(dir.join(BACKBONE_WEIGHTS_FILE), b"").unwrap();
        std::fs::write(dir.join(EMPTY_PROMPT_FILE), b"").unwrap();
    }

    #[test]
    fn guarded_config_refusal_precedes_read_buffer_and_parser() {
        let dir = tempfile::tempdir().unwrap();
        export(dir.path(), "not valid YAML: [");
        let spec = LoadSpec::new(WeightsSource::Dir(dir.path().to_owned()));
        let mut calls = 0;
        let error =
            TaskExport::from_spec_observed(&spec, IrisTask::Depth, "iris_3b_depth", &mut |event| {
                assert!(matches!(
                    event,
                    gen_core::weightsmeta::SourceMetadataRead::Input(_)
                ));
                calls += 1;
                Err(Error::Unsupported("capacity sentinel".into()))
            })
            .unwrap_err();
        assert!(matches!(error, Error::Unsupported(ref reason) if reason == "capacity sentinel"));
        assert_eq!(calls, 1);
    }

    #[test]
    fn guarded_task_stamp_uses_existing_header_read_before_config_decode() {
        let dir = tempfile::tempdir().unwrap();
        export(dir.path(), DEPTH_CONFIG);
        let raw = br#"{"__metadata__":{"irisTask":"depth"}}"#;
        let mut bytes = (raw.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(raw);
        std::fs::write(dir.path().join(BACKBONE_WEIGHTS_FILE), bytes).unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(dir.path().to_owned()));
        let mut calls = Vec::new();
        let checked =
            TaskExport::from_spec_observed(&spec, IrisTask::Depth, "iris_3b_depth", &mut |event| {
                calls.push(match event {
                    gen_core::weightsmeta::SourceMetadataRead::Input(_) => 0,
                    gen_core::weightsmeta::SourceMetadataRead::Yaml(_) => 1,
                    gen_core::weightsmeta::SourceMetadataRead::YamlParse(_) => 3,
                    gen_core::weightsmeta::SourceMetadataRead::Json { bytes, .. } => {
                        assert_eq!(bytes, raw);
                        2
                    }
                });
                Ok(())
            })
            .unwrap();
        assert_eq!(calls, [0, 1, 3, 0, 2, 3]);
        assert_eq!(
            checked,
            TaskExport::from_spec(&spec, IrisTask::Depth, "iris_3b_depth").unwrap()
        );
    }

    #[test]
    fn task_section_is_read() {
        let s = parse_task_settings(
            "model:\n  depth: 2\ntask:\n  name: restoration\n  sigma: 0.5\n  tile: 1024\n",
        )
        .unwrap()
        .unwrap();
        assert_eq!(s.name, "restoration");
        assert_eq!(s.get("sigma"), Some("0.5"));
        assert_eq!(s.get("tile"), Some("1024"));
        assert_eq!(parse_task_settings("model:\n  depth: 2\n").unwrap(), None);
        assert!(parse_task_settings("task:\n  sigma: 0.5\n").is_err());
    }

    /// A task export staged as a generation backbone is refused and names the provider that
    /// loads it — both exports a native trainer writes (depth, sc-25687; restoration, sc-25688).
    #[test]
    fn a_task_export_is_refused_as_a_generation_backbone_naming_its_provider() {
        for (task, provider) in [
            ("depth", super::super::DEPTH_MODEL_ID),
            ("restoration", super::super::restoration::MODEL_ID),
        ] {
            let dir = tempfile::tempdir().unwrap();
            export(
                dir.path(),
                &format!("model:\n  patch_size: 16\ntask:\n  name: {task}\n"),
            );
            let err = refuse_task_export(dir.path(), "iris_3b").unwrap_err();
            assert!(matches!(err, Error::Unsupported(_)), "{task}: {err}");
            let text = err.to_string();
            assert!(text.contains(&format!("'{task}' task export")), "{text}");
            assert!(text.contains(&format!("load it with {provider}")), "{text}");
        }
        let generation = tempfile::tempdir().unwrap();
        export(generation.path(), "model:\n  patch_size: 16\n");
        refuse_task_export(generation.path(), "iris_3b").unwrap();
    }

    #[test]
    fn the_depth_export_resolves_with_its_identity() {
        let dir = tempfile::tempdir().unwrap();
        export(dir.path(), DEPTH_CONFIG);
        let spec = LoadSpec::new(WeightsSource::Dir(dir.path().to_path_buf()));
        let e = TaskExport::from_spec(&spec, IrisTask::Depth, "iris_3b_depth").unwrap();
        assert_eq!(e.task, IrisTask::Depth);
        assert_eq!(e.settings.name, "depth");
        assert_eq!(e.config_sha256.len(), 64);
        assert_eq!(
            e.config_sha256,
            format!("{:x}", Sha256::digest(DEPTH_CONFIG.as_bytes()))
        );
    }

    #[test]
    fn wrong_task_artifacts_are_typed_refusals() {
        let dir = tempfile::tempdir().unwrap();
        export(dir.path(), DEPTH_CONFIG);
        let unsupported =
            |spec: &LoadSpec| match TaskExport::from_spec(spec, IrisTask::Depth, "iris_3b_depth") {
                Err(Error::Unsupported(m)) => m,
                other => panic!("expected a typed refusal, got {other:?}"),
            };
        // a text-encoder component
        let mut spec = LoadSpec::new(WeightsSource::Dir(dir.path().to_path_buf()));
        spec.components.insert(
            super::super::TEXT_ENCODER_COMPONENT.into(),
            WeightsSource::Dir(dir.path().to_path_buf()),
        );
        assert!(unsupported(&spec).contains("text_encoder"));
        // LoadSpec::text_encoder
        let mut spec = LoadSpec::new(WeightsSource::Dir(dir.path().to_path_buf()));
        spec.text_encoder = Some(WeightsSource::Dir(dir.path().to_path_buf()));
        assert!(unsupported(&spec).contains("text encoder"));
        // the generation checkpoint (no `task` section)
        let gen = tempfile::tempdir().unwrap();
        export(gen.path(), "model:\n  patch_size: 16\n");
        let spec = LoadSpec::new(WeightsSource::Dir(gen.path().to_path_buf()));
        assert!(unsupported(&spec).contains("generation checkpoint"));
        // the restoration export
        let up = tempfile::tempdir().unwrap();
        export(up.path(), &DEPTH_CONFIG.replace("depth\n", "restoration\n"));
        let spec = LoadSpec::new(WeightsSource::Dir(up.path().to_path_buf()));
        assert!(unsupported(&spec).contains("'restoration' export"));
    }

    #[test]
    fn a_missing_empty_prompt_is_a_load_error() {
        let dir = tempfile::tempdir().unwrap();
        export(dir.path(), DEPTH_CONFIG);
        std::fs::remove_file(dir.path().join(EMPTY_PROMPT_FILE)).unwrap();
        let err = TaskExport::from_dir(dir.path(), IrisTask::Depth, "iris_3b_depth").unwrap_err();
        assert!(err.to_string().contains(EMPTY_PROMPT_FILE), "{err}");
    }

    #[test]
    fn empty_prompt_shapes_are_checked() {
        let config = IrisConfig::parse(
            "model:\n  text_len: 3\n  text_dim: 2\n  text_lap_num_layers: 1\n  text_lap_num_heads: 1\ntext_encoder:\n  dim: 2\n  max_length: 3\n  hidden_layers:\n  - 2\n",
        )
        .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(EMPTY_PROMPT_FILE);
        let write = |emb_shape: Vec<usize>, mask: &[u8]| {
            let n: usize = emb_shape.iter().product();
            let emb: Vec<u8> = (0..n).flat_map(|i| (i as f32).to_le_bytes()).collect();
            let views = [
                (
                    "embeddings",
                    safetensors::tensor::TensorView::new(safetensors::Dtype::F32, emb_shape, &emb)
                        .unwrap(),
                ),
                (
                    "mask",
                    safetensors::tensor::TensorView::new(
                        safetensors::Dtype::BOOL,
                        vec![1, mask.len()],
                        mask,
                    )
                    .unwrap(),
                ),
            ];
            safetensors::serialize_to_file(views, &None, &path).unwrap();
        };
        write(vec![1, 3, 1, 2], &[1, 1, 0]);
        let p = EmptyPrompt::read(&path, &config).unwrap();
        assert_eq!(p.mask, [1, 1, 0]);
        assert_eq!(p.shape, [1, 3, 1, 2]);
        assert_eq!(p.embeddings, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        write(vec![1, 3, 2, 2], &[1, 1, 0]);
        assert!(EmptyPrompt::read(&path, &config).is_err());
        write(vec![1, 3, 1, 2], &[0, 0, 0]);
        assert!(EmptyPrompt::read(&path, &config).is_err());
    }
}
