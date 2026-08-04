//! Exact HF-cache artifact pin for the calibrated FLUX.2 Klein BF16 ladder.

use std::collections::{BTreeSet, HashMap};
use std::io::{BufReader, Read};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use mlx_gen::gen_core::{Error as CoreError, Result as CoreResult};
use mlx_gen::{LoadSpec, WeightsSource};
use sha2::{Digest, Sha256};

const REVISIONS: &[&str] = &[
    crate::memory_strategy::KLEIN_CALIBRATED_REVISION,
    "acf05e8d5103838baba6a5e32dc91d6997a56023",
];
const TRUE_V2_TRANSFORMER_SHA256: &str =
    "72ae74528050cd97bf056568000fcb7915012b4d0fd0807de205513e0fdc64b9";

const FILES: &[(&str, &str)] = &[
    (
        "model_index.json",
        "554a39bf1019138dd0b28809b745df8ef33657c8",
    ),
    (
        "text_encoder/config.json",
        "dd0d909c489845c5042d866f0d9f258784a90304",
    ),
    (
        "text_encoder/generation_config.json",
        "f5af93d0c7764fa38e3b1cd3aaba1b6307246d3b",
    ),
    (
        "text_encoder/model-00001-of-00004.safetensors",
        "c0dc64934ae0f730ddc80d99af44968d01a89e8454df07d762096ea1356446bc",
    ),
    (
        "text_encoder/model-00002-of-00004.safetensors",
        "d58533b468c31caa0222540a8aefddea1d74dc6e1fee928da8556d3d85729d6e",
    ),
    (
        "text_encoder/model-00003-of-00004.safetensors",
        "74927fec432e050365bf757bb30348f560a44394efac89f492680b7c910b64fd",
    ),
    (
        "text_encoder/model-00004-of-00004.safetensors",
        "cb73f466fade5716702bda38d4e3b321c9358c39889e46fb9d613fb038bfcb2f",
    ),
    (
        "text_encoder/model.safetensors.index.json",
        "991332e2a7dada9479948259f715fe1f6c69db54",
    ),
    (
        "tokenizer/added_tokens.json",
        "b54f9135e44c1e81047e8d05cb027af8bc039eed",
    ),
    (
        "tokenizer/chat_template.jinja",
        "01be9b307daa2d425f7c168c9fb145a286e0afb4",
    ),
    (
        "tokenizer/merges.txt",
        "31349551d90c7606f325fe0f11bbb8bd5fa0d7c7",
    ),
    (
        "tokenizer/special_tokens_map.json",
        "ac23c0aaa2434523c494330aeb79c58395378103",
    ),
    (
        "tokenizer/tokenizer.json",
        "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    ),
    (
        "tokenizer/tokenizer_config.json",
        "ddaf69808214a44fdd26d3785b66c1367c78277a",
    ),
    (
        "tokenizer/vocab.json",
        "4783fe10ac3adce15ac8f358ef5462739852c569",
    ),
    (
        "transformer/config.json",
        "028532655afc211b01866fc5059b61a53af736be",
    ),
    (
        "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
        "cb942a7072865a1d06e47a3361f9ba8746e68ad207c8499083bcb735869f5102",
    ),
    (
        "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
        "ca568a31d19c03ddbcfd8b2d4ec7dbd16dcefbaa50b7aef1b8ceefd6e6eb0970",
    ),
    (
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        "53b7e151560e69b6b242208b98da4eb25b0111c6",
    ),
    (
        "vae/config.json",
        "c3f38eb4c188a96a462519159356bd0fcc28cd14",
    ),
    (
        "vae/diffusion_pytorch_model.safetensors",
        "ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04",
    ),
];

fn expected_sha256<'a>(relative: &str, blob: &'a str) -> &'a str {
    match relative {
        "model_index.json" => "51a76cb1cf3ed37423a1128c79c22faee8e6fbe7f5aaeb737f0a258930dbaac0",
        "text_encoder/config.json" => {
            "57866e90a1d6328a7ed53eca732bce106f86c76ab99d7629e01f0a319fa57998"
        }
        "text_encoder/generation_config.json" => {
            "4347b1aeed2b2b78bc059920a0b7f5fec71482e1344952b76d7665d638d71f13"
        }
        "text_encoder/model.safetensors.index.json" => {
            "e7e4bdc58d3302c97357f27979b270987dd620cf0cd9b1c130e3a51c9d64df95"
        }
        "tokenizer/added_tokens.json" => {
            "c0284b582e14987fbd3d5a2cb2bd139084371ed9acbae488829a1c900833c680"
        }
        "tokenizer/chat_template.jinja" => {
            "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
        }
        "tokenizer/merges.txt" => {
            "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"
        }
        "tokenizer/special_tokens_map.json" => {
            "76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd"
        }
        "tokenizer/tokenizer_config.json" => {
            "443bfa629eb16387a12edbf92a76f6a6f10b2af3b53d87ba1550adfcf45f7fa0"
        }
        "tokenizer/vocab.json" => {
            "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"
        }
        "transformer/config.json" => {
            "e82d0d325aff03c3b3b33a1634c47a5f88867478f53071b9c9a39c99010c5d46"
        }
        "transformer/diffusion_pytorch_model.safetensors.index.json" => {
            "96fe598b12bdfa266f54c2a6027dedc6e87016cce7e19cb2821a8b7deaf9deb9"
        }
        "vae/config.json" => "0d6dfb69ae95a5e2ac9836284bbb63d8b38ce67b25ba2dff380752b2a10ab948",
        _ => blob,
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Identity {
    canonical: PathBuf,
    device: u64,
    inode: u64,
    size: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
}

fn identity(path: &Path) -> std::io::Result<Identity> {
    let canonical = std::fs::canonicalize(path)?;
    let metadata = std::fs::metadata(&canonical)?;
    Ok(Identity {
        canonical,
        device: metadata.dev(),
        inode: metadata.ino(),
        size: metadata.size(),
        modified_seconds: metadata.mtime(),
        modified_nanoseconds: metadata.mtime_nsec(),
        changed_seconds: metadata.ctime(),
        changed_nanoseconds: metadata.ctime_nsec(),
    })
}

fn digest_cache() -> &'static Mutex<HashMap<Identity, String>> {
    static CACHE: OnceLock<Mutex<HashMap<Identity, String>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn sha256(file: &Path, id: &Identity) -> CoreResult<String> {
    if let Some(value) = digest_cache()
        .lock()
        .map_err(|_| CoreError::Msg("flux2 artifact digest cache poisoned".to_owned()))?
        .get(id)
        .cloned()
    {
        return Ok(value);
    }
    let file = std::fs::File::open(file)
        .map_err(|error| CoreError::Msg(format!("open {} for hashing: {error}", file.display())))?;
    let mut reader = BufReader::with_capacity(8 * 1024 * 1024, file);
    let mut buffer = vec![0_u8; 8 * 1024 * 1024];
    let mut hash = Sha256::new();
    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|error| CoreError::Msg(format!("hash {}: {error}", id.canonical.display())))?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    let value = format!("{:x}", hash.finalize());
    if identity(&id.canonical).ok().as_ref() != Some(id) {
        return Err(CoreError::Unsupported(
            "flux2 calibrated artifact changed while it was hashed".to_owned(),
        ));
    }
    digest_cache()
        .lock()
        .map_err(|_| CoreError::Msg("flux2 artifact digest cache poisoned".to_owned()))?
        .insert(id.clone(), value.clone());
    Ok(value)
}

#[derive(Clone, Debug)]
struct Entry {
    source: PathBuf,
    link_target: Option<PathBuf>,
    identity: Identity,
}

#[derive(Clone, Debug)]
pub(crate) struct KleinArtifactInventory {
    root: PathBuf,
    entries: Vec<Entry>,
    true_v2: bool,
}

impl KleinArtifactInventory {
    pub(crate) fn verify(spec: &LoadSpec) -> CoreResult<Option<Self>> {
        let WeightsSource::Dir(root) = &spec.weights else {
            return Ok(None);
        };
        let root = std::path::absolute(root)
            .map_err(|error| CoreError::Msg(format!("absolute FLUX.2 root: {error}")))?;
        if root
            .join("transformer/diffusion_pytorch_model.safetensors")
            .is_file()
            && std::fs::symlink_metadata(root.join("text_encoder"))
                .map(|metadata| metadata.file_type().is_symlink())
                .unwrap_or(false)
        {
            return Self::verify_true_v2(root).map(Some);
        }
        if !root.components().any(|part| {
            REVISIONS
                .iter()
                .any(|revision| part.as_os_str() == *revision)
        }) {
            return Ok(None);
        }
        let expected = FILES.iter().map(|(path, _)| *path).collect::<BTreeSet<_>>();
        let mut visible = BTreeSet::new();
        for directory in ["", "text_encoder", "tokenizer", "transformer", "vae"] {
            let directory = root.join(directory);
            for item in std::fs::read_dir(&directory).map_err(|error| {
                CoreError::Msg(format!(
                    "read calibrated directory {}: {error}",
                    directory.display()
                ))
            })? {
                let path = item
                    .map_err(|error| CoreError::Msg(error.to_string()))?
                    .path();
                if std::fs::symlink_metadata(&path)
                    .map(|metadata| metadata.file_type().is_symlink())
                    .unwrap_or(false)
                {
                    visible.insert(
                        path.strip_prefix(&root)
                            .unwrap()
                            .to_string_lossy()
                            .into_owned(),
                    );
                }
            }
        }
        let expected_owned = expected
            .iter()
            .map(|path| (*path).to_owned())
            .collect::<BTreeSet<_>>();
        if visible != expected_owned {
            return Err(CoreError::Unsupported(
                "flux2 calibrated HF snapshot membership does not match the measured BF16 inventory"
                    .to_owned(),
            ));
        }
        let mut entries = Vec::with_capacity(FILES.len());
        for &(relative, expected_blob) in FILES {
            let source = root.join(relative);
            let link_target = std::fs::read_link(&source).map_err(|error| {
                CoreError::Msg(format!(
                    "read calibrated link {}: {error}",
                    source.display()
                ))
            })?;
            if link_target.file_name().and_then(|name| name.to_str()) != Some(expected_blob) {
                return Err(CoreError::Unsupported(format!(
                    "flux2 calibrated snapshot entry {relative} resolves to an unexpected HF blob"
                )));
            }
            let id = identity(&source).map_err(|error| {
                CoreError::Msg(format!("stat calibrated entry {relative}: {error}"))
            })?;
            if sha256(&source, &id)? != expected_sha256(relative, expected_blob) {
                return Err(CoreError::Unsupported(format!(
                    "flux2 calibrated snapshot entry {relative} failed its SHA-256 content pin"
                )));
            }
            entries.push(Entry {
                source,
                link_target: Some(link_target),
                identity: id,
            });
        }
        let inventory = Self {
            root,
            entries,
            true_v2: false,
        };
        inventory.ensure_unchanged()?;
        Ok(Some(inventory))
    }

    fn verify_true_v2(root: PathBuf) -> CoreResult<Self> {
        let text_target = std::fs::read_link(root.join("text_encoder"))
            .map_err(|error| CoreError::Msg(format!("read True-V2 text-encoder link: {error}")))?;
        let base_root = text_target.parent().ok_or_else(|| {
            CoreError::Unsupported("True-V2 text-encoder link has no base root".to_owned())
        })?;
        let base_spec = LoadSpec::new(WeightsSource::Dir(base_root.to_path_buf()));
        let base = Self::verify(&base_spec)?.ok_or_else(|| {
            CoreError::Unsupported(
                "True-V2 borrowed components are not the exact calibrated Klein base".to_owned(),
            )
        })?;
        let mut entries = base.entries.clone();
        for directory in ["vae", "text_encoder", "tokenizer", "scheduler"] {
            let source = root.join(directory);
            let link_target = std::fs::read_link(&source).map_err(|error| {
                CoreError::Msg(format!("read True-V2 {directory} link: {error}"))
            })?;
            let expected = base_root.join(directory);
            if std::fs::canonicalize(&source).ok() != std::fs::canonicalize(&expected).ok() {
                return Err(CoreError::Unsupported(format!(
                    "True-V2 {directory} does not borrow the admitted Klein base"
                )));
            }
            entries.push(Entry {
                identity: identity(&source).map_err(|error| CoreError::Msg(error.to_string()))?,
                source,
                link_target: Some(link_target),
            });
        }
        for (relative, expected) in [
            ("model_index.json", expected_sha256("model_index.json", "")),
            (
                "transformer/config.json",
                expected_sha256("transformer/config.json", ""),
            ),
            (
                "transformer/diffusion_pytorch_model.safetensors",
                TRUE_V2_TRANSFORMER_SHA256,
            ),
        ] {
            let source = root.join(relative);
            let id = identity(&source)
                .map_err(|error| CoreError::Msg(format!("stat True-V2 {relative}: {error}")))?;
            if sha256(&source, &id)? != expected {
                return Err(CoreError::Unsupported(format!(
                    "True-V2 {relative} failed its exact converted-content pin"
                )));
            }
            entries.push(Entry {
                source,
                link_target: None,
                identity: id,
            });
        }
        let inventory = Self {
            root,
            entries,
            true_v2: true,
        };
        inventory.ensure_unchanged()?;
        Ok(inventory)
    }

    pub(crate) fn transformer_dir(&self) -> PathBuf {
        self.root.join("transformer")
    }

    pub(crate) fn calibration_tag(&self) -> &'static str {
        if self.true_v2 {
            "true-two"
        } else {
            "base"
        }
    }

    pub(crate) fn ensure_unchanged(&self) -> CoreResult<()> {
        for entry in &self.entries {
            if entry.link_target.as_ref().is_some_and(|target| {
                std::fs::read_link(&entry.source).ok().as_ref() != Some(target)
            }) || identity(&entry.source).ok().as_ref() != Some(&entry.identity)
            {
                return Err(CoreError::Unsupported(
                    "flux2 calibrated HF snapshot changed after admission".to_owned(),
                ));
            }
        }
        Ok(())
    }
}
