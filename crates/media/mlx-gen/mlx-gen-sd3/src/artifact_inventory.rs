//! Exact immutable HF-cache inventory for the calibrated SD3.5-Large BF16 ladder.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::{BufReader, Read};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use mlx_gen::{Error, LoadSpec, Result, WeightsSource};
use sha2::{Digest, Sha256};

pub const CALIBRATED_REVISION: &str = "ceddf0a7fdf2064ea28e2213e3b84e4afa170a0f";

const FILES: &[(&str, &str)] = &[
    (
        "model_index.json",
        "d1e85db2887d95ee9e0f6206771a4c52f9233e2f",
    ),
    (
        "scheduler/scheduler_config.json",
        "eca5b2eed3727b3ad375eb20e467e4264120fd5d",
    ),
    (
        "text_encoder/config.json",
        "dbda34b6008657d7be7423051c47abd4c94e263a",
    ),
    (
        "text_encoder/model.safetensors",
        "71e183d11db0c6b6282a4d9e0abb74125edc8692393e89ed8ee5571005f35cb1",
    ),
    (
        "text_encoder_2/config.json",
        "35886d089e214b9e35db394d4d3d122a19baf231",
    ),
    (
        "text_encoder_2/model.safetensors",
        "ec310df2af79c318e24d20511b601a591ca8cd4f1fce1d8dff822a356bcdb1f4",
    ),
    (
        "text_encoder_3/config.json",
        "5f020057278e664f8d4875a172b8266a9707c249",
    ),
    (
        "text_encoder_3/model-00001-of-00002.safetensors",
        "4f2751ceeb2a96edd693e539dc5d6bba0b8d3814f49a9b3798403a0cec4b2e3d",
    ),
    (
        "text_encoder_3/model-00002-of-00002.safetensors",
        "f63154532130422309532ff56f11945fbea8266c958e3133e8e5aef85c6293c7",
    ),
    (
        "text_encoder_3/model.safetensors.index.json",
        "c8728bc3dca59d2616a2a594cbac3ddb9eb77d5b",
    ),
    (
        "tokenizer/merges.txt",
        "76e821f1b6f0a9709293c3b6b51ed90980b3166b",
    ),
    (
        "tokenizer/special_tokens_map.json",
        "cf0682d6de72c1547f41b4f6d7c59f62deffef94",
    ),
    (
        "tokenizer/tokenizer_config.json",
        "180a4e1be2a7d0b38a44108d6f24f585f35147b1",
    ),
    (
        "tokenizer/vocab.json",
        "469be27c5c010538f845f518c4f5e8574c78f7c8",
    ),
    (
        "tokenizer_2/merges.txt",
        "76e821f1b6f0a9709293c3b6b51ed90980b3166b",
    ),
    (
        "tokenizer_2/special_tokens_map.json",
        "b3b93aa8e7fd8552501067e63e04e2958298c2b2",
    ),
    (
        "tokenizer_2/tokenizer_config.json",
        "f4299b251265219ef1369503c04d837bd9528f0a",
    ),
    (
        "tokenizer_2/vocab.json",
        "469be27c5c010538f845f518c4f5e8574c78f7c8",
    ),
    (
        "tokenizer_3/special_tokens_map.json",
        "17ade346a1042cbe0c1436f5bedcbd85c099d582",
    ),
    (
        "tokenizer_3/spiece.model",
        "d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86",
    ),
    (
        "tokenizer_3/tokenizer.json",
        "6f055030106071d446e559644b285557a82fefd4",
    ),
    (
        "tokenizer_3/tokenizer_config.json",
        "15377e70a0e43e060a98bbc5cb6245664d906e6c",
    ),
    (
        "transformer/config.json",
        "feda980c13a51d63beaa5074de3e03bbadb5b0ab",
    ),
    (
        "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
        "8ec9c4af8093417fded1eec8638f313f5671c7dd0a17a3bf64807bee9954a3e0",
    ),
    (
        "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
        "a84bad7103aa22d8ddf6baafc7664868b1edcd316ca6dd9c7e1f667345bd99db",
    ),
    (
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        "8b54a2ceceea4454cc015aa5f808cf31a179e75f",
    ),
    (
        "vae/config.json",
        "059ae000f22d9c502030ead3c3f1db2718ae6f1f",
    ),
    (
        "vae/diffusion_pytorch_model.safetensors",
        "8f53304a79335b55e13ec50f63e5157fee4deb2f30d5fae0654e2b2653c109dc",
    ),
];

fn expected_sha256<'a>(relative: &str, blob: &'a str) -> &'a str {
    match relative {
        "model_index.json" => "b5e97ca0d3c0cd9a1e3ed91a71bbdd489de7941b23fe038b3aaa9acf36fd1543",
        "scheduler/scheduler_config.json" => {
            "677e1ed36ca0de4ac8cc766ca58869e8649673d5f1d47eb8f8aef511a18ae8f7"
        }
        "text_encoder/config.json" => {
            "3ae637e6409ed10c125e13b935461060d629ee069eea7cb8c3e9eee26b01ff1a"
        }
        "text_encoder_2/config.json" => {
            "786ac791f8ae2168eca583978d89e97b704579e4873a2ca36e2d8b5af561f625"
        }
        "text_encoder_3/config.json" => {
            "9035948b0904c536918a88b5d984a52cfca26a5c57efa9e90149790cf38c151e"
        }
        "text_encoder_3/model.safetensors.index.json" => {
            "3bacec0f0cf392399d4a385908f67dd73df99c9e9cfee669f148858ba9fbdb0a"
        }
        "tokenizer/merges.txt" | "tokenizer_2/merges.txt" => {
            "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a"
        }
        "tokenizer/special_tokens_map.json" => {
            "2cdb3b8331a60c92fc1e55a13e9fd61fd2293c5a51275fdcccd62b780052530e"
        }
        "tokenizer/tokenizer_config.json" => {
            "6bdcee9ccce2a16ca2b4c0c5ed00b42c50ea225f4472a8c4c1e963a2902c2881"
        }
        "tokenizer/vocab.json" | "tokenizer_2/vocab.json" => {
            "e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349"
        }
        "tokenizer_2/special_tokens_map.json" => {
            "c4dbb96da703fb38f10ccf0490df2fd476811c5a3e71b7e0189cffeed3224e25"
        }
        "tokenizer_2/tokenizer_config.json" => {
            "e70eb3e2b85a508e32bb81c32c13fc3f2b015548ca469b94627c2f32e445cba1"
        }
        "tokenizer_3/special_tokens_map.json" => {
            "7a1985a994c41886db38c719d2a3d2f40606663cc19d7c5d6a85d349320e06d2"
        }
        "tokenizer_3/tokenizer.json" => {
            "652ffdfc379606bad8edfb653f92dcf28e2e5dbf1cdfe50d685d29b2cad12dd6"
        }
        "tokenizer_3/tokenizer_config.json" => {
            "83b3e7737eef754efca607d9f580798cc35457daa96052cc60179f095a375c14"
        }
        "transformer/config.json" => {
            "ee8af543f7c9ec4219a536b3d6cbc8bd0b2e3374b0f6826dbe1ba534483787f9"
        }
        "transformer/diffusion_pytorch_model.safetensors.index.json" => {
            "d00651348ec872f9d13ccd6417572c590cb25e7a46e2d96e4145c5deb7498ce2"
        }
        "vae/config.json" => "58557f2439dfa867450caef425b5d11160be8aa9c34d60dbf23a94a6a94cb060",
        _ => blob,
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Identity {
    canonical: PathBuf,
    device: u64,
    inode: u64,
    size: u64,
    mtime: i64,
    mtime_ns: i64,
    ctime: i64,
    ctime_ns: i64,
}

fn identity(path: &Path) -> Result<Identity> {
    let canonical = std::fs::canonicalize(path)?;
    let metadata = std::fs::metadata(&canonical)?;
    Ok(Identity {
        canonical,
        device: metadata.dev(),
        inode: metadata.ino(),
        size: metadata.size(),
        mtime: metadata.mtime(),
        mtime_ns: metadata.mtime_nsec(),
        ctime: metadata.ctime(),
        ctime_ns: metadata.ctime_nsec(),
    })
}

fn digest_cache() -> &'static Mutex<HashMap<Identity, String>> {
    static CACHE: OnceLock<Mutex<HashMap<Identity, String>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn sha256(path: &Path, id: &Identity) -> Result<String> {
    if let Some(value) = digest_cache()
        .lock()
        .map_err(|_| Error::Msg("sd3 artifact digest cache poisoned".to_owned()))?
        .get(id)
        .cloned()
    {
        return Ok(value);
    }
    let file = std::fs::File::open(path)?;
    let mut reader = BufReader::with_capacity(8 * 1024 * 1024, file);
    let mut buffer = vec![0_u8; 8 * 1024 * 1024];
    let mut hash = Sha256::new();
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    let value = format!("{:x}", hash.finalize());
    if identity(path).ok().as_ref() != Some(id) {
        return Err(Error::Unsupported(
            "sd3 calibrated artifact changed while it was hashed".to_owned(),
        ));
    }
    digest_cache()
        .lock()
        .map_err(|_| Error::Msg("sd3 artifact digest cache poisoned".to_owned()))?
        .insert(id.clone(), value.clone());
    Ok(value)
}

fn visible_files(root: &Path) -> Result<BTreeSet<String>> {
    fn walk(root: &Path, dir: &Path, out: &mut BTreeSet<String>) -> Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                walk(root, &entry.path(), out)?;
            } else {
                out.insert(
                    entry
                        .path()
                        .strip_prefix(root)
                        .map_err(|error| Error::Msg(error.to_string()))?
                        .to_string_lossy()
                        .into_owned(),
                );
            }
        }
        Ok(())
    }
    let mut out = BTreeSet::new();
    walk(root, root, &mut out)?;
    Ok(out)
}

#[derive(Clone, Debug)]
struct Entry {
    source: PathBuf,
    link_target: PathBuf,
    identity: Identity,
}

#[derive(Clone, Debug)]
pub(crate) struct Sd3ArtifactInventory {
    root: PathBuf,
    entries: Vec<Entry>,
}

impl Sd3ArtifactInventory {
    pub(crate) fn verify(spec: &LoadSpec) -> Result<Option<Self>> {
        let WeightsSource::Dir(root) = &spec.weights else {
            return Ok(None);
        };
        let root = std::path::absolute(root)?;
        if !root
            .components()
            .any(|part| part.as_os_str() == CALIBRATED_REVISION)
        {
            return Ok(None);
        }
        let expected: BTreeMap<&str, &str> = FILES.iter().copied().collect();
        let expected_files = expected.keys().map(|path| (*path).to_owned()).collect();
        let visible = visible_files(&root)?;
        if visible != expected_files {
            return Err(Error::Unsupported(format!(
                "sd3 calibrated snapshot membership differs: expected {:?}, found {:?}",
                expected_files, visible
            )));
        }
        let mut entries = Vec::with_capacity(FILES.len());
        for (relative, blob) in &expected {
            let source = root.join(relative);
            let meta = std::fs::symlink_metadata(&source).map_err(|error| {
                Error::Msg(format!(
                    "sd3 calibrated artifact {}: {error}",
                    source.display()
                ))
            })?;
            if !meta.file_type().is_symlink() {
                return Err(Error::Unsupported(format!(
                    "sd3 calibrated artifact {} is not an immutable HF-cache link",
                    source.display()
                )));
            }
            let link_target = std::fs::read_link(&source)?;
            if link_target.file_name().and_then(|name| name.to_str()) != Some(blob) {
                return Err(Error::Unsupported(format!(
                    "sd3 calibrated artifact {} does not resolve to expected HF blob {blob}",
                    source.display()
                )));
            }
            let file_identity = identity(&source)?;
            let digest = sha256(&source, &file_identity)?;
            let expected_digest = expected_sha256(relative, blob);
            if digest != expected_digest {
                return Err(Error::Unsupported(format!(
                    "sd3 calibrated artifact {} has SHA-256 {digest}, expected {expected_digest}",
                    source.display()
                )));
            }
            entries.push(Entry {
                identity: file_identity,
                source,
                link_target,
            });
        }
        Ok(Some(Self { root, entries }))
    }

    pub(crate) fn transformer_dir(&self) -> PathBuf {
        self.root.join("transformer")
    }

    pub(crate) fn ensure_unchanged(&self) -> Result<()> {
        for entry in &self.entries {
            if std::fs::read_link(&entry.source).ok().as_ref() != Some(&entry.link_target)
                || identity(&entry.source).ok().as_ref() != Some(&entry.identity)
            {
                return Err(Error::Unsupported(format!(
                    "sd3 calibrated artifact changed after admission: {}",
                    entry.source.display()
                )));
            }
        }
        Ok(())
    }
}
