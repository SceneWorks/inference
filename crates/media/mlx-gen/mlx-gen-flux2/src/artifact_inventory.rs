//! Exact HF-cache artifact pin for the calibrated FLUX.2 Klein BF16 ladder.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::{BufReader, Read};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use mlx_gen::gen_core::weightsmeta::{
    safetensors_path_tensor_headers, Dtype, SafetensorsTensorHeader,
};
use mlx_gen::gen_core::{
    EncoderContract, Error as CoreError, Result as CoreResult, ValidatedEncoderSource,
};
use mlx_gen::{LoadSpec, Quant, WeightsSource};
use sha2::{Digest, Sha256};

const REVISIONS: &[&str] = &[
    crate::memory_strategy::KLEIN_CALIBRATED_REVISION,
    "1902693279fcfb828919370dfac2b8922d99499a",
];
pub(crate) const KLEIN_REHOST_REVISION: &str = "1902693279fcfb828919370dfac2b8922d99499a";
pub(crate) const KLEIN_KV_REHOST_REVISION: &str = "bbf22de8d654789de3b177632d2e283cc4f77729";
const KLEIN_REHOST_CACHE_DIR: &str = "models--SceneWorks--flux2-klein-9b-mlx";
const KLEIN_KV_REHOST_CACHE_DIR: &str = "models--SceneWorks--flux2-klein-9b-kv-mlx";
const TRUE_V2_TRANSFORMER_SHA256: &str =
    "72ae74528050cd97bf056568000fcb7915012b4d0fd0807de205513e0fdc64b9";

/// Discovery roots for a turnkey tier directory. A tier lives at
/// `<hub>/models--SceneWorks--<repo>/snapshots/<revision>/<tier>` and every file in it is a
/// symlink into the repository's sibling `blobs/` tree, so the shared helper authorizes the
/// repository directory alongside the tier itself (sc-22727); a target outside the repository
/// directory still fails confinement.
fn discovery_roots(root: &Path) -> CoreResult<Vec<PathBuf>> {
    Ok(mlx_gen::gen_core::hf_cache_discovery_roots(root)?)
}

/// A component's `(bits, group_size)` packed marker, or `None` for a dense component.
fn packed_marker(directory: &Path, component: &str) -> CoreResult<Option<(i32, i32)>> {
    match (
        mlx_gen::quant::packed_quant_bits_at(directory)
            .map_err(|error| CoreError::Unsupported(error.to_string()))?,
        mlx_gen::quant::packed_quant_group_size_at(directory)
            .map_err(|error| CoreError::Unsupported(error.to_string()))?,
    ) {
        (None, None) => Ok(None),
        (Some(bits), Some(group_size)) => Ok(Some((bits, group_size))),
        _ => Err(CoreError::Unsupported(format!(
            "flux2 Klein {component} quantization marker is incomplete"
        ))),
    }
}

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
    visible_safetensors: Vec<(PathBuf, BTreeSet<String>)>,
    kind: KleinArtifactKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KleinArtifactKind {
    CalibratedBase,
    TrueV2,
    BaseRehost(Option<Quant>),
    KvRehost(Option<Quant>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TurnkeyFamily {
    Base,
    Kv,
}

fn path_ends_with(root: &Path, suffix: &[&str]) -> bool {
    let components = root
        .components()
        .map(|component| component.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    components.len() >= suffix.len()
        && components[components.len() - suffix.len()..]
            .iter()
            .map(String::as_str)
            .eq(suffix.iter().copied())
}

fn turnkey_identity(root: &Path) -> Option<(TurnkeyFamily, Option<Quant>)> {
    for (family, cache_dir, revision) in [
        (
            TurnkeyFamily::Base,
            KLEIN_REHOST_CACHE_DIR,
            KLEIN_REHOST_REVISION,
        ),
        (
            TurnkeyFamily::Kv,
            KLEIN_KV_REHOST_CACHE_DIR,
            KLEIN_KV_REHOST_REVISION,
        ),
    ] {
        for (tier, quant) in [
            ("bf16", None),
            ("q4", Some(Quant::Q4)),
            ("q8", Some(Quant::Q8)),
        ] {
            if path_ends_with(root, &[cache_dir, "snapshots", revision, tier]) {
                return Some((family, quant));
            }
        }
    }
    None
}

fn visible_safetensors(directory: &Path) -> CoreResult<BTreeSet<String>> {
    let mut visible = BTreeSet::new();
    for entry in std::fs::read_dir(directory).map_err(|error| {
        CoreError::Msg(format!(
            "read FLUX.2 Klein component {}: {error}",
            directory.display()
        ))
    })? {
        let path = entry
            .map_err(|error| CoreError::Msg(error.to_string()))?
            .path();
        if path.extension().and_then(|extension| extension.to_str()) == Some("safetensors")
            && !path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with('.'))
        {
            visible.insert(
                path.file_name()
                    .expect("directory entry has a file name")
                    .to_string_lossy()
                    .into_owned(),
            );
        }
    }
    Ok(visible)
}

/// Every visible safetensors shard of one turnkey component, in name order, with the membership
/// set [`KleinArtifactInventory::ensure_unchanged`] re-checks. A packed q4/q8 component is one
/// file; the bf16 tier ships the dense Qwen3 encoder as four `model-0000N-of-00004.safetensors`
/// shards and the transformer as two, exactly as the upstream diffusers snapshot does.
fn component_safetensors(directory: &Path) -> CoreResult<(Vec<PathBuf>, BTreeSet<String>)> {
    let visible = visible_safetensors(directory)?;
    if visible.is_empty() {
        return Err(CoreError::Unsupported(format!(
            "flux2 Klein {} must contain at least one visible safetensors file",
            directory.display()
        )));
    }
    let shards = visible.iter().map(|name| directory.join(name)).collect();
    Ok((shards, visible))
}

/// The tensor headers of one component across all of its shards. A tensor name that appears in
/// two shards is refused: the loaders would resolve it to whichever shard they read last.
fn component_tensor_headers(shards: &[PathBuf]) -> CoreResult<Vec<SafetensorsTensorHeader>> {
    let mut seen = BTreeSet::new();
    let mut headers = Vec::new();
    for shard in shards {
        for header in safetensors_path_tensor_headers(shard)? {
            if !seen.insert(header.name.clone()) {
                return Err(CoreError::Unsupported(format!(
                    "flux2 Klein tensor {} appears in more than one shard of {}",
                    header.name,
                    shard.parent().unwrap_or(shard).display()
                )));
            }
            headers.push(header);
        }
    }
    Ok(headers)
}

fn pinned_entry(source: PathBuf) -> CoreResult<Entry> {
    let link_target = std::fs::symlink_metadata(&source)
        .map_err(|error| CoreError::Msg(format!("stat {}: {error}", source.display())))?
        .file_type()
        .is_symlink()
        .then(|| std::fs::read_link(&source))
        .transpose()
        .map_err(|error| CoreError::Msg(format!("read link {}: {error}", source.display())))?;
    let identity = identity(&source)
        .map_err(|error| CoreError::Msg(format!("pin {}: {error}", source.display())))?;
    Ok(Entry {
        source,
        link_target,
        identity,
    })
}

fn validate_dense_tensor_headers(
    headers: &[mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader],
    component: &str,
) -> CoreResult<()> {
    if headers.is_empty()
        || headers.iter().any(|header| {
            header.dtype == Dtype::U32
                || header.name.ends_with(".scales")
                || header.name.ends_with(".biases")
        })
    {
        return Err(CoreError::Unsupported(format!(
            "flux2 Klein {component} must be a non-empty dense inventory"
        )));
    }
    Ok(())
}

fn validate_exact_shape(
    component: &str,
    header: &mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader,
    expected: &[usize],
) -> CoreResult<()> {
    if header.shape != expected {
        return Err(CoreError::Unsupported(format!(
            "flux2 Klein {component} tensor {} has shape {:?}, expected {expected:?}",
            header.name, header.shape
        )));
    }
    Ok(())
}

fn validate_exact_tensor_names(
    component: &str,
    observed: impl Iterator<Item = String>,
    expected: impl Iterator<Item = String>,
) -> CoreResult<()> {
    let observed = observed.collect::<BTreeSet<_>>();
    let expected = expected.collect::<BTreeSet<_>>();
    if observed == expected {
        return Ok(());
    }
    let missing = expected.difference(&observed).cloned().collect::<Vec<_>>();
    let unexpected = observed.difference(&expected).cloned().collect::<Vec<_>>();
    Err(CoreError::Unsupported(format!(
        "flux2 Klein {component} tensor inventory is not exact: missing={missing:?} unexpected={unexpected:?}"
    )))
}

fn validate_vae_tensor_headers(
    headers: &[mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader],
) -> CoreResult<()> {
    validate_dense_tensor_headers(headers, "vae")?;
    let observed = headers
        .iter()
        .cloned()
        .map(|header| (header.name.clone(), header))
        .collect::<BTreeMap<_, _>>();
    let required = klein_required_vae_weights();
    validate_exact_tensor_names(
        "VAE",
        observed.keys().cloned(),
        required
            .keys()
            .cloned()
            .chain(std::iter::once("bn.num_batches_tracked".to_owned())),
    )?;
    let batch_counter = observed
        .get("bn.num_batches_tracked")
        .expect("exact VAE tensor-name equality requires the batch counter");
    if batch_counter.dtype != Dtype::I64 || !batch_counter.shape.is_empty() {
        return Err(CoreError::Unsupported(format!(
            "flux2 Klein VAE bn.num_batches_tracked must be an I64 scalar, found {:?} {:?}",
            batch_counter.dtype, batch_counter.shape
        )));
    }
    for (name, shape) in required {
        let header = observed.get(&name).ok_or_else(|| {
            CoreError::Unsupported(format!("flux2 Klein VAE is missing required tensor {name}"))
        })?;
        if !header.is_float() {
            return Err(CoreError::Unsupported(format!(
                "flux2 Klein VAE tensor {name} must be floating point"
            )));
        }
        validate_exact_shape("VAE", header, &shape)?;
    }
    Ok(())
}

fn validate_vae_headers(shards: &[PathBuf]) -> CoreResult<()> {
    validate_vae_tensor_headers(&component_tensor_headers(shards)?)
}

fn klein_required_vae_weights() -> BTreeMap<String, Vec<usize>> {
    fn insert(required: &mut BTreeMap<String, Vec<usize>>, name: String, shape: &[usize]) {
        required.insert(name, shape.to_vec());
    }
    fn add_resnet(
        required: &mut BTreeMap<String, Vec<usize>>,
        prefix: &str,
        input: usize,
        output: usize,
    ) {
        for suffix in ["norm1.weight", "norm1.bias"] {
            insert(required, format!("{prefix}.{suffix}"), &[input]);
        }
        insert(
            required,
            format!("{prefix}.conv1.weight"),
            &[output, input, 3, 3],
        );
        insert(required, format!("{prefix}.conv1.bias"), &[output]);
        for suffix in ["norm2.weight", "norm2.bias"] {
            insert(required, format!("{prefix}.{suffix}"), &[output]);
        }
        insert(
            required,
            format!("{prefix}.conv2.weight"),
            &[output, output, 3, 3],
        );
        insert(required, format!("{prefix}.conv2.bias"), &[output]);
        if input != output {
            insert(
                required,
                format!("{prefix}.conv_shortcut.weight"),
                &[output, input, 1, 1],
            );
            insert(required, format!("{prefix}.conv_shortcut.bias"), &[output]);
        }
    }
    fn add_attention(required: &mut BTreeMap<String, Vec<usize>>, prefix: &str, channels: usize) {
        for suffix in ["group_norm.weight", "group_norm.bias"] {
            insert(required, format!("{prefix}.{suffix}"), &[channels]);
        }
        for projection in ["to_q", "to_k", "to_v", "to_out.0"] {
            insert(
                required,
                format!("{prefix}.{projection}.weight"),
                &[channels, channels],
            );
            insert(required, format!("{prefix}.{projection}.bias"), &[channels]);
        }
    }

    let mut required = BTreeMap::new();
    for name in ["bn.running_mean", "bn.running_var"] {
        insert(&mut required, name.to_owned(), &[128]);
    }
    insert(
        &mut required,
        "quant_conv.weight".to_owned(),
        &[64, 64, 1, 1],
    );
    insert(&mut required, "quant_conv.bias".to_owned(), &[64]);
    insert(
        &mut required,
        "post_quant_conv.weight".to_owned(),
        &[32, 32, 1, 1],
    );
    insert(&mut required, "post_quant_conv.bias".to_owned(), &[32]);
    insert(
        &mut required,
        "encoder.conv_in.weight".to_owned(),
        &[128, 3, 3, 3],
    );
    insert(&mut required, "encoder.conv_in.bias".to_owned(), &[128]);

    let channels = [128, 256, 512, 512];
    let mut input = 128;
    for (block, output) in channels.into_iter().enumerate() {
        for resnet in 0..2 {
            add_resnet(
                &mut required,
                &format!("encoder.down_blocks.{block}.resnets.{resnet}"),
                input,
                output,
            );
            input = output;
        }
        if block < 3 {
            insert(
                &mut required,
                format!("encoder.down_blocks.{block}.downsamplers.0.conv.weight"),
                &[output, output, 3, 3],
            );
            insert(
                &mut required,
                format!("encoder.down_blocks.{block}.downsamplers.0.conv.bias"),
                &[output],
            );
        }
    }
    add_resnet(&mut required, "encoder.mid_block.resnets.0", 512, 512);
    add_attention(&mut required, "encoder.mid_block.attentions.0", 512);
    add_resnet(&mut required, "encoder.mid_block.resnets.1", 512, 512);
    for suffix in ["weight", "bias"] {
        insert(
            &mut required,
            format!("encoder.conv_norm_out.{suffix}"),
            &[512],
        );
    }
    insert(
        &mut required,
        "encoder.conv_out.weight".to_owned(),
        &[64, 512, 3, 3],
    );
    insert(&mut required, "encoder.conv_out.bias".to_owned(), &[64]);

    insert(
        &mut required,
        "decoder.conv_in.weight".to_owned(),
        &[512, 32, 3, 3],
    );
    insert(&mut required, "decoder.conv_in.bias".to_owned(), &[512]);
    add_resnet(&mut required, "decoder.mid_block.resnets.0", 512, 512);
    add_attention(&mut required, "decoder.mid_block.attentions.0", 512);
    add_resnet(&mut required, "decoder.mid_block.resnets.1", 512, 512);
    input = 512;
    for (block, output) in [512, 512, 256, 128].into_iter().enumerate() {
        for resnet in 0..3 {
            add_resnet(
                &mut required,
                &format!("decoder.up_blocks.{block}.resnets.{resnet}"),
                input,
                output,
            );
            input = output;
        }
        if block < 3 {
            insert(
                &mut required,
                format!("decoder.up_blocks.{block}.upsamplers.0.conv.weight"),
                &[output, output, 3, 3],
            );
            insert(
                &mut required,
                format!("decoder.up_blocks.{block}.upsamplers.0.conv.bias"),
                &[output],
            );
        }
    }
    for suffix in ["weight", "bias"] {
        insert(
            &mut required,
            format!("decoder.conv_norm_out.{suffix}"),
            &[128],
        );
    }
    insert(
        &mut required,
        "decoder.conv_out.weight".to_owned(),
        &[3, 128, 3, 3],
    );
    insert(&mut required, "decoder.conv_out.bias".to_owned(), &[3]);
    required
}

fn validate_transformer_tensor_headers(
    headers: &[mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader],
    quant: Option<Quant>,
) -> CoreResult<()> {
    if headers.is_empty() {
        return Err(CoreError::Unsupported(
            "flux2 Klein transformer inventory is empty".to_owned(),
        ));
    }
    let by_name = headers
        .iter()
        .map(|header| (header.name.as_str(), header))
        .collect::<std::collections::BTreeMap<_, _>>();
    let (linear_weights, dense_weights) = klein_required_transformer_weights();
    for (name, shape) in &dense_weights {
        let header = by_name.get(name.as_str()).ok_or_else(|| {
            CoreError::Unsupported(format!(
                "flux2 Klein transformer is missing required tensor {name}"
            ))
        })?;
        if !header.is_float() {
            return Err(CoreError::Unsupported(format!(
                "flux2 Klein transformer requires dense tensor {name}"
            )));
        }
        validate_exact_shape("transformer", header, shape)?;
    }
    match quant {
        None => {
            validate_dense_tensor_headers(headers, "transformer")?;
            validate_exact_tensor_names(
                "dense transformer",
                by_name.keys().map(|name| (*name).to_owned()),
                linear_weights.keys().chain(dense_weights.keys()).cloned(),
            )?;
            for (name, shape) in &linear_weights {
                let header = by_name.get(name.as_str()).ok_or_else(|| {
                    CoreError::Unsupported(format!(
                        "flux2 Klein transformer is missing required tensor {name}"
                    ))
                })?;
                if !header.is_float() {
                    return Err(CoreError::Unsupported(format!(
                        "flux2 Klein transformer tensor {name} must be floating point"
                    )));
                }
                validate_exact_shape("transformer", header, shape)?;
            }
            Ok(())
        }
        Some(Quant::Q4 | Quant::Q8) => {
            let expected = dense_weights
                .keys()
                .cloned()
                .chain(linear_weights.keys().flat_map(|weight| {
                    let base = weight
                        .strip_suffix(".weight")
                        .expect("linear inventory keys end in .weight");
                    [
                        weight.clone(),
                        format!("{base}.scales"),
                        format!("{base}.biases"),
                    ]
                }));
            validate_exact_tensor_names(
                "packed transformer",
                by_name.keys().map(|name| (*name).to_owned()),
                expected,
            )?;
            let mut packed = 0_usize;
            for header in headers {
                if let Some(base) = header.name.strip_suffix(".weight") {
                    if header.dtype == Dtype::U32 {
                        let scales = format!("{base}.scales");
                        let biases = format!("{base}.biases");
                        if !by_name.contains_key(scales.as_str())
                            || !by_name.contains_key(biases.as_str())
                        {
                            return Err(CoreError::Unsupported(format!(
                                "flux2 Klein packed transformer has an incomplete triple for {base}"
                            )));
                        }
                        packed += 1;
                    }
                }
                if let Some(base) = header
                    .name
                    .strip_suffix(".scales")
                    .or_else(|| header.name.strip_suffix(".biases"))
                {
                    let weight = format!("{base}.weight");
                    if by_name.get(weight.as_str()).map(|header| header.dtype) != Some(Dtype::U32) {
                        return Err(CoreError::Unsupported(format!(
                            "flux2 Klein packed transformer has orphan metadata for {base}"
                        )));
                    }
                }
            }
            if packed == 0 {
                return Err(CoreError::Unsupported(
                    "flux2 Klein packed transformer marker has no packed content".to_owned(),
                ));
            }
            for (weight, dense_shape) in &linear_weights {
                let base = weight
                    .strip_suffix(".weight")
                    .expect("linear inventory keys end in .weight");
                let scales = format!("{base}.scales");
                let biases = format!("{base}.biases");
                let weight_header = by_name.get(weight.as_str());
                let scales_header = by_name.get(scales.as_str());
                let biases_header = by_name.get(biases.as_str());
                if weight_header.map(|header| header.dtype) != Some(Dtype::U32)
                    || scales_header.map(|header| header.dtype) != Some(Dtype::BF16)
                    || biases_header.map(|header| header.dtype) != Some(Dtype::BF16)
                {
                    return Err(CoreError::Unsupported(format!(
                        "flux2 Klein packed transformer is missing the required U32/BF16 triple for {base}"
                    )));
                }
                let [output, input] = dense_shape.as_slice() else {
                    return Err(CoreError::Unsupported(format!(
                        "flux2 Klein linear shape definition for {base} is not rank 2"
                    )));
                };
                let packed_shape = [
                    *output,
                    input * quant.expect("packed branch").bits() as usize / 32,
                ];
                let affine_shape = [*output, input / 64];
                validate_exact_shape(
                    "packed transformer",
                    weight_header.expect("dtype checked"),
                    &packed_shape,
                )?;
                validate_exact_shape(
                    "packed transformer scales",
                    scales_header.expect("dtype checked"),
                    &affine_shape,
                )?;
                validate_exact_shape(
                    "packed transformer biases",
                    biases_header.expect("dtype checked"),
                    &affine_shape,
                )?;
            }
            Ok(())
        }
        Some(Quant::Nvfp4) => Err(CoreError::Unsupported(
            "flux2 Klein MLX does not support an NVFP4 artifact tier".to_owned(),
        )),
    }
}

fn validate_transformer_headers(shards: &[PathBuf], quant: Option<Quant>) -> CoreResult<()> {
    validate_transformer_tensor_headers(&component_tensor_headers(shards)?, quant)
}

fn klein_required_transformer_weights(
) -> (BTreeMap<String, Vec<usize>>, BTreeMap<String, Vec<usize>>) {
    let config = crate::config::Flux2Config::klein_9b();
    let inner = config.inner_dim();
    let mlp = (config.mlp_ratio * inner as f32) as usize;
    let mut linear = BTreeMap::new();
    let mut dense = BTreeMap::new();
    let mut add_linear = |name: String, output: usize, input: usize| {
        linear.insert(name, vec![output, input]);
    };
    add_linear(
        "time_guidance_embed.timestep_embedder.linear_1.weight".to_owned(),
        inner,
        config.timestep_channels,
    );
    add_linear(
        "time_guidance_embed.timestep_embedder.linear_2.weight".to_owned(),
        inner,
        inner,
    );
    for name in [
        "double_stream_modulation_img.linear.weight",
        "double_stream_modulation_txt.linear.weight",
    ] {
        add_linear(name.to_owned(), 6 * inner, inner);
    }
    add_linear(
        "single_stream_modulation.linear.weight".to_owned(),
        3 * inner,
        inner,
    );
    add_linear("x_embedder.weight".to_owned(), inner, config.in_channels);
    add_linear(
        "context_embedder.weight".to_owned(),
        inner,
        config.joint_attention_dim,
    );
    add_linear("norm_out.linear.weight".to_owned(), 2 * inner, inner);
    add_linear("proj_out.weight".to_owned(), config.out_channels, inner);
    for block in 0..config.num_double_layers {
        let prefix = format!("transformer_blocks.{block}");
        for name in [
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_q_proj",
            "attn.add_k_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
        ] {
            add_linear(format!("{prefix}.{name}.weight"), inner, inner);
        }
        for stream in ["ff", "ff_context"] {
            add_linear(
                format!("{prefix}.{stream}.linear_in.weight"),
                2 * mlp,
                inner,
            );
            add_linear(format!("{prefix}.{stream}.linear_out.weight"), inner, mlp);
        }
        for name in [
            "attn.norm_q.weight",
            "attn.norm_k.weight",
            "attn.norm_added_q.weight",
            "attn.norm_added_k.weight",
        ] {
            dense.insert(format!("{prefix}.{name}"), vec![config.head_dim]);
        }
    }
    for block in 0..config.num_single_layers {
        let prefix = format!("single_transformer_blocks.{block}.attn");
        add_linear(
            format!("{prefix}.to_qkv_mlp_proj.weight"),
            3 * inner + 2 * mlp,
            inner,
        );
        add_linear(format!("{prefix}.to_out.weight"), inner, inner + mlp);
        dense.insert(format!("{prefix}.norm_q.weight"), vec![config.head_dim]);
        dense.insert(format!("{prefix}.norm_k.weight"), vec![config.head_dim]);
    }
    (linear, dense)
}

impl KleinArtifactInventory {
    pub(crate) fn verify_for_provider(
        provider_id: &str,
        spec: &LoadSpec,
    ) -> CoreResult<Option<Self>> {
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
            let inventory = Self::verify_true_v2(root)?;
            inventory.validate_provider(provider_id)?;
            inventory.validate_resolved_route(spec.resolved_route.as_deref())?;
            return Ok(Some(inventory));
        }
        if let Some((family, tier)) = turnkey_identity(&root) {
            let inventory = Self::verify_turnkey(root, family, tier, spec)?;
            inventory.validate_provider(provider_id)?;
            inventory.validate_resolved_route(spec.resolved_route.as_deref())?;
            return Ok(Some(inventory));
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
            visible_safetensors: Vec::new(),
            kind: KleinArtifactKind::CalibratedBase,
        };
        inventory.ensure_unchanged()?;
        inventory.validate_provider(provider_id)?;
        inventory.validate_resolved_route(spec.resolved_route.as_deref())?;
        Ok(Some(inventory))
    }

    fn verify_turnkey(
        root: PathBuf,
        family: TurnkeyFamily,
        tier: Option<Quant>,
        spec: &LoadSpec,
    ) -> CoreResult<Self> {
        Self::verify_turnkey_with_contracts(
            root,
            family,
            tier,
            spec,
            crate::config::KLEIN_ENCODER_CONTRACT,
            validate_transformer_headers,
            validate_vae_headers,
        )
    }

    fn verify_turnkey_with_contracts(
        root: PathBuf,
        family: TurnkeyFamily,
        tier: Option<Quant>,
        spec: &LoadSpec,
        encoder_contract: mlx_gen::gen_core::EncoderContract,
        validate_transformer: impl Fn(&[PathBuf], Option<Quant>) -> CoreResult<()>,
        validate_vae: impl Fn(&[PathBuf]) -> CoreResult<()>,
    ) -> CoreResult<Self> {
        if spec.precision != mlx_gen::Precision::Bf16 || spec.quantize.is_some() {
            return Err(CoreError::Unsupported(
                "flux2 Klein turnkey tiers require BF16 execution with LoadSpec.quantize=None"
                    .to_owned(),
            ));
        }
        let observed = packed_marker(&root.join("transformer"), "transformer")?;
        let expected = tier.map(|quant| (quant.bits(), 64));
        if observed != expected {
            return Err(CoreError::Unsupported(format!(
                "flux2 Klein turnkey transformer quantization {:?} does not match resolved tier {:?}",
                observed, tier
            )));
        }
        // Every published tier ships the Qwen3 text encoder and the VAE dense; only the transformer
        // packs. A packed marker on either is a foreign or defective artifact, not a tier
        // (sc-22760: the corrected rehost revisions carry no such marker).
        for component in ["text_encoder", "vae"] {
            if packed_marker(&root.join(component), component)?.is_some() {
                return Err(CoreError::Unsupported(format!(
                    "flux2 Klein {component} must stay dense at every turnkey tier"
                )));
            }
        }
        let text_encoder_dir = root.join("text_encoder");
        let (text_encoder_shards, text_encoder_membership) =
            component_safetensors(&text_encoder_dir)?;
        encoder_contract.validate_source_for_discovery(
            &WeightsSource::Dir(text_encoder_dir.clone()),
            &discovery_roots(&root)?,
        )?;

        let tokenizer = root.join("tokenizer/tokenizer.json");
        let mut entries = vec![
            pinned_entry(tokenizer)?,
            pinned_entry(root.join("text_encoder/config.json"))?,
            pinned_entry(root.join("transformer/config.json"))?,
            pinned_entry(root.join("vae/config.json"))?,
        ];
        let mut visible = vec![(text_encoder_dir, text_encoder_membership)];
        for shard in text_encoder_shards {
            entries.push(pinned_entry(shard)?);
        }
        for component in ["transformer", "vae"] {
            let directory = root.join(component);
            let (shards, membership) = component_safetensors(&directory)?;
            match component {
                "transformer" => validate_transformer(&shards, tier)?,
                "vae" => validate_vae(&shards)?,
                _ => unreachable!("turnkey component list is exhaustive"),
            }
            for shard in shards {
                entries.push(pinned_entry(shard)?);
            }
            visible.push((directory, membership));
        }
        let inventory = Self {
            root,
            entries,
            visible_safetensors: visible,
            kind: match family {
                TurnkeyFamily::Base => KleinArtifactKind::BaseRehost(tier),
                TurnkeyFamily::Kv => KleinArtifactKind::KvRehost(tier),
            },
        };
        inventory.ensure_unchanged()?;
        Ok(inventory)
    }

    /// The bundled text encoder of a Klein snapshot, validated for one generator load.
    ///
    /// A pinned turnkey is verified first and a failed inventory does not fall back to the
    /// ordinary path: the load refuses with the inventory's reason. Every admitted Klein source —
    /// a dense-tagged artifact, a verified turnkey, an unpinned snapshot, or a
    /// [`LoadSpec::text_encoder`] override (which skips verification) — then takes
    /// [`EncoderContract::source_for_load`] unchanged; the inventory admits no text encoder the
    /// shared contract would refuse.
    pub(crate) fn text_encoder_source_for_load(
        encoder_contract: EncoderContract,
        provider_id: &str,
        spec: &LoadSpec,
        root: &Path,
    ) -> CoreResult<ValidatedEncoderSource> {
        if spec.text_encoder.is_none() {
            Self::verify_for_provider(provider_id, spec)?;
        }
        encoder_contract.source_for_load(spec, root)
    }

    fn validate_provider(&self, provider_id: &str) -> CoreResult<()> {
        let accepted = match self.kind {
            KleinArtifactKind::CalibratedBase
            | KleinArtifactKind::TrueV2
            | KleinArtifactKind::BaseRehost(_) => matches!(
                provider_id,
                crate::FLUX2_KLEIN_9B_ID | crate::FLUX2_KLEIN_9B_EDIT_ID
            ),
            KleinArtifactKind::KvRehost(_) => matches!(
                provider_id,
                crate::FLUX2_KLEIN_9B_ID | crate::FLUX2_KLEIN_9B_KV_EDIT_ID
            ),
        };
        if accepted {
            Ok(())
        } else {
            Err(CoreError::Unsupported(format!(
                "FLUX.2 Klein artifact {:?} does not belong to provider {provider_id}",
                self.kind
            )))
        }
    }

    fn validate_resolved_route(&self, resolved_route: Option<&str>) -> CoreResult<()> {
        let Some(resolved_route) = resolved_route else {
            return Ok(());
        };
        let expected = match self.kind {
            KleinArtifactKind::CalibratedBase | KleinArtifactKind::BaseRehost(_) => {
                "flux2_klein_9b"
            }
            KleinArtifactKind::TrueV2 => "flux2_klein_9b_true_v2",
            KleinArtifactKind::KvRehost(_) => "flux2_klein_9b_kv",
        };
        if resolved_route == expected {
            Ok(())
        } else {
            Err(CoreError::Unsupported(format!(
                "FLUX.2 Klein artifact {:?} belongs to resolved route {expected}, not {resolved_route}",
                self.kind
            )))
        }
    }

    fn verify_true_v2(root: PathBuf) -> CoreResult<Self> {
        let text_target = std::fs::read_link(root.join("text_encoder"))
            .map_err(|error| CoreError::Msg(format!("read True-V2 text-encoder link: {error}")))?;
        let base_root = text_target.parent().ok_or_else(|| {
            CoreError::Unsupported("True-V2 text-encoder link has no base root".to_owned())
        })?;
        let base_spec = LoadSpec::new(WeightsSource::Dir(base_root.to_path_buf()));
        let base =
            Self::verify_for_provider(crate::FLUX2_KLEIN_9B_ID, &base_spec)?.ok_or_else(|| {
                CoreError::Unsupported(
                    "True-V2 borrowed components are not the exact calibrated Klein base"
                        .to_owned(),
                )
            })?;
        if matches!(base.kind, KleinArtifactKind::KvRehost(_)) {
            return Err(CoreError::Unsupported(
                "True-V2 must borrow components from the exact base Klein artifact, not KV"
                    .to_owned(),
            ));
        }
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
            visible_safetensors: Vec::new(),
            kind: KleinArtifactKind::TrueV2,
        };
        inventory.ensure_unchanged()?;
        Ok(inventory)
    }

    pub(crate) fn transformer_dir(&self) -> PathBuf {
        self.root.join("transformer")
    }

    /// Measured-snapshot tag of the two dense tagged artifacts; `None` for the packed SceneWorks
    /// turnkeys, which are a family of tiers rather than one measured snapshot. The production
    /// identity is keyed on [`Self::artifact_tag`] for every kind (sc-22727).
    #[cfg(test)]
    pub(crate) fn calibration_tag(&self) -> Option<&'static str> {
        match self.kind {
            KleinArtifactKind::CalibratedBase | KleinArtifactKind::TrueV2 => {
                Some(self.artifact_tag())
            }
            KleinArtifactKind::BaseRehost(_) | KleinArtifactKind::KvRehost(_) => None,
        }
    }

    /// Artifact family segment of the production calibration identity (sc-22727): the two dense
    /// tagged artifacts are named by their [`Self::calibration_tag`], and the packed SceneWorks
    /// turnkeys by family, taking their tier from [`Self::resolved_quant`].
    pub(crate) fn artifact_tag(&self) -> &'static str {
        match self.kind {
            KleinArtifactKind::CalibratedBase => "base",
            KleinArtifactKind::TrueV2 => "true-two",
            KleinArtifactKind::BaseRehost(_) => "rehost",
            KleinArtifactKind::KvRehost(_) => "kv-rehost",
        }
    }

    pub(crate) fn resolved_quant(&self) -> Option<Quant> {
        match self.kind {
            KleinArtifactKind::BaseRehost(quant) | KleinArtifactKind::KvRehost(quant) => quant,
            KleinArtifactKind::TrueV2 => None,
            KleinArtifactKind::CalibratedBase => None,
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
        for (directory, expected) in &self.visible_safetensors {
            if &visible_safetensors(directory)? != expected {
                return Err(CoreError::Unsupported(format!(
                    "flux2 Klein {} safetensors membership changed after admission",
                    directory.display()
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transformer_inventory_uses_raw_diffusers_keys_before_loader_aliasing() {
        let (linear, _) = klein_required_transformer_weights();
        assert!(linear.contains_key("time_guidance_embed.timestep_embedder.linear_1.weight"));
        assert!(linear.contains_key("transformer_blocks.0.attn.to_out.0.weight"));
        assert!(!linear.contains_key("time_guidance_embed.linear_1.weight"));
        assert!(!linear.contains_key("transformer_blocks.0.attn.to_out.weight"));
    }

    #[test]
    fn vae_inventory_includes_the_exact_unused_pytorch_batch_counter() {
        assert_eq!(klein_required_vae_weights().len(), 250);
        let tensors = vae_tensors();
        assert_eq!(tensors.len(), 251);
        assert_eq!(
            tensors
                .iter()
                .find(|(name, _, _)| name == "bn.num_batches_tracked")
                .map(|(_, dtype, shape)| (*dtype, shape.as_slice())),
            Some(("I64", &[][..]))
        );
    }

    fn write_tensor_file(path: &Path, tensors: Vec<(String, &'static str, Vec<usize>)>) {
        use std::io::Write;

        let mut offset = 0_u64;
        let mut header = serde_json::Map::new();
        for (name, dtype, shape) in tensors {
            let width = match dtype {
                "U32" => 4_u64,
                "F16" => 2_u64,
                "BF16" => 2_u64,
                "I64" => 8_u64,
                _ => unreachable!("fixture dtypes are exhaustive"),
            };
            let bytes = shape
                .iter()
                .try_fold(width, |bytes, dimension| {
                    bytes.checked_mul(*dimension as u64)
                })
                .expect("fixture tensor extent");
            let end = offset.checked_add(bytes).expect("fixture file extent");
            header.insert(
                name,
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [offset, end],
                }),
            );
            offset = end;
        }
        let mut encoded = serde_json::to_vec(&header).unwrap();
        while !encoded.len().is_multiple_of(8) {
            encoded.push(b' ');
        }
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(&(encoded.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&encoded).unwrap();
        file.set_len(8 + encoded.len() as u64 + offset).unwrap();
    }

    fn tensor_headers(
        tensors: Vec<(String, &'static str, Vec<usize>)>,
    ) -> Vec<mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader> {
        tensors
            .into_iter()
            .map(|(name, dtype, shape)| {
                let dtype = match dtype {
                    "U32" => Dtype::U32,
                    "BF16" => Dtype::BF16,
                    "I64" => Dtype::I64,
                    _ => unreachable!("fixture dtypes are exhaustive"),
                };
                let width = match dtype {
                    Dtype::U32 => 4_u64,
                    Dtype::BF16 => 2_u64,
                    Dtype::I64 => 8_u64,
                    _ => unreachable!("fixture dtypes are exhaustive"),
                };
                let data_bytes = shape
                    .iter()
                    .try_fold(width, |bytes, dimension| {
                        bytes.checked_mul(*dimension as u64)
                    })
                    .expect("fixture tensor extent");
                mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader {
                    name,
                    dtype,
                    shape,
                    data_bytes,
                }
            })
            .collect()
    }

    fn write_safetensors(path: &Path, packed: bool) {
        let mut header = if packed {
            br#"{"block.weight":{"dtype":"U32","shape":[1],"data_offsets":[0,4]},"block.scales":{"dtype":"BF16","shape":[1],"data_offsets":[4,6]},"block.biases":{"dtype":"BF16","shape":[1],"data_offsets":[6,8]}}"#.to_vec()
        } else {
            br#"{"probe":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#.to_vec()
        };
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let payload = if packed { 8 } else { 2 };
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(vec![0_u8; payload]);
        std::fs::write(path, bytes).unwrap();
    }

    fn transformer_tensors(quant: Option<Quant>) -> Vec<(String, &'static str, Vec<usize>)> {
        let (linear, dense) = klein_required_transformer_weights();
        let mut tensors = Vec::new();
        for (name, shape) in dense {
            tensors.push((name, "BF16", shape));
        }
        for (weight, dense_shape) in linear {
            if let Some(quant) = quant {
                let base = weight.strip_suffix(".weight").unwrap().to_owned();
                let output = dense_shape[0];
                let input = dense_shape[1];
                tensors.push((
                    weight,
                    "U32",
                    vec![output, input * quant.bits() as usize / 32],
                ));
                tensors.push((format!("{base}.scales"), "BF16", vec![output, input / 64]));
                tensors.push((format!("{base}.biases"), "BF16", vec![output, input / 64]));
            } else {
                tensors.push((weight, "BF16", dense_shape));
            }
        }
        tensors
    }

    fn bounded_transformer_tensors(
        quant: Option<Quant>,
    ) -> Vec<(String, &'static str, Vec<usize>)> {
        match quant {
            None => vec![
                ("block.weight".to_owned(), "BF16", vec![8, 8]),
                ("block.bias".to_owned(), "BF16", vec![8]),
            ],
            Some(Quant::Q4 | Quant::Q8) => {
                let bits = quant.expect("packed tier").bits() as usize;
                vec![
                    ("block.weight".to_owned(), "U32", vec![8, 8 * bits / 32]),
                    ("block.scales".to_owned(), "BF16", vec![8, 1]),
                    ("block.biases".to_owned(), "BF16", vec![8, 1]),
                ]
            }
            Some(Quant::Nvfp4) => unreachable!(),
        }
    }

    fn write_transformer_safetensors(path: &Path, quant: Option<Quant>) {
        write_tensor_file(path, bounded_transformer_tensors(quant));
    }

    fn vae_tensors() -> Vec<(String, &'static str, Vec<usize>)> {
        let mut tensors = klein_required_vae_weights()
            .into_iter()
            .map(|(name, shape)| (name, "BF16", shape))
            .collect::<Vec<_>>();
        tensors.push(("bn.num_batches_tracked".to_owned(), "I64", Vec::new()));
        tensors
    }

    fn bounded_vae_tensors() -> Vec<(String, &'static str, Vec<usize>)> {
        vec![("probe".to_owned(), "BF16", vec![1])]
    }

    fn write_vae_safetensors(path: &Path) {
        write_tensor_file(path, bounded_vae_tensors());
    }

    fn quantization_config(bits: Option<i32>) -> String {
        bits.map_or_else(
            || "{}".to_owned(),
            |bits| format!(r#"{{"quantization":{{"bits":{bits},"group_size":64}}}}"#),
        )
    }

    fn validate_bounded_transformer(shards: &[PathBuf], quant: Option<Quant>) -> CoreResult<()> {
        let headers = component_tensor_headers(shards)?;
        let expected = tensor_headers(bounded_transformer_tensors(quant));
        if headers != expected {
            return Err(CoreError::Unsupported(
                "bounded transformer inventory is not exact".to_owned(),
            ));
        }
        Ok(())
    }

    fn validate_bounded_vae(shards: &[PathBuf]) -> CoreResult<()> {
        let headers = component_tensor_headers(shards)?;
        if headers != tensor_headers(bounded_vae_tensors()) {
            return Err(CoreError::Unsupported(
                "bounded VAE inventory is not exact".to_owned(),
            ));
        }
        Ok(())
    }

    /// One tier of a SceneWorks Klein rehost exactly as `huggingface_hub` lays it out on disk:
    /// `<hub>/models--SceneWorks--<repo>/snapshots/<revision>/<tier>/…`, every file a relative
    /// symlink into the repository's sibling `blobs/` tree. The Qwen3 text encoder is dense with
    /// no `quantization` marker at every tier (the corrected rehost revisions, sc-22760); bf16
    /// mirrors the shipped sharding (a multi-shard text encoder and transformer, each with its
    /// `*.index.json`), and q4/q8 pack only the transformer.
    fn turnkey_fixture(family: TurnkeyFamily, tier: Option<Quant>) -> tempfile::TempDir {
        let tmp = tempfile::tempdir().unwrap();
        let (cache, revision) = match family {
            TurnkeyFamily::Base => (KLEIN_REHOST_CACHE_DIR, KLEIN_REHOST_REVISION),
            TurnkeyFamily::Kv => (KLEIN_KV_REHOST_CACHE_DIR, KLEIN_KV_REHOST_REVISION),
        };
        let tier_dir = match tier {
            None => "bf16",
            Some(Quant::Q4) => "q4",
            Some(Quant::Q8) => "q8",
            Some(Quant::Nvfp4) => unreachable!(),
        };
        let repository = tmp.path().join(cache);
        let root = repository.join("snapshots").join(revision).join(tier_dir);
        for component in ["tokenizer", "text_encoder", "transformer", "vae"] {
            std::fs::create_dir_all(root.join(component)).unwrap();
        }
        let contract = crate::config::bounded_klein_encoder_contract();
        gen_core_testkit::write_encoder_contract_fixture(&root.join("text_encoder"), contract)
            .unwrap();
        if tier.is_none() {
            let headers =
                gen_core_testkit::encoder_contract_fixture_tensor_headers(contract, None).unwrap();
            std::fs::remove_file(root.join("text_encoder/model.safetensors")).unwrap();
            write_sharded_tensor_files(
                &root.join("text_encoder"),
                "model",
                header_tuples(headers),
                4,
            );
        }
        std::fs::write(root.join("vae/config.json"), "{}").unwrap();
        let transformer_config = tier.map_or_else(
            || "{}".to_owned(),
            |quant| {
                format!(
                    r#"{{"quantization":{{"bits":{},"group_size":64}}}}"#,
                    quant.bits()
                )
            },
        );
        std::fs::write(root.join("transformer/config.json"), transformer_config).unwrap();
        if tier.is_none() {
            write_sharded_tensor_files(
                &root.join("transformer"),
                "diffusion_pytorch_model",
                bounded_transformer_tensors(None),
                2,
            );
        } else {
            write_transformer_safetensors(
                &root.join("transformer/diffusion_pytorch_model.safetensors"),
                tier,
            );
        }
        write_vae_safetensors(&root.join("vae/diffusion_pytorch_model.safetensors"));
        relink_into_blobs(&repository, &root);
        tmp
    }

    fn header_tuples(
        headers: Vec<mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader>,
    ) -> Vec<(String, &'static str, Vec<usize>)> {
        headers
            .into_iter()
            .map(|header| {
                let dtype = match header.dtype {
                    Dtype::F16 => "F16",
                    Dtype::BF16 => "BF16",
                    Dtype::U32 => "U32",
                    other => unreachable!("bounded encoder fixture dtype {other:?}"),
                };
                (header.name, dtype, header.shape)
            })
            .collect()
    }

    /// `<stem>-0000N-of-0000M.safetensors` shards plus `<stem>.safetensors.index.json`, as
    /// diffusers writes a sharded component.
    fn write_sharded_tensor_files(
        directory: &Path,
        stem: &str,
        tensors: Vec<(String, &'static str, Vec<usize>)>,
        shards: usize,
    ) {
        assert!(
            tensors.len() >= shards,
            "{stem}: {} tensors for {shards} shards",
            tensors.len()
        );
        let per_shard = tensors.len().div_ceil(shards);
        let mut weight_map = serde_json::Map::new();
        for (index, chunk) in tensors.chunks(per_shard).enumerate() {
            let name = format!("{stem}-{:05}-of-{shards:05}.safetensors", index + 1);
            for (tensor, _, _) in chunk {
                weight_map.insert(tensor.clone(), serde_json::json!(name));
            }
            write_tensor_file(&directory.join(name), chunk.to_vec());
        }
        std::fs::write(
            directory.join(format!("{stem}.safetensors.index.json")),
            serde_json::to_vec(&serde_json::json!({"metadata": {}, "weight_map": weight_map}))
                .unwrap(),
        )
        .unwrap();
    }

    /// Move every regular file under `root` into `<repository>/blobs/<sha256>` and leave the
    /// relative symlink `huggingface_hub` would have written in its place.
    fn relink_into_blobs(repository: &Path, root: &Path) {
        fn walk(path: &Path, files: &mut Vec<PathBuf>) {
            for entry in std::fs::read_dir(path).unwrap() {
                let path = entry.unwrap().path();
                let metadata = std::fs::symlink_metadata(&path).unwrap();
                if metadata.is_dir() {
                    walk(&path, files);
                } else if metadata.is_file() {
                    files.push(path);
                }
            }
        }
        let blobs = repository.join("blobs");
        std::fs::create_dir_all(&blobs).unwrap();
        let mut files = Vec::new();
        walk(root, &mut files);
        for file in files {
            let bytes = std::fs::read(&file).unwrap();
            let blob = format!("{:x}", Sha256::digest(&bytes));
            let target = blobs.join(&blob);
            if !target.exists() {
                std::fs::rename(&file, &target).unwrap();
            } else {
                std::fs::remove_file(&file).unwrap();
            }
            let depth = file
                .parent()
                .unwrap()
                .strip_prefix(repository)
                .unwrap()
                .components()
                .count();
            let mut relative = PathBuf::new();
            for _ in 0..depth {
                relative.push("..");
            }
            relative.push("blobs");
            relative.push(&blob);
            std::os::unix::fs::symlink(relative, &file).unwrap();
        }
    }

    fn fixture_root(
        tmp: &tempfile::TempDir,
        family: TurnkeyFamily,
        tier: Option<Quant>,
    ) -> PathBuf {
        let (cache, revision) = match family {
            TurnkeyFamily::Base => (KLEIN_REHOST_CACHE_DIR, KLEIN_REHOST_REVISION),
            TurnkeyFamily::Kv => (KLEIN_KV_REHOST_CACHE_DIR, KLEIN_KV_REHOST_REVISION),
        };
        tmp.path()
            .join(cache)
            .join("snapshots")
            .join(revision)
            .join(match tier {
                None => "bf16",
                Some(Quant::Q4) => "q4",
                Some(Quant::Q8) => "q8",
                Some(Quant::Nvfp4) => unreachable!(),
            })
    }

    fn verify_bounded_turnkey(
        root: PathBuf,
        family: TurnkeyFamily,
        tier: Option<Quant>,
        spec: &LoadSpec,
    ) -> CoreResult<KleinArtifactInventory> {
        KleinArtifactInventory::verify_turnkey_with_contracts(
            root,
            family,
            tier,
            spec,
            crate::config::bounded_klein_encoder_contract(),
            validate_bounded_transformer,
            validate_bounded_vae,
        )
    }

    struct ImmutableTurnkeyFixture {
        _tmp: tempfile::TempDir,
        root: PathBuf,
        inventory: KleinArtifactInventory,
    }

    fn immutable_turnkey_fixture(
        family: TurnkeyFamily,
        tier: Option<Quant>,
    ) -> &'static ImmutableTurnkeyFixture {
        static BASE_BF16: OnceLock<ImmutableTurnkeyFixture> = OnceLock::new();
        static BASE_Q4: OnceLock<ImmutableTurnkeyFixture> = OnceLock::new();
        static BASE_Q8: OnceLock<ImmutableTurnkeyFixture> = OnceLock::new();
        static KV_BF16: OnceLock<ImmutableTurnkeyFixture> = OnceLock::new();
        static KV_Q4: OnceLock<ImmutableTurnkeyFixture> = OnceLock::new();
        static KV_Q8: OnceLock<ImmutableTurnkeyFixture> = OnceLock::new();

        let fixture = match (family, tier) {
            (TurnkeyFamily::Base, None) => &BASE_BF16,
            (TurnkeyFamily::Base, Some(Quant::Q4)) => &BASE_Q4,
            (TurnkeyFamily::Base, Some(Quant::Q8)) => &BASE_Q8,
            (TurnkeyFamily::Kv, None) => &KV_BF16,
            (TurnkeyFamily::Kv, Some(Quant::Q4)) => &KV_Q4,
            (TurnkeyFamily::Kv, Some(Quant::Q8)) => &KV_Q8,
            (_, Some(Quant::Nvfp4)) => unreachable!(),
        };
        fixture.get_or_init(|| {
            let tmp = turnkey_fixture(family, tier);
            let root = fixture_root(&tmp, family, tier);
            let provider_id = match family {
                TurnkeyFamily::Base => crate::FLUX2_KLEIN_9B_ID,
                TurnkeyFamily::Kv => crate::FLUX2_KLEIN_9B_KV_EDIT_ID,
            };
            let inventory = verify_bounded_turnkey(
                root.clone(),
                family,
                tier,
                &LoadSpec::new(WeightsSource::Dir(root.clone())),
            )
            .unwrap();
            inventory.validate_provider(provider_id).unwrap();
            ImmutableTurnkeyFixture {
                _tmp: tmp,
                root,
                inventory,
            }
        })
    }

    #[test]
    fn turnkey_inventory_binds_family_tier_provider_and_production_quantize_none() {
        for tier in [None, Some(Quant::Q4), Some(Quant::Q8)] {
            validate_transformer_tensor_headers(&tensor_headers(transformer_tensors(tier)), tier)
                .unwrap();
        }
        validate_vae_tensor_headers(&tensor_headers(vae_tensors())).unwrap();

        for family in [TurnkeyFamily::Base, TurnkeyFamily::Kv] {
            for tier in [None, Some(Quant::Q4), Some(Quant::Q8)] {
                let fixture = immutable_turnkey_fixture(family, tier);
                let spec = LoadSpec::new(WeightsSource::Dir(fixture.root.clone()));
                let allowed = match family {
                    TurnkeyFamily::Base => {
                        [crate::FLUX2_KLEIN_9B_ID, crate::FLUX2_KLEIN_9B_EDIT_ID]
                    }
                    TurnkeyFamily::Kv => {
                        [crate::FLUX2_KLEIN_9B_ID, crate::FLUX2_KLEIN_9B_KV_EDIT_ID]
                    }
                };
                for provider_id in allowed {
                    fixture.inventory.validate_provider(provider_id).unwrap();
                }
                assert_eq!(fixture.inventory.resolved_quant(), tier);
                assert!(fixture.inventory.calibration_tag().is_none());
                let resolved_route = match family {
                    TurnkeyFamily::Base => "flux2_klein_9b",
                    TurnkeyFamily::Kv => "flux2_klein_9b_kv",
                };
                fixture
                    .inventory
                    .validate_resolved_route(Some(resolved_route))
                    .unwrap();
                for wrong_route in [
                    "flux2_klein_9b",
                    "flux2_klein_9b_kv",
                    "flux2_klein_9b_true_v2",
                ]
                .into_iter()
                .filter(|route| *route != resolved_route)
                {
                    assert!(fixture
                        .inventory
                        .validate_resolved_route(Some(wrong_route))
                        .is_err());
                }
                let refused = match family {
                    TurnkeyFamily::Base => crate::FLUX2_KLEIN_9B_KV_EDIT_ID,
                    TurnkeyFamily::Kv => crate::FLUX2_KLEIN_9B_EDIT_ID,
                };
                assert!(fixture.inventory.validate_provider(refused).is_err());
                assert!(verify_bounded_turnkey(
                    fixture.root.clone(),
                    family,
                    tier,
                    &spec.clone().with_quant(Quant::Q4),
                )
                .is_err());
                fixture.inventory.ensure_unchanged().unwrap();
            }
        }
    }

    #[test]
    fn turnkey_inventory_rejects_quant_config_membership_and_identity_mutation() {
        let tmp = turnkey_fixture(TurnkeyFamily::Base, Some(Quant::Q4));
        let root = fixture_root(&tmp, TurnkeyFamily::Base, Some(Quant::Q4));
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
        let transformer_dir = root.join("transformer");
        let (mut shards, membership) = component_safetensors(&transformer_dir).unwrap();
        assert_eq!(shards.len(), 1, "a packed tier ships one transformer shard");
        let transformer = shards.remove(0);
        let inventory = KleinArtifactInventory {
            root: root.clone(),
            entries: vec![pinned_entry(transformer.clone()).unwrap()],
            visible_safetensors: vec![(transformer_dir, membership)],
            kind: KleinArtifactKind::BaseRehost(Some(Quant::Q4)),
        };
        inventory.ensure_unchanged().unwrap();

        write_safetensors(&root.join("transformer/extra.safetensors"), true);
        assert!(inventory.ensure_unchanged().is_err());
        std::fs::remove_file(root.join("transformer/extra.safetensors")).unwrap();
        inventory.ensure_unchanged().unwrap();
        write_safetensors(&transformer, true);
        assert!(inventory.ensure_unchanged().is_err());
        assert!(
            validate_transformer_headers(std::slice::from_ref(&transformer), Some(Quant::Q4))
                .is_err()
        );
        write_transformer_safetensors(&transformer, Some(Quant::Q4));
        std::fs::write(
            root.join("transformer/config.json"),
            r#"{"quantization":{"bits":8,"group_size":64}}"#,
        )
        .unwrap();
        assert!(
            verify_bounded_turnkey(root, TurnkeyFamily::Base, Some(Quant::Q4), &spec,).is_err()
        );

        let tmp = turnkey_fixture(TurnkeyFamily::Kv, Some(Quant::Q8));
        let root = fixture_root(&tmp, TurnkeyFamily::Kv, Some(Quant::Q8));
        std::fs::write(
            root.join("transformer/config.json"),
            r#"{"quantization":{"bits":8,"group_size":32}}"#,
        )
        .unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
        assert!(verify_bounded_turnkey(root, TurnkeyFamily::Kv, Some(Quant::Q8), &spec,).is_err());

        for malformed in [
            r#"{"quantization":{}}"#,
            r#"{"quantization":{"bits":"4","group_size":64}}"#,
            r#"{"quantization":{"bits":4}}"#,
        ] {
            let tmp = turnkey_fixture(TurnkeyFamily::Base, Some(Quant::Q4));
            let root = fixture_root(&tmp, TurnkeyFamily::Base, Some(Quant::Q4));
            std::fs::write(root.join("transformer/config.json"), malformed).unwrap();
            let spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
            assert!(
                verify_bounded_turnkey(root, TurnkeyFamily::Base, Some(Quant::Q4), &spec,).is_err()
            );
        }
    }

    #[test]
    fn turnkey_inventory_rejects_transformer_vae_and_encoder_shape_mutations() {
        let mut transformer = transformer_tensors(Some(Quant::Q4));
        transformer
            .iter_mut()
            .find(|(name, _, _)| name == "transformer_blocks.0.attn.to_q.weight")
            .unwrap()
            .2[0] = 1;
        assert!(
            validate_transformer_tensor_headers(&tensor_headers(transformer), Some(Quant::Q4),)
                .is_err()
        );

        let mut vae = vae_tensors();
        vae.iter_mut()
            .find(|(name, _, _)| name == "decoder.conv_out.weight")
            .unwrap()
            .2[1] = 1;
        assert!(validate_vae_tensor_headers(&tensor_headers(vae)).is_err());

        let tmp = turnkey_fixture(TurnkeyFamily::Base, Some(Quant::Q8));
        let root = fixture_root(&tmp, TurnkeyFamily::Base, Some(Quant::Q8));
        let bounded_contract = crate::config::bounded_klein_encoder_contract();
        let encoder_source = WeightsSource::Dir(root.join("text_encoder"));
        let validated = bounded_contract
            .validate_source_for_planning(&encoder_source)
            .expect("the bounded encoder fixture must be valid before mutation");
        let encoder_file = component_safetensors(&root.join("text_encoder"))
            .unwrap()
            .0
            .remove(0);
        let mut mutated_headers =
            gen_core_testkit::encoder_contract_fixture_tensor_headers(bounded_contract, None)
                .unwrap();
        mutated_headers
            .iter_mut()
            .find(|header| header.name == "model.layers.0.self_attn.q_proj.weight")
            .unwrap()
            .shape[0] += 1;
        write_tensor_file(&encoder_file, header_tuples(mutated_headers));
        let seal_error = validated
            .materialized_language_tensor_headers(&bounded_contract)
            .unwrap_err()
            .to_string();
        // The fixture shard is an HF-cache symlink, so the rewrite lands on its blob target.
        assert!(
            seal_error.contains("pinned weights entry changed after load")
                || seal_error.contains("pinned weights target changed after load")
                || seal_error.contains("artifact seal mismatch after load"),
            "{seal_error}"
        );

        let mut transformer = transformer_tensors(Some(Quant::Q8));
        for suffix in ["weight", "scales", "biases"] {
            transformer.push((
                format!("time_guidance_embed.guidance_embedder.linear_1.{suffix}"),
                if suffix == "weight" { "U32" } else { "BF16" },
                if suffix == "weight" {
                    vec![4096, 1024]
                } else {
                    vec![4096, 4]
                },
            ));
        }
        assert!(
            validate_transformer_tensor_headers(&tensor_headers(transformer), Some(Quant::Q8),)
                .is_err()
        );

        let mut vae = vae_tensors();
        vae.push((
            "encoder.down_blocks.0.resnets.0.conv_shortcut.weight".to_owned(),
            "BF16",
            vec![128, 128, 1, 1],
        ));
        vae.push((
            "encoder.down_blocks.0.resnets.0.conv_shortcut.bias".to_owned(),
            "BF16",
            vec![128],
        ));
        assert!(validate_vae_tensor_headers(&tensor_headers(vae)).is_err());

        let mut vae = vae_tensors();
        let counter = vae
            .iter_mut()
            .find(|(name, _, _)| name == "bn.num_batches_tracked")
            .unwrap();
        counter.1 = "BF16";
        counter.2 = vec![1];
        assert!(validate_vae_tensor_headers(&tensor_headers(vae)).is_err());
    }

    #[test]
    fn turnkey_inventory_reaches_only_the_exact_production_rung_four_contract() {
        use mlx_gen::gen_core::{MemoryStrategy, MemoryStrategySupport};
        use mlx_gen::{LoadShape, OffloadPolicy};

        for (family, provider_id, resolved_route) in [
            (
                TurnkeyFamily::Base,
                crate::FLUX2_KLEIN_9B_ID,
                "flux2_klein_9b",
            ),
            (
                TurnkeyFamily::Kv,
                crate::FLUX2_KLEIN_9B_KV_EDIT_ID,
                "flux2_klein_9b_kv",
            ),
        ] {
            for tier in [None, Some(Quant::Q4), Some(Quant::Q8)] {
                let fixture = immutable_turnkey_fixture(family, tier);
                let spec = LoadSpec::new(WeightsSource::Dir(fixture.root.clone()))
                    .with_resolved_route(resolved_route)
                    .with_offload_policy(OffloadPolicy::Sequential)
                    .with_load_shape(LoadShape::DeferredMaterialization);
                fixture.inventory.validate_provider(provider_id).unwrap();
                fixture
                    .inventory
                    .validate_resolved_route(Some(resolved_route))
                    .unwrap();
                assert_eq!(fixture.inventory.resolved_quant(), tier);
                assert!(crate::memory_strategy::klein_streamable(&spec));
                // The bounded sealed inventory above owns source/identity admission. Exercise the
                // production contract builder proper with it — everything `klein_contract_for`
                // does after artifact verification and footprint pricing — without resealing the
                // fixture or pricing a Qwen tower the bounded fixture does not carry.
                let contract = crate::memory_strategy::klein_contract_from_parts(
                    provider_id,
                    &spec,
                    Some(&fixture.inventory),
                    Default::default(),
                )
                .unwrap_or_else(|error| panic!("{provider_id} {tier:?}: {error}"));
                assert_eq!(
                    contract
                        .capability(MemoryStrategy::BoundedTransformerResidency)
                        .unwrap()
                        .support,
                    MemoryStrategySupport::Implemented
                );
                assert!(contract.conformance_errors().is_empty());
                assert_eq!(
                    contract.calibration.as_ref().unwrap().fingerprint,
                    crate::memory_strategy::klein_production_calibration_fingerprint(
                        provider_id,
                        fixture.inventory.artifact_tag(),
                        tier,
                    )
                    .unwrap()
                );
                // Without an admitted inventory the same spec reaches neither rung 4 nor an
                // identity.
                let unadmitted = crate::memory_strategy::klein_contract_from_parts(
                    provider_id,
                    &spec,
                    None,
                    Default::default(),
                )
                .unwrap();
                assert_eq!(
                    unadmitted
                        .capability(MemoryStrategy::BoundedTransformerResidency)
                        .unwrap()
                        .support,
                    MemoryStrategySupport::Missing
                );
                assert!(unadmitted.calibration.is_none());
            }
        }
    }

    /// sc-22727 (epic sc-22723 E1/E4): every admitted turnkey publishes a production calibration
    /// identity keyed on (provider, artifact family, tier) for the worker's
    /// `Resident + EagerMaterialization` rung as well as the deferred evidence shape, the six
    /// (family, tier) cells are pairwise distinct, and none of them is the weights-free registry
    /// string the production path used to hand out for a tagless turnkey.
    ///
    /// Mutation that fails this: restoring `if streamable { .. } else { None }` around the
    /// identity in `klein_contract_from_parts` (every resident cell drops to `None`), or the
    /// `calibration_tag()`/`KLEIN_STATIC_BEHAVIOR_FINGERPRINT` fallback (every turnkey collides
    /// with its weights-free string and the tiers collapse onto one identity), or restoring
    /// `spec.quantize.or(inventory.resolved_quant())` in `klein_production_calibration` (a
    /// request knob that disagrees with the artifact publishes the requested tier's cell).
    #[test]
    fn turnkey_inventory_publishes_a_per_tier_production_identity_for_every_load_shape() {
        use mlx_gen::{LoadShape, OffloadPolicy};

        let mut published = std::collections::BTreeSet::new();
        for (family, provider_id, resolved_route, artifact) in [
            (
                TurnkeyFamily::Base,
                crate::FLUX2_KLEIN_9B_ID,
                "flux2_klein_9b",
                "rehost",
            ),
            (
                TurnkeyFamily::Kv,
                crate::FLUX2_KLEIN_9B_KV_EDIT_ID,
                "flux2_klein_9b_kv",
                "kv-rehost",
            ),
        ] {
            let route = match family {
                TurnkeyFamily::Base => "t2i",
                TurnkeyFamily::Kv => "kv-edit",
            };
            for (tier, label) in [
                (None, "bf16"),
                (Some(Quant::Q4), "q4"),
                (Some(Quant::Q8), "q8"),
            ] {
                let fixture = immutable_turnkey_fixture(family, tier);
                assert_eq!(fixture.inventory.artifact_tag(), artifact);
                let expected =
                    format!("flux2-klein-9b-{label}-mlx-shared-ladder-{artifact}-{route}-v1");
                let weights_free = format!(
                    "{}-{}",
                    crate::memory_strategy::KLEIN_STATIC_BEHAVIOR_FINGERPRINT,
                    provider_id.replace('_', "-")
                );
                for (offload_policy, load_shape) in [
                    (OffloadPolicy::Resident, LoadShape::EagerMaterialization),
                    (
                        OffloadPolicy::Sequential,
                        LoadShape::DeferredMaterialization,
                    ),
                ] {
                    let spec = LoadSpec::new(WeightsSource::Dir(fixture.root.clone()))
                        .with_resolved_route(resolved_route)
                        .with_offload_policy(offload_policy)
                        .with_load_shape(load_shape);
                    let label = format!("{provider_id} {tier:?} {offload_policy:?} {load_shape:?}");
                    // The production contract builder, fed the sealed fixture inventory exactly as
                    // `klein_contract_for` feeds it the verified one (and a declaration-only
                    // footprint, since the bounded fixture carries no priceable Qwen tower).
                    let contract = crate::memory_strategy::klein_contract_from_parts(
                        provider_id,
                        &spec,
                        Some(&fixture.inventory),
                        Default::default(),
                    )
                    .unwrap_or_else(|error| panic!("{label}: {error}"));
                    let identity = contract
                        .calibration
                        .as_ref()
                        .unwrap_or_else(|| panic!("{label}: no identity"));
                    assert_eq!(identity.fingerprint, expected, "{label}");
                    assert_eq!(identity.load_shape, load_shape);
                    assert_eq!(contract.load_shape, load_shape);
                    assert_ne!(identity.fingerprint, weights_free);
                    assert!(
                        contract.conformance_errors().is_empty(),
                        "{label}: {:?}",
                        contract.conformance_errors()
                    );
                    // An overlay on the route has no clean base cell to bind.
                    let with_adapter = spec.clone().with_adapters(vec![mlx_gen::AdapterSpec::new(
                        fixture.root.join("lora.safetensors"),
                        1.0,
                        mlx_gen::AdapterKind::Lora,
                    )]);
                    assert!(crate::memory_strategy::klein_production_calibration(
                        provider_id,
                        &with_adapter,
                        &fixture.inventory,
                    )
                    .unwrap()
                    .is_none());
                    // The request knob never outranks the admitted artifact: `LoadSpec::quantize`
                    // set against the fixture's tier publishes nothing, and set equal to it
                    // publishes the artifact's own cell.
                    for requested in [Quant::Q4, Quant::Q8] {
                        let knob = spec.clone().with_quant(requested);
                        let identity = crate::memory_strategy::klein_production_calibration(
                            provider_id,
                            &knob,
                            &fixture.inventory,
                        )
                        .unwrap();
                        if Some(requested) == tier {
                            assert_eq!(
                                identity.map(|identity| identity.fingerprint).as_deref(),
                                Some(expected.as_str()),
                                "{label} requested {requested:?}"
                            );
                        } else {
                            assert!(
                                identity.is_none(),
                                "{label} requested {requested:?} published {identity:?}"
                            );
                        }
                    }
                }
                published.insert(expected);
            }
        }
        assert_eq!(
            published.len(),
            6,
            "two (family, tier) cells share one identity"
        );
    }

    /// sc-22727: the six shipped (family, tier) cells are admitted in the layout the app installs —
    /// every file a symlink into the repository's `blobs/` tree, the bf16 text encoder and
    /// transformer sharded, and the text encoder dense at every tier (sc-22760).
    ///
    /// Mutations that fail this: restoring the snapshot-only `discovery_roots` (every cell fails
    /// confinement on its `blobs/` target) or restoring the single-file component rule (both
    /// bf16 cells).
    #[test]
    fn turnkey_inventory_admits_the_shipped_hf_cache_layout_at_every_tier() {
        for family in [TurnkeyFamily::Base, TurnkeyFamily::Kv] {
            for tier in [None, Some(Quant::Q4), Some(Quant::Q8)] {
                let fixture = immutable_turnkey_fixture(family, tier);
                let inventory = &fixture.inventory;
                assert!(
                    inventory
                        .entries
                        .iter()
                        .all(|entry| entry.link_target.is_some()),
                    "{family:?} {tier:?}: every snapshot file is a symlink into blobs/"
                );
                let membership = |component: &str| {
                    inventory
                        .visible_safetensors
                        .iter()
                        .find(|(directory, _)| directory.ends_with(component))
                        .map(|(_, membership)| membership.len())
                        .unwrap_or_else(|| panic!("{family:?} {tier:?}: no {component} membership"))
                };
                let (text_encoder_shards, transformer_shards) =
                    if tier.is_none() { (4, 2) } else { (1, 1) };
                assert_eq!(membership("text_encoder"), text_encoder_shards);
                assert_eq!(membership("transformer"), transformer_shards);
                assert_eq!(membership("vae"), 1);
            }
        }
    }

    /// sc-22727: authorizing the repository directory does not authorize anything above it. A
    /// text-encoder shard whose symlink escapes `models--SceneWorks--…` is still refused by the
    /// shared confinement, and a shard mutated in place (a swapped blob) still fails the seal.
    #[cfg(unix)]
    #[test]
    fn turnkey_inventory_refuses_a_text_encoder_symlink_escaping_the_repository() {
        let tmp = turnkey_fixture(TurnkeyFamily::Base, Some(Quant::Q4));
        let root = fixture_root(&tmp, TurnkeyFamily::Base, Some(Quant::Q4));
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
        verify_bounded_turnkey(root.clone(), TurnkeyFamily::Base, Some(Quant::Q4), &spec).unwrap();

        let shard = root.join("text_encoder/model.safetensors");
        let outside = tmp.path().join("outside/model.safetensors");
        std::fs::create_dir_all(outside.parent().unwrap()).unwrap();
        std::fs::copy(&shard, &outside).unwrap();
        std::fs::remove_file(&shard).unwrap();
        std::os::unix::fs::symlink(&outside, &shard).unwrap();
        let error =
            verify_bounded_turnkey(root.clone(), TurnkeyFamily::Base, Some(Quant::Q4), &spec)
                .unwrap_err()
                .to_string();
        assert!(error.contains("canonical target"), "{error}");
        assert!(error.contains("escapes authorized model roots"), "{error}");
    }

    /// sc-22727: a sharded component is one inventory, so a tensor present in two shards is
    /// refused before any loader could pick whichever shard it reads last.
    #[test]
    fn turnkey_inventory_refuses_a_tensor_duplicated_across_shards() {
        let tmp = turnkey_fixture(TurnkeyFamily::Base, None);
        let root = fixture_root(&tmp, TurnkeyFamily::Base, None);
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
        verify_bounded_turnkey(root.clone(), TurnkeyFamily::Base, None, &spec).unwrap();

        let second = root.join("transformer/diffusion_pytorch_model-00002-of-00002.safetensors");
        std::fs::remove_file(&second).unwrap();
        write_tensor_file(&second, bounded_transformer_tensors(None));
        let error = verify_bounded_turnkey(root, TurnkeyFamily::Base, None, &spec)
            .unwrap_err()
            .to_string();
        assert!(error.contains("appears in more than one shard"), "{error}");
    }

    /// sc-22727 scope gap: the public entry `text_encoder_source_for_load` — the one `loader.rs`
    /// calls — driven end to end. Two behaviors:
    ///
    /// * the `spec.text_encoder.is_none()` gate — a user-selected alternate skips artifact
    ///   verification entirely and takes the ordinary encoder path, and
    /// * no silent fallback — a pinned turnkey whose inventory refuses propagates the inventory's
    ///   reason even though the ordinary encoder path over the same directory would have succeeded.
    ///
    /// The `provider_id` argument is forwarded to `verify_for_provider`; the refusal it produces
    /// (`validate_provider`) is asserted over a bounded inventory in the provider-binding test,
    /// because reaching it through this entry would need a text encoder satisfying the
    /// *production* 9B encoder contract.
    #[test]
    fn text_encoder_source_for_load_gates_artifact_verification_on_the_spec_override() {
        let contract = crate::config::bounded_klein_encoder_contract();
        let fixture = immutable_turnkey_fixture(TurnkeyFamily::Base, None);
        let spec = LoadSpec::new(WeightsSource::Dir(fixture.root.clone()));

        // The ordinary bounded encoder path over this very snapshot succeeds …
        contract.source_for_load(&spec, &fixture.root).unwrap();
        // … so this refusal is the inventory's, not the encoder contract's: a pinned turnkey that
        // fails artifact verification never falls back to the ordinary path.
        let error = KleinArtifactInventory::text_encoder_source_for_load(
            contract,
            crate::FLUX2_KLEIN_9B_ID,
            &spec,
            &fixture.root,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("text encoder contract mismatch"), "{error}");

        // A `LoadSpec::text_encoder` override skips artifact verification altogether: the same
        // root that just refused now resolves through the ordinary encoder path.
        let selected = spec
            .clone()
            .with_text_encoder(WeightsSource::Dir(fixture.root.join("text_encoder")));
        let validated = KleinArtifactInventory::text_encoder_source_for_load(
            contract,
            crate::FLUX2_KLEIN_9B_ID,
            &selected,
            &fixture.root,
        )
        .unwrap();
        assert_eq!(validated.packed_quant_bits(), None);
    }

    /// sc-22760: a `quantization` marker on the text encoder or the VAE is refused at every tier,
    /// whatever the shards under it carry. The corrected rehost revisions ship none; the earlier
    /// stale-marker tolerance (sc-22727) is gone, so a pinned tier carrying one is not a tier.
    #[test]
    fn turnkey_inventory_keeps_the_text_encoder_and_vae_dense_at_every_tier() {
        // (family, tier, encoder marker, vae marker)
        let cases = [
            (TurnkeyFamily::Base, Some(Quant::Q4), Some(4), None),
            (TurnkeyFamily::Kv, Some(Quant::Q8), Some(8), None),
            (TurnkeyFamily::Base, None, Some(4), None),
            (TurnkeyFamily::Kv, None, None, Some(4)),
            (TurnkeyFamily::Base, Some(Quant::Q8), None, Some(8)),
        ];
        for (family, tier, encoder_bits, vae_bits) in cases {
            let tmp = turnkey_fixture(family, tier);
            let root = fixture_root(&tmp, family, tier);
            gen_core_testkit::write_encoder_contract_fixture_with_quant(
                &root.join("text_encoder"),
                crate::config::bounded_klein_encoder_contract(),
                encoder_bits,
            )
            .unwrap();
            // The fixture is content-addressed (`relink_into_blobs`): on the bf16 tier the empty
            // transformer and VAE configs share one blob, so replace the VAE's symlink rather than
            // writing through it and re-tagging the transformer too.
            let vae_config = root.join("vae/config.json");
            std::fs::remove_file(&vae_config).unwrap();
            std::fs::write(&vae_config, quantization_config(vae_bits)).unwrap();
            let spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
            let error = verify_bounded_turnkey(root, family, tier, &spec)
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("must stay dense at every turnkey tier"),
                "{family:?} {tier:?} encoder={encoder_bits:?} vae={vae_bits:?}: {error}"
            );
        }
    }

    #[test]
    fn non_pinned_or_single_file_sources_never_become_turnkey_inventory() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(KleinArtifactInventory::verify_for_provider(
            crate::FLUX2_KLEIN_9B_ID,
            &LoadSpec::new(WeightsSource::Dir(tmp.path().to_path_buf())),
        )
        .unwrap()
        .is_none());
        assert!(KleinArtifactInventory::verify_for_provider(
            crate::FLUX2_KLEIN_9B_ID,
            &LoadSpec::new(WeightsSource::File(tmp.path().join("model.safetensors"))),
        )
        .unwrap()
        .is_none());
    }

    #[test]
    fn true_v2_inventory_is_dense_bf16_like_its_conversion_source() {
        let inventory = |kind| KleinArtifactInventory {
            root: PathBuf::new(),
            entries: Vec::new(),
            visible_safetensors: Vec::new(),
            kind,
        };
        assert_eq!(inventory(KleinArtifactKind::TrueV2).resolved_quant(), None);
        assert_eq!(
            inventory(KleinArtifactKind::CalibratedBase).resolved_quant(),
            None
        );
        assert!(inventory(KleinArtifactKind::TrueV2)
            .validate_resolved_route(Some("flux2_klein_9b_true_v2"))
            .is_ok());
        assert!(inventory(KleinArtifactKind::TrueV2)
            .validate_resolved_route(Some("flux2_klein_9b"))
            .is_err());
    }
}
