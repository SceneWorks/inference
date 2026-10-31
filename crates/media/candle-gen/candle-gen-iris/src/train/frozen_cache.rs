//! Comparison-only, run-owned immutable decoding of Iris restoration frozen checkpoints.
//! Raw source dtype/value tensors are shared; no mutable Run or stage-specific pack is cached.
use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::{Instant, UNIX_EPOCH};

use candle_gen::candle_core::{safetensors::Load, Device, Tensor};
use candle_gen::{CandleError as Error, Result};
use gen_core_iris::restoration_train::{sha256_file, verify_resource, TrainingResource};
use safetensors::{SafeTensors, View};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const MARKER: &str = ".sc-26850-owned-runtime";
const CACHE_SCOPE: &str =
    "Iris restoration immutable frozen checkpoint decode; constant all stages, not E2";

fn expected_owner(source: &str, run: &str, attempt: &str, marker: &str) -> Value {
    json!({"schema_version":1,"protocol":"conditional-v1",
        "binding":{"package":"candle-gen-iris","target":"integration",
        "fqn":"restoration_train_real_weights::iris_restoration_bounded_acceleration_acceptance"},
        "source_revision":source,"github_run_id":run,"github_run_attempt":attempt,
        "parent_marker_sha256":marker,"cache_scope":CACHE_SCOPE})
}

fn validate_owner(
    owner: &Value,
    source: &str,
    run: &str,
    attempt: &str,
    parent_digest: &str,
) -> Result<()> {
    if source.len() != 40
        || !source.bytes().all(|b| b.is_ascii_hexdigit())
        || run.is_empty()
        || !run.bytes().all(|b| b.is_ascii_digit())
        || attempt.parse::<u32>().ok().is_none_or(|n| n == 0)
        || *owner != expected_owner(source, run, attempt, parent_digest)
    {
        return Err(refuse("owner source/run/case/marker/schema mismatch"));
    }
    Ok(())
}

fn refuse(message: impl Into<String>) -> Error {
    Error::Msg(format!("iris comparison frozen cache: {}", message.into()))
}
fn io<T>(value: std::io::Result<T>) -> Result<T> {
    value.map_err(|e| refuse(e.to_string()))
}
fn read_json(path: &Path) -> Result<Value> {
    serde_json::from_slice(&io(fs::read(path))?).map_err(|e| refuse(e.to_string()))
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Both paths and the source/run/case authority are fixed by the comparison driver.
fn owned_root() -> Result<(PathBuf, Value)> {
    if candle_gen::train::comparison::active_stage().is_none() {
        return Err(refuse("requires an active comparison"));
    }
    let declared = std::env::var_os("MODEL_PROOF_IRIS_FROZEN_CACHE_DIR")
        .map(PathBuf::from)
        .ok_or_else(|| refuse("missing source-owned directory"))?;
    if !declared.is_absolute()
        || declared.file_name().and_then(|n| n.to_str()) != Some("comparison-iris-frozen")
    {
        return Err(refuse("unexpected directory"));
    }
    let root = io(declared.canonicalize())?;
    let parent = root.parent().ok_or_else(|| refuse("missing parent"))?;
    if root != io(parent.join("comparison-iris-frozen").canonicalize())? {
        return Err(refuse("directory escapes its owned parent"));
    }
    let runtime = std::env::var_os("BESPOKE_RUNTIME_DIR")
        .map(PathBuf::from)
        .ok_or_else(|| refuse("missing child runtime"))?;
    let runtime = io(runtime.canonicalize())?;
    if runtime.parent() != Some(parent)
        || !parent.join(MARKER).is_file()
        || !root.join(MARKER).is_file()
        || !runtime.join(MARKER).is_file()
    {
        return Err(refuse(
            "cache and child must share the authoritative runtime parent",
        ));
    }
    let parent_digest = sha256_file(&parent.join(MARKER)).map_err(|e| refuse(e.to_string()))?;
    for path in [root.join(MARKER), runtime.join(MARKER)] {
        if sha256_file(&path).map_err(|e| refuse(e.to_string()))? != parent_digest {
            return Err(refuse("owned marker mismatch"));
        }
    }
    let owner = read_json(&root.join("comparison-cache-owner.json"))?;
    let source = std::env::var("GITHUB_SHA").map_err(|_| refuse("missing GITHUB_SHA"))?;
    let run = std::env::var("GITHUB_RUN_ID").map_err(|_| refuse("missing GITHUB_RUN_ID"))?;
    let attempt =
        std::env::var("GITHUB_RUN_ATTEMPT").map_err(|_| refuse("missing GITHUB_RUN_ATTEMPT"))?;
    validate_owner(&owner, &source, &run, &attempt, &parent_digest)?;
    Ok((root, owner))
}

fn tensor_description(
    name: &str,
    dtype: safetensors::Dtype,
    shape: &[usize],
    bytes: &[u8],
) -> Value {
    json!({"name":name,"dtype":format!("{dtype:?}"),"shape":shape,"bytes":bytes.len(),"sha256":digest(bytes)})
}
fn describe_source(map: &HashMap<String, Tensor>) -> Value {
    let ordered: BTreeMap<_, _> = map.iter().collect();
    Value::Array(
        ordered
            .into_iter()
            .map(|(name, tensor)| {
                tensor_description(
                    name,
                    View::dtype(tensor),
                    tensor.dims(),
                    View::data(tensor).as_ref(),
                )
            })
            .collect(),
    )
}

/// One metadata parse and one ZIP archive per checkpoint; unlike pickle::read_all,
/// this never reopens/rebuilds the archive for each requested tensor.
fn decode_once(source: &Path, keep: &dyn Fn(&str) -> bool) -> Result<HashMap<String, Tensor>> {
    if source.extension().and_then(|x| x.to_str()) == Some("safetensors") {
        return Ok(
            candle_gen::candle_core::safetensors::load(source, &Device::Cpu)?
                .into_iter()
                .filter(|(name, _)| keep(name))
                .collect(),
        );
    }
    let infos = candle_gen::candle_core::pickle::read_pth_tensor_info(source, false, None)?;
    let mut archive = zip::ZipArchive::new(BufReader::new(io(File::open(source))?))
        .map_err(|e| refuse(e.to_string()))?;
    let mut out = HashMap::new();
    for info in infos.into_iter().filter(|info| keep(&info.name)) {
        let mut entry = archive
            .by_name(&info.path)
            .map_err(|e| refuse(e.to_string()))?;
        let layout = &info.layout;
        let fortran = layout.is_fortran_contiguous();
        if !layout.is_contiguous() && !fortran {
            return Err(refuse(format!("unsupported source layout {}", info.name)));
        }
        let count = layout
            .dims()
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d))
            .and_then(|n| n.checked_mul(info.dtype.size_in_bytes()))
            .ok_or_else(|| refuse("source tensor size overflow"))?;
        let offset = layout.start_offset();
        if offset
            .checked_add(count)
            .is_none_or(|end| end as u64 > entry.size())
        {
            return Err(refuse("source tensor exceeds archive storage"));
        }
        let skipped = io(std::io::copy(
            &mut entry.by_ref().take(offset as u64),
            &mut std::io::sink(),
        ))?;
        if skipped != offset as u64 {
            return Err(refuse("short source offset"));
        }
        let mut bytes = vec![0u8; count];
        io(entry.read_exact(&mut bytes))?;
        let mut tensor = Tensor::from_raw_buffer(&bytes, info.dtype, layout.dims(), &Device::Cpu)?;
        if layout.dims().len() > 1 && fortran {
            let shape: Vec<_> = layout.dims().iter().rev().copied().collect();
            let axes: Vec<_> = (0..shape.len()).rev().collect();
            tensor = tensor.reshape(shape)?.permute(axes)?.contiguous()?;
        }
        if out.insert(info.name.clone(), tensor).is_some() {
            return Err(refuse("duplicate source name"));
        }
    }
    if out.is_empty() {
        return Err(refuse("no selected frozen tensors"));
    }
    Ok(out)
}

fn store_read(path: &Path, manifest: &Value) -> Result<HashMap<String, Tensor>> {
    let stored_size = io(fs::metadata(path))?.len();
    if manifest["cache_bytes"].as_u64() != Some(stored_size) || stored_size > (8u64 << 30) {
        return Err(refuse(
            "store length is invalid or exceeds the fixed frozen-resource bound",
        ));
    }
    let bytes = io(fs::read(path))?;
    if manifest["cache_sha256"] != digest(&bytes)
        || manifest["cache_bytes"].as_u64() != Some(bytes.len() as u64)
    {
        return Err(refuse("store byte hash/size mismatch"));
    }
    let store = SafeTensors::deserialize(&bytes).map_err(|e| refuse(e.to_string()))?;
    let ordered: BTreeMap<_, _> = store.tensors().into_iter().collect();
    let descriptions = Value::Array(
        ordered
            .iter()
            .map(|(name, view)| tensor_description(name, view.dtype(), view.shape(), view.data()))
            .collect(),
    );
    if manifest["tensors"] != descriptions {
        return Err(refuse("store names/dtypes/shapes/content mismatch"));
    }
    ordered
        .into_iter()
        .map(|(name, view)| Ok((name, view.load(&Device::Cpu)?)))
        .collect()
}

/// Fixed role, fixed resource, fixed owned path. All E2/other stages use this exact
/// constant bootstrap; returned raw CPU tensors undergo the original native casts.
pub(super) fn load(
    role: &str,
    source: &Path,
    resource: &TrainingResource,
    keep: &dyn Fn(&str) -> bool,
) -> Result<HashMap<String, Tensor>> {
    if !["convnext", "vgg", "lpips"].contains(&role) {
        return Err(refuse("unknown role"));
    }
    let start = Instant::now();
    let (root, owner) = owned_root()?;
    verify_resource(source, resource).map_err(|e| refuse(e.to_string()))?;
    let metadata = io(fs::metadata(source))?;
    let modified = io(metadata.modified())?
        .duration_since(UNIX_EPOCH)
        .map_err(|e| refuse(e.to_string()))?
        .as_nanos()
        .to_string();
    let original = json!({"canonical_path":io(source.canonicalize())?.to_string_lossy(),
        "bytes":metadata.len(),"modified_unix_nanos":modified,"sha256":resource.sha256,
        "resource_revision":resource.revision,"resource_file":resource.file});
    let path = root.join(format!("{role}.safetensors"));
    let manifest_path = root.join(format!("{role}.json"));
    let hit = manifest_path.is_file();
    if !hit {
        let lock = root.join(format!("{role}.lock"));
        let _lock = io(OpenOptions::new().write(true).create_new(true).open(&lock))?;
        let map = decode_once(source, keep)?;
        let tensors = describe_source(&map);
        let staged = root.join(format!("{role}.tmp-{}.safetensors", std::process::id()));
        if staged.exists() || path.exists() {
            return Err(refuse("partial/duplicate store refuses overwrite"));
        }
        safetensors::serialize_to_file(map.iter().map(|(n, t)| (n.as_str(), t)), None, &staged)
            .map_err(|e| refuse(e.to_string()))?;
        io(File::open(&staged))?
            .sync_all()
            .map_err(|e| refuse(e.to_string()))?;
        let manifest = json!({"schema_version":1,"format":"iris-native-frozen-v1","role":role,
            "owner":owner,"original":original,"cache_sha256":sha256_file(&staged).map_err(|e| refuse(e.to_string()))?,
            "cache_bytes":io(fs::metadata(&staged))?.len(),"tensors":tensors});
        drop(map);
        verify_resource(source, resource).map_err(|e| refuse(e.to_string()))?;
        // Actual native roundtrip and per-tensor byte comparison before publish.
        drop(store_read(&staged, &manifest)?);
        io(fs::rename(&staged, &path))?;
        let staged_meta = root.join(format!("{role}.tmp-{}.json", std::process::id()));
        let mut file = io(OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&staged_meta))?;
        use std::io::Write;
        io(file
            .write_all(&serde_json::to_vec_pretty(&manifest).map_err(|e| refuse(e.to_string()))?))?;
        io(file.sync_all())?;
        drop(file);
        io(fs::rename(staged_meta, &manifest_path))?;
        drop(_lock);
        io(fs::remove_file(lock))?;
    }
    if io(path.canonicalize())?.parent() != Some(root.as_path())
        || io(manifest_path.canonicalize())?.parent() != Some(root.as_path())
    {
        return Err(refuse("store escapes owned directory"));
    }
    let manifest = read_json(&manifest_path)?;
    if manifest["schema_version"] != 1
        || manifest["format"] != "iris-native-frozen-v1"
        || manifest["role"] != role
        || manifest["owner"] != owner
        || manifest["original"] != original
    {
        return Err(refuse("store/source/owner join mismatch"));
    }
    let (map, reader) = match role {
        "convnext" => (
            candle_gen::gen_core::__model_input_result!(
                store_read(&path, &manifest),
                "candle-gen-iris::train::frozen_cache::convnext_native_store",
                TensorInput,
                &[source, path.as_path()]
            )?,
            "candle-gen-iris::train::frozen_cache::convnext_native_store",
        ),
        "vgg" => (
            candle_gen::gen_core::__model_input_result!(
                store_read(&path, &manifest),
                "candle-gen-iris::train::frozen_cache::vgg_native_store",
                TensorInput,
                &[source, path.as_path()]
            )?,
            "candle-gen-iris::train::frozen_cache::vgg_native_store",
        ),
        "lpips" => (
            candle_gen::gen_core::__model_input_result!(
                store_read(&path, &manifest),
                "candle-gen-iris::train::frozen_cache::lpips_native_store",
                TensorInput,
                &[source, path.as_path()]
            )?,
            "candle-gen-iris::train::frozen_cache::lpips_native_store",
        ),
        _ => unreachable!("fixed cache role already checked"),
    };
    if map.keys().any(|name| !keep(name)) {
        return Err(refuse("unexpected store tensor"));
    }
    candle_gen::train::comparison::record_initialization(&json!({
        "owner":"iris-restoration-frozen-bootstrap","role":role,"reader":reader,"cache_owner":owner,"native_read_stage":"raw CPU frozen store read before original native dtype casts",
        "original":original,"cache_sha256":manifest["cache_sha256"],"cache_bytes":manifest["cache_bytes"],
        "tensors":manifest["tensors"],"qualification":"same immutable raw native checkpoint tensors in every arm; not whole-model byte attestation"}));
    candle_gen::train::profile::cohort_reuse("iris:immutable-frozen-bootstrap-read");
    eprintln!(
        "IRIS_COMPARISON_FROZEN_CACHE role={role} hit={hit} elapsed_seconds={} reader={reader}",
        start.elapsed().as_secs_f64()
    );
    Ok(map)
}

#[cfg(test)]
mod metadata_tests {
    use super::*;
    #[test]
    fn frozen_cache_owner_is_exact_source_run_and_case() {
        let expected = expected_owner(&"a".repeat(40), "123", "1", &"b".repeat(64));
        assert!(validate_owner(&expected, &"a".repeat(40), "123", "1", &"b".repeat(64)).is_ok());
        for (field, changed) in [
            ("source_revision", json!("c".repeat(40))),
            ("github_run_id", json!("124")),
            ("github_run_attempt", json!(2)),
            ("parent_marker_sha256", json!("d".repeat(64))),
            ("cache_scope", json!("E2")),
        ] {
            let mut wrong = expected.clone();
            wrong[field] = changed;
            assert!(
                validate_owner(&wrong, &"a".repeat(40), "123", "1", &"b".repeat(64)).is_err(),
                "{field} is authoritative"
            );
        }
        let mut wrong = expected.clone();
        wrong["binding"]["fqn"] = json!("other_test");
        assert!(validate_owner(&wrong, &"a".repeat(40), "123", "1", &"b".repeat(64)).is_err());
        assert!(validate_owner(&expected, "bad", "123", "1", &"b".repeat(64)).is_err());
    }
    #[test]
    fn frozen_tensor_signature_preserves_type_shape_and_bytes() {
        // Raw mock bytes only: no Tensor constructor/model/device/optimizer is called.
        let a = tensor_description("weight", safetensors::Dtype::F32, &[1], &[0, 0, 0, 0]);
        assert_ne!(
            a,
            tensor_description("weight", safetensors::Dtype::BF16, &[2], &[0, 0, 0, 0])
        );
        assert_ne!(
            a,
            tensor_description("other", safetensors::Dtype::F32, &[1], &[0, 0, 0, 0])
        );
        assert_ne!(
            a,
            tensor_description("weight", safetensors::Dtype::F32, &[1], &[1, 0, 0, 0])
        );
        assert_eq!(
            a,
            tensor_description("weight", safetensors::Dtype::F32, &[1], &[0, 0, 0, 0])
        );
    }
}
