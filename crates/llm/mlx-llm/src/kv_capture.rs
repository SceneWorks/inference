//! SC-20677 real K/V capture for the matched-budget candidate comparison.
//!
//! Loads a campaign snapshot through the same product provider and causal decoder the SC-20671
//! campaign decodes through (`LlamaProvider::load_for_campaign`, `Decode::make_cache`,
//! `Decode::step`), prefills `N-1` prompt tokens, then runs one real single-token decode step for
//! token `N-1`. For each requested layer it writes a safetensors file that
//! [`crate::primitives::kv_candidates::compare::load_captured_case`] reads directly:
//!
//! - `q` `[B, Hq, 1, D]` — that decode step's query, post-RoPE at position `N-1`, exactly as the
//!   attention layer handed it to its cache;
//! - `k`/`v` `[B, Hkv, N, D]` — the post-RoPE keys/values as stored in the dense cache after the
//!   step (the decode token's own K/V included), i.e. what that step attended over;
//! - string metadata: `scale` (the layer's own attention scale), `mask` (`causal`), layer, rope
//!   offset, GQA geometry, dtype, KV length, snapshot inventory sha256, prompt sha256, and the
//!   inference git revision.
//!
//! A `capture-manifest.json` and a `SHA256SUMS` file cover every written file. Layers are written
//! one at a time: at most one layer's `(q, k, v)` handles exist outside the model's own cache.
//!
//! The worker runs only under the campaign supervisor (`parent` spawns it with the SC-20671/76
//! safety policy: child footprint cap, host free reserve, deadline, bounded logs). One command
//! captures and compares (from the inference checkout, prebuilt MLX exported):
//!
//! ```text
//! cargo run --locked --release -p mlx-llm --bin sc20677_capture_kv -- parent \
//!   --snapshot <llama-4bit-snapshot> --prompt-file <prompt.txt> --tokens 8192 \
//!   --layers 0,mid,last --safety-policy <policy.json> --out /abs/sc20677-kv-llama && \
//! cargo run --locked --release -p mlx-llm --bin sc20677_kv_candidates -- \
//!   $(for f in /abs/sc20677-kv-llama/*.safetensors; do printf -- '--kv %s ' "$f"; done) \
//!   --out /abs/sc20677-comparison-llama.json
//! ```

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use mlx_rs::Array;

use crate::campaign::{self, load_campaign_safety_policy};
use crate::campaign_supervisor::{self, RunRequest, SystemProbe};
use crate::decode::Decode;
use crate::error::Result as EngineResult;
use crate::primitives::kv_cache::{
    CacheRoute, CompressedCacheStorage, ContiguousKvCache, KvCache, PackedAttentionMask,
    PackedCacheEvidence,
};

pub const CAPTURE_SCHEMA: &str = "sc-20677-kv-capture/v1";
pub const CAPTURE_MANIFEST: &str = "capture-manifest.json";
pub const CAPTURE_SHA256SUMS: &str = "SHA256SUMS";
const CAPTURE_TOOL: &str = "sc20677_capture_kv";
/// Admission estimate source of the capture worker: the snapshot inventory bytes (its resident
/// weights).
pub const SC20677_ESTIMATE_SOURCE: &str = "sc20677-snapshot-inventory-bytes";
const TOKENIZATION: &str = "encode(prompt, special=true) then encode(prompt, special=false) repeated, truncated to --tokens";

/// Resolve `--layers` (`0,mid,last` or integers) against the loaded decoder's layer count.
pub fn parse_layers(spec: &str, num_layers: usize) -> Result<Vec<usize>, String> {
    if num_layers == 0 {
        return Err("decoder has no layers".into());
    }
    let mut layers = BTreeSet::new();
    for item in spec.split(',').map(str::trim) {
        let layer = match item {
            "first" => 0,
            "mid" => num_layers / 2,
            "last" => num_layers - 1,
            number => number
                .parse::<usize>()
                .map_err(|_| format!("--layers entry `{number}` is not an index, mid, or last"))?,
        };
        if layer >= num_layers {
            return Err(format!(
                "--layers entry {layer} is outside the decoder's {num_layers} layers"
            ));
        }
        layers.insert(layer);
    }
    Ok(layers.into_iter().collect())
}

/// `first` (the prompt with special tokens) followed by `repeat` (the prompt without them) until
/// `target` tokens, then truncated to exactly `target`.
pub fn repeat_to_length(first: &[i32], repeat: &[i32], target: usize) -> Result<Vec<i32>, String> {
    if target < 2 {
        return Err("--tokens must be at least 2 (a prefill plus one decode step)".into());
    }
    if first.is_empty() || (first.len() < target && repeat.is_empty()) {
        return Err("prompt tokenizes to nothing it could be repeated with".into());
    }
    let mut tokens = first.to_vec();
    while tokens.len() < target {
        tokens.extend_from_slice(repeat);
    }
    tokens.truncate(target);
    Ok(tokens)
}

#[cfg(test)]
thread_local! {
    /// `(live, max live)` [`LayerTensors`] on this thread: the one-layer-at-a-time invariant.
    static LAYER_TENSORS: std::cell::Cell<(usize, usize)> = const { std::cell::Cell::new((0, 0)) };
}

/// One layer's `(q, k, v)` handles, alive only while that layer's file is written.
struct LayerTensors {
    query: Array,
    keys: Array,
    values: Array,
}

impl LayerTensors {
    fn new(query: Array, keys: Array, values: Array) -> Self {
        #[cfg(test)]
        LAYER_TENSORS.with(|count| {
            let (live, max) = count.get();
            count.set((live + 1, max.max(live + 1)));
        });
        Self {
            query,
            keys,
            values,
        }
    }
}

impl Drop for LayerTensors {
    fn drop(&mut self) {
        #[cfg(test)]
        LAYER_TENSORS.with(|count| {
            let (live, max) = count.get();
            count.set((live.saturating_sub(1), max));
        });
    }
}

/// The decode step's query for one layer, as the attention layer presented it to its cache.
#[derive(Clone)]
struct CapturedQuery {
    query: Array,
    scale: f32,
    mask: PackedAttentionMask,
}

/// A transparent wrapper over the decoder's own dense cache. While armed it records the query
/// (never K/V) the attention layer passes at the model boundary; every cache operation is
/// delegated unchanged, so the forward pass is the production one.
struct CaptureCache {
    inner: Box<dyn KvCache>,
    targets: BTreeSet<usize>,
    armed: bool,
    queries: BTreeMap<usize, CapturedQuery>,
}

impl KvCache for CaptureCache {
    fn preflight_packed(&self, query_length: usize, mask: bool) -> CacheRoute {
        self.inner.preflight_packed(query_length, mask)
    }

    #[allow(clippy::too_many_arguments)]
    fn try_packed_attention(
        &mut self,
        layer: usize,
        query: &Array,
        keys: &Array,
        values: &Array,
        mask: PackedAttentionMask,
        scale: f32,
        retained_for_sharing: bool,
    ) -> EngineResult<Option<Array>> {
        if self.armed && self.targets.contains(&layer) {
            let previous = self.queries.insert(
                layer,
                CapturedQuery {
                    query: query.clone(),
                    scale,
                    mask,
                },
            );
            if previous.is_some() {
                return Err(crate::error::Error::Msg(format!(
                    "layer {layer} presented more than one query in the captured decode step"
                )));
            }
        }
        self.inner.try_packed_attention(
            layer,
            query,
            keys,
            values,
            mask,
            scale,
            retained_for_sharing,
        )
    }

    fn import_prefix(&mut self, layers: &[(Array, Array)]) -> EngineResult<bool> {
        self.inner.import_prefix(layers)
    }

    fn prepare_dense_fallback(&mut self, operation: &str, reason: &str) -> EngineResult<()> {
        self.inner.prepare_dense_fallback(operation, reason)
    }

    fn packed_evidence(&self) -> Option<PackedCacheEvidence> {
        self.inner.packed_evidence()
    }

    fn compressed_storage(&self) -> EngineResult<Option<CompressedCacheStorage>> {
        self.inner.compressed_storage()
    }

    fn compressed_dense_fallback(&self) -> Option<&ContiguousKvCache> {
        self.inner.compressed_dense_fallback()
    }

    fn record_events(&mut self) {
        self.inner.record_events()
    }

    fn update(
        &mut self,
        layer: usize,
        keys: &Array,
        values: &Array,
    ) -> EngineResult<(Array, Array)> {
        self.inner.update(layer, keys, values)
    }

    fn offset(&self) -> i32 {
        self.inner.offset()
    }

    fn batch_size(&self) -> i32 {
        self.inner.batch_size()
    }

    fn num_layers(&self) -> usize {
        self.inner.num_layers()
    }

    fn retain_sequences(&mut self, keep: &[i32]) -> EngineResult<()> {
        self.inner.retain_sequences(keep)
    }

    fn truncate(&mut self, len: i32) -> EngineResult<()> {
        self.inner.truncate(len)
    }

    fn reset(&mut self) -> EngineResult<()> {
        self.inner.reset()
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self.inner.as_any_mut()
    }
}

/// The state after the captured decode step: the decoder's dense cache plus each target layer's
/// query. No K/V is copied out until [`write_capture`] visits a layer.
pub struct DecodeCapture {
    cache: CaptureCache,
    queries: BTreeMap<usize, CapturedQuery>,
    /// Positions cached after the decode step (= requested tokens).
    pub kv_len: usize,
    /// RoPE position of the captured query (= `kv_len - 1`).
    pub rope_offset: usize,
    pub decode_token: i32,
}

impl DecodeCapture {
    pub fn layers(&self) -> Vec<usize> {
        self.queries.keys().copied().collect()
    }
}

fn host_eval(array: &Array) -> Result<(), String> {
    array.eval().map_err(|e| e.to_string())
}

/// Prefill `tokens[..N-1]` at offset 0, then decode `tokens[N-1]` at offset `N-1` with the query
/// capture armed for `layers` — both through `decoder`'s own `make_cache`/`step`.
pub fn capture_decode_step(
    decoder: &dyn Decode,
    tokens: &[i32],
    layers: &[usize],
) -> Result<DecodeCapture, String> {
    let n = tokens.len();
    if n < 2 {
        return Err("capture needs at least two tokens".into());
    }
    if layers.is_empty() {
        return Err("capture needs at least one layer".into());
    }
    let mut cache = CaptureCache {
        inner: decoder.make_cache(),
        targets: layers.iter().copied().collect(),
        armed: false,
        queries: BTreeMap::new(),
    };
    if let Some(&layer) = layers.iter().find(|&&layer| layer >= cache.num_layers()) {
        return Err(format!("layer {layer} is outside the decoder cache"));
    }
    let prefill = crate::primitives::input_ids(&tokens[..n - 1]);
    host_eval(
        &decoder
            .step(&prefill, &mut cache, 0)
            .map_err(|e| e.to_string())?,
    )?;
    let rope_offset = n - 1;
    if cache.offset() as usize != rope_offset {
        return Err(format!(
            "prefill cached {} positions, expected {rope_offset}",
            cache.offset()
        ));
    }
    cache.armed = true;
    let step = crate::primitives::input_ids(&tokens[n - 1..]);
    let logits = decoder
        .step(&step, &mut cache, rope_offset as i32)
        .map_err(|e| e.to_string())?;
    cache.armed = false;
    host_eval(&logits)?;
    drop(logits);
    if cache.offset() as usize != n {
        return Err(format!(
            "decode step cached {} positions, expected {n}",
            cache.offset()
        ));
    }
    let queries = std::mem::take(&mut cache.queries);
    if let Some(missing) = layers.iter().find(|layer| !queries.contains_key(layer)) {
        return Err(format!(
            "layer {missing} did not present its decode query at the cache boundary"
        ));
    }
    Ok(DecodeCapture {
        cache,
        queries,
        kv_len: n,
        rope_offset,
        decode_token: tokens[n - 1],
    })
}

fn mask_text(mask: PackedAttentionMask) -> Result<String, String> {
    match mask {
        PackedAttentionMask::None => Ok("none".into()),
        PackedAttentionMask::Causal => Ok("causal".into()),
        PackedAttentionMask::SlidingWindow(window) => Ok(format!("window:{window}")),
        PackedAttentionMask::Additive => {
            Err("an additive-mask decode cannot be replayed by the comparison".into())
        }
    }
}

fn dim(array: &Array, axis: usize) -> usize {
    usize::try_from(array.shape()[axis]).unwrap_or(0)
}

/// One written capture file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapturedFile {
    pub file: String,
    pub layer: usize,
    pub sha256: String,
    pub bytes: u64,
    pub metadata: BTreeMap<String, String>,
}

/// Write one safetensors file per captured layer — visiting layers one at a time — then the
/// manifest and `SHA256SUMS`. `base` metadata (snapshot, prompt, revision) is stamped on every
/// file alongside the per-layer geometry.
pub fn write_capture(
    capture: DecodeCapture,
    out: &Path,
    base: &BTreeMap<String, String>,
) -> Result<Vec<CapturedFile>, String> {
    let DecodeCapture {
        mut cache,
        queries,
        kv_len,
        rope_offset,
        decode_token,
    } = capture;
    let num_layers = cache.num_layers();
    let dense = cache
        .as_any_mut()
        .downcast_mut::<ContiguousKvCache>()
        .ok_or("capture requires the decoder's dense contiguous cache")?;
    fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let mut files = Vec::with_capacity(queries.len());
    for (layer, captured) in queries {
        let (keys, values) = dense
            .peek(layer)
            .map_err(|e| e.to_string())?
            .ok_or_else(|| format!("layer {layer} has no cached K/V"))?;
        let tensors = LayerTensors::new(captured.query, keys, values);
        if tensors.query.shape().len() != 4 || tensors.keys.shape().len() != 4 {
            return Err(format!("layer {layer} capture is not rank 4"));
        }
        let (batch, query_heads, query_len, head_dim) = (
            dim(&tensors.query, 0),
            dim(&tensors.query, 1),
            dim(&tensors.query, 2),
            dim(&tensors.query, 3),
        );
        let kv_heads = dim(&tensors.keys, 1);
        if tensors.keys.shape() != tensors.values.shape()
            || tensors.keys.shape()
                != [
                    batch as i32,
                    kv_heads as i32,
                    kv_len as i32,
                    head_dim as i32,
                ]
            || query_len != 1
            || kv_heads == 0
            || query_heads % kv_heads != 0
        {
            return Err(format!(
                "layer {layer} capture geometry q{:?} k{:?} v{:?} is not [B,Hq,1,D] / [B,Hkv,{kv_len},D]",
                tensors.query.shape(),
                tensors.keys.shape(),
                tensors.values.shape()
            ));
        }
        let mut metadata = base.clone();
        for (key, value) in [
            ("schema", CAPTURE_SCHEMA.to_string()),
            ("captureTool", CAPTURE_TOOL.to_string()),
            ("layer", layer.to_string()),
            ("numLayers", num_layers.to_string()),
            ("ropeOffset", rope_offset.to_string()),
            ("kvLen", kv_len.to_string()),
            ("queryLen", query_len.to_string()),
            ("batch", batch.to_string()),
            ("queryHeads", query_heads.to_string()),
            ("kvHeads", kv_heads.to_string()),
            ("gqaGroup", (query_heads / kv_heads).to_string()),
            ("headDim", head_dim.to_string()),
            ("dtype", format!("{:?}", tensors.keys.dtype())),
            ("queryDtype", format!("{:?}", tensors.query.dtype())),
            ("scale", captured.scale.to_string()),
            ("mask", mask_text(captured.mask)?),
            ("decodeTokenId", decode_token.to_string()),
        ] {
            metadata.insert(key.to_string(), value);
        }
        let file = format!("layer-{layer:03}.safetensors");
        let path = out.join(&file);
        if path.exists() {
            return Err(format!("capture file {} already exists", path.display()));
        }
        let hash_map: HashMap<String, String> = metadata.clone().into_iter().collect();
        Array::save_safetensors(
            [
                ("q", &tensors.query),
                ("k", &tensors.keys),
                ("v", &tensors.values),
            ],
            &hash_map,
            &path,
        )
        .map_err(|e| format!("save {}: {e}", path.display()))?;
        drop(tensors);
        files.push(CapturedFile {
            sha256: campaign::file_seal(&path)?,
            bytes: fs::metadata(&path).map_err(|e| e.to_string())?.len(),
            file,
            layer,
            metadata,
        });
    }
    drop(cache);
    write_manifest(out, base, &files)?;
    Ok(files)
}

fn write_manifest(
    out: &Path,
    base: &BTreeMap<String, String>,
    files: &[CapturedFile],
) -> Result<(), String> {
    let manifest = serde_json::json!({
        "schema": CAPTURE_SCHEMA,
        "captureTool": CAPTURE_TOOL,
        "inputs": base,
        "files": files.iter().map(|file| serde_json::json!({
            "file": file.file,
            "layer": file.layer,
            "sha256": file.sha256,
            "bytes": file.bytes,
            "metadata": file.metadata,
        })).collect::<Vec<_>>(),
    });
    let bytes = campaign::canonical_json_bytes(&manifest).map_err(|e| e.to_string())?;
    fs::write(out.join(CAPTURE_MANIFEST), &bytes).map_err(|e| e.to_string())?;
    let mut sums = files
        .iter()
        .map(|file| format!("{}  {}\n", file.sha256, file.file))
        .collect::<String>();
    sums.push_str(&format!(
        "{}  {CAPTURE_MANIFEST}\n",
        campaign::seal_bytes(&bytes)
    ));
    fs::write(out.join(CAPTURE_SHA256SUMS), sums).map_err(|e| e.to_string())
}

/// Verify `SHA256SUMS` against the directory: every listed file matches, the manifest is listed,
/// and every safetensors file present is listed. Returns the verified capture file names.
pub fn verify_capture_dir(dir: &Path) -> Result<Vec<String>, String> {
    let sums = fs::read_to_string(dir.join(CAPTURE_SHA256SUMS))
        .map_err(|e| format!("read {CAPTURE_SHA256SUMS}: {e}"))?;
    let mut listed = BTreeSet::new();
    for line in sums.lines() {
        let (sha256, name) = line
            .split_once("  ")
            .ok_or_else(|| format!("malformed {CAPTURE_SHA256SUMS} line `{line}`"))?;
        if name.contains('/') || !listed.insert(name.to_string()) {
            return Err(format!("{CAPTURE_SHA256SUMS} lists `{name}` unsafely"));
        }
        if campaign::file_seal(&dir.join(name))? != sha256 {
            return Err(format!("capture file {name} does not match its sha256"));
        }
    }
    if !listed.contains(CAPTURE_MANIFEST) {
        return Err(format!(
            "{CAPTURE_SHA256SUMS} does not cover {CAPTURE_MANIFEST}"
        ));
    }
    let mut captures = Vec::new();
    for entry in fs::read_dir(dir).map_err(|e| e.to_string())? {
        let name = entry.map_err(|e| e.to_string())?.file_name();
        let name = name.to_string_lossy().to_string();
        if name.ends_with(".safetensors") {
            if !listed.contains(&name) {
                return Err(format!(
                    "capture file {name} is not in {CAPTURE_SHA256SUMS}"
                ));
            }
            captures.push(name);
        }
    }
    if captures.is_empty() {
        return Err("capture directory has no safetensors files".into());
    }
    captures.sort();
    Ok(captures)
}

fn flag(args: &[String], name: &str) -> Result<String, String> {
    args.windows(2)
        .find_map(|window| (window[0] == name).then(|| window[1].clone()))
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("missing {name}"))
}

fn optional_flag(args: &[String], name: &str, default: &str) -> Result<String, String> {
    if args.iter().any(|arg| arg == name) {
        flag(args, name)
    } else {
        Ok(default.to_string())
    }
}

const USAGE: &str =
    "usage: sc20677_capture_kv parent --snapshot DIR --prompt-file FILE --tokens N \
[--layers 0,mid,last] --safety-policy POLICY.json --out /abs/DIR";

/// CLI for the standalone `sc20677_capture_kv` binary: `parent` validates inputs and runs the
/// `worker` under the campaign supervisor, then verifies and publishes the capture directory.
pub fn cli(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("parent") => parent(args),
        Some("worker") => worker(args),
        _ => Err(USAGE.into()),
    }
}

fn parent(args: &[String]) -> Result<(), String> {
    let snapshot = PathBuf::from(flag(args, "--snapshot")?);
    let prompt_file = PathBuf::from(flag(args, "--prompt-file")?);
    let policy_path = PathBuf::from(flag(args, "--safety-policy")?);
    let out = PathBuf::from(flag(args, "--out")?);
    let layers = optional_flag(args, "--layers", "0,mid,last")?;
    let tokens: u64 = flag(args, "--tokens")?
        .parse()
        .map_err(|_| "--tokens must be an integer".to_string())?;
    if tokens < 2 {
        return Err("--tokens must be at least 2".into());
    }
    if !out.is_absolute() || out.exists() {
        return Err("--out must be an absolute path that does not exist".into());
    }
    let staging = PathBuf::from(format!("{}.partial", out.display()));
    if staging.exists() {
        return Err(format!(
            "partial capture {} exists; inspect and remove it explicitly",
            staging.display()
        ));
    }
    let prompt = fs::read_to_string(&prompt_file).map_err(|e| format!("read prompt: {e}"))?;
    if prompt.trim().is_empty() {
        return Err("capture prompt must not be empty".into());
    }
    let policy = load_campaign_safety_policy(&policy_path)?;
    let policy_sha256 = policy.seal()?;
    let inventory =
        campaign::inventory_snapshot(&snapshot).map_err(|e| format!("snapshot inventory: {e}"))?;
    // Resident weights are the static floor and the admission estimate: the supervisor admits the
    // worker when host available memory covers them plus the reserve, and its runtime guards (cap,
    // host reserve, deadline) catch a capture that grows beyond them.
    let admission =
        campaign::runtime_guarded_admission(&policy, inventory.bytes, SC20677_ESTIMATE_SOURCE)?;
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    fs::create_dir(&staging).map_err(|e| e.to_string())?;
    let mut command = Command::new(std::env::current_exe().map_err(|e| e.to_string())?);
    command
        .arg("worker")
        .arg("--snapshot")
        .arg(&snapshot)
        .arg("--snapshot-sha256")
        .arg(&inventory.sha256)
        .arg("--prompt-file")
        .arg(&prompt_file)
        .arg("--tokens")
        .arg(tokens.to_string())
        .arg("--layers")
        .arg(&layers)
        .arg("--safety-policy")
        .arg(&policy_path)
        .arg("--policy-sha256")
        .arg(&policy_sha256)
        .arg("--out")
        .arg(&staging);
    let logs = staging.join("logs");
    let prefix = campaign::unused_attempt_prefix(&logs, "capture")
        .ok_or("no unused bounded worker log path")?;
    let request = RunRequest {
        context_tokens: tokens,
        request_tokens: tokens,
        estimate: admission.estimate(),
        stdout_path: logs.join(format!("{prefix}.stdout.log")),
        stderr_path: logs.join(format!("{prefix}.stderr.log")),
    };
    let unaccepted = logs.join(format!("{prefix}.unaccepted.json"));
    let status = campaign_supervisor::run_guarded(
        &mut command,
        &request,
        &policy.supervisor(),
        &mut SystemProbe,
    )
    .map_err(|failure| {
        let reason = format!("{:?}", failure.reason);
        campaign::unaccepted_row_error(
            &unaccepted,
            "sc-20677-unaccepted-capture",
            "capture",
            &admission.with_host_memory(failure.host_memory.as_deref().cloned()),
            (
                &reason,
                &failure.detail,
                failure.pid,
                failure.watchdog_host_memory.as_deref(),
            ),
            format!(
                "capture worker stopped ({reason}): {}; child {:?} reaped; partial output {}",
                failure.detail,
                failure.pid,
                staging.display()
            ),
        )
    })?;
    if !status.success() {
        return Err(campaign::unaccepted_row_error(
            &unaccepted,
            "sc-20677-unaccepted-capture",
            "capture",
            &admission,
            ("ChildExit", &status.to_string(), None, None),
            format!(
                "capture worker failed with {status}; stderr {}; partial output {}",
                request.stderr_path.display(),
                staging.display()
            ),
        ));
    }
    let files = verify_capture_dir(&staging)?;
    fs::rename(&staging, &out).map_err(|e| e.to_string())?;
    let kv_args = files
        .iter()
        .map(|file| format!("--kv {}", out.join(file).display()))
        .collect::<Vec<_>>()
        .join(" ");
    println!(
        "{}",
        serde_json::json!({
            "status": "captured",
            "out": out.display().to_string(),
            "files": files,
            "compare": format!("cargo run --locked --release -p mlx-llm --bin sc20677_kv_candidates -- {kv_args} --out <report.json>"),
        })
    );
    Ok(())
}

fn worker(args: &[String]) -> Result<(), String> {
    let policy = load_campaign_safety_policy(Path::new(&flag(args, "--safety-policy")?))?;
    if policy.seal()? != flag(args, "--policy-sha256")? {
        return Err("worker safety policy seal differs from the parent's".into());
    }
    let snapshot = PathBuf::from(flag(args, "--snapshot")?);
    let out = PathBuf::from(flag(args, "--out")?);
    let prompt_path = PathBuf::from(flag(args, "--prompt-file")?);
    let prompt = fs::read_to_string(&prompt_path).map_err(|e| format!("read prompt: {e}"))?;
    let tokens: usize = flag(args, "--tokens")?
        .parse()
        .map_err(|_| "--tokens must be an integer".to_string())?;
    let inventory =
        campaign::inventory_snapshot(&snapshot).map_err(|e| format!("snapshot inventory: {e}"))?;
    if inventory.sha256 != flag(args, "--snapshot-sha256")? {
        return Err("snapshot inventory changed between parent and worker".into());
    }
    // The supervisor's admission decision (rule, estimate, pre-spawn host measurement) travels
    // with the sealed capture metadata.
    let admission =
        campaign::supervised_admission(&policy, inventory.bytes, SC20677_ESTIMATE_SOURCE)?;
    let inference_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .ok_or("inference root")?;
    let inference_revision = campaign::checked_git_revision(inference_root)?;
    let provider = crate::provider::LlamaProvider::load_for_campaign(&core_llm::LoadSpec::dense(
        snapshot.to_string_lossy().to_string(),
    ))
    .map_err(|e| format!("load campaign provider: {e}"))?;
    let family = provider.campaign_family().map_err(|e| e.to_string())?;
    let context_window = provider
        .campaign_context_window()
        .map_err(|e| e.to_string())?;
    if tokens as u64 > context_window {
        return Err(format!(
            "--tokens {tokens} exceeds the model's {context_window}-token context"
        ));
    }
    let (model, tokenizer) = provider
        .campaign_causal_decoder()
        .map_err(|e| e.to_string())?;
    let encode = |special: bool| -> Result<Vec<i32>, String> {
        tokenizer
            .encode(&prompt, special)
            .map_err(|e| e.to_string())?
            .into_iter()
            .map(|id| i32::try_from(id).map_err(|_| "token id overflows i32".to_string()))
            .collect()
    };
    let ids = repeat_to_length(&encode(true)?, &encode(false)?, tokens)?;
    let layers = parse_layers(&flag(args, "--layers")?, model.config().num_layers)?;
    let base = BTreeMap::from([
        ("family".to_string(), family.to_string()),
        ("model".to_string(), snapshot.display().to_string()),
        ("modelSnapshotSha256".to_string(), inventory.sha256.clone()),
        (
            "modelSnapshotBytes".to_string(),
            inventory.bytes.to_string(),
        ),
        (
            "promptSha256".to_string(),
            campaign::file_seal(&prompt_path)?,
        ),
        ("tokens".to_string(), tokens.to_string()),
        ("tokenization".to_string(), TOKENIZATION.to_string()),
        ("inferenceRevision".to_string(), inference_revision),
        (
            "hostMemoryAdmission".to_string(),
            serde_json::to_string(&admission).map_err(|e| e.to_string())?,
        ),
    ]);
    let capture = capture_decode_step(model, &ids, &layers)?;
    let files = write_capture(capture, &out, &base)?;
    eprintln!(
        "sc20677 capture: {} layer file(s) at {} tokens written to {}",
        files.len(),
        tokens,
        out.display()
    );
    Ok(())
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use crate::models::CausalLm;
    use crate::primitives::kv_candidates::compare::load_captured_case;
    use crate::primitives::Weights;

    /// Tiny synthetic three-layer GQA Llama (head dim 64, 2 query heads over 1 KV head).
    fn tiny_model() -> CausalLm {
        use crate::primitives::sampler::{SplitMix64, TokenRng};
        let cfg = crate::config::ModelConfig {
            hidden_size: 128,
            intermediate_size: 64,
            num_layers: 3,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 64,
            vocab_size: 32,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            rope_scaling: None,
            tie_word_embeddings: false,
            architecture: crate::config::Architecture::Llama,
            max_position_embeddings: 0,
            quantization: None,
            moe: None,
            attn_logit_softcap: None,
            final_logit_softcap: None,
            query_pre_attn_scalar: None,
            partial_rotary_factor: 1.0,
            mla: None,
            yarn: None,
            mrope_section: None,
            gemma4: None,
            activation_role: Default::default(),
        };
        let mut rng = SplitMix64::new(0x5c20677);
        let mut randn = |shape: &[i32]| {
            let n: i32 = shape.iter().product();
            let data: Vec<f32> = (0..n).map(|_| (rng.next_f32() - 0.5) * 0.4).collect();
            Array::from_slice(&data, shape)
        };
        let (h, v, inter) = (cfg.hidden_size, cfg.vocab_size, cfg.intermediate_size);
        let (qd, kvd) = (
            cfg.num_heads * cfg.head_dim,
            cfg.num_kv_heads * cfg.head_dim,
        );
        let mut m = std::collections::HashMap::new();
        m.insert("model.embed_tokens.weight".to_string(), randn(&[v, h]));
        m.insert(
            "model.norm.weight".into(),
            Array::ones::<f32>(&[h]).unwrap(),
        );
        m.insert("lm_head.weight".into(), randn(&[v, h]));
        for i in 0..cfg.num_layers {
            let p = |s: &str| format!("model.layers.{i}.{s}");
            m.insert(
                p("input_layernorm.weight"),
                Array::ones::<f32>(&[h]).unwrap(),
            );
            m.insert(
                p("post_attention_layernorm.weight"),
                Array::ones::<f32>(&[h]).unwrap(),
            );
            m.insert(p("self_attn.q_proj.weight"), randn(&[qd, h]));
            m.insert(p("self_attn.k_proj.weight"), randn(&[kvd, h]));
            m.insert(p("self_attn.v_proj.weight"), randn(&[kvd, h]));
            m.insert(p("self_attn.o_proj.weight"), randn(&[h, qd]));
            m.insert(p("mlp.gate_proj.weight"), randn(&[inter, h]));
            m.insert(p("mlp.up_proj.weight"), randn(&[inter, h]));
            m.insert(p("mlp.down_proj.weight"), randn(&[h, inter]));
        }
        CausalLm::from_weights(&Weights::from_map(m), "", cfg).unwrap()
    }

    fn host(array: &Array) -> Vec<f32> {
        crate::primitives::nn::to_f32_host(array).unwrap()
    }

    /// One bf16 ulp at magnitude `x` (8 significant bits): `2^(⌊log2 x⌋ − 7)`, or the smallest
    /// normal's ulp for zero/subnormals.
    fn bf16_ulp(x: f32) -> f32 {
        let power = f32::from_bits(x.abs().to_bits() & 0x7f80_0000);
        power.max(f32::MIN_POSITIVE) / 128.0
    }

    #[test]
    fn bf16_ulp_is_the_bf16_spacing() {
        assert_eq!(bf16_ulp(1.0), 1.0 / 128.0);
        assert_eq!(bf16_ulp(4.406), 1.0 / 32.0);
        assert_eq!(bf16_ulp(-0.75), 1.0 / 256.0);
        assert_eq!(bf16_ulp(0.0), f32::MIN_POSITIVE / 128.0);
    }

    /// Independent reference: one full-prompt prefill through the same decoder, with the same
    /// query tap armed so the last row's query and the dense cache can be read back.
    fn full_prefill_reference(
        model: &CausalLm,
        tokens: &[i32],
        layer: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut cache = CaptureCache {
            inner: model.make_cache(),
            targets: BTreeSet::from([layer]),
            armed: true,
            queries: BTreeMap::new(),
        };
        host_eval(
            &model
                .step(&crate::primitives::input_ids(tokens), &mut cache, 0)
                .unwrap(),
        )
        .unwrap();
        let query = cache.queries.remove(&layer).unwrap().query;
        let s = query.shape()[2];
        let last = query.index((.., .., (s - 1)..s, ..));
        let dense = cache
            .as_any_mut()
            .downcast_mut::<ContiguousKvCache>()
            .unwrap();
        let (k, v) = dense.peek(layer).unwrap().unwrap();
        (host(&last), host(&k), host(&v))
    }

    use mlx_rs::ops::indexing::IndexOp;

    #[test]
    fn layer_spec_and_token_repetition_resolve_exactly() {
        assert_eq!(parse_layers("0,mid,last", 28).unwrap(), vec![0, 14, 27]);
        assert_eq!(parse_layers("last, 3 ,first", 4).unwrap(), vec![0, 3]);
        assert!(parse_layers("4", 4).is_err());
        assert!(parse_layers("middle", 4).is_err());
        assert_eq!(
            repeat_to_length(&[1, 7, 8], &[7, 8], 6).unwrap(),
            vec![1, 7, 8, 7, 8, 7]
        );
        assert_eq!(
            repeat_to_length(&[1, 7, 8, 9], &[7], 2).unwrap(),
            vec![1, 7]
        );
        assert!(repeat_to_length(&[1], &[], 3).is_err());
        assert!(repeat_to_length(&[1, 2], &[2], 1).is_err());
    }

    #[test]
    fn captured_files_round_trip_through_the_compare_loader_with_decode_geometry() {
        let model = tiny_model();
        let tokens: Vec<i32> = (1..=9).collect();
        let layers = parse_layers("0,mid,last", model.config().num_layers).unwrap();
        assert_eq!(layers, vec![0, 1, 2]);
        let capture = capture_decode_step(&model, &tokens, &layers).unwrap();
        assert_eq!(
            (capture.kv_len, capture.rope_offset, capture.decode_token),
            (9, 8, 9)
        );
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("capture");
        let base = BTreeMap::from([("modelSnapshotSha256".to_string(), "c".repeat(64))]);
        LAYER_TENSORS.with(|count| count.set((0, 0)));
        let files = write_capture(capture, &out, &base).unwrap();
        // One layer's (q, k, v) handles at a time, none retained afterwards.
        assert_eq!(LAYER_TENSORS.with(|count| count.get()), (0, 1));
        assert_eq!(files.len(), 3);
        assert_eq!(
            verify_capture_dir(&out).unwrap(),
            vec![
                "layer-000.safetensors",
                "layer-001.safetensors",
                "layer-002.safetensors"
            ]
        );
        for file in &files {
            let case = load_captured_case(&out.join(&file.file), None, None).unwrap();
            let r = &case.request;
            assert_eq!(
                (
                    r.batch,
                    r.query_heads,
                    r.kv_heads,
                    r.query_len,
                    r.kv_len,
                    r.head_dim
                ),
                (1, 2, 1, 1, 9, 64)
            );
            assert_eq!(r.scale, 0.125);
            assert_eq!(r.mask, PackedAttentionMask::Causal);
            let crate::primitives::kv_candidates::compare::CaseSource::Captured {
                metadata,
                sha256,
                ..
            } = &case.source
            else {
                panic!("captured source");
            };
            assert_eq!(sha256, &file.sha256);
            assert_eq!(metadata["layer"], file.layer.to_string());
            assert_eq!(metadata["ropeOffset"], "8");
            assert_eq!(metadata["kvLen"], "9");
            assert_eq!(metadata["gqaGroup"], "2");
            assert_eq!(metadata["modelSnapshotSha256"], "c".repeat(64));
            assert_eq!(metadata["schema"], CAPTURE_SCHEMA);
            // Prefill(8) + decode(1) reproduces a 9-token prefill's cache and last query row. Layer 0
            // (no attention upstream) must match exactly. From layer 1 on the 9-row prefill attends
            // with MLX's fused full kernel and the split path with its vector kernel (sc-20676): a
            // one-ulp rounding difference in the attention output is summed through the next
            // projection, so the error scales with the tensor's largest elements, not each
            // element's own magnitude (measured: 1 ulp of the largest key, 8 ulps of a small one).
            // Tolerance: 2 bf16 ulps of the tensor's largest magnitude.
            let (query, keys, values) = full_prefill_reference(&model, &tokens, file.layer);
            for (name, got, want) in [
                ("keys", &case.keys, &keys),
                ("values", &case.values, &values),
                ("query", &case.query, &query),
            ] {
                assert_eq!(got.len(), want.len());
                let largest = got
                    .iter()
                    .chain(want.iter())
                    .fold(0.0f32, |m, v| m.max(v.abs()));
                let tolerance = if file.layer == 0 {
                    0.0
                } else {
                    2.0 * bf16_ulp(largest)
                };
                for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
                    assert!(
                        (g - w).abs() <= tolerance,
                        "layer {} {name}[{i}]: {g} vs {w} (tolerance {tolerance})",
                        file.layer
                    );
                }
            }
        }
        // An unlisted capture file, or a tampered one, no longer verifies.
        fs::copy(
            out.join("layer-000.safetensors"),
            out.join("layer-999.safetensors"),
        )
        .unwrap();
        assert!(verify_capture_dir(&out).is_err());
        fs::remove_file(out.join("layer-999.safetensors")).unwrap();
        verify_capture_dir(&out).unwrap();
        fs::write(out.join("layer-001.safetensors"), b"tampered").unwrap();
        assert!(verify_capture_dir(&out).is_err());
    }

    /// A decoder that appends K/V straight through `update`, presenting each layer's query
    /// `presentations` times per step.
    struct MockDecoder {
        presentations: usize,
    }

    impl Decode for MockDecoder {
        fn make_cache(&self) -> Box<dyn KvCache> {
            Box::new(ContiguousKvCache::new(2))
        }

        fn step(
            &self,
            input_ids: &Array,
            cache: &mut dyn KvCache,
            _offset: i32,
        ) -> EngineResult<Array> {
            let s = input_ids.shape()[1];
            let kv = Array::zeros::<f32>(&[1, 1, s, 64])?;
            let query = Array::zeros::<f32>(&[1, 2, s, 64])?;
            for layer in 0..2 {
                for _ in 0..self.presentations {
                    cache.try_packed_attention(
                        layer,
                        &query,
                        &kv,
                        &kv,
                        PackedAttentionMask::Causal,
                        0.125,
                        false,
                    )?;
                }
                cache.update(layer, &kv, &kv)?;
            }
            Ok(Array::zeros::<f32>(&[1, 4])?)
        }
    }

    #[test]
    fn capture_refuses_layers_that_never_presented_a_query() {
        let model = tiny_model();
        let tokens: Vec<i32> = (1..=4).collect();
        assert!(capture_decode_step(&model, &tokens, &[7]).is_err());
        assert!(capture_decode_step(&model, &tokens[..1], &[0]).is_err());
        let error = capture_decode_step(&MockDecoder { presentations: 0 }, &tokens, &[1])
            .err()
            .expect("a layer without a decode query is refused");
        assert!(
            error.contains("did not present its decode query"),
            "{error}"
        );
        let error = capture_decode_step(&MockDecoder { presentations: 2 }, &tokens, &[1])
            .err()
            .expect("a layer presenting two decode queries is refused");
        assert!(error.contains("more than one query"), "{error}");
        let single =
            capture_decode_step(&MockDecoder { presentations: 1 }, &tokens, &[0, 1]).unwrap();
        assert_eq!(single.layers(), vec![0, 1]);
    }
}
