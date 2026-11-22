//! Quantization — group-wise affine Q4/Q8 primitives re-exported from [`mlx_layers::quant`]
//! plus offline turnkey converter utilities (`SW-6955`).

use std::collections::HashMap;
use std::path::Path;

pub use mlx_layers::quantization_cache::{quantize_cached, quantize_linear_cached};
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};

use crate::weights::Weights;
use crate::Result;

pub use mlx_layers::quant::{
    embedding, embedding_compute, is_quant_aligned, lin, lin_with_bias, native_lin,
    needs_load_time_quant, packed_bits, packed_bits_from_headers, packed_bits_from_metadata,
    packed_quant_bits, packed_quant_bits_at, packed_quant_group_size_at, TokenEmbedding,
    DEFAULT_GROUP_SIZE, HEADER_PACKED_GROUP_SIZE,
};

/// Read every tensor of `dir` (possibly sharded safetensors) into an owned key→`Array` map (MLX
/// arrays are ref-counted, so the clone is a handle copy, not a buffer copy). The read side of the
/// offline pre-quantization converters.
pub fn load_dir_map(dir: &Path) -> Result<HashMap<String, Array>> {
    let w = Weights::from_dir(dir)?;
    Ok(w.keys()
        .map(|k| (k.to_string(), w.get(k).expect("listed key").clone()))
        .collect())
}

/// Materialize (`eval`) + write a key→`Array` map to a single `path` safetensors (one file — a packed
/// component is small enough not to need sharding; the loaders glob `*.safetensors`, so one file
/// replaces the source's shards). Keys are sorted before serialization so rebuilding an identical
/// validated artifact produces byte-identical files that can be bound to publication by SHA-256.
/// The write side of the converters.
pub fn save_map(path: &Path, map: &HashMap<String, Array>) -> Result<()> {
    eval(map.values().collect::<Vec<_>>())?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut entries = map.iter().collect::<Vec<_>>();
    entries.sort_unstable_by_key(|(key, _)| *key);
    Array::save_safetensors(
        entries.into_iter().map(|(k, v)| (k.as_str(), v)),
        None::<&HashMap<String, String>>,
        path,
    )?;
    Ok(())
}

/// Pack every `is_target` `{base}.weight` (group-quantizable 2-D, `in % group_size == 0`,
/// `in >= group_size`) into the triple `{base}.weight` (u32 codes) + `.scales` + `.biases` — the
/// weight cast to **bf16 first** so the pack is byte-identical to the load-time
/// [`AdaptableLinear::quantize`](crate::adapters::AdaptableLinear::quantize) (and the fork's
/// `nn.quantize(bf16)`). Every other tensor (norms, 1-D, non-divisible, non-target) passes through
/// unchanged. `is_target` receives the **base** (the key minus its `.weight` suffix). The shape guard
/// keeps an odd-shaped or 1-D target dense rather than crashing `quantize`.
///
/// Each packed target is **evaluated one tensor at a time** (source first, then the packed triple;
/// sc-24114). Those evals are load-bearing, not redundant: MLX runs a safetensors `Load` on its
/// CPU stream, and a GPU `quantize` over an unread `Load` makes the Metal command buffer wait on
/// the disk read — deferred to one eval over the whole map, that trips the GPU watchdog
/// (`kIOGPUCommandBufferCallbackErrorTimeout` → `SubmissionsIgnored`) on a slow volume.
pub fn quantize_map(
    map: HashMap<String, Array>,
    bits: i32,
    group_size: i32,
    is_target: impl Fn(&str) -> bool,
) -> Result<HashMap<String, Array>> {
    let mut out = HashMap::with_capacity(map.len());
    for (k, v) in map {
        let base = k.strip_suffix(".weight").filter(|b| is_target(b));
        let packable = base.is_some()
            && v.shape().len() == 2
            && is_quant_aligned(v.shape()[1], group_size)
            && v.shape()[1] >= group_size;
        if let (Some(base), true) = (base, packable) {
            // sc-24114: consume the source bytes BEFORE the GPU quantize kernel is scheduled
            // behind them. A safetensors `Load` runs on MLX's CPU stream; a GPU consumer of a
            // not-yet-read `Load` makes `eval` encode a GPU-timeline wait on that read
            // (`Event::wait(stream)` → `encodeWait`), so a graph of hundreds of lazy
            // `Load → quantize` pairs turns into command buffers that sit on the Metal timeline
            // waiting for gigabytes of `pread` from disk — and past the GPU watchdog that is
            // `kIOGPUCommandBufferCallbackErrorTimeout`, after which every later submission of
            // the process is `SubmissionsIgnored`. Reproduced at 90 production tensors from a
            // ~300 MiB/s disk. Evaluating the source on its own stream first leaves the quantize
            // with an already-resident input (no GPU wait at all), and evaluating the packed
            // triple right away bounds the dense transient to one tensor.
            v.eval()?;
            let wbf16 = v.as_dtype(Dtype::Bfloat16)?;
            let (wq, scales, biases) = quantize_cached(&wbf16, group_size, bits)?;
            eval([&wq, &scales, &biases])?;
            out.insert(format!("{base}.weight"), wq);
            out.insert(format!("{base}.scales"), scales);
            out.insert(format!("{base}.biases"), biases);
        } else {
            out.insert(k, v);
        }
    }
    Ok(out)
}

// ============================================================================================
// Turnkey-assembly glue — the Group-B converter tail the provider `convert.rs` modules used to
// clone verbatim (sc-9108 / F-045). Each provider still owns its per-component pack predicates and
// its `prequantize_turnkey` wiring, but routes the config annotation, the symlink-resolving
// directory copy, and the non-weight asset tail through these shared helpers so the six converters
// stay byte-identical (they had drifted: z-image dropped `LICENSE.txt`; krea's copy skipped the
// symlink deref; sensenova never annotated its config).
// ============================================================================================

/// The canonical set of top-level non-weight assets a turnkey snapshot copies verbatim from the
/// dense source (in addition to the component dirs it packs / passes through): the diffusers pipeline
/// manifest and every license/readme variant an upstream repo might ship. A converter iterates this,
/// copying each that exists (deref'ing HF-cache symlinks). Owning it here fixes the F-045 drift where
/// z-image omitted `LICENSE.txt` and shipped license-less rehosted tiers.
pub const TURNKEY_ASSET_FILES: &[&str] = &[
    "model_index.json",
    "LICENSE",
    "LICENSE.md",
    "LICENSE.txt",
    "README.md",
];

/// Copy `src/config.json` to `dst/config.json` with a `"quantization": {"bits", "group_size"}` block
/// added. A missing source config starts from an empty object. The written bytes are
/// `serde_json::to_string_pretty` of the merged value, byte-identical across every provider that
/// packs per-component dirs.
///
/// **This block is load-bearing, not provenance.** An earlier version of this doc said the Rust
/// loaders auto-detect packed weights via `{base}.scales` and "ignore this block" — true of the
/// *loaders*, false of everything else, and the SC-15525 review traced a real coverage gap back to
/// believing it. [`packed_quant_bits`] reads this marker and **only** this marker, so a packed
/// component shipped without it is reported as dense and three separate decisions go the wrong way:
///
/// * [`needs_load_time_quant`] answers "yes", which is what the F-144 requested-vs-packed tier guard
///   consults — so a Q4 request against an unmarked packed-Q8 component passes the guard and is then
///   served Q8 by a `quantize()` that no-ops on already-packed weights;
/// * a memory-strategy contract's "does this load leave the transformer lazy" predicate answers "no"
///   and withholds rung 4. That is exactly why `illustrious_xl_v1`/`illustrious_xl_v2` at q8 publish
///   no bounded transformer residency — see `mlx_gen_sdxl::memory_strategy::load_leaves_blocks_lazy`;
/// * anything sizing a fit from the declared tier reads the wrong one.
///
/// A packing path that omits this call produces a snapshot that *works* and is *mislabelled*, which
/// is the hardest kind to notice. Every packing path must write it.
pub fn write_quantized_config(src: &Path, dst: &Path, bits: i32, group_size: i32) -> Result<()> {
    let src_cfg = src.join("config.json");
    let mut v: serde_json::Value = if src_cfg.exists() {
        serde_json::from_str(&gen_core::__model_input_reader!(
std::fs::read_to_string(&src_cfg),
"crates/media/mlx-gen/src/quant.rs::write_quantized_config::std::fs::read_to_string@25585",
ArtifactInput,
&src_cfg,
|__model_input_path| std::fs::read_to_string(__model_input_path)
)?)
        .map_err(|e| {
            crate::Error::Msg(format!("quant convert: parse {}: {e}", src_cfg.display()))
        })?
    } else {
        serde_json::json!({})
    };
    v["quantization"] = serde_json::json!({ "bits": bits, "group_size": group_size });
    let text = serde_json::to_string_pretty(&v)
        .map_err(|e| crate::Error::Msg(format!("quant convert: serialize config.json: {e}")))?;
    std::fs::create_dir_all(dst)?;
    std::fs::write(dst.join("config.json"), text)?;
    Ok(())
}

/// Recursively copy a directory's files, resolving symlinks (HF snapshots symlink into
/// `../../blobs/…`) to real bytes so the assembled tier is self-contained and HF-uploadable. This is
/// the symlink-resolving copy the sibling converters share (the sc-9108 unification point where
/// krea's non-deref'ing copy is brought into line).
///
/// A symlinked subdirectory is followed, but a directory symlink that leads back to one of the
/// directories being copied (a cycle) is a typed error rather than unbounded recursion → stack
/// overflow (sc-5330, the hardening bernini's private copy carried).
pub fn copy_dir(src: &Path, dst: &Path) -> Result<()> {
    copy_dir_guarded(src, dst, &mut Vec::new())
}

fn copy_dir_guarded(src: &Path, dst: &Path, ancestors: &mut Vec<std::path::PathBuf>) -> Result<()> {
    let canonical = std::fs::canonicalize(src)?;
    if ancestors.contains(&canonical) {
        return Err(crate::Error::Msg(format!(
            "quant convert: directory symlink cycle at {} (resolves to {})",
            src.display(),
            canonical.display()
        )));
    }
    ancestors.push(canonical);
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let path = entry.path();
        // Never carry a hidden entry into an assembled tier: an AppleDouble sidecar copied in here
        // would be uploaded with the turnkey and then break `Weights::from_dir` for everyone who
        // downloads it (SceneWorks#1333).
        if gen_core::weightsmeta::is_hidden_file(&path) {
            continue;
        }
        let target = dst.join(entry.file_name());
        if path.is_dir() {
            copy_dir_guarded(&path, &target, ancestors)?;
        } else {
            let real = std::fs::canonicalize(&path)?;
            std::fs::copy(&real, &target)?;
        }
    }
    ancestors.pop();
    Ok(())
}

/// Merge every tensor of `w` into `map`, refusing a key `map` already holds (overlapping shards or
/// variants are a corrupt snapshot). `context` prefixes the error (e.g. `"sdxl convert: unet"`).
pub fn merge_weights_unique(
    map: &mut HashMap<String, Array>,
    w: &Weights,
    context: &str,
) -> Result<()> {
    for k in w.keys() {
        let v = w.get(k).expect("listed key").clone();
        if map.insert(k.to_string(), v).is_some() {
            return Err(crate::Error::Msg(format!("{context}: duplicate key `{k}`")));
        }
    }
    Ok(())
}

/// Copy a single top-level asset `name` from `src_root` to `dst_root` if it exists, deref'ing an
/// HF-cache symlink to real bytes. Returns whether the file was present and copied. The per-file
/// building block behind [`copy_turnkey_assets`] (some converters — e.g. sensenova — need the copied?
/// signal per file).
pub fn copy_asset(src_root: &Path, dst_root: &Path, name: &str) -> Result<bool> {
    let src = src_root.join(name);
    if !src.exists() {
        return Ok(false);
    }
    let real = std::fs::canonicalize(&src)?;
    std::fs::create_dir_all(dst_root)?;
    std::fs::copy(&real, dst_root.join(name))?;
    Ok(true)
}

/// Copy every [`TURNKEY_ASSET_FILES`] entry that exists from `src_root` to `dst_root` (deref'ing
/// symlinks). The shared non-weight tail of `prequantize_turnkey`.
pub fn copy_turnkey_assets(src_root: &Path, dst_root: &Path) -> Result<()> {
    for name in TURNKEY_ASSET_FILES {
        copy_asset(src_root, dst_root, name)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    /// sc-5330: a directory symlink back to an ancestor is refused instead of recursing forever,
    /// while a file symlink is still copied as its target's bytes and a non-cyclic directory
    /// symlink is still followed.
    #[cfg(unix)]
    #[test]
    fn copy_dir_refuses_a_directory_symlink_cycle_and_derefs_file_links() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        let blobs = tmp.path().join("blobs");
        std::fs::create_dir_all(src.join("sub")).unwrap();
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::write(blobs.join("blob"), b"payload").unwrap();
        std::os::unix::fs::symlink(blobs.join("blob"), src.join("sub/config.json")).unwrap();
        std::os::unix::fs::symlink(&blobs, src.join("linked")).unwrap();

        let ok = tmp.path().join("ok");
        super::copy_dir(&src, &ok).unwrap();
        let copied = ok.join("sub/config.json");
        assert!(!std::fs::symlink_metadata(&copied).unwrap().is_symlink());
        assert_eq!(std::fs::read(&copied).unwrap(), b"payload");
        assert_eq!(std::fs::read(ok.join("linked/blob")).unwrap(), b"payload");

        std::os::unix::fs::symlink(&src, src.join("sub/loop")).unwrap();
        let err = super::copy_dir(&src, &tmp.path().join("cyclic")).unwrap_err();
        assert!(err.to_string().contains("symlink cycle"), "{err}");
    }

    use super::*;

    fn marker_fixture(tmp: &tempfile::TempDir, body: Option<&str>) -> std::path::PathBuf {
        let root = tmp.path().join(format!(
            "mlx-gen-quant-marker-{:?}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
        ));
        let dir = root.join("transformer");
        std::fs::create_dir_all(&dir).unwrap();
        if let Some(body) = body {
            std::fs::write(dir.join("config.json"), body).unwrap();
        }
        root
    }

    #[test]
    fn quant_tier_missing_or_unmarked_config_is_dense() {
        let tmp = tempfile::tempdir().unwrap();
        for body in [None, Some("{}"), Some(r#"{"hidden_size": 64}"#)] {
            let root = marker_fixture(&tmp, body);
            assert_eq!(packed_quant_bits(&root, "transformer").unwrap(), None);
            assert!(needs_load_time_quant(&root, "transformer", 4, "model").unwrap());
            std::fs::remove_dir_all(root).ok();
        }
    }

    /// An UNMARKED packed component (an older published q8 revision with no `quantization` block)
    /// is read as packed from its headers — never as dense — and an inconsistent pack is an error.
    #[test]
    fn an_unmarked_packed_component_is_detected_from_its_headers() {
        fn write(dir: &Path, tensors: &[(&str, &str, &[usize])]) {
            std::fs::create_dir_all(dir).unwrap();
            let mut header = serde_json::Map::new();
            let mut offset = 0usize;
            for (name, dtype, shape) in tensors {
                let width = if *dtype == "F32" || *dtype == "U32" {
                    4
                } else {
                    2
                };
                let bytes = shape.iter().product::<usize>() * width;
                header.insert(
                    (*name).into(),
                    serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + bytes]}),
                );
                offset += bytes;
            }
            let text = serde_json::Value::Object(header).to_string();
            let mut data = (text.len() as u64).to_le_bytes().to_vec();
            data.extend(text.as_bytes());
            data.resize(data.len() + offset, 0);
            std::fs::write(dir.join("diffusion_pytorch_model.safetensors"), data).unwrap();
        }
        let tmp = tempfile::tempdir().unwrap();
        // in = 128: Q8 codes are [out, 128·8/32 = 32], scales [out, 128/64 = 2].
        let q8 = tmp.path().join("q8/unet");
        write(
            &q8,
            &[
                ("a.weight", "U32", &[4, 32]),
                ("a.scales", "BF16", &[4, 2]),
                ("a.biases", "BF16", &[4, 2]),
                ("norm.weight", "BF16", &[4]),
            ],
        );
        std::fs::write(q8.join("config.json"), r#"{"sample_size": 128}"#).unwrap();
        assert_eq!(
            packed_quant_bits(&tmp.path().join("q8"), "unet").unwrap(),
            Some(8)
        );
        assert_eq!(
            packed_quant_group_size_at(&q8).unwrap(),
            Some(HEADER_PACKED_GROUP_SIZE)
        );
        assert!(!needs_load_time_quant(&tmp.path().join("q8"), "unet", 8, "sdxl").unwrap());
        assert!(needs_load_time_quant(&tmp.path().join("q8"), "unet", 4, "sdxl").is_err());
        let q4 = tmp.path().join("q4/unet");
        write(
            &q4,
            &[("a.weight", "U32", &[4, 16]), ("a.scales", "BF16", &[4, 2])],
        );
        assert_eq!(
            packed_quant_bits(&tmp.path().join("q4"), "unet").unwrap(),
            Some(4)
        );
        // Dense stays dense; a mixed or non-Q4/Q8 pack is an error, never "dense".
        let dense = tmp.path().join("dense/unet");
        write(&dense, &[("a.weight", "F16", &[4, 128])]);
        assert_eq!(
            packed_quant_bits(&tmp.path().join("dense"), "unet").unwrap(),
            None
        );
        let mixed = tmp.path().join("mixed/unet");
        write(
            &mixed,
            &[
                ("a.weight", "U32", &[4, 32]),
                ("a.scales", "BF16", &[4, 2]),
                ("b.weight", "U32", &[4, 16]),
                ("b.scales", "BF16", &[4, 2]),
            ],
        );
        assert!(packed_quant_bits(&tmp.path().join("mixed"), "unet").is_err());
        let odd = tmp.path().join("odd/unet");
        write(
            &odd,
            &[("a.weight", "U32", &[4, 24]), ("a.scales", "BF16", &[4, 2])],
        );
        assert!(packed_quant_bits(&tmp.path().join("odd"), "unet").is_err());
    }

    #[test]
    fn quant_tier_match_skips_and_mismatch_names_context() {
        let tmp = tempfile::tempdir().unwrap();
        let root = marker_fixture(&tmp, Some(r#"{"quantization":{"bits":8,"group_size":64}}"#));
        assert!(!needs_load_time_quant(&root, "transformer", 8, "lens_turbo").unwrap());
        let error = needs_load_time_quant(&root, "transformer", 4, "lens_turbo")
            .unwrap_err()
            .to_string();
        assert!(error.contains("lens_turbo"), "{error}");
        assert!(error.contains("transformer"), "{error}");
        assert!(error.contains("Q8") && error.contains("Q4"), "{error}");
        std::fs::remove_dir_all(root).ok();
    }

    /// sc-24114: `quantize_map` consumes every packed target's source bytes before it returns —
    /// the eval boundary that keeps a GPU quantize kernel from waiting on a disk read inside a
    /// Metal command buffer (`kIOGPUCommandBufferCallbackErrorTimeout` on a slow volume).
    ///
    /// Mechanical pin: load a file lazily, run `quantize_map`, then **truncate the file to zero
    /// bytes** before anything downstream evaluates. If the reads were still pending (the
    /// pre-fix graph deferred them to `save_map`'s single eval), the later eval reads a
    /// zero-length file and cannot reproduce the eager reference; if they were consumed inside
    /// `quantize_map`, the packed triple is byte-identical to the reference.
    ///
    /// *Mutation that reds this:* removing the `v.eval()` / `eval([&wq, &scales, &biases])` pair
    /// from `quantize_map`.
    #[test]
    fn quantize_map_consumes_its_source_bytes_before_returning() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("model.safetensors");
        let w = Array::from_slice(
            &(0..256 * 128)
                .map(|i| (i as f32 * 0.37).sin())
                .collect::<Vec<_>>(),
            &[256, 128],
        );
        let norm = Array::ones::<f32>(&[128]).unwrap();
        Array::save_safetensors(
            [
                ("blocks.0.proj.weight", &w),
                ("blocks.0.norm.weight", &norm),
            ],
            None::<&HashMap<String, String>>,
            &path,
        )
        .unwrap();
        // The eager reference, from bytes read while the file is intact.
        let (ewq, esc, ebi) =
            mlx_rs::ops::quantize(w.as_dtype(Dtype::Bfloat16).unwrap(), 64, 8).unwrap();
        eval([&ewq, &esc, &ebi]).unwrap();

        let lazy = load_dir_map(tmp.path()).unwrap();
        let out = quantize_map(lazy, 8, 64, |_| true).unwrap();
        // Nothing has been evaluated by this test yet: pull the source out from under the map.
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(0)
            .unwrap();

        for (key, expected) in [
            ("blocks.0.proj.weight", &ewq),
            ("blocks.0.proj.scales", &esc),
            ("blocks.0.proj.biases", &ebi),
        ] {
            let got = out.get(key).expect(key);
            let same = mlx_rs::ops::eq(got, expected)
                .and_then(|m| m.all(None))
                .and_then(|a| a.try_item::<bool>());
            assert!(
                matches!(same, Ok(true)),
                "{key}: packed output does not match the eager reference after the source was \
                 truncated ({same:?}) — quantize_map deferred its reads"
            );
        }
    }

    #[test]
    fn packed_group_size_accepts_only_mlx_affine_geometries() {
        let tmp = tempfile::tempdir().unwrap();
        for group_size in [32, 64, 128] {
            let body = format!(r#"{{"quantization":{{"bits":8,"group_size":{group_size}}}}}"#);
            let root = marker_fixture(&tmp, Some(&body));
            assert_eq!(
                packed_quant_group_size_at(&root.join("transformer")).unwrap(),
                Some(group_size)
            );
            std::fs::remove_dir_all(root).ok();
        }
        for marker in [
            r#"{"quantization":{"bits":8}}"#,
            r#"{"quantization":{"bits":8,"group_size":"32"}}"#,
            r#"{"quantization":{"bits":8,"group_size":16}}"#,
            r#"{"quantization":{"bits":8,"group_size":2147483648}}"#,
        ] {
            let root = marker_fixture(&tmp, Some(marker));
            assert!(
                packed_quant_group_size_at(&root.join("transformer")).is_err(),
                "group-size marker must fail: {marker}"
            );
            std::fs::remove_dir_all(root).ok();
        }
    }

    #[test]
    fn quant_tier_rejects_every_present_but_malformed_marker() {
        let tmp = tempfile::tempdir().unwrap();
        for body in [
            "{",
            r#"{"quantization":null}"#,
            r#"{"quantization":[]}"#,
            r#"{"quantization":{}}"#,
            r#"{"quantization":{"bits":"4"}}"#,
            r#"{"quantization":{"bits":4.0}}"#,
            r#"{"quantization":{"bits":3}}"#,
            r#"{"quantization":{"bits":2147483648}}"#,
        ] {
            let root = marker_fixture(&tmp, Some(body));
            assert!(
                packed_quant_bits(&root, "transformer").is_err(),
                "marker must fail: {body}"
            );
            std::fs::remove_dir_all(root).ok();
        }
    }

    #[test]
    fn quant_tier_only_swallows_not_found_and_rejects_unsafe_components() {
        let tmp = tempfile::tempdir().unwrap();
        let root = marker_fixture(&tmp, None);
        std::fs::create_dir(root.join("transformer/config.json")).unwrap();
        assert!(packed_quant_bits(&root, "transformer").is_err());
        for component in ["", "../transformer", "/transformer", "transformer/../vae"] {
            assert!(
                packed_quant_bits(&root, component).is_err(),
                "{component:?}"
            );
        }
        assert!(needs_load_time_quant(&root, "vae", 3, "model").is_err());
        std::fs::remove_dir_all(root).ok();
    }

    /// Pure production validator coverage: no native device/stream/tensor is constructed.
    #[test]
    fn packed_metadata_preserves_exact_load_geometry_refusals() {
        let check = |w: &[i32], wd, s: &[i32], sd, g| packed_bits_from_metadata(w, wd, s, sd, g);
        assert_eq!(
            check(&[128, 32], Dtype::Uint32, &[128, 4], Dtype::Float32, 64).unwrap(),
            4
        );
        assert_eq!(
            check(&[64, 32], Dtype::Uint32, &[64, 2], Dtype::Float16, 64).unwrap(),
            8
        );
        assert!(check(&[2, 33], Dtype::Uint32, &[2, 2], Dtype::Bfloat16, 64).is_err());
        assert!(check(&[3, 32], Dtype::Uint32, &[2, 2], Dtype::Float32, 64).is_err());
        assert!(check(&[2, 32], Dtype::Float32, &[2, 2], Dtype::Float32, 64).is_err());
        assert!(check(&[2, 32], Dtype::Uint32, &[2, 2], Dtype::Uint32, 64).is_err());
        assert!(check(&[2, 32], Dtype::Uint32, &[2, 0], Dtype::Float32, 64).is_err());
        assert!(check(&[2, 24], Dtype::Uint32, &[2, 4], Dtype::Float32, 64).is_err());
        assert!(check(&[2], Dtype::Uint32, &[2, 4], Dtype::Float32, 64).is_err());
        assert!(check(&[2, 32], Dtype::Uint32, &[2, i32::MAX], Dtype::Float32, 64).is_err());
        assert!(check(&[2, i32::MAX], Dtype::Uint32, &[2, 2], Dtype::Float32, 64).is_err());
    }

    /// F-011: a Q4 pack at group_size 64 derives bits == 4 from the standard shapes.
    #[test]
    fn packed_bits_derives_q4() {
        // scales [out, in/gs] = [128, 4] ⇒ in_dim 256; wq [out, in·4/32] = [128, 32] ⇒ bits 4.
        let scales = Array::zeros::<f32>(&[128, 4]).unwrap();
        let wq = Array::zeros::<u32>(&[128, 32]).unwrap();
        assert_eq!(packed_bits(&wq, &scales, 64).unwrap(), 4);
    }

    #[test]
    fn packed_bits_derives_q8() {
        // scales [out, in/gs] = [64, 2] ⇒ in_dim 128; wq [out, in·8/32] = [64, 32] ⇒ bits 8.
        let scales = Array::zeros::<f32>(&[64, 2]).unwrap();
        let wq = Array::zeros::<u32>(&[64, 32]).unwrap();
        assert_eq!(packed_bits(&wq, &scales, 64).unwrap(), 8);
    }

    #[test]
    fn packed_bits_rejects_1d_shapes() {
        let scales = Array::zeros::<f32>(&[64]).unwrap(); // 1-D
        let wq = Array::zeros::<u32>(&[64, 32]).unwrap();
        let err = packed_bits(&wq, &scales, 64).unwrap_err().to_string();
        assert!(err.contains("must be 2-D"), "{err}");
    }

    #[test]
    fn packed_bits_rejects_zero_in_dim() {
        // [out, 0] scales ⇒ in_dim 0 ⇒ integer divide-by-zero today; now a typed error.
        let scales = Array::zeros::<f32>(&[64, 0]).unwrap();
        let wq = Array::zeros::<u32>(&[64, 32]).unwrap();
        let err = packed_bits(&wq, &scales, 64).unwrap_err().to_string();
        assert!(err.contains("zero input dim"), "{err}");
    }

    #[test]
    fn packed_bits_rejects_non_q4_q8_width() {
        // in_dim 256, wq cols 24 ⇒ bits 24·32/256 = 3 ∉ {4,8}.
        let scales = Array::zeros::<f32>(&[64, 4]).unwrap(); // in_dim 256
        let wq = Array::zeros::<u32>(&[64, 24]).unwrap(); // bits 3
        let err = packed_bits(&wq, &scales, 64).unwrap_err().to_string();
        assert!(err.contains("∉ {4, 8}"), "{err}");
    }

    #[test]
    fn packed_bits_rejects_truncated_width_wrong_rows_and_non_u32_codes() {
        let scales = Array::zeros::<f32>(&[2, 2]).unwrap();
        let truncated = Array::zeros::<u32>(&[2, 33]).unwrap();
        assert!(packed_bits(&truncated, &scales, 64).is_err());

        let wrong_rows = Array::zeros::<u32>(&[3, 32]).unwrap();
        assert!(packed_bits(&wrong_rows, &scales, 64).is_err());

        let float_codes = Array::zeros::<f32>(&[2, 32]).unwrap();
        assert!(packed_bits(&float_codes, &scales, 64).is_err());
    }
}
