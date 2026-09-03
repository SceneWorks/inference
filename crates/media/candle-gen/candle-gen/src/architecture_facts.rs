//! Snapshot-read architecture axes for [`gen_core::MemoryArchitectureFacts`] (epic SC-22657, E2).
//!
//! Every Candle provider declares the same eight axes, and almost every one of them reads those
//! axes out of the component `config.json` files its loader already parses. The reading is where
//! the honesty rules live — a zero is never a legitimate axis, a stage count that cannot be turned
//! into a scale has no honest value to publish, and a missing config is not an error because the
//! contract is constructible before any asset exists on disk. Keeping those rules in one place
//! means a provider's own `architecture_facts` function is just its key names.
//!
//! Nothing here infers an axis from a model id: a snapshot whose config disagrees with the
//! reference config publishes what it actually says.

use std::path::Path;

use serde_json::Value;

/// The materialized snapshot directory a contract is being built for, or `None`.
///
/// This is the single gate that keeps a **weights-free** contract's architecture facts empty. The
/// registry builds its contract surfaces against the sentinel path
/// `/__sceneworks_memory_contract_surface__`, which is deliberately not on disk: nothing about the
/// pipeline that *would* be loaded has been resolved there, so no axis is knowable and every one
/// stays `None` (E2). A single-file import is likewise not a component snapshot.
pub fn snapshot_root(spec: &gen_core::LoadSpec) -> Option<&Path> {
    match &spec.weights {
        gen_core::WeightsSource::Dir(root) if root.is_dir() => Some(root.as_path()),
        _ => None,
    }
}

/// Narrow a geometry constant the loader already holds — a field of the crate's own model config
/// struct — to a declared axis, applying the same no-zero rule as [`axis`].
pub fn declared(value: usize) -> Option<u32> {
    u32::try_from(value).ok().filter(|value| *value != 0)
}

/// Read one component `config.json` from a snapshot root (`<root>/<component>/config.json`).
///
/// A missing or unparseable file is not an error: the contract is constructible before any asset
/// exists on disk, and every architecture axis it feeds is `Option`. Pass `""` as `component` to
/// read the root's own `config.json`.
pub fn component_config(root: &Path, component: &str) -> Option<Value> {
    let dir = if component.is_empty() {
        root.to_path_buf()
    } else {
        root.join(component)
    };
    config_file(&dir.join("config.json"))
}

/// Read an explicitly located JSON config. Missing or unparseable is `None`, never an error.
pub fn config_file(path: &Path) -> Option<Value> {
    serde_json::from_slice(&std::fs::read(path).ok()?).ok()
}

/// Read the first component config that exists, so a provider can name the layouts it accepts
/// (`transformer` for a diffusers snapshot, `unet` for the SD-family layout) in one call.
pub fn first_component_config(root: &Path, components: &[&str]) -> Option<Value> {
    components
        .iter()
        .find_map(|component| component_config(root, component))
}

/// Narrow a JSON number to a declared architecture axis.
///
/// Zero and out-of-range values are rejected rather than published: no axis has a legitimate zero,
/// and `Some(0)` would silently zero any activation estimate that multiplied by it (E2).
pub fn axis(value: Option<&Value>) -> Option<u32> {
    let value = u32::try_from(value?.as_u64()?).ok()?;
    (value != 0).then_some(value)
}

/// The first of `keys` the config actually declares, narrowed by [`axis`].
///
/// Diffusers configs spell the same axis differently across families (`num_attention_heads` versus
/// `num_heads`, `num_layers` versus `depth`); listing the accepted spellings in priority order lets
/// a provider read whichever one its own snapshots ship without asserting a value.
pub fn axis_of(config: Option<&Value>, keys: &[&str]) -> Option<u32> {
    keys.iter()
        .find_map(|key| axis(config.and_then(|config| config.get(*key))))
}

/// One element of a list-valued axis, e.g. the spatial entry of a `[temporal, height, width]`
/// `patch_size`. A scalar-valued key is read directly, so both spellings are accepted.
pub fn axis_at(config: Option<&Value>, key: &str, index: usize) -> Option<u32> {
    let value = config?.get(key)?;
    match value.as_array() {
        Some(entries) => axis(entries.get(index)),
        None => axis(Some(value)),
    }
}

/// Per-head channel width for a uniform-head architecture.
///
/// Publishing the quotient only when it divides evenly keeps a non-uniform snapshot from claiming
/// a head width it does not have.
pub fn head_dim(hidden_dim: Option<u32>, attention_heads: Option<u32>) -> Option<u32> {
    match (hidden_dim, attention_heads) {
        (Some(dim), Some(heads)) if dim % heads == 0 => Some(dim / heads),
        _ => None,
    }
}

/// Pixels per latent unit, read from a VAE's per-stage channel list (`block_out_channels` on a
/// diffusers autoencoder, `dim_mult` on the WAN-family 3D autoencoder).
///
/// Each stage after the first halves both spatial axes, so four stages give the x8 scale. A
/// pathological stage count has no honest scale to publish: clamping the shift would invent one,
/// so the axis is declined instead (E2). A VAE that declares its own `scale_factor_spatial` is
/// read through [`axis_of`] instead — the stage list is only the fallback, and it misses any extra
/// patchification the encoder applies on top of the stages.
pub fn spatial_scale_from_stages(vae: Option<&Value>, keys: &[&str]) -> Option<u32> {
    keys.iter()
        .find_map(|key| stage_count(vae, key))
        .and_then(|stages| stages.checked_sub(1))
        .and_then(shift_scale)
}

/// Frames per latent unit, read from a video VAE's boolean per-stage downsample list (WAN's
/// `temperal_downsample`, and the same shape under the `temporal_downsample` spelling).
///
/// Only the `true` stages halve the temporal axis, so this counts them rather than the stages.
/// A list of all-`false` flags means the VAE does not compress time at all, which is a scale of 1.
pub fn temporal_scale_from_flags(vae: Option<&Value>, keys: &[&str]) -> Option<u32> {
    let flags = keys.iter().find_map(|key| {
        vae.and_then(|vae| vae.get(*key))
            .and_then(Value::as_array)
            .filter(|entries| entries.iter().all(|entry| entry.is_boolean()))
    })?;
    shift_scale(
        u32::try_from(
            flags
                .iter()
                .filter(|entry| entry.as_bool() == Some(true))
                .count(),
        )
        .ok()?,
    )
}

/// The number of entries in a list-valued config key, e.g. a VAE's `block_out_channels`.
pub fn stage_count(config: Option<&Value>, key: &str) -> Option<u32> {
    u32::try_from(
        config?
            .get(key)?
            .as_array()
            .filter(|stages| !stages.is_empty())?
            .len(),
    )
    .ok()
}

/// `2^downsamples`, declined for a shift no real VAE stage list could produce.
fn shift_scale(downsamples: u32) -> Option<u32> {
    (downsamples <= 5).then(|| 1_u32 << downsamples)
}

/// Bytes per element of the pipeline's pinned activation dtype.
pub fn dtype_width(dtype: candle_core::DType) -> Option<u32> {
    u32::try_from(dtype.size_in_bytes()).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn json(text: &str) -> Value {
        serde_json::from_str(text).unwrap()
    }

    #[test]
    fn a_missing_or_unparseable_config_is_absent_not_an_error() {
        let tmp = tempfile::tempdir().unwrap();
        assert_eq!(component_config(tmp.path(), "transformer"), None);
        std::fs::create_dir_all(tmp.path().join("transformer")).unwrap();
        std::fs::write(tmp.path().join("transformer/config.json"), b"{ not json").unwrap();
        assert_eq!(component_config(tmp.path(), "transformer"), None);
        std::fs::write(tmp.path().join("transformer/config.json"), br#"{"n":1}"#).unwrap();
        assert_eq!(
            component_config(tmp.path(), "transformer"),
            Some(json(r#"{"n":1}"#))
        );
        // `""` reads the root's own config, the layout a single-component snapshot ships.
        std::fs::write(tmp.path().join("config.json"), br#"{"n":2}"#).unwrap();
        assert_eq!(component_config(tmp.path(), ""), Some(json(r#"{"n":2}"#)));
    }

    #[test]
    fn the_first_declared_layout_wins() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("unet")).unwrap();
        std::fs::write(tmp.path().join("unet/config.json"), br#"{"which":"unet"}"#).unwrap();
        assert_eq!(
            first_component_config(tmp.path(), &["transformer", "unet"]),
            Some(json(r#"{"which":"unet"}"#))
        );
        assert_eq!(first_component_config(tmp.path(), &["transformer"]), None);
    }

    #[test]
    fn zero_and_out_of_range_axes_are_declined_rather_than_published() {
        let config = json(r#"{"zero": 0, "negative": -4, "huge": 68719476736, "good": 24}"#);
        for key in ["zero", "negative", "huge", "absent"] {
            assert_eq!(axis(config.get(key)), None, "{key} must not be published");
        }
        assert_eq!(axis(config.get("good")), Some(24));
    }

    #[test]
    fn axis_of_reads_the_first_spelling_the_config_declares() {
        let config = json(r#"{"num_heads": 16, "num_attention_heads": 0}"#);
        // The zero-valued preferred spelling is not a value, so the fallback spelling is read.
        assert_eq!(
            axis_of(Some(&config), &["num_attention_heads", "num_heads"]),
            Some(16)
        );
        assert_eq!(axis_of(Some(&config), &["depth"]), None);
        assert_eq!(axis_of(None, &["num_heads"]), None);
    }

    #[test]
    fn a_list_valued_axis_reads_its_own_index_and_a_scalar_reads_directly() {
        let config = json(r#"{"patch_size": [1, 2, 2], "scalar_patch": 2}"#);
        assert_eq!(axis_at(Some(&config), "patch_size", 1), Some(2));
        assert_eq!(axis_at(Some(&config), "patch_size", 0), Some(1));
        assert_eq!(axis_at(Some(&config), "patch_size", 7), None);
        assert_eq!(axis_at(Some(&config), "scalar_patch", 1), Some(2));
    }

    #[test]
    fn head_dim_is_published_only_for_a_uniform_head_architecture() {
        assert_eq!(head_dim(Some(3840), Some(30)), Some(128));
        assert_eq!(head_dim(Some(3841), Some(30)), None);
        assert_eq!(head_dim(None, Some(30)), None);
        assert_eq!(head_dim(Some(3840), None), None);
    }

    #[test]
    fn the_spatial_scale_follows_the_stage_list_and_declines_a_pathological_one() {
        assert_eq!(
            spatial_scale_from_stages(
                Some(&json(r#"{"block_out_channels": [1,2,3,4]}"#)),
                &["block_out_channels"]
            ),
            Some(8)
        );
        assert_eq!(
            spatial_scale_from_stages(
                Some(&json(r#"{"block_out_channels": [1]}"#)),
                &["block_out_channels"]
            ),
            Some(1)
        );
        assert_eq!(
            spatial_scale_from_stages(
                Some(&json(r#"{"block_out_channels": []}"#)),
                &["block_out_channels"]
            ),
            None
        );
        assert_eq!(
            spatial_scale_from_stages(
                Some(&json(r#"{"block_out_channels": [1,2,3,4,5,6,7,8,9]}"#)),
                &["block_out_channels"]
            ),
            None,
            "a stage count no VAE ships has no honest scale to publish"
        );
        assert_eq!(
            spatial_scale_from_stages(None, &["block_out_channels"]),
            None
        );
    }

    #[test]
    fn the_temporal_scale_counts_only_the_downsampling_stages() {
        let wan = json(r#"{"temperal_downsample": [false, true, true]}"#);
        assert_eq!(
            temporal_scale_from_flags(Some(&wan), &["temporal_downsample", "temperal_downsample"]),
            Some(4)
        );
        // An image VAE's stage list carries no booleans, so no temporal axis is invented from it.
        let image = json(r#"{"temporal_downsample": [128, 256]}"#);
        assert_eq!(
            temporal_scale_from_flags(Some(&image), &["temporal_downsample"]),
            None
        );
        let uncompressed = json(r#"{"temperal_downsample": [false, false]}"#);
        assert_eq!(
            temporal_scale_from_flags(Some(&uncompressed), &["temperal_downsample"]),
            Some(1)
        );
        assert_eq!(
            temporal_scale_from_flags(None, &["temperal_downsample"]),
            None
        );
    }

    #[test]
    fn only_a_materialized_snapshot_directory_is_a_root() {
        let tmp = tempfile::tempdir().unwrap();
        let spec = gen_core::LoadSpec::new(gen_core::WeightsSource::Dir(tmp.path().to_path_buf()));
        assert_eq!(snapshot_root(&spec), Some(tmp.path()));
        // The registry's weights-free contract surface names a sentinel that is not on disk.
        let weights_free = gen_core::LoadSpec::new(gen_core::WeightsSource::Dir(
            "/__sceneworks_memory_contract_surface__".into(),
        ));
        assert_eq!(snapshot_root(&weights_free), None);
        let single_file = gen_core::LoadSpec::new(gen_core::WeightsSource::File(
            tmp.path().join("model.safetensors"),
        ));
        assert_eq!(snapshot_root(&single_file), None);
    }

    #[test]
    fn a_declared_constant_keeps_the_no_zero_rule() {
        assert_eq!(declared(128), Some(128));
        assert_eq!(declared(0), None);
        assert_eq!(declared(usize::MAX), None);
    }

    #[test]
    fn dtype_width_is_the_pinned_activation_width() {
        assert_eq!(dtype_width(candle_core::DType::BF16), Some(2));
        assert_eq!(dtype_width(candle_core::DType::F16), Some(2));
        assert_eq!(dtype_width(candle_core::DType::F32), Some(4));
    }
}
