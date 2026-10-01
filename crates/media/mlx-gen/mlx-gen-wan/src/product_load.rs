//! The SceneWorks Mac product's load decisions for the five MLX Wan routes.
//!
//! This module is the single source of those decisions: the SceneWorks worker calls the decision
//! functions below when it builds a Wan `LoadSpec`, and the SC-20686 campaign entrypoint builds its
//! spec through [`product_load_spec`], so the two cannot drift.
//!
//! | Route | Weights | Load quantization ([`load_quant`]) | Lightning |
//! | --- | --- | --- | --- |
//! | `wan2_2_ti2v_5b` | a pre-packed quant-matrix tier (`q4/` is the default) | none on a packed tier (its `config.json` is authoritative); the requested quant on a legacy flat root | none |
//! | `wan2_2_t2v_14b` / `wan2_2_i2v_14b` | as above | as above | **on by default**: the per-architecture high/low LoRA pair, 4 steps at guidance 1 |
//! | `wan_vace` | the assembled snapshot: the **Wan2.1-VACE-1.3B** transformer + a base-Wan tier's UMT5/VAE/tokenizer | the requested quant (none: dense bf16) | none |
//! | `wan2_2_vace_fun_14b` | the assembled snapshot: both dense VACE-Fun 14B experts + the same shared components | the requested quant, **Q4 when none is requested** | none |
//!
//! Every route loads at `Precision::Bf16`. Residency is not decided here (the worker's fit gate
//! chooses it per host), so callers pass it in.

use std::path::{Path, PathBuf};

use mlx_gen::{
    AdapterKind, AdapterSpec, Error, LoadSpec, MoeExpert, OffloadPolicy, Precision, Quant, Result,
    WeightsSource,
};

use crate::config::{WanModelConfig, WanVaceConfig};
use crate::model::{MODEL_ID, MODEL_ID_I2V_14B, MODEL_ID_T2V_14B};
use crate::model_vace::{MODEL_ID_VACE, MODEL_ID_VACE_FUN};

/// The bit width of the product's default packed tier (`q4/`) for the tiered routes.
pub const PRODUCT_TIER_BITS: u64 = 4;
/// The Lightning distill recipe's step count.
pub const LIGHTNING_STEPS: u32 = 4;
/// The Lightning distill recipe's guidance (CFG off).
pub const LIGHTNING_GUIDANCE: f32 = 1.0;
/// The Hugging Face repository holding the Lightning LoRA pairs.
pub const LIGHTNING_REPO: &str = "lightx2v/Wan2.2-Lightning";
/// The [`LIGHTNING_REPO`] revision the product pins for its download.
pub const LIGHTNING_REVISION: &str = "18bccf8884ec0a078eed79785eb4ef13ea16ce1e";

fn unknown(route: &str) -> Error {
    Error::Msg(format!(
        "no SceneWorks MLX Wan product load decision for route {route}"
    ))
}

/// Whether `route` ships as a pre-packed quant-matrix tier (`q4/`, `q8/`, `bf16/`).
pub fn uses_packed_tiers(route: &str) -> bool {
    matches!(route, MODEL_ID | MODEL_ID_T2V_14B | MODEL_ID_I2V_14B)
}

/// The load-time quantization for `route` given the request's explicit pick (`requested`, from
/// `advanced.mlxQuantize`) and whether `weights` is a pre-packed tier.
pub fn load_quant(
    route: &str,
    requested: Option<Quant>,
    packed_tier: bool,
) -> Result<Option<Quant>> {
    match route {
        // A packed tier's config is authoritative; the pick chose WHICH tier, never a requant.
        MODEL_ID | MODEL_ID_T2V_14B | MODEL_ID_I2V_14B if packed_tier => Ok(None),
        MODEL_ID | MODEL_ID_T2V_14B | MODEL_ID_I2V_14B | MODEL_ID_VACE => Ok(requested),
        // Both 14B experts at bf16 would risk OOM on a 128 GB Mac: Q4 unless the user picks.
        MODEL_ID_VACE_FUN => Ok(requested.or(Some(Quant::Q4))),
        other => Err(unknown(other)),
    }
}

/// Whether `route` runs the Lightning distill when the request leaves `advanced.lightning` unset.
pub fn lightning_default(route: &str) -> bool {
    lightning_subdir(route).is_some()
}

/// The per-architecture Lightning LoRA subdirectory of [`LIGHTNING_REPO`] (not cross-compatible).
pub fn lightning_subdir(route: &str) -> Option<&'static str> {
    match route {
        MODEL_ID_T2V_14B => Some("Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1"),
        MODEL_ID_I2V_14B => Some("Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1"),
        _ => None,
    }
}

/// The [`LIGHTNING_REPO`] snapshot dir in a Hugging Face hub cache
/// (`<hub>/models--lightx2v--Wan2.2-Lightning/snapshots/<revision>`): the revision `refs/main` names
/// when that snapshot holds files (as the product worker's HF-cache resolver prefers), else the
/// pinned [`LIGHTNING_REVISION`]. Never the repository root.
pub fn lightning_snapshot_in_hub(hub: &Path) -> Result<PathBuf> {
    let repo = hub.join(format!("models--{}", LIGHTNING_REPO.replace('/', "--")));
    let has_files = |dir: &Path| {
        std::fs::read_dir(dir)
            .map(|mut entries| entries.next().is_some())
            .unwrap_or(false)
    };
    let refs_main = std::fs::read_to_string(repo.join("refs").join("main"))
        .ok()
        .map(|revision| revision.trim().to_owned())
        .filter(|revision| !revision.is_empty() && !revision.contains(['/', '\\', '.']));
    refs_main
        .into_iter()
        .chain(std::iter::once(LIGHTNING_REVISION.to_owned()))
        .map(|revision| repo.join("snapshots").join(revision))
        .find(|snapshot| has_files(snapshot))
        .ok_or_else(|| {
            Error::Msg(format!(
                "{LIGHTNING_REPO} has no installed snapshot under {}",
                repo.display()
            ))
        })
}

/// The `(high, low)` Lightning LoRA files for `route` inside a [`LIGHTNING_REPO`] snapshot dir
/// (`.../snapshots/<revision>`, as [`lightning_snapshot_in_hub`] or the worker's HF-cache resolver
/// returns it). A repository root or any other dir is refused: the pair only exists per snapshot.
pub fn lightning_lora_files(route: &str, snapshot: &Path) -> Result<(PathBuf, PathBuf)> {
    if snapshot.parent().and_then(Path::file_name) != Some(std::ffi::OsStr::new("snapshots")) {
        return Err(Error::Msg(format!(
            "{route}: {} is not a {LIGHTNING_REPO} snapshot dir (.../snapshots/<revision>)",
            snapshot.display()
        )));
    }
    let subdir = lightning_subdir(route).ok_or_else(|| {
        Error::Msg(format!(
            "{route}: no Lightning distill LoRA; only the A14B MoE models bake Lightning"
        ))
    })?;
    let high = snapshot.join(subdir).join("high_noise_model.safetensors");
    let low = snapshot.join(subdir).join("low_noise_model.safetensors");
    for file in [&high, &low] {
        if !file.is_file() {
            return Err(Error::Msg(format!(
                "{route}: Lightning LoRA file missing: {}",
                file.display()
            )));
        }
    }
    Ok((high, low))
}

/// The Lightning pair as per-expert adapters at strength 1.0 (high → high-noise, low → low-noise).
pub fn lightning_adapters(high: PathBuf, low: PathBuf) -> Vec<AdapterSpec> {
    [(high, MoeExpert::High), (low, MoeExpert::Low)]
        .into_iter()
        .map(|(path, expert)| AdapterSpec {
            path,
            scale: 1.0,
            kind: AdapterKind::Lora,
            pass_scales: None,
            moe_expert: Some(expert),
        })
        .collect()
}

/// The `quantization.bits` a packed tier root declares in its `config.json` (`None` = dense).
pub fn packed_tier_bits(root: &Path) -> Result<Option<u64>> {
    let path = root.join("config.json");
    let text = std::fs::read_to_string(&path)
        .map_err(|error| Error::Msg(format!("read {}: {error}", path.display())))?;
    let config: serde_json::Value = serde_json::from_str(&text)
        .map_err(|error| Error::Msg(format!("parse {}: {error}", path.display())))?;
    Ok(config
        .get("quantization")
        .and_then(|quantization| quantization.get("bits"))
        .and_then(serde_json::Value::as_u64))
}

/// The `LoadSpec` of the product's DEFAULT request for `route` (no `advanced.mlxQuantize`, no user
/// LoRA): what a measurement entrypoint must load instead of assembling its own spec.
///
/// `lightning` is the request's Lightning toggle and `lightning_snapshot` the [`LIGHTNING_REPO`]
/// snapshot it reads (required when on; Lightning is refused on routes that do not bake it). The
/// default request's weights are also checked: a tiered route must be the `q4/` tier, and
/// `wan_vace` must hold the Wan2.1-VACE-1.3B transformer the Mac product assembles (never the 14B
/// tree the Candle lane reads).
pub fn product_load_spec(
    route: &str,
    weights: &Path,
    offload_policy: OffloadPolicy,
    lightning: bool,
    lightning_snapshot: Option<&Path>,
) -> Result<LoadSpec> {
    let packed_tier = uses_packed_tiers(route);
    let quantize = load_quant(route, None, packed_tier)?;
    if packed_tier {
        let bits = packed_tier_bits(weights)?;
        if bits != Some(PRODUCT_TIER_BITS) {
            return Err(Error::Msg(format!(
                "{route}: the product's default tier is q{PRODUCT_TIER_BITS}; {} is packed at {}",
                weights.display(),
                bits.map_or_else(|| "dense bf16".to_owned(), |bits| format!("{bits} bits"))
            )));
        }
    }
    if route == MODEL_ID_VACE {
        let base = WanVaceConfig::from_model_dir(weights)?.base;
        let product = WanModelConfig::wan21_t2v_1_3b();
        if (base.dim, base.num_layers) != (product.dim, product.num_layers) {
            return Err(Error::Msg(format!(
                "wan_vace: the Mac product loads the Wan2.1-VACE-1.3B transformer (dim {}, {} \
                 layers); {} holds dim {}, {} layers",
                product.dim,
                product.num_layers,
                weights.display(),
                base.dim,
                base.num_layers
            )));
        }
    }
    let adapters = match (lightning, lightning_snapshot) {
        (false, None) => Vec::new(),
        (true, Some(snapshot)) => {
            let (high, low) = lightning_lora_files(route, snapshot)?;
            lightning_adapters(high, low)
        }
        (true, None) => {
            return Err(Error::Msg(format!(
                "{route}: Lightning is on but no {LIGHTNING_REPO} snapshot was given"
            )))
        }
        (false, Some(_)) => {
            return Err(Error::Msg(format!(
                "{route}: a Lightning snapshot was given with Lightning off"
            )))
        }
    };
    let mut spec = LoadSpec::new(WeightsSource::Dir(weights.to_path_buf()))
        .with_offload_policy(offload_policy);
    spec.quantize = quantize;
    spec.precision = Precision::Bf16;
    spec.adapters = adapters;
    Ok(spec)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tier(bits: Option<u64>) -> tempfile::TempDir {
        let root = tempfile::tempdir().expect("temp root");
        let config = match bits {
            Some(bits) => serde_json::json!({"quantization": {"bits": bits, "group_size": 64}}),
            None => serde_json::json!({}),
        };
        std::fs::write(root.path().join("config.json"), config.to_string()).expect("config");
        root
    }

    fn vace_snapshot(heads: u64, layers: u64) -> tempfile::TempDir {
        let root = tempfile::tempdir().expect("temp root");
        let transformer = root.path().join("transformer");
        std::fs::create_dir_all(&transformer).expect("transformer dir");
        let config = serde_json::json!({
            "num_attention_heads": heads,
            "attention_head_dim": 128,
            "num_layers": layers,
        });
        std::fs::write(transformer.join("config.json"), config.to_string()).expect("config");
        root
    }

    /// A synthetic Hugging Face hub holding the Lightning repo the way the cache lays it out:
    /// `models--lightx2v--Wan2.2-Lightning/{refs/main, snapshots/<rev>/<subdir>/* -> blobs/*}`.
    struct LightningHub {
        hub: tempfile::TempDir,
    }

    impl LightningHub {
        fn repo(&self) -> PathBuf {
            self.hub.path().join("models--lightx2v--Wan2.2-Lightning")
        }

        fn path(&self) -> PathBuf {
            self.repo().join("snapshots").join(LIGHTNING_REVISION)
        }
    }

    fn lightning_snapshot() -> LightningHub {
        let hub = LightningHub {
            hub: tempfile::tempdir().expect("temp hub"),
        };
        let repo = hub.repo();
        std::fs::create_dir_all(repo.join("refs")).expect("refs");
        std::fs::write(repo.join("refs/main"), LIGHTNING_REVISION).expect("refs/main");
        std::fs::create_dir_all(repo.join("blobs")).expect("blobs");
        for route in [MODEL_ID_T2V_14B, MODEL_ID_I2V_14B] {
            let subdir = lightning_subdir(route).expect("A14B");
            let dir = hub.path().join(subdir);
            std::fs::create_dir_all(&dir).expect("subdir");
            for name in [
                "high_noise_model.safetensors",
                "low_noise_model.safetensors",
            ] {
                let blob = repo.join("blobs").join(format!("{subdir}-{name}"));
                std::fs::write(&blob, b"lora").expect("blob");
                std::os::unix::fs::symlink(&blob, dir.join(name)).expect("snapshot link");
            }
        }
        hub
    }

    #[test]
    fn lightning_resolves_the_hf_snapshot_never_the_repo_root() {
        let hub = lightning_snapshot();
        assert_eq!(
            lightning_snapshot_in_hub(hub.hub.path()).unwrap(),
            hub.path()
        );
        let (high, low) = lightning_lora_files(MODEL_ID_T2V_14B, &hub.path()).unwrap();
        assert!(high.starts_with(hub.path()) && low.starts_with(hub.path()));
        // The SC-20686 W2 defect: a path built on the repository root (`models--…/<subdir>/…`,
        // what a snapshot derived from a symlink-resolved blob path yields) is refused outright.
        let error = lightning_lora_files(MODEL_ID_T2V_14B, &hub.repo()).unwrap_err();
        assert!(error.to_string().contains("not a"), "{error}");
        // A stale `refs/main` naming an absent snapshot falls back to the pinned revision.
        std::fs::write(hub.repo().join("refs/main"), "0".repeat(40)).unwrap();
        assert_eq!(
            lightning_snapshot_in_hub(hub.hub.path()).unwrap(),
            hub.path()
        );
        std::fs::remove_dir_all(hub.repo().join("snapshots")).unwrap();
        assert!(lightning_snapshot_in_hub(hub.hub.path()).is_err());
    }

    #[test]
    fn load_quant_matches_the_worker_decisions() {
        for route in [MODEL_ID, MODEL_ID_T2V_14B, MODEL_ID_I2V_14B] {
            assert_eq!(load_quant(route, Some(Quant::Q8), true).unwrap(), None);
            assert_eq!(
                load_quant(route, Some(Quant::Q8), false).unwrap(),
                Some(Quant::Q8)
            );
            assert_eq!(load_quant(route, None, false).unwrap(), None);
        }
        assert_eq!(load_quant(MODEL_ID_VACE, None, false).unwrap(), None);
        assert_eq!(
            load_quant(MODEL_ID_VACE, Some(Quant::Q8), false).unwrap(),
            Some(Quant::Q8)
        );
        assert_eq!(
            load_quant(MODEL_ID_VACE_FUN, None, false).unwrap(),
            Some(Quant::Q4)
        );
        assert_eq!(
            load_quant(MODEL_ID_VACE_FUN, Some(Quant::Q8), false).unwrap(),
            Some(Quant::Q8)
        );
        assert!(load_quant("wan_2_2", None, false).is_err());
    }

    #[test]
    fn lightning_is_on_by_default_only_for_the_a14b_routes() {
        assert!(lightning_default(MODEL_ID_T2V_14B));
        assert!(lightning_default(MODEL_ID_I2V_14B));
        for route in [MODEL_ID, MODEL_ID_VACE, MODEL_ID_VACE_FUN] {
            assert!(!lightning_default(route), "{route}");
        }
        assert_eq!((LIGHTNING_STEPS, LIGHTNING_GUIDANCE), (4, 1.0));
    }

    #[test]
    fn default_request_specs_match_the_product() {
        let q4 = tier(Some(4));
        let vace = vace_snapshot(12, 30);
        let loras = lightning_snapshot();
        for (route, weights, quant) in [
            (MODEL_ID, q4.path(), None),
            (MODEL_ID_T2V_14B, q4.path(), None),
            (MODEL_ID_I2V_14B, q4.path(), None),
            (MODEL_ID_VACE, vace.path(), None),
            (
                MODEL_ID_VACE_FUN,
                Path::new("/assembled/vace_fun"),
                Some(Quant::Q4),
            ),
        ] {
            let lightning = lightning_default(route);
            let snapshot = lightning.then(|| loras.path());
            let spec = product_load_spec(
                route,
                weights,
                OffloadPolicy::Sequential,
                lightning,
                snapshot.as_deref(),
            )
            .expect(route);
            assert!(matches!(&spec.weights, WeightsSource::Dir(dir) if dir == weights));
            assert_eq!(spec.quantize, quant, "{route}");
            assert_eq!(spec.precision, Precision::Bf16, "{route}");
            assert_eq!(spec.offload_policy, OffloadPolicy::Sequential, "{route}");
            assert!(spec.text_encoder.is_none(), "{route}");
            assert!(spec.components.is_empty(), "{route}");
            assert!(spec.resolved_route.is_none(), "{route}");
            if lightning {
                let subdir = loras.path().join(lightning_subdir(route).unwrap());
                let adapters: Vec<_> = spec
                    .adapters
                    .iter()
                    .map(|a| (a.path.clone(), a.scale, a.moe_expert))
                    .collect();
                assert_eq!(
                    adapters,
                    vec![
                        (
                            subdir.join("high_noise_model.safetensors"),
                            1.0,
                            Some(MoeExpert::High)
                        ),
                        (
                            subdir.join("low_noise_model.safetensors"),
                            1.0,
                            Some(MoeExpert::Low)
                        ),
                    ],
                    "{route}"
                );
            } else {
                assert!(spec.adapters.is_empty(), "{route}");
            }
        }
        // The Lightning-off A14B request is also a product request: no adapter.
        let off = product_load_spec(
            MODEL_ID_T2V_14B,
            q4.path(),
            OffloadPolicy::Sequential,
            false,
            None,
        )
        .unwrap();
        assert!(off.adapters.is_empty());
    }

    #[test]
    fn lightning_is_refused_where_the_product_does_not_bake_it() {
        let q4 = tier(Some(4));
        let loras = lightning_snapshot();
        let error = product_load_spec(
            MODEL_ID,
            q4.path(),
            OffloadPolicy::Sequential,
            true,
            Some(loras.path().as_path()),
        )
        .expect_err("the 5B has no Lightning distill");
        assert!(error.to_string().contains("only the A14B"), "{error}");
        assert!(product_load_spec(
            MODEL_ID_T2V_14B,
            q4.path(),
            OffloadPolicy::Sequential,
            true,
            None
        )
        .is_err());
    }

    #[test]
    fn a_tiered_route_refuses_a_non_default_tier() {
        for bits in [Some(8), None] {
            let root = tier(bits);
            let error = product_load_spec(
                MODEL_ID_T2V_14B,
                root.path(),
                OffloadPolicy::Sequential,
                false,
                None,
            )
            .expect_err("only q4 is the product default");
            assert!(error.to_string().contains("default tier is q4"), "{error}");
        }
    }

    #[test]
    fn wan_vace_refuses_a_transformer_the_product_does_not_assemble() {
        let fourteen_b = vace_snapshot(40, 40);
        let error = product_load_spec(
            MODEL_ID_VACE,
            fourteen_b.path(),
            OffloadPolicy::Resident,
            false,
            None,
        )
        .expect_err("a 14B VACE transformer is not the Mac product's wan_vace");
        assert!(error.to_string().contains("Wan2.1-VACE-1.3B"), "{error}");
    }
}
