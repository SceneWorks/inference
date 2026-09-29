//! The SceneWorks Mac product's load decisions for the MLX FLUX.2 Klein routes.
//!
//! This module is the single source of those decisions: the SceneWorks worker's Klein routes call
//! [`loads_packed_tier_unquantized`] and [`resolved_route`], and the SC-20686 campaign entrypoint
//! builds its spec through [`product_load_spec`].
//!
//! Klein ships as pre-packed quant-matrix tiers (`q4/` is the default download; `q8/`, `bf16/`
//! are opt-in) whose transformer self-describes its quantization while the Qwen3 text encoder stays
//! dense bf16, so the load never re-quantizes (`LoadSpec::quantize` is `None`). The resolved route
//! is the catalog model the request named; each catalog model maps to exactly one provider route.

use std::path::Path;

use mlx_gen::{Error, LoadSpec, OffloadPolicy, Precision, Result, WeightsSource};

/// The bit width of the product's default packed Klein tier (`q4/`).
pub const PRODUCT_TIER_BITS: u64 = 4;

/// Whether `route` loads its resolved packed tier with no load-time quantization.
pub fn loads_packed_tier_unquantized(route: &str) -> bool {
    matches!(
        route,
        "flux2_klein_9b" | "flux2_klein_9b_edit" | "flux2_klein_9b_kv_edit"
    )
}

/// The `resolved_route` the product records for provider `route` serving `catalog_model`, refusing
/// a catalog model the route does not serve.
pub fn resolved_route<'a>(route: &str, catalog_model: &'a str) -> Result<&'a str> {
    let served = match route {
        "flux2_klein_9b_edit" => {
            matches!(catalog_model, "flux2_klein_9b" | "flux2_klein_9b_true_v2")
        }
        "flux2_klein_9b_kv_edit" => catalog_model == "flux2_klein_9b_kv",
        other => {
            return Err(Error::Msg(format!(
                "no SceneWorks MLX FLUX.2 product load decision for route {other}"
            )))
        }
    };
    if served {
        Ok(catalog_model)
    } else {
        Err(Error::Msg(format!(
            "{route} does not serve catalog model {catalog_model}"
        )))
    }
}

/// The catalog model of the product's default request on a Klein edit `route`.
pub fn default_catalog_model(route: &str) -> Result<&'static str> {
    match route {
        "flux2_klein_9b_edit" => Ok("flux2_klein_9b"),
        "flux2_klein_9b_kv_edit" => Ok("flux2_klein_9b_kv"),
        other => Err(Error::Msg(format!(
            "no SceneWorks MLX FLUX.2 product load decision for route {other}"
        ))),
    }
}

/// The `quantization.bits` a Klein tier's packed transformer declares (`None` = dense).
pub fn packed_tier_bits(root: &Path) -> Result<Option<u64>> {
    let path = root.join("transformer").join("config.json");
    let text = std::fs::read_to_string(&path)
        .map_err(|error| Error::Msg(format!("read {}: {error}", path.display())))?;
    let config: serde_json::Value = serde_json::from_str(&text)
        .map_err(|error| Error::Msg(format!("parse {}: {error}", path.display())))?;
    Ok(config
        .get("quantization")
        .and_then(|quantization| quantization.get("bits"))
        .and_then(serde_json::Value::as_u64))
}

/// The `LoadSpec` of the product's DEFAULT request on the Klein edit `route`: the `q4/` tier root
/// (any other tier is refused), no load-time quantization, and the default catalog model as the
/// resolved route. What a measurement entrypoint must load instead of assembling its own spec.
pub fn product_load_spec(
    route: &str,
    weights: &Path,
    offload_policy: OffloadPolicy,
) -> Result<LoadSpec> {
    let catalog = resolved_route(route, default_catalog_model(route)?)?;
    let bits = packed_tier_bits(weights)?;
    if bits != Some(PRODUCT_TIER_BITS) {
        return Err(Error::Msg(format!(
            "{route}: the product's default tier is q{PRODUCT_TIER_BITS}; {} is packed at {}",
            weights.display(),
            bits.map_or_else(|| "dense bf16".to_owned(), |bits| format!("{bits} bits"))
        )));
    }
    let mut spec = LoadSpec::new(WeightsSource::Dir(weights.to_path_buf()))
        .with_resolved_route(catalog)
        .with_offload_policy(offload_policy);
    spec.quantize = None;
    spec.precision = Precision::Bf16;
    Ok(spec)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tier(bits: Option<u64>) -> tempfile::TempDir {
        let root = tempfile::tempdir().expect("temp root");
        std::fs::create_dir_all(root.path().join("transformer")).expect("transformer");
        let config = match bits {
            Some(bits) => serde_json::json!({"quantization": {"bits": bits, "group_size": 64}}),
            None => serde_json::json!({}),
        };
        std::fs::write(
            root.path().join("transformer/config.json"),
            config.to_string(),
        )
        .expect("config");
        root
    }

    #[test]
    fn both_edit_routes_load_with_the_product_settings() {
        let q4 = tier(Some(4));
        for (route, catalog) in [
            ("flux2_klein_9b_edit", "flux2_klein_9b"),
            ("flux2_klein_9b_kv_edit", "flux2_klein_9b_kv"),
        ] {
            assert!(loads_packed_tier_unquantized(route));
            let spec = product_load_spec(route, q4.path(), OffloadPolicy::Sequential).expect(route);
            assert!(matches!(&spec.weights, WeightsSource::Dir(dir) if dir == q4.path()));
            assert_eq!(spec.resolved_route.as_deref(), Some(catalog), "{route}");
            assert_eq!(spec.quantize, None, "{route}");
            assert_eq!(spec.precision, Precision::Bf16, "{route}");
            assert_eq!(spec.offload_policy, OffloadPolicy::Sequential, "{route}");
            assert!(spec.adapters.is_empty(), "{route}");
            assert!(spec.pid.is_none(), "{route}");
            assert!(spec.text_encoder.is_none(), "{route}");
            assert!(spec.components.is_empty(), "{route}");
        }
        assert!(product_load_spec("flux2_dev_edit", q4.path(), OffloadPolicy::Resident).is_err());
    }

    #[test]
    fn a_non_default_tier_is_refused() {
        for bits in [Some(8), None] {
            let root = tier(bits);
            let error = product_load_spec(
                "flux2_klein_9b_edit",
                root.path(),
                OffloadPolicy::Sequential,
            )
            .expect_err("only q4 is the product default");
            assert!(error.to_string().contains("default tier is q4"), "{error}");
        }
    }

    #[test]
    fn resolved_routes_follow_the_catalog() {
        assert_eq!(
            resolved_route("flux2_klein_9b_edit", "flux2_klein_9b_true_v2").unwrap(),
            "flux2_klein_9b_true_v2"
        );
        assert!(resolved_route("flux2_klein_9b_edit", "flux2_klein_9b_kv").is_err());
        assert!(resolved_route("flux2_klein_9b_kv_edit", "flux2_klein_9b").is_err());
        assert!(!loads_packed_tier_unquantized("flux2_dev_edit"));
    }
}
