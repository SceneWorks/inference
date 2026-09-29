//! The SceneWorks Mac worker's load decisions for the two MLX FLUX.2 Klein edit routes.
//!
//! The worker's bespoke edit route (`image_jobs/flux2.rs::generate_flux2_edit_stream`) resolves the
//! Klein turnkey tier root, then builds `load_spec(weights, load_quant, adapters, None)` with
//! `load_quant = None` (`mlx_load_quant_for_resolved_artifact`: the tier is pre-packed and its dense
//! bf16 text encoder must not be re-quantized) and `with_resolved_route(<catalog model id>)` —
//! `flux2_klein_9b` for `flux2_klein_9b_edit` and `flux2_klein_9b_kv` for `flux2_klein_9b_kv_edit`.
//! A default request adds no adapter, PiD, component, decoder, or text-encoder selection. Residency
//! is the caller's (the worker's declared memory rungs choose it per request).
//!
//! [`product_load_spec`] is what a measurement entrypoint must call instead of assembling its own
//! `LoadSpec`.

use std::path::Path;

use mlx_gen::{Error, LoadSpec, OffloadPolicy, Precision, Result, WeightsSource};

/// The catalog model id the worker records as `resolved_route` for a Klein edit `route`.
pub fn product_resolved_route(route: &str) -> Result<&'static str> {
    match route {
        "flux2_klein_9b_edit" => Ok("flux2_klein_9b"),
        "flux2_klein_9b_kv_edit" => Ok("flux2_klein_9b_kv"),
        other => Err(Error::Msg(format!(
            "no SceneWorks MLX FLUX.2 product load decision for route {other}"
        ))),
    }
}

/// The `LoadSpec` the worker hands the Klein edit `route` loader for `weights`.
pub fn product_load_spec(
    route: &str,
    weights: &Path,
    offload_policy: OffloadPolicy,
) -> Result<LoadSpec> {
    let mut spec = LoadSpec::new(WeightsSource::Dir(weights.to_path_buf()))
        .with_resolved_route(product_resolved_route(route)?)
        .with_offload_policy(offload_policy);
    spec.quantize = None;
    spec.precision = Precision::Bf16;
    Ok(spec)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_edit_routes_load_with_the_worker_decisions() {
        let weights = Path::new("/snapshots/rev/q8");
        for (route, catalog) in [
            ("flux2_klein_9b_edit", "flux2_klein_9b"),
            ("flux2_klein_9b_kv_edit", "flux2_klein_9b_kv"),
        ] {
            let spec = product_load_spec(route, weights, OffloadPolicy::Sequential).expect(route);
            assert!(matches!(&spec.weights, WeightsSource::Dir(dir) if dir == weights));
            assert_eq!(spec.resolved_route.as_deref(), Some(catalog), "{route}");
            assert_eq!(spec.quantize, None, "{route}");
            assert_eq!(spec.precision, Precision::Bf16, "{route}");
            assert_eq!(spec.offload_policy, OffloadPolicy::Sequential, "{route}");
            assert!(spec.adapters.is_empty(), "{route}");
            assert!(spec.pid.is_none(), "{route}");
            assert!(spec.text_encoder.is_none(), "{route}");
            assert!(spec.components.is_empty(), "{route}");
        }
        assert!(product_load_spec("flux2_dev_edit", weights, OffloadPolicy::Resident).is_err());
    }
}
