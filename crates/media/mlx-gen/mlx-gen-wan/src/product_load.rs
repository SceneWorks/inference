//! The SceneWorks Mac worker's load decisions for the five MLX Wan routes, as one constructor.
//!
//! The worker builds every MLX Wan `LoadSpec` in `video_jobs/wan.rs::video_load_spec` from a
//! `VideoGenInput` whose per-route fields are fixed by the route's handler. For a default request
//! (no `advanced.mlxQuantize`, no user LoRA) those handler decisions are:
//!
//! | Route | Weights the worker resolves | `LoadSpec::quantize` |
//! | --- | --- | --- |
//! | `wan2_2_ti2v_5b` / `wan2_2_t2v_14b` / `wan2_2_i2v_14b` | the pre-packed quant-matrix tier root (`q4/` first) | `None` — the packed tier's `config.json` is authoritative (`resolve_wan_tier_dir_and_quant`) |
//! | `wan_vace` | `<data>/models/mlx/wan_vace`: the dense **Wan2.1-VACE-1.3B** transformer + the base-Wan 14B UMT5/z16-VAE/tokenizer | `None` — dense bf16 (`resolve_wan_quant`) |
//! | `wan2_2_vace_fun_14b` | `<data>/models/mlx/wan_2_2_vace_fun`: both dense VACE-Fun 14B experts + the same shared components | `Some(Q4)` — forced (`resolve_wan_quant(request).or(Some(Quant::Q4))`) |
//!
//! Every route loads at `Precision::Bf16` with no adapters, text-encoder override, named
//! components, or `resolved_route` (the video spec never sets one). Residency is not decided here:
//! the worker's fit gate chooses it per host, so the caller passes the frozen campaign residency.
//!
//! [`product_load_spec`] is what a measurement entrypoint must call instead of assembling its own
//! `LoadSpec`, so a campaign loads each route in the product's memory shape (without the forced Q4,
//! a VACE-Fun load builds both 14B experts dense bf16 instead of the product's Q4).

use std::path::Path;

use mlx_gen::{Error, LoadSpec, OffloadPolicy, Precision, Quant, Result, WeightsSource};

use crate::config::{WanModelConfig, WanVaceConfig};
use crate::model::{MODEL_ID, MODEL_ID_I2V_14B, MODEL_ID_T2V_14B};
use crate::model_vace::{MODEL_ID_VACE, MODEL_ID_VACE_FUN};

/// The load-time quantization the worker applies to `route` on a default request.
pub fn product_load_quant(route: &str) -> Result<Option<Quant>> {
    match route {
        MODEL_ID | MODEL_ID_T2V_14B | MODEL_ID_I2V_14B | MODEL_ID_VACE => Ok(None),
        MODEL_ID_VACE_FUN => Ok(Some(Quant::Q4)),
        other => Err(Error::Msg(format!(
            "no SceneWorks MLX Wan product load decision for route {other}"
        ))),
    }
}

/// The `LoadSpec` the worker hands the `route` loader for `weights`, with the caller's residency.
///
/// `wan_vace` additionally requires the product's transformer: the Mac worker assembles the
/// Wan2.1-VACE-**1.3B** transformer, never the 14B one the Candle lane reads, so a snapshot of any
/// other size is refused rather than measured as if it were the product.
pub fn product_load_spec(
    route: &str,
    weights: &Path,
    offload_policy: OffloadPolicy,
) -> Result<LoadSpec> {
    let quantize = product_load_quant(route)?;
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
    let mut spec = LoadSpec::new(WeightsSource::Dir(weights.to_path_buf()))
        .with_offload_policy(offload_policy);
    spec.quantize = quantize;
    spec.precision = Precision::Bf16;
    Ok(spec)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROUTES: [&str; 5] = [
        MODEL_ID,
        MODEL_ID_T2V_14B,
        MODEL_ID_I2V_14B,
        MODEL_ID_VACE,
        MODEL_ID_VACE_FUN,
    ];

    /// The worker's default-request load quant per route (SceneWorks `video_jobs/{wan,vace}.rs`).
    fn worker_quant(route: &str) -> Option<Quant> {
        match route {
            "wan2_2_vace_fun_14b" => Some(Quant::Q4),
            _ => None,
        }
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

    #[test]
    fn every_route_loads_with_the_worker_decisions() {
        let vace = vace_snapshot(12, 30);
        for route in ROUTES {
            for policy in [OffloadPolicy::Resident, OffloadPolicy::Sequential] {
                let weights = if route == MODEL_ID_VACE {
                    vace.path().to_path_buf()
                } else {
                    std::path::PathBuf::from("/snapshots/rev/q4")
                };
                let spec = product_load_spec(route, &weights, policy).expect(route);
                assert!(matches!(&spec.weights, WeightsSource::Dir(dir) if *dir == weights));
                assert_eq!(spec.quantize, worker_quant(route), "{route}");
                assert_eq!(spec.precision, Precision::Bf16, "{route}");
                assert_eq!(spec.offload_policy, policy, "{route}");
                assert!(spec.adapters.is_empty(), "{route}");
                assert!(spec.text_encoder.is_none(), "{route}");
                assert!(spec.components.is_empty(), "{route}");
                assert!(spec.resolved_route.is_none(), "{route}");
                assert!(spec.control.is_none(), "{route}");
            }
        }
    }

    #[test]
    fn wan_vace_refuses_a_transformer_the_product_does_not_assemble() {
        let fourteen_b = vace_snapshot(40, 40);
        let error = product_load_spec(MODEL_ID_VACE, fourteen_b.path(), OffloadPolicy::Resident)
            .expect_err("a 14B VACE transformer is not the Mac product's wan_vace");
        assert!(error.to_string().contains("Wan2.1-VACE-1.3B"), "{error}");
    }

    #[test]
    fn an_unknown_route_has_no_product_decision() {
        assert!(product_load_spec("wan_2_2", Path::new("/x"), OffloadPolicy::Resident).is_err());
    }
}
