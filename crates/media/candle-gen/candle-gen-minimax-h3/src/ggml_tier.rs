//! Offline conversion of a published MiniMax-H3 **MLX-affine** tier into a **Candle-native GGML**
//! tier — the artifact `SceneWorks/minimax-h3-candle` publishes, so a Windows/Linux q4/q8 install
//! loads its projections as-is instead of repacking ~130 GB of MLX triples on first load.
//!
//! The conversion itself is family-generic and lives in [`candle_gen::ggml_tier`] (SW-7285): MLX
//! triple -> one stored `{base}.weight` GGML block tensor (`Q4_1` lossless for q4, `Q8_0` for q8),
//! every other tensor copied verbatim, `*.index.json` / `config.json` rewritten, output written to
//! `<dst>.partial` and renamed. This module pins it to H3's component names and error prefix; the
//! output layout is unchanged.

use std::collections::HashMap;
use std::path::Path;

use candle_gen::candle_core::Tensor;
use candle_gen::ggml_tier::{GgmlTierError, SourceMode};
use candle_gen::quant::MLX_GROUP_SIZE;
use candle_gen::{CandleError, Result};

pub use candle_gen::ggml_tier::ComponentReport;

/// The three tiered component directories a Candle-native tier carries, matching the MLX rehost's
/// `{q4,q8}/{transformer,transformer_ref,text_encoder}` layout.
pub const COMPONENTS: [&str; 3] = ["transformer", "transformer_ref", "text_encoder"];

fn wrap(e: GgmlTierError) -> CandleError {
    CandleError::Msg(format!("minimax-h3 {e}"))
}

/// The rewritten `config.json` text for a GGML tier. Refuses a config with no packed marker.
pub fn ggml_config(text: &str) -> Result<String> {
    candle_gen::ggml_tier::ggml_config(text).map_err(wrap)
}

/// Convert one shard's tensors held in memory: each MLX triple (group 64) becomes stored GGML
/// blocks, everything else passes through unchanged. `report` is updated in place.
pub fn convert_tensors(
    tensors: HashMap<String, Tensor>,
    report: &mut ComponentReport,
) -> Result<HashMap<String, Tensor>> {
    candle_gen::ggml_tier::convert_tensors(tensors, MLX_GROUP_SIZE, report).map_err(wrap)
}

/// Convert the MLX-tier component directory `src` into a GGML-tier component at `dst` (from the
/// MLX triples; the group size comes from the component's `config.json`).
///
/// `dst` must not exist. `progress` is called after each shard with that shard's output path and
/// the running report.
pub fn convert_component(
    src: &Path,
    dst: &Path,
    progress: &mut dyn FnMut(&Path, &ComponentReport) -> Result<()>,
) -> Result<ComponentReport> {
    candle_gen::ggml_tier::convert_component(src, dst, &SourceMode::FromMlx, &mut |p, r| {
        progress(p, r).map_err(|e| GgmlTierError::Callback(e.to_string()))
    })
    .map_err(wrap)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::testkit::q4_pack;
    use candle_gen::candle_core::{DType, Device};
    use candle_gen::quant::repack_packed_weight_uncached;

    #[test]
    fn the_config_marker_becomes_ggml_storage_and_keeps_the_width() {
        let out =
            ggml_config(r#"{"hidden_size": 8, "quantization": {"bits": 4, "group_size": 64}}"#)
                .unwrap();
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(
            v["quantization"],
            serde_json::json!({"bits": 4, "storage": "ggml"})
        );
        assert_eq!(v["hidden_size"], 8);
        assert!(
            ggml_config(r#"{"hidden_size": 8}"#).is_err(),
            "a dense config must refuse"
        );
    }

    #[test]
    fn a_triple_becomes_the_exact_blocks_the_load_time_repack_builds() {
        let (wq, scales, biases, _grid) = q4_pack(8, 128, 3);
        let mut tensors = HashMap::new();
        tensors.insert("p.weight".to_string(), wq.clone());
        tensors.insert("p.scales".to_string(), scales.clone());
        tensors.insert("p.biases".to_string(), biases.clone());
        tensors.insert(
            "p.bias".to_string(),
            Tensor::zeros(8, DType::F32, &Device::Cpu).unwrap(),
        );
        let mut report = ComponentReport::default();
        let out = convert_tensors(tensors, &mut report).unwrap();
        assert_eq!(report.converted, 1);
        assert_eq!(report.passthrough, 1);
        assert!(!out.contains_key("p.scales") && !out.contains_key("p.biases"));
        let stored = &out["p.weight"];
        assert_eq!(stored.dtype(), DType::U8);
        assert_eq!(
            stored.dims(),
            [8, 4, 20],
            "Q4_1: 128 / 32 blocks of 20 B per row"
        );
        let expected =
            repack_packed_weight_uncached(&wq, &scales, &biases, MLX_GROUP_SIZE, &Device::Cpu)
                .unwrap();
        let stored_bytes = stored.flatten_all().unwrap().to_vec1::<u8>().unwrap();
        assert_eq!(stored_bytes, expected.data().unwrap().into_owned());
    }
}
