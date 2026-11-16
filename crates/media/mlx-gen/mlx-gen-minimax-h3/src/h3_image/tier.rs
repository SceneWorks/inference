//! MiniMax-H3-Image **tier resolution** on MLX (SW-7093): which directory the DiT and the text
//! encoder are read from, and whether the staged DiT really is the tier the caller asserted.
//!
//! Layout and names are [`gen_core_h3::h3_image_tier`]; this module is the load-time half. The
//! packed DiT is the staged [`IMAGE_DIT_COMPONENT`], and the text encoder is the staged
//! `text_encoder` component -- MiniMax-H3's own tier artifact, whose group size is checked by H3's
//! `reconcile_text_encoder`. Neither is derived from `spec.quantize`, which only asserts which tier
//! was staged: the `quantization` marker in `config.json` is authoritative ([`reconcile`]).

use std::path::{Path, PathBuf};

use mlx_gen::gen_core::{LoadSpec, WeightsSource};
use mlx_gen::{Error, Result};

use crate::h3_image::dit::IMAGE_DIT_COMPONENT;
use crate::h3_image::model::IMAGE_MODEL_ID;
use crate::model::reconcile_text_encoder;

/// The DiT directory a load reads: the staged [`IMAGE_DIT_COMPONENT`] (a tier's
/// `transformer_diffusers/`), else `root/transformer_diffusers` (the upstream bf16 layout).
pub fn resolve_dit_dir(root: &Path, spec: &LoadSpec) -> PathBuf {
    match spec.components.get(IMAGE_DIT_COMPONENT) {
        Some(WeightsSource::Dir(p)) => p.clone(),
        _ => root.join(IMAGE_DIT_COMPONENT),
    }
}

/// Check the staged tier against the request: the DiT's `quantization` marker must agree with
/// `spec.quantize` (a dense DiT under a packed request, or the wrong width, is refused), and a packed
/// text encoder must declare the group size this engine reads.
pub fn reconcile(dit_dir: &Path, text_encoder_dir: &Path, spec: &LoadSpec) -> Result<()> {
    let packed = mlx_gen::quant::packed_quant_bits_at(dit_dir)?;
    match (packed, spec.quantize) {
        (Some(bits), Some(q)) if bits != q.bits() => {
            return Err(Error::Msg(format!(
                "{IMAGE_MODEL_ID}: spec.quantize={q:?} (bits {}) disagrees with the staged tier's \
                 config.json quantization marker (bits {bits}) in {} -- the on-disk tier is \
                 authoritative; stage the tier you asked for",
                q.bits(),
                dit_dir.display()
            )))
        }
        (None, Some(q)) => {
            return Err(Error::Unsupported(format!(
                "{IMAGE_MODEL_ID}: spec.quantize={q:?} but {} carries no quantization marker -- \
                 this model never quantizes at load; stage the packed tier's \
                 `{IMAGE_DIT_COMPONENT}` directory as the '{IMAGE_DIT_COMPONENT}' component",
                dit_dir.display()
            )))
        }
        _ => {}
    }
    reconcile_text_encoder(text_encoder_dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::Quant;

    fn marked(dir: &Path, bits: Option<i32>) {
        std::fs::create_dir_all(dir).unwrap();
        let cfg = match bits {
            Some(b) => format!(r#"{{"quantization": {{"bits": {b}, "group_size": 64}}}}"#),
            None => "{}".to_owned(),
        };
        std::fs::write(dir.join("config.json"), cfg).unwrap();
    }

    #[test]
    fn the_staged_component_wins_over_the_snapshot_root() {
        let spec = LoadSpec::new(WeightsSource::Dir("/snap".into()));
        assert_eq!(
            resolve_dit_dir(Path::new("/snap"), &spec),
            Path::new("/snap/transformer_diffusers")
        );
        let mut staged = spec.clone();
        staged.components.insert(
            IMAGE_DIT_COMPONENT.into(),
            WeightsSource::Dir("/tiers/q4/transformer_diffusers".into()),
        );
        assert_eq!(
            resolve_dit_dir(Path::new("/snap"), &staged),
            Path::new("/tiers/q4/transformer_diffusers")
        );
    }

    #[test]
    fn a_tier_must_match_the_marker_on_disk() {
        let tmp = tempfile::tempdir().unwrap();
        let te = tmp.path().join("te");
        marked(&te, Some(4));
        for (marker, asked, ok) in [
            (Some(4), Some(Quant::Q4), true),
            (Some(8), Some(Quant::Q8), true),
            (Some(8), None, true),
            (None, None, true),
            (Some(4), Some(Quant::Q8), false),
            (None, Some(Quant::Q4), false),
        ] {
            let dit = tmp.path().join(format!("d{marker:?}{asked:?}"));
            marked(&dit, marker);
            let mut spec = LoadSpec::new(WeightsSource::Dir(tmp.path().into()));
            spec.quantize = asked;
            assert_eq!(
                reconcile(&dit, &te, &spec).is_ok(),
                ok,
                "{marker:?} {asked:?}"
            );
        }
    }

    #[test]
    fn a_text_encoder_packed_at_another_group_size_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let dit = tmp.path().join("dit");
        marked(&dit, None);
        let te = tmp.path().join("te");
        std::fs::create_dir_all(&te).unwrap();
        std::fs::write(
            te.join("config.json"),
            r#"{"quantization": {"bits": 4, "group_size": 32}}"#,
        )
        .unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(tmp.path().into()));
        assert!(reconcile(&dit, &te, &spec).is_err());
    }
}
