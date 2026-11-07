//! Header/JSON-only contract for the ignored arithmetic witness. No tensors.
use crate::denoise::{JointGeometry, JointSchedule};
use crate::dit::config::MiniMaxH3DitConfig;
use candle_gen::{CandleError, Result};
use serde_json::Value;

pub(super) const FROZEN: &str = include_str!("residency_arithmetic_metadata.json");

pub(super) fn selected(name: &str) -> bool {
    name.starts_with("transformer_blocks.0.")
        || name.starts_with("time_embedder.")
        || name.starts_with("norm_out.")
}

pub(super) fn validate(
    config: &Value,
    selected: &Value,
    turbo_ffn: &Value,
    turbo_metadata: &Value,
) -> Result<()> {
    let frozen: Value =
        serde_json::from_str(FROZEN).map_err(|e| CandleError::Msg(e.to_string()))?;
    for (role, actual) in [
        ("config", config),
        ("selected", selected),
        ("turboFfn", turbo_ffn),
        ("turboMetadata", turbo_metadata),
    ] {
        if actual != &frozen[role] {
            return Err(CandleError::Msg(format!(
                "arithmetic frozen metadata mismatch: {role}"
            )));
        }
    }
    Ok(())
}

pub(super) fn geometry(cfg: &MiniMaxH3DitConfig) -> Result<JointGeometry> {
    // Use the actual production spatial->latent resolver (VAE16), not a
    // second guessed spatial scale. DiT then patches this48x84 latent grid by2.
    let geometry = crate::pipeline::resolve_geometry(1344, 768, 345)?.joint;
    validate_geometry(cfg, geometry)?;
    Ok(geometry)
}

fn validate_geometry(cfg: &MiniMaxH3DitConfig, geometry: JointGeometry) -> Result<()> {
    if cfg.patch_size != [1, 2, 2] {
        return Err(CandleError::Msg("arithmetic actual patch mismatch".into()));
    }
    let video = geometry.num_latent_frames
        * (geometry.latent_height / cfg.patch_size[1])
        * (geometry.latent_width / cfg.patch_size[2]);
    let audio = geometry.num_audio_latents * 2;
    if (
        geometry.latent_height,
        geometry.latent_width,
        geometry.num_latent_frames,
        geometry.num_audio_latents,
        video,
        audio,
        588 + video + audio,
    ) != (48, 84, 102, 575, 102816, 1150, 104554)
    {
        return Err(CandleError::Msg(
            "arithmetic actual packed geometry mismatch".into(),
        ));
    }
    Ok(())
}

pub(super) fn schedule() -> Result<JointSchedule> {
    // Match model::generate: effective evaluations + terminal sigma point;
    // accepted source recipe records video shift12/audio shift3, eight forwards.
    JointSchedule::with_shifts(8 + 1, 12.0, 3.0).map_err(|e| CandleError::Msg(e.to_string()))
}

#[test]
fn frozen_metadata_rejects_wrong_patch_missing_native_keys_and_wrong_schedule_count() {
    let frozen: Value = serde_json::from_str(FROZEN).unwrap();
    let cfg = MiniMaxH3DitConfig::from_diffusers_json(&frozen["config"].to_string()).unwrap();
    validate(
        &frozen["config"],
        &frozen["selected"],
        &frozen["turboFfn"],
        &frozen["turboMetadata"],
    )
    .unwrap();
    let actual_geometry = geometry(&cfg).unwrap();
    let mut wrong_height = actual_geometry;
    wrong_height.latent_height *= 2;
    assert!(validate_geometry(&cfg, wrong_height).is_err());
    let mut wrong_width = actual_geometry;
    wrong_width.latent_width *= 2;
    assert!(validate_geometry(&cfg, wrong_width).is_err());
    let mut wrong = frozen.clone();
    wrong["config"]["patch_size"] = serde_json::json!([1, 4, 4]);
    assert!(validate(
        &wrong["config"],
        &frozen["selected"],
        &frozen["turboFfn"],
        &frozen["turboMetadata"]
    )
    .is_err());
    let wrong_cfg = MiniMaxH3DitConfig::from_diffusers_json(&wrong["config"].to_string()).unwrap();
    assert!(geometry(&wrong_cfg).is_err());
    for key in frozen["selected"].as_object().unwrap().keys() {
        let mut missing = frozen["selected"].clone();
        missing.as_object_mut().unwrap().remove(key);
        assert!(validate(
            &frozen["config"],
            &missing,
            &frozen["turboFfn"],
            &frozen["turboMetadata"]
        )
        .is_err());
        let mut dtype = frozen["selected"].clone();
        dtype[key]["dtype"] = serde_json::json!("F16");
        assert!(validate(
            &frozen["config"],
            &dtype,
            &frozen["turboFfn"],
            &frozen["turboMetadata"]
        )
        .is_err());
        let mut shape = frozen["selected"].clone();
        shape[key]["shape"][0] = serde_json::json!(1);
        assert!(validate(
            &frozen["config"],
            &shape,
            &frozen["turboFfn"],
            &frozen["turboMetadata"]
        )
        .is_err());
    }
    for key in frozen["turboFfn"].as_object().unwrap().keys() {
        let mut missing = frozen["turboFfn"].clone();
        missing.as_object_mut().unwrap().remove(key);
        assert!(validate(
            &frozen["config"],
            &frozen["selected"],
            &missing,
            &frozen["turboMetadata"]
        )
        .is_err());
    }
    for key in ["alpha", "format", "floating_dtype"] {
        let mut wrong = frozen["turboMetadata"].clone();
        wrong[key] = serde_json::json!("invalid");
        assert!(validate(
            &frozen["config"],
            &frozen["selected"],
            &frozen["turboFfn"],
            &wrong
        )
        .is_err());
    }
    let actual = schedule().unwrap();
    assert_eq!(
        (
            actual.num_evals(),
            actual.video().shift(),
            actual.audio().shift()
        ),
        (8, 12.0, 3.0)
    );
    let adaln = crate::denoise::adaln_schedule(&actual).unwrap();
    assert_eq!((adaln.num_steps(), adaln.num_row_classes()), (8, 4));
    assert_eq!(adaln.num_distinct_timesteps(), 17);
    // The original fixture's guessed constructor would have silently used seven
    // forwards, then failed its eight-row assertion after preparing the model.
    assert_eq!(JointSchedule::new(8).unwrap().num_evals(), 7);
}
