//! CPU-only verification of the frozen native terminal asset inventory; never creates a device.
use candle_gen::gen_core::PinnedWeightsFile;
use gen_core_moss_image::moss_image::*;
use std::path::PathBuf;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    for asset in MOSS_IMAGE_ASSETS {
        let variable = match asset.repository {
            BASE_REPO => "MOSS_BASE_SNAPSHOT",
            SFT_REPO => "MOSS_SHARED_SNAPSHOT",
            DMD_REPO => "MOSS_DMD_SNAPSHOT",
            _ => return Err("unknown frozen repository".into()),
        };
        let root = PathBuf::from(std::env::var_os(variable).ok_or(variable)?);
        if !root.is_absolute() {
            return Err(format!("{variable} must be absolute").into());
        }
        let pin = PinnedWeightsFile::pin(root.join(asset.relative_path))?;
        verify_moss_image_asset(&pin, asset.repository, asset.revision, asset.relative_path)?;
        println!(
            "{}",
            serde_json::json!({"repository":asset.repository,"revision":asset.revision,
            "relative_path":asset.relative_path,"path":pin.loader_path(),"bytes":asset.bytes,"sha256":asset.sha256})
        );
    }
    Ok(())
}
