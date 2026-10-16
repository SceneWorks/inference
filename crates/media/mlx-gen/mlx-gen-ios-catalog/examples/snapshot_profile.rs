//! Header-only SANA artifact inspection; never constructs a model or reads tensor payloads.
//! cargo run -p mlx-gen-ios-catalog --example snapshot_profile -- /path/to/snapshot

use mlx_gen::gen_core::WeightsSource;
use mlx_gen_ios_catalog::{sana_load_spec, sana_snapshot_profile};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::env::args_os()
        .nth(1)
        .ok_or("usage: snapshot_profile SNAPSHOT_DIR")?;
    let profile = sana_snapshot_profile(&sana_load_spec(WeightsSource::Dir(root.into())))?;
    let hex: String = profile
        .layout_sha256
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    println!("layout_sha256={hex}");
    println!(
        "component_payload_bytes (transformer, text_encoder, vae)={:?}",
        profile.component_bytes
    );
    println!(
        "embedding={:?} {:?}",
        profile.embedding_dtype, profile.embedding_shape
    );
    let payload = profile.component_bytes[1]
        .max(profile.component_bytes[0].saturating_add(profile.component_bytes[2]));
    println!(
        "Sequential payload allowance={} MiB; payload only, NOT a safe working-set/render budget",
        payload.div_ceil(1024 * 1024)
    );
    Ok(())
}
