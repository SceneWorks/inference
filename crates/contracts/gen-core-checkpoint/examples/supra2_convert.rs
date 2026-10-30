//! Install-time Supra2-IMG conversion (sc-24213): verify the pinned upstream checkpoint and write
//! the native bundle. Optionally verifies the pinned component snapshots (sizes + SHA-256).
//!
//! ```sh
//! cargo run --release -p sceneworks-gen-core --example supra2_convert -- \
//!   <Supra2-IMG snapshot>/model_final_ema.pt <out bundle dir> [<flan-t5-base dir> <sd-vae-ft-mse dir>]
//! ```

use std::path::PathBuf;

use gen_core_checkpoint::supra2::{convert_pinned_checkpoint, FLAN_T5_BASE, SD_VAE};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    if args.len() != 2 && args.len() != 4 {
        eprintln!("usage: supra2_convert <model_final_ema.pt> <out_dir> [<flan-t5-base dir> <sd-vae-ft-mse dir>]");
        std::process::exit(2);
    }
    if args.len() == 4 {
        FLAN_T5_BASE.verify(&args[2])?;
        SD_VAE.verify(&args[3])?;
        eprintln!("verified pinned components");
    }
    let start = std::time::Instant::now();
    let manifest = convert_pinned_checkpoint(&args[0], &args[1])?;
    eprintln!(
        "converted {} ({} tensors, stored unconditional: {}) in {:.2}s",
        args[1].display(),
        manifest.denoiser_tensors,
        manifest.conditioning.is_some(),
        start.elapsed().as_secs_f32()
    );
    Ok(())
}
