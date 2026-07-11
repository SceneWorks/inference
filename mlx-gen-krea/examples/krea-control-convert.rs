//! Offline convert-once: candle Krea pose overlay → MLX-native pose overlay (sc-8465, epic 8459 S5).
//!
//! The candle overlay (`SceneWorks/krea2-pose-controlnet-beta/control_step5000.safetensors`) stores its
//! four RMSNorm scales pre-folded as `scale + 1` (`*.weight_p1`); MLX's `RmsScale` re-folds `+1` at
//! load, so this un-folds them back to the raw `*.weight` and copies everything else verbatim
//! (see [`mlx_gen_krea::control::convert_candle_overlay`]). Run once; re-host the output on SceneWorks HF
//! and reference it from the worker control-overlay manifest.
//!
//! ```text
//! cargo run -p mlx-gen-krea --example krea-control-convert -- \
//!     control_step5000.safetensors control_step5000.mlx.safetensors
//! ```

use mlx_gen_krea::control::convert_candle_overlay_file;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!(
            "usage: {} <candle_overlay.safetensors> <out_mlx_overlay.safetensors>",
            args.first()
                .map(String::as_str)
                .unwrap_or("krea-control-convert")
        );
        std::process::exit(2);
    }
    match convert_candle_overlay_file(&args[1], &args[2]) {
        Ok(()) => println!("wrote MLX pose overlay → {}", args[2]),
        Err(e) => {
            eprintln!("convert failed: {e}");
            std::process::exit(1);
        }
    }
}
