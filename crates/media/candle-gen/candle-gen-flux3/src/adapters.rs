//! Native additive LoRA installation after tensor-neutral identity/key/rank admission.
use crate::dit::Flux3Dit;
use candle_gen::{
    candle_core::DType,
    gen_core::{
        flux3::{adapters, DitConfig},
        AdapterSpec,
    },
    CandleError as Error, Result,
};

pub fn validate(specs: &[AdapterSpec], config: &DitConfig) -> Result<()> {
    for spec in specs {
        adapters::inspect(spec, config).map_err(|e| Error::Unsupported(e.to_string()))?;
    }
    Ok(())
}

pub fn install(dit: &mut Flux3Dit, specs: &[AdapterSpec]) -> Result<()> {
    install_with_reader(dit, specs, |path| std::fs::read(path))
}

/// `install` with the file read injected, so the single-read property is directly testable.
fn install_with_reader(
    dit: &mut Flux3Dit,
    specs: &[AdapterSpec],
    mut read: impl FnMut(&std::path::Path) -> std::io::Result<Vec<u8>>,
) -> Result<()> {
    // Read each file ONCE; validate and load tensors from those same bytes (no TOCTOU), and
    // validate the entire stack before changing even one projection.
    let loaded = specs
        .iter()
        .map(|spec| {
            let bytes = read(&spec.path).map_err(|e| {
                Error::Unsupported(format!("flux3 LoRA {}: {e}", spec.path.display()))
            })?;
            let plan = adapters::inspect_bytes(spec, dit.config(), &bytes)
                .map_err(|e| Error::Unsupported(e.to_string()))?;
            Ok((bytes, plan))
        })
        .collect::<Result<Vec<_>>>()?;
    for (spec, (bytes, plan)) in specs.iter().zip(loaded) {
        let tensors = candle_gen::candle_core::safetensors::load_buffer(&bytes, dit.device())
            .map_err(|e| Error::Unsupported(format!("flux3 LoRA {}: {e}", spec.path.display())))?;
        for projection in plan {
            let take = |key: &str| {
                tensors
                    .get(key)
                    .ok_or_else(|| Error::Unsupported(format!("flux3 LoRA: missing tensor {key}")))?
                    .to_dtype(DType::F32)
                    .map_err(Error::from)
            };
            let down = take(&projection.down)?;
            let up = take(&projection.up)?;
            dit.install_lora(&projection.target, down, up, projection.scale)
                .map_err(|e| {
                    Error::Unsupported(format!(
                        "flux3 LoRA {} target {}: {e}",
                        spec.path.display(),
                        projection.target
                    ))
                })?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;
    use candle_gen::candle_nn::VarBuilder;
    use candle_gen::gen_core::{flux3::Flux3Config, AdapterKind};

    fn fixtures() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../mlx-gen/mlx-gen-flux3/tests/fixtures")
    }

    fn base_dit() -> Flux3Dit {
        let device = Device::Cpu;
        let config = Flux3Config::from_dir(&fixtures().join("tiny-base-snapshot")).unwrap();
        let weights = candle_gen::candle_core::safetensors::load(
            fixtures().join("tiny-base-snapshot/model.safetensors"),
            &device,
        )
        .unwrap();
        let vb = VarBuilder::from_tensors(weights, DType::F32, &device);
        Flux3Dit::from_var_builder(&vb, &config.dit, DType::F32, &device).unwrap()
    }

    /// The bytes validated are the bytes loaded: each adapter file is read exactly once, so a file
    /// replaced after validation cannot be installed. *Mutation that reds this:* loading tensors
    /// with a second read (e.g. `safetensors::load(&spec.path)` after `inspect`).
    #[test]
    fn each_adapter_file_is_read_once_for_both_validation_and_tensors() {
        let valid = std::fs::read(fixtures().join("flux3_base_lora.safetensors")).unwrap();
        let mut swapped = valid.clone();
        // A later read would see a file whose header no longer parses.
        swapped[8..16].copy_from_slice(b"garbage!");
        let spec = AdapterSpec::new(
            fixtures().join("flux3_base_lora.safetensors"),
            0.5,
            AdapterKind::Lora,
        );
        let mut dit = base_dit();
        let mut reads = 0usize;
        install_with_reader(&mut dit, &[spec], |_| {
            reads += 1;
            Ok(if reads == 1 {
                valid.clone()
            } else {
                swapped.clone()
            })
        })
        .unwrap();
        assert_eq!(
            reads, 1,
            "validation and tensor loading must share one read"
        );
        // And a file that fails validation installs nothing.
        let spec = AdapterSpec::new(
            fixtures().join("flux3_base_lora.safetensors"),
            0.5,
            AdapterKind::Lora,
        );
        let mut dit = base_dit();
        assert!(matches!(
            install_with_reader(&mut dit, &[spec], |_| Ok(swapped.clone())),
            Err(Error::Unsupported(_))
        ));
    }
}
