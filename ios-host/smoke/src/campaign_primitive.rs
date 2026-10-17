//! Bounded synthetic PRIMITIVE characterization (sc-16780): one SANA-block quantum boundary on
//! the real device, without any model file. A synthetic f32 activation meets a resident F16 1x1
//! convolution weight shaped like a SANA GLUMBConv `conv_inverted`, so MLX materializes the
//! promoted F32 weight copy inside one guarded quantum. The row records what the device actually
//! charged for that quantum (kernel footprint, lifetime peak, OS headroom, MLX accounting).
//!
//! It is NOT a render, NOT admission evidence for any model, and NOT a cap. Its only consumer is
//! the engineering reserve: a measured per-quantum overhead may replace the unvalidated 512 MiB.
use super::Sampler;
use mlx_gen::ios_quantum::{self, SanaBlockGeometry, PROCESS_RESERVE_BYTES};
use serde_json::{json, Value};
use std::time::Instant;

#[cfg(test)]
#[path = "campaign_primitive_tests.rs"]
mod tests;

/// Reserved campaign key. Not a catalog case and never enumerated by `listing()`.
pub(super) const PROBE_KEY: &str = "primitive/sana_block_quantum";
pub(super) const EVIDENCE_CLASS: &str = "primitive_characterization";
const LABEL: &str = "SANA block quantum";
/// Image edge whose latent geometry the probe uses (256 px / 32 = 8x8 tokens).
const EDGE: u32 = 256;

/// Everything derived from code before any allocation: geometry, shapes and the requirement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Plan {
    pub geometry: SanaBlockGeometry,
    /// NHWC activation `[B, H, W, dim]`, f32.
    pub activation_shape: [i32; 4],
    /// OHWI 1x1 weight `[2*hidden, 1, 1, dim]`, F16.
    pub weight_shape: [i32; 4],
    pub scratch_bytes: u64,
    /// f32 activation + F16 weight, resident BEFORE the quantum's own live recheck.
    pub resident_bytes: u64,
    pub reserve_bytes: u64,
    /// scratch + resident + reserve: passing this guarantees the inner recheck also passes
    /// unless OS headroom moves for reasons outside the probe.
    pub required_bytes: u64,
}

pub(super) fn plan() -> Result<Plan, String> {
    let cfg = mlx_gen_sana::SanaTransformerConfig::sana_1600m();
    let side = (EDGE / mlx_gen_sana::pipeline::SPATIAL_SCALE) as i32;
    let caption = mlx_gen_sana::text_encoder::MAX_SEQUENCE_LENGTH;
    let geometry =
        mlx_gen_sana::transformer::configured_block_geometry(&cfg, 1, side, side, caption);
    let dim = i32::try_from(geometry.dim).map_err(|_| "probe dim overflow")?;
    let channels = i32::try_from(geometry.ff_channels).map_err(|_| "probe channel overflow")?;
    let scratch_bytes = ios_quantum::sana_block_bytes(geometry);
    let reserve_bytes = PROCESS_RESERVE_BYTES;
    let activation_shape = [1, side, side, dim];
    let weight_shape = [channels, 1, 1, dim];
    let elements = |shape: &[i32; 4]| shape.iter().map(|d| *d as u64).product::<u64>();
    let resident_bytes = elements(&activation_shape)
        .checked_mul(4)
        .and_then(|a| elements(&weight_shape).checked_mul(2)?.checked_add(a))
        .ok_or("probe resident overflow")?;
    let required_bytes = scratch_bytes
        .checked_add(resident_bytes)
        .and_then(|b| b.checked_add(reserve_bytes))
        .ok_or("probe requirement overflow")?;
    Ok(Plan {
        geometry,
        activation_shape,
        weight_shape,
        scratch_bytes,
        resident_bytes,
        reserve_bytes,
        required_bytes,
    })
}

/// One point-in-time reading. `None` is "unavailable", never zero evidence.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Reading {
    pub available_bytes: Option<u64>,
    pub footprint_bytes: Option<u64>,
    pub lifetime_peak_bytes: Option<u64>,
    pub mlx_active_bytes: u64,
    pub mlx_cache_bytes: u64,
    pub mlx_peak_bytes: u64,
}
impl Reading {
    fn json(&self) -> Value {
        json!({"available_bytes": self.available_bytes, "footprint_bytes": self.footprint_bytes,
            "kernel_lifetime_peak_bytes": self.lifetime_peak_bytes, "mlx_active_bytes": self.mlx_active_bytes,
            "mlx_cache_bytes": self.mlx_cache_bytes, "mlx_peak_bytes": self.mlx_peak_bytes})
    }
}

/// The device seam. Tests inject fixtures; the real device reads the OS and runs MLX.
pub(super) trait ProbeDevice {
    fn read(&mut self) -> Reading;
    fn reset_mlx_peak(&mut self);
    /// Materialize the resident F16 weight + activation, return a reading, then run exactly one
    /// guarded quantum and release everything. Only ever called after the headroom guard passed.
    fn allocate_and_run(&mut self, plan: &Plan) -> Result<Reading, String>;
}

/// Pure orchestration: guard (before ANY allocation) → residents → one quantum → readings.
pub(super) fn characterize(device: &mut impl ProbeDevice, diagnostic: bool) -> Value {
    let mut row = json!({"evidence_class": EVIDENCE_CLASS, "model_admission_evidence": false,
        "cap_validated": false, "production_success": false, "diagnostic": true,
        "description": "synthetic SANA-block quantum (f32 activation x resident F16 1x1 conv weight); primitive characterization only: not a model render, not admission evidence, not a cap",
        "allocation_attempted": false, "refused_before_allocation": false});
    let plan = match plan() {
        Ok(plan) => plan,
        Err(reason) => return failed(row, reason),
    };
    let g = plan.geometry;
    row["primitive"] = json!({"edge": EDGE, "activation_shape": plan.activation_shape, "activation_dtype": "float32",
        "weight_shape": plan.weight_shape, "weight_dtype": "float16",
        "geometry": {"batch": g.batch, "tokens": g.tokens, "dim": g.dim, "caption_tokens": g.caption_tokens,
            "ff_channels": g.ff_channels, "cross_heads": g.cross_heads, "conv_weight_elements": g.conv_weight_elements},
        "scratch_floor_bytes": plan.scratch_bytes, "resident_bytes": plan.resident_bytes, "engineering_reserve_bytes": plan.reserve_bytes,
        "required_headroom_bytes": plan.required_bytes});
    if !diagnostic {
        return failed(
            row,
            "primitive characterization must be launched as a diagnostic; refused before allocation".into(),
        );
    }
    let before = device.read();
    row["before"] = before.json();
    // Residents are allocated before the quantum's live recheck, so they are charged here too.
    if let Err(reason) = ios_quantum::check_headroom(
        before.available_bytes,
        plan.scratch_bytes + plan.resident_bytes,
        plan.reserve_bytes,
    ) {
        row["refused_before_allocation"] = json!(true);
        return failed(
            row,
            format!(
                "{reason}: requires {} bytes (SANA block floor {} + residents {} + engineering reserve {}), live {:?}",
                plan.required_bytes, plan.scratch_bytes, plan.resident_bytes, plan.reserve_bytes, before.available_bytes
            ),
        );
    }
    device.reset_mlx_peak();
    row["allocation_attempted"] = json!(true);
    let resident = match device.allocate_and_run(&plan) {
        Ok(resident) => resident,
        Err(reason) => {
            row["after"] = device.read().json();
            return failed(row, reason);
        }
    };
    let after = device.read();
    row["resident"] = resident.json();
    row["after"] = after.json();
    // Device charge for the quantum above the pre-probe footprint. Only defined when the quantum
    // set a NEW lifetime peak; otherwise an earlier launch excursion masks it (null, not zero).
    let over_baseline = match (
        before.footprint_bytes,
        before.lifetime_peak_bytes,
        after.lifetime_peak_bytes,
    ) {
        (Some(base), Some(old), Some(new)) if new > old => Some(new.saturating_sub(base)),
        _ => None,
    };
    row["quantum_peak_over_baseline_bytes"] = json!(over_baseline);
    row["unaccounted_by_mlx_peak_bytes"] =
        json!(over_baseline.map(|bytes| bytes as i128 - i128::from(after.mlx_peak_bytes)));
    row["status"] = json!("diagnostic_success");
    row["reason"] = json!(if over_baseline.is_some() {
        "primitive characterized"
    } else {
        "primitive ran; quantum did not exceed the prior lifetime peak, overhead unresolved"
    });
    row
}

fn failed(mut row: Value, reason: String) -> Value {
    row["status"] = json!("diagnostic_failure");
    row["reason"] = json!(reason);
    row
}

struct Device;
impl ProbeDevice for Device {
    fn read(&mut self) -> Reading {
        let (footprint_bytes, lifetime_peak_bytes) = match super::phys_footprint() {
            Some((current, peak)) => (Some(current), peak),
            None => (None, None),
        };
        Reading {
            available_bytes: ios_quantum::available_bytes(),
            footprint_bytes,
            lifetime_peak_bytes,
            mlx_active_bytes: mlx_rs::memory::get_active_memory() as u64,
            mlx_cache_bytes: mlx_rs::memory::get_cache_memory() as u64,
            mlx_peak_bytes: mlx_rs::memory::get_peak_memory() as u64,
        }
    }
    fn reset_mlx_peak(&mut self) {
        mlx_rs::memory::reset_peak_memory();
    }
    fn allocate_and_run(&mut self, plan: &Plan) -> Result<Reading, String> {
        run_quantum(plan, || self.read())
    }
}

/// The real primitive. Cache limit zero for its lifetime (same scope as the engine's quanta).
pub(super) fn run_quantum(
    plan: &Plan,
    mut read: impl FnMut() -> Reading,
) -> Result<Reading, String> {
    let _scope = ios_quantum::Scope::new();
    let activation = mlx_rs::ops::ones_dtype(&plan.activation_shape, mlx_rs::Dtype::Float32)
        .map_err(|e| e.to_string())?;
    let weight = mlx_rs::ops::ones_dtype(&plan.weight_shape, mlx_rs::Dtype::Float16)
        .map_err(|e| e.to_string())?;
    ios_quantum::finish("primitive residents", &[&activation, &weight])
        .map_err(|e| e.to_string())?;
    let resident = read();
    // The engine's own guard API: rechecks LIVE headroom, builds, evals, synchronizes, drains.
    let out = ios_quantum::run(LABEL, plan.scratch_bytes, || {
        Ok(mlx_rs::ops::conv2d(
            &activation,
            &weight,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )?)
    })
    .map_err(|e| e.to_string())?;
    let expected: Vec<i32> = [
        plan.activation_shape[0],
        plan.activation_shape[1],
        plan.activation_shape[2],
        plan.weight_shape[0],
    ]
    .into();
    if out.dtype() != mlx_rs::Dtype::Float32 || out.shape() != expected.as_slice() {
        return Err(format!(
            "primitive output {:?} {:?}; expected Float32 {expected:?} (F16 weight promotion)",
            out.dtype(),
            out.shape()
        ));
    }
    // ones x ones over `dim` input channels: every output is exactly `dim` in f32.
    let dim = plan.activation_shape[3] as f32;
    if out.as_slice::<f32>().iter().any(|v| *v != dim) {
        return Err("primitive output values are wrong".into());
    }
    Ok(resident)
}

/// Device entry. Host and simulator refuse before any MLX allocation.
pub(super) fn report(diagnostic: bool) -> Result<Value, String> {
    let started = Instant::now();
    if !cfg!(all(target_os = "ios", not(target_abi = "sim"))) {
        return Err("primitive characterization requires a real iOS device; host/simulator never allocate it".into());
    }
    runtime_ios::bound_mlx_to_platform_limits()
        .ok_or("platform allocator limits could not be bound; refused before allocation")?;
    if let Some(docs) = super::super::dirs_documents() {
        // No renders: an empty manifest so a prior case's PNG list is never pulled as this one's.
        std::fs::write(docs.join(super::super::RENDERS_MANIFEST), "").map_err(|e| e.to_string())?;
    }
    let sampler = Sampler::start();
    let mut row = characterize(&mut Device, diagnostic);
    let e = sampler.finish()?;
    row["schema"] = json!(1);
    row["case"] = json!(PROBE_KEY);
    row["elapsed_seconds"] = json!(started.elapsed().as_secs_f64());
    e.write(&mut row);
    row["mlx_peak_bytes"] = json!(mlx_rs::memory::get_peak_memory());
    row["device"] = json!(runtime_ios::device_machine());
    Ok(row)
}
