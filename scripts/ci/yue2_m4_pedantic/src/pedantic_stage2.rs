//! Diagnostic-only same-operand M4 stage-2 native ConvTranspose at cuBLAS modes 16 and 18.
//! Mode 18 is the documented PEDANTIC (2) | DISALLOW_REDUCED_REDUCTION (16) bitmask.

use super::*;
use crate::pedantic_io::{
    compare_native_contributors, native_column_record, raw_values, tensor_record, weight_record,
};
use candle_audio::candle_core::cuda::cudarc::cublas::{self, sys};
use std::sync::Arc;

const MODE16: u32 = 16;
const MODE18: u32 = 18;
const ORIGINAL_BOUND: f32 = 1.0 / 64.0;

struct ModeGuard {
    device: Device,
    blas: Arc<cublas::CudaBlas>,
    active: bool,
    events: std::fs::File,
    handle: String,
    stream: String,
    thread: String,
}

impl ModeGuard {
    fn new(vae: &Yue2Vae, shared: &Device, out: &Path) -> Result<Self, Box<dyn Error>> {
        let private = vae.device().as_cuda_device()?;
        let shared_blas = shared.as_cuda_device()?.cublas_handle();
        let blas = private.cublas_handle();
        if vae.dtype() != DType::BF16
            || vae.device().same_device(shared)
            || Arc::ptr_eq(&shared_blas, &blas)
        {
            return Err("diagnostic requires the distinct M4 private BF16 VAE handle".into());
        }
        let stream = private.cuda_stream();
        let mut guard = Self {
            device: vae.device().clone(),
            handle: format!("{:p}", *blas.handle()),
            stream: format!("{:p}", Arc::as_ptr(&stream)),
            thread: format!("{:?}", std::thread::current().id()),
            blas,
            active: false,
            events: OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(out.join("math-events.jsonl"))?,
        };
        guard.sync("initial_sync")?;
        guard.expect_mode("initial_get", MODE16)?;
        if guard.shared_mode(shared)? != 0 {
            return Err("shared MoT/audio handle was not default mode 0".into());
        }
        Ok(guard)
    }

    fn event(
        &mut self,
        action: &str,
        status: &str,
        mode: Option<u32>,
    ) -> Result<(), Box<dyn Error>> {
        writeln!(
            self.events,
            "{}",
            json!({"action":action,"status":status,"rawMode":mode,
            "handleAddress":self.handle,"streamObjectAddress":self.stream,
            "thread":self.thread,"deviceOrdinal":0,
            "bf16ReducedPrecisionAtomic":candle_core::cuda::gemm_reduced_precision_bf16()})
        )?;
        self.events.flush()?;
        Ok(())
    }

    fn bind(&self) -> Result<(), Box<dyn Error>> {
        self.device
            .as_cuda_device()?
            .cuda_stream()
            .context()
            .bind_to_thread()?;
        Ok(())
    }

    fn sync(&mut self, action: &str) -> Result<(), Box<dyn Error>> {
        let status = self.device.synchronize();
        self.event(
            action,
            if status.is_ok() { "success" } else { "failure" },
            None,
        )?;
        status?;
        Ok(())
    }

    fn raw_mode(&self) -> Result<u32, Box<dyn Error>> {
        self.bind()?;
        let mut bits = u32::MAX;
        let status =
            unsafe { sys::cublasGetMathMode(*self.blas.handle(), (&mut bits as *mut u32).cast()) };
        status.result()?;
        Ok(bits)
    }

    fn expect_mode(&mut self, action: &str, expected: u32) -> Result<(), Box<dyn Error>> {
        let observed = self.raw_mode()?;
        self.event(
            action,
            if observed == expected {
                "success"
            } else {
                "mismatch"
            },
            Some(observed),
        )?;
        if observed != expected {
            return Err(
                format!("{action}: private cuBLAS mode {observed}, expected {expected}").into(),
            );
        }
        Ok(())
    }

    fn shared_mode(&self, shared: &Device) -> Result<u32, Box<dyn Error>> {
        let cuda = shared.as_cuda_device()?;
        cuda.cuda_stream().context().bind_to_thread()?;
        let blas = cuda.cublas_handle();
        let mut bits = u32::MAX;
        let status =
            unsafe { sys::cublasGetMathMode(*blas.handle(), (&mut bits as *mut u32).cast()) };
        status.result()?;
        Ok(bits)
    }

    // cudarc's Rust cublasMath_t enum has values 0,1,2,3,16 but not the valid C bitmask 18.
    // Call the same cublasSetMathMode symbol with its C ABI integer parameter; never construct
    // or transmute an invalid Rust enum discriminant. The DLL is the same CUDA 12 cuBLAS used by
    // cudarc's dynamically loaded get/set wrapper and the handle belongs to this private VAE.
    fn set_bits(&mut self, action: &str, bits: u32) -> Result<(), Box<dyn Error>> {
        if bits != MODE16 && bits != MODE18 {
            return Err("diagnostic permits only documented mode 16 or 18".into());
        }
        self.bind()?;
        #[cfg(target_os = "windows")]
        let library = "cublas64_12.dll";
        #[cfg(not(target_os = "windows"))]
        let library = "libcublas.so.12";
        let status = unsafe {
            let lib = libloading::Library::new(library)?;
            let function: libloading::Symbol<
                unsafe extern "C" fn(sys::cublasHandle_t, u32) -> sys::cublasStatus_t,
            > = lib.get(b"cublasSetMathMode\0")?;
            function(*self.blas.handle(), bits)
        };
        self.event(action, &format!("{status:?}"), Some(bits))?;
        status.result()?;
        Ok(())
    }

    fn enable_18(&mut self) -> Result<(), Box<dyn Error>> {
        self.sync("before18_sync")?;
        self.expect_mode("before18_get", MODE16)?;
        // A failing setter may have changed the handle, so restoration is armed first.
        self.active = true;
        self.set_bits("set18", MODE18)?;
        self.expect_mode("read18", MODE18)
    }

    fn restore_16(&mut self) -> Result<(), Box<dyn Error>> {
        if !self.active {
            return Ok(());
        }
        self.sync("before_restore_sync")?;
        self.set_bits("restore16", MODE16)?;
        self.sync("after_restore_sync")?;
        self.expect_mode("read_restored16", MODE16)?;
        self.active = false;
        Ok(())
    }
}

impl Drop for ModeGuard {
    fn drop(&mut self) {
        if self.active && self.restore_16().is_err() {
            eprintln!("M4 diagnostic private cuBLAS restoration failed; aborting owned child");
            std::process::abort();
        }
    }
}

fn compare_records(out: &Path, a: &Value, b: &Value) -> Result<Value, Box<dyn Error>> {
    if a["shape"] != b["shape"] || a["dtype"] != "BF16" || b["dtype"] != "BF16" {
        return Err("stage-2 comparison shape/dtype changed".into());
    }
    let (x, y) = (raw_values(out, a)?, raw_values(out, b)?);
    if x.len() != y.len() {
        return Err("stage-2 comparison length changed".into());
    }
    let mut bits = 0usize;
    let mut positive = 0usize;
    let mut max_abs = 0f32;
    for (a, b) in x.iter().zip(y.iter()) {
        if a.0 != b.0 {
            bits += 1;
        }
        let difference = (a.1 - b.1).abs();
        if difference > 0.0 {
            positive += 1;
            max_abs = max_abs.max(difference);
        }
    }
    Ok(json!({"bitDifferences":bits,"positiveDifferences":positive,"maxAbs":max_abs}))
}

fn contributor_input_alignment(
    out: &Path,
    full: &Value,
    tile: &Value,
) -> Result<Value, Box<dyn Error>> {
    if full["shape"] != json!([1, 2048, 75]) || tile["shape"] != json!([1, 2048, 32]) {
        return Err("stage-2 full/tile source input geometry changed".into());
    }
    let full = raw_values(out, full)?;
    let tile = raw_values(out, tile)?;
    let mut boundary_differences = 0usize;
    for channel in 0..2048 {
        for row in 0..32 {
            if full[channel * 75 + row].0 != tile[channel * 32 + row].0 {
                if row < 29 {
                    return Err(format!(
                        "stage-2 contributing input changed at channel {channel}, row {row}"
                    )
                    .into());
                }
                boundary_differences += 1;
            }
        }
    }
    Ok(
        json!({"bitIdenticalPrefixRows":[0,29],"comparedElements":2048*29,
              "boundaryRowsNotComparedAsEqual":[29,32],
              "boundaryBitDifferences":boundary_differences}),
    )
}

fn arm(
    vae: &Yue2Vae,
    out: &Path,
    label: &str,
    mode: u32,
    inputs: &[Tensor; 2],
    input_records: &[Value; 2],
    weight_sha: &str,
    guard: &ModeGuard,
) -> Result<Value, Box<dyn Error>> {
    let mut rows = serde_json::Map::new();
    for (index, slot) in ["full", "tile0"].iter().enumerate() {
        let input = &inputs[index];
        let current = tensor_record(input, out, &format!("{label}-{slot}-input"), 0)?;
        if current["sha256"] != input_records[index]["sha256"]
            || current["shape"] != input_records[index]["shape"]
        {
            return Err("mode arm changed the exact stage-2 BF16 input bits".into());
        }
        let expected_len = if index == 0 { 75 } else { 32 };
        let (output, capture) = candle_core::cuda::with_yue2_native_convt_column(
            DType::BF16,
            [1, expected_len, 1024, 12],
            || {
                vae.diagnostic_stage2_forward(input)
                    .map_err(pedantic_io::io_candle)
            },
        )?;
        let column =
            native_column_record(capture, out, label, slot, expected_len, weight_sha, true)?;
        if column["mathModeReadback"] != mode
            || column["effectiveComputeType"] != "CUBLAS_COMPUTE_32F"
            || column["bf16ReducedPrecisionAtomic"] != false
            || column["handleAddress"] != guard.handle
            || column["streamObjectAddress"] != guard.stream
        {
            return Err("native stage-2 GEMM mode/compute/handle changed".into());
        }
        let output = tensor_record(&output, out, &format!("{label}-{slot}-output"), 0)?;
        rows.insert(
            (*slot).into(),
            json!({"input":current,"nativeColumn":column,
                                          "output":output}),
        );
    }
    Ok(Value::Object(rows))
}

pub(super) fn run(
    out: &Path,
    source: &AcousticLatents,
    verified: &candle_audio_yue2::snapshot::VerifiedComponent,
    shared: &Device,
    meta: &Value,
    reference_hash: &str,
) -> Result<(), Box<dyn Error>> {
    if source.frames() != 75
        || meta["long_core_frames"].as_u64() != Some(16)
        || candle_core::cuda::gemm_reduced_precision_bf16()
    {
        return Err("exact M4 75-frame/core16 BF16 diagnostic prerequisites changed".into());
    }
    let standard = &meta["decoders"]["standard"];
    let component = verified.component();
    if component.repo.id != standard["repo"]
        || component.repo.revision != standard["revision"]
        || component.weights().map(|file| file.sha256) != standard["weights_sha256"].as_str()
        || component.file("config.json").map(|file| file.sha256)
            != standard["config_sha256"].as_str()
    {
        return Err("verified standard VAE component differs from authenticated teacher".into());
    }
    let vae = Yue2Vae::load_with_dtype(verified, VaeParts::Full, shared, DType::BF16)?;
    if vae.dtype() != DType::BF16 || vae.required_halo(16) > 16 {
        return Err("M4 BF16 VAE dtype or halo changed".into());
    }
    let mut guard = ModeGuard::new(&vae, shared, out)?;
    let z = source
        .to_decoder_input(vae.device())?
        .to_dtype(DType::BF16)?;
    let latent = tensor_record(&z, out, "m4-bf16-latent75", 0)?;
    let inputs = [
        vae.diagnostic_stage2_input(&z, 0, 75)?,
        vae.diagnostic_stage2_input(&z, 0, 32)?,
    ];
    let input_records = [
        tensor_record(&inputs[0], out, "source-full-input", 0)?,
        tensor_record(&inputs[1], out, "source-tile0-input", 0)?,
    ];
    let input_alignment = contributor_input_alignment(out, &input_records[0], &input_records[1])?;
    let weight = weight_record(&vae, out, "before16", DType::BF16)?;
    if weight["shape"] != json!([2048, 1024, 12]) {
        return Err("standard stage-2 folded weight geometry changed".into());
    }
    let weight_sha = weight["sha256"]
        .as_str()
        .ok_or("folded weight SHA absent")?;

    let mode16 = arm(
        &vae,
        out,
        "mode16",
        MODE16,
        &inputs,
        &input_records,
        weight_sha,
        &guard,
    )?;
    guard.sync("after16_sync")?;
    guard.expect_mode("after16_get", MODE16)?;
    let contributors16 = compare_native_contributors(
        out,
        &mode16["full"]["nativeColumn"],
        &mode16["tile0"]["nativeColumn"],
    )?;

    // This is the unchanged product entry path and rechecks the original frozen 1/64 failure.
    let full = capture(&vae, source, false, 16, 16, out, "mode16-full-vae75")?;
    let tiled = capture(&vae, source, true, 16, 16, out, "mode16-tiled-vae75")?;
    let seams: Vec<usize> = (16..75).step_by(16).map(|frame| frame * RATIO).collect();
    let whole_vae16 = pair(&tiled, &full, &seams);
    if whole_vae16["clamped"]["maxAbs"].as_f64().unwrap_or(0.0) <= ORIGINAL_BOUND as f64 {
        return Err("same-input M4 mode-16 full/tiled discrepancy did not reproduce".into());
    }

    guard.enable_18()?;
    let experimental_result = arm(
        &vae,
        out,
        "mode18",
        MODE18,
        &inputs,
        &input_records,
        weight_sha,
        &guard,
    );
    guard.restore_16()?;
    let mode18 = experimental_result?;
    if guard.shared_mode(shared)? != 0 {
        return Err("shared handle changed during arm".into());
    }
    let after_weight = weight_record(&vae, out, "after18", DType::BF16)?;
    if after_weight["sha256"] != weight["sha256"] {
        return Err("folded BF16 stage-2 weight changed between arms".into());
    }
    let contributors18 = compare_native_contributors(
        out,
        &mode18["full"]["nativeColumn"],
        &mode18["tile0"]["nativeColumn"],
    )?;
    let mut cross = serde_json::Map::new();
    for slot in ["full", "tile0"] {
        cross.insert(
            slot.into(),
            json!({
                "nativeColumn":compare_records(out, &mode16[slot]["nativeColumn"]["raw"],
                                                  &mode18[slot]["nativeColumn"]["raw"])?,
                "output":compare_records(out, &mode16[slot]["output"],
                                           &mode18[slot]["output"])?,
            }),
        );
    }
    let source_provenance: Value =
        serde_json::from_slice(&fs::read(required_env("YUE2_M4_PEDANTIC_PROVENANCE")?)?)?;
    let event_bytes = fs::read(out.join("math-events.jsonl"))?;
    let report = json!({"schemaVersion":7,"selector":"pedantic_stage2",
        "purpose":"diagnostic_only_no_gate_change","acceptanceSatisfied":false,
        "engineSha":ENGINE_SHA,"sourceProvenance":source_provenance,
        "referenceSha256":reference_hash,"backend":"cuda","deviceOrdinal":0,
        "frames":75,"coreFrames":16,"haloFrames":16,"originalBound":ORIGINAL_BOUND,
        "latent":latent,"foldedWeightBefore":weight,"foldedWeightAfter":after_weight,
        "sourceInputs":{"full":input_records[0],"tile0":input_records[1]},
        "contributorInputAlignment":input_alignment,
        "arms":{"mode16":mode16,"mode18":mode18},
        "nativeContributors":{"mode16":contributors16,"mode18":contributors18},
        "crossArm":cross,"wholeVaeMode16":whole_vae16,
        "wholeVaeMode16Captures":{"full":describe_capture(&full),"tiled":describe_capture(&tiled)},
        "priorObserved":{"run":37139279967u64,"testLine":442,
            "bf16StandardLongMaxAbs":0.037109375,"frozenMaximum":ORIGINAL_BOUND},
        "mathEvents":{"file":"math-events.jsonl","sha256":sha256(&event_bytes),
            "bytes":event_bytes.len(),"handleAddress":guard.handle,
            "streamObjectAddress":guard.stream,"thread":guard.thread}});
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out.join("report.json"))?;
    file.write_all(serde_json::to_string_pretty(&report)?.as_bytes())?;
    file.write_all(b"\n")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn valid_math_bitmask_is_integer_only() {
        assert_eq!(MODE18, 2 | MODE16);
        assert_eq!(MODE16, 16);
    }
}
