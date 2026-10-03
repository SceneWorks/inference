//! Diagnostic-only native stage-2 replay. The immutable parent supplies every input byte.

use super::*;
use candle_audio::candle_core::cuda::cudarc::cublas::{self, sys};
use decoder_trace::{
    compare_native_contributors, native_column_record, raw_values, tensor_from_record,
    tensor_record, weight_record,
};
use std::sync::Arc;

const PARENT: &str = include_str!("../native-convt-math-parent.json");

fn parent_record<'a>(anchor: &'a Value, key: &str) -> Result<&'a Value, Box<dyn Error>> {
    anchor["records"][key]
        .as_object()
        .map(|_| &anchor["records"][key])
        .ok_or_else(|| format!("parent stage-2 record {key} absent").into())
}

fn check_record(actual: &Value, expected: &Value, what: &str) -> Result<(), Box<dyn Error>> {
    for key in ["sha256", "bytes", "dtype", "shape", "layout"] {
        if actual[key] != expected[key] || actual[key].is_null() {
            return Err(format!("{what} changed parent {key}").into());
        }
    }
    for key in ["origin", "coordinateSystem", "cropPadding"] {
        if !expected[key].is_null() && actual[key] != expected[key] {
            return Err(format!("{what} changed parent {key}").into());
        }
    }
    Ok(())
}

struct NativeMathGuard {
    device: Device,
    blas: Arc<cublas::CudaBlas>,
    events: std::fs::File,
    active: bool,
    handle_address: String,
    stream_address: String,
    thread: String,
}

impl NativeMathGuard {
    fn new(device: &Device, out: &Path) -> Result<Self, Box<dyn Error>> {
        let cuda = device.as_cuda_device()?;
        let blas = cuda.cublas_handle();
        let stream = cuda.cuda_stream();
        let mut result = Self {
            device: device.clone(),
            handle_address: format!("{:p}", *blas.handle()),
            stream_address: format!("{:p}", Arc::as_ptr(&stream)),
            thread: format!("{:?}", std::thread::current().id()),
            blas,
            events: OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(out.join("native-math-events.jsonl"))?,
            active: false,
        };
        result.sync("before_default_sync")?;
        if result.get("before_default_get")? != 0 {
            return Err("native stage-2 cuBLAS did not start at default mode 0".into());
        }
        Ok(result)
    }

    fn event(
        &mut self,
        action: &str,
        status: &str,
        raw: Option<u32>,
    ) -> Result<(), Box<dyn Error>> {
        writeln!(
            self.events,
            "{}",
            json!({"action":action,"status":status,
            "rawMode":raw,"handleAddress":self.handle_address,
            "streamObjectAddress":self.stream_address,"thread":self.thread,
            "deviceOrdinal":0,
            "bf16ReducedPrecisionAtomic":
                candle_core::cuda::gemm_reduced_precision_bf16()})
        )?;
        self.events.flush()?;
        Ok(())
    }

    fn sync(&mut self, action: &str) -> Result<(), Box<dyn Error>> {
        let result = self.device.synchronize();
        self.event(
            action,
            if result.is_ok() { "success" } else { "failure" },
            None,
        )?;
        result?;
        Ok(())
    }

    fn get(&mut self, action: &str) -> Result<u32, Box<dyn Error>> {
        let mut raw = u32::MAX;
        let status =
            unsafe { sys::cublasGetMathMode(*self.blas.handle(), (&mut raw as *mut u32).cast()) };
        self.event(
            action,
            &format!("{status:?}"),
            (status == sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS).then_some(raw),
        )?;
        status.result()?;
        Ok(raw)
    }

    fn set(&mut self, action: &str, mode: sys::cublasMath_t) -> Result<(), Box<dyn Error>> {
        let status = unsafe { sys::cublasSetMathMode(*self.blas.handle(), mode) };
        self.event(action, &format!("{status:?}"), Some(mode as u32))?;
        status.result()?;
        Ok(())
    }

    fn after_default(&mut self) -> Result<(), Box<dyn Error>> {
        self.sync("after_default_sync")?;
        if self.get("after_default_get")? != 0 {
            return Err("native default arm changed math mode".into());
        }
        Ok(())
    }

    fn enable(&mut self) -> Result<(), Box<dyn Error>> {
        self.sync("before_flag_sync")?;
        if self.get("before_flag_get")? != 0 {
            return Err("native mode changed before flagged arm".into());
        }
        // A failed setter can still modify state; arm the Drop restoration first.
        self.active = true;
        self.set(
            "set_disallow",
            sys::cublasMath_t::CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION,
        )?;
        if self.get("read_disallow")? != 16 {
            return Err("native disallow-reduced-precision mode was not 16".into());
        }
        Ok(())
    }

    fn restore(&mut self) -> Result<(), Box<dyn Error>> {
        if !self.active {
            return Ok(());
        }
        self.sync("before_restore_sync")?;
        self.set("restore_default", sys::cublasMath_t::CUBLAS_DEFAULT_MATH)?;
        self.sync("after_restore_sync")?;
        if self.get("read_restored")? != 0 {
            return Err("native math-mode restore readback was not 0".into());
        }
        self.active = false;
        Ok(())
    }
}

impl Drop for NativeMathGuard {
    fn drop(&mut self) {
        if self.active && self.restore().is_err() {
            eprintln!("native stage-2 math-mode restoration failed; aborting owned child");
            std::process::abort();
        }
    }
}

fn replay(
    vae: &Yue2Vae,
    parent: &Path,
    out: &Path,
    anchor: &Value,
    arm: &str,
    slot: &str,
    mode: u32,
    weight_sha: &str,
    handle_address: &str,
    stream_address: &str,
) -> Result<Value, Box<dyn Error>> {
    let input_key = format!("{slot}.replayInput");
    let source = parent_record(anchor, &format!("{slot}.sourceInput"))?;
    let replay_source = parent_record(anchor, &input_key)?;
    if source["sha256"] != replay_source["sha256"]
        || source["shape"] != replay_source["shape"]
        || source["origin"] != replay_source["origin"]
    {
        return Err("parent stage-2 source/replay input ceased to be bit-identical".into());
    }
    let input = tensor_from_record(
        replay_source,
        &parent.join("data"),
        vae.device(),
        DType::BF16,
    )?;
    let input_record = tensor_record(
        &input,
        out,
        &format!("{arm}-{slot}-input"),
        replay_source["origin"]
            .as_i64()
            .ok_or("parent input origin absent")?,
    )?;
    check_record(&input_record, replay_source, "stage-2 input")?;
    let mut substeps = Vec::new();
    let mut run = || {
        vae.trace_replay_stage(2, &input, &mut |name, tensor| {
            let mut record = tensor_record(tensor, out, &format!("{arm}-{slot}-{name}"), 0)
                .map_err(decoder_trace::io_candle)?;
            record["coordinateSystem"] = json!(if name == "native_unpadded_conv_transpose" {
                "native_uncropped_conv_transpose"
            } else {
                "cropped_stage"
            });
            if name == "native_unpadded_conv_transpose" {
                record["cropPadding"] = json!(3);
            }
            substeps.push(json!({"name":name,"capture":record}));
            Ok(())
        })
    };
    let expected_length = if slot == "full" { 75 } else { 32 };
    let (output, capture) = candle_core::cuda::with_yue2_native_convt_column(
        DType::BF16,
        [1, expected_length, 1024, 12],
        || run().map_err(decoder_trace::io_candle),
    )?;
    let native = native_column_record(capture, out, arm, slot, expected_length, weight_sha, true)?;
    let output = tensor_record(&output, out, &format!("{arm}-{slot}-output"), 0)?;
    if native["mathModeReadback"] != mode
        || native["weightSha256"] != weight_sha
        || native["bf16ReducedPrecisionAtomic"] != false
        || native["effectiveComputeType"] != "CUBLAS_COMPUTE_32F"
        || native["handleAddress"] != handle_address
        || native["streamObjectAddress"] != stream_address
    {
        return Err("native stage-2 capture mode, compute type, or weight changed".into());
    }
    let result = json!({"sourceInput":source,"replayInput":input_record,
        "weightSha256":weight_sha,"nativeColumn":native,"substeps":substeps,
        "replayOutput":output});
    if arm == "default" {
        for (name, key) in [
            (
                "native_unpadded_conv_transpose",
                "native_unpadded_conv_transpose",
            ),
            ("cropped", "cropped"),
            ("biased", "biased"),
        ] {
            let row = substeps
                .iter()
                .find(|row| row["name"] == name)
                .ok_or("native stage-2 substep absent")?;
            check_record(
                &row["capture"],
                parent_record(anchor, &format!("{slot}.{key}"))?,
                name,
            )?;
        }
        check_record(
            &result["nativeColumn"]["raw"],
            parent_record(anchor, &format!("{slot}.nativeColumn"))?,
            "native column",
        )?;
        check_record(
            &result["replayOutput"],
            parent_record(anchor, &format!("{slot}.replayOutput"))?,
            "stage-2 output",
        )?;
        check_record(
            &result["replayOutput"],
            parent_record(anchor, &format!("{slot}.originalOutput"))?,
            "original stage-2 output",
        )?;
    }
    Ok(result)
}

fn compare_same_shape(out: &Path, a: &Value, b: &Value) -> Result<Value, Box<dyn Error>> {
    if a["shape"] != b["shape"] || a["dtype"] != "BF16" || b["dtype"] != "BF16" {
        return Err("native cross-arm arrays changed dtype or shape".into());
    }
    let av = raw_values(out, a)?;
    let bv = raw_values(out, b)?;
    if av.len() != bv.len() {
        return Err("native cross-arm arrays changed extent".into());
    }
    let mut bit_differences = 0usize;
    let mut positive_differences = 0usize;
    let mut max_abs = 0.0f32;
    for ((bits_a, value_a), (bits_b, value_b)) in av.into_iter().zip(bv) {
        if bits_a != bits_b {
            bit_differences += 1;
        }
        let delta = (value_a - value_b).abs();
        if delta > 0.0 {
            positive_differences += 1;
            max_abs = max_abs.max(delta);
        }
    }
    Ok(
        json!({"bitDifferences":bit_differences,"positiveDifferences":positive_differences,
        "maxAbs":max_abs,"sameShape":true}),
    )
}

pub(super) fn run_native_math(
    out: &Path,
    parent: &Path,
    verified: &candle_audio_yue2::snapshot::VerifiedComponent,
    device: &Device,
    meta: &Value,
    reference_hash: &str,
) -> Result<(), Box<dyn Error>> {
    let anchor: Value = serde_json::from_str(PARENT)?;
    let proof: Value = serde_json::from_slice(&fs::read(parent.join("parent-proof.json"))?)?;
    if proof["runId"] != anchor["runId"]
        || proof["controlSha"] != anchor["controlSha"]
        || proof["metricsZipSha256"] != anchor["metricsZipSha256"]
        || sha256(&fs::read(parent.join("data/report.json"))?) != anchor["reportSha256"]
    {
        return Err("native stage-2 parent artifact lost immutable identity".into());
    }
    let parent_report: Value = serde_json::from_slice(&fs::read(parent.join("data/report.json"))?)?;
    if parent_report["schemaVersion"] != 5
        || parent_report["waveformParity"] != true
        || parent_report["nativeColumns"]["status"] != "collected"
    {
        return Err("native stage-2 parent proof is incomplete".into());
    }
    let standard = &meta["decoders"]["standard"];
    let component = verified.component();
    if component.repo.id != standard["repo"]
        || component.repo.revision != standard["revision"]
        || component.weights().map(|file| file.sha256) != standard["weights_sha256"].as_str()
        || component.file("config.json").map(|file| file.sha256)
            != standard["config_sha256"].as_str()
    {
        return Err("native math standard VAE component changed".into());
    }
    let vae = Yue2Vae::load_with_dtype(verified, VaeParts::Full, device, DType::BF16)?;
    if vae.dtype() != DType::BF16 {
        return Err("native math VAE resident dtype changed".into());
    }
    let weight = weight_record(&vae, out, "math", DType::BF16)?;
    check_record(
        &weight,
        parent_record(&anchor, "weight")?,
        "folded stage-2 weight",
    )?;
    let weight_sha = weight["sha256"].as_str().ok_or("weight SHA absent")?;
    let mut guard = NativeMathGuard::new(device, out)?;
    if candle_core::cuda::gemm_reduced_precision_bf16() {
        return Err("Candle BF16 reduced-precision compute atomic was enabled".into());
    }
    let mut default = serde_json::Map::new();
    for slot in ["full", "window"] {
        default.insert(
            slot.into(),
            replay(
                &vae,
                parent,
                out,
                &anchor,
                "default",
                slot,
                0,
                weight_sha,
                &guard.handle_address,
                &guard.stream_address,
            )?,
        );
    }
    guard.after_default()?;
    guard.enable()?;
    let flagged_result = (|| {
        let mut flagged = serde_json::Map::new();
        for slot in ["full", "window"] {
            flagged.insert(
                slot.into(),
                replay(
                    &vae,
                    parent,
                    out,
                    &anchor,
                    "flagged",
                    slot,
                    16,
                    weight_sha,
                    &guard.handle_address,
                    &guard.stream_address,
                )?,
            );
        }
        Ok::<_, Box<dyn Error>>(flagged)
    })();
    guard.restore()?;
    let flagged = flagged_result?;
    let mut cross_arm = serde_json::Map::new();
    for slot in ["full", "window"] {
        let a = &default[slot];
        let b = &flagged[slot];
        let mut comparisons = serde_json::Map::new();
        comparisons.insert(
            "nativeColumn".into(),
            compare_same_shape(out, &a["nativeColumn"]["raw"], &b["nativeColumn"]["raw"])?,
        );
        comparisons.insert(
            "replayOutput".into(),
            compare_same_shape(out, &a["replayOutput"], &b["replayOutput"])?,
        );
        cross_arm.insert(slot.into(), Value::Object(comparisons));
    }
    let default_contributors = compare_native_contributors(
        out,
        &default["full"]["nativeColumn"],
        &default["window"]["nativeColumn"],
    )?;
    let flagged_contributors = compare_native_contributors(
        out,
        &flagged["full"]["nativeColumn"],
        &flagged["window"]["nativeColumn"],
    )?;
    let mode_events = fs::read(out.join("native-math-events.jsonl"))?;
    let provenance: Value =
        serde_json::from_slice(&fs::read(required_env("YUE2_DECODER_TRACE_PROVENANCE")?)?)?;
    let report = json!({"schemaVersion":6,"selector":"native_convt_math_mode",
        "purpose":"diagnostic_only_no_gate_change","engineSha":ENGINE_SHA,
        "derivativeSource":provenance,"referenceSha256":reference_hash,
        "backend":"cuda","deviceOrdinal":0,"stage":2,"dtype":"BF16",
        "parentProof":proof,"parentAuditSha256":anchor["independentAuditSha256"],
        "parentSourceZipSha256":anchor["sourceZipSha256"],
        "parentWaveformParity":parent_report["waveformParity"],
        "parentRawCheckpointsPerDtype":198,"parentHistoricalRecords":28,
        "foldedWeight":weight,"runs":{"default":default,"disallowReducedPrecision":flagged},
        "nativeContributors":{"default":default_contributors,
            "disallowReducedPrecision":flagged_contributors},
        "crossArm":cross_arm,
        "mathModeEvents":{"file":"native-math-events.jsonl","sha256":sha256(&mode_events),
            "bytes":mode_events.len(),"handleAddress":guard.handle_address,
            "streamObjectAddress":guard.stream_address,"thread":guard.thread},
        "acceptanceSatisfied":false});
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out.join("report.json"))?;
    output.write_all(serde_json::to_string_pretty(&report)?.as_bytes())?;
    output.write_all(b"\n")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_parent_proof_has_all_bf16_stage2_records() {
        let parent: Value = serde_json::from_str(PARENT).unwrap();
        assert_eq!(parent["runId"], 37080478002u64);
        assert_eq!(parent["records"].as_object().unwrap().len(), 17);
        for slot in ["full", "window"] {
            for name in [
                "replayInput",
                "nativeColumn",
                "native_unpadded_conv_transpose",
                "cropped",
                "biased",
                "replayOutput",
            ] {
                let row = parent_record(&parent, &format!("{slot}.{name}")).unwrap();
                assert_eq!(row["dtype"], "BF16");
                assert!(row["sha256"].as_str().unwrap().len() == 64);
            }
        }
    }

    #[test]
    fn changed_parent_substep_is_refused_even_when_other_records_match() {
        let parent: Value = serde_json::from_str(PARENT).unwrap();
        for slot in ["full", "window"] {
            for name in ["nativeColumn", "cropped", "biased", "replayOutput"] {
                let expected = parent_record(&parent, &format!("{slot}.{name}")).unwrap();
                let mut altered = expected.clone();
                altered["sha256"] = json!("0".repeat(64));
                assert!(check_record(&altered, expected, name).is_err());
            }
        }
    }
}
