//! Observational, derivative-source decoder trace. No numerical acceptance gate is changed.

use super::*;
use std::io::{BufWriter, Read};

const STAGES: usize = 33;
#[cfg(feature = "native_convt_columns")]
const HISTORICAL_STAGE2: &str = include_str!("../historical-stage2.json");
#[cfg(not(feature = "native_convt_columns"))]
const HISTORICAL_STAGE2: &str = "{}";
const ORIGINAL_RAW: [[&str; 2]; 2] = [
    [
        "c12b2b7dca34ee4810536fcee12872674376c24ba484fa600c47fcb35317a908",
        "aa032057a51ad3451ab5826ffe18b2f14afae223adcae245b7663d787676f42c",
    ],
    [
        "742f7878e04093aee8abbb0389ebd4fe4a3e734d8c16a3a3f449cc0281bfad99",
        "2cc2dc0824b1e2115d366829ef1dbc97c6a935e02dfc91237a5f4e7b7211a499",
    ],
];
const ORIGINAL_CLAMPED: [[&str; 2]; 2] = [
    [
        "33046e24cd16c415a9963711807abfc1579159036671ea4d2f08a6ca0a61a9f3",
        "a09be42f9ce9d0129c0640daadef40f711a6c3f65751a28b3e73a14de65d0a17",
    ],
    [
        "dcd3a7091bb2e0a442951ae94f81c2abc27b5956853645e37a998738af3a9646",
        "9cd3aace4657165201b692c2a20b63750d7d004ac68c8dc0136b38101eae9ec3",
    ],
];

pub(super) fn io_candle(error: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(format!("decoder trace observation: {error}"))
}

fn encode_native(value: f32, dtype: DType) -> Vec<u8> {
    let bits = value.to_bits();
    if dtype == DType::BF16 {
        ((bits >> 16) as u16).to_le_bytes().to_vec()
    } else {
        bits.to_le_bytes().to_vec()
    }
}

pub(super) fn tensor_record(
    tensor: &Tensor,
    out: &Path,
    stem: &str,
    origin: i64,
) -> Result<Value, Box<dyn Error>> {
    let [1, channels, length] = tensor.dims() else {
        return Err(format!("{stem} is not BCT: {:?}", tensor.dims()).into());
    };
    if !matches!(tensor.dtype(), DType::BF16 | DType::F32) || *channels == 0 || *length == 0 {
        return Err(format!("{stem} dtype or extent changed").into());
    }
    let dtype = tensor.dtype();
    // BF16 -> F32 is exact. Its high sixteen F32 bits are the original BF16 payload,
    // including the sign bit; do not round through a numeric BF16 constructor.
    let values = tensor
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{stem} has non-finite values").into());
    }
    let suffix = if dtype == DType::BF16 {
        "bf16le"
    } else {
        "f32le"
    };
    let file = format!("{stem}.{suffix}");
    let mut writer = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(out.join(&file))?,
    );
    let mut digest = Sha256::new();
    for value in values {
        let encoded = encode_native(value, dtype);
        writer.write_all(&encoded)?;
        digest.update(&encoded);
    }
    writer.flush()?;
    let bytes = channels * length * if dtype == DType::BF16 { 2 } else { 4 };
    Ok(
        json!({"file":file,"sha256":format!("{:x}",digest.finalize()),"bytes":bytes,
        "layout":format!("bct_{suffix}"),"dtype":format!("{dtype:?}"),
        "shape":[1,channels,length],"origin":origin}),
    )
}

pub(super) fn raw_values(out: &Path, record: &Value) -> Result<Vec<(u32, f32)>, Box<dyn Error>> {
    let file = record["file"].as_str().ok_or("missing trace file")?;
    let mut bytes = Vec::new();
    std::fs::File::open(out.join(file))?.read_to_end(&mut bytes)?;
    if sha256(&bytes) != record["sha256"] || bytes.len() != record["bytes"] {
        return Err(format!("trace array {file} was altered").into());
    }
    let bf16 = record["dtype"] == "BF16";
    let size = if bf16 { 2 } else { 4 };
    if bytes.len() % size != 0 {
        return Err("trace array has fractional element".into());
    }
    bytes
        .chunks_exact(size)
        .map(|part| {
            let bits = if bf16 {
                u32::from(u16::from_le_bytes([part[0], part[1]])) << 16
            } else {
                u32::from_le_bytes([part[0], part[1], part[2], part[3]])
            };
            let value = f32::from_bits(bits);
            if !value.is_finite() {
                return Err("non-finite trace array".into());
            }
            Ok((bits, value))
        })
        .collect()
}

#[cfg(feature = "native_convt_columns")]
pub(super) fn weight_record(
    vae: &Yue2Vae,
    out: &Path,
    label: &str,
    dtype: DType,
) -> Result<Value, Box<dyn Error>> {
    let weight = vae
        .trace_conv_transpose_weight(2)
        .ok_or("native column trace lost stage-2 folded weight")?;
    if weight.dtype() != dtype || weight.dims() != [2048, 1024, 12] {
        return Err("native ConvT weight dtype/shape changed".into());
    }
    let suffix = if dtype == DType::BF16 {
        "bf16le"
    } else {
        "f32le"
    };
    let file = format!("{label}-stage-02-folded-weight.{suffix}");
    let mut writer = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(out.join(&file))?,
    );
    let mut digest = Sha256::new();
    let values = weight
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()?;
    for value in values {
        if !value.is_finite() {
            return Err("native ConvT folded weight is non-finite".into());
        }
        let encoded = encode_native(value, dtype);
        writer.write_all(&encoded)?;
        digest.update(&encoded);
    }
    writer.flush()?;
    Ok(
        json!({"file":file,"sha256":format!("{:x}",digest.finalize()),
        "bytes":2048*1024*12*if dtype==DType::BF16{2}else{4},
        "layout":format!("cick_{suffix}"),"dtype":format!("{dtype:?}"),
        "shape":[2048,1024,12]}),
    )
}

#[cfg(all(feature = "native_convt_columns", feature = "cuda"))]
pub(super) fn native_column_record(
    capture: candle_core::cuda::Yue2NativeConvtColumn,
    out: &Path,
    label: &str,
    slot: &str,
    expected_length: usize,
    weight_sha256: &str,
    include_math_details: bool,
) -> Result<Value, Box<dyn Error>> {
    let dtype = capture.dtype;
    if !matches!(dtype, DType::BF16 | DType::F32)
        || capture.shape != [1, expected_length, 1024, 12]
        || capture.gemm != [1, expected_length, 12288, 2048]
        || capture.kernel_layout_strides != [0, 12288, 1]
        || capture.branch != "native_col2im"
        || capture.bytes.len()
            != expected_length * 1024 * 12 * if dtype == DType::BF16 { 2 } else { 4 }
    {
        return Err("captured native column branch, shape, or GEMM layout changed".into());
    }
    let suffix = if dtype == DType::BF16 {
        "bf16le"
    } else {
        "f32le"
    };
    let file = format!("{label}-stage-02-{slot}-native-column.{suffix}");
    let bytes = capture.bytes.len();
    let hash = sha256(&capture.bytes);
    let mut writer = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(out.join(&file))?,
    );
    writer.write_all(&capture.bytes)?;
    writer.flush()?;
    let raw = json!({"file":file,"sha256":hash,"bytes":bytes,"layout":format!("blck_{suffix}"),
        "dtype":format!("{dtype:?}"),"shape":capture.shape});
    let _ = raw_values(out, &raw)?;
    let mut record = json!({"raw":raw,"branch":capture.branch,"gemm":capture.gemm,
        "kernelLayoutStrides":capture.kernel_layout_strides,
        "mathModeReadback":capture.math_mode_readback,"captureCount":1,
        "weightSha256":weight_sha256});
    if include_math_details {
        record["bf16ReducedPrecisionAtomic"] = json!(capture.bf16_reduced_precision_atomic);
        record["effectiveComputeType"] = json!(capture.effective_compute_type);
        record["handleAddress"] = json!(capture.handle_address);
        record["streamObjectAddress"] = json!(capture.stream_object_address);
    }
    Ok(record)
}

#[cfg(feature = "native_convt_columns")]
pub(super) fn compare_native_contributors(
    out: &Path,
    full: &Value,
    tile: &Value,
) -> Result<Value, Box<dyn Error>> {
    let a = &full["raw"];
    let b = &tile["raw"];
    let dtype = a["dtype"].as_str().ok_or("native column dtype absent")?;
    if b["dtype"] != dtype
        || a["shape"] != json!([1, 75, 1024, 12])
        || b["shape"] != json!([1, 32, 1024, 12])
    {
        return Err("native column comparison shape/dtype changed".into());
    }
    let (a, b) = (raw_values(out, a)?, raw_values(out, b)?);
    let mut compared = 0usize;
    let mut bit_differences = 0usize;
    let mut positive_differences = 0usize;
    let mut signed_zero_only = 0usize;
    let mut max_abs = 0f32;
    let mut first = Value::Null;
    for raw_frame in 3usize..=173 {
        let current = raw_frame / 6;
        let tap = raw_frame % 6;
        for channel in 0..1024 {
            for (row, kernel_tap) in [(current, tap), (current.saturating_sub(1), tap + 6)] {
                if kernel_tap >= 6 && current == 0 {
                    continue;
                }
                let at = (row * 1024 + channel) * 12 + kernel_tap;
                let (x, y) = (a[at], b[at]);
                let delta = (x.1 - y.1).abs();
                compared += 1;
                max_abs = max_abs.max(delta);
                if x.0 != y.0 {
                    bit_differences += 1;
                    if delta > 0.0 {
                        positive_differences += 1;
                    }
                    if x.1 == 0.0 && y.1 == 0.0 {
                        signed_zero_only += 1;
                    }
                    if first.is_null() {
                        first = json!({"rawFrame":raw_frame,"inputRow":row,
                            "kernelTap":kernel_tap,"channel":channel,
                            "full":x.1,"tile":y.1,"absError":delta});
                    }
                }
            }
        }
    }
    Ok(
        json!({"rawValidInclusive":[3,173],"comparedContributors":compared,
        "bitDifferences":bit_differences,"positiveDifferences":positive_differences,
        "signedZeroOnly":signed_zero_only,"maxAbs":max_abs,"firstDifferent":first}),
    )
}

fn first_true(mut lo: usize, mut hi: usize, mut predicate: impl FnMut(usize) -> bool) -> usize {
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if predicate(mid) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo
}

#[allow(clippy::too_many_arguments)]
fn valid_interval(
    vae: &Yue2Vae,
    stage: usize,
    full_len: usize,
    tile_len: usize,
    left: usize,
    right: usize,
    frames: usize,
    scale: usize,
) -> Result<(usize, usize), Box<dyn Error>> {
    let origin = left * scale;
    let end = full_len.min(origin + tile_len);
    if origin >= end {
        return Err("no trace overlap".into());
    }
    let low = first_true(origin, end, |index| {
        let (_, _, (a, _)) = vae
            .trace_prefix_geometry(stage, frames, index)
            .expect("overlap index");
        left == 0 || a >= left as i64
    });
    let high = first_true(low, end, |index| {
        let (_, _, (_, b)) = vae
            .trace_prefix_geometry(stage, frames, index)
            .expect("overlap index");
        right != frames && b >= right as i64
    });
    if low >= high {
        return Err("empty prefix-valid trace interval".into());
    }
    Ok((low, high - 1)) // inclusive, explicitly recorded in report
}

#[allow(clippy::too_many_arguments)]
fn compare(
    vae: &Yue2Vae,
    out: &Path,
    stage: usize,
    full: &Value,
    tile: &Value,
    left: usize,
    right: usize,
    frames: usize,
    core_start: usize,
    core_end: usize,
) -> Result<(Value, bool), Box<dyn Error>> {
    let full_shape = full["shape"].as_array().ok_or("full shape absent")?;
    let tile_shape = tile["shape"].as_array().ok_or("tile shape absent")?;
    let (channels, full_len, tile_len) = (
        full_shape[1].as_u64().unwrap() as usize,
        full_shape[2].as_u64().unwrap() as usize,
        tile_shape[2].as_u64().unwrap() as usize,
    );
    if full_shape[1] != tile_shape[1] || full["dtype"] != tile["dtype"] {
        return Err("stage tensor shape/dtype mismatch".into());
    }
    let (scale, expected_full, _) = vae
        .trace_prefix_geometry(stage, frames, 0)
        .ok_or("missing stage geometry")?;
    let (_, expected_tile, _) = vae
        .trace_prefix_geometry(stage, right - left, 0)
        .ok_or("missing tile geometry")?;
    if full_len != expected_full || tile_len != expected_tile {
        return Err("stage output length differs from M3".into());
    }
    let (lo, hi) = valid_interval(vae, stage, full_len, tile_len, left, right, frames, scale)?;
    let core_lo = core_start * scale;
    let core_hi = (core_end * scale).min(full_len);
    if !(lo <= core_lo && hi + 1 >= core_hi) {
        return Err("nominal core escaped prefix-valid overlap".into());
    }
    let (a, b) = (raw_values(out, full)?, raw_values(out, tile)?);
    if a.len() != channels * full_len || b.len() != channels * tile_len {
        return Err("stage element count mismatch".into());
    }
    let mut bit_differences = 0usize;
    let mut positive_differences = 0usize;
    let mut signed_zero_only = 0usize;
    let mut max_abs = 0f32;
    let mut first = Value::Null;
    for channel in 0..channels {
        for global in lo..=hi {
            let x = a[channel * full_len + global];
            let y = b[channel * tile_len + global - left * scale];
            let delta = (x.1 - y.1).abs();
            max_abs = max_abs.max(delta);
            if x.0 != y.0 {
                bit_differences += 1;
                if delta > 0.0 {
                    positive_differences += 1;
                }
                if x.1 == 0.0 && y.1 == 0.0 {
                    signed_zero_only += 1;
                }
                if first.is_null() {
                    first = json!({"channel":channel,"globalFrame":global,
                    "full":x.1,"tile":y.1,"absError":delta});
                }
            }
        }
    }
    Ok((
        json!({"validInclusive":[lo,hi],"comparedValues":channels*(hi-lo+1),
        "bitDifferences":bit_differences,"positiveDifferences":positive_differences,
        "signedZeroOnly":signed_zero_only,"maxAbs":max_abs,"firstDifferent":first}),
        positive_differences > 0,
    ))
}

fn waveform_record(
    tensor: &Tensor,
    dtype: DType,
    out: &Path,
    stem: &str,
    expected: [&str; 2],
) -> Result<Value, Box<dyn Error>> {
    if tensor.dtype() != dtype || tensor.dims() != [1, 2, 143936] {
        return Err("traced waveform dtype/shape changed".into());
    }
    let raw = interleaved(tensor, false)?;
    let clamped = interleaved(tensor, true)?;
    if raw.iter().chain(&clamped).any(|v| !v.is_finite()) {
        return Err("traced waveform is non-finite".into());
    }
    let raw_sha = samples_sha256(&raw);
    let clamped_sha = samples_sha256(&clamped);
    let raw_artifact = write_samples(out, &format!("{stem}-raw"), &raw)?;
    let clamped_artifact = write_samples(out, &format!("{stem}-clamped"), &clamped)?;
    Ok(
        json!({"rawSha256":raw_sha,"clampedSha256":clamped_sha,"rawDtype":format!("{dtype:?}"),
        "rawArtifact":raw_artifact,"clampedArtifact":clamped_artifact,
        "matchesOriginal":raw_sha==expected[0] && clamped_sha==expected[1]}),
    )
}

fn waveform_peak(out: &Path, a: &Value, b: &Value) -> Result<f32, Box<dyn Error>> {
    let read = |record: &Value| -> Result<Vec<f32>, Box<dyn Error>> {
        let file = record["clampedArtifact"]["file"]
            .as_str()
            .ok_or("clamped file absent")?;
        let bytes = fs::read(out.join(file))?;
        if bytes.len() != record["clampedArtifact"]["bytes"]
            || sha256(&bytes) != record["clampedSha256"]
        {
            return Err("clamped waveform artifact changed".into());
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes([v[0], v[1], v[2], v[3]]))
            .collect())
    };
    let (x, y) = (read(a)?, read(b)?);
    if x.len() != y.len() {
        return Err("full/tile waveform extent differs".into());
    }
    Ok(x.into_iter()
        .zip(y)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max))
}

pub(super) fn tensor_from_record(
    record: &Value,
    out: &Path,
    device: &Device,
    dtype: DType,
) -> Result<Tensor, Box<dyn Error>> {
    let shape = record["shape"].as_array().ok_or("replay shape absent")?;
    let dims = (
        shape[0].as_u64().ok_or("batch absent")? as usize,
        shape[1].as_u64().ok_or("channels absent")? as usize,
        shape[2].as_u64().ok_or("frames absent")? as usize,
    );
    if dims.0 != 1 || record["dtype"] != format!("{dtype:?}") {
        return Err("replay source dtype/shape changed".into());
    }
    let values = raw_values(out, record)?
        .into_iter()
        .map(|(_, value)| value)
        .collect::<Vec<_>>();
    if values.len() != dims.0 * dims.1 * dims.2 {
        return Err("replay source element count changed".into());
    }
    Ok(Tensor::from_vec(values, dims, &Device::Cpu)?
        .to_device(device)?
        .to_dtype(dtype)?)
}

#[allow(clippy::too_many_arguments)]
fn replay_one(
    vae: &Yue2Vae,
    z: &Tensor,
    out: &Path,
    label: &str,
    stage: usize,
    window: Option<usize>,
    observed: &[Value],
    native_weight_sha256: Option<&str>,
) -> Result<Value, Box<dyn Error>> {
    let (slot, left, right) = if let Some(index) = window {
        let row = &observed[stage]["windows"][index];
        (
            format!("tile-{index}"),
            row["left"].as_u64().unwrap() as usize,
            row["right"].as_u64().unwrap() as usize,
        )
    } else {
        ("full".to_owned(), 0, z.dim(2)?)
    };
    let source_record = if stage == 0 {
        let input = if window.is_some() {
            z.narrow(2, left, right - left)?
        } else {
            z.clone()
        };
        tensor_record(
            &input,
            out,
            &format!("{label}-replay-stage-{stage:02}-{slot}-source"),
            left as i64,
        )?
    } else if let Some(index) = window {
        observed[stage - 1]["windows"][index]["capture"].clone()
    } else {
        observed[stage - 1]["full"].clone()
    };
    let input = tensor_from_record(&source_record, out, vae.device(), vae.dtype())?;
    let input_capture = tensor_record(
        &input,
        out,
        &format!("{label}-replay-stage-{stage:02}-{slot}-input"),
        source_record["origin"]
            .as_i64()
            .ok_or("replay input origin absent")?,
    )?;
    let input_exact = input_capture["sha256"] == source_record["sha256"];
    let mut substeps = Vec::new();
    let target_record = if let Some(index) = window {
        &observed[stage]["windows"][index]["capture"]
    } else {
        &observed[stage]["full"]
    };
    let stage_origin = target_record["origin"]
        .as_i64()
        .ok_or("stage origin absent")?;
    let mut replay = || {
        vae.trace_replay_stage(stage, &input, &mut |name, tensor| {
            let mut record = tensor_record(
                tensor,
                out,
                &format!("{label}-replay-stage-{stage:02}-{slot}-{name}"),
                stage_origin,
            )
            .map_err(io_candle)?;
            if name == "native_unpadded_conv_transpose" {
                record["coordinateSystem"] = json!("native_uncropped_conv_transpose");
                record["cropPadding"] = json!(vae
                    .trace_conv_transpose_padding(stage)
                    .ok_or_else(|| io_candle("ConvT padding absent"))?);
            } else {
                record["coordinateSystem"] = json!("cropped_stage");
            }
            substeps.push(json!({"name":name,"capture":record}));
            Ok(())
        })
    };
    let (output, native_column) = if let Some(weight_sha256) = native_weight_sha256 {
        #[cfg(all(feature = "native_convt_columns", feature = "cuda"))]
        {
            let (output, capture) = candle_core::cuda::with_yue2_native_convt_column(
                vae.dtype(),
                [1, right - left, 1024, 12],
                || replay().map_err(io_candle),
            )?;
            let record = native_column_record(
                capture,
                out,
                label,
                &slot,
                right - left,
                weight_sha256,
                false,
            )?;
            (output, Some(record))
        }
        #[cfg(not(all(feature = "native_convt_columns", feature = "cuda")))]
        {
            let _ = weight_sha256;
            return Err("native column capture requires the CUDA derivative build".into());
        }
    } else {
        (replay()?, None)
    };
    let output_capture = tensor_record(
        &output,
        out,
        &format!("{label}-replay-stage-{stage:02}-{slot}-output"),
        stage_origin,
    )?;
    let output_exact = output_capture["sha256"] == target_record["sha256"];
    let mut result = json!({"slot":slot,"sourceInput":source_record,"replayInput":input_capture,
        "originalOutput":target_record,"replayOutput":output_capture,"inputBitExact":input_exact,
        "outputBitExact":output_exact,"substeps":substeps});
    if let Some(record) = native_column {
        result["nativeColumn"] = record;
    }
    Ok(result)
}

fn matches_historical_record(actual: &Value, expected: &Value) -> bool {
    ["file", "sha256", "bytes", "dtype", "shape"]
        .iter()
        .all(|key| !expected[key].is_null() && actual[key] == expected[key])
}

fn matches_historical_stage2_slot(replay: &Value, expected: &Value) -> bool {
    for key in [
        "sourceInput",
        "replayInput",
        "originalOutput",
        "replayOutput",
    ] {
        if !matches_historical_record(&replay[key], &expected[key]) {
            return false;
        }
    }
    let Some(steps) = replay["substeps"].as_array() else {
        return false;
    };
    if steps.len() != 3 {
        return false;
    }
    for name in ["native_unpadded_conv_transpose", "cropped", "biased"] {
        let mut matching = steps.iter().filter(|step| step["name"] == name);
        let Some(step) = matching.next() else {
            return false;
        };
        if matching.next().is_some()
            || !matches_historical_record(&step["capture"], &expected[name])
        {
            return false;
        }
    }
    true
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_decoder_trace(
    engine: &Path,
    out: &Path,
    source: &AcousticLatents,
    verified: &candle_audio_yue2::snapshot::VerifiedComponent,
    device: &Device,
    meta: &Value,
    reference_hash: &str,
    frames: usize,
    core: usize,
    halo: usize,
    native_columns: bool,
) -> Result<(), Box<dyn Error>> {
    if (frames, core, halo) != (75, 16, 16) {
        return Err("decoder trace geometry changed".into());
    }
    let standard = &meta["decoders"]["standard"];
    let component = verified.component();
    if component.repo.id != standard["repo"]
        || component.repo.revision != standard["revision"]
        || component.weights().map(|f| f.sha256) != standard["weights_sha256"].as_str()
        || component.file("config.json").map(|f| f.sha256) != standard["config_sha256"].as_str()
    {
        return Err("decoder trace component differs from pinned standard VAE".into());
    }
    let mut runs = BTreeMap::new();
    let mut earliest = None;
    let mut first_window = None;
    let mut adaptive_runs = BTreeMap::new();
    let historical: Value = serde_json::from_str(HISTORICAL_STAGE2)?;
    if native_columns
        && (historical["source_run"] != 36979353496u64
            || historical["source_control"] != "4f89f150e13647c53d37dddfa7d36e3f1d28ac85"
            || historical["engine"] != ENGINE_SHA
            || historical["report_sha256"]
                != "9dd0a4113f43a185e573f79c99c2d3e01eb8525923e271cc56a341e21344301f"
            || historical["metrics_zip_sha256"]
                != "a26439ced8b5244b75a65d31b565341ff4bf0280cc3d02e6fb982563ccca379c"
            || historical["stage"] != 2
            || historical["window"] != 0)
    {
        return Err("historical stage-2 receipt identity changed".into());
    }
    let mut historical_checks = BTreeMap::new();
    let mut native_runs = BTreeMap::<&str, Value>::new();
    let mut bf16_clamped_peak = None;
    for (rank, (label, dtype)) in [("bf16", DType::BF16), ("f32", DType::F32)]
        .into_iter()
        .enumerate()
    {
        let vae = Yue2Vae::load_with_dtype(verified, VaeParts::Full, device, dtype)?;
        if vae.dtype() != dtype
            || vae.required_halo(core) > halo
            || vae.identity().weights_sha256 != standard["weights_sha256"]
        {
            return Err("resident decoder identity/dtype/halo changed".into());
        }
        let native_weight: Option<Value> = if native_columns {
            #[cfg(feature = "native_convt_columns")]
            {
                Some(weight_record(&vae, out, label, dtype)?)
            }
            #[cfg(not(feature = "native_convt_columns"))]
            {
                return Err("native column capture requires a derivative harness".into());
            }
        } else {
            None
        };
        let z = source.to_decoder_input(device)?.to_dtype(dtype)?;
        let mut stages = Vec::<Value>::new();
        let full = vae.trace_decode_full(&z, &mut |stage, _, output| {
            if stage != stages.len() {
                return Err(io_candle("stage order changed"));
            }
            let (scale, len, _) = vae
                .trace_prefix_geometry(stage, frames, 0)
                .ok_or_else(|| io_candle("stage geometry absent"))?;
            if output.dim(2)? != len || output.dtype() != dtype {
                return Err(io_candle("stage dtype/length changed"));
            }
            let capture = tensor_record(output, out, &format!("{label}-stage-{stage:02}-full"), 0)
                .map_err(io_candle)?;
            let kind = vae
                .trace_stage_kind(stage)
                .ok_or_else(|| io_candle("stage kind absent"))?;
            stages
                .push(json!({"index":stage,"kind":kind,"scale":scale,"full":capture,"windows":[]}));
            Ok(())
        })?;
        if stages.len() != STAGES {
            return Err("decoder no longer has 33 stages".into());
        }
        let mut windows_seen = [0usize; STAGES];
        let tiled = vae.trace_decode_tiled(
            &z,
            core,
            halo,
            &mut |window, left, right, stage, _, output| {
                if stage >= STAGES || window != windows_seen[stage] {
                    return Err(io_candle("tile stage order changed"));
                }
                windows_seen[stage] += 1;
                let start = window * core;
                let end = frames.min(start + core);
                let scale = stages[stage]["scale"]
                    .as_u64()
                    .ok_or_else(|| io_candle("scale absent"))? as usize;
                let capture = tensor_record(
                    output,
                    out,
                    &format!("{label}-stage-{stage:02}-tile-{window}"),
                    (left * scale) as i64,
                )
                .map_err(io_candle)?;
                let (comparison, positive) = compare(
                    &vae,
                    out,
                    stage,
                    &stages[stage]["full"],
                    &capture,
                    left,
                    right,
                    frames,
                    start,
                    end,
                )
                .map_err(io_candle)?;
                if dtype == DType::BF16 && positive && earliest.is_none_or(|old| stage < old) {
                    earliest = Some(stage);
                    first_window = Some(window);
                }
                stages[stage]["windows"]
                    .as_array_mut()
                    .ok_or_else(|| io_candle("window list absent"))?
                    .push(
                        json!({"index":window,"start":start,"end":end,"left":left,"right":right,
                    "origin":left*scale,"capture":capture,"comparison":comparison}),
                    );
                Ok(())
            },
        )?;
        if windows_seen.iter().any(|count| *count != 5) {
            return Err("decoder trace window coverage incomplete".into());
        }
        let full_wave = waveform_record(
            &full,
            dtype,
            out,
            &format!("{label}-full"),
            [ORIGINAL_RAW[rank][0], ORIGINAL_CLAMPED[rank][0]],
        )?;
        let tiled_wave = waveform_record(
            &tiled,
            dtype,
            out,
            &format!("{label}-tiled"),
            [ORIGINAL_RAW[rank][1], ORIGINAL_CLAMPED[rank][1]],
        )?;
        if dtype == DType::BF16 {
            bf16_clamped_peak = Some(waveform_peak(out, &full_wave, &tiled_wave)?);
        }
        if native_columns && (earliest, first_window) != (Some(2), Some(0)) {
            return Err(
                "native column selector requires the observed first stage-2 tile-0 divergence"
                    .into(),
            );
        }
        if let (Some(stage), Some(window)) = (earliest, first_window) {
            let weight_sha256 = native_weight
                .as_ref()
                .and_then(|weight| weight["sha256"].as_str());
            let full_replay =
                replay_one(&vae, &z, out, label, stage, None, &stages, weight_sha256)?;
            let tile_replay = replay_one(
                &vae,
                &z,
                out,
                label,
                stage,
                Some(window),
                &stages,
                weight_sha256,
            )?;
            if native_columns {
                let expected = &historical["records"][label];
                historical_checks.insert(
                    label,
                    json!({
                        "full": matches_historical_stage2_slot(&full_replay, &expected["full"]),
                        "window": matches_historical_stage2_slot(&tile_replay, &expected["window"]),
                    }),
                );
            }
            if native_columns {
                #[cfg(feature = "native_convt_columns")]
                {
                    let full = &full_replay["nativeColumn"];
                    let window = &tile_replay["nativeColumn"];
                    let comparison = compare_native_contributors(out, full, window)?;
                    native_runs.insert(
                        label,
                        json!({"weight":native_weight,
                        "full":full,"window":window,"comparison":comparison}),
                    );
                }
            }
            adaptive_runs.insert(label, json!({"full":full_replay,"window":tile_replay}));
        }
        runs.insert(
            label,
            json!({"residentDtype":format!("{dtype:?}"),"decoderIdentity":vae.identity().to_json(),
            "stages":stages,"waveform":{"full":full_wave,"tiled":tiled_wave}}),
        );
    }
    let parity = bf16_clamped_peak == Some(0.03125)
        && runs.values().all(|run| {
            run["waveform"]["full"]["matchesOriginal"] == true
                && run["waveform"]["tiled"]["matchesOriginal"] == true
        });
    let replay_exact = adaptive_runs.values().all(|run: &Value| {
        run["full"]["inputBitExact"] == true
            && run["full"]["outputBitExact"] == true
            && run["window"]["inputBitExact"] == true
            && run["window"]["outputBitExact"] == true
    });
    let historical_exact = historical_checks.len() == 2
        && historical_checks
            .values()
            .all(|check: &Value| check["full"] == true && check["window"] == true);
    let adaptive = if let (Some(stage), Some(window)) = (earliest, first_window) {
        json!({"status":if replay_exact{"collected"}else{"inconclusive_replay_mismatch"},
            "source":"single_native_layer_replay","stage":stage,"window":window,
            "kind":runs["bf16"]["stages"][stage]["kind"],"runs":adaptive_runs})
    } else {
        json!({"status":"not_applicable","reason":"no_positive_prefix_valid_stage_residual"})
    };
    let derivative_source: Value =
        serde_json::from_slice(&fs::read(required_env("YUE2_DECODER_TRACE_PROVENANCE")?)?)?;
    if derivative_source["engine_sha"] != ENGINE_SHA {
        return Err("derivative provenance lost M3 source".into());
    }
    let mut report = json!({"schemaVersion":if native_columns {5} else {4},
        "selector":if native_columns {"native_convt_columns"} else {"decoder_trace"},
        "purpose":"diagnostic_only_no_gate_change","engineSha":ENGINE_SHA,
        "derivativeSource":derivative_source,
        "referenceSha256":reference_hash,
        "referenceMetadataSha256":sha256(&fs::read(engine.join("crates/audio/candle-audio-yue2/tests/fixtures/vae_real_reference.json"))?),
        "latentIdentity":source.identity().to_json(),"backend":"cuda","deviceOrdinal":0,
        "decoderIdentity":{"repo":YUE2_VAE_REPO.id,"revision":YUE2_VAE_REPO.revision,
            "weights_sha256":standard["weights_sha256"],"config_sha256":standard["config_sha256"]},
        "frames":frames,"coreFrames":core,"haloFrames":halo,"stageCount":STAGES,
        "rawEncoding":{"BF16":"bct_bf16le_lossless_via_f32_upper16","F32":"bct_f32le"},
        "originalWaveformRunId":"36884387320","originalBound":ORIGINAL_BOUND,
        "waveformParity":parity,"bf16ClampedMaxAbs":bf16_clamped_peak,
        "earliestBf16Stage":earliest,"adaptive":adaptive,"runs":runs});
    if native_columns {
        report["historicalStage2"] = json!({
            "sourceRunId":historical["source_run"],
            "sourceControlSha":historical["source_control"],
            "sourceReportSha256":historical["report_sha256"],
            "sourceMetricsZipSha256":historical["metrics_zip_sha256"],
            "expectedMapSha256":sha256(HISTORICAL_STAGE2.as_bytes()),
            "checks":historical_checks,
            "matchesHistorical":historical_exact,
        });
        report["nativeColumns"] = json!({"status":if parity && replay_exact && historical_exact {"collected"}
            else {"inconclusive_parity"},"stage":2,"window":0,
            "geometry":{"stride":6,"kernel":12,"cropPadding":3,
                "rawValidInclusive":[3,173],"inputValidInclusive":[0,28]},
            "runs":native_runs});
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out.join("report.json"))?;
    file.write_all(serde_json::to_string_pretty(&report)?.as_bytes())?;
    file.write_all(b"\n")?;
    if !parity || (native_columns && (!replay_exact || !historical_exact)) {
        return Err(
            "trace failed original waveform, replay, or historical stage-2 identity: inconclusive"
                .into(),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_bf16_encoding_retains_bits_without_rounding() {
        assert_eq!(encode_native(-0.0, DType::BF16), [0, 0x80]);
        assert_eq!(encode_native(1.5, DType::BF16), [0xc0, 0x3f]);
        assert_eq!(encode_native(1.5, DType::F32), 1.5f32.to_le_bytes());
    }
    #[test]
    fn signed_zero_is_not_positive_numeric_divergence() {
        let a = f32::from_bits(0x8000_0000);
        let b = 0f32;
        assert_ne!(a.to_bits(), b.to_bits());
        assert_eq!((a - b).abs(), 0f32);
    }
    #[test]
    fn binary_search_valid_interval_edges() {
        assert_eq!(first_true(0, 100, |x| x >= 19), 19);
        assert_eq!(first_true(19, 100, |x| x >= 61), 61);
        assert_eq!(first_true(0, 100, |_| false), 100);
    }

    #[cfg(feature = "native_convt_columns")]
    #[test]
    fn historical_stage2_rejects_consistent_current_run_mutations() {
        assert_eq!(
            sha256(HISTORICAL_STAGE2.as_bytes()),
            "a09106a9e5602c89bb44c21e31388f8f26a45233a393a15dd2016d355bb0e9f9"
        );
        let manifest: Value = serde_json::from_str(HISTORICAL_STAGE2).unwrap();
        for dtype in ["bf16", "f32"] {
            for slot in ["full", "window"] {
                let expected = &manifest["records"][dtype][slot];
                let steps: Vec<Value> = ["native_unpadded_conv_transpose", "cropped", "biased"]
                    .iter()
                    .map(|name| json!({"name":name,"capture":expected[name]}))
                    .collect();
                let mut replay = json!({"sourceInput":expected["sourceInput"],
                    "replayInput":expected["replayInput"],
                    "originalOutput":expected["originalOutput"],
                    "replayOutput":expected["replayOutput"],"substeps":steps});
                assert!(matches_historical_stage2_slot(&replay, expected));
                for key in ["sourceInput", "replayInput"] {
                    replay[key]["sha256"] = json!("paired-new-input");
                }
                assert!(!matches_historical_stage2_slot(&replay, expected));
                for key in ["sourceInput", "replayInput"] {
                    replay[key] = expected[key].clone();
                }
                for key in ["originalOutput", "replayOutput"] {
                    replay[key]["sha256"] = json!("paired-new-output");
                }
                assert!(!matches_historical_stage2_slot(&replay, expected));
                for key in ["originalOutput", "replayOutput"] {
                    replay[key] = expected[key].clone();
                }
                for index in 0..3 {
                    replay["substeps"][index]["capture"]["sha256"] = json!("changed-substep");
                    assert!(!matches_historical_stage2_slot(&replay, expected));
                    replay["substeps"][index]["capture"] =
                        expected[replay["substeps"][index]["name"].as_str().unwrap()].clone();
                }
            }
        }
    }
}
