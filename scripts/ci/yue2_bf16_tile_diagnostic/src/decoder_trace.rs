//! Observational, derivative-source decoder trace. No numerical acceptance gate is changed.

use super::*;
use std::io::{BufWriter, Read};

const STAGES: usize = 33;
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

fn io_candle(error: impl std::fmt::Display) -> candle_core::Error {
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

fn tensor_record(
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

fn raw_values(out: &Path, record: &Value) -> Result<Vec<(u32, f32)>, Box<dyn Error>> {
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

fn tensor_from_record(
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
    let output = vae.trace_replay_stage(stage, &input, &mut |name, tensor| {
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
    })?;
    let output_capture = tensor_record(
        &output,
        out,
        &format!("{label}-replay-stage-{stage:02}-{slot}-output"),
        stage_origin,
    )?;
    let output_exact = output_capture["sha256"] == target_record["sha256"];
    Ok(
        json!({"slot":slot,"sourceInput":source_record,"replayInput":input_capture,
        "originalOutput":target_record,"replayOutput":output_capture,"inputBitExact":input_exact,
        "outputBitExact":output_exact,"substeps":substeps}),
    )
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
        if let (Some(stage), Some(window)) = (earliest, first_window) {
            let full_replay = replay_one(&vae, &z, out, label, stage, None, &stages)?;
            let tile_replay = replay_one(&vae, &z, out, label, stage, Some(window), &stages)?;
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
    let report = json!({"schemaVersion":4,"selector":"decoder_trace",
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
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out.join("report.json"))?;
    file.write_all(serde_json::to_string_pretty(&report)?.as_bytes())?;
    file.write_all(b"\n")?;
    if !parity {
        return Err("traced final waveform differs from original M3: inconclusive".into());
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
}
