//! Raw BF16 operand/column/output recording copied from the reviewed native-column diagnostic.
use super::*;
use std::io::{BufWriter, Read};

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
    let weight = vae.diagnostic_stage2_weight()?;
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
