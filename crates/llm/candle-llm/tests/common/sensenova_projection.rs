//! Production projection slice for the registered CUDA proof. The reference
//! decodes GGML blocks on the host independently of Candle's CUDA kernels.
use candle_core::{
    quantized::{GgmlDType, QTensor},
    DType, Device, Tensor,
};
use candle_llm::primitives::{projection::Projection, quant::from_ggml_block_tensor};
use serde_json::Value;
use std::path::Path;

const KEY: &str = "model.language_model.layers.4.linear_attn.in_proj_z.weight";
const COLS: usize = 4096;
const ROWS: usize = 32;
const FIRST_ROW: usize = 2032;

fn rows(root: &Path, dtype: DType, row_shape: &[usize]) -> Tensor {
    let index: Value = serde_json::from_reader(
        std::fs::File::open(root.join("model.safetensors.index.json")).unwrap(),
    )
    .unwrap();
    let path = root.join(index["weight_map"][KEY].as_str().unwrap());
    // SAFETY: the endpoint stages immutable, digest-verified release files.
    let mapping = unsafe { candle_core::safetensors::MmapedSafetensors::new(path) }.unwrap();
    let view = mapping.get(KEY).unwrap();
    assert_eq!(view.shape()[0], 4096);
    assert_eq!(&view.shape()[1..], row_shape);
    let stride = row_shape.iter().product::<usize>() * dtype.size_in_bytes();
    let bytes = &view.data()[FIRST_ROW * stride..(FIRST_ROW + ROWS) * stride];
    let mut dims = vec![ROWS];
    dims.extend_from_slice(row_shape);
    Tensor::from_raw_buffer(bytes, dtype, &dims, &Device::Cpu).unwrap()
}

fn half_value(bytes: &[u8]) -> f32 {
    half::f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32()
}

fn q4_cell(block: &[u8], index: usize) -> (f32, f32, f32) {
    let scales = &block[4..16];
    let j = index / 32;
    let (scale, min) = if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 15) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    };
    let nibble = block[16 + (index / 64) * 32 + index % 32];
    let code = if j.is_multiple_of(2) {
        nibble & 15
    } else {
        nibble >> 4
    };
    let step = half_value(block) * f32::from(scale);
    let min = half_value(&block[2..]) * f32::from(min);
    (step * f32::from(code) - min, step, min)
}

pub(super) fn verify(original: &Path, prepared: &Path, format: GgmlDType) {
    let dense = rows(original, DType::BF16, &[COLS])
        .to_dtype(DType::F32)
        .unwrap();
    let stored = rows(
        prepared,
        DType::U8,
        &[COLS / format.block_size(), format.type_size()],
    );
    let bytes = stored.flatten_all().unwrap().to_vec1::<u8>().unwrap();
    // A published tier must reproduce the exact named codec applied to the
    // pinned source, including its sub-block scales, minima and integer codes.
    let canonical = QTensor::quantize(&dense, format).unwrap();
    assert!(
        bytes.as_slice() == canonical.data().unwrap().as_ref(),
        "stored {format:?} bytes differ from source encoding for {KEY}"
    );
    let source = dense.to_vec2::<f32>().unwrap();
    let device = candle_llm::device::select_device().unwrap();
    assert!(device.is_cuda());
    let projection =
        Projection::load_qtensor(from_ggml_block_tensor(&stored, &device).unwrap(), None).unwrap();
    // 127 is exactly representable in BF16 and gives the MMVQ Q8_1 activation
    // block an exact unit scale. The true FP32 dot is 127 times one coefficient.
    let probes = [0, 63, 1281, 4095];
    let mut input = vec![0f32; probes.len() * COLS];
    for (i, &column) in probes.iter().enumerate() {
        input[i * COLS + column] = 127.;
    }
    let input = Tensor::from_vec(input, (1, probes.len(), COLS), &device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let output = projection.forward(&input).unwrap();
    assert!(output.device().is_cuda());
    let output = output
        .reshape((probes.len(), ROWS))
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    let mut source_error = 0f32;
    let mut dot_error = 0f32;
    for (r, row) in source.iter().enumerate() {
        for (b, values) in row.chunks_exact(format.block_size()).enumerate() {
            let start = (r * COLS / format.block_size() + b) * format.type_size();
            let block = &bytes[start..start + format.type_size()];
            let max = values.iter().map(|n| n.abs()).fold(0., f32::max);
            for (i, &x) in values.iter().enumerate() {
                let (value, depth, budget) = if format == GgmlDType::Q4K {
                    let (value, step, min) = q4_cell(block, i);
                    // Nearest 4-bit code within the emitted affine range. For
                    // clipping, include exactly the source's distance outside it.
                    let clip = (-min - x).max(x - (15. * step - min)).max(0.);
                    (value, min + 15. * step, clip + 0.5 * step)
                } else {
                    let step = half_value(block);
                    let exact = max / 127.;
                    (
                        step * f32::from(block[2 + i] as i8),
                        127. * step,
                        0.5 * exact + 127. * (step - exact).abs(),
                    )
                };
                let error = (value - x).abs();
                assert!(
                    error <= budget + 8. * f32::EPSILON * (depth + x.abs()),
                    "{format:?} row={r} block={b} cell={i} error={error} budget={budget}"
                );
                source_error = source_error.max(error);
                if let Some(p) = probes
                    .iter()
                    .position(|&col| col == b * format.block_size() + i)
                {
                    let expected = 127.0 * value;
                    // Up to eight FP32 arithmetic roundings of the affine
                    // terms and the final BF16 round (one unit roundoff).
                    let bound = 127. * depth * 8. * f32::EPSILON
                        + expected.abs() / 256.
                        + f32::MIN_POSITIVE;
                    let error = (output[p][r] - expected).abs();
                    assert!(
                        error <= bound,
                        "{format:?} row={r} probe={p}: {error} > {bound}"
                    );
                    dot_error = dot_error.max(error);
                }
            }
        }
    }
    println!("\nSenseNova {format:?} exact source encoding: {ROWS} rows of {KEY}, max source error {source_error}, max CUDA-vs-FP32 dot error {dot_error}");
}
