//! Shape-stable BF16 convolution leaves for the YuE2 VAE on CUDA.
//!
//! One output element owns its complete, fixed-order FP32 reduction. The input, folded weight,
//! output, and the surrounding VAE layers remain BF16. Unlike a GEMM chosen for the complete song
//! length, equal contributing BF16 input/weight bits therefore produce equal interior output bits
//! in a full decode and a halo tile. Compilation uses the workspace's shared NVRTC seam.

#[cfg(any(feature = "cuda", test))]
use candle_core::{DType, Tensor};

use crate::nvrtc::KernelSource;

/// The two kernels are compiled and cached together for each CUDA ordinal.
pub const YUE2_STABLE_CONV_SRC: KernelSource = KernelSource {
    name: "candle_quant_kernels_yue2_stable_conv_v1",
    src: include_str!("yue2_stable_conv.cu"),
    // Only software BF16 conversion and FP32 FMA are used; no toolkit header is required.
    cc_floor: (7, 0),
};

#[cfg(any(feature = "cuda", test))]
const THREADS: usize = 256;

#[cfg(any(feature = "cuda", test))]
macro_rules! ensure {
    ($condition:expr, $message:literal) => {
        if !$condition {
            candle_core::bail!($message)
        }
    };
}

#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Plan {
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    in_length: usize,
    out_length: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
}

#[cfg(feature = "cuda")]
fn plan(
    input: &Tensor,
    weight: &Tensor,
    stride: usize,
    padding: usize,
    dilation: usize,
    transposed: bool,
) -> candle_core::Result<Plan> {
    ensure!(
        input.dtype() == DType::BF16 && weight.dtype() == DType::BF16,
        "YuE2 stable convolution requires BF16 input and weight"
    );
    ensure!(
        input.device().is_cuda() && input.device().same_device(weight.device()),
        "YuE2 stable convolution requires one CUDA device"
    );
    let input_shape: [usize; 3] = input.dims().try_into().map_err(|_| {
        candle_core::Error::Msg("YuE2 stable convolution input must be [B,C,L]".into())
    })?;
    let weight_shape: [usize; 3] = weight.dims().try_into().map_err(|_| {
        candle_core::Error::Msg("YuE2 stable convolution weight must have rank three".into())
    })?;
    geometry(
        input_shape,
        weight_shape,
        stride,
        padding,
        dilation,
        transposed,
    )
}

#[cfg(any(feature = "cuda", test))]
fn geometry(
    [batch, in_channels, in_length]: [usize; 3],
    [first, second, kernel]: [usize; 3],
    stride: usize,
    padding: usize,
    dilation: usize,
    transposed: bool,
) -> candle_core::Result<Plan> {
    let out_channels = if transposed {
        ensure!(first == in_channels, "YuE2 ConvT input channels differ");
        second
    } else {
        ensure!(second == in_channels, "YuE2 Conv input channels differ");
        first
    };
    ensure!(
        batch > 0 && in_channels > 0 && in_length > 0 && out_channels > 0 && kernel > 0,
        "YuE2 stable convolution has an empty dimension"
    );
    ensure!(
        stride > 0 && dilation > 0,
        "YuE2 convolution stride/dilation is zero"
    );
    let out_length = if transposed {
        ensure!(
            dilation == 1 && padding == 0,
            "YuE2 ConvT requires raw unpadded output"
        );
        in_length
            .checked_sub(1)
            .and_then(|n| n.checked_mul(stride))
            .and_then(|n| n.checked_add(kernel))
    } else {
        let effective = kernel.checked_sub(1).and_then(|n| n.checked_mul(dilation));
        padding
            .checked_mul(2)
            .and_then(|p| in_length.checked_add(p))
            .and_then(|n| n.checked_sub(effective?.checked_add(1)?))
            .map(|n| n / stride + 1)
    }
    .ok_or_else(|| {
        candle_core::Error::Msg("YuE2 convolution output length overflow/underflow".into())
    })?;
    let count = batch
        .checked_mul(out_channels)
        .and_then(|n| n.checked_mul(out_length))
        .ok_or_else(|| candle_core::Error::Msg("YuE2 convolution output size overflow".into()))?;
    ensure!(
        [
            batch,
            in_channels,
            out_channels,
            in_length,
            out_length,
            kernel,
            stride,
            padding,
            dilation
        ]
        .into_iter()
        .all(|n| i32::try_from(n).is_ok())
            && count.div_ceil(THREADS) <= u32::MAX as usize,
        "YuE2 convolution geometry exceeds CUDA launch/index bounds"
    );
    Ok(Plan {
        batch,
        in_channels,
        out_channels,
        in_length,
        out_length,
        kernel,
        stride,
        padding,
        dilation,
    })
}

/// Compute a BF16 Conv1d. Bias stays with the caller so its existing BF16 broadcast semantics
/// are unchanged.
#[cfg(feature = "cuda")]
pub fn conv1d(
    input: &Tensor,
    weight: &Tensor,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> candle_core::Result<Tensor> {
    let plan = plan(input, weight, stride, padding, dilation, false)?;
    cuda::launch(input, weight, plan, false)
}

/// Compute the raw unpadded BF16 ConvTranspose1d. The caller retains torch's crop and BF16 bias.
#[cfg(feature = "cuda")]
pub fn conv_transpose1d(
    input: &Tensor,
    weight: &Tensor,
    stride: usize,
) -> candle_core::Result<Tensor> {
    let plan = plan(input, weight, stride, 0, 1, true)?;
    cuda::launch(input, weight, plan, true)
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use candle_core::cuda_backend::cudarc;
    use candle_core::op::BackpropOp;
    use candle_core::{CudaStorage, Device, Shape, Storage};
    use cudarc::driver::{LaunchConfig, PushKernelArg};

    pub(super) fn launch(
        input: &Tensor,
        weight: &Tensor,
        p: Plan,
        transposed: bool,
    ) -> candle_core::Result<Tensor> {
        let Device::Cuda(dev) = input.device() else {
            unreachable!("plan checked CUDA")
        };
        let symbol = if transposed {
            "yue2_conv_transpose1d_bf16"
        } else {
            "yue2_conv1d_bf16"
        };
        let func = YUE2_STABLE_CONV_SRC
            .compiled(dev)
            .and_then(|module| module.function(symbol))
            .map_err(|error| {
                candle_core::Error::Msg(format!("YuE2 stable convolution: {error}"))
            })?;
        let input = input.contiguous()?;
        let weight = weight.contiguous()?;
        let count = p.batch * p.out_channels * p.out_length;
        let mut output = unsafe { dev.alloc::<half::bf16>(count) }?;
        let cfg = LaunchConfig {
            grid_dim: (count.div_ceil(THREADS) as u32, 1, 1),
            block_dim: (THREADS as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let (xs, xl) = input.storage_and_layout();
        let (ws, wl) = weight.storage_and_layout();
        let Storage::Cuda(xs) = &*xs else {
            unreachable!("plan checked CUDA")
        };
        let Storage::Cuda(ws) = &*ws else {
            unreachable!("plan checked CUDA")
        };
        let xs = xs.as_cuda_slice::<half::bf16>()?.slice(xl.start_offset()..);
        let ws = ws.as_cuda_slice::<half::bf16>()?.slice(wl.start_offset()..);
        let [b, ci, co, lin, lout, k, stride, padding, dilation] = [
            p.batch,
            p.in_channels,
            p.out_channels,
            p.in_length,
            p.out_length,
            p.kernel,
            p.stride,
            p.padding,
            p.dilation,
        ]
        .map(|n| n as i32);
        let stream = dev.cuda_stream();
        let mut launch = stream.launch_builder(&func);
        launch
            .arg(&xs)
            .arg(&ws)
            .arg(&mut output)
            .arg(&b)
            .arg(&ci)
            .arg(&co)
            .arg(&lin)
            .arg(&lout)
            .arg(&k)
            .arg(&stride);
        if !transposed {
            launch.arg(&padding).arg(&dilation);
        }
        unsafe { launch.launch(cfg) }.map_err(|error| {
            candle_core::Error::Cuda(format!("YuE2 stable convolution launch: {error:?}").into())
        })?;
        Ok(Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(output, dev.clone())),
            Shape::from((p.batch, p.out_channels, p.out_length)),
            BackpropOp::none(),
            false,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn geometry_covers_both_vae_leaves_and_rejects_invalid_shapes() {
        let conv = geometry([1, 4, 32], [3, 4, 7], 2, 3, 3, false).unwrap();
        assert_eq!(conv.out_length, 10);
        let transposed = geometry([1, 4, 32], [4, 3, 12], 6, 0, 1, true).unwrap();
        assert_eq!(transposed.out_length, 198);
        assert!(geometry([1, 4, 32], [5, 3, 12], 6, 0, 1, true).is_err());
        assert!(geometry([1, 4, 32], [3, 4, 7], 0, 3, 1, false).is_err());
        assert!(geometry([1, 4, 32], [3, 4, 7], 2, 0, 9, false).is_err());
        assert!(geometry([1, 4, usize::MAX], [4, 3, 12], 6, 0, 1, true).is_err());
        let x = Tensor::zeros((1, 4, 32), DType::BF16, &Device::Cpu).unwrap();
        let w = Tensor::zeros((4, 3, 12), DType::BF16, &Device::Cpu).unwrap();
        assert_eq!(x.dims(), &[1, 4, 32]);
        assert_eq!(w.dims(), &[4, 3, 12]);
        assert!(YUE2_STABLE_CONV_SRC.src.contains("__fmaf_rn"));
        assert!(YUE2_STABLE_CONV_SRC
            .src
            .contains("yue2_conv_transpose1d_bf16"));
    }

    #[test]
    fn a_transposed_window_keeps_the_same_ordered_contributors_as_full_decode() {
        // The standard decoder's first upsampling leaf is k=12, stride=6. A nonzero latent
        // window origin must preserve both its tap residue and the input index of each tap.
        let left = 16;
        let stride = 6;
        let kernel = 12;
        for local_raw in 11..174 {
            let global_raw = left * stride + local_raw;
            let window: Vec<_> = (local_raw % stride..kernel)
                .step_by(stride)
                .filter_map(|k| local_raw.checked_sub(k).map(|source| (k, source / stride)))
                .collect();
            let full: Vec<_> = (global_raw % stride..kernel)
                .step_by(stride)
                .filter_map(|k| global_raw.checked_sub(k).map(|source| (k, source / stride)))
                .collect();
            assert_eq!(window.len(), full.len());
            for ((tile_tap, tile_i), (full_tap, full_i)) in window.into_iter().zip(full) {
                assert_eq!(tile_tap, full_tap);
                assert_eq!(tile_i + left, full_i);
            }
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use super::*;
    use candle_core::Device;

    #[test]
    #[ignore = "requires an owned CUDA device; no real weights"]
    fn bf16_conv_and_transpose_match_same_input_prefix_across_lengths() {
        let device = Device::new_cuda(0).unwrap();
        let simple = Tensor::from_vec(
            [1.0, 2.0, 3.0, 4.0].map(half::bf16::from_f32).to_vec(),
            (1, 1, 4),
            &device,
        )
        .unwrap();
        let simple_weight = Tensor::from_vec(
            [1.0, 2.0, 3.0].map(half::bf16::from_f32).to_vec(),
            (1, 1, 3),
            &device,
        )
        .unwrap();
        let actual = conv1d(&simple, &simple_weight, 1, 1, 1)
            .unwrap()
            .to_vec3::<half::bf16>()
            .unwrap();
        assert_eq!(
            actual[0][0].iter().map(|v| v.to_f32()).collect::<Vec<_>>(),
            [8.0, 14.0, 20.0, 11.0]
        );
        let simple_t = simple.narrow(2, 0, 3).unwrap();
        let simple_t_weight = Tensor::from_vec(
            [1.0, 2.0, 3.0, 4.0].map(half::bf16::from_f32).to_vec(),
            (1, 1, 4),
            &device,
        )
        .unwrap();
        let actual = conv_transpose1d(&simple_t, &simple_t_weight, 2)
            .unwrap()
            .to_vec3::<half::bf16>()
            .unwrap();
        assert_eq!(
            actual[0][0].iter().map(|v| v.to_f32()).collect::<Vec<_>>(),
            [1.0, 2.0, 5.0, 8.0, 9.0, 14.0, 9.0, 12.0]
        );

        let values: Vec<_> = (0..24)
            .map(|n| half::bf16::from_f32((n as f32 - 11.0) / 17.0))
            .collect();
        let x = Tensor::from_vec(values, (1, 2, 12), &device).unwrap();
        let tile = x.narrow(2, 0, 8).unwrap();
        let conv_weight = Tensor::from_vec(
            (0..18)
                .map(|n| half::bf16::from_f32((n as f32 - 8.0) / 31.0))
                .collect(),
            (3, 2, 3),
            &device,
        )
        .unwrap();
        let full = conv1d(&x, &conv_weight, 1, 1, 1).unwrap();
        let short = conv1d(&tile, &conv_weight, 1, 1, 1).unwrap();
        assert_eq!(full.dtype(), DType::BF16);
        let full = full.to_vec3::<half::bf16>().unwrap();
        let short = short.to_vec3::<half::bf16>().unwrap();
        for channel in 0..3 {
            for t in 0..7 {
                assert_eq!(
                    full[0][channel][t].to_bits(),
                    short[0][channel][t].to_bits()
                );
            }
        }

        let transpose_weight = Tensor::from_vec(
            (0..36)
                .map(|n| half::bf16::from_f32((n as f32 - 17.0) / 43.0))
                .collect(),
            (2, 3, 6),
            &device,
        )
        .unwrap();
        let full = conv_transpose1d(&x, &transpose_weight, 3).unwrap();
        let short = conv_transpose1d(&tile, &transpose_weight, 3).unwrap();
        assert_eq!(full.dtype(), DType::BF16);
        let full = full.to_vec3::<half::bf16>().unwrap();
        let short = short.to_vec3::<half::bf16>().unwrap();
        for channel in 0..3 {
            for t in 0..24 {
                assert_eq!(
                    full[0][channel][t].to_bits(),
                    short[0][channel][t].to_bits()
                );
            }
        }
    }
}
