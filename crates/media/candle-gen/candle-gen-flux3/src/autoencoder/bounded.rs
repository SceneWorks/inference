//! Exact operation-level launch bounds, not independently normalized spatial VAE tiles.
//!
//! Candle CUDA maps use u32 element counts/relative indices. A released final-upsample tensor
//! [1,256,4096,4096] has exactly 2^32 elements. Keep every kernel/copy bounded while retaining
//! usize global offsets and full-image GroupNorm statistics.

use std::ops::Range;

use candle_gen::candle_core::{Module, Tensor};
use candle_gen::candle_nn::{Conv2d, GroupNorm};
use candle_gen::{CandleError as Error, Result};

const WORK_ELEMENTS: usize = 1 << 28;
// Leave ample headroom for rounded launch/grid-stride increments below u32::MAX.
const MAX_LAUNCH_ELEMENTS: usize = 1 << 30;

fn invalid(message: &str) -> Error {
    Error::Unsupported(format!("flux3 autoencoder launch plan: {message}"))
}

fn product(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(1usize, |n, &value| {
        if value == 0 {
            return Err(invalid("empty geometry"));
        }
        n.checked_mul(value)
            .ok_or_else(|| invalid("geometry overflow"))
    })
}

fn relative_last(dims: &[usize], strides: &[usize]) -> Result<usize> {
    dims.iter()
        .zip(strides)
        .try_fold(0usize, |n, (&dim, &stride)| {
            n.checked_add(
                dim.saturating_sub(1)
                    .checked_mul(stride)
                    .ok_or_else(|| invalid("stride overflow"))?,
            )
            .ok_or_else(|| invalid("stride overflow"))
        })
}

fn nchw_offset(shape: [usize; 4], batch: usize, channel: usize, row: usize) -> Result<usize> {
    product(&shape)?;
    let [b, c, h, w] = shape;
    if batch >= b || channel >= c || row >= h {
        return Err(invalid("NCHW copy coordinate is out of range"));
    }
    batch
        .checked_mul(c)
        .and_then(|v| v.checked_add(channel))
        .and_then(|v| v.checked_mul(h))
        .and_then(|v| v.checked_add(row))
        .and_then(|v| v.checked_mul(w))
        .ok_or_else(|| invalid("NCHW destination offset overflow"))
}

fn flat_range(total: usize, offset: usize, len: usize) -> Result<Range<usize>> {
    if len > MAX_LAUNCH_ELEMENTS {
        return Err(invalid("unbounded flat copy"));
    }
    let end = offset
        .checked_add(len)
        .ok_or_else(|| invalid("copy offset overflow"))?;
    if end > total {
        return Err(invalid("copy exceeds destination"));
    }
    Ok(offset..end)
}

/// Narrow the destination BEFORE copying: d1=1 and d2=len stay bounded, whereas the global
/// destination start offset is applied by Candle to its pointer as usize, not a u32 kernel pitch.
fn copy_flat(destination: &Tensor, offset: usize, source: &Tensor) -> Result<()> {
    if !source.is_contiguous() {
        return Err(invalid("copy source is not contiguous"));
    }
    let range = flat_range(destination.elem_count(), offset, source.elem_count())?;
    if range.is_empty() {
        return Ok(());
    }
    let dst = destination
        .flatten_all()?
        .narrow(0, range.start, range.len())?;
    dst.slice_set(&source.flatten_all()?, 0, 0)?;
    Ok(())
}

/// Materialize a view only when both its launch count and relative strided address fit. For
/// a larger layout, copy individual NCHW planes with 64-bit pointer offsets instead.
fn contiguous_nchw(x: &Tensor) -> Result<Tensor> {
    if x.is_contiguous() {
        return Ok(x.clone());
    }
    if x.elem_count() <= MAX_LAUNCH_ELEMENTS
        && relative_last(x.dims(), x.stride())? <= u32::MAX as usize
    {
        return Ok(x.contiguous()?);
    }
    let (b, c, h, w) = x.dims4()?;
    let plane = product(&[h, w])?;
    if plane > MAX_LAUNCH_ELEMENTS {
        return Err(invalid("a single spatial plane exceeds the launch bound"));
    }
    let output = Tensor::zeros((b, c, h, w), x.dtype(), x.device())?;
    for batch in 0..b {
        for channel in 0..c {
            let view = x.narrow(0, batch, 1)?.narrow(1, channel, 1)?;
            if relative_last(view.dims(), view.stride())? > u32::MAX as usize {
                return Err(invalid(
                    "a spatial plane has an unsupported relative stride span",
                ));
            }
            copy_flat(
                &output,
                nchw_offset([b, c, h, w], batch, channel, 0)?,
                &view.contiguous()?,
            )?;
        }
    }
    Ok(output)
}

fn channel_step(channels: usize, height: usize, width: usize, limit: usize) -> Result<usize> {
    let plane = product(&[height, width])?;
    let limit = limit.min(MAX_LAUNCH_ELEMENTS);
    if limit < plane || channels == 0 {
        return Err(invalid("one spatial plane does not fit the working bound"));
    }
    Ok((limit / plane).min(channels))
}

pub(super) fn silu(x: &Tensor) -> Result<Tensor> {
    unary_with_limit(x, WORK_ELEMENTS, |x| {
        Ok(candle_gen::candle_nn::ops::silu(x)?)
    })
}

fn unary_with_limit(
    x: &Tensor,
    limit: usize,
    apply: impl Fn(&Tensor) -> Result<Tensor>,
) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    if x.elem_count() <= limit.min(MAX_LAUNCH_ELEMENTS) {
        return apply(&contiguous_nchw(x)?);
    }
    let step = channel_step(c, h, w, limit)?;
    let output = Tensor::zeros((b, c, h, w), x.dtype(), x.device())?;
    for batch in 0..b {
        for start in (0..c).step_by(step) {
            let len = step.min(c - start);
            let part = contiguous_nchw(&x.narrow(0, batch, 1)?.narrow(1, start, len)?)?;
            let value = contiguous_nchw(&apply(&part)?)?;
            copy_flat(&output, nchw_offset([b, c, h, w], batch, start, 0)?, &value)?;
        }
    }
    Ok(output)
}

pub(super) fn add(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    add_with_limit(lhs, rhs, WORK_ELEMENTS)
}

fn add_with_limit(lhs: &Tensor, rhs: &Tensor, limit: usize) -> Result<Tensor> {
    if lhs.dims() != rhs.dims() {
        return Err(invalid("residual shapes differ"));
    }
    let (b, c, h, w) = lhs.dims4()?;
    if lhs.elem_count() <= limit.min(MAX_LAUNCH_ELEMENTS) {
        return Ok(contiguous_nchw(lhs)?.add(&contiguous_nchw(rhs)?)?);
    }
    let step = channel_step(c, h, w, limit)?;
    let output = Tensor::zeros((b, c, h, w), lhs.dtype(), lhs.device())?;
    for batch in 0..b {
        for start in (0..c).step_by(step) {
            let len = step.min(c - start);
            let left = contiguous_nchw(&lhs.narrow(0, batch, 1)?.narrow(1, start, len)?)?;
            let right = contiguous_nchw(&rhs.narrow(0, batch, 1)?.narrow(1, start, len)?)?;
            copy_flat(
                &output,
                nchw_offset([b, c, h, w], batch, start, 0)?,
                &left.add(&right)?,
            )?;
        }
    }
    Ok(output)
}

pub(super) fn nearest2d(x: &Tensor, height: usize, width: usize) -> Result<Tensor> {
    nearest_with_limit(x, height, width, WORK_ELEMENTS)
}

fn nearest_with_limit(x: &Tensor, height: usize, width: usize, limit: usize) -> Result<Tensor> {
    let (b, c, _, _) = x.dims4()?;
    product(&[b, c, height, width])?;
    let step = channel_step(c, height, width, limit)?;
    if b == 1 && step == c {
        return Ok(contiguous_nchw(x)?.upsample_nearest2d(height, width)?);
    }
    let output = Tensor::zeros((b, c, height, width), x.dtype(), x.device())?;
    for batch in 0..b {
        for start in (0..c).step_by(step) {
            let len = step.min(c - start);
            let input = contiguous_nchw(&x.narrow(0, batch, 1)?.narrow(1, start, len)?)?;
            let value = input.upsample_nearest2d(height, width)?;
            copy_flat(
                &output,
                nchw_offset([b, c, height, width], batch, start, 0)?,
                &value,
            )?;
        }
    }
    Ok(output)
}

/// The four sides are [top,bottom,left,right]. Asymmetric encoder downsampling uses [0,1,0,1]
/// without ever constructing the full padded high-resolution activation.
#[derive(Clone, Copy)]
struct Padding([usize; 4]);

struct ConvPlan {
    height: usize,
    width: usize,
    rows: usize,
    span: usize,
}

fn conv_plan(shape: [usize; 4], conv: &Conv2d, padding: Padding, limit: usize) -> Result<ConvPlan> {
    product(&shape)?;
    let [_, channels, height, width] = shape;
    let (out_channels, _, kh, kw) = conv.weight().dims4()?;
    let cfg = conv.config();
    if cfg.stride == 0 || kh == 0 || kw == 0 {
        return Err(invalid("invalid convolution geometry"));
    }
    let span = cfg
        .dilation
        .checked_mul(kh - 1)
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| invalid("kernel span overflow"))?;
    let width_span = cfg
        .dilation
        .checked_mul(kw - 1)
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| invalid("kernel span overflow"))?;
    let [top, bottom, left, right] = padding.0;
    let ph = height
        .checked_add(top)
        .and_then(|v| v.checked_add(bottom))
        .ok_or_else(|| invalid("padding overflow"))?;
    let pw = width
        .checked_add(left)
        .and_then(|v| v.checked_add(right))
        .ok_or_else(|| invalid("padding overflow"))?;
    if ph < span || pw < width_span {
        return Err(invalid("kernel exceeds padded input"));
    }
    let out_h = (ph - span) / cfg.stride + 1;
    let out_w = (pw - width_span) / cfg.stride + 1;
    let limit = limit.min(MAX_LAUNCH_ELEMENTS);
    let im_row = product(&[channels, out_w, kh, kw])?;
    let out_row = product(&[out_channels, out_w])?;
    let input_row = product(&[channels, pw])?;
    let max_input_rows = limit / input_row;
    if max_input_rows < span {
        return Err(invalid(
            "one convolution receptive field exceeds the launch bound",
        ));
    }
    let rows = out_h
        .min(candle_gen::CONV_IM2COL_BUDGET.min(limit) / im_row)
        .min(limit / out_row)
        .min((max_input_rows - span) / cfg.stride + 1);
    if rows == 0 {
        return Err(invalid(
            "one output row exceeds the convolution launch bound",
        ));
    }
    Ok(ConvPlan {
        height: out_h,
        width: out_w,
        rows,
        span,
    })
}

/// All AE convolution fields use this type, including shortcuts and attention projections;
/// an ordinary `.forward()` cannot accidentally bypass the large-activation launch plan.
pub(super) struct BoundedConv2d {
    inner: Conv2d,
}

impl BoundedConv2d {
    pub(super) fn new(
        weight: Tensor,
        bias: Option<Tensor>,
        config: candle_gen::candle_nn::Conv2dConfig,
    ) -> Self {
        Self {
            inner: Conv2d::new(weight, bias, config),
        }
    }
    pub(super) fn weight(&self) -> &Tensor {
        self.inner.weight()
    }
    pub(super) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let p = self.inner.config().padding;
        conv_with_limit(x, &self.inner, Padding([p, p, p, p]), WORK_ELEMENTS)
    }
}

pub(super) fn conv2d(x: &Tensor, conv: &BoundedConv2d) -> Result<Tensor> {
    conv.forward(x)
}

pub(super) fn downsample(x: &Tensor, conv: &BoundedConv2d) -> Result<Tensor> {
    conv_with_limit(x, &conv.inner, Padding([0, 1, 0, 1]), WORK_ELEMENTS)
}

fn conv_with_limit(x: &Tensor, conv: &Conv2d, padding: Padding, limit: usize) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    let plan = conv_plan([b, c, h, w], conv, padding, limit)?;
    let out_c = conv.weight().dim(0)?;
    product(&[b, out_c, plan.height, plan.width])?;
    let cfg = conv.config();
    if b == 1
        && plan.rows == plan.height
        && padding.0 == [cfg.padding; 4]
        && product(&[
            c,
            h + padding.0[0] + padding.0[1],
            w + padding.0[2] + padding.0[3],
        ])? <= limit.min(MAX_LAUNCH_ELEMENTS)
    {
        return Ok(conv.forward(&contiguous_nchw(x)?)?);
    }
    let output = Tensor::zeros((b, out_c, plan.height, plan.width), x.dtype(), x.device())?;
    let [top, _, left, right] = padding.0;
    for batch in 0..b {
        for row in (0..plan.height).step_by(plan.rows) {
            let rows = plan.rows.min(plan.height - row);
            let virtual_start = row * cfg.stride;
            let virtual_end = virtual_start + (rows - 1) * cfg.stride + plan.span;
            let begin = virtual_start.saturating_sub(top).min(h);
            let end = virtual_end.saturating_sub(top).min(h);
            let pad_top = top.saturating_sub(virtual_start);
            let pad_bottom = virtual_end.saturating_sub(top + h);
            if end <= begin {
                return Err(invalid("convolution tile contains no input"));
            }
            let input = contiguous_nchw(&x.narrow(0, batch, 1)?.narrow(2, begin, end - begin)?)?;
            let input = input
                .pad_with_zeros(2, pad_top, pad_bottom)?
                .pad_with_zeros(3, left, right)?;
            if input.elem_count() > limit.min(MAX_LAUNCH_ELEMENTS) {
                return Err(invalid("padded tile exceeds planned bound"));
            }
            let value = input.conv2d_with_algo(
                conv.weight(),
                0,
                cfg.stride,
                cfg.dilation,
                cfg.groups,
                cfg.cudnn_fwd_algo,
            )?;
            let value = match conv.bias() {
                Some(bias) => value.broadcast_add(&bias.reshape((1, out_c, 1, 1))?)?,
                None => value,
            };
            if value.dims() != [1, out_c, rows, plan.width] {
                return Err(invalid("convolution tile output differs from plan"));
            }
            // Do not spatial-cat tiles: copy a bounded contiguous row segment of each channel.
            for channel in 0..out_c {
                let plane = contiguous_nchw(&value.narrow(1, channel, 1)?)?;
                let offset = nchw_offset([b, out_c, plan.height, plan.width], batch, channel, row)?;
                copy_flat(&output, offset, &plane)?;
            }
        }
    }
    Ok(output)
}

/// Training-only bounded dtype conversion; never launches across the large-tensor bound.
pub(super) fn training_cast(x: &Tensor, dtype: candle_gen::candle_core::DType) -> Result<Tensor> {
    unary_with_limit(x, WORK_ELEMENTS, |part| Ok(part.to_dtype(dtype)?))
}

/// Training-only source opmath: retained parameters are selected precision. Convert only
/// this operation's rounded operands, accumulate convolution AND bias in F32, then write
/// selected precision ONCE. CPU half convolution is explicitly emulated; this is not a
/// claim of a CUDA half/tensor-core kernel. Both paths reuse the safe spatial planner.
pub(super) fn training_conv_f32_opmath(
    x: &Tensor,
    conv: &BoundedConv2d,
    asymmetric_downsample: bool,
) -> Result<Tensor> {
    use candle_gen::candle_core::DType;
    let inner = &conv.inner;
    let float_conv = Conv2d::new(
        inner.weight().to_dtype(DType::F32)?,
        inner.bias().map(|b| b.to_dtype(DType::F32)).transpose()?,
        *inner.config(),
    );
    let p = inner.config().padding;
    let padding = if asymmetric_downsample {
        Padding([0, 1, 0, 1])
    } else {
        Padding([p, p, p, p])
    };
    let value = conv_with_limit(
        &training_cast(x, DType::F32)?,
        &float_conv,
        padding,
        WORK_ELEMENTS,
    )?;
    training_cast(&value, x.dtype())
}

/// Retain affine tensors so a large norm can be split at WHOLE original group boundaries.
/// Splitting spatial tiles and recomputing their norms would change the image and is not used.
pub(super) struct BoundedGroupNorm {
    inner: GroupNorm,
    weight: Tensor,
    bias: Tensor,
    channels: usize,
    groups: usize,
    eps: f64,
}

fn group_channel_step(
    channels: usize,
    groups: usize,
    h: usize,
    w: usize,
    limit: usize,
) -> Result<usize> {
    if groups == 0 || channels == 0 || !channels.is_multiple_of(groups) {
        return Err(invalid("invalid GroupNorm groups"));
    }
    let per_group = product(&[channels / groups, h, w])?;
    if per_group > MAX_LAUNCH_ELEMENTS {
        return Err(invalid(
            "a complete GroupNorm group exceeds the kernel limit",
        ));
    }
    let groups_per_chunk = (limit.max(per_group).min(MAX_LAUNCH_ELEMENTS) / per_group).min(groups);
    Ok(groups_per_chunk * (channels / groups))
}

impl BoundedGroupNorm {
    pub(super) fn new(
        weight: Tensor,
        bias: Tensor,
        channels: usize,
        groups: usize,
        eps: f64,
    ) -> Result<Self> {
        let inner = GroupNorm::new(weight.clone(), bias.clone(), channels, groups, eps)?;
        Ok(Self {
            inner,
            weight,
            bias,
            channels,
            groups,
            eps,
        })
    }

    pub(super) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_with_limit(x, WORK_ELEMENTS)
    }

    /// Training-only F32 statistics/affine opmath with one selected-dtype output write.
    /// Whole-group splitting and usize assembly are exactly the inference planner's.
    pub(super) fn training_forward_f32_opmath(&self, x: &Tensor) -> Result<Tensor> {
        use candle_gen::candle_core::DType;
        let norm = Self::new(
            self.weight.to_dtype(DType::F32)?,
            self.bias.to_dtype(DType::F32)?,
            self.channels,
            self.groups,
            self.eps,
        )?;
        let output = norm.forward(&training_cast(x, DType::F32)?)?;
        training_cast(&output, x.dtype())
    }

    fn forward_with_limit(&self, x: &Tensor, limit: usize) -> Result<Tensor> {
        let (b, c, h, w) = x.dims4()?;
        if c != self.channels {
            return Err(invalid("GroupNorm channel mismatch"));
        }
        if x.elem_count() <= limit.min(MAX_LAUNCH_ELEMENTS) {
            return Ok(self.inner.forward(&contiguous_nchw(x)?)?);
        }
        let step = group_channel_step(c, self.groups, h, w, limit)?;
        let per_group = c / self.groups;
        let output = Tensor::zeros((b, c, h, w), x.dtype(), x.device())?;
        for batch in 0..b {
            for start in (0..c).step_by(step) {
                let len = step.min(c - start);
                let norm = GroupNorm::new(
                    self.weight.narrow(0, start, len)?,
                    self.bias.narrow(0, start, len)?,
                    len,
                    len / per_group,
                    self.eps,
                )?;
                let part = contiguous_nchw(&x.narrow(0, batch, 1)?.narrow(1, start, len)?)?;
                copy_flat(
                    &output,
                    nchw_offset([b, c, h, w], batch, start, 0)?,
                    &norm.forward(&part)?,
                )?;
            }
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::{DType, Device};
    use candle_gen::candle_nn::Conv2dConfig;

    #[test]
    fn training_bf16_conv_cpu_emulation_fuses_bias_before_write_rounding() {
        let low = |v: Vec<f32>, shape| {
            Tensor::from_vec(v, shape, &Device::Cpu)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
        };
        let x = low(vec![1.0078125], (1, 1, 1, 1));
        let weight = low(vec![1.0078125], (1, 1, 1, 1));
        let bias = Tensor::from_vec(vec![0.00390625f32], 1, &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let conv = BoundedConv2d::new(weight, Some(bias), Conv2dConfig::default());
        let got = training_conv_f32_opmath(&x, &conv, false).unwrap();
        assert_eq!(got.dtype(), DType::BF16);
        assert_eq!(
            got.to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![1.0234375]
        );
        assert_eq!(conv.weight().dtype(), DType::BF16);
    }

    fn values(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }
    fn close(lhs: &Tensor, rhs: &Tensor) {
        assert_eq!(lhs.dims(), rhs.dims());
        let delta = values(lhs)
            .into_iter()
            .zip(values(rhs))
            .map(|(a, b)| {
                assert!(
                    a.is_finite() && b.is_finite(),
                    "non-finite bounded operation result"
                );
                (a - b).abs()
            })
            .fold(0f32, f32::max);
        assert!(delta < 1e-4, "forced bounded path differs by {delta}");
    }

    #[test]
    #[should_panic(expected = "non-finite bounded operation result")]
    fn comparison_rejects_nan_instead_of_treating_it_as_zero_error() {
        let lhs = Tensor::from_vec(vec![f32::NAN], (1,), &Device::Cpu).unwrap();
        let rhs = Tensor::zeros((1,), DType::F32, &Device::Cpu).unwrap();
        close(&lhs, &rhs);
    }

    #[test]
    fn released_four_k_boundary_is_planned_without_allocating_the_activation() {
        let elements = product(&[1, 256, 4096, 4096]).unwrap();
        assert_eq!(elements as u64, 1u64 << 32);
        let step = channel_step(256, 4096, 4096, WORK_ELEMENTS).unwrap();
        assert_eq!(step, 16);
        assert_eq!(
            group_channel_step(256, 32, 4096, 4096, WORK_ELEMENTS).unwrap(),
            16
        );
        assert_eq!(
            group_channel_step(128, 32, 4096, 4096, WORK_ELEMENTS).unwrap(),
            16
        );
        let mut copied = 0;
        for channel in (0..256).step_by(step) {
            let len = product(&[step, 4096, 4096]).unwrap();
            let offset = nchw_offset([1, 256, 4096, 4096], 0, channel, 0).unwrap();
            let range = flat_range(elements, offset, len).unwrap();
            assert_eq!(range.start, copied);
            assert!(range.len() <= WORK_ELEMENTS);
            copied = range.end;
        }
        assert_eq!(copied, elements);
        assert_eq!(
            nchw_offset([2, 256, 4096, 4096], 1, 0, 0).unwrap(),
            elements
        );
        assert!(flat_range(elements, elements, 0).unwrap().is_empty());
        assert!(flat_range(elements, elements, 1).is_err());
        assert!(flat_range(elements, usize::MAX, 2).is_err());
        assert!(flat_range(elements, 0, MAX_LAUNCH_ELEMENTS + 1).is_err());
        assert!(nchw_offset([usize::MAX, 256, 4096, 4096], 0, 0, 0).is_err());
        assert!(
            relative_last(&[1, 256, 7, 4096], &[elements, 4096 * 4096, 4096, 1]).unwrap()
                <= u32::MAX as usize
        );
        assert!(
            relative_last(&[1, 512, 7, 4096], &[elements * 2, 4096 * 4096, 4096, 1]).unwrap()
                > u32::MAX as usize
        );
        assert_eq!(
            flat_range(elements * 2, elements + 17, 31).unwrap().start,
            elements + 17
        );
        // Real 256->128 final-resolution shortcut and a 3x3 follow-up, no giant tensor allocation.
        for kernel in [1, 3] {
            let weight =
                Tensor::zeros((128, 256, kernel, kernel), DType::F32, &Device::Cpu).unwrap();
            let conv = Conv2d::new(
                weight,
                None,
                Conv2dConfig {
                    padding: kernel / 2,
                    ..Default::default()
                },
            );
            let plan = conv_plan(
                [1, 256, 4096, 4096],
                &conv,
                Padding([kernel / 2; 4]),
                WORK_ELEMENTS,
            )
            .unwrap();
            assert_eq!((plan.height, plan.width), (4096, 4096));
            assert!(plan.rows > 0 && plan.rows < 4096);
            assert!(
                product(&[256, plan.rows, 4096, kernel, kernel]).unwrap()
                    <= candle_gen::CONV_IM2COL_BUDGET
            );
            assert!(
                product(&[256, (plan.rows - 1) + kernel, 4096 + 2 * (kernel / 2)]).unwrap()
                    <= WORK_ELEMENTS
            );
        }
    }

    #[test]
    fn forced_chunked_pointwise_upsample_and_residual_preserve_pixels() {
        let x = Tensor::arange(0f32, 96f32, &Device::Cpu)
            .unwrap()
            .reshape((2, 3, 4, 4))
            .unwrap()
            .affine(0.02, -0.7)
            .unwrap();
        close(
            &unary_with_limit(&x, 32, |t| Ok(candle_gen::candle_nn::ops::silu(t)?)).unwrap(),
            &candle_gen::candle_nn::ops::silu(&x).unwrap(),
        );
        close(&add_with_limit(&x, &x, 32).unwrap(), &x.add(&x).unwrap());
        close(
            &nearest_with_limit(&x, 8, 8, 128).unwrap(),
            &x.upsample_nearest2d(8, 8).unwrap(),
        );
    }

    #[test]
    fn forced_group_chunks_keep_full_image_statistics_and_affine_parameters() {
        let x = Tensor::arange(0f32, 4096f32, &Device::Cpu)
            .unwrap()
            .reshape((1, 64, 8, 8))
            .unwrap()
            .affine(0.01, -0.4)
            .unwrap();
        let weight = Tensor::arange(1f32, 65f32, &Device::Cpu)
            .unwrap()
            .affine(0.02, 0.7)
            .unwrap();
        let bias = weight.affine(-0.1, 0.3).unwrap();
        let norm = BoundedGroupNorm::new(weight, bias, 64, 32, 1e-6).unwrap();
        close(
            &norm.forward_with_limit(&x, 256).unwrap(),
            &norm.inner.forward(&x).unwrap(),
        );
    }

    #[test]
    fn forced_convolution_tiles_preserve_halos_shortcuts_and_asymmetric_padding() {
        let x = Tensor::arange(0f32, 252f32, &Device::Cpu)
            .unwrap()
            .reshape((1, 4, 9, 7))
            .unwrap()
            .affine(0.01, -0.5)
            .unwrap();
        for kernel in [1, 3] {
            let weight = Tensor::arange(0f32, (3 * 4 * kernel * kernel) as f32, &Device::Cpu)
                .unwrap()
                .reshape((3, 4, kernel, kernel))
                .unwrap()
                .affine(0.001, -0.04)
                .unwrap();
            let bias = Tensor::from_vec(vec![0.2f32, -0.1, 0.4], (3,), &Device::Cpu).unwrap();
            let conv = Conv2d::new(
                weight,
                Some(bias),
                Conv2dConfig {
                    padding: kernel / 2,
                    ..Default::default()
                },
            );
            close(
                &conv_with_limit(
                    &x,
                    &conv,
                    Padding([kernel / 2; 4]),
                    if kernel == 1 { 128 } else { 1024 },
                )
                .unwrap(),
                &conv.forward(&x).unwrap(),
            );
            if kernel == 3 {
                let down = Conv2d::new(
                    conv.weight().clone(),
                    conv.bias().cloned(),
                    Conv2dConfig {
                        stride: 2,
                        ..Default::default()
                    },
                );
                let padded = x
                    .pad_with_zeros(3, 0, 1)
                    .unwrap()
                    .pad_with_zeros(2, 0, 1)
                    .unwrap();
                close(
                    &conv_with_limit(&x, &down, Padding([0, 1, 0, 1]), 256).unwrap(),
                    &down.forward(&padded).unwrap(),
                );
            }
        }
    }
}
