//! Source traversal for SANA native allocation accounting.
//!
//! This is a cumulative allocation walk, not a live logical-scratch sum. Every
//! primitive operand is charged independently by `native_alloc`; no output,
//! source, cast, layout repair or workspace is retired. Floating operands use a
//! four-byte envelope (all advertised source/compute dtypes are F16/BF16/F32).
//! This deliberately counts conditional copies even when a particular execution
//! aliases them. A native bound is not a process/physical-footprint certificate.

use super::native_alloc::{NativeBackend, NativeLedger};
use super::*;

fn invalid(message: &str) -> Error {
    Error::Unsupported(format!("SANA native source walk: {message}"))
}
fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("allocation extent overflow"))
}
fn sum(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b)
        .ok_or_else(|| invalid("allocation extent overflow"))
}
fn dim(value: i32) -> Result<u64> {
    u64::try_from(value)
        .ok()
        .filter(|n| *n > 0)
        .ok_or_else(|| invalid("nonpositive source dimension"))
}
fn tensor<'a>(inventory: &'a Inventory, key: &str) -> Result<&'a Tensor> {
    inventory
        .0
        .get(key)
        .ok_or_else(|| invalid(&format!("missing header {key}")))
}
fn shape(tensor: &Tensor, rank: usize) -> Result<&[u64]> {
    if tensor.shape.len() != rank || tensor.shape.contains(&0) {
        return Err(invalid("source tensor rank/extent mismatch"));
    }
    Ok(&tensor.shape)
}

struct Walk<'a> {
    ledger: NativeLedger,
    facts: &'a SanaPricingFacts,
}
impl<'a> Walk<'a> {
    fn new(facts: &'a SanaPricingFacts, page: u64, backend: NativeBackend) -> Result<Self> {
        Ok(Self {
            ledger: NativeLedger::new(page, backend)?,
            facts,
        })
    }
    fn a(&mut self, elements: u64) -> Result<()> {
        self.ledger.array(4, elements)
    }
    fn ops(&mut self, elements: u64, count: u64) -> Result<()> {
        self.ledger.repeated(4, elements, count)
    }
    fn scalars(&mut self, count: u64) -> Result<()> {
        self.ops(1, count)
    }
    fn mm(&mut self, batch: u64, m: u64, n: u64, k: u64) -> Result<()> {
        self.ledger.matmul(4, batch, m, n, k)
    }
    /// Header leaf + possible dtype constructor target. Raw header storage and
    /// target are distinct; equal-width F16/BF16 conversions cannot disappear.
    fn inventory(&mut self, inventory: &Inventory) -> Result<()> {
        for tensor in inventory.0.values() {
            if !matches!(
                tensor.dtype,
                Dtype::Float16 | Dtype::Bfloat16 | Dtype::Float32 | Dtype::Uint32
            ) {
                return Err(invalid(
                    "native lowering has no advertised source dtype authority",
                ));
            }
            let elements = tensor.shape.iter().try_fold(1u64, |a, &b| mul(a, b))?;
            if elements != tensor.elements {
                return Err(invalid("header shape/product changed"));
            }
            let stored_width = if matches!(tensor.dtype, Dtype::Float16 | Dtype::Bfloat16) {
                2
            } else {
                4
            };
            if tensor.bytes != mul(elements, stored_width)? {
                return Err(invalid(
                    "header byte extent does not match source shape/dtype",
                ));
            }
            self.ledger.bytes(tensor.bytes)?;
            self.a(tensor.elements)?;
            if tensor.bytes > 0 {
                // Explicit default-GPU load checks are independent of a CPU
                // task-local COMPUTE stream. Charge their complete source
                // envelope on BOTH compute backends; CPU-only fixtures may
                // skip the real checks, never the conservative source price.
                // weights::materialize_batch -> coherence::verify_gpu_view:
                // one CPU checksum, then up to MAX_RETRIES+1 GPU checksums.
                // Safetensors source leaves are row-contiguous, so flatten and
                // bitcast are views. An odd byte-size instead widens EACH byte
                // to U32 before the reduction. No retry/drop retirement credit.
                let words = if tensor.bytes.is_multiple_of(4) {
                    tensor.bytes / 4
                } else {
                    tensor.bytes
                };
                let mut cpu = NativeLedger::new(self.ledger.page, NativeBackend::Cpu)?;
                if !tensor.bytes.is_multiple_of(4) {
                    cpu.array(4, words)?;
                }
                cpu.reduce(4, words, 1, words, true)?;
                self.ledger.total = sum(self.ledger.total, cpu.total)?;
                let mut gpu = NativeLedger::new(self.ledger.page, NativeBackend::Metal)?;
                for _ in 0..=mlx_gen::coherence::MAX_RETRIES {
                    if !tensor.bytes.is_multiple_of(4) {
                        gpu.array(4, words)?;
                    }
                    gpu.reduce(4, words, 1, words, true)?;
                }
                self.ledger.total = sum(self.ledger.total, gpu.total)?;
            }
        }
        Ok(())
    }
    /// `quant::lin` -> `AdaptableLinear::forward`, dense or affine Q4/Q8.
    /// The native shape is read from the actual header triple, never the tier name.
    /// Count the possible activation cast and bias cast independently of QMM's
    /// normalized input/weight copies. SANA refuses adapters in its load contract.
    fn linear(&mut self, inventory: &Inventory, base: &str, rows: u64, input: u64) -> Result<u64> {
        let weight = tensor(inventory, &format!("{base}.weight"))?;
        let ws = shape(weight, 2)?;
        let output = ws[0];
        self.a(mul(rows, input)?)?;
        if let Some(scales) = inventory.0.get(&format!("{base}.scales")) {
            let biases = tensor(inventory, &format!("{base}.biases"))?;
            let ss = shape(scales, 2)?;
            if weight.dtype != Dtype::Uint32
                || ss[0] != output
                || ss[1] == 0
                || !input.is_multiple_of(ss[1])
                || biases.shape != scales.shape
            {
                return Err(invalid("incoherent affine linear headers"));
            }
            let group = input / ss[1];
            let bits = mul(ws[1], 32)?
                .checked_div(input)
                .ok_or_else(|| invalid("empty affine input"))?;
            if !matches!(bits, 4 | 8) || mul(ws[1], 32)? != mul(input, bits)? {
                return Err(invalid("affine codes do not describe a Q4/Q8 input"));
            }
            // QuantizedLinear's dtype casts can allocate both affine parameters.
            self.a(scales.elements)?;
            self.a(biases.elements)?;
            self.ledger.qmm(
                4,
                rows,
                input,
                output,
                [weight.elements, scales.elements, group],
            )?;
        } else {
            if ws[1] != input || weight.dtype == Dtype::Uint32 {
                return Err(invalid("dense linear header input mismatch"));
            }
            self.a(weight.elements)?; // promotion, even same-width source dtype changes
            self.mm(1, rows, output, input)?;
        }
        if let Some(bias) = inventory.0.get(&format!("{base}.bias")) {
            if bias.elements != output {
                return Err(invalid("linear bias extent mismatch"));
            }
            self.a(output)?;
            self.a(mul(rows, output)?)?;
        }
        Ok(output)
    }
    /// Conv::forward -> nn::conv2d_general, including bias epilogue and
    /// exact-epilogue fallback. Weight shape is source OIHW, not guessed NHWC.
    #[allow(clippy::too_many_arguments)] // Native source operands, kept explicit for audit.
    fn conv(
        &mut self,
        inventory: &Inventory,
        base: &str,
        h: u64,
        w: u64,
        cin: u64,
        stride: u64,
        pad: u64,
        groups: u64,
    ) -> Result<(u64, u64, u64)> {
        let weight = tensor(inventory, &format!("{base}.weight"))?;
        let ws = shape(weight, 4)?;
        let (cout, kh, kw) = (ws[0], ws[2], ws[3]);
        if groups == 0 || mul(ws[1], groups)? != cin || stride == 0 {
            return Err(invalid("convolution header input mismatch"));
        }
        let output_axis = |n, k| -> Result<u64> {
            let padded = sum(n, mul(2, pad)?)?;
            sum(
                padded
                    .checked_sub(k)
                    .ok_or_else(|| invalid("convolution exceeds input"))?
                    / stride,
                1,
            )
        };
        let (oh, ow) = (output_axis(h, kh)?, output_axis(w, kw)?);
        self.a(weight.elements)?; // promoted/transposed constructor parent
        self.a(mul(mul(h, w)?, cin)?)?; // mixed-input promotion
        self.ledger.conv2d(
            [4, 1, h, w, cin, cout, kh, kw, oh, ow],
            [stride, stride],
            [pad, pad],
            [1, 1],
            groups,
        )?;
        // CPU F16/BF16 convolution additionally has a distinct F32 gemm output;
        // using F32 primitive geometry must not silently erase that possible node.
        if self.ledger.backend == NativeBackend::Cpu {
            self.a(mul(mul(oh, ow)?, cout)?)?;
        }
        if let Some(bias) = inventory.0.get(&format!("{base}.bias")) {
            if bias.elements != cout {
                return Err(invalid("convolution bias extent mismatch"));
            }
            self.a(cout)?;
            self.a(mul(mul(oh, ow)?, cout)?)?;
        }
        Ok((oh, ow, cout))
    }
    /// Explicit SANA/DC-AE RMS: cast, square, mean, epsilon add,
    /// sqrt/rsqrt, normalize, affine multiply, optional bias, cast back.
    fn rms(&mut self, rows: u64, channels: u64, bias: bool) -> Result<()> {
        let n = mul(rows, channels)?;
        self.a(n)?;
        self.a(channels)?;
        self.a(n)?;
        self.ledger.reduce(4, n, rows, channels, true)?;
        self.scalars(2)?; // mean normalizer and epsilon
        self.ops(rows, 3)?; // mean normalization, epsilon add, sqrt/rsqrt
        self.ops(n, 3)?; // normalized product/division, output cast, gain product
        if bias {
            self.a(channels)?;
            self.a(n)?;
        }
        Ok(())
    }
    fn silu(&mut self, n: u64) -> Result<()> {
        self.ops(n, 2)
    }
    fn relu(&mut self, n: u64) -> Result<()> {
        // mlx-rs nn/activation.rs:36-38 maximum(x,array!(0)): original
        // I32 zero and its possible F32 promotion are distinct scalar owners.
        self.scalars(2)?;
        self.a(n)
    }
    /// nn::gelu_tanh's eager nine-operation expansion also bounds the exact
    /// kernel path's input repair/output. Four typed constants, integer power.
    fn gelu(&mut self, n: u64) -> Result<()> {
        self.scalars(9)?;
        self.ops(n, 9)
    }
    fn norm(&mut self, rows: u64, channels: u64) -> Result<()> {
        self.ledger
            .layernorm(mul(rows, channels)?, rows, 4, 4, 0, 0)
    }
    fn modulate(&mut self, rows: u64, channels: u64) -> Result<()> {
        self.scalars(2)?;
        self.a(channels)?;
        self.ops(mul(rows, channels)?, 2)
    }
    /// nn::timestep_sincos(256): arange, exponent mul/div/exp,
    /// timestep cast, emb product, cos/sin, concatenate; no host bytes.
    fn timestep(&mut self) -> Result<()> {
        self.scalars(3)?;
        self.ops(128, 7)?;
        self.a(256)
    }
    fn random(&mut self, n: u64) -> Result<()> {
        self.a(2)?;
        self.a(4)?;
        self.scalars(6)?;
        self.ops(n, 8)
    }

    fn dcae_res(&mut self, prefix: &str, h: u64, w: u64, c: u64) -> Result<()> {
        let inv = &self.facts.vae;
        self.conv(inv, &format!("{prefix}.conv1"), h, w, c, 1, 1, 1)?;
        self.silu(mul(mul(h, w)?, c)?)?;
        self.conv(inv, &format!("{prefix}.conv2"), h, w, c, 1, 1, 1)?;
        self.rms(mul(h, w)?, c, true)?;
        self.a(mul(mul(h, w)?, c)?)
    }
    /// dc_ae::LinearAttn::forward, including every multiscale depthwise and
    /// grouped projection, both concatenations, gram/denominator products.
    fn dcae_attn(&mut self, prefix: &str, h: u64, w: u64, c: u64) -> Result<()> {
        let inv = &self.facts.vae;
        let rows = mul(h, w)?;
        for name in ["to_q", "to_k", "to_v"] {
            self.linear(inv, &format!("{prefix}.{name}"), rows, c)?;
        }
        let qkv = mul(3, c)?;
        self.a(mul(rows, qkv)?)?;
        let cfg = &self.facts.dc_ae;
        let heads = c / dim(cfg.attention_head_dim)?;
        for (i, &kernel) in cfg.qkv_multiscales.iter().enumerate() {
            self.conv(
                inv,
                &format!("{prefix}.to_qkv_multiscale.{i}.proj_in"),
                h,
                w,
                qkv,
                1,
                dim(kernel)? / 2,
                qkv,
            )?;
            self.conv(
                inv,
                &format!("{prefix}.to_qkv_multiscale.{i}.proj_out"),
                h,
                w,
                qkv,
                1,
                0,
                mul(3, heads)?,
            )?;
        }
        let groups = mul(heads, sum(cfg.qkv_multiscales.len() as u64, 1)?)?;
        let hd = dim(cfg.attention_head_dim)?;
        let expanded = mul(groups, hd)?;
        self.a(mul(rows, mul(3, expanded)?)?)?; // concatenate + possible layout reshape
        self.a(mul(rows, mul(3, expanded)?)?)?;
        self.relu(mul(rows, expanded)?)?;
        self.relu(mul(rows, expanded)?)?;
        self.mm(groups, hd, hd, rows)?;
        self.mm(groups, hd, rows, hd)?;
        self.ledger
            .reduce(4, mul(rows, expanded)?, expanded, rows, true)?;
        self.mm(groups, 1, rows, hd)?;
        self.scalars(1)?;
        self.a(mul(groups, rows)?)?;
        self.a(mul(rows, expanded)?)?;
        self.a(mul(rows, expanded)?)?; // output reshape materialization
        self.linear(inv, &format!("{prefix}.to_out"), rows, expanded)?;
        self.rms(rows, c, true)?;
        self.a(mul(rows, c)?)
    }
    fn glu(
        &mut self,
        inv: &Inventory,
        prefix: &str,
        h: u64,
        w: u64,
        c: u64,
        norm: bool,
    ) -> Result<()> {
        let rows = mul(h, w)?;
        let (_, _, expanded) =
            self.conv(inv, &format!("{prefix}.conv_inverted"), h, w, c, 1, 0, 1)?;
        self.silu(mul(rows, expanded)?)?;
        self.conv(
            inv,
            &format!("{prefix}.conv_depth"),
            h,
            w,
            expanded,
            1,
            1,
            expanded,
        )?;
        let hidden = expanded / 2;
        self.silu(mul(rows, hidden)?)?;
        self.a(mul(rows, hidden)?)?;
        self.conv(inv, &format!("{prefix}.conv_point"), h, w, hidden, 1, 0, 1)?;
        if norm {
            self.rms(rows, c, true)?;
            self.a(mul(rows, c)?)?;
        }
        Ok(())
    }
    fn dcae_blocks(&mut self, encode: bool, i: usize, h: u64, w: u64) -> Result<()> {
        let cfg = &self.facts.dc_ae;
        let count = if encode {
            cfg.encoder_layers_per_block[i]
        } else {
            cfg.layers_per_block[i]
        };
        for j in 0..count {
            self.dcae_block(encode, i, j, h, w)?;
        }
        Ok(())
    }
    fn dcae_block(&mut self, encode: bool, i: usize, j: i32, h: u64, w: u64) -> Result<()> {
        let cfg = &self.facts.dc_ae;
        let c = dim(cfg.block_out_channels[i])?;
        let side = if encode { "encoder" } else { "decoder" };
        let prefix = if encode {
            format!("{side}.down_blocks.{i}.{j}")
        } else {
            format!(
                "decoder.up_blocks.{i}.{}",
                j + i32::from(i + 1 < cfg.num_stages())
            )
        };
        match cfg.block_types[i] {
            BlockType::Res => self.dcae_res(&prefix, h, w, c)?,
            BlockType::EfficientVit => {
                self.dcae_attn(&format!("{prefix}.attn"), h, w, c)?;
                self.glu(
                    &self.facts.vae,
                    &format!("{prefix}.conv_out"),
                    h,
                    w,
                    c,
                    true,
                )?;
            }
        }

        Ok(())
    }
}

impl Walk<'_> {
    fn encode_stages(
        &mut self,
        range: std::ops::Range<usize>,
        mut h: u64,
        mut w: u64,
    ) -> Result<(u64, u64)> {
        let cfg = &self.facts.dc_ae;
        for i in range {
            self.dcae_blocks(true, i, h, w)?;
            if i + 1 < cfg.num_stages() {
                let c = dim(cfg.block_out_channels[i])?;
                let base = format!(
                    "encoder.down_blocks.{i}.{}.conv",
                    cfg.encoder_layers_per_block[i]
                );
                let (oh, ow, cout) = self.conv(&self.facts.vae, &base, h, w, c, 2, 1, 1)?;
                self.a(mul(mul(h, w)?, c)?)?; // pixel_unshuffle reshape materialization
                let out = mul(mul(oh, ow)?, cout)?;
                let input = mul(mul(h, w)?, c)?;
                if out == 0 || !input.is_multiple_of(out) {
                    return Err(invalid("encoder group shortcut extent mismatch"));
                }
                self.ledger.reduce(4, input, out, input / out, true)?;
                self.scalars(1)?;
                self.ops(out, 2)?; // mean normalizer + residual add
                (h, w) = (oh, ow);
            }
        }
        Ok((h, w))
    }
    fn encode_prefix(&mut self, h: u64, w: u64, prefix: usize) -> Result<()> {
        let cfg = &self.facts.dc_ae;
        self.conv(
            &self.facts.vae,
            "encoder.conv_in",
            h,
            w,
            dim(cfg.in_channels)?,
            1,
            1,
            1,
        )?;
        self.encode_stages(0..prefix, h, w)?;
        Ok(())
    }
    fn encode(&mut self, plan: &SanaPricingPlan) -> Result<()> {
        let cfg = &self.facts.dc_ae;
        let (h, w) = (u64::from(plan.height), u64::from(plan.width));
        self.a(mul(mul(h, w)?, dim(cfg.in_channels)?)?)?; // preprocess_init_image Array upload
        let prefix = cfg
            .block_types
            .iter()
            .take_while(|t| **t == BlockType::Res)
            .count();
        let down = (0..prefix).filter(|i| i + 1 < cfg.num_stages()).count();
        let cell = 1u64
            .checked_shl(down as u32)
            .ok_or_else(|| invalid("encode prefix scale overflow"))?;
        let core = plan
            .encode_tile
            .map(|n| dim(n).map(|n| n.div_ceil(cell) * cell))
            .transpose()?;
        let (rh, rw) =
            if let Some(core) = core.filter(|core| prefix > 0 && (h > *core || w > *core)) {
                let (mut radius, mut scale) = (1u64, 1u64);
                for i in 0..prefix {
                    radius = sum(
                        radius,
                        mul(mul(2, dim(cfg.encoder_layers_per_block[i])?)?, scale)?,
                    )?;
                    if i + 1 < cfg.num_stages() {
                        radius = sum(radius, scale)?;
                        scale = mul(scale, 2)?;
                    }
                }
                let halo = mul(radius.div_ceil(cell), cell)?;
                let c = dim(cfg.block_out_channels[prefix.min(cfg.num_stages() - 1)])?;
                let mut y = 0;
                while y < h {
                    let mut x = 0;
                    while x < w {
                        let th = sum(y, core)?.min(h).saturating_add(halo).min(h)
                            - y.saturating_sub(halo);
                        let tw = sum(x, core)?.min(w).saturating_add(halo).min(w)
                            - x.saturating_sub(halo);
                        self.encode_prefix(th, tw, prefix)?;
                        x = sum(x, core)?;
                    }
                    let row_h = sum(y, core)?.min(h) - y;
                    self.a(mul(mul(row_h / cell, w / cell)?, c)?)?; // every row concatenate
                    y = sum(y, core)?;
                }
                self.a(mul(mul(h / cell, w / cell)?, c)?)?; // concatenate rows
                self.encode_stages(prefix..cfg.num_stages(), h / cell, w / cell)?
            } else {
                self.conv(
                    &self.facts.vae,
                    "encoder.conv_in",
                    h,
                    w,
                    dim(cfg.in_channels)?,
                    1,
                    1,
                    1,
                )?;
                self.encode_stages(0..cfg.num_stages(), h, w)?
            };
        let c = dim(*cfg
            .block_out_channels
            .last()
            .ok_or_else(|| invalid("no DC-AE stages"))?)?;
        let out = mul(mul(rh, rw)?, dim(cfg.latent_channels)?)?;
        self.conv(&self.facts.vae, "encoder.conv_out", rh, rw, c, 1, 1, 1)?;
        let input = mul(mul(rh, rw)?, c)?;
        if out == 0 || !input.is_multiple_of(out) {
            return Err(invalid("encoder output shortcut extent mismatch"));
        }
        self.ledger.reduce(4, input, out, input / out, true)?;
        self.scalars(2)?; // mean normalization and latent scaling factor
        self.ops(out, 3) // mean normalization, residual, scaling multiply
    }
    fn decode_stages(
        &mut self,
        range: std::ops::Range<usize>,
        mut h: u64,
        mut w: u64,
    ) -> Result<(u64, u64)> {
        let cfg = &self.facts.dc_ae;
        for i in range.rev() {
            if i + 1 < cfg.num_stages() {
                let cin = dim(cfg.block_out_channels[i + 1])?;
                let cout = dim(cfg.block_out_channels[i])?;
                (h, w) = (mul(h, 2)?, mul(w, 2)?);
                self.a(mul(mul(h, w)?, cin)?)?; // nearest broadcast reshape
                self.conv(
                    &self.facts.vae,
                    &format!("decoder.up_blocks.{i}.0.conv"),
                    h,
                    w,
                    cin,
                    1,
                    1,
                    1,
                )?;
                // repeat_interleave then pixel shuffle: both conditional reshape
                // materializations and the residual are distinct source requests.
                self.ops(mul(mul(h, w)?, cout)?, 3)?;
            }
            self.dcae_blocks(false, i, h, w)?;
        }
        Ok((h, w))
    }
    fn decode_output(&mut self, h: u64, w: u64) -> Result<()> {
        let cfg = &self.facts.dc_ae;
        let c = dim(cfg.block_out_channels[0])?;
        self.rms(mul(h, w)?, c, true)?;
        self.relu(mul(mul(h, w)?, c)?)?;
        self.conv(&self.facts.vae, "decoder.conv_out", h, w, c, 1, 1, 1)?;
        Ok(())
    }
    fn decode(&mut self, plan: &SanaPricingPlan) -> Result<()> {
        let cfg = &self.facts.dc_ae;
        let (lh, lw) = (u64::from(plan.height / 32), u64::from(plan.width / 32));
        let deepest = dim(*cfg
            .block_out_channels
            .last()
            .ok_or_else(|| invalid("no DC-AE stages"))?)?;
        self.scalars(1)?;
        self.a(mul(mul(lh, lw)?, dim(cfg.latent_channels)?)?)?; // unscale
        self.conv(
            &self.facts.vae,
            "decoder.conv_in",
            lh,
            lw,
            dim(cfg.latent_channels)?,
            1,
            1,
            1,
        )?;
        self.ops(mul(mul(lh, lw)?, deepest)?, 2)?; // repeat reshape + residual
        let tail = cfg
            .block_types
            .iter()
            .take_while(|t| **t == BlockType::Res)
            .count();
        let (hh, hw) = self.decode_stages(tail..cfg.num_stages(), lh, lw)?;
        let scale = 1u64
            .checked_shl((0..tail).filter(|i| i + 1 < cfg.num_stages()).count() as u32)
            .ok_or_else(|| invalid("decode tail scale overflow"))?;
        let vae = VaeTiling {
            spatial_scale: i32::try_from(scale)
                .map_err(|_| invalid("decode scale out of range"))?,
            temporal_scale: 1,
            causal_temporal: false,
            full_res_channels: 3,
        };
        let tiling = plan
            .decode_tiling
            .map(|(edge, overlap)| TilingConfig::spatial_only(edge, overlap));
        if let Some(tiling) =
            tiling.filter(|t| tail > 0 && t.needs_tiling(vae, 1, hh as i32, hw as i32))
        {
            let tiles = tiling.plan(vae, 1, hh as i32, hw as i32);
            let (oh, ow) = (mul(hh, scale)?, mul(hw, scale)?);
            for ht in &tiles.h {
                let th = u64::try_from(ht.end - ht.start)
                    .map_err(|_| invalid("negative decode tile"))?;
                for wt in &tiles.w {
                    let tw = u64::try_from(wt.end - wt.start)
                        .map_err(|_| invalid("negative decode tile"))?;
                    let (h, w) = self.decode_stages(0..tail, th, tw)?;
                    self.decode_output(h, w)?;
                    // vae_tiling::tiled_decode_with_finalizer: three masks,
                    // two outer products, weighted tile, then pad+add on EACH
                    // fold axis (also charge conditional first/sole-tile branches).
                    self.scalars(4)?; // temporal mask + each Pad's default zero
                    self.a(w)?;
                    self.ops(h, 2)?; // width/height masks and t*h
                    self.a(mul(h, w)?)?; // (t*h)*w
                    self.a(mul(mul(h, w)?, 3)?)?; // decoded * blend
                    self.ops(mul(mul(h, ow)?, 3)?, 2)?; // width Pad + accumulation
                    self.ops(mul(mul(oh, ow)?, 3)?, 4)?; // height and temporal Pad + accumulation
                }
            }
            // AxisCoverage::normalizer: t/h/w uploads, two outer products;
            // maximum(epsilon), division, and contiguous reshape/readback copy.
            self.scalars(2)?;
            self.ops(oh, 2)?;
            self.a(ow)?;
            self.ops(mul(oh, ow)?, 2)?;
            self.ops(mul(mul(oh, ow)?, 3)?, 2)?;
        } else {
            let (h, w) = self.decode_stages(0..tail, hh, hw)?;
            self.decode_output(h, w)?;
        }
        // decoded_to_image: transpose view, dtype cast, add/divide, clip,
        // multiply255, round, Uint8 cast. Uint8 is bounded by 4-byte output.
        self.scalars(4)?;
        self.ops(
            mul(mul(u64::from(plan.height), u64::from(plan.width))?, 3)?,
            7,
        )
    }
}

const NATIVE_EPOCH_LABELS: usize = labels::ALL.len() + GEMMA_LABELS.len() + 1;
pub(super) fn native_epoch_labels() -> impl Iterator<Item = &'static str> {
    labels::ALL
        .iter()
        .copied()
        .chain(GEMMA_LABELS.iter().copied())
        .chain(std::iter::once(mlx_gen::sampler::SAMPLER_STEP_QUANTUM))
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct NativeEpochCounts {
    epochs: [u64; NATIVE_EPOCH_LABELS],
    pub(super) solver_marks: u64,
}
impl NativeEpochCounts {
    fn new() -> Self {
        Self {
            epochs: [0; NATIVE_EPOCH_LABELS],
            solver_marks: 0,
        }
    }
    fn note(&mut self, label: &str) -> Result<()> {
        let index = native_epoch_labels()
            .position(|known| known == label)
            .ok_or_else(|| invalid("unknown closed source quantum label"))?;
        self.epochs[index] = sum(self.epochs[index], 1)?;
        Ok(())
    }
    pub(super) fn count(&self, label: &str) -> Result<u64> {
        let index = native_epoch_labels()
            .position(|known| known == label)
            .ok_or_else(|| invalid("unknown observed quantum label"))?;
        Ok(self.epochs[index])
    }
}

include!("native_gemma.rs");
include!("native_trunk.rs");
include!("native_sampler.rs");
#[path = "native_retired.rs"]
mod retired;
pub(super) fn retired_upper(
    facts: &SanaPricingFacts,
    plan: &SanaPricingPlan,
    sampling: &NativeSampling,
    page: u64,
    backend: NativeBackend,
    budget: u64,
) -> Result<u64> {
    retired::retired_upper(facts, plan, sampling, page, backend, budget)
}
pub(super) fn retired_profile(
    facts: &SanaPricingFacts,
    plan: &SanaPricingPlan,
    sampling: &NativeSampling,
    page: u64,
    backend: NativeBackend,
    budget: u64,
) -> Result<(u64, NativeEpochCounts)> {
    retired::retired_profile(facts, plan, sampling, page, backend, budget)
}

pub(super) fn source_epoch_counts(
    facts: &SanaPricingFacts,
    plan: &SanaPricingPlan,
    sampling: &NativeSampling,
    page: u64,
    budget: u64,
) -> Result<NativeEpochCounts> {
    retired::source_epoch_counts(facts, plan, sampling, page, budget)
}

#[cfg(test)]
#[path = "native_graph_tests.rs"]
mod tests;
