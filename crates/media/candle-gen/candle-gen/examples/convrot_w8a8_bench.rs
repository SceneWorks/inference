//! sc-9299 8-bit GEMM bench — the decisive compute number for the epic's 8-bit pivot. Extends the
//! sc-8523 spike bench (which measured only bf16 / dequant / candle-int8-MMQ and concluded the MMQ
//! path was a NO-GO) with the two **cuBLASLt** columns that changed the verdict: **fp8 E4M3** and
//! **int8 IGEMM** driven through `candle_gen::quant::cublaslt::CublasLt`.
//!
//! Per DiT-representative `(M tokens, K in, N out)` shape it times, ms/iter, and reports TFLOP/s:
//! 1. `bf16` — dense bf16 `Tensor::matmul` (cuBLAS; the bf16-tier baseline)
//! 2. `dequant` — per-forward `QTensor::dequantize` (Q4_1) + dense bf16 matmul (production
//!    `QLinear`, sc-7702; the plain Q4/Q8-tier baseline)
//! 3. `mmq` — `QMatMul::forward` (candle's activation-q8_1 int8 MMQ kernels — the sc-8523 NO-GO path)
//! 4. `cublasLt-fp8` — this crate's fp8 E4M3 GEMM (weight pre-quant excluded from timing; dynamic
//!    activation quant INCLUDED, since it is per-forward in production)
//! 5. `cublasLt-int8` — this crate's int8 IGEMM (int32 accumulate + scale fold; activation quant
//!    INCLUDED)
//! 6. `w4a8-ref` / `w4a8-fused` (SW-6134) — the full `W4A8Linear::forward` of a weight packed by the
//!    offline packer (`gen_core::w4a8_int8::quantize_weight`): ConvRot FHT + kitchen int8
//!    activation quant, then either the dequant-tile + cuBLASLt IGEMM reference or the fused
//!    int4-weight int8-MMA GEMM (the `CANDLE_LLM_FUSED_KERNELS` switch selects the path; both are
//!    bit-identical). `prep-ref` / `prep-fused` time that path's activation prep alone (the eager
//!    candle-op FHT + `kitchen_int8` quantizer vs the fused FHT+quantizer kernel), so a path's GEMM
//!    stage is its forward column minus its prep column.
//!
//! A second table reports **peak device memory** above the call's baseline (the stream-ordered
//! pool's `USED_MEM_HIGH`) for one call of bf16, int8 (staged GEMM), and both W4A8 forwards, plus
//! the resident weight bytes of bf16 vs W4A8 (packed codes + scales). Krea 2 Turbo projection
//! shapes (hidden 6144, kv 1536, MLP 16384, 4096 tokens at 1024²) are included.
//!
//! ```text
//! (vcvars) cargo run --release --example convrot_w8a8_bench -p candle-gen --features cuda
//! ```
//!
//! NOTE (locked): this box is **sm_120** (Blackwell). The numbers are a Blackwell ceiling / parity
//! proof; the marketing audience runs sm_89 (RTX 40xx), where the bf16-vs-fp8/int8 ratio differs and
//! must be measured on that hardware separately (do not extrapolate this box's ratio).

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with --features cuda");
}

#[cfg(feature = "cuda")]
fn main() -> candle_gen::candle_core::Result<()> {
    use candle_gen::candle_core::quantized::{GgmlDType, QMatMul, QTensor};
    use candle_gen::candle_core::{DType, Device, Module, Tensor};
    use candle_gen::quant::cublaslt::{
        quantize_activation_fp8, quantize_activation_int8, quantize_weight_fp8,
        quantize_weight_int8, CublasLt,
    };
    use std::time::Instant;

    let dev = Device::new_cuda(0)?;
    let lt = CublasLt::new(&dev)?;
    let cap = lt.compute_cap()?;
    println!(
        "GPU compute cap {}.{}  meets fp8 floor (sm_89): {}",
        cap.0,
        cap.1,
        lt.meets_fp8_floor()?
    );

    // (label, M tokens, K in, N out) — z-image DiT attn/ffn @1024² (4096 tokens, dim 3840),
    // flux2-dev joint attn (dim 15360), and a small-batch tail case. All dims 16-aligned (cuBLASLt
    // 8-bit requirement).
    let shapes: &[(&str, usize, usize, usize)] = &[
        ("zimage-attn ", 4096, 3840, 3840),
        ("zimage-ffn  ", 4096, 3840, 10240),
        ("flux2-joint ", 4096, 15360, 3840),
        ("small-batch ", 16, 3840, 3840),
        ("krea-qo     ", 4096, 6144, 6144),
        ("krea-kv     ", 4096, 6144, 1536),
        ("krea-mlp-up ", 4096, 6144, 16384),
        ("krea-mlp-dn ", 4096, 16384, 6144),
        ("krea-decode ", 16, 6144, 6144),
    ];
    use candle_gen::cuda_mempool::MemPool;
    use candle_gen::quant::W4A8Linear;
    use candle_layers::fused::set_fused_kernels;
    let lt_arc = std::sync::Arc::new(CublasLt::new(&dev)?);
    let pool = MemPool::device_current(0).expect("the CUDA device's current pool");
    let peak = |f: &mut dyn FnMut() -> candle_gen::candle_core::Result<()>|
        -> candle_gen::candle_core::Result<f64> {
        dev.synchronize()?;
        let base = pool.used().unwrap_or(0);
        pool.reset_used_high_water();
        f()?;
        dev.synchronize()?;
        Ok(pool.used_high().unwrap_or(0).saturating_sub(base) as f64 / 1048576.0)
    };
    let mut peaks = Vec::new();
    let device_label = format!("sm_{}{}", cap.0, cap.1);

    let time = |f: &mut dyn FnMut() -> candle_gen::candle_core::Result<()>|
        -> candle_gen::candle_core::Result<f64> {
        for _ in 0..3 {
            f()?;
        }
        dev.synchronize()?;
        let t = Instant::now();
        let iters = 20;
        for _ in 0..iters {
            f()?;
        }
        dev.synchronize()?;
        Ok(t.elapsed().as_secs_f64() * 1e3 / iters as f64)
    };

    println!("\n(ms/iter over 20, 3 warmup; TFLOP/s in parens)");
    println!(
        "{:<13} {:>6} {:>6} {:>6}  {:>15} {:>15} {:>15} {:>15} {:>15} {:>15} {:>15} {:>9} {:>9}",
        "shape",
        "M",
        "K",
        "N",
        "bf16",
        "dequant+mm",
        "mmq(q8_1)",
        "cublasLt-fp8",
        "cublasLt-int8",
        "w4a8-ref",
        "w4a8-fused",
        "prep-ref",
        "prep-fused"
    );

    for &(label, m, k, n) in shapes {
        let w_cpu = Tensor::randn(0f32, 1f32, (n, k), &Device::Cpu)?;
        let w_bf16 = w_cpu.to_device(&dev)?.to_dtype(DType::BF16)?;
        let x = Tensor::randn(0f32, 1f32, (m, k), &dev)?.to_dtype(DType::BF16)?;
        let wt = w_bf16.t()?.contiguous()?;

        let flops = 2.0 * m as f64 * k as f64 * n as f64;
        let tf = |ms: f64| flops / (ms * 1e-3) / 1e12;

        // 1. bf16 dense.
        let t_bf16 = time(&mut || x.matmul(&wt).map(|_| ()))?;

        // 2. dequant (Q4_1) + dense.
        let qt = std::sync::Arc::new(QTensor::quantize_onto(&w_cpu, GgmlDType::Q4_1, &dev)?);
        let t_dq = time(&mut || {
            let wd = qt
                .dequantize(&dev)?
                .to_dtype(DType::BF16)?
                .t()?
                .contiguous()?;
            x.matmul(&wd).map(|_| ())
        })?;

        // 3. candle int8 MMQ (the sc-8523 path).
        let qmm = QMatMul::from_arc(qt.clone())?;
        let t_mmq = time(&mut || qmm.forward(&x).map(|_| ()))?;

        // 4. cuBLASLt fp8 E4M3 — the honest GEMM-compute ceiling: weight AND activation staged on
        //    device ONCE (per-forward activation quant is a fixed elementwise cost measured
        //    separately below), then only the matmul kernel is timed.
        let qw_fp8 = quantize_weight_fp8(&w_bf16)?;
        let qx_fp8 = quantize_activation_fp8(&x)?;
        let dev_w_fp8 = lt.stage_fp8(&qw_fp8.q)?;
        let dev_x_fp8 = lt.stage_fp8(&qx_fp8.q)?;
        let t_fp8 = time(&mut || {
            lt.matmul_fp8_staged(&dev_w_fp8, qw_fp8.scale, &dev_x_fp8, qx_fp8.scale)
                .map(|_| ())
        })?;

        // 5. cuBLASLt int8 IGEMM (same staging discipline).
        let qw_i8 = quantize_weight_int8(&w_bf16)?;
        let qx_i8 = quantize_activation_int8(&x)?;
        let dev_w_i8 = lt.stage_int8(&qw_i8.q)?;
        let dev_x_i8 = lt.stage_int8(&qx_i8.q)?;
        let t_i8 = time(&mut || lt.matmul_int8_staged(&dev_w_i8, &dev_x_i8).map(|_| ()))?;

        // 6. W4A8: the offline packer's output, the full native forward on each path.
        let packed = candle_gen::gen_core::w4a8_int8::quantize_weight(
            &w_cpu
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            n,
            k,
        )
        .map_err(candle_gen::candle_core::Error::wrap)?;
        let w4 = W4A8Linear::from_packed(&packed, None, lt_arc.clone())
            .map_err(|e| candle_gen::candle_core::Error::Msg(e.to_string()))?;
        set_fused_kernels(Some(false));
        let t_w4_ref = time(&mut || w4.forward(&x).map(|_| ()))?;
        let p_w4_ref = peak(&mut || w4.forward(&x).map(|_| ()))?;
        set_fused_kernels(Some(true));
        let t_w4_fused = time(&mut || w4.forward(&x).map(|_| ()))?;
        let p_w4_fused = peak(&mut || w4.forward(&x).map(|_| ()))?;
        set_fused_kernels(None);
        let t_prep = time(&mut || {
            let r = candle_gen::quant::convrot::convrot_fht_f32(&x, 256)?;
            lt_arc.quantize_kitchen_int8(&r, DType::BF16).map(|_| ())
        })?;
        let t_prep_fused = time(&mut || {
            candle_quant_kernels::w4a8_gemm::w4a8_quantize_activation(&x)
                .map(|_| ())
                .map_err(|e| candle_gen::candle_core::Error::Msg(e.to_string()))
        })?;
        let p_bf16 = peak(&mut || x.matmul(&wt).map(|_| ()))?;
        let p_i8 = peak(&mut || lt.matmul_int8_staged(&dev_w_i8, &dev_x_i8).map(|_| ()))?;
        peaks.push((
            label,
            m,
            k,
            n,
            p_bf16,
            p_i8,
            p_w4_ref,
            p_w4_fused,
            (n * k * 2) as f64 / 1048576.0,
            w4.resident_bytes() as f64 / 1048576.0,
        ));

        println!(
            "{label} {m:>6} {k:>6} {n:>6}  {:>7.3}({:>5.1}) {:>7.3}({:>5.1}) {:>7.3}({:>5.1}) {:>7.3}({:>5.1}) {:>7.3}({:>5.1}) {:>7.3}({:>5.1}) {:>7.3}({:>5.1}) {:>9.3} {:>9.3}",
            t_bf16, tf(t_bf16),
            t_dq, tf(t_dq),
            t_mmq, tf(t_mmq),
            t_fp8, tf(t_fp8),
            t_i8, tf(t_i8),
            t_w4_ref, tf(t_w4_ref),
            t_w4_fused, tf(t_w4_fused),
            t_prep,
            t_prep_fused,
        );
    }

    println!(
        "
peak device MiB above baseline for one call (pool USED_MEM_HIGH), device {device_label};          resident weight MiB"
    );
    println!(
        "{:<13} {:>6} {:>6} {:>6}  {:>9} {:>9} {:>9} {:>10}  {:>9} {:>9}",
        "shape", "M", "K", "N", "bf16", "int8", "w4a8-ref", "w4a8-fused", "w bf16", "w w4a8"
    );
    for (label, m, k, n, pb, pi, pr, pf, wb, ww) in peaks {
        println!(
            "{label} {m:>6} {k:>6} {n:>6}  {pb:>9.1} {pi:>9.1} {pr:>9.1} {pf:>10.1}  {wb:>9.1} {ww:>9.1}"
        );
    }
    Ok(())
}
