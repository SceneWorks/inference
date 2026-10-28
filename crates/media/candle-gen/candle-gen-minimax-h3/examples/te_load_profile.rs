//! Real-weight load profiler for the MiniMax-H3 q4 text encoder (sc-26691). NOT part of CI: run by hand
//!
//! ```text
//! H3_TE_DIR=<snapshot>/q4/text_encoder CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID \
//!   cargo run --release --example te_load_profile --features cuda
//! ```
//!
//! Does what the product does (`Weights::from_files_filtered*` + `MiniMaxH3TextEncoder::from_weights`)
//! and prints a per-phase breakdown. The loader decomposition (mmap open, header scan, per-tensor
//! read, device upload) is measured separately on the same shards so the phases are attributable.
use std::time::Instant;

use candle_gen::candle_core::safetensors::MmapedSafetensors;
use candle_gen::candle_core::Device;
use candle_gen_minimax_h3::text_encoder::{lm_prefixes, MiniMaxH3TeConfig, TE_STORE_DTYPE};
use candle_gen_minimax_h3::{MiniMaxH3TextEncoder, LM_PREFIX};

fn main() {
    let dir = std::path::PathBuf::from(std::env::var("H3_TE_DIR").expect("H3_TE_DIR"));
    let device = if std::env::var("H3_CPU").is_ok() {
        Device::Cpu
    } else {
        Device::new_cuda(0).expect("cuda:0 (set CUDA_VISIBLE_DEVICES=1)")
    };
    let cfg = MiniMaxH3TeConfig::from_component_dir(&dir).unwrap();
    let shards = candle_gen::loader::sorted_safetensors(&dir, "te").unwrap();
    let prefixes = lm_prefixes(LM_PREFIX, &cfg);
    let refs: Vec<&str> = prefixes.iter().map(String::as_str).collect();

    if std::env::var("H3_DECOMPOSE").is_ok() {
        let (mut open, mut hdr, mut read_cpu, mut n, mut bytes) =
            (0f64, 0f64, 0f64, 0usize, 0usize);
        for p in &shards {
            let t = Instant::now();
            let st = unsafe { MmapedSafetensors::new(p).unwrap() };
            open += t.elapsed().as_secs_f64();
            let t = Instant::now();
            let keys: Vec<String> = st
                .tensors()
                .into_iter()
                .map(|(k, _)| k)
                .filter(|k| refs.iter().any(|p| k.starts_with(p)))
                .collect();
            hdr += t.elapsed().as_secs_f64();
            let t = Instant::now();
            for k in &keys {
                let x = st.load(k, &Device::Cpu).unwrap();
                bytes += x.elem_count() * x.dtype().size_in_bytes();
                n += 1;
            }
            read_cpu += t.elapsed().as_secs_f64();
        }
        eprintln!("decompose: mmap-open {open:.2}s  header-scan {hdr:.2}s  read-to-host {read_cpu:.2}s  ({n} tensors, {:.2} GiB)", bytes as f64 / (1u64 << 30) as f64);
    }

    let t = Instant::now();
    // (The legacy-path run replaces this call with `from_files_filtered` on a detached origin/main tree.)
    let w = candle_gen::Weights::from_files_filtered_host_packed(
        &shards,
        &device,
        TE_STORE_DTYPE,
        &refs,
    )
    .unwrap();
    let t_weights = t.elapsed();
    let t = Instant::now();
    let te = MiniMaxH3TextEncoder::from_weights(&w, LM_PREFIX, &cfg, TE_STORE_DTYPE).unwrap();
    let t_te = t.elapsed();
    let (calls, parts, repack, upload) = candle_gen::quant::repack::profile::snapshot();
    eprintln!(
        "Weights load {t_weights:.1?}  from_weights {t_te:.1?}  | q4 tensors (cumulative, process) {calls}: parts/D2H {parts:.2}s  repack {repack:.2}s  upload {upload:.2}s | layers {}",
        te.num_loaded_layers()
    );
}
