//! Backend-owned tensor primitives (epic 7153).
//!
//! These are the decode leaves `candle-llm` owns — the Candle reimplementation of the `mlx-llm`
//! foundation: the batch-capable KV caches (growing and preallocated), the sampler, the RoPE family, GQA attention helpers,
//! group-wise quantization (Candle's `QTensor`/`QMatMul`), the `nn` leaves (linear / RMSNorm /
//! activations / embedding), and a safetensors weights loader. They own Candle `Tensor`s directly.
//!
//! The RMSNorm / SwiGLU / QK-norm+RoPE leaves have a fused CUDA implementation behind the same
//! entry points (sc-24137): bit-identical to the op chain, on by default in a `cuda` build, with
//! which path ran recorded per thread in [`fused`] and per request in the decode record.
//!
//! NVFP4 projections pick the fused decode GEMV (≤ 8 rows) or the cuBLASLt W4A4 GEMM per call
//! (sc-24136), with the path recorded per thread in [`nvfp4_path`] and per request in the decode
//! record.
//!
//! Shapes are **batch-capable from day one**: the batch axis is a real dimension everywhere, even
//! though the first decoders run batch-1. The [`KvCache`] trait is the seam a paged cache slots in
//! behind without touching decoders.

pub mod attention;
pub mod decode_cache;
pub mod fused;
pub mod gated_delta;
pub mod host_sync;
pub mod kv_cache;
pub mod nn;
pub mod nvfp4_path;
pub mod paged_kv_cache;
pub mod prism;
pub mod projection;
pub mod quant;
pub mod rope;
pub mod sampler;
pub mod step_kv_cache;
pub mod switch;
pub mod weights;

pub use attention::{
    repeat_kv, sdpa, sdpa_causal, sdpa_gqa_causal, sliding_causal_mask, AttnFormulation, AttnMask,
};
pub use decode_cache::{tensor_bytes, CacheMemory, DecodeCache};
pub use fused::{
    fused_kernels_enabled, fused_tally, set_fused_kernels, FusedTally, FUSED_KERNELS_ENV,
};
#[doc(hidden)]
pub use fused::{fused_policy_guard, FusedPolicyGuard};
pub use gated_delta::{
    causal_depthwise_conv, compute_g, gated_delta_recurrence, rms_norm_gated, DeltaNetCache,
};
pub use host_sync::{
    host_sync_count, last_host_reason, note_host_sync, note_logits_to_host, note_sampler_path,
    sampler_counters, SamplerCounters,
};
pub use kv_cache::{
    kv_materialize_count, note_kv_materialize, storage_address, ContiguousKvCache, KvCache,
    KvCacheKind, StaticKvCache,
};
pub use nn::{
    conv2d, embed, gelu, gelu_erf, input_ids, input_ids_batch, layer_norm, linear, rms_norm,
    rms_norm_reference, rms_norm_residual, rms_norm_unscaled, silu, soft_cap, swiglu,
};
pub use nvfp4_path::{
    nvfp4_gemv_enabled, nvfp4_path_tally, set_nvfp4_gemv, Nvfp4PathTally, NVFP4_GEMV_ENV,
};
#[doc(hidden)]
pub use nvfp4_path::{nvfp4_gemv_policy_guard, Nvfp4GemvPolicyGuard};
pub use paged_kv_cache::{BlockPool, PagedKvCache};
pub use prism::{GdnRowMap, PrismPackedWeight, PrismRegistry};
pub use projection::{
    KvProjection, Projection, ProjectionCensus, ProjectionFormat, ProjectionKind, ProjectionTally,
    QuantSpec, WeightCensus,
};
pub use quant::QuantizedLinear;
pub use rope::{apply_rope, rms_norm_rope, Rope};
pub use sampler::{
    device_sampler_available, sample, sample_device, sample_host, sampler_path, shaped_candidates,
    uniform_device, with_reference_sampler, HostSampleReason, SamplerPath, SamplingParams,
    SplitMix64, TokenRng,
};
pub use step_kv_cache::{KvLayout, LayerKvShape, StepKvCache};
pub use switch::{ProcessSwitch, SwitchGuard};
pub use weights::Weights;

/// Every kernel source `candle-llm` compiles through the shared nvrtc seam
/// ([`candle_quant_kernels::nvrtc`]): the Prism packed operators and the device sampler.
///
/// With [`candle_quant_kernels::NVRTC_SOURCES`] this is every runtime-compiled kernel in the
/// workspace. The CUDA test `cuda_nvrtc_kernels_use_no_local_memory` walks both lists (every
/// entry point of every source, as compiled for the live device), and
/// `every_workspace_kernel_source_is_registered` fails when a `KernelSource` anywhere under
/// `crates/` is listed in neither (sc-24164).
pub const NVRTC_SOURCES: &[candle_quant_kernels::KernelSource] =
    &[prism::PRISM_SRC, sampler::SAMPLER_SRC];

#[cfg(test)]
mod nvrtc_source_tests {
    use super::NVRTC_SOURCES;
    use candle_quant_kernels::KernelSource;
    use std::path::{Path, PathBuf};

    /// Every source registered for the all-kernels checks: this crate's and `candle-quant-kernels`'.
    fn registered() -> impl Iterator<Item = &'static KernelSource> {
        candle_quant_kernels::NVRTC_SOURCES
            .iter()
            .chain(NVRTC_SOURCES)
    }

    /// The `.rs` files under `dir`, leaving out integration-test, bench and example targets.
    fn rust_files(dir: &Path, files: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            if path.is_dir() {
                if !name.starts_with('.')
                    && !matches!(name.as_str(), "target" | "tests" | "benches" | "examples")
                {
                    rust_files(&path, files);
                }
            } else if name.ends_with(".rs") {
                files.push(path);
            }
        }
    }

    /// A file's production code: everything before its first inline `#[cfg(test)]` module.
    fn production_part(text: &str) -> &str {
        let mut offset = 0;
        let mut test_attribute_at = None;
        for line in text.split_inclusive('\n') {
            let trimmed = line.trim_start();
            if trimmed.starts_with("#[cfg(test)]") || trimmed.starts_with("#[cfg(all(test") {
                test_attribute_at.get_or_insert(offset);
            } else if let Some(at) = test_attribute_at {
                let declares_module = ["mod ", "pub mod ", "pub(crate) mod ", "pub(super) mod "]
                    .iter()
                    .any(|prefix| trimmed.starts_with(prefix));
                if declares_module && !trimmed.trim_end().ends_with(';') {
                    return &text[..at];
                }
                if !trimmed.starts_with("#[") && !trimmed.starts_with("//") {
                    test_attribute_at = None;
                }
            }
            offset += line.len();
        }
        text
    }

    /// The text up to the first `}` outside a string literal.
    fn braced_body(text: &str) -> &str {
        let mut in_string = false;
        let mut escaped = false;
        for (at, character) in text.char_indices() {
            match character {
                _ if escaped => escaped = false,
                '\\' if in_string => escaped = true,
                '"' => in_string = !in_string,
                '}' if !in_string => return &text[..at],
                _ => {}
            }
        }
        text
    }

    /// The `name` of every `KernelSource { .. }` struct literal in `text` (`None` for a literal
    /// whose name is not a string literal, which the caller reports). A literal is a braced body
    /// setting all three fields; the struct's own definition, its `impl` blocks and functions
    /// returning one are not.
    fn declared_names(text: &str) -> Vec<Option<String>> {
        const LITERAL: &str = "KernelSource {";
        let mut names = Vec::new();
        let mut rest = text;
        while let Some(at) = rest.find(LITERAL) {
            let is_definition = rest[..at].trim_end().ends_with("struct");
            rest = &rest[at + LITERAL.len()..];
            let body = braced_body(rest);
            let sets_every_field = ["name:", "src:", "cc_floor:"]
                .iter()
                .all(|field| body.contains(field));
            if is_definition || !sets_every_field {
                continue;
            }
            names.push(
                body.split_once("name:")
                    .and_then(|(_, value)| value.trim_start().strip_prefix('"'))
                    .and_then(|value| value.split_once('"'))
                    .map(|(name, _)| name.to_string()),
            );
        }
        names
    }

    #[test]
    fn declared_names_reads_struct_literals_only() {
        let text = r#"
            pub struct KernelSource {
                pub name: &'static str,
                pub src: &'static str,
                pub cc_floor: (i32, i32),
            }
            impl KernelSource {
                fn key(&self) -> &str { self.name }
            }
            fn source() -> KernelSource {
                SRC
            }
            const A: KernelSource = KernelSource {
                name: "a_v1",
                src: "__global__ void k(char* s) { s[0] = '\"'; }",
                cc_floor: (7, 0),
            };
            const B: crate::nvrtc::KernelSource =
                crate::nvrtc::KernelSource { name: NAME, src: include_str!("b.cu"), cc_floor: (8, 0) };
        "#;
        assert_eq!(declared_names(text), [Some("a_v1".to_string()), None]);
    }

    #[test]
    fn production_part_stops_at_the_inline_test_module_only() {
        let text = "const A: u8 = 1;\n#[cfg(test)]\nmod tests;\nconst B: u8 = 2;\n\
                    #[cfg(test)]\nthread_local! {}\nconst C: u8 = 3;\n\
                    #[cfg(all(test, feature = \"cuda\"))]\n// note\nmod cuda_tests {\n}\n";
        let production = production_part(text);
        assert!(production.contains("const C"), "{production}");
        assert!(!production.contains("cuda_tests"), "{production}");
    }

    /// sc-24164: every `KernelSource` in the workspace's production code is registered, so the
    /// all-kernels checks cover a new kernel the day it lands (and no registered source is stale).
    #[test]
    fn every_workspace_kernel_source_is_registered() {
        let crates = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut files = Vec::new();
        rust_files(&crates, &mut files);
        let mut declared = Vec::new();
        for file in &files {
            let text = String::from_utf8_lossy(&std::fs::read(file).unwrap()).into_owned();
            if text.contains("KernelSource {") {
                for name in declared_names(production_part(&text)) {
                    declared.push((name, file.clone()));
                }
            }
        }
        let registered: Vec<&str> = registered().map(|source| source.name).collect();
        let unregistered: Vec<_> = declared
            .iter()
            .filter(|(name, _)| {
                !name
                    .as_deref()
                    .is_some_and(|name| registered.contains(&name))
            })
            .collect();
        assert!(
            unregistered.is_empty(),
            "nvrtc KernelSource(s) listed in neither candle_quant_kernels::NVRTC_SOURCES nor \
             candle_llm::primitives::NVRTC_SOURCES, so no all-kernels check covers them: \
             {unregistered:?}"
        );
        let mut declared: Vec<&str> = declared
            .iter()
            .flat_map(|(name, _)| name.as_deref())
            .collect();
        let mut registered = registered;
        declared.sort_unstable();
        registered.sort_unstable();
        assert_eq!(
            declared, registered,
            "every registered source is declared exactly once"
        );
    }

    /// sc-24164: the seam compiles every kernel for the device's own architecture (`compute_120`
    /// on Blackwell since sc-24137), and there an indexed local array spills to per-thread local
    /// memory. Prism's PTQ `pow3[5]` table cost 24 B/thread, which made `prism_ptq_matmul_f32`
    /// 3.2x slower and Bonsai GGUF prefill 2.5x slower. No kernel compiled through the seam may
    /// use local memory: this checks every entry point of every registered source, as compiled
    /// for the live device.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_nvrtc_kernels_use_no_local_memory() {
        let device = crate::device::new_cuda_for_test().expect("cuda device");
        let dev = device.as_cuda_device().unwrap();
        let mut checked = Vec::new();
        let mut spills = Vec::new();
        for source in registered() {
            let kernel = match source.compiled(dev) {
                Ok(kernel) => kernel,
                Err(error @ candle_quant_kernels::KernelCompileError::BelowComputeFloor { .. }) => {
                    eprintln!("[nvrtc-local] skipping {}: {error}", source.name);
                    continue;
                }
                Err(error) => panic!("{error}"),
            };
            assert!(
                !kernel.entry_points().is_empty(),
                "{} defines no kernel entry point",
                source.name
            );
            for entry in kernel.entry_points() {
                let function = kernel.function(entry).unwrap();
                let local = function.local_size_bytes().unwrap();
                eprintln!(
                    "[nvrtc-local] {}::{entry}: {local} B local, {} registers",
                    source.name,
                    function.num_regs().unwrap()
                );
                if local != 0 {
                    spills.push(format!("{}::{entry}: {local} B/thread", source.name));
                }
                checked.push(entry.clone());
            }
        }
        assert!(
            checked.iter().any(|entry| entry == "prism_ptq_matmul_f32"),
            "the walk missed the Prism PTQ matmul: {checked:?}"
        );
        assert!(
            spills.is_empty(),
            "kernels spill to per-thread local memory under the seam's architecture: {spills:?}"
        );
    }
}
