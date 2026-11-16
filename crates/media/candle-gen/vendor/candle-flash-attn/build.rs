// Build script to run nvcc and generate the C glue code for launching the flash-attention kernel.
// The cuda build time is very long so one can set the CANDLE_FLASH_ATTN_BUILD_DIR environment
// variable in order to cache the compiled artifacts and avoid recompiling too often.
use anyhow::Result;
use cudaforge::KernelBuilder;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Read;
use std::path::PathBuf;
use std::process::Command;
const CUTLASS_COMMIT: &str = "7d49e6c7e2f8896c47f586706e67e1fb215529dc";

const KERNEL_FILES: [&str; 37] = [
    "kernels/flash_api.cu",
    "kernels/flash_fwd_hdim128_fp16_sm80.cu",
    "kernels/flash_fwd_hdim160_fp16_sm80.cu",
    "kernels/flash_fwd_hdim192_fp16_sm80.cu",
    "kernels/flash_fwd_hdim224_fp16_sm80.cu",
    "kernels/flash_fwd_hdim256_fp16_sm80.cu",
    "kernels/flash_fwd_hdim512_fp16_sm80.cu",
    "kernels/flash_fwd_hdim32_fp16_sm80.cu",
    "kernels/flash_fwd_hdim64_fp16_sm80.cu",
    "kernels/flash_fwd_hdim96_fp16_sm80.cu",
    "kernels/flash_fwd_hdim128_bf16_sm80.cu",
    "kernels/flash_fwd_hdim160_bf16_sm80.cu",
    "kernels/flash_fwd_hdim192_bf16_sm80.cu",
    "kernels/flash_fwd_hdim224_bf16_sm80.cu",
    "kernels/flash_fwd_hdim256_bf16_sm80.cu",
    "kernels/flash_fwd_hdim512_bf16_sm80.cu",
    "kernels/flash_fwd_hdim32_bf16_sm80.cu",
    "kernels/flash_fwd_hdim64_bf16_sm80.cu",
    "kernels/flash_fwd_hdim96_bf16_sm80.cu",
    "kernels/flash_fwd_hdim128_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim160_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim192_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim224_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim256_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim512_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim32_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim64_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim96_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim128_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim160_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim192_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim224_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim256_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim512_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim32_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim64_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim96_bf16_causal_sm80.cu",
];

fn file_sha256(path: &std::path::Path) -> Result<String> {
    let mut input = fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = input.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

// Match scripts/ci/flash_attn_prebuilt.py. The source key includes every CUDA source/header,
// the CUTLASS pin and the flags in this build script, but not checkout-specific paths.
fn source_sha256() -> Result<String> {
    let mut files = vec![PathBuf::from("build.rs"), PathBuf::from("Cargo.toml")];
    for entry in fs::read_dir("kernels")? {
        let path = entry?.path();
        if path.is_file() {
            files.push(path);
        }
    }
    files.sort();
    let mut hash = Sha256::new();
    for path in files {
        hash.update(path.to_string_lossy().replace('\\', "/").as_bytes());
        hash.update([0]);
        hash.update(fs::read(path)?);
        hash.update([0]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn nvcc_version() -> Option<String> {
    let nvcc = std::env::var_os("NVCC")
        .map(PathBuf::from)
        .or_else(|| {
            ["CUDA_HOME", "CUDA_PATH"]
                .iter()
                .filter_map(|key| std::env::var_os(key))
                .map(|root| {
                    PathBuf::from(root).join("bin").join(if cfg!(windows) {
                        "nvcc.exe"
                    } else {
                        "nvcc"
                    })
                })
                .find(|path| path.is_file())
        })
        .unwrap_or_else(|| PathBuf::from("nvcc"));
    let output = Command::new(nvcc).arg("--version").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    text.split("release ")
        .nth(1)?
        .split(',')
        .next()
        .map(str::trim)
        .map(str::to_owned)
}

fn host_compiler_version(target: &str) -> Option<String> {
    let output = if target.contains("msvc") {
        Command::new("cl").output().ok()?
    } else {
        Command::new("g++").arg("-dumpfullversion").output().ok()?
    };
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    if target.contains("msvc") {
        text.split("Version ")
            .nth(1)?
            .split_whitespace()
            .next()
            .map(str::to_owned)
    } else {
        text.trim().split_whitespace().next().map(str::to_owned)
    }
}

fn link_prebuilt(target: &str) -> Result<bool> {
    println!("cargo::rerun-if-env-changed=CANDLE_FLASH_ATTN_PREBUILT_DIR");
    let Some(directory) = std::env::var_os("CANDLE_FLASH_ATTN_PREBUILT_DIR") else {
        return Ok(false);
    };
    let directory = PathBuf::from(directory);
    let manifest = directory.join("flash-attn-prebuilt.txt");
    let archive = directory.join("libflashattention.a");
    println!("cargo::rerun-if-changed={}", manifest.display());
    println!("cargo::rerun-if-changed={}", archive.display());
    if !manifest.is_file() || !archive.is_file() {
        println!("cargo::warning=FlashAttention prebuilt is incomplete; compiling from source");
        return Ok(false);
    }
    let manifest_text = fs::read_to_string(manifest)?;
    let fields: std::collections::HashMap<_, _> = manifest_text
        .lines()
        .filter_map(|line| line.split_once('='))
        .collect();
    let source = source_sha256()?;
    let cap = std::env::var("CUDA_COMPUTE_CAP").unwrap_or_default();
    let cuda_version = nvcc_version();
    let compiler_version = host_compiler_version(target);
    let matches = fields.get("format") == Some(&"1")
        && fields.get("source_sha256").copied() == Some(source.as_str())
        && fields.get("target").copied() == Some(target)
        // A cap-80 fatbin contains sm_80, sm_90, sm_120 and compute_120.
        // Only these known baselines keep candle-gen's source-derived dispatch guard honest.
        && fields.get("cuda_compute_cap") == Some(&"80")
        && (cap.is_empty() || cap == "80" || cap == "120")
        && cuda_version.as_deref().is_some()
        && fields.get("cuda_version").copied() == cuda_version.as_deref()
        && compiler_version.as_deref().is_some()
        && fields.get("host_compiler_version").copied() == compiler_version.as_deref();
    if !matches {
        println!("cargo::warning=FlashAttention prebuilt does not match source, target or toolchain; compiling from source");
        return Ok(false);
    }
    let expected = fields
        .get("archive_sha256")
        .ok_or_else(|| anyhow::anyhow!("prebuilt manifest lacks archive_sha256"))?;
    let actual = file_sha256(&archive)?;
    anyhow::ensure!(
        &actual == expected,
        "FlashAttention prebuilt archive checksum mismatch"
    );
    println!(
        "cargo::rustc-link-search={}",
        directory.canonicalize()?.display()
    );
    println!("cargo::rustc-link-lib=flashattention");
    println!("cargo::rustc-link-lib=dylib=cudart");
    if !target.contains("msvc") {
        println!("cargo::rustc-link-lib=dylib=stdc++");
    }
    println!("cargo::warning=linking verified prebuilt FlashAttention archive");
    Ok(true)
}

fn main() -> Result<()> {
    println!("cargo::rerun-if-changed=build.rs");
    for kernel_file in KERNEL_FILES.iter() {
        println!("cargo::rerun-if-changed={kernel_file}");
    }
    println!("cargo::rerun-if-changed=kernels/flash_fwd_kernel.h");
    println!("cargo::rerun-if-changed=kernels/flash_fwd_launch_template.h");
    println!("cargo::rerun-if-changed=kernels/flash.h");
    println!("cargo::rerun-if-changed=kernels/philox.cuh");
    println!("cargo::rerun-if-changed=kernels/softmax.h");
    println!("cargo::rerun-if-changed=kernels/utils.h");
    println!("cargo::rerun-if-changed=kernels/kernel_traits.h");
    println!("cargo::rerun-if-changed=kernels/block_info.h");
    println!("cargo::rerun-if-changed=kernels/static_switch.h");
    println!("cargo::rerun-if-changed=kernels/hardware_info.h");
    // sc-26691 (vendored delta 4): the list above misses several headers the kernels include
    // (alibi.h, dropout.h, mask.h, rotary.h, kernel_traits_sm90.h, ...). Watching the whole
    // directory reruns this script on ANY kernel-tree edit, and `.watch(["kernels"])` below folds
    // every header's CONTENT into cudaforge's per-object rebuild key, so a persistent
    // CANDLE_FLASH_ATTN_BUILD_DIR cache can never serve objects compiled against an older header.
    // (nvcc flags and the -gencode ladder are already in that key via cudaforge's args hash; the
    // CUDA_COMPUTE_CAP arch via its per-object arch.)
    println!("cargo::rerun-if-changed=kernels");
    println!("cargo::rerun-if-env-changed=CANDLE_FLASH_ATTN_BUILD_DIR");
    let target = std::env::var("TARGET")?;
    if link_prebuilt(&target)? {
        return Ok(());
    }
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set"));
    let build_dir = match std::env::var("CANDLE_FLASH_ATTN_BUILD_DIR") {
        Err(_) =>
        {
            #[allow(clippy::redundant_clone)]
            out_dir.clone()
        }
        Ok(build_dir) => {
            let path = PathBuf::from(build_dir);
            path.canonicalize().expect(&format!(
                "Directory doesn't exists: {} (the current directory is {})",
                &path.display(),
                std::env::current_dir()?.display()
            ))
        }
    };

    let kernels: Vec<_> = KERNEL_FILES.iter().collect();
    let mut builder = KernelBuilder::new()
        .source_files(kernels)
        .watch(["kernels"])
        .out_dir(&build_dir)
        .with_cutlass(Some(CUTLASS_COMMIT)) // ✅ Auto-fetch and include CUTLASS from GitHub
        .arg("-std=c++17")
        .arg("-O3")
        .arg("-U__CUDA_NO_HALF_OPERATORS__")
        .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
        .arg("-U__CUDA_NO_HALF2_OPERATORS__")
        .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
        .arg("--expt-relaxed-constexpr")
        .arg("--expt-extended-lambda")
        .arg("--use_fast_math")
        .arg("--verbose")
        // sc-26691 (vendored delta 1): a multi-arch fatbin. cudaforge prepends ONE `-gencode`
        // derived from CUDA_COMPUTE_CAP — `code=sm_80` at the SceneWorks packaging baseline
        // (cap=80), a SASS-only cubin with no PTX. These kernels are linked as a static library of
        // cubins, so on a Blackwell sm_120 card a cap=80 build has no image to launch. nvcc
        // accumulates `-gencode` flags, so the extras below add native sm_90 + sm_120 SASS and a
        // compute_120 PTX floor — the same ladder vendor/candle-kernels ships (sc-7544). The
        // runtime guard in candle-gen (`fused_attention`) reads these flags back through its
        // build script and only dispatches to the kernel on a device this ladder covers.
        .arg("-gencode=arch=compute_90,code=sm_90")
        .arg("-gencode=arch=compute_120,code=sm_120")
        .arg("-gencode=arch=compute_120,code=compute_120")
        .thread_percentage(0.5); // Use up to 50% of available threads

    let mut is_target_msvc = false;
    if let Ok(target) = std::env::var("TARGET") {
        if target.contains("msvc") {
            is_target_msvc = true;
            builder = builder.arg("-D_USE_MATH_DEFINES");
        }
    }

    if !is_target_msvc {
        builder = builder.arg("-Xcompiler").arg("-fPIC");
    }

    let out_file = build_dir.join("libflashattention.a");
    builder.build_lib(out_file)?;

    println!("cargo::rustc-link-search={}", build_dir.display());
    println!("cargo::rustc-link-lib=flashattention");
    println!("cargo::rustc-link-lib=dylib=cudart");
    if !is_target_msvc {
        println!("cargo::rustc-link-lib=dylib=stdc++");
    }
    Ok(())
}
