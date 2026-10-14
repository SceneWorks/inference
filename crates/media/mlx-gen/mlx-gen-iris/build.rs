//! Link MLX's JACCL distributed backend (sc-25689).
//!
//! The Iris trainer's data-parallel mode calls `mlx::core::distributed::init`, whose backend switch
//! references `::jaccl::init` / `::jaccl::is_available` (MLX's RDMA-over-Thunderbolt backend, a
//! separate `libjaccl.a`). `mlx-sys` links only `libmlx.a` + `libmlxc.a`, so a source-built MLX
//! (whose build directory holds `libjaccl.a` beside `mlx.metallib`) gets JACCL linked here; a
//! prebuilt MLX cell that ships without `libjaccl.a` gets `jaccl_absent.cpp` — JACCL reports
//! unavailable (the trainer refuses `transport: mlx_jaccl` typed) and `ring` / `mpi` are unaffected.

use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=DEP_MLX_METALLIB");
    println!("cargo:rerun-if-changed=src/train/jaccl_absent.cpp");
    let lib_dir = std::env::var_os("DEP_MLX_METALLIB")
        .map(PathBuf::from)
        .and_then(|p| p.parent().map(|d| d.to_path_buf()));
    match lib_dir.filter(|d| d.join("libjaccl.a").is_file()) {
        Some(dir) => {
            println!("cargo:rustc-link-search=native={}", dir.display());
            println!("cargo:rustc-link-lib=static=jaccl");
            println!("cargo:rustc-cfg=iris_jaccl_linked");
        }
        None => {
            cc::Build::new()
                .cpp(true)
                .std("c++17")
                .file("src/train/jaccl_absent.cpp")
                .compile("iris_jaccl_absent");
        }
    }
    println!("cargo:rustc-check-cfg=cfg(iris_jaccl_linked)");
}
