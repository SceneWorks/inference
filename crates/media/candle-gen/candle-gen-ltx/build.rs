use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn git(repo: &Path, args: &[&str]) -> String {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(args)
        .output()
        .expect("run git for LTX quant build identity");
    assert!(
        output.status.success(),
        "git {} failed while binding LTX quant build identity: {}",
        args.join(" "),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .expect("git output is UTF-8")
        .trim()
        .to_owned()
}

fn collect_rs(dir: &Path, files: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("read source directory") {
        let path = entry.expect("read source entry").path();
        if path.is_dir() {
            collect_rs(&path, files);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            files.push(path);
        }
    }
}

fn main() {
    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let repo = PathBuf::from(git(&manifest, &["rev-parse", "--show-toplevel"]));
    let revision = git(&repo, &["rev-parse", "HEAD"]);
    assert!(
        revision.len() == 40 && revision.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "git HEAD is not a 40-character revision"
    );
    let git_head = PathBuf::from(git(&repo, &["rev-parse", "--git-path", "HEAD"]));
    let git_head = if git_head.is_absolute() {
        git_head
    } else {
        repo.join(git_head)
    };
    println!("cargo:rerun-if-changed={}", git_head.display());

    // This digest is the immutable executable *contract*: all LTX source plus the shared operator
    // implementations it dispatches to and the dependency lock. The runtime additionally hashes the
    // exact executable bytes. Together they avoid the old runtime-CWD-git claim (which could name a
    // different checkout than the process that was actually executing).
    let mut files = vec![
        repo.join("Cargo.lock"),
        manifest.join("Cargo.toml"),
        manifest.join("build.rs"),
    ];
    collect_rs(&manifest.join("src"), &mut files);
    collect_rs(&manifest.join("examples"), &mut files);
    for name in [
        "convrot.rs",
        "eight_bit_linear.rs",
        "nvfp4.rs",
        "nvfp4_linear.rs",
        "cublaslt.rs",
    ] {
        files.push(
            repo.join("crates/media/candle-gen/candle-gen/src/quant")
                .join(name),
        );
    }
    files.sort();
    files.dedup();
    let mut digest = Sha256::new();
    for path in files {
        let relative = path.strip_prefix(&repo).expect("contract file under repo");
        let bytes = fs::read(&path).unwrap_or_else(|error| {
            panic!(
                "read executable-contract source {}: {error}",
                path.display()
            )
        });
        println!("cargo:rerun-if-changed={}", path.display());
        let name = relative.to_string_lossy();
        digest.update((name.len() as u64).to_le_bytes());
        digest.update(name.as_bytes());
        digest.update((bytes.len() as u64).to_le_bytes());
        digest.update(bytes);
    }
    let contract = format!("{:x}", digest.finalize());
    let dirty = !git(&repo, &["status", "--porcelain", "--untracked-files=no"]).is_empty();
    println!("cargo:rustc-env=LTX25_BUILD_INFERENCE_REVISION={revision}");
    println!("cargo:rustc-env=LTX25_EXECUTABLE_CONTRACT_SHA256={contract}");
    println!(
        "cargo:rustc-env=LTX25_BUILD_SOURCE_DIRTY={}",
        if dirty { "1" } else { "0" }
    );
}
