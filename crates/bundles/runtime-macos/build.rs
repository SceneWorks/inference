use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn command(root: &Path, args: &[&str]) -> String {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .unwrap_or_else(|error| panic!("start git {}: {error}", args.join(" ")));
    assert!(
        output.status.success(),
        "git {} failed: {}",
        args.join(" "),
        String::from_utf8_lossy(&output.stderr).trim()
    );
    String::from_utf8(output.stdout)
        .expect("git output must be UTF-8")
        .trim()
        .to_owned()
}

fn mlx_revision(lockfile: &Path) -> String {
    let lock = fs::read_to_string(lockfile)
        .unwrap_or_else(|error| panic!("read {}: {error}", lockfile.display()));
    let marker = "git+https://github.com/michaeltrefry/mlx-rs?rev=";
    let source = lock
        .lines()
        .find(|line| line.contains(marker))
        .expect("Cargo.lock must contain the pinned pmetal mlx-rs source");
    let revision = source
        .split("?rev=")
        .nth(1)
        .and_then(|tail| tail.split('#').next())
        .expect("parse mlx-rs revision from Cargo.lock");
    assert!(
        revision.len() == 40 && revision.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "Cargo.lock mlx-rs revision must be a full SHA"
    );
    revision.to_ascii_lowercase()
}

fn main() {
    if env::var_os("CARGO_FEATURE_PERF_BENCH").is_none() {
        return;
    }

    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let root = manifest
        .ancestors()
        .nth(3)
        .expect("runtime-macos must live under crates/bundles")
        .to_path_buf();
    let lockfile = root.join("Cargo.lock");
    println!("cargo:rerun-if-changed={}", lockfile.display());

    let git_dir = PathBuf::from(command(&root, &["rev-parse", "--absolute-git-dir"]));
    for path in [git_dir.join("HEAD"), git_dir.join("index")] {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    let common_dir = {
        let path = PathBuf::from(command(&root, &["rev-parse", "--git-common-dir"]));
        if path.is_absolute() {
            path
        } else {
            root.join(path)
        }
    };
    println!(
        "cargo:rerun-if-changed={}",
        common_dir.join("packed-refs").display()
    );
    if let Ok(head) = fs::read_to_string(git_dir.join("HEAD")) {
        if let Some(reference) = head.trim().strip_prefix("ref: ") {
            println!(
                "cargo:rerun-if-changed={}",
                common_dir.join(reference).display()
            );
        }
    }

    let source_revision = command(&root, &["rev-parse", "HEAD"]);
    let dirty = !command(&root, &["status", "--porcelain", "--untracked-files=all"]).is_empty();
    println!("cargo:rustc-env=SCENEWORKS_BENCH_SOURCE_REVISION={source_revision}");
    println!(
        "cargo:rustc-env=SCENEWORKS_BENCH_MLX_REVISION={}",
        mlx_revision(&lockfile)
    );
    println!("cargo:rustc-env=SCENEWORKS_BENCH_SOURCE_DIRTY={dirty}");
}
