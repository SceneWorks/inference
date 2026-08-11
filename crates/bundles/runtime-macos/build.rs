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

fn program_output(program: &str, args: &[&str]) -> String {
    let output = Command::new(program)
        .args(args)
        .output()
        .unwrap_or_else(|error| panic!("start {program} {}: {error}", args.join(" ")));
    assert!(
        output.status.success(),
        "{program} {} failed: {}",
        args.join(" "),
        String::from_utf8_lossy(&output.stderr).trim()
    );
    String::from_utf8(output.stdout)
        .expect("build-tool output must be UTF-8")
        .trim()
        .replace('\n', "; ")
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

/// Every tracked repository file can affect the benchmark executable through Rust compilation,
/// feature wiring, generated includes, or workspace dependency resolution. Once a build script emits
/// any `rerun-if-changed` directive Cargo stops applying its package-wide default, so enumerate the
/// complete tracked input set explicitly rather than watching only Cargo.lock and Git metadata.
pub(crate) fn tracked_build_inputs(root: &Path) -> Vec<PathBuf> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["ls-files", "-z", "--cached"])
        .output()
        .unwrap_or_else(|error| panic!("start git ls-files: {error}"));
    assert!(
        output.status.success(),
        "git ls-files failed: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    let mut paths = output
        .stdout
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
        .map(|path| {
            let relative = std::str::from_utf8(path)
                .expect("tracked repository paths must be UTF-8 for Cargo change tracking");
            root.join(relative)
        })
        .collect::<Vec<_>>();
    paths.sort();
    paths
}

pub(crate) fn source_state(root: &Path) -> (String, bool) {
    let revision = command(root, &["rev-parse", "HEAD"]);
    let dirty = !command(root, &["status", "--porcelain", "--untracked-files=all"]).is_empty();
    (revision, dirty)
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
    for path in tracked_build_inputs(&root) {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    println!("cargo:rerun-if-env-changed=RUSTFLAGS");
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");

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

    let (source_revision, dirty) = source_state(&root);
    println!("cargo:rustc-env=SCENEWORKS_BENCH_SOURCE_REVISION={source_revision}");
    println!(
        "cargo:rustc-env=SCENEWORKS_BENCH_MLX_REVISION={}",
        mlx_revision(&lockfile)
    );
    println!("cargo:rustc-env=SCENEWORKS_BENCH_SOURCE_DIRTY={dirty}");

    let mut features: Vec<_> = env::vars()
        .filter_map(|(name, _)| {
            name.strip_prefix("CARGO_FEATURE_")
                .map(|feature| feature.to_ascii_lowercase().replace('_', "-"))
        })
        .collect();
    features.sort();
    let mut target_features: Vec<_> = env::var("CARGO_CFG_TARGET_FEATURE")
        .unwrap_or_default()
        .split(',')
        .filter(|feature| !feature.is_empty())
        .map(str::to_owned)
        .collect();
    target_features.sort();
    let rustflags = env::var("CARGO_ENCODED_RUSTFLAGS")
        .unwrap_or_default()
        .split('\x1f')
        .filter(|flag| !flag.is_empty())
        .collect::<Vec<_>>()
        .join("\u{241f}");
    let rustc = env::var("RUSTC").expect("RUSTC");
    println!(
        "cargo:rustc-env=SCENEWORKS_BENCH_CARGO_PROFILE={}",
        env::var("PROFILE").expect("PROFILE")
    );
    println!(
        "cargo:rustc-env=SCENEWORKS_BENCH_OPT_LEVEL={}",
        env::var("OPT_LEVEL").expect("OPT_LEVEL")
    );
    println!(
        "cargo:rustc-env=SCENEWORKS_BENCH_TARGET={}",
        env::var("TARGET").expect("TARGET")
    );
    println!(
        "cargo:rustc-env=SCENEWORKS_BENCH_CARGO_FEATURES={}",
        features.join(",")
    );
    println!(
        "cargo:rustc-env=SCENEWORKS_BENCH_TARGET_FEATURES={}",
        target_features.join(",")
    );
    println!("cargo:rustc-env=SCENEWORKS_BENCH_RUSTFLAGS={rustflags}");
    println!(
        "cargo:rustc-env=SCENEWORKS_BENCH_RUSTC_VERSION={}",
        program_output(&rustc, &["--version", "--verbose"])
    );
}
