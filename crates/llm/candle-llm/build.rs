//! Build provenance for the decode-perf bench (sc-24140 feature-end review).
//!
//! With `CANDLE_LLM_BUILD_PROVENANCE=1` in the build environment this embeds the checkout's `HEAD`
//! and whether its tree differed from it (`git status --porcelain`, untracked files included) as
//! `CANDLE_LLM_BUILD_GIT_SHA` / `CANDLE_LLM_BUILD_GIT_DIRTY` (`1` / `0`), which the `decode_bench`
//! test records and `scripts/release/decode_bench.py` checks against the runtime SHA it was given
//! and a clean tree. Without it both are set **empty**, so a value inherited from the environment
//! can never pose as provenance, and no `git` runs: an ordinary build (a product's, CI's) pays
//! nothing and is never rebuilt because a commit moved `HEAD`.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const SWITCH: &str = "CANDLE_LLM_BUILD_PROVENANCE";

fn git(repo: &Path, args: &[&str]) -> String {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(args)
        .output()
        .unwrap_or_else(|error| panic!("{SWITCH}=1 needs git: {error}"));
    assert!(
        output.status.success(),
        "{SWITCH}=1: git {} failed: {}",
        args.join(" "),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .expect("git output is UTF-8")
        .trim_end()
        .to_owned()
}

/// A `git rev-parse --git-path` answer, absolute.
fn git_path(repo: &Path, name: &str) -> PathBuf {
    let path = PathBuf::from(git(repo, &["rev-parse", "--git-path", name]));
    if path.is_absolute() {
        path
    } else {
        repo.join(path)
    }
}

fn main() {
    println!("cargo:rerun-if-env-changed={SWITCH}");
    if env::var(SWITCH).as_deref() != Ok("1") {
        println!("cargo:rustc-env=CANDLE_LLM_BUILD_GIT_SHA=");
        println!("cargo:rustc-env=CANDLE_LLM_BUILD_GIT_DIRTY=");
        return;
    }
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let repo = PathBuf::from(git(&manifest, &["rev-parse", "--show-toplevel"]));
    let sha = git(&repo, &["rev-parse", "HEAD"]);
    assert!(
        sha.len() == 40 && sha.bytes().all(|b| b.is_ascii_hexdigit()),
        "git HEAD is not a 40-hex commit: {sha:?}"
    );
    let dirty = !git(&repo, &["status", "--porcelain"]).is_empty();

    // Re-run when HEAD moves (a checkout, a commit on the current branch) or any source the
    // build compiles changes, so the embedded state follows the tree the binary is built from.
    println!(
        "cargo:rerun-if-changed={}",
        git_path(&repo, "HEAD").display()
    );
    let head = std::fs::read_to_string(git_path(&repo, "HEAD")).unwrap_or_default();
    if let Some(reference) = head.trim().strip_prefix("ref: ") {
        for path in [git_path(&repo, reference), git_path(&repo, "packed-refs")] {
            if path.exists() {
                println!("cargo:rerun-if-changed={}", path.display());
            }
        }
    }
    for path in ["crates", "Cargo.toml", "Cargo.lock"] {
        println!("cargo:rerun-if-changed={}", repo.join(path).display());
    }
    println!("cargo:rustc-env=CANDLE_LLM_BUILD_GIT_SHA={sha}");
    println!(
        "cargo:rustc-env=CANDLE_LLM_BUILD_GIT_DIRTY={}",
        if dirty { "1" } else { "0" }
    );
}
