use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const MLX_REVISION: &str = "0123456789abcdef0123456789abcdef01234567";
const CLEAN_SOURCE: &str = r#"const PAYLOAD: &str = "clean";

fn main() {
    let source_dirty = env!("SCENEWORKS_BENCH_SOURCE_DIRTY");
    if source_dirty == "true" {
        eprintln!(
            "refused source_dirty={} revision={} payload={}",
            source_dirty,
            env!("SCENEWORKS_BENCH_SOURCE_REVISION"),
            PAYLOAD
        );
        std::process::exit(86);
    }
    println!(
        "accepted source_dirty={} revision={} payload={}",
        source_dirty,
        env!("SCENEWORKS_BENCH_SOURCE_REVISION"),
        PAYLOAD
    );
}
"#;

fn run(command: &mut Command, description: &str) -> Output {
    let output = command
        .output()
        .unwrap_or_else(|error| panic!("start {description}: {error}"));
    assert!(
        output.status.success(),
        "{description} failed with {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

fn git(root: &Path, args: &[&str]) -> Output {
    run(
        Command::new("git").arg("-C").arg(root).args(args),
        &format!("git {}", args.join(" ")),
    )
}

fn cargo(root: &Path, cargo_home: &Path, target_dir: &Path, args: &[&str]) -> Output {
    run(
        Command::new(env!("CARGO"))
            .current_dir(root)
            .env("CARGO_HOME", cargo_home)
            .env("CARGO_TARGET_DIR", target_dir)
            .env("CARGO_NET_OFFLINE", "true")
            .env("CARGO_TERM_COLOR", "never")
            .env_remove("RUSTFLAGS")
            .env_remove("CARGO_ENCODED_RUSTFLAGS")
            .args(args),
        &format!("cargo {}", args.join(" ")),
    )
}

fn run_artifact(binary: &Path) -> Output {
    Command::new(binary)
        .output()
        .unwrap_or_else(|error| panic!("run {}: {error}", binary.display()))
}

fn fixture_was_fresh(output: &Output) -> bool {
    String::from_utf8_lossy(&output.stderr)
        .lines()
        .any(|line| line.trim_start().starts_with("Fresh receipt-fixture "))
}

fn assert_accepted(output: &Output, revision: &str, payload: &str) {
    assert!(
        output.status.success(),
        "clean receipt was refused: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("accepted source_dirty=false"), "{stdout}");
    assert!(stdout.contains(&format!("revision={revision}")), "{stdout}");
    assert!(stdout.contains(&format!("payload={payload}")), "{stdout}");
}

fn assert_refused(output: &Output, revision: &str, payload: &str) {
    assert_eq!(output.status.code(), Some(86));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("refused source_dirty=true"), "{stderr}");
    assert!(stderr.contains(&format!("revision={revision}")), "{stderr}");
    assert!(stderr.contains(&format!("payload={payload}")), "{stderr}");
}

#[test]
fn source_only_edit_rebuilds_dirty_receipt_that_stays_refused_after_revert() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("repository");
    let manifest = root.join("crates/bundles/runtime-macos");
    let cargo_home = temp.path().join("cargo-home");
    let target_dir = temp.path().join("target");
    fs::create_dir_all(manifest.join("src")).unwrap();
    fs::create_dir(&cargo_home).unwrap();

    fs::write(
        root.join("Cargo.toml"),
        "[workspace]\nmembers = [\"crates/bundles/runtime-macos\"]\nresolver = \"2\"\n",
    )
    .unwrap();
    fs::write(
        manifest.join("Cargo.toml"),
        "[package]\nname = \"receipt-fixture\"\nversion = \"0.0.0\"\nedition = \"2021\"\nbuild = \"build.rs\"\n\n[features]\ndefault = []\nperf-bench = []\n",
    )
    .unwrap();
    fs::copy(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("build.rs"),
        manifest.join("build.rs"),
    )
    .unwrap();
    let source = manifest.join("src/main.rs");
    fs::write(&source, CLEAN_SOURCE).unwrap();

    cargo(
        &root,
        &cargo_home,
        &target_dir,
        &["generate-lockfile", "--offline"],
    );
    writeln!(
        OpenOptions::new()
            .append(true)
            .open(root.join("Cargo.lock"))
            .unwrap(),
        "\n# git+https://github.com/michaeltrefry/mlx-rs?rev={MLX_REVISION}#{MLX_REVISION}"
    )
    .unwrap();

    git(&root, &["init", "--quiet"]);
    git(&root, &["add", "--all"]);
    git(
        &root,
        &[
            "-c",
            "user.name=receipt-test",
            "-c",
            "user.email=receipt-test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ],
    );

    // `build.rs` watches both packed-refs and the loose current-branch ref. Make both exist so a
    // missing metadata path cannot spuriously rerun the script and conceal a disconnected source
    // watch in this regression.
    git(&root, &["pack-refs", "--all", "--prune"]);
    git(
        &root,
        &[
            "-c",
            "user.name=receipt-test",
            "-c",
            "user.email=receipt-test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "--allow-empty",
            "-m",
            "loose-head-ref",
        ],
    );
    assert!(root.join(".git/packed-refs").is_file());
    let head = fs::read_to_string(root.join(".git/HEAD")).unwrap();
    let head_ref = head.trim().strip_prefix("ref: ").unwrap();
    assert!(root.join(".git").join(head_ref).is_file());
    assert!(
        git(&root, &["status", "--porcelain", "--untracked-files=all"])
            .stdout
            .is_empty()
    );

    let revision = String::from_utf8(git(&root, &["rev-parse", "HEAD"]).stdout)
        .unwrap()
        .trim()
        .to_owned();
    let build_args = [
        "build",
        "--offline",
        "--locked",
        "--no-default-features",
        "--verbose",
        "-p",
        "receipt-fixture",
        "--features",
        "perf-bench",
    ];
    // The second unchanged build must be genuinely Fresh. In particular, build.rs may not rewrite
    // its own watched Git index while determining the source dirty receipt.
    cargo(&root, &cargo_home, &target_dir, &build_args);
    let clean_build = cargo(&root, &cargo_home, &target_dir, &build_args);
    assert!(
        fixture_was_fresh(&clean_build),
        "could not establish a fully fresh Cargo baseline before the source-only edit:\n{}",
        String::from_utf8_lossy(&clean_build.stderr)
    );
    let binary = target_dir
        .join("debug")
        .join(format!("receipt-fixture{}", std::env::consts::EXE_SUFFIX));
    assert_accepted(&run_artifact(&binary), &revision, "clean");

    // Do not run Git between this source-only edit and Cargo: refreshing the index would itself
    // touch another watched input and could make a disconnected source-watch loop look correct.
    let edited_source = CLEAN_SOURCE.replace(
        "const PAYLOAD: &str = \"clean\";",
        "const PAYLOAD: &str = \"edited\";",
    );
    assert_ne!(edited_source, CLEAN_SOURCE);
    fs::write(&source, edited_source).unwrap();
    cargo(&root, &cargo_home, &target_dir, &build_args);
    let dirty_artifact = fs::read(&binary).unwrap();
    assert_refused(&run_artifact(&binary), &revision, "edited");

    // Restore the checkout without asking Cargo to rebuild. The directly executed artifact must
    // retain its dirty build receipt and refuse even though the runtime checkout is now clean.
    fs::write(&source, CLEAN_SOURCE).unwrap();
    assert_eq!(fs::read_to_string(&source).unwrap(), CLEAN_SOURCE);
    assert!(
        git(&root, &["status", "--porcelain", "--untracked-files=all"])
            .stdout
            .is_empty()
    );
    assert_eq!(fs::read(&binary).unwrap(), dirty_artifact);
    assert_refused(&run_artifact(&binary), &revision, "edited");
}
