//! Deterministic execution-source identity, independent of Git HEAD and evidence/policy data.
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use toml_edit::{DocumentMut, Item};

fn document(path: &Path) -> DocumentMut {
    fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
        .parse()
        .unwrap_or_else(|e| panic!("cannot parse {}: {e}", path.display()))
}

fn collect(path: &Path, files: &mut BTreeSet<PathBuf>) {
    if path.is_dir() {
        // Watch directories too, so newly added source files invalidate Cargo's cached identity.
        println!("cargo:rerun-if-changed={}", path.display());
        for entry in fs::read_dir(path).expect("read source directory") {
            collect(&entry.expect("source directory entry").path(), files);
        }
    } else if path.is_file() {
        if path.extension().is_some_and(|e| e == "md") {
            return;
        }
        files.insert(path.to_path_buf());
    }
}

fn hash_file(digest: &mut Sha256, label: &str, path: &Path) {
    println!("cargo:rerun-if-changed={}", path.display());
    digest.update(label.as_bytes());
    digest.update([0]);
    let mut file = fs::File::open(path).expect("open identity input");
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer).expect("read identity input");
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    digest.update([255]);
}

fn dependencies(
    table: &Item,
    directory: &Path,
    workspace: &DocumentMut,
    root: &Path,
    pending: &mut Vec<PathBuf>,
) {
    let Some(table) = table.as_table_like() else {
        return;
    };
    for (name, dependency) in table.iter() {
        // The supported iOS graph has no optional inference dependency except its catalog,
        // seeded explicitly below. The optional provider edges are trainers, omitted on iOS.
        if dependency.get("optional").and_then(Item::as_bool) == Some(true) {
            continue;
        }
        let (dependency, base) =
            if dependency.get("workspace").and_then(Item::as_bool) == Some(true) {
                (&workspace["workspace"]["dependencies"][name], root)
            } else {
                (dependency, directory)
            };
        if let Some(path) = dependency.get("path").and_then(Item::as_str) {
            pending.push(base.join(path));
        }
    }
}

fn source_closure(root: &Path) -> String {
    let workspace = document(&root.join("Cargo.toml"));
    let mut pending = vec![
        root.join("crates/bundles/runtime-ios"),
        root.join("crates/media/mlx-gen/mlx-gen-ios-catalog"),
    ];
    // Local registry patches affect CLIP tokenization in the same execution graph.
    if let Some(patches) = workspace["patch"]["crates-io"].as_table_like() {
        for (_, patch) in patches.iter() {
            if let Some(path) = patch.get("path").and_then(Item::as_str) {
                pending.push(root.join(path));
            }
        }
    }
    let mut visited = BTreeSet::new();
    let mut files = BTreeSet::new();
    while let Some(directory) = pending.pop() {
        let directory = directory
            .canonicalize()
            .expect("local dependency directory");
        assert!(
            directory.starts_with(root),
            "local dependency escapes repository"
        );
        if !visited.insert(directory.clone()) {
            continue;
        }
        let manifest = directory.join("Cargo.toml");
        let parsed = document(&manifest);
        collect(&manifest, &mut files);
        for relative in ["src", "assets", "build.rs", ".cargo/config.toml"] {
            collect(&directory.join(relative), &mut files);
        }
        for table in ["dependencies", "build-dependencies"] {
            dependencies(
                parsed.get(table).unwrap_or(&Item::None),
                &directory,
                &workspace,
                root,
                &mut pending,
            );
        }
        if let Some(targets) = parsed.get("target").and_then(Item::as_table_like) {
            // A conservative superset of target-conditioned path dependencies.
            for (_, target) in targets.iter() {
                for table in ["dependencies", "build-dependencies"] {
                    dependencies(
                        target.get(table).unwrap_or(&Item::None),
                        &directory,
                        &workspace,
                        root,
                        &mut pending,
                    );
                }
            }
        }
    }
    for relative in [
        "Cargo.toml",
        "Cargo.lock",
        "rust-toolchain.toml",
        ".cargo/config.toml",
        "ios-host/smoke/Cargo.toml",
        "ios-host/smoke/Cargo.lock",
        "ios-host/smoke/src",
        "scripts/ios/run_smoke.sh",
        "scripts/ios/bundle_metallib.py",
    ] {
        let path = root.join(relative);
        assert!(path.exists(), "missing iOS execution input {relative}");
        collect(&path, &mut files);
    }
    // Only src/assets/build inputs are traversed: evidence/*.json, including the reviewed
    // ios_engineering_policy.json, and documentation are data, never part of this closure.
    // Their parsers, admission rules, and this hash algorithm ARE included verbatim.
    let mut digest = Sha256::new();
    digest.update(b"sceneworks-ios-execution-source-v1\0");
    let named_files: BTreeMap<_, _> = files
        .into_iter()
        .map(|path| {
            let relative = path.strip_prefix(root).expect("source within repository");
            let label = relative
                .to_str()
                .expect("UTF-8 source path")
                .replace('\\', "/");
            (label, path)
        })
        .collect();
    for (label, path) in named_files {
        hash_file(&mut digest, &label, &path);
    }
    format!("{:x}", digest.finalize())
}

fn native_identity() -> String {
    // mlx-sys's direct `links=mlx` metadata identifies this build's library directory.
    // Never infer it by scanning target/ or a shared prebuilt cache.
    println!("cargo:rerun-if-env-changed=DEP_MLX_METALLIB");
    let Some(directory) = env::var_os("DEP_MLX_METALLIB")
        .map(PathBuf::from)
        .and_then(|p| p.parent().map(Path::to_path_buf))
    else {
        return "unavailable".into();
    };
    let inputs = [
        "pmetal-mlx-prebuilt.txt",
        "libmlx.a",
        "libmlxc.a",
        "mlx.metallib",
    ];
    for input in inputs {
        println!("cargo:rerun-if-changed={}", directory.join(input).display());
        if !directory.join(input).is_file() {
            return "unavailable".into();
        }
    }
    let manifest = fs::read_to_string(directory.join(inputs[0])).expect("native build manifest");
    if !manifest.lines().any(|line| {
        line.strip_prefix("fingerprint=")
            .is_some_and(|f| f.len() == 64 && f.bytes().all(|b| b.is_ascii_hexdigit()))
    }) {
        return "unavailable".into();
    }
    let mut digest = Sha256::new();
    for input in inputs {
        hash_file(&mut digest, input, &directory.join(input));
    }
    format!("{:x}", digest.finalize())
}

fn main() {
    let directory = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("Cargo manifest dir"));
    let root = directory
        .ancestors()
        .nth(4)
        .expect("inference root")
        .canonicalize()
        .expect("repository root");
    let rustc = env::var_os("RUSTC").expect("Cargo rustc");
    println!("cargo:rerun-if-env-changed=RUSTC");
    let output = Command::new(&rustc)
        .arg("-vV")
        .output()
        .expect("rustc identity");
    assert!(output.status.success(), "rustc identity failed");
    let rustc = String::from_utf8(output.stdout).expect("UTF-8 rustc identity");
    let target = env::var("TARGET").expect("Cargo target");
    let features: BTreeSet<_> = env::vars()
        .filter_map(|(key, _)| {
            key.strip_prefix("CARGO_FEATURE_")
                .map(|f| f.to_ascii_lowercase().replace('_', "-"))
        })
        .collect();
    let mut config = BTreeMap::new();
    for key in [
        "PROFILE",
        "OPT_LEVEL",
        "DEBUG",
        "CARGO_CFG_TARGET_FEATURE",
        "CARGO_ENCODED_RUSTFLAGS",
        "RUSTFLAGS",
        "RUSTC_WRAPPER",
        "RUSTC_WORKSPACE_WRAPPER",
        "IPHONEOS_DEPLOYMENT_TARGET",
        "MACOSX_DEPLOYMENT_TARGET",
        "SDKROOT",
    ] {
        println!("cargo:rerun-if-env-changed={key}");
        config.insert(key, env::var(key).unwrap_or_default());
    }
    let config = format!("{:x}", Sha256::digest(format!("{config:?}").as_bytes()));
    let constants = [
        ("SOURCE_CLOSURE_SHA256", source_closure(&root)),
        ("BUILD_TARGET", target),
        ("BUILD_RUSTC", rustc),
        (
            "BUILD_FEATURES",
            features.into_iter().collect::<Vec<_>>().join(","),
        ),
        ("BUILD_CONFIG_SHA256", config),
        ("NATIVE_MLX_FINGERPRINT", native_identity()),
    ];
    let generated: String = constants
        .into_iter()
        .map(|(name, value)| format!("pub const {name}: &str = {value:?};\n"))
        .collect();
    fs::write(
        PathBuf::from(env::var_os("OUT_DIR").expect("Cargo output dir"))
            .join("ios_source_identity.rs"),
        generated,
    )
    .expect("write iOS source identity");
}
