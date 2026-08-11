#[allow(dead_code)]
#[path = "../build.rs"]
mod build_script;

use std::path::Path;
use std::process::Command;

fn git(root: &Path, args: &[&str]) {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "git {} failed: {}",
        args.join(" "),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn tracked_rust_edit_is_a_receipt_input_and_marks_source_dirty() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    std::fs::create_dir(root.join("src")).unwrap();
    std::fs::write(
        root.join("Cargo.toml"),
        "[package]\nname = \"receipt-fixture\"\nversion = \"0.0.0\"\n",
    )
    .unwrap();
    let source = root.join("src/main.rs");
    std::fs::write(&source, "fn main() {}\n").unwrap();
    git(root, &["init", "--quiet"]);
    git(root, &["add", "Cargo.toml", "src/main.rs"]);
    git(
        root,
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

    let inputs = build_script::tracked_build_inputs(root);
    assert!(inputs.contains(&source));
    let (clean_revision, clean_dirty) = build_script::source_state(root);
    assert_eq!(clean_revision.len(), 40);
    assert!(!clean_dirty);

    // This is the P1 reproduction: only executable Rust changes. Cargo must rerun build.rs because
    // the source is one of its explicit inputs, and the recomputed receipt must refuse acceptance.
    std::fs::write(&source, "fn main() { println!(\"changed\"); }\n").unwrap();
    let (edited_revision, edited_dirty) = build_script::source_state(root);
    assert_eq!(edited_revision, clean_revision);
    assert!(edited_dirty);
}
