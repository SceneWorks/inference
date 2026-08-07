//! Shared test-support for the candle-llm integration suites.
//!
//! `tests/common/` is a module directory, not a test target, so this compiles into each suite that
//! declares `mod common;` rather than running as its own binary.

use std::path::{Path, PathBuf};

/// A temp fixture path that owns its `TempDir` guard (sc-17755).
///
/// Derefs (and `AsRef`s) to `Path`, so call sites read exactly as they did when these helpers
/// returned a bare `PathBuf` — but the tree now leaves on `Drop`, **including out of a panicking
/// test**, rather than depending on trailing `remove_dir_all(..).ok()` lines. Those ran only on the
/// happy path, and several fixtures had no cleanup line at all, which is how `candle-llm-*` entries
/// accumulated under `%TEMP%`.
///
/// Bind it for as long as the path is read: a dropped `Fixture` takes the directory with it.
pub struct Fixture {
    path: PathBuf,
    _guard: tempfile::TempDir,
}

impl Fixture {
    /// A fresh fixture root. `name`, when given, names a file *inside* that root to point at
    /// instead — the guard still owns the enclosing directory, so the file cannot outlive it.
    pub fn new(prefix: &str, name: Option<&str>) -> Self {
        let guard = tempfile::Builder::new()
            .prefix(prefix)
            .tempdir()
            .expect("fixture temp dir");
        let path = match name {
            Some(file) => guard.path().join(file),
            None => guard.path().to_path_buf(),
        };
        Self {
            path,
            _guard: guard,
        }
    }
}

impl std::ops::Deref for Fixture {
    type Target = Path;
    fn deref(&self) -> &Path {
        &self.path
    }
}

// `AsRef<Path>` as well as `Deref`: deref coercion does not fire for a generic `impl AsRef<Path>`
// parameter, which is what the `std::fs` entry points take.
impl AsRef<Path> for Fixture {
    fn as_ref(&self) -> &Path {
        &self.path
    }
}

/// Guards the sc-17755 fix for [`Fixture`] itself: the root, and any file the helper pointed at
/// inside it, leave with the value. Give the builder `disable_cleanup(true)` (or go back to a bare
/// `create_dir_all` on a `temp_dir()` join) and this goes RED.
///
/// This lives beside the type rather than in one suite because the leak it guards is invisible from
/// the assertions the suites actually make — a leaking fixture still passes every test that reads it.
#[test]
fn fixture_removes_its_tree_on_drop() {
    let (root, file) = {
        let fixture = Fixture::new("candle-llm-fixture-guard-", Some("inner.bin"));
        std::fs::write(&fixture, b"bytes").unwrap();
        let file = fixture.to_path_buf();
        let root = file.parent().unwrap().to_path_buf();
        assert!(file.is_file(), "fixture file not written");
        (root, file)
    };
    assert!(!file.exists(), "fixture file survived: {}", file.display());
    assert!(!root.exists(), "fixture root survived: {}", root.display());
}

/// Two fixtures with the same prefix get different roots — the property the old
/// `{prefix}{pid}` naming lacked, and the reason a recycled PID could reopen a previous run.
#[test]
fn fixtures_with_the_same_prefix_do_not_collide() {
    let (a, b) = (
        Fixture::new("candle-llm-fixture-guard-", None),
        Fixture::new("candle-llm-fixture-guard-", None),
    );
    assert_ne!(a.to_path_buf(), b.to_path_buf());
}
