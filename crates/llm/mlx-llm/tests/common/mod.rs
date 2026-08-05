//! Shared test-support for the mlx-llm suites (sc-17768).
//!
//! `tests/common/` is a module directory, not a test target, so this compiles into each integration
//! suite that declares `mod common;` rather than running as its own binary. The crate's own
//! `#[cfg(test)]` unit suites reach the same code through `#[path]` (see `src/lib.rs`) — an
//! integration test links against the public API and so cannot see a `#[cfg(test)]` module, and the
//! reverse (making this `pub` in the library) would put test scaffolding in the shipped surface and
//! force `tempfile` out of `[dev-dependencies]`. One file, both contexts, nothing to keep in step.

use std::path::{Path, PathBuf};

/// A temp fixture path that owns its `TempDir` guard.
///
/// Derefs (and `AsRef`s) to `Path`, so call sites read exactly as they did when these helpers
/// returned a bare `PathBuf` — but the tree now leaves on `Drop`, **including out of a panicking
/// test**, rather than depending on trailing `remove_dir_all(..).ok()` lines. Those ran only on the
/// happy path, which is why the leak was invisible to a green run: measured on `main`, eleven
/// panicking probes left ten `mlx-llm-*` trees under `$TMPDIR`; with this type, none.
///
/// Bind it for as long as the path is read: a dropped `Fixture` takes the directory with it. That
/// matters more here than on the candle side — MLX's `Array::load_safetensors` is lazy, so a
/// `Weights` value or provider read after its fixture dropped would be reading a deleted tree.
pub struct Fixture {
    path: PathBuf,
    guard: tempfile::TempDir,
}

impl Fixture {
    /// A fresh fixture root. `name`, when given, names an entry *inside* that root to point at
    /// instead — the guard still owns the enclosing directory, so the entry cannot outlive it.
    ///
    /// Pass a `name` whenever the code under test wants to create the directory itself: the
    /// snapshot writers, the GGUF converter, and the preparer all refuse a non-empty output, and
    /// they allocate their staging directory as a *sibling* of it. Both land inside the guarded
    /// root this way. See [`assert_fixture_is_a_guarded_entry`].
    pub fn new(prefix: &str, name: Option<&str>) -> Self {
        let guard = tempfile::Builder::new()
            .prefix(prefix)
            .tempdir()
            .expect("fixture temp dir");
        let path = match name {
            Some(entry) => guard.path().join(entry),
            None => guard.path().to_path_buf(),
        };
        Self { path, guard }
    }

    /// The guarded root — the directory that leaves on `Drop`. Equal to the fixture path itself
    /// unless the fixture was minted with a `name`.
    pub fn root(&self) -> &Path {
        self.guard.path()
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

// `PrepareSpec` and friends take `impl Into<PathBuf>`; without this, every such call site would
// have to spell out `&*fixture` or `.to_path_buf()`.
impl From<&Fixture> for PathBuf {
    fn from(fixture: &Fixture) -> Self {
        fixture.path.clone()
    }
}

/// Asserts a helper's fixture takes its whole tree with it. Every suite that mints fixtures calls
/// this on its own helper, because the risk this file cannot cover is a *helper* that drops the
/// guard before returning — the leak is invisible from the assertions the suites actually make.
///
/// Give [`Fixture::new`]'s builder `disable_cleanup(true)` and every caller of this goes RED.
pub fn assert_fixture_is_self_removing(fixture: Fixture) {
    let root = fixture.root().to_path_buf();
    assert!(
        root.is_dir(),
        "fixture root was never created: {}",
        root.display()
    );
    drop(fixture);
    assert!(
        !root.exists(),
        "fixture root survived its guard: {}",
        root.display()
    );
}

/// [`assert_fixture_is_self_removing`] plus the containment invariant: the helper points at an
/// entry *inside* its guarded root, not at the root itself.
///
/// Every helper feeding a snapshot writer, the GGUF converter, or the preparer must satisfy this.
/// Those allocate staging as a sibling of the output path (`.{name}.tmp-{pid}-{nonce}`), so a
/// helper quietly switched back to a root-level fixture would push staging *out* of the guard and
/// into `$TMPDIR` — a leak no assertion in the owning suite can see, because the test still passes.
/// Drop the `Some(..)` from the helper's `Fixture::new` and this goes RED.
// This module compiles into eight test binaries and each uses only the subset it needs: the suites
// whose helpers mint root fixtures (`conformance`, `contract_roundtrip`) never call this one.
#[allow(dead_code)]
pub fn assert_fixture_is_a_guarded_entry(fixture: Fixture) {
    assert_eq!(
        fixture.parent(),
        Some(fixture.root()),
        "fixture must point at an entry inside its guarded root ({}), not at the root itself; \
         otherwise the writer's staging sibling escapes the guard",
        fixture.root().display()
    );
    assert_fixture_is_self_removing(fixture);
}

/// Guards [`Fixture`] itself. Compiled into every suite that declares `mod common;` (and into the
/// lib's unit suite), so it runs once per test binary — cheap, and it keeps the guarantee local.
#[test]
fn fixture_removes_its_tree_on_drop() {
    let (root, file) = {
        let fixture = Fixture::new("mlx-llm-fixture-guard-", Some("inner.bin"));
        std::fs::write(&fixture, b"bytes").unwrap();
        let file = fixture.to_path_buf();
        let root = file.parent().unwrap().to_path_buf();
        assert!(file.is_file(), "fixture file not written");
        (root, file)
    };
    assert!(!file.exists(), "fixture file survived: {}", file.display());
    assert!(!root.exists(), "fixture root survived: {}", root.display());
}

/// Two fixtures with the same prefix get different roots — the property the old `{prefix}{pid}`
/// naming lacked, and the reason two `#[test]`s sharing a label (they run concurrently in one
/// process) could collide, or a recycled PID reopen a previous run.
#[test]
fn fixtures_with_the_same_prefix_do_not_collide() {
    let (a, b) = (
        Fixture::new("mlx-llm-fixture-guard-", None),
        Fixture::new("mlx-llm-fixture-guard-", None),
    );
    assert_ne!(a.to_path_buf(), b.to_path_buf());
}
