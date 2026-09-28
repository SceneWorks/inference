//! The sm_120 test gate (sc-24140, epic sc-24128 AT1).
//!
//! A GPU test that needs an NVFP4-capable (sm_120) CUDA device skips — loudly, on stderr — when
//! the host has none, so the CPU lanes and a sub-sm_120 box stay green. That skip is also how such
//! a test *passes without running*, which is exactly what an acceptance run on the authoritative
//! box must rule out. Setting [`REQUIRE_SM120_ENV`] (`REQUIRE_SM120=1`) turns every skip routed
//! through [`skip_without_sm120`] into a hard failure naming the reason, so a green run under it
//! proves the sm_120 tests executed.
//!
//! Pure host code: builds (and is tested) on every target.

/// The environment variable that turns an sm_120 test skip into a failure.
pub const REQUIRE_SM120_ENV: &str = "REQUIRE_SM120";

/// Whether a [`REQUIRE_SM120_ENV`] value demands an sm_120 device: any value but unset, empty or
/// `0`.
pub fn sm120_required_by(value: Option<&str>) -> bool {
    value.is_some_and(|v| !v.trim().is_empty() && v.trim() != "0")
}

/// Whether this process's environment demands an sm_120 device.
pub fn sm120_required() -> bool {
    sm120_required_by(std::env::var(REQUIRE_SM120_ENV).ok().as_deref())
}

/// What a test does when it found no sm_120 device (`reason` says what was missing): print
/// `skipping: {reason}` and return when `required` is false; panic naming the reason when it is
/// true. Tests call [`skip_without_sm120`]; this form takes the policy explicitly so the gate
/// itself is testable without touching the process environment.
pub fn skip_without_sm120_as(required: bool, reason: &str) {
    if required {
        panic!(
            "{REQUIRE_SM120_ENV} is set but {reason}: this sm_120 test must run, not skip, on \
             this host"
        );
    }
    eprintln!("skipping: {reason}");
}

/// [`skip_without_sm120_as`] under this process's [`REQUIRE_SM120_ENV`]: call it on a GPU test's
/// no-sm_120 path, right before its early return.
pub fn skip_without_sm120(reason: &str) {
    skip_without_sm120_as(sm120_required(), reason);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_set_non_zero_value_requires_sm120() {
        assert!(!sm120_required_by(None));
        assert!(!sm120_required_by(Some("")));
        assert!(!sm120_required_by(Some("  ")));
        assert!(!sm120_required_by(Some("0")));
        assert!(sm120_required_by(Some("1")));
        assert!(sm120_required_by(Some(" 1 ")));
        assert!(sm120_required_by(Some("yes")));
    }

    #[test]
    fn an_unrequired_skip_returns() {
        skip_without_sm120_as(false, "no sm_120 CUDA device");
    }

    #[test]
    #[should_panic(expected = "REQUIRE_SM120 is set but no sm_120 CUDA device")]
    fn a_required_skip_fails() {
        skip_without_sm120_as(true, "no sm_120 CUDA device");
    }
}
