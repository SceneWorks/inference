//! The process-wide on/off switch (epic sc-24128, sc-24140): **one** implementation of the
//! pattern the fused primitives ([`fused`](super::fused)), the NVFP4 decode GEMV
//! ([`nvfp4_path`](super::nvfp4_path)) and the CUDA-graph runner
//! ([`graph`](crate::decode::graph)) each switch through.
//!
//! A [`ProcessSwitch`] has three layers, most specific first:
//!
//! 1. a runtime **override** ([`ProcessSwitch::set`]: `Some(true)` / `Some(false)`), which the
//!    decode bench and the parity tests flip to compare two paths in one process;
//! 2. the **environment** variable, read **once** per process and cached (a hot path asks the
//!    switch on every call);
//! 3. the switch's **default** when the variable is unset.
//!
//! Tests that flip a switch — or assert a reason the switch could change — hold its
//! [`SwitchGuard`] ([`ProcessSwitch::guard`]): the switch's own lock, so parallel test threads
//! cannot race on the global, and the policy it found, restored on drop — including a policy that
//! was deferring to the environment (`None`), which a hand-rolled "read then set `Some(was)`"
//! restore would pin instead.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};

const POLICY_ENV: u8 = 0;
const POLICY_ON: u8 = 1;
const POLICY_OFF: u8 = 2;

/// A process-wide switch: a runtime override over a cached environment variable over a default.
/// Declared as a `static`; see the [module docs](self).
pub struct ProcessSwitch {
    env: &'static str,
    unset: bool,
    parse: fn(&str) -> bool,
    policy: AtomicU8,
    from_env: OnceLock<bool>,
    lock: Mutex<()>,
}

impl ProcessSwitch {
    /// A switch read from `env`: `unset` is its state when the variable is not set, and `parse`
    /// says whether a set value (trimmed and lower-cased first) means **on**.
    pub const fn new(env: &'static str, unset: bool, parse: fn(&str) -> bool) -> Self {
        Self {
            env,
            unset,
            parse,
            policy: AtomicU8::new(POLICY_ENV),
            from_env: OnceLock::new(),
            lock: Mutex::new(()),
        }
    }

    /// The environment variable this switch reads.
    pub fn env(&self) -> &'static str {
        self.env
    }

    /// Whether the switch is on: the override if one is set, else the environment's setting.
    pub fn enabled(&self) -> bool {
        match self.policy.load(Ordering::Relaxed) {
            POLICY_ON => true,
            POLICY_OFF => false,
            _ => self.env_enabled(),
        }
    }

    /// The environment's setting, read once per process.
    fn env_enabled(&self) -> bool {
        *self.from_env.get_or_init(|| match std::env::var(self.env) {
            Ok(v) => (self.parse)(&v.trim().to_ascii_lowercase()),
            Err(_) => self.unset,
        })
    }

    /// Override the switch for the process: `Some(true)` / `Some(false)` force it, `None` returns
    /// to the environment's setting.
    pub fn set(&self, enabled: Option<bool>) {
        self.policy.store(encode(enabled), Ordering::Relaxed);
    }

    /// Take the switch's lock, apply `enabled` (as [`set`](Self::set)) and hand back a guard that
    /// restores the policy it found — override or deferral — when dropped. [`set`](Self::set)
    /// may still be called while it is held.
    pub fn guard(&'static self, enabled: Option<bool>) -> SwitchGuard {
        let lock = self
            .lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let previous = self.policy.load(Ordering::Relaxed);
        self.set(enabled);
        SwitchGuard {
            switch: self,
            previous,
            _lock: lock,
        }
    }
}

fn encode(enabled: Option<bool>) -> u8 {
    match enabled {
        Some(true) => POLICY_ON,
        Some(false) => POLICY_OFF,
        None => POLICY_ENV,
    }
}

/// Holds a [`ProcessSwitch`]'s lock; restores the policy it found when dropped. Returned by
/// [`ProcessSwitch::guard`].
#[must_use = "the policy is only held (and restored) while the guard is alive"]
pub struct SwitchGuard {
    switch: &'static ProcessSwitch,
    previous: u8,
    _lock: MutexGuard<'static, ()>,
}

impl SwitchGuard {
    /// The policy the guard found — and restores when dropped: `Some(bool)` for an override,
    /// `None` for a switch that deferred to the environment.
    pub fn found(&self) -> Option<bool> {
        match self.previous {
            POLICY_ON => Some(true),
            POLICY_OFF => Some(false),
            _ => None,
        }
    }
}

impl Drop for SwitchGuard {
    fn drop(&mut self) {
        self.switch.policy.store(self.previous, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn on_words(v: &str) -> bool {
        matches!(v, "1" | "on")
    }

    /// No test sets these variables, so the environment layer is the default.
    static DEFAULT_OFF: ProcessSwitch =
        ProcessSwitch::new("CANDLE_LLM_TEST_SWITCH_UNSET_A", false, on_words);
    static DEFAULT_ON: ProcessSwitch =
        ProcessSwitch::new("CANDLE_LLM_TEST_SWITCH_UNSET_B", true, on_words);

    #[test]
    fn the_override_wins_and_none_returns_to_the_default() {
        let _guard = DEFAULT_OFF.guard(None);
        assert!(!DEFAULT_OFF.enabled(), "unset: the default");
        DEFAULT_OFF.set(Some(true));
        assert!(DEFAULT_OFF.enabled());
        DEFAULT_OFF.set(None);
        assert!(!DEFAULT_OFF.enabled());
        let _other = DEFAULT_ON.guard(None);
        assert!(DEFAULT_ON.enabled(), "each switch has its own default");
        assert_eq!(DEFAULT_ON.env(), "CANDLE_LLM_TEST_SWITCH_UNSET_B");
    }

    #[test]
    fn the_guard_restores_a_deferral_not_the_state_it_observed() {
        let outer = DEFAULT_ON.guard(None);
        drop(outer);
        {
            let _off = DEFAULT_ON.guard(Some(false));
            assert!(!DEFAULT_ON.enabled());
        }
        // The guard found `None` (defer to the environment), so it restored `None` — not
        // `Some(true)`, the state a "read enabled(), set Some(was)" restore would pin.
        let after = DEFAULT_ON.guard(None);
        assert_eq!(after.found(), None);
        assert!(DEFAULT_ON.enabled());
        DEFAULT_ON.set(Some(false));
        drop(after);
        assert_eq!(
            DEFAULT_ON.guard(None).found(),
            None,
            "an override set while a guard is held is undone by it"
        );
    }

    #[test]
    fn the_environment_is_read_once_and_parsed_after_trimming_and_lower_casing() {
        static READ_ONCE: ProcessSwitch =
            ProcessSwitch::new("CANDLE_LLM_TEST_SWITCH_READ_ONCE", false, on_words);
        let _guard = READ_ONCE.guard(None);
        std::env::set_var(READ_ONCE.env(), "  ON ");
        assert!(
            READ_ONCE.enabled(),
            "trimmed and lower-cased before parsing"
        );
        std::env::set_var(READ_ONCE.env(), "0");
        assert!(
            READ_ONCE.enabled(),
            "the first read is cached for the process"
        );
        std::env::remove_var(READ_ONCE.env());
        assert!(READ_ONCE.enabled());
    }
}
