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
//!
//! The lock is **re-entrant on one thread** (sc-24140 feature-end review): the CUDA-graph
//! switch's lock is also the CUDA unit tests' serialization lock, which a test thread takes when
//! it opens a CUDA device ([`ProcessSwitch::hold`]) and keeps until it exits, so the same thread
//! must still be able to take a [`SwitchGuard`] afterwards. Another thread still waits.

use std::marker::PhantomData;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Condvar, Mutex, OnceLock, PoisonError};
use std::thread::ThreadId;

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
    lock: ThreadLock,
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
            lock: ThreadLock::new(),
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
        let lock = self.hold();
        let previous = self.policy.load(Ordering::Relaxed);
        self.set(enabled);
        SwitchGuard {
            switch: self,
            previous,
            _lock: lock,
        }
    }

    /// Take the switch's lock without touching its policy: the lock is released when the
    /// returned [`SwitchLock`] drops, and nothing is restored. Re-entrant on the holding thread.
    pub fn hold(&'static self) -> SwitchLock {
        self.lock.acquire();
        SwitchLock {
            lock: &self.lock,
            _thread_bound: PhantomData,
        }
    }

    /// [`hold`](Self::hold) without waiting: `None` while another thread holds the lock — the
    /// clock-free way for a test to observe that the lock is held elsewhere.
    #[cfg(test)]
    pub(crate) fn try_hold(&'static self) -> Option<SwitchLock> {
        // Lazily: a `SwitchLock` built for a failed attempt would release someone else's hold
        // when dropped.
        self.lock.try_acquire().then(|| SwitchLock {
            lock: &self.lock,
            _thread_bound: PhantomData,
        })
    }
}

/// A lock that one thread may take any number of times (each [`SwitchLock`] releases one) while
/// every other thread waits for the last release.
struct ThreadLock {
    state: Mutex<LockState>,
    released: Condvar,
}

struct LockState {
    owner: Option<ThreadId>,
    depth: usize,
    /// Threads blocked in [`ThreadLock::acquire`] (tests read it to know a waiter is parked).
    waiting: usize,
}

impl LockState {
    /// Take one hold for `me` unless another thread owns the lock.
    fn take(&mut self, me: ThreadId) -> bool {
        if self.owner.is_some_and(|owner| owner != me) {
            return false;
        }
        self.owner = Some(me);
        self.depth += 1;
        true
    }
}

impl ThreadLock {
    const fn new() -> Self {
        Self {
            state: Mutex::new(LockState {
                owner: None,
                depth: 0,
                waiting: 0,
            }),
            released: Condvar::new(),
        }
    }

    fn acquire(&self) {
        let me = std::thread::current().id();
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        while !state.take(me) {
            state.waiting += 1;
            state = self
                .released
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
            state.waiting -= 1;
        }
    }

    #[cfg(test)]
    fn try_acquire(&self) -> bool {
        let me = std::thread::current().id();
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.take(me)
    }

    /// How many threads are parked waiting for the lock.
    #[cfg(test)]
    fn waiting(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .waiting
    }

    /// Release one hold. Never asks for the current thread, so it also runs from a thread-local
    /// destructor at thread exit.
    fn release(&self) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.depth = state.depth.saturating_sub(1);
        if state.depth == 0 {
            state.owner = None;
            self.released.notify_all();
        }
    }
}

/// One hold of a [`ProcessSwitch`]'s lock ([`ProcessSwitch::hold`]); releases it when dropped.
/// Bound to the thread that took it.
#[must_use = "the lock is only held while the value is alive"]
pub struct SwitchLock {
    lock: &'static ThreadLock,
    _thread_bound: PhantomData<*const ()>,
}

impl Drop for SwitchLock {
    fn drop(&mut self) {
        self.lock.release();
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
    // Declared last: dropped after `Drop::drop` has restored the policy.
    _lock: SwitchLock,
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

    /// sc-24140 feature-end review: the lock is re-entrant on the thread that holds it — a
    /// thread-lifetime [`ProcessSwitch::hold`] followed by a [`SwitchGuard`] and another hold does
    /// not deadlock, and the hold restores no policy — while no other thread can take it, and a
    /// thread parked waiting for it gets it at the last release. Nothing is read through a clock:
    /// exclusion is a non-waiting `try_hold`, the waiter is known parked by the lock's own count;
    /// the only timeouts bound a regression's hang.
    #[test]
    fn the_lock_is_reentrant_on_its_thread_and_excludes_every_other_thread() {
        use std::sync::mpsc::channel;
        use std::time::Duration;
        static SHARED: ProcessSwitch =
            ProcessSwitch::new("CANDLE_LLM_TEST_SWITCH_UNSET_C", false, on_words);

        let (holding, held) = channel();
        let (release, released) = channel::<()>();
        let owner = std::thread::spawn(move || {
            let outer = SHARED.hold();
            {
                let _guard = SHARED.guard(Some(true));
                let _again = SHARED.hold();
                assert!(SHARED.enabled());
            }
            let restored = SHARED.enabled();
            holding.send(restored).unwrap();
            released.recv().unwrap();
            drop(outer);
        });
        let restored = held
            .recv_timeout(Duration::from_secs(10))
            .expect("the holding thread re-entered its own lock");
        assert!(
            !restored,
            "the guard restores its policy; the outer hold restores nothing"
        );

        let taken_elsewhere = std::thread::spawn(|| SHARED.try_hold().is_some())
            .join()
            .unwrap();
        assert!(
            !taken_elsewhere,
            "no other thread can take the lock while it is held"
        );

        let (acquired_tx, acquired) = channel();
        let waiter = std::thread::spawn(move || {
            let _held = SHARED.hold();
            acquired_tx.send(()).unwrap();
        });
        // Release only once the waiter is parked on the lock, so the hand-off is what wakes it.
        while SHARED.lock.waiting() == 0 {
            if acquired.try_recv().is_ok() {
                panic!("the waiter took a lock another thread holds");
            }
            std::thread::yield_now();
        }
        release.send(()).unwrap();
        acquired
            .recv_timeout(Duration::from_secs(10))
            .expect("the last release hands the lock to the waiting thread");
        owner.join().unwrap();
        waiter.join().unwrap();
    }
}
