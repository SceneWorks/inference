//! Process-level runtime switches for mlx-llm's decode optimizations (epic sc-24432 E5, sc-24446):
//! one [`ProcessSwitch`] per optimization the terminal campaign isolates on MLX, on the shared
//! switch implementation Candle's `CANDLE_LLM_*` switches use.
//!
//! Each switch is a runtime override ([`ProcessSwitch::set`] / [`ProcessSwitch::guard`]) over its
//! environment variable (read once per process) over its default — the MLX row of the per-backend
//! defaults table ([`core_llm::defaults::MLX`]), where the value and its justification live. A set
//! variable turns the optimization off with `0` / `off` / `false` / `no` and on with anything else.
//!
//! | switch | variable | path when off |
//! |---|---|---|
//! | [`PIPELINING`] | `MLX_LLM_PIPELINING` | every token is read back before the next step is enqueued |
//! | [`DEVICE_SAMPLER`] | `MLX_LLM_DEVICE_SAMPLER` | every token is drawn by the host reference (`host:reference`) |
//! | [`FUSED_ROTATION`] | `MLX_LLM_FUSED_ROTATION` | the Prism rotation runs the unfused op chain |
//! | [`GDN_KERNEL`] | `MLX_LLM_GDN_KERNEL` | the gated-delta recurrence runs the op-by-op reference |
//!
//! Every path a switch selects matches the other (E1: pipelining is token-identical; the fused
//! rotation is bit-identical; the device sampler keeps the distribution; the GDN kernel matches
//! the reference within the recurrence's f32 tolerance), so flipping one changes the speed, not
//! the greedy output.

use std::cell::Cell;
use std::thread::LocalKey;

use core_llm::defaults::MLX;
use core_llm::switch::ProcessSwitch;

/// Environment variable of [`PIPELINING`].
pub const PIPELINING_ENV: &str = "MLX_LLM_PIPELINING";
/// Environment variable of [`DEVICE_SAMPLER`].
pub const DEVICE_SAMPLER_ENV: &str = "MLX_LLM_DEVICE_SAMPLER";
/// Environment variable of [`FUSED_ROTATION`].
pub const FUSED_ROTATION_ENV: &str = "MLX_LLM_FUSED_ROTATION";
/// Environment variable of [`GDN_KERNEL`].
pub const GDN_KERNEL_ENV: &str = "MLX_LLM_GDN_KERNEL";

/// A set variable means **on** unless it is one of the off words.
fn env_value_enables(v: &str) -> bool {
    !matches!(v, "0" | "off" | "false" | "no")
}

/// One MLX optimization switch: a thread-scoped override ([`scoped`](Self::scoped)) over the
/// process switch ([`process`](Self::process): runtime override, environment, defaults table).
/// The thread layer lets a test (or a bench comparing two paths) flip the optimization for its
/// own generation without changing it under every other thread of the process.
pub struct MlxSwitch {
    process: ProcessSwitch,
    thread: &'static LocalKey<Cell<Option<bool>>>,
}

impl MlxSwitch {
    const fn new(
        env: &'static str,
        default: bool,
        thread: &'static LocalKey<Cell<Option<bool>>>,
    ) -> Self {
        Self {
            process: ProcessSwitch::new(env, default, env_value_enables),
            thread,
        }
    }

    /// Whether the optimization is on for the current thread: its [`scoped`](Self::scoped)
    /// override if one is active, else the process switch.
    pub fn enabled(&self) -> bool {
        self.thread
            .with(Cell::get)
            .unwrap_or_else(|| self.process.enabled())
    }

    /// The process-wide layer: [`ProcessSwitch::set`] / [`ProcessSwitch::guard`] change it for
    /// every thread; its environment variable is [`ProcessSwitch::env`].
    pub fn process(&self) -> &ProcessSwitch {
        &self.process
    }

    /// Run `f` with the optimization forced `enabled` on this thread only, restoring the previous
    /// thread setting afterwards (also on unwind).
    pub fn scoped<R>(&self, enabled: bool, f: impl FnOnce() -> R) -> R {
        struct Restore(&'static LocalKey<Cell<Option<bool>>>, Option<bool>);
        impl Drop for Restore {
            fn drop(&mut self) {
                self.0.with(|c| c.set(self.1));
            }
        }
        let _restore = Restore(self.thread, self.thread.with(|c| c.replace(Some(enabled))));
        f()
    }
}

thread_local! {
    static PIPELINING_THREAD: Cell<Option<bool>> = const { Cell::new(None) };
    static DEVICE_SAMPLER_THREAD: Cell<Option<bool>> = const { Cell::new(None) };
    static FUSED_ROTATION_THREAD: Cell<Option<bool>> = const { Cell::new(None) };
    static GDN_KERNEL_THREAD: Cell<Option<bool>> = const { Cell::new(None) };
}

/// Whether the shared engine pipelines its token-at-a-time loop (sc-24439) where a request allows
/// it. Off is [`Pipelining::Off`](crate::decode::Pipelining::Off) for every request.
pub static PIPELINING: MlxSwitch =
    MlxSwitch::new(PIPELINING_ENV, MLX.pipelining, &PIPELINING_THREAD);

/// Whether tokens are drawn on the device (the argmax, and the device sampler for stochastic
/// rows; sc-24439). Off routes every draw to the host reference, reported as
/// [`HostSampleReason::Reference`](core_llm::HostSampleReason::Reference).
pub static DEVICE_SAMPLER: MlxSwitch = MlxSwitch::new(
    DEVICE_SAMPLER_ENV,
    MLX.device_sampler,
    &DEVICE_SAMPLER_THREAD,
);

/// Whether the Prism block-Hadamard rotation dispatches the fused Metal kernel (sc-24444).
pub static FUSED_ROTATION: MlxSwitch = MlxSwitch::new(
    FUSED_ROTATION_ENV,
    MLX.fused_rotation,
    &FUSED_ROTATION_THREAD,
);

/// Whether the gated-delta recurrence dispatches the fused Metal kernel on a GPU stream
/// (sc-24443). Off runs the op-by-op reference there (the pre-sc-24443 path).
pub static GDN_KERNEL: MlxSwitch =
    MlxSwitch::new(GDN_KERNEL_ENV, MLX.gdn_kernel, &GDN_KERNEL_THREAD);

#[cfg(test)]
mod tests {
    use super::*;

    /// Unset, every switch is at its MLX defaults-table value; a thread-scoped override wins on
    /// its thread only and is restored afterwards. (The process layer's override and guard are
    /// `core_llm::switch`'s own tests; flipping it here would race every parallel test.)
    #[test]
    fn each_switch_defaults_to_the_mlx_row_and_a_scoped_override_wins_on_its_thread() {
        for (switch, default, env) in [
            (&PIPELINING, MLX.pipelining, PIPELINING_ENV),
            (&DEVICE_SAMPLER, MLX.device_sampler, DEVICE_SAMPLER_ENV),
            (&FUSED_ROTATION, MLX.fused_rotation, FUSED_ROTATION_ENV),
            (&GDN_KERNEL, MLX.gdn_kernel, GDN_KERNEL_ENV),
        ] {
            assert_eq!(switch.process().env(), env);
            if std::env::var_os(env).is_none() && switch.process().explicit().is_none() {
                assert_eq!(switch.enabled(), default, "{env}: the defaults table");
            }
            let before = switch.enabled();
            switch.scoped(!before, || {
                assert_eq!(switch.enabled(), !before, "{env}");
                let other = std::thread::scope(|s| s.spawn(|| switch.enabled()).join().unwrap());
                assert_eq!(other, before, "{env}: another thread is unaffected");
            });
            assert_eq!(switch.enabled(), before, "{env}: restored");
        }
        assert!(env_value_enables("1") && env_value_enables("on"));
        assert!(!env_value_enables("0") && !env_value_enables("off"));
    }
}
