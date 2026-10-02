//! Device-resident decode positions (epic sc-24432, story sc-24441) — the **one** position
//! mechanism every static-cache decoder shares (E8): the llama family ([`CausalLm`]) and the
//! Qwen3.5/3.8 hybrid ([`Qwen35Model`]) both read their step's positions from here.
//!
//! A CUDA graph records a decode step once and replays it at every later position, so no kernel
//! the step launches may take a position as a host-side number: the RoPE angles, the KV write
//! index, the attention length and — on the hybrid — the DeltaNet checkpoint-ring slots all move
//! every step. [`DevicePositions`] keeps them in two small device buffers at stable addresses,
//! written by [`DevicePositions::stage`] **outside** any capture (one tiny upload per step, the
//! cache's [`stage_positions`](crate::primitives::DecodeCache::stage_positions)); a step reads
//! them through device-side ops only:
//!
//! * **RoPE** — [`DeviceRope::cos_sin`] builds the step's `(cos, sin)` tables from the staged
//!   positions on the device, with exactly the host tables' arithmetic (`pos as f32 · inv_freq`,
//!   then `cos` / `sin`, then the dtype cast), so they are bit-identical to
//!   [`Rope::cos_sin`](crate::primitives::Rope::cos_sin) at the same positions.
//! * **KV write + attention length + causal mask** — the step start
//!   ([`start`](DevicePositions::start)) is the row a static cache writes the step's K/V at
//!   ([`candle_quant_kernels::write_rows_at`]) and the position of the step's first query in the
//!   length-aware decode attention ([`candle_quant_kernels::decode_attention()`]), which derives each
//!   query's visible keys (causal, optionally windowed) from it.
//! * **DeltaNet ring slots** — the slot the live recurrent state is read from
//!   ([`ring_read`](DevicePositions::ring_read)) and the slot each token's post-step state is
//!   written to ([`ring_write`](DevicePositions::ring_write)).
//!
//! The path is taken only for a step of at most [`MAX_DEVICE_STEP_TOKENS`] tokens (a decode or a
//! verify step — exactly the shapes the graph runner captures); a prompt prefill — from an empty
//! cache, longer than that, or marked as one ([`prefill_scope`], e.g. a prefix-cache hit's
//! suffix) — keeps the host positions and the `sdpa_gqa` attention, and both paths keep the
//! cache's host-side length in step, so they interleave freely.
//!
//! [`CausalLm`]: crate::models::CausalLm
//! [`Qwen35Model`]: crate::models::Qwen35Model

use candle_core::{DType, Device, Tensor};

use crate::decode::graph::capturing;
use crate::error::{Error, Result};
use crate::primitives::rope::Rope;
use crate::primitives::switch::{ProcessSwitch, SwitchGuard};

/// Environment variable that sets whether decoders stage device positions: `1` / `on` / `true`
/// / `yes` turns the path on for **every** device (the CPU runs the same logic through the
/// reference kernel), anything else turns it off; unset leaves [`DEVICE_POSITIONS_DEFAULT`].
pub const DEVICE_POSITIONS_ENV: &str = "CANDLE_LLM_DEVICE_POSITIONS";

/// Whether a decoder built on a **CUDA** device stages device positions when nothing asks
/// otherwise, read by both static-cache decoders' constructors
/// ([`CausalLm`](crate::models::CausalLm), [`Qwen35Model`](crate::models::Qwen35Model); E8) —
/// the Candle CUDA row of the per-backend defaults table ([`core_llm::defaults::CANDLE_CUDA`]),
/// where the value and its justification live. A CPU or Metal decoder takes its own row's
/// default (off: the host path) unless a caller asks for the device path ([`DEVICE_POSITIONS_ENV`],
/// [`set_device_positions_default`], or the model's own `set_device_positions`).
pub const DEVICE_POSITIONS_DEFAULT: bool = core_llm::defaults::CANDLE_CUDA.device_positions;

fn env_value_enables(v: &str) -> bool {
    matches!(v, "1" | "on" | "true" | "yes")
}

/// The process device-positions switch, on the crate's one switch implementation.
static SWITCH: ProcessSwitch = ProcessSwitch::new(
    DEVICE_POSITIONS_ENV,
    DEVICE_POSITIONS_DEFAULT,
    env_value_enables,
);

/// Whether a decoder built now on `device` stages device positions: what was asked for
/// explicitly — [`set_device_positions_default`], else [`DEVICE_POSITIONS_ENV`] — on any device;
/// else the device's row of the defaults table ([`crate::device::decode_defaults`]: on for CUDA,
/// [`DEVICE_POSITIONS_DEFAULT`]; off on Metal and the CPU). The model still serves the path only
/// where it can (`device_positions_support`).
pub fn device_positions_default(device: &Device) -> bool {
    SWITCH
        .explicit()
        .unwrap_or_else(|| crate::device::decode_defaults(device).device_positions)
}

/// [`device_positions_default`] for the device [`select_device`](crate::device::select_device)
/// would open in this process — CUDA in a `cuda` build unless `CANDLE_LLM_DEVICE=cpu`, else
/// Metal in a `metal` build, else the CPU — **without opening it** (a second CUDA device would
/// put a second stream on the context). The speculative benchmark records it as the switch's
/// effective state (sc-24446).
pub fn device_positions_default_for_selected_device() -> bool {
    let cpu_forced = std::env::var_os("CANDLE_LLM_DEVICE")
        .is_some_and(|s| s.to_string_lossy().eq_ignore_ascii_case("cpu"));
    SWITCH.explicit().unwrap_or_else(|| {
        crate::device::decode_backend_for(cfg!(feature = "cuda") && !cpu_forced)
            .defaults()
            .device_positions
    })
}

/// Override the device-positions default for the process: `Some(true)` / `Some(false)` force it
/// on every device, `None` returns to the environment's setting (else the CUDA default).
pub fn set_device_positions_default(enabled: Option<bool>) {
    SWITCH.set(enabled);
}

/// Test seam: take the switch's lock, apply `enabled` (as [`set_device_positions_default`]) and
/// hand back a guard that restores the previous policy when dropped.
#[doc(hidden)]
pub fn device_positions_policy_guard(enabled: Option<bool>) -> SwitchGuard {
    SWITCH.guard(enabled)
}

thread_local! {
    /// Set while the current thread runs a prompt-prefill forward ([`prefill_scope`]).
    static PREFILL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Mark the forwards the current thread runs until the returned scope drops as a prompt
/// **prefill** (`on`) or not — [`StepRequest::prefill`](crate::decode::StepRequest::prefill),
/// set by each model's `forward_step`. A prefill segment runs the host-position path and the
/// reference (`sdpa_gqa`) attention even from a non-empty cache — a prefix-cache hit's suffix
/// (sc-24437) — so a restored request attends its prompt exactly as a cold prefill does; only
/// decode / verify steps take the device-positions path. A hybrid's linear layers ring only a
/// prefill's final state (sc-24446).
pub fn prefill_scope(on: bool) -> PrefillScope {
    PrefillScope {
        previous: PREFILL.with(|p| p.replace(on)),
    }
}

/// Whether the current thread is inside a prefill forward ([`prefill_scope`]).
pub fn in_prefill() -> bool {
    PREFILL.with(std::cell::Cell::get)
}

/// Restores the thread's previous prefill marking when dropped ([`prefill_scope`]).
#[must_use = "the prefill marking only applies while the scope is alive"]
pub struct PrefillScope {
    previous: bool,
}

impl Drop for PrefillScope {
    fn drop(&mut self) {
        PREFILL.with(|p| p.set(self.previous));
    }
}

/// The longest step the device-positions path serves: the graph runner's largest captured step
/// ([`GraphRunner::MAX_CAPTURED_TOKENS`](crate::decode::GraphRunner::MAX_CAPTURED_TOKENS)). A
/// longer step (a prompt prefill) runs the host-position path.
pub const MAX_DEVICE_STEP_TOKENS: usize = 16;

/// `index` layout: `[step start, ring read slot, ring write slot of token 0 .. MAX)`.
const INDEX_LEN: usize = 2 + MAX_DEVICE_STEP_TOKENS;

/// The staged positions of one cache (see the module docs). Two stable device buffers:
///
/// * `index` (`u32`): `[0]` the step start — the cache length before the step, where its K/V are
///   written and its first query sits; `[1]` the checkpoint-ring slot holding the live recurrent
///   state; `[2 + t]` the slot token `t`'s post-step state goes to;
/// * `rope` (`f32`): `[t]` the RoPE position of token `t` (`len + rope_delta + t`), as `f32`
///   exactly like the host tables' `pos as f32`.
#[derive(Debug)]
pub struct DevicePositions {
    index: Tensor,
    rope: Tensor,
    device: Device,
    /// What the buffers hold now: `(len, rope_delta, ring_slots)` of the last stage.
    staged: std::cell::Cell<Option<(i32, i32, Option<usize>)>>,
}

impl DevicePositions {
    /// Bytes the two buffers hold on the device (admission prices them, E7).
    pub const BYTES: usize = INDEX_LEN * 4 + MAX_DEVICE_STEP_TOKENS * 4;

    /// Allocate the buffers on `device` (zeroed).
    pub fn new(device: &Device) -> Result<Self> {
        Ok(Self {
            index: Tensor::zeros(INDEX_LEN, DType::U32, device)?,
            rope: Tensor::zeros(MAX_DEVICE_STEP_TOKENS, DType::F32, device)?,
            device: device.clone(),
            staged: std::cell::Cell::new(None),
        })
    }

    /// The device the buffers live on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The host values [`stage`](Self::stage) writes: `(index, rope)`.
    fn values(
        len: i32,
        rope_delta: i32,
        ring_slots: Option<usize>,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        let start = u32::try_from(len)
            .map_err(|_| Error::Msg(format!("DevicePositions: negative cache length {len}")))?;
        let mut index = vec![0u32; INDEX_LEN];
        index[0] = start;
        if let Some(slots) = ring_slots.filter(|s| *s > 0) {
            let slot = |position: u64| (position % slots as u64) as u32;
            index[1] = slot(u64::from(start));
            for (t, entry) in index[2..].iter_mut().enumerate() {
                *entry = slot(u64::from(start) + t as u64 + 1);
            }
        }
        let rope = (0..MAX_DEVICE_STEP_TOKENS as i32)
            .map(|t| (len + rope_delta + t) as f32)
            .collect();
        Ok((index, rope))
    }

    /// Write the positions of a step starting at cache length `len` (RoPE shifted by
    /// `rope_delta`; ring slots modulo `ring_slots` when the cache keeps a checkpoint ring) into
    /// the device buffers, in place. A no-op while a capture is recording
    /// ([`capturing`]): the runner staged them before the capture began, and an upload inside it
    /// would never replay. Also a no-op when the buffers already hold exactly these positions
    /// (the runner stages before the step, the model's forward stages again).
    pub fn stage(&self, len: i32, rope_delta: i32, ring_slots: Option<usize>) -> Result<()> {
        let key = (len, rope_delta, ring_slots);
        if capturing() || self.staged.get() == Some(key) {
            return Ok(());
        }
        // Invalidated first: a failed upload leaves nothing claimed.
        self.staged.set(None);
        let (index, rope) = Self::values(len, rope_delta, ring_slots)?;
        self.index
            .slice_set(&Tensor::from_vec(index, INDEX_LEN, &self.device)?, 0, 0)?;
        self.rope.slice_set(
            &Tensor::from_vec(rope, MAX_DEVICE_STEP_TOKENS, &self.device)?,
            0,
            0,
        )?;
        self.staged.set(Some(key));
        Ok(())
    }

    /// The storage addresses of the `(index, rope)` buffers
    /// ([`storage_address`](crate::primitives::storage_address)): a captured graph reads them, so
    /// a cache's [`graph_identity`](crate::primitives::DecodeCache::graph_identity) folds them in.
    pub fn addresses(&self) -> Result<(usize, usize)> {
        Ok((
            crate::primitives::storage_address(&self.index)?,
            crate::primitives::storage_address(&self.rope)?,
        ))
    }

    /// The step start (`[1]` `u32`): the KV write row and the first query's position.
    pub fn start(&self) -> Result<Tensor> {
        Ok(self.index.narrow(0, 0, 1)?)
    }

    /// The checkpoint-ring slot of the live recurrent state (`[1]` `u32`).
    pub fn ring_read(&self) -> Result<Tensor> {
        Ok(self.index.narrow(0, 1, 1)?)
    }

    /// The checkpoint-ring slot token `t` of the step writes its post-step state to (`[1]`
    /// `u32`).
    pub fn ring_write(&self, t: usize) -> Result<Tensor> {
        if t >= MAX_DEVICE_STEP_TOKENS {
            return Err(Error::Msg(format!(
                "DevicePositions: token {t} past the {MAX_DEVICE_STEP_TOKENS}-token step bound"
            )));
        }
        Ok(self.index.narrow(0, 2 + t, 1)?)
    }

    /// The RoPE positions of the step's first `n` tokens (`[n]` `f32`).
    pub fn rope_positions(&self, n: usize) -> Result<Tensor> {
        if n > MAX_DEVICE_STEP_TOKENS {
            return Err(Error::Msg(format!(
                "DevicePositions: a {n}-token step exceeds {MAX_DEVICE_STEP_TOKENS}"
            )));
        }
        Ok(self.rope.narrow(0, 0, n)?)
    }
}

/// A [`Rope`]'s inverse frequencies resident on the device, so a step's `(cos, sin)` tables are
/// built from [`DevicePositions::rope_positions`] without a host upload (see the module docs).
#[derive(Clone, Debug)]
pub struct DeviceRope {
    /// `[1, rotary_dim / 2]` f32.
    inv_freq: Tensor,
    interleaved: bool,
}

impl DeviceRope {
    /// Lift `rope`'s inverse frequencies onto `device`.
    pub fn new(rope: &Rope, device: &Device) -> Result<Self> {
        let half = rope.inv_freq().len();
        Ok(Self {
            inv_freq: Tensor::from_slice(rope.inv_freq(), (1, half), device)?,
            interleaved: rope.interleaved(),
        })
    }

    /// `(cos, sin)` `[1, n, rotary_dim]` in `dtype` for the `n` positions `positions` (`[n]`
    /// f32): the angle table `pos · inv_freq` laid out like [`Rope::cos_sin`]'s (NeoX
    /// `cat(freqs, freqs)`, or each frequency twice for the interleaved pairing), then `cos` /
    /// `sin` and the cast. Every element is the same IEEE f32 product and the same elementwise
    /// kernels as the host table, so the result is bit-identical to `Rope::cos_sin` there.
    pub fn cos_sin(&self, positions: &Tensor, dtype: DType) -> Result<(Tensor, Tensor)> {
        let n = positions.dim(0)?;
        let half = self.inv_freq.dim(1)?;
        let freqs = positions.reshape((n, 1))?.broadcast_mul(&self.inv_freq)?; // [n, half]
        let emb = if self.interleaved {
            let f = freqs.unsqueeze(2)?;
            Tensor::cat(&[&f, &f], 2)?.reshape((n, 2 * half))?
        } else {
            Tensor::cat(&[&freqs, &freqs], 1)?
        };
        let emb = emb.reshape((1, n, 2 * half))?;
        Ok((emb.cos()?.to_dtype(dtype)?, emb.sin()?.to_dtype(dtype)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bench's record follows an explicit setting on any build, and otherwise the default
    /// row of the device this build selects.
    #[test]
    fn the_selected_device_default_follows_the_switch_then_the_selected_row() {
        for explicit in [true, false] {
            let _guard = device_positions_policy_guard(Some(explicit));
            assert_eq!(device_positions_default_for_selected_device(), explicit);
        }
        let _guard = device_positions_policy_guard(None);
        if SWITCH.explicit().is_none() && !cfg!(feature = "cuda") {
            // CPU (or Metal) row: the host path.
            assert!(!device_positions_default_for_selected_device());
        }
    }

    fn host(t: &Tensor) -> Vec<f32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    /// The device tables equal the host tables bit for bit, for both pairings, a partial rotary
    /// width and a shifted (M-RoPE delta) position.
    #[test]
    fn device_rope_tables_are_the_host_tables() {
        let device = Device::Cpu;
        for rope in [
            Rope::standard(16, 10_000.0),
            Rope::partial(8, 1_000_000.0, false),
            Rope::partial(8, 10_000.0, true),
            Rope::proportional(16, 1_000_000.0, 0.25, 1.0),
            Rope::yarn(8, 10_000.0, 40.0, 32.0, 1.0, 4096.0),
            Rope::llama3(16, 500_000.0, 8.0, 1.0, 4.0, 8192.0),
        ] {
            let d = DeviceRope::new(&rope, &device).unwrap();
            let positions = DevicePositions::new(&device).unwrap();
            for (len, delta) in [(0, 0), (37, 0), (4095, -3), (131_071, 17)] {
                positions.stage(len, delta, None).unwrap();
                for n in [1usize, 4, MAX_DEVICE_STEP_TOKENS] {
                    for dtype in [DType::F32, DType::BF16] {
                        let (cos, sin) = d
                            .cos_sin(&positions.rope_positions(n).unwrap(), dtype)
                            .unwrap();
                        let (hc, hs) = rope.cos_sin(n as i32, len + delta, dtype, &device).unwrap();
                        assert_eq!(cos.dims(), hc.dims());
                        assert_eq!(host(&cos), host(&hc), "cos len={len} n={n}");
                        assert_eq!(host(&sin), host(&hs), "sin len={len} n={n}");
                    }
                }
            }
        }
    }

    /// sc-24441 (E8): both static-cache decoders take their device-positions setting from the
    /// one process default — asked for explicitly it applies on any device (the CPU here), and
    /// unset it is on only for a CUDA device. The environment is read once per process, so each
    /// explicit setting runs in a child process of its own.
    #[test]
    fn both_decoders_take_the_process_device_positions_default() {
        const CHILD: &str = "CANDLE_LLM_DEVICE_POSITIONS_TEST_CHILD";
        let settings = || {
            let (weights, cfg) = crate::models::llama::tests::tiny_qwen3(32, true, &Device::Cpu);
            let causal = crate::models::CausalLm::from_weights(&weights, "", cfg).unwrap();
            let (_, hybrid) = crate::models::qwen35::tests::text_model();
            (causal.device_positions(), hybrid.device_positions())
        };
        match std::env::var(CHILD).ok().as_deref() {
            Some(expected) => {
                let on = expected == "on";
                assert_eq!(device_positions_default(&Device::Cpu), on);
                assert_eq!(settings(), (on, on), "both constructors take the default");
            }
            None => {
                let name = std::thread::current().name().unwrap().to_owned();
                for (value, expected) in [("1", "on"), ("0", "off")] {
                    let status = std::process::Command::new(std::env::current_exe().unwrap())
                        .args(["--exact", &name, "--nocapture"])
                        .env(CHILD, expected)
                        .env(DEVICE_POSITIONS_ENV, value)
                        .status()
                        .unwrap();
                    assert!(status.success(), "{DEVICE_POSITIONS_ENV}={value}");
                }
                if std::env::var_os(DEVICE_POSITIONS_ENV).is_none() {
                    let _guard = device_positions_policy_guard(None);
                    assert!(!device_positions_default(&Device::Cpu), "unset: CUDA only");
                    assert_eq!(settings(), (false, false));
                }
            }
        }
    }

    #[test]
    fn staged_index_holds_start_and_ring_slots() {
        let positions = DevicePositions::new(&Device::Cpu).unwrap();
        positions.stage(10, 0, Some(4)).unwrap();
        let index: Vec<u32> = positions.index.to_vec1::<u32>().unwrap();
        assert_eq!(index[0], 10);
        assert_eq!(
            index[1],
            10 % 4,
            "the live state sits in the slot of the current position"
        );
        for t in 0..MAX_DEVICE_STEP_TOKENS {
            assert_eq!(index[2 + t], (10 + t as u32 + 1) % 4, "token {t}");
        }
        assert_eq!(
            positions.start().unwrap().to_vec1::<u32>().unwrap(),
            vec![10]
        );
        assert_eq!(
            positions.ring_write(3).unwrap().to_vec1::<u32>().unwrap(),
            vec![(10 + 4) % 4]
        );
        assert!(positions.ring_write(MAX_DEVICE_STEP_TOKENS).is_err());
        assert!(positions
            .rope_positions(MAX_DEVICE_STEP_TOKENS + 1)
            .is_err());
        assert!(positions.stage(-1, 0, None).is_err());
        assert_eq!(DevicePositions::BYTES, (2 + 16) * 4 + 16 * 4);
    }
}
