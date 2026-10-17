//! The retention model every counted Iris iOS quantum price follows (epic 16774, sc-26270 round 4).
//!
//! MLX allocates an op's output when it ENCODES the op, and keeps the op's inputs referenced until
//! its command buffer (Metal) or queued task (CPU) completes; up to ten command buffers (tasks) are
//! encoded ahead of execution. Within one evaluated graph every new buffer can therefore be live at
//! once — the quantum's peak is the SUM of its allocations, not its live set — with two exceptions
//! the counts honour:
//!
//! * **donation**: an elementwise op (or a compiled elementwise kernel, softmax, a fused norm) whose
//!   operand has no other reference and the output's size and element width reuses the operand's
//!   buffer and allocates nothing. A buffer the quantum's code still holds a handle on is never
//!   donated.
//! * **barriers**: [`crate::nn::barrier`] evaluates a block's sub-layer on the device before the next
//!   sub-layer is built. A buffer only the graph referenced is released there; a buffer the block's
//!   code holds a handle on lives to the end of the quantum.
//!
//! So a quantum is priced as a sequence of phases ([`Retention::barrier`]): each phase's peak is
//! what earlier phases left held plus every buffer the phase allocates, and the quantum's price is
//! the largest phase. Every buffer is rounded as MLX's allocator rounds it ([`allocated`]).
//!
//! The device runs the fused kernels (`rms_norm`, SDPA); the CPU — where every count is proven
//! against MLX's measured peak, `crate::quantum_proofs` — runs MLX's unfused fallbacks, which
//! allocate more. [`unfused`] selects which the price counts ([`crate::nn::fused_kernels`]).

/// MLX's allocator page (`vm_page_size` on Apple silicon).
pub const PAGE_BYTES: u64 = 16 * 1024;

/// Scalar constants a phase's graph creates (norm epsilons, the modulation's `1`, the attention
/// scale, …): host arrays of a few bytes each, allocated while the graph is built.
const SCALARS_BYTES: u64 = 256;

/// One buffer of `bytes` as MLX's allocator holds it: above one page it is rounded up to whole
/// pages (`MetalAllocator::malloc`, also the macOS CPU backend's allocator).
pub fn allocated(bytes: u64) -> u64 {
    if bytes > PAGE_BYTES {
        bytes.div_ceil(PAGE_BYTES).saturating_mul(PAGE_BYTES)
    } else {
        bytes
    }
}

/// Whether the current default device runs MLX's unfused CPU fallbacks (`rms_norm` as square /
/// mean / rsqrt / products, SDPA as scaled-query · key copy → scores → value copy), or the fused
/// device kernels.
pub fn unfused() -> bool {
    !crate::nn::fused_kernels()
}

/// Who still holds a buffer when its phase ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Only the graph: released at the next barrier.
    Temp,
    /// The quantum's code (a binding, a returned output): held to the end of the quantum.
    Named,
}

/// How a norm's input arrives (see [`Retention::norm`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormInput {
    /// At the compute width (bf16): the norm's own f32 cast is a new buffer the result reuses.
    Cast,
    /// An f32 temporary nothing else reads: the result reuses its buffer (the producer counts it).
    Donor,
    /// An f32 tensor something else still reads (a residual stream, a bound handle).
    Held,
}

/// One quantum's counted retention, phase by phase.
#[derive(Clone, Copy, Debug)]
pub struct Retention {
    /// Bytes of the earlier phases' named buffers.
    held: u64,
    /// Named bytes allocated in the current phase.
    named: u64,
    /// Every byte allocated in the current phase.
    phase: u64,
    peak: u64,
    /// The device runs the unfused fallbacks ([`unfused`]).
    pub unfused: bool,
    /// The compute width (bytes per element of the compute dtype).
    pub cw: u64,
}

impl Retention {
    /// A quantum at compute width `cw`, pricing the unfused fallbacks when `unfused`.
    pub fn on(cw: u64, unfused: bool) -> Self {
        Self {
            held: 0,
            named: 0,
            phase: SCALARS_BYTES,
            peak: 0,
            unfused,
            cw,
        }
    }

    /// Whether the compute dtype is narrower than f32 (bf16).
    pub fn narrow(&self) -> bool {
        self.cw < 4
    }

    /// `n` buffers of `elements` at `width` bytes, held as `kind`.
    pub fn alloc(&mut self, kind: Kind, n: u64, width: u64, elements: u64) -> &mut Self {
        let bytes = allocated(width.saturating_mul(elements)).saturating_mul(n);
        self.phase = self.phase.saturating_add(bytes);
        if kind == Kind::Named {
            self.named = self.named.saturating_add(bytes);
        }
        self
    }

    /// Graph-only bytes already rounded as allocated.
    pub fn temp_bytes(&mut self, bytes: u64) -> &mut Self {
        self.phase = self.phase.saturating_add(bytes);
        self
    }

    /// One graph-only buffer.
    pub fn temp(&mut self, width: u64, elements: u64) -> &mut Self {
        self.alloc(Kind::Temp, 1, width, elements)
    }

    /// One buffer the quantum's code holds.
    pub fn named(&mut self, width: u64, elements: u64) -> &mut Self {
        self.alloc(Kind::Named, 1, width, elements)
    }

    /// A sub-layer barrier: this phase's peak is the held bytes plus everything it allocated; its
    /// graph-only buffers are released, its named ones held on.
    pub fn barrier(&mut self) -> &mut Self {
        self.peak = self.peak.max(self.held.saturating_add(self.phase));
        self.held = self.held.saturating_add(self.named);
        self.named = 0;
        self.phase = SCALARS_BYTES;
        self
    }

    /// Start the quantum's LAST phase — the graph a block returns, evaluated by the quantum after
    /// the block's code (and every binding in it) is gone: only what that graph reads stays held.
    /// Follow with [`Self::hold`] for each such input.
    pub fn last_phase(&mut self) -> &mut Self {
        self.barrier();
        self.held = 0;
        self
    }

    /// A buffer an earlier phase allocated that the current phase's graph still reads.
    pub fn hold(&mut self, width: u64, elements: u64) -> &mut Self {
        self.held = self
            .held
            .saturating_add(allocated(width.saturating_mul(elements)));
        self
    }

    /// A peak reached INSIDE the current phase by `extra` transient bytes on top of everything the
    /// phase holds so far (a per-chunk evaluation's own buffers).
    pub fn transient(&mut self, extra: u64) -> &mut Self {
        self.peak = self
            .peak
            .max(self.held.saturating_add(self.phase).saturating_add(extra));
        self
    }

    /// The quantum's price: its largest phase.
    pub fn price(mut self) -> u64 {
        self.barrier();
        self.peak
    }

    /// A compute-width projection `[rows, in] → [rows, out]` (`crate::nn::Linear::forward`): the
    /// compute-width cast of an f32 input when `cast`, the zero-padded input of an unaligned packed
    /// base (`pad` extra columns), and the output, held as `out_kind`.
    pub fn linear(
        &mut self,
        rows: u64,
        in_features: u64,
        out_features: u64,
        cast: bool,
        pad: u64,
        out_kind: Kind,
    ) -> &mut Self {
        let cw = self.cw;
        if cast {
            self.temp(cw, rows * in_features);
        }
        if pad > 0 {
            self.temp(cw, rows * (in_features + pad));
        }
        self.alloc(out_kind, 1, cw, rows * out_features)
    }

    /// `crate::nn::RmsNorm::forward` over `elements` in `rows` rows, its f32 result held as
    /// `out_kind`. Unfused: the f32 cast of a compute-width input, `square` and the per-row mean
    /// (the epsilon add and `rsqrt` donate) are graph temporaries, and the normalizing product is
    /// always a new buffer — its operand is still referenced by the `square` in flight, so it cannot
    /// donate (the gain product donates). Fused: one result buffer, unless the input donates it
    /// ([`NormInput::Donor`]; a compute-width input's f32 cast donates into the result).
    pub fn norm(
        &mut self,
        elements: u64,
        rows: u64,
        input: NormInput,
        out_kind: Kind,
    ) -> &mut Self {
        if self.unfused {
            if input == NormInput::Cast {
                self.temp(4, elements);
            }
            self.temp(4, elements)
                .temp(4, rows)
                .alloc(out_kind, 1, 4, elements)
        } else if input == NormInput::Donor {
            self
        } else {
            self.alloc(out_kind, 1, 4, elements)
        }
    }

    /// Who holds the buffer of a [`NormInput::Donor`] whose norm's result is held as `out_kind`:
    /// fused, the result reuses it; unfused, it is a graph temporary.
    pub fn donor(&self, out_kind: Kind) -> Kind {
        if self.unfused {
            Kind::Temp
        } else {
            out_kind
        }
    }

    /// `crate::nn::Rope::apply` over an f32 input of `elements`: the four rotated half-planes
    /// (`re·cos`, `im·sin`, `re·sin`, `im·cos`; the difference and the sum donate) and their stacked
    /// result, held as `out_kind`.
    pub fn rope(&mut self, elements: u64, out_kind: Kind) -> &mut Self {
        self.alloc(Kind::Temp, 4, 4, elements / 2)
            .alloc(out_kind, 1, 4, elements)
    }

    /// One attention call (`crate::nn::attention_budgeted`) — see [`Attention`].
    pub fn attention(&mut self, a: &Attention) -> &mut Self {
        let cw = self.cw;
        let q = a.batch * a.heads * a.tokens * a.head_dim;
        let kv = a.batch * a.kv_heads * a.tokens * a.head_dim;
        // The compute-width views of q / k / v (`t`): a cast of an f32 operand is a new buffer.
        if a.q_cast {
            self.temp(cw, q);
        }
        if a.k_cast {
            self.temp(cw, kv);
        }
        if a.v_cast {
            self.temp(cw, kv);
        }
        if let Some((elements, cast)) = a.mask {
            if cast {
                self.temp(cw, elements);
            }
        }
        let rows = a.query_rows.min(a.tokens).max(1);
        let chunked = rows < a.tokens;
        let chunks = a.tokens.div_ceil(rows);
        let q_rows = a.batch * a.heads * rows * a.head_dim;
        let scores = a.batch * a.heads * rows * a.tokens;
        // One SDPA call's own buffers besides its output: unfused, the scaled query, the key and
        // value copies its batched matmuls take (broadcast over the query groups, so query-sized)
        // and the scores (the mask add and the softmax donate); fused, none. A DENSE query (its own
        // buffer, not a slice of a fused qkv) leaves the scaled query in the query's transposed
        // layout, which the score matmul copies once more (a chunk of it stays dense only at batch
        // one).
        let relaid = a.q_dense && (!chunked || a.batch == 1);
        let call = if self.unfused {
            (1 + u64::from(relaid)) * allocated(cw * q_rows)
                + 2 * allocated(cw * q)
                + allocated(cw * scores)
        } else {
            0
        };
        if !chunked {
            self.phase = self.phase.saturating_add(call);
            // The SDPA output, then (unfused, `[B, H, N, D]` re-laid out) its merged copy; the
            // fused kernel writes `[B, N, H, D]`, which merges as a view.
            if self.unfused {
                self.temp(cw, q);
            }
            return self.alloc(a.out_kind, 1, cw, q);
        }
        if a.eval_per_chunk {
            // Each chunk is evaluated before the next is built: its own buffers are transient, the
            // chunk outputs accumulate (counted whole: they are the query-sized result).
            self.transient(call.saturating_add(allocated(cw * q)));
        } else {
            self.phase = self
                .phase
                .saturating_add(call.saturating_mul(chunks))
                .saturating_add(allocated(cw * q));
        }
        // The chunk outputs, their concatenation and its merged (re-laid out) copy.
        if a.eval_per_chunk {
            self.temp(cw, q);
        }
        self.temp(cw, q).alloc(a.out_kind, 1, cw, q)
    }
}

/// One attention call's geometry (`crate::nn::attention_budgeted` over `[B, N, H, D]` q and
/// `[B, N, KV, D]` k / v).
#[derive(Clone, Copy, Debug)]
pub struct Attention {
    pub batch: u64,
    pub tokens: u64,
    pub heads: u64,
    pub kv_heads: u64,
    pub head_dim: u64,
    /// Query rows per SDPA call (`tokens` when the plan does not chunk).
    pub query_rows: u64,
    /// Whether a chunking plan evaluates each chunk before building the next.
    pub eval_per_chunk: bool,
    /// Whether q is a buffer of its own (not a slice of a fused qkv projection).
    pub q_dense: bool,
    /// Whether q / k / v reach the attention in f32 (their compute-width cast is a new buffer).
    pub q_cast: bool,
    pub k_cast: bool,
    pub v_cast: bool,
    /// An additive key mask: its elements, and whether it is cast to the compute width.
    pub mask: Option<(u64, bool)>,
    /// Who holds the merged `[B, N, H·D]` result.
    pub out_kind: Kind,
}
