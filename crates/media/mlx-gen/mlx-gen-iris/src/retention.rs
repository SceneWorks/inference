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
pub struct Retention<'a> {
    adapters: Option<(&'a crate::adapters::ProjectionAdapterFacts, &'a str)>,
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

impl<'a> Retention<'a> {
    /// A quantum at compute width `cw`, pricing the unfused fallbacks when `unfused`.
    pub fn on(cw: u64, unfused: bool) -> Self {
        Self {
            adapters: None,
            held: 0,
            named: 0,
            phase: SCALARS_BYTES,
            peak: 0,
            unfused,
            cw,
        }
    }

    /// Bind upstream projection identities for this counted quantum. `prefix` is the actual
    /// block path (or empty for skeleton projections), shared with the header-only price.
    pub fn with_adapters(
        mut self,
        facts: &'a crate::adapters::ProjectionAdapterFacts,
        prefix: &'a str,
    ) -> Self {
        self.adapters = Some((facts, prefix));
        self
    }

    /// Count one named production projection, with its exact attached residuals and request rows.
    pub fn linear_at(
        &mut self,
        path: &str,
        rows: u64,
        input: u64,
        output: u64,
        cast: bool,
        pad: u64,
        kind: Kind,
    ) -> &mut Self {
        let facts = self.adapters.and_then(|(facts, prefix)| {
            let key = if prefix.is_empty() {
                path.to_owned()
            } else {
                format!("{prefix}.{path}")
            };
            facts.get(&key)
        });
        self.adapted_linear(
            rows,
            input,
            output,
            cast,
            pad,
            kind,
            facts.map(Vec::as_slice).unwrap_or(&[]),
        )
    }

    /// Residual-only allocations for a skeleton projection whose base output the pre-existing
    /// law already counted (e.g. the two modulation cores' combined outputs). No base/cast/padding
    /// is charged twice. Each stack's own factors give the exact input/output widths.
    pub fn adapter_extras_at(&mut self, path: &str, rows: u64, copy_input: bool) -> &mut Self {
        let facts = self.adapters.and_then(|(facts, prefix)| {
            let key = if prefix.is_empty() {
                path.to_owned()
            } else {
                format!("{prefix}.{path}")
            };
            facts.get(&key)
        });
        if let Some(facts) = facts {
            let compute = if self.cw == 4 {
                mlx_rs::Dtype::Float32
            } else {
                mlx_rs::Dtype::Bfloat16
            };
            for adapter in facts.iter().filter(|a| !a.disabled()) {
                use mlx_gen::adapters::facts::AdapterFacts;
                let (input, output) = match adapter {
                    AdapterFacts::Lora { a, b, .. } => (a.shape[0], b.shape[1]),
                    AdapterFacts::Materialized { delta, .. } => (delta.shape[1], delta.shape[0]),
                    AdapterFacts::Structured { w1, w2, .. } => {
                        (w1.shape[1] * w2.shape[1], w1.shape[0] * w2.shape[0])
                    }
                };
                if copy_input {
                    let dtype = match adapter {
                        AdapterFacts::Lora { a, .. } => {
                            mlx_rs::Dtype::from_promoting_types(compute, a.dtype)
                        }
                        _ => compute,
                    };
                    self.temp(
                        mlx_gen::adapters::facts::dtype_width(dtype),
                        rows * input.max(0) as u64,
                    );
                }
                self.adapter_residual(
                    rows,
                    input.max(0) as u64,
                    output.max(0) as u64,
                    compute,
                    adapter,
                );
                self.temp(self.cw, rows * output.max(0) as u64);
            }
        }
        self
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

    /// The same base projection as [`Self::linear`], followed by its actual ordered residual
    /// stack. No-adapter (including all-disabled) pricing is exactly the old base law. Parameters
    /// are already resident; only casts/copies and newly produced forward buffers belong here.
    ///
    /// This is a source-counted allocation upper bound: it does not require multiply/add donation
    /// to succeed. It is NOT physical-footprint evidence or a subtractable anchor basis.
    pub fn adapted_linear(
        &mut self,
        rows: u64,
        in_features: u64,
        out_features: u64,
        cast: bool,
        pad: u64,
        out_kind: Kind,
        adapters: &[mlx_gen::adapters::facts::AdapterFacts],
    ) -> &mut Self {
        let live = adapters.iter().filter(|a| !a.disabled()).count();
        self.linear(
            rows,
            in_features,
            out_features,
            cast,
            pad,
            if live == 0 { out_kind } else { Kind::Temp },
        );
        if live == 0 {
            return self;
        }
        let compute = if self.cw == 4 {
            mlx_rs::Dtype::Float32
        } else {
            mlx_rs::Dtype::Bfloat16
        };
        for (index, adapter) in adapters.iter().filter(|a| !a.disabled()).enumerate() {
            self.adapter_residual(rows, in_features, out_features, compute, adapter);
            // `apply_adapters`: narrow the natural residual to the base output dtype, then add.
            // Its preceding output remains referenced by the add's input graph.
            self.alloc(
                if index + 1 == live {
                    out_kind
                } else {
                    Kind::Temp
                },
                1,
                self.cw,
                rows * out_features,
            );
        }
        self
    }

    fn adapter_residual(
        &mut self,
        rows: u64,
        input: u64,
        output: u64,
        compute: mlx_rs::Dtype,
        adapter: &mlx_gen::adapters::facts::AdapterFacts,
    ) {
        use mlx_gen::adapters::facts::{dtype_width, AdapterFacts};
        use mlx_rs::Dtype;
        let cast = |r: &mut Self, from: Dtype, to: Dtype, n| {
            if from != to {
                r.temp(dtype_width(to), n);
            }
        };
        let residual_dtype = match adapter {
            AdapterFacts::Lora { a, b, .. } => {
                let rank = a.shape[1].max(0) as u64;
                let first = Dtype::from_promoting_types(compute, a.dtype);
                let second = Dtype::from_promoting_types(first, b.dtype);
                cast(self, compute, first, rows * input);
                cast(self, a.dtype, first, a.elements());
                self.temp(dtype_width(first), rows * rank);
                cast(self, first, second, rows * rank);
                cast(self, b.dtype, second, b.elements());
                self.temp(dtype_width(second), rows * output);
                second
            }
            AdapterFacts::Materialized { delta, .. } => {
                cast(self, delta.dtype, compute, delta.elements());
                self.temp(self.cw, rows * output);
                compute
            }
            AdapterFacts::Structured { w1, w2, .. } => {
                cast(self, w1.dtype, compute, w1.elements());
                cast(self, w2.dtype, compute, w2.elements());
                // w1 [a,c] @ x [rows,c,d] -> [rows,a,d], then @ w2.T -> [rows,a,b].
                let cw = self.cw;
                self.temp(
                    cw,
                    rows * w1.shape[0].max(0) as u64 * w2.shape[1].max(0) as u64,
                )
                .temp(cw, rows * output);
                // The structured scale is already baked into w2, so no scale op here.
                return;
            }
        };
        // `scalar(scale).as_dtype(r.dtype())`, followed by multiply, even at scale one.
        self.temp(4, 1);
        cast(self, Dtype::Float32, residual_dtype, 1);
        self.temp(dtype_width(residual_dtype), rows * output);
        cast(self, residual_dtype, compute, rows * output);
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
