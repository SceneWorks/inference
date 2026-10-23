// Gemma closed graph, included by native_graph (not a second module).
impl Walk<'_> {
    fn gemma_embedding(&mut self, l: u64, hidden: u64) -> Result<()> {
        let inv = &self.facts.text_encoder;
        let weight = tensor(inv, "model.embed_tokens.weight")?;
        let ws = shape(weight, 2)?;
        self.ops(l, 2)?; // ids and pad-mask uploads
        if let Some(scales) = inv.0.get("model.embed_tokens.scales") {
            let ss = shape(scales, 2)?;
            let biases = tensor(inv, "model.embed_tokens.biases")?;
            if biases.shape != scales.shape
                || ss[0] != ws[0]
                || weight.dtype != Dtype::Uint32
                || !hidden.is_multiple_of(ss[1])
            {
                return Err(invalid("packed embedding affine extent mismatch"));
            }
            let bits = mul(ws[1], 32)?
                .checked_div(hidden)
                .ok_or_else(|| invalid("empty embedding width"))?;
            if !matches!(bits, 4 | 8) || mul(ws[1], 32)? != mul(hidden, bits)? {
                return Err(invalid("packed embedding codes are not Q4/Q8"));
            }
            // ops.cpp:3394-3490/3510-3553, cpu/indexing.cpp:177-235,
            // metal/indexing.cpp:57ff: a single I32 index array is already the
            // promoted index type. Gather allocates only its result, even for
            // a strided source; Take's transpose/squeeze are views.
            self.a(mul(l, ws[1])?)?;
            self.ops(mul(l, ss[1])?, 2)?;
            if self.ledger.backend == NativeBackend::Metal {
                // metal/quantized.cpp:1801-1827: dense output plus possible
                // independent row-contiguous repairs of codes/scales/biases.
                self.a(mul(l, hidden)?)?;
                self.a(mul(l, ws[1])?)?;
                self.ops(mul(l, ss[1])?, 2)?;
            } else {
                // ops.cpp:5715-5768 CPU affine fallback, NOT a Metal kernel:
                // per-field left/right scalar and output; concatenate U32;
                // cast for affine product; product and bias addition.
                for _ in 0..32 / bits {
                    self.scalars(2)?;
                    self.ops(mul(l, ws[1])?, 2)?;
                }
                self.ops(mul(l, hidden)?, 4)?;
            }
        } else {
            if ws[1] != hidden {
                return Err(invalid("dense embedding width mismatch"));
            }
            self.a(mul(l, hidden)?)?;
        }
        self.ops(mul(l, hidden)?, 3)?; // common f32 return, Gemma dtype cast, scaling multiply
        self.scalars(2)?;
        let hd = dim(self.facts.gemma.head_dim)?;
        self.ops(mul(l, hd)?, 2)?; // host RoPE table uploads
        self.a(mul(l, l)?)?; // host causal mask upload
        self.a(l)?; // pad mask Float32 cast
        self.scalars(2)?;
        self.ops(l, 2)?; // subtract pad from one, multiply neg
        self.a(mul(l, l)?) // causal + pad mask
    }
    fn gemma_rope(&mut self, l: u64, heads: u64, hd: u64) -> Result<()> {
        let n = mul(mul(l, heads)?, hd)?;
        self.ops(mul(l, hd)?, 2)?; // cos/sin casts
        self.a(n / 2)?; // negate rotate_half's second half
        self.a(n)?; // concatenate halves
        self.ops(n, 3) // x*cos, rotate_half*sin, sum
    }
    fn gemma_layer(&mut self, i: i32) -> Result<()> {
        let cfg = &self.facts.gemma;
        let inv = &self.facts.text_encoder;
        let (l, hidden, heads, kv, hd) = (
            self.facts.gemma_tokens,
            dim(cfg.hidden_size)?,
            dim(cfg.num_heads)?,
            dim(cfg.num_kv_heads)?,
            dim(cfg.head_dim)?,
        );
        if !heads.is_multiple_of(kv) {
            return Err(invalid("Gemma head ratio mismatch"));
        }
        let prefix = format!("model.layers.{i}");
        self.rms(l, hidden, false)?;
        for name in ["q_proj", "k_proj", "v_proj"] {
            self.linear(inv, &format!("{prefix}.self_attn.{name}"), l, hidden)?;
        }
        self.gemma_rope(l, heads, hd)?;
        self.gemma_rope(l, kv, hd)?;
        if heads != kv {
            self.ops(mul(mul(l, heads)?, hd)?, 2)?;
        }
        self.mm(heads, l, l, hd)?;
        self.scalars(6)?; // typed scale, cap and inverse cap
        let scores = mul(heads, mul(l, l)?)?;
        self.ops(scores, 4)?; // scale, inverse-cap product, tanh, cap product
        self.a(scores)?; // scores Float32 cast
        self.a(mul(l, l)?)?; // mask Float32 cast
        self.a(scores)?; // additive mask
        self.ledger.softmax(4, scores)?;
        self.a(scores)?; // attention cast back
        self.mm(heads, l, hd, l)?;
        self.a(mul(mul(l, heads)?, hd)?)?; // transposed output reshape copy
        self.linear(
            inv,
            &format!("{prefix}.self_attn.o_proj"),
            l,
            mul(heads, hd)?,
        )?;
        self.rms(l, hidden, false)?;
        self.a(mul(l, hidden)?)?;
        self.rms(l, hidden, false)?;
        let intermediate = self.linear(inv, &format!("{prefix}.mlp.gate_proj"), l, hidden)?;
        self.gelu(mul(l, intermediate)?)?;
        self.linear(inv, &format!("{prefix}.mlp.up_proj"), l, hidden)?;
        self.a(mul(l, intermediate)?)?;
        self.linear(inv, &format!("{prefix}.mlp.down_proj"), l, intermediate)?;
        self.rms(l, hidden, false)?;
        self.a(mul(l, hidden)?)?;

        Ok(())
    }
    fn gemma(&mut self, guided: bool) -> Result<()> {
        let cfg = &self.facts.gemma;
        let (l, hidden, heads, kv, _hd) = (
            self.facts.gemma_tokens,
            dim(cfg.hidden_size)?,
            dim(cfg.num_heads)?,
            dim(cfg.num_kv_heads)?,
            dim(cfg.head_dim)?,
        );
        if !heads.is_multiple_of(kv) {
            return Err(invalid("Gemma head ratio mismatch"));
        }
        // Layer::from_weights/Gemma2::from_weights: typed one + weight owners.
        for _ in 0..sum(mul(dim(cfg.num_layers)?, 4)?, 1)? {
            self.scalars(2)?;
            self.a(hidden)?;
        }
        for _ in 0..if guided { 2 } else { 1 } {
            self.gemma_embedding(l, hidden)?;
            for i in 0..cfg.num_layers {
                self.gemma_layer(i)?;
            }
            self.rms(l, hidden, false)?;
            self.a(mul(self.facts.caption_tokens, hidden)?)?;
            self.ops(self.facts.caption_tokens, 2)?; // selected-index and selected-mask uploads
        }
        Ok(())
    }
}
