//! Z-owned constructor and forward recipes. Shapes/dtypes come from the SAME selected header
//! dictionaries as the constructors, never resident/scratch/work-basis numbers.
use super::*;
use crate::quantum as q;
use mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader;

fn error(message: impl Into<String>) -> Error {
    Error::Msg(format!("Z source graph: {}", message.into()))
}
fn dim(n: i32) -> Result<u64> {
    u64::try_from(n).map_err(|_| error("negative architecture dimension"))
}
fn header_tensor(h: &SafetensorsTensorHeader) -> Result<Tensor> {
    Tensor::row(
        &h.shape.iter().map(|&n| n as u64).collect::<Vec<_>>(),
        header_dtype(h.dtype)?,
    )
}

/// One concrete dictionary/source-instance. Reopening a streamed block creates a NEW instance;
/// its raw loads and packing are charged again, with no presumed cache hit or retirement credit.
#[derive(Clone)]
pub(super) struct Catalog {
    pub values: BTreeMap<String, Tensor>,
    raw: BTreeMap<String, (u64, Tensor)>,
    origins: BTreeMap<String, String>,
    loaded: BTreeSet<String>,
    conversions: BTreeMap<String, bool>, // true only for TokenEmbedding's no-BF16-cast quantizer
    converted: BTreeSet<String>,
}
impl Catalog {
    fn component(
        effective: &Component,
        raw: &Component,
        conversions: &[(String, u64, u64, u64)],
        embedding: bool,
        dit: bool,
    ) -> Result<Self> {
        let mut headers = effective
            .headers
            .iter()
            .map(|h| (h.name.clone(), h.clone()))
            .collect::<BTreeMap<_, _>>();
        let mut raw_headers = raw
            .headers
            .iter()
            .map(|h| (h.name.clone(), h.clone()))
            .collect::<BTreeMap<_, _>>();
        let origins = if dit {
            crate::loader::transformer_header_origins(&raw_headers)
        } else {
            raw_headers.keys().map(|k| (k.clone(), k.clone())).collect()
        };
        if dit {
            crate::loader::remap_transformer_header_keys(&mut headers);
            crate::loader::remap_transformer_header_keys(&mut raw_headers);
        }
        let values = headers
            .iter()
            .map(|(k, h)| Ok((k.clone(), header_tensor(h)?)))
            .collect::<Result<_>>()?;
        let raw = raw
            .headers
            .iter()
            .map(|h| Ok((h.name.clone(), (h.data_bytes, header_tensor(h)?))))
            .collect::<Result<_>>()?;
        Ok(Self {
            values,
            raw,
            origins,
            loaded: BTreeSet::new(),
            conversions: conversions
                .iter()
                .map(|(p, ..)| (p.clone(), embedding && p == "model.embed_tokens"))
                .collect(),
            converted: BTreeSet::new(),
        })
    }
    pub(super) fn from_vae(facts: &Facts, owners: &VaeOwnerHistory) -> Result<Self> {
        let raw = facts.load_components.as_ref().map_or(&facts.vae, |c| &c[2]);
        let source = Component {
            files: Vec::new(),
            headers: owners
                .remap
                .current
                .values()
                .map(|v| v.header.clone())
                .collect(),
        };
        let targets = ["", "encoder."]
            .into_iter()
            .flat_map(|p| {
                crate::vae::attention::QUANTIZATION_LINEAR_PATHS
                    .into_iter()
                    .map(move |leaf| (format!("{p}mid_block.attentions.0.{leaf}"), false))
            })
            .collect::<Vec<_>>();
        let (effective, conversions) = if let Some(bits) = facts.quant_bits {
            source.load_quantized_with_bias(&targets, bits, |_| true)?
        } else {
            (source, Vec::new())
        };
        let mut out = Self::component(&effective, raw, &conversions, false, false)?;
        out.origins = owners
            .remap
            .current
            .iter()
            .map(|(k, v)| (k.clone(), v.raw.clone()))
            .collect();
        for (key, value) in &owners.remap.current {
            if conversions.iter().any(|(p, ..)| {
                key == &format!("{p}.weight")
                    || key == &format!("{p}.scales")
                    || key == &format!("{p}.biases")
            }) {
                continue;
            }
            let mut t = out
                .raw
                .get(&value.raw)
                .ok_or_else(|| error(format!("missing VAE raw origin {}", value.raw)))?
                .1
                .clone();
            for step in &value.path {
                if step.transpose {
                    t = t.transpose(&[0, 2, 3, 1])?;
                }
            }
            if let Some(effective) = out.values.get(key) {
                // The quantizer may cast an output bias; that cast is separately charged on use.
                if t.dtype == effective.dtype && t.shape == effective.shape {
                    out.values.insert(key.clone(), t);
                }
            }
        }
        Ok(out)
    }
    pub fn value(&self, key: &str) -> Result<Tensor> {
        self.values
            .get(key)
            .cloned()
            .ok_or_else(|| error(format!("constructor-selected key missing: {key}")))
    }
    pub fn has(&self, key: &str) -> bool {
        self.values.contains_key(key)
    }
    fn raw_value(&self, key: &str) -> Result<Tensor> {
        let raw = self.origins.get(key).map_or(key, String::as_str);
        self.raw
            .get(raw)
            .map(|(_, t)| t.clone())
            .ok_or_else(|| error(format!("missing raw source {key}")))
    }
    fn load_raw(&mut self, g: &mut Graph, key: &str) -> Result<()> {
        let raw = self.origins.get(key).map_or(key, String::as_str).to_owned();
        if self.loaded.insert(raw.clone()) {
            let (bytes, _) = self
                .raw
                .get(&raw)
                .ok_or_else(|| error(format!("missing materialized raw source {raw}")))?;
            g.load(*bytes, &raw)?;
        }
        Ok(())
    }
    pub fn preload(&mut self, g: &mut Graph, keys: impl IntoIterator<Item = String>) -> Result<()> {
        for key in keys {
            let converted = self
                .conversions
                .keys()
                .find(|p| key.starts_with(&format!("{p}.")))
                .cloned();
            if let Some(base) = converted {
                self.load_raw(g, &format!("{base}.weight"))?;
                if self.origins.contains_key(&format!("{base}.bias")) {
                    self.load_raw(g, &format!("{base}.bias"))?;
                }
            } else {
                self.load_raw(g, &key)?;
            }
        }
        Ok(())
    }
    pub fn linear_keys(&self, base: &str, bias: bool) -> Vec<String> {
        ["weight", "scales", "biases", "bias"]
            .into_iter()
            .filter(|s| *s != "bias" || bias)
            .map(|s| format!("{base}.{s}"))
            .filter(|k| self.has(k))
            .collect()
    }
    pub fn prepare_linear(&mut self, g: &mut Graph, base: &str, bias: bool) -> Result<()> {
        self.preload(g, self.linear_keys(base, bias))?;
        if let Some(&embedding) = self.conversions.get(base) {
            if self.converted.insert(base.into()) {
                let w = self.raw_value(&format!("{base}.weight"))?;
                let w = if embedding {
                    w
                } else {
                    g.cast(&w, Dtype::Bfloat16, base)?
                };
                let output = self.value(&format!("{base}.weight"))?;
                let bits = mul(
                    *output
                        .shape
                        .last()
                        .ok_or_else(|| error("packed weight rank"))?,
                    32,
                )? / *w.shape.last().unwrap();
                let generated = g.affine_quantize(&w, bits, base)?;
                for (suffix, t) in ["weight", "scales", "biases"].into_iter().zip(generated) {
                    self.values.insert(format!("{base}.{suffix}"), t);
                }
                if bias && self.has(&format!("{base}.bias")) {
                    let b = self.raw_value(&format!("{base}.bias"))?;
                    let b = g.cast(&b, Dtype::Bfloat16, base)?;
                    self.values.insert(format!("{base}.bias"), b);
                }
            }
        }
        Ok(())
    }
    pub fn linear(&mut self, g: &mut Graph, base: &str, x: &Tensor, bias: bool) -> Result<Tensor> {
        self.prepare_linear(g, base, bias)?;
        let weight = self.value(&format!("{base}.weight"))?;
        let output_bias = if bias && self.has(&format!("{base}.bias")) {
            Some(self.value(&format!("{base}.bias"))?)
        } else {
            None
        };
        if self.has(&format!("{base}.scales")) {
            g.qmm(
                x,
                &weight,
                &self.value(&format!("{base}.scales"))?,
                &self.value(&format!("{base}.biases"))?,
                output_bias.as_ref(),
                base,
            )
        } else {
            g.matrix_product(x, &weight.transpose(&[1, 0])?, output_bias.as_ref(), base)
        }
    }
    fn embedding(&mut self, g: &mut Graph, tokens: u64) -> Result<Tensor> {
        let base = "model.embed_tokens";
        self.prepare_linear(g, base, false)?;
        let w = self.value(&format!("{base}.weight"))?;
        let gathered = g.gather_rows(&w, tokens, base)?;
        let out = if self.has(&format!("{base}.scales")) {
            let sc = g.gather_rows(&self.value(&format!("{base}.scales"))?, tokens, base)?;
            let bi = g.gather_rows(&self.value(&format!("{base}.biases"))?, tokens, base)?;
            let hidden = mul(*sc.shape.last().unwrap(), 64)?;
            let bits = mul(*w.shape.last().unwrap(), 32)? / hidden;
            g.affine_dequantize(&gathered, &sc, &bi, bits, base)?
        } else {
            gathered
        };
        let out = g.cast(&out, Dtype::Float32, base)?;
        g.reshape(&out, &[1, tokens, out.shape[1]], base)
    }
    fn fresh_stream(&self) -> Self {
        Self {
            loaded: BTreeSet::new(),
            converted: BTreeSet::new(),
            ..self.clone()
        }
    }
}

fn norm(g: &mut Graph, c: &mut Catalog, key: &str, x: &Tensor) -> Result<Tensor> {
    c.preload(g, [key.into()])?;
    g.rms(x, &c.value(key)?, key)
}
fn te_rope(g: &mut Graph, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let last = x.shape.len() - 1;
    let half = x.split(last, x.shape[last] / 2)?;
    let neg = g.unary(&half, "text_encoder/attention.rs rotate_half negative")?;
    let rot = g.concat(
        &[neg, half],
        last,
        "text_encoder/attention.rs rotate_half concat",
    )?;
    let a = g.binary(x, cos, "text_encoder/attention.rs x*cos")?;
    let b = g.binary(&rot, sin, "text_encoder/attention.rs rotate*sin")?;
    g.binary(&a, &b, "text_encoder/attention.rs RoPE add")
}
fn te_layer(
    g: &mut Graph,
    c: &mut Catalog,
    index: usize,
    x: &Tensor,
    cfg: crate::text_encoder::ZTextEncoderConfig,
    cos: &Tensor,
    sin: &Tensor,
    mask: &Tensor,
) -> Result<Tensor> {
    let p = format!("model.layers.{index}");
    let n = x.shape[1];
    let a = norm(g, c, &format!("{p}.input_layernorm.weight"), x)?;
    let mut projected = Vec::new();
    for (leaf, heads) in [
        ("q_proj", cfg.n_heads),
        ("k_proj", cfg.n_kv_heads),
        ("v_proj", cfg.n_kv_heads),
    ] {
        let y = c.linear(g, &format!("{p}.self_attn.{leaf}"), &a, false)?;
        projected.push(g.reshape(&y, &[1, n, dim(heads)?, dim(cfg.head_dim)?], &p)?);
    }
    let q = norm(g, c, &format!("{p}.self_attn.q_norm.weight"), &projected[0])?;
    let k = norm(g, c, &format!("{p}.self_attn.k_norm.weight"), &projected[1])?;
    let q = te_rope(g, &q, cos, sin)?;
    let k = te_rope(g, &k, cos, sin)?;
    let groups = dim(cfg.n_heads / cfg.n_kv_heads)?;
    let repeat = |g: &mut Graph, t: &Tensor| -> Result<Tensor> {
        if groups == 1 {
            return Ok(t.clone());
        }
        let t = g.reshape(t, &[1, n, dim(cfg.n_kv_heads)?, 1, dim(cfg.head_dim)?], &p)?;
        let t = t.broadcast(&[1, n, dim(cfg.n_kv_heads)?, groups, dim(cfg.head_dim)?])?;
        g.reshape(&t, &[1, n, dim(cfg.n_heads)?, dim(cfg.head_dim)?], &p)
    };
    let k = repeat(g, &k)?;
    let v = repeat(g, &projected[2])?;
    let y = g.sdpa(
        &q.transpose(&[0, 2, 1, 3])?,
        &k.transpose(&[0, 2, 1, 3])?,
        &v.transpose(&[0, 2, 1, 3])?,
        Some(mask),
        &p,
    )?;
    let y = g.reshape(
        &y.transpose(&[0, 2, 1, 3])?,
        &[1, n, mul(dim(cfg.n_heads)?, dim(cfg.head_dim)?)?],
        &p,
    )?;
    let y = c.linear(g, &format!("{p}.self_attn.o_proj"), &y, false)?;
    let h = g.binary(x, &y, &p)?;
    let a = norm(g, c, &format!("{p}.post_attention_layernorm.weight"), &h)?;
    let gate = c.linear(g, &format!("{p}.mlp.gate_proj"), &a, false)?;
    let gate = g.silu(&gate, &p)?;
    let up = c.linear(g, &format!("{p}.mlp.up_proj"), &a, false)?;
    let gate = g.binary(&gate, &up, &p)?;
    let y = c.linear(g, &format!("{p}.mlp.down_proj"), &gate, false)?;
    g.binary(&h, &y, &p)
}

fn dit_rope(g: &mut Graph, x: &Tensor, freqs: &Tensor) -> Result<Tensor> {
    let [b, n, h, d]: [u64; 4] = x
        .shape
        .clone()
        .try_into()
        .map_err(|_| error("DiT RoPE rank"))?;
    let half = d / 2;
    let a = g.reshape(x, &[b, n, h, half, 2], "attention.rs interleaved input")?;
    let a = a.split(4, 1)?;
    let real = g.reshape(&a, &[b, n, h, half], "attention.rs real view")?;
    let imag = real.clone();
    let fc = g.reshape(freqs, &[1, n, 1, half, 2], "attention.rs frequencies")?;
    let fc = fc.split(4, 1)?;
    let cos = g.reshape(&fc, &[1, n, 1, half], "attention.rs cosine view")?;
    let sin = cos.clone();
    let ac = g.binary(&real, &cos, "attention.rs rope xr*cos")?;
    let bs = g.binary(&imag, &sin, "attention.rs rope xi*sin")?;
    let r = g.binary(&ac, &bs, "attention.rs rope subtract")?;
    let as_ = g.binary(&real, &sin, "attention.rs rope xr*sin")?;
    let bc = g.binary(&imag, &cos, "attention.rs rope xi*cos")?;
    let i = g.binary(&as_, &bc, "attention.rs rope add")?;
    let r = g.reshape(&r, &[b, n, h, half, 1], "attention.rs stack real view")?;
    let i = g.reshape(&i, &[b, n, h, half, 1], "attention.rs stack imag view")?;
    let out = g.concat(&[r, i], 4, "attention.rs stack_axis")?;
    g.reshape(&out, &[b, n, h, d], "attention.rs stack reshape")
}
fn attention(
    g: &mut Graph,
    c: &mut Catalog,
    p: &str,
    x: &Tensor,
    freqs: &Tensor,
    heads: u64,
    query_rows: u64,
) -> Result<Tensor> {
    let (b, n) = (x.shape[0], x.shape[1]);
    let mut qkv = Vec::new();
    for leaf in ["to_q", "to_k", "to_v"] {
        let y = c.linear(g, &format!("{p}.{leaf}"), x, false)?;
        let hd = *y.shape.last().unwrap() / heads;
        let mut y = g.reshape(&y, &[b, n, heads, hd], p)?;
        let norm_key = if leaf == "to_q" {
            Some(format!("{p}.norm_q.weight"))
        } else if leaf == "to_k" {
            Some(format!("{p}.norm_k.weight"))
        } else {
            None
        };
        if let Some(key) = norm_key {
            if c.has(&key) {
                y = norm(g, c, &key, &y)?;
            }
        }
        if leaf != "to_v" {
            y = dit_rope(g, &y, freqs)?;
        }
        qkv.push(y.transpose(&[0, 2, 1, 3])?);
    }
    let y = budgeted_sdpa(g, &qkv[0], &qkv[1], &qkv[2], query_rows, p)?;
    let y = g.reshape(
        &y.transpose(&[0, 2, 1, 3])?,
        &[b, n, mul(heads, y.shape[3])?],
        p,
    )?;
    c.linear(g, &format!("{p}.to_out.0"), &y, false)
}
pub(super) fn budgeted_sdpa(
    g: &mut Graph,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    rows: u64,
    site: &str,
) -> Result<Tensor> {
    if rows == 0 {
        return Err(error("zero attention query block"));
    }
    let n = q.shape[2];
    if rows >= n {
        return g.sdpa(q, k, v, None, site);
    }
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < n {
        let len = rows.min(n - start);
        let chunk = q.split(2, len)?;
        chunks.push(g.sdpa(&chunk, k, v, None, site)?);
        start = add(start, len)?;
    }
    g.concat(&chunks, 2, "attention.rs query chunk concatenate")
}
fn block(
    g: &mut Graph,
    c: &mut Catalog,
    p: &str,
    x: &Tensor,
    freqs: &Tensor,
    t: Option<&Tensor>,
    heads: u64,
    query_rows: u64,
) -> Result<Tensor> {
    let dim = *x.shape.last().unwrap();
    let modulation = if let Some(t) = t {
        let m = c.linear(g, &format!("{p}.adaLN_modulation.0"), t, true)?;
        let m = g.reshape(&m, &[1, 1, mul(4, dim)?], p)?;
        let part = m.split(2, dim)?;
        let one = g.rust_scalar(Dtype::Float32, p)?;
        let one = g.cast(&one, m.dtype, p)?;
        Some([
            g.binary(&part, &one, p)?,
            g.unary(&part, p)?,
            g.binary(&part, &one, p)?,
            g.unary(&part, p)?,
        ])
    } else {
        None
    };
    let mut a = norm(g, c, &format!("{p}.attention_norm1.weight"), x)?;
    if let Some(m) = &modulation {
        a = g.binary(&a, &m[0], p)?;
    }
    let a = attention(
        g,
        c,
        &format!("{p}.attention"),
        &a,
        freqs,
        heads,
        query_rows,
    )?;
    let mut a = norm(g, c, &format!("{p}.attention_norm2.weight"), &a)?;
    if let Some(m) = &modulation {
        a = g.binary(&m[1], &a, p)?;
    }
    let h = g.binary(x, &a, p)?;
    let mut a = norm(g, c, &format!("{p}.ffn_norm1.weight"), &h)?;
    if let Some(m) = &modulation {
        a = g.binary(&a, &m[2], p)?;
    }
    let ff = c.linear(g, &format!("{p}.feed_forward.w1"), &a, false)?;
    let ff = g.silu(&ff, p)?;
    let other = c.linear(g, &format!("{p}.feed_forward.w3"), &a, false)?;
    let ff = g.binary(&ff, &other, p)?;
    let ff = c.linear(g, &format!("{p}.feed_forward.w2"), &ff, false)?;
    let mut a = norm(g, c, &format!("{p}.ffn_norm2.weight"), &ff)?;
    if let Some(m) = &modulation {
        a = g.binary(&m[3], &a, p)?;
    }
    g.binary(&h, &a, p)
}
fn pad(g: &mut Graph, c: &Catalog, x: &Tensor, keep: &Tensor, key: &str) -> Result<Tensor> {
    let pad = c.value(key)?;
    let keep = g.cast(keep, pad.dtype, "transformer.rs apply_pad keep")?;
    let one = g.rust_scalar(Dtype::Float32, "transformer.rs apply_pad one")?;
    let one = g.cast(&one, pad.dtype, "transformer.rs apply_pad cast one")?;
    let inv = g.binary(&one, &keep, "transformer.rs apply_pad inverse")?;
    let a = g.binary(x, &keep, "transformer.rs apply_pad embedding product")?;
    let b = g.binary(&pad, &inv, "transformer.rs apply_pad token product")?;
    g.binary(&a, &b, "transformer.rs apply_pad sum")
}
fn pad_rows(g: &mut Graph, x: &Tensor, pad: u64) -> Result<Tensor> {
    if pad == 0 {
        return Ok(x.clone());
    }
    let zeros = g.immediate(
        &[pad, x.shape[1]],
        Dtype::Float32,
        "transformer.rs pad_rows host zeros upload",
    )?;
    g.concat(
        &[x.clone(), zeros],
        0,
        "transformer.rs pad_rows concatenate",
    )
}
fn patchify(g: &mut Graph, x: &Tensor, patch: u64, fpatch: u64, pad: u64) -> Result<Tensor> {
    let [c, f, h, w]: [u64; 4] = x
        .shape
        .clone()
        .try_into()
        .map_err(|_| error("patchify rank"))?;
    let t = g.reshape(
        x,
        &[c, f / fpatch, fpatch, h / patch, patch, w / patch, patch],
        "transformer.rs patchify first reshape",
    )?;
    let t = t.transpose(&[1, 3, 5, 2, 4, 6, 0])?;
    let t = g.reshape(
        &t,
        &[
            mul(mul(f / fpatch, h / patch)?, w / patch)?,
            mul(mul(mul(fpatch, patch)?, patch)?, c)?,
        ],
        "transformer.rs patchify transpose reshape",
    )?;
    pad_rows(g, &t, pad)
}

struct Prepared {
    cap: Tensor,
    x_keep: Tensor,
    cap_keep: Tensor,
    x_freqs: Tensor,
    cap_freqs: Tensor,
    unified_freqs: Tensor,
    n: u64,
    c: u64,
    img_pad: u64,
    control: Option<Tensor>,
}
fn prepare(
    p: &mut Program,
    model: &Model,
    route: &Route,
    c: &mut Catalog,
    control: &mut Option<Catalog>,
    caption: &Tensor,
) -> Result<Prepared> {
    let cfg = &model.facts.dit_cfg;
    let patch = dim(cfg.patch_size)?;
    let fp = dim(cfg.f_patch_size)?;
    let n0 = mul(
        mul(1 / fp, u64::from(route.height / 8) / patch)?,
        u64::from(route.width / 8) / patch,
    )?;
    let n = mul(n0.div_ceil(32), 32)?;
    let rows = caption.shape[0];
    let cn = mul(rows.div_ceil(32), 32)?;
    let cap = pad_rows(&mut p.graph, caption, cn - rows)?;
    p.graph
        .immediate(&[n, 3], Dtype::Int32, "transformer.rs image position ids")?;
    p.graph.immediate(
        &[cn, 3],
        Dtype::Int32,
        "transformer.rs caption position ids",
    )?;
    let x_keep = p
        .graph
        .immediate(&[n, 1], Dtype::Float32, "transformer.rs image keep")?;
    let cap_keep = p
        .graph
        .immediate(&[cn, 1], Dtype::Float32, "transformer.rs caption keep")?;
    let mut rope = |rows: u64| -> Result<Tensor> {
        let mut parts = Vec::new();
        for (&d, &len) in cfg.axes_dims.iter().zip(&cfg.axes_lens) {
            p.graph.immediate(
                &[rows],
                Dtype::Int32,
                "rope_embedder.rs axis gather indices",
            )?;
            let table = Tensor::row(&[dim(len)?, dim(d)? / 2, 2], Dtype::Float32)?;
            parts.push(
                p.graph
                    .gather_rows(&table, rows, "rope_embedder.rs gather axis table")?,
            );
        }
        p.graph
            .concat(&parts, 1, "rope_embedder.rs concatenate axes")
    };
    let xf = rope(n)?;
    let cf = rope(cn)?;
    let uf = p.graph.concat(
        &[xf.clone(), cf.clone()],
        0,
        "transformer.rs concatenate unified frequencies",
    )?;
    p.mark(q::DIT_PREPARE)?;
    let control_emb = if let Some(ctrl) = control {
        let latent = Tensor::row(
            &[
                33,
                1,
                u64::from(route.height / 8),
                u64::from(route.width / 8),
            ],
            Dtype::Float32,
        )?;
        let tokens = patchify(&mut p.graph, &latent, patch, fp, n - n0)?;
        let e = ctrl.linear(
            &mut p.graph,
            &format!("control_all_x_embedder.{patch}-{fp}"),
            &tokens,
            true,
        )?;
        let e = pad(&mut p.graph, c, &e, &x_keep, "x_pad_token")?;
        let e = p.graph.reshape(
            &e,
            &[1, n, dim(cfg.dim)?],
            "control_transformer.rs control embedding batch view",
        )?;
        p.mark(q::CONTROL_EMBED)?;
        Some(e)
    } else {
        None
    };
    Ok(Prepared {
        cap,
        x_keep,
        cap_keep,
        x_freqs: xf,
        cap_freqs: cf,
        unified_freqs: uf,
        n,
        c: cn,
        img_pad: n - n0,
        control: control_emb,
    })
}

fn timestep(
    g: &mut Graph,
    c: &mut Catalog,
    cfg: &crate::transformer::ZImageTransformerConfig,
) -> Result<Tensor> {
    let site = "timestep_embedder.rs forward";
    let t = g.immediate(&[1], Dtype::Float32, site)?;
    let freqs = g.immediate(
        &[1, dim(cfg.frequency_embedding_size)? / 2],
        Dtype::Float32,
        site,
    )?;
    let t = g.reshape(&t, &[1, 1], site)?;
    let args = g.binary(&t, &freqs, site)?;
    let cos = g.unary(&args, site)?;
    let sin = g.unary(&args, site)?;
    let mut x = g.concat(&[cos, sin], 1, site)?;
    let base = "t_embedder.linear1";
    if !c.has(&format!("{base}.scales")) {
        x = g.cast(&x, c.value(&format!("{base}.weight"))?.dtype, site)?;
    }
    let x = c.linear(g, base, &x, true)?;
    let x = g.silu(&x, site)?;
    c.linear(g, "t_embedder.linear2", &x, true)
}
fn hint(g: &mut Graph, x: &Tensor, h: &Tensor) -> Result<Tensor> {
    let scale = g.rust_scalar(Dtype::Float32, "control_transformer.rs add_hint scale")?;
    let scaled = g.binary(h, &scale, "control_transformer.rs add_hint multiply")?;
    g.binary(x, &scaled, "control_transformer.rs add_hint residual")
}
fn control_stack(
    p: &mut Program,
    catalog: &mut Catalog,
    prefix: &str,
    count: usize,
    input: &Tensor,
    base: &Tensor,
    freqs: &Tensor,
    t: &Tensor,
    heads: u64,
    query_rows: u64,
    streamed: bool,
) -> Result<(Tensor, Vec<Tensor>)> {
    let mut x = input.clone();
    let mut hints = Vec::new();
    for i in 0..count {
        let key = format!("{prefix}.{i}");
        let mut local;
        let c = if streamed {
            local = catalog.fresh_stream();
            &mut local
        } else {
            &mut *catalog
        };
        // These loads/quantizers are evaluated in THIS block's quantum on every streamed visit.
        if i == 0 {
            x = c.linear(&mut p.graph, &format!("{key}.before_proj"), &x, true)?;
            x = p
                .graph
                .binary(&x, base, "control_transformer.rs before_proj + base")?;
        }
        x = block(&mut p.graph, c, &key, &x, freqs, Some(t), heads, query_rows)?;
        hints.push(c.linear(&mut p.graph, &format!("{key}.after_proj"), &x, true)?);
        p.mark(q::CONTROL_BLOCK)?;
    }
    Ok((x, hints))
}
fn forward(
    p: &mut Program,
    model: &Model,
    route: &Route,
    c: &mut Catalog,
    control: &mut Option<Catalog>,
    prep: &Prepared,
    input: &Tensor,
) -> Result<Tensor> {
    p.reset_phase_points(IosPhase::Denoise);
    // LATENTS/PREPARE/CONTROL_EMBED are outside the callback. Their point identities remain 0.
    let cfg = &model.facts.dit_cfg;
    let heads = dim(cfg.n_heads)?;
    let patch = dim(cfg.patch_size)?;
    let fp = dim(cfg.f_patch_size)?;
    let plan = mlx_gen::attention::AttentionPlan::budgeted(crate::pipeline::attention_budget_for(
        Some(route.execution),
    ));
    let query_rows = |tokens: u64| -> Result<u64> {
        let tokens =
            i32::try_from(tokens).map_err(|_| error("attention token shape exceeds native i32"))?;
        dim(plan
            .budget
            .query_block(1, cfg.n_heads, tokens, tokens)
            .min(tokens))
    };
    let t = timestep(&mut p.graph, c, cfg)?;
    let tokens = patchify(&mut p.graph, input, patch, fp, prep.img_pad)?;
    let x = c.linear(
        &mut p.graph,
        &format!("all_x_embedder.{patch}-{fp}"),
        &tokens,
        true,
    )?;
    let x = pad(&mut p.graph, c, &x, &prep.x_keep, "x_pad_token")?;
    let mut x = p.graph.reshape(
        &x,
        &[1, prep.n, dim(cfg.dim)?],
        "transformer.rs image embedding batch",
    )?;
    p.mark_at(
        q::DIT_EMBED,
        if input.dtype == Dtype::Bfloat16 && !route.mode.img2img() {
            1
        } else {
            0
        },
    )?;
    let (threaded, refiner_hints) = if let Some(ctrl) = control {
        let control_input = prep
            .control
            .as_ref()
            .ok_or_else(|| error("missing control prepared embedding"))?;
        let (threaded, hints) = control_stack(
            p,
            ctrl,
            "control_noise_refiner",
            crate::control_transformer::CONTROL_REFINER_PLACES.len(),
            control_input,
            &x,
            &prep.x_freqs,
            &t,
            heads,
            query_rows(prep.n)?,
            false,
        )?;
        (Some(threaded), hints)
    } else {
        (None, Vec::new())
    };
    for i in 0..cfg.n_refiner_layers {
        x = block(
            &mut p.graph,
            c,
            &format!("noise_refiner.{i}"),
            &x,
            &prep.x_freqs,
            Some(&t),
            heads,
            query_rows(prep.n)?,
        )?;
        if control.is_some() {
            if let Some(k) = crate::control_transformer::hint_index(
                &crate::control_transformer::CONTROL_REFINER_PLACES,
                i,
            ) {
                x = hint(&mut p.graph, &x, &refiner_hints[k])?;
            }
        }
        p.mark(q::DIT_BLOCK)?;
    }
    let cap = norm(&mut p.graph, c, "cap_embedder.0.weight", &prep.cap)?;
    let cap = c.linear(&mut p.graph, "cap_embedder.1", &cap, true)?;
    let cap = pad(&mut p.graph, c, &cap, &prep.cap_keep, "cap_pad_token")?;
    let mut cap = p.graph.reshape(
        &cap,
        &[1, prep.c, dim(cfg.dim)?],
        "transformer.rs caption batch",
    )?;
    p.mark(q::DIT_CAPTION)?;
    for i in 0..cfg.n_refiner_layers {
        cap = block(
            &mut p.graph,
            c,
            &format!("context_refiner.{i}"),
            &cap,
            &prep.cap_freqs,
            None,
            heads,
            query_rows(prep.c)?,
        )?;
        p.mark(q::DIT_CONTEXT_BLOCK)?;
    }
    let mut unified = p
        .graph
        .concat(&[x, cap.clone()], 1, "transformer.rs unify image+caption")?;
    p.mark(q::DIT_UNIFY)?;
    let main_hints = if let Some(ctrl) = control {
        let control_unified = p.graph.concat(
            &[
                threaded.ok_or_else(|| error("missing threaded control"))?,
                cap,
            ],
            1,
            "control_transformer.rs unify control+caption",
        )?;
        p.mark(q::DIT_UNIFY)?;
        control_stack(
            p,
            ctrl,
            "control_layers",
            crate::control_transformer::CONTROL_LAYERS_PLACES.len(),
            &control_unified,
            &unified,
            &prep.unified_freqs,
            &t,
            heads,
            query_rows(add(prep.n, prep.c)?)?,
            model.facts.streamable,
        )?
        .1
    } else {
        Vec::new()
    };
    for i in 0..cfg.n_layers {
        let mut local;
        let source = if model.facts.streamable {
            local = c.fresh_stream();
            &mut local
        } else {
            &mut *c
        };
        unified = block(
            &mut p.graph,
            source,
            &format!("layers.{i}"),
            &unified,
            &prep.unified_freqs,
            Some(&t),
            heads,
            query_rows(add(prep.n, prep.c)?)?,
        )?;
        if control.is_some() {
            if let Some(k) = crate::control_transformer::hint_index(
                &crate::control_transformer::CONTROL_LAYERS_PLACES,
                i,
            ) {
                unified = hint(&mut p.graph, &unified, &main_hints[k])?;
            }
        }
        p.mark(q::DIT_BLOCK)?;
    }
    let final_key = format!("all_final_layer.{patch}-{fp}");
    let silu = p.graph.silu(&t, "final_layer.rs timestep SiLU")?;
    let ada = c.linear(
        &mut p.graph,
        &format!("{final_key}.adaLN_modulation.0"),
        &silu,
        true,
    )?;
    let one = p.graph.rust_scalar(Dtype::Float32, "final_layer.rs one")?;
    let one = p.graph.cast(&one, ada.dtype, "final_layer.rs one cast")?;
    let scale = p.graph.binary(&ada, &one, "final_layer.rs scale")?;
    let scale = p.graph.reshape(
        &scale,
        &[1, 1, dim(cfg.dim)?],
        "final_layer.rs scale expand",
    )?;
    let normed = p
        .graph
        .layer_norm(&unified, "final_layer.rs affine-free LN")?;
    let normed = p
        .graph
        .binary(&normed, &scale, "final_layer.rs scale normed")?;
    let output = c.linear(&mut p.graph, &format!("{final_key}.linear"), &normed, true)?;
    let output = p.graph.reshape(
        &output,
        &[add(prep.n, prep.c)?, output.shape[2]],
        "transformer.rs output flatten",
    )?;
    p.graph
        .immediate(&[prep.n], Dtype::Int32, "transformer.rs head row indices")?;
    let output = p
        .graph
        .gather_rows(&output, prep.n, "transformer.rs image head gather")?;
    let (h, w) = (u64::from(route.height / 8), u64::from(route.width / 8));
    let n0 = mul(mul(1 / fp, h / patch)?, w / patch)?;
    p.graph
        .immediate(&[n0], Dtype::Int32, "transformer.rs unpatchify indices")?;
    let output = p
        .graph
        .gather_rows(&output, n0, "transformer.rs unpatchify gather")?;
    let output = p.graph.reshape(
        &output,
        &[
            1 / fp,
            h / patch,
            w / patch,
            fp,
            patch,
            patch,
            dim(cfg.in_channels)?,
        ],
        "transformer.rs unpatchify reshape",
    )?;
    let output = output.transpose(&[6, 0, 3, 1, 4, 2, 5])?;
    let output = p.graph.reshape(
        &output,
        &[dim(cfg.in_channels)?, 1, h, w],
        "transformer.rs unpatchify transpose copy",
    )?;
    let negative = p
        .graph
        .rust_scalar(Dtype::Float32, "transformer.rs output -1")?;
    let output = p
        .graph
        .binary(&output, &negative, "transformer.rs velocity negation")?;
    p.mark(q::DIT_OUTPUT)?;
    Ok(output)
}

fn te_condition(p: &mut Program, model: &Model, c: &mut Catalog, route: &Route) -> Result<Tensor> {
    let f = &model.facts;
    let cfg = f.te_cfg;
    let s = u64::from(f.window_tokens);
    // Tokenizer outputs are uploaded as two actual immediate int32 Arrays before the embed guard.
    p.graph
        .immediate(&[1, s], Dtype::Int32, "text_encoder input ids upload")?;
    p.graph
        .immediate(&[1, s], Dtype::Int32, "text_encoder attention mask upload")?;
    let mut x = c.embedding(&mut p.graph, s)?;
    let angles = p.graph.immediate(
        &[s, dim(cfg.head_dim)? / 2],
        Dtype::Float32,
        "nn.rs TextRope host frequency table",
    )?;
    let angles = p.graph.concat(
        &[angles.clone(), angles],
        1,
        "nn.rs TextRope duplicate frequencies",
    )?;
    let cos = p.graph.unary(&angles, "nn.rs TextRope cos")?;
    let sin = p.graph.unary(&angles, "nn.rs TextRope sin")?;
    let cos = p.graph.reshape(
        &cos,
        &[1, s, 1, dim(cfg.head_dim)?],
        "text_encoder cosine expand heads",
    )?;
    let sin = p.graph.reshape(
        &sin,
        &[1, s, 1, dim(cfg.head_dim)?],
        "text_encoder sine expand heads",
    )?;
    let mask = p.graph.immediate(
        &[1, 1, s, s],
        Dtype::Float32,
        "text_encoder/encoder.rs build_mask upload",
    )?;
    p.mark(q::TE_EMBED)?;
    let mut previous = x.clone();
    for index in 0..cfg.n_layers {
        let mut local;
        let source = if f.streamable {
            local = c.fresh_stream();
            &mut local
        } else {
            &mut *c
        };
        previous = x.clone();
        x = te_layer(&mut p.graph, source, index, &x, cfg, &cos, &sin, &mask)?;
        p.mark(q::TE_LAYER)?;
    }
    let valid = u64::from(f.caption_tokens.min(f.window_tokens));
    let flat = p.graph.reshape(
        &previous,
        &[s, dim(cfg.hidden_size)?],
        "pipeline.rs slice_valid flatten",
    )?;
    p.graph.immediate(
        &[valid],
        Dtype::Int32,
        "pipeline.rs slice_valid index upload",
    )?;
    let mut cap = p
        .graph
        .gather_rows(&flat, valid, "pipeline.rs slice_valid gather")?;
    p.mark(q::CAPTION_FEATURES)?;
    if !route.mode.img2img() {
        cap = p
            .graph
            .cast(&cap, Dtype::Bfloat16, "pipeline.rs caption bf16")?;
        p.mark(q::CAPTION_FEATURES)?;
    }
    Ok(cap)
}

fn block_keys(
    c: &Catalog,
    prefix: &str,
    modulated: bool,
    control: bool,
    before: bool,
) -> Vec<String> {
    let mut keys = Vec::new();
    for leaf in [
        "attention.to_q",
        "attention.to_k",
        "attention.to_v",
        "attention.to_out.0",
        "feed_forward.w1",
        "feed_forward.w2",
        "feed_forward.w3",
    ] {
        keys.extend(c.linear_keys(&format!("{prefix}.{leaf}"), false));
    }
    if modulated {
        keys.extend(c.linear_keys(&format!("{prefix}.adaLN_modulation.0"), true));
    }
    for leaf in [
        "attention_norm1.weight",
        "attention_norm2.weight",
        "ffn_norm1.weight",
        "ffn_norm2.weight",
        "attention.norm_q.weight",
        "attention.norm_k.weight",
    ] {
        let key = format!("{prefix}.{leaf}");
        if c.has(&key) {
            keys.push(key);
        }
    }
    if control {
        keys.extend(c.linear_keys(&format!("{prefix}.after_proj"), true));
        if before {
            keys.extend(c.linear_keys(&format!("{prefix}.before_proj"), true));
        }
    }
    keys
}
fn quantize_control(p: &mut Program, model: &Model, control: &mut Option<Catalog>) -> Result<()> {
    if model.facts.quant_bits.is_none() {
        return Ok(());
    }
    if let Some(c) = control {
        for (prefix, len, resident) in [
            (
                "control_layers",
                crate::control_transformer::CONTROL_LAYERS_PLACES.len(),
                !model.facts.streamable,
            ),
            (
                "control_noise_refiner",
                crate::control_transformer::CONTROL_REFINER_PLACES.len(),
                true,
            ),
        ] {
            if !resident {
                continue;
            }
            for i in 0..len {
                let base = format!("{prefix}.{i}");
                for (target,_) in crate::control_transformer_block::ZImageControlBlock::quantization_header_targets(&base,i==0) {
                    c.prepare_linear(&mut p.graph,&target,crate::quant::dit_quantization_has_bias(&target))?;
                }
                p.mark(q::CONTROL_QUANTIZE)?;
            }
        }
    }
    Ok(())
}
fn heavy_load(
    p: &mut Program,
    model: &Model,
    c: &mut Catalog,
    control: &mut Option<Catalog>,
    vae: &mut Catalog,
    owners: &VaeOwnerHistory,
    order: HeavyLoadOrder,
) -> Result<()> {
    let f = &model.facts;
    c.preload(
        &mut p.graph,
        model
            .dit_constructor_origins
            .keys()
            .filter(|name| crate::loader::dit_load_keeps(f.streamable, name))
            .cloned(),
    )?;
    for (&d, &len) in f.dit_cfg.axes_dims.iter().zip(&f.dit_cfg.axes_lens) {
        p.graph.immediate(
            &[dim(len)?, dim(d)? / 2, 2],
            Dtype::Float32,
            "rope_embedder.rs new constructor cache",
        )?;
    }
    p.mark(q::DIT_LOAD)?;
    if let Some(ctrl) = control {
        let key = format!(
            "control_all_x_embedder.{}-{}",
            f.dit_cfg.patch_size, f.dit_cfg.f_patch_size
        );
        let mut selected = ctrl.linear_keys(&key, true);
        for (prefix, len, loaded) in [
            (
                "control_layers",
                crate::control_transformer::CONTROL_LAYERS_PLACES.len(),
                !f.streamable,
            ),
            (
                "control_noise_refiner",
                crate::control_transformer::CONTROL_REFINER_PLACES.len(),
                true,
            ),
        ] {
            if loaded {
                for i in 0..len {
                    selected.extend(block_keys(
                        ctrl,
                        &format!("{prefix}.{i}"),
                        true,
                        true,
                        i == 0,
                    ));
                }
            }
        }
        ctrl.preload(&mut p.graph, selected)?;
        p.mark(q::CONTROL_LOAD)?;
    }
    if matches!(order, HeavyLoadOrder::QuantizeBeforeVae) {
        quantize_control(p, model, control)?;
    }
    let raw = owners
        .unchanged
        .keys()
        .cloned()
        .chain(
            owners
                .conversions
                .iter()
                .flat_map(|v| v.parents.keys().cloned()),
        )
        .chain(owners.remap.reads.iter().map(|read| read.read.raw.clone()))
        .collect::<BTreeSet<_>>();
    for key in raw {
        // These are RAW identities from the actual remap/read history, not lookup keys in the
        // final dictionary (which may overwrite the same spelling with another source handle).
        if vae.loaded.insert(key.clone()) {
            let bytes = vae
                .raw
                .get(&key)
                .ok_or_else(|| error(format!("VAE materialized raw identity missing {key}")))?
                .0;
            p.graph.load(bytes, &key)?;
        }
    }
    p.mark(q::VAE_LOAD)?;
    if matches!(order, HeavyLoadOrder::VaeBeforeQuantize) {
        quantize_control(p, model, control)?;
    }
    // Both the original fixture and production call vae.quantize(bits) right after the transformer
    // quantization/VAE load pair. Dense mid-block attention Linears are cast to BF16 and packed by
    // the native Quantize primitive; their outputs stay in EVERY later cumulative prefix.
    let targets = vae.conversions.keys().cloned().collect::<Vec<_>>();
    for base in targets {
        vae.prepare_linear(&mut p.graph, &base, true)?;
    }
    Ok(())
}

#[derive(Clone)]
struct SolverOps {
    program: std::rc::Rc<std::cell::RefCell<Program>>,
}
impl SolverOps {
    fn apply<T>(&self, f: impl FnOnce(&mut Graph) -> Result<T>) -> mlx_gen::gen_core::Result<T> {
        f(&mut self.program.borrow_mut().graph).map_err(mlx_gen::gen_core::Error::from)
    }
    fn multiply(&self, x: &Tensor) -> mlx_gen::gen_core::Result<Tensor> {
        self.apply(|g| {
            let s = g.rust_scalar(Dtype::Float32, "sampler.rs MlxLatentOps scalar")?;
            let s = g.cast(&s, x.dtype, "sampler.rs scalar dtype cast")?;
            g.binary(x, &s, "sampler.rs scale multiply")
        })
    }
}
impl mlx_gen::gen_core::sampling::LatentOps for SolverOps {
    type Latent = Tensor;
    fn scale(&self, x: &Tensor, scale: f32) -> mlx_gen::gen_core::Result<Tensor> {
        if scale == 1.0 {
            Ok(x.clone())
        } else {
            self.multiply(x)
        }
    }
    fn add(&self, a: &Tensor, b: &Tensor) -> mlx_gen::gen_core::Result<Tensor> {
        self.apply(|g| g.binary(a, b, "sampler.rs MlxLatentOps add"))
    }
    fn sub(&self, a: &Tensor, b: &Tensor) -> mlx_gen::gen_core::Result<Tensor> {
        self.apply(|g| g.binary(a, b, "sampler.rs MlxLatentOps sub"))
    }
    fn axpy(&self, a: f32, x: &Tensor, _b: f32, y: &Tensor) -> mlx_gen::gen_core::Result<Tensor> {
        use mlx_gen::gen_core::sampling::LatentOps;
        let by = self.multiply(y)?;
        if a == 1.0 {
            self.add(x, &by)
        } else {
            let ax = self.multiply(x)?;
            self.add(&ax, &by)
        }
    }
    fn randn_like(
        &self,
        x: &Tensor,
        _seed: u64,
        _step: usize,
    ) -> mlx_gen::gen_core::Result<Tensor> {
        self.apply(|g| g.normal_f32(&x.shape, "sampler.rs StepRng explicit keyed normal"))
    }
}

pub(super) fn request(
    p: &mut Program,
    model: &Model,
    route: &Route,
    sampling: &ResolvedFlowSampling,
    load_order: HeavyLoadOrder,
    _vae: &crate::vae::Vae,
    owners: &VaeOwnerHistory,
) -> Result<()> {
    let f = &model.facts;
    if route.count == 0
        || route.width == 0
        || route.height == 0
        || f.dit_cfg.patch_size <= 0
        || f.dit_cfg.f_patch_size != 1
        || f.dit_cfg.n_heads <= 0
    {
        return Err(error("invalid served geometry/configuration"));
    }
    if route.mode.control() != f.control.is_some() {
        return Err(error("loaded Control family does not match request roles"));
    }
    // These two providers are explicitly CFG-free; arbitrary authored guidance is ignored by the
    // REAL Turbo resolver. A resolved base-model plan must not be silently reinterpreted as Turbo.
    if sampling.effective_guidance().is_some() {
        return Err(error(
            "supplied plan is not the CFG-free Turbo provider's resolved execution plan",
        ));
    }
    if f.quant_conversions.iter().any(|c| !c.is_empty()) {
        // Production Facts refuse load-time TE/DiT requantization before any pricing exists.
        return Err(Error::Unsupported("Z certificate: load-time TE/DiT requantization is refused by the provider's pricing contract".into()));
    }
    let raw = f.load_components.as_ref();
    let mut te = Catalog::component(
        &f.te,
        raw.map_or(&f.te, |c| &c[0]),
        &f.quant_conversions[0],
        true,
        false,
    )?;
    let mut dit = Catalog::component(
        &f.dit,
        raw.map_or(&f.dit, |c| &c[1]),
        &f.quant_conversions[1],
        false,
        true,
    )?;
    let mut control = if let Some(raw) = &f.control {
        let (effective, conversions) = if let Some(bits) = f.quant_bits {
            raw.load_quantized_with_bias(
                &super::super::control_quantization_targets(),
                bits,
                crate::quant::dit_quantization_has_bias,
            )?
        } else {
            (raw.clone(), Vec::new())
        };
        Some(Catalog::component(
            &effective,
            raw,
            &conversions,
            false,
            false,
        )?)
    } else {
        None
    };
    let mut vae = Catalog::from_vae(f, owners)?;
    te.preload(&mut p.graph, model.te_constructor_keys.iter().cloned())?;
    p.mark(q::TE_LOAD)?;
    if !route.execution.stage_residency {
        heavy_load(
            p,
            model,
            &mut dit,
            &mut control,
            &mut vae,
            owners,
            load_order,
        )?;
    }
    let caption = te_condition(p, model, &mut te, route)?;
    if route.execution.stage_residency {
        heavy_load(
            p,
            model,
            &mut dit,
            &mut control,
            &mut vae,
            owners,
            load_order,
        )?;
    }
    let (h, w) = (u64::from(route.height), u64::from(route.width));
    let mut clean = None;
    for is_control in [true, false] {
        if (is_control && !route.mode.control()) || (!is_control && !route.mode.img2img()) {
            continue;
        }
        let upload = p.graph.immediate(
            &[1, h, w, 3],
            Dtype::Float32,
            "img2img.rs preprocessed NHWC upload",
        )?;
        let image = upload.transpose(&[0, 3, 1, 2])?;
        p.mark(q::INIT_IMAGE)?;
        let latent = super::vae_graph::encode(p, &mut vae, &image)?;
        let latent = p.graph.reshape(
            &latent,
            &[16, 1, h / 8, w / 8],
            "pipeline.rs encoded latent layout",
        )?;
        if is_control {
            let mask = p.graph.immediate(
                &[1, 1, h / 8, w / 8],
                Dtype::Float32,
                "pipeline.rs control zero mask upload",
            )?;
            let inpaint = p.graph.immediate(
                &[16, 1, h / 8, w / 8],
                Dtype::Float32,
                "pipeline.rs control zero inpaint upload",
            )?;
            p.graph.concat(
                &[latent, mask, inpaint],
                0,
                "pipeline.rs control context concatenate",
            )?;
            p.mark(q::CONTROL_CONTEXT)?;
        } else {
            clean = Some(latent);
        }
    }
    // Execute the ACTUAL generic solver with metadata descriptors. This preserves its scalar
    // branches, ordered primary/corrector/midpoint callbacks, histories, stochastic draws, and final
    // advancement for every Solver variant. No logical liveness or drop is used as a counter law.
    let program = std::rc::Rc::new(std::cell::RefCell::new(p.clone()));
    let ops = SolverOps {
        program: program.clone(),
    };
    let ms = mlx_gen::gen_core::sampling::FlowModelSampling::new(sampling.convention());
    let mut latents = Vec::new();
    for _output in 0..route.count {
        let (initial, prep) = {
            let mut p = program.borrow_mut();
            p.reset_phase_points(IosPhase::Denoise);
            let noise = p.graph.normal_f32(
                &[16, 1, h / 8, w / 8],
                "pipeline.rs create_noise explicit keyed normal",
            )?;
            let noise = p
                .graph
                .cast(&noise, Dtype::Bfloat16, "pipeline.rs initial noise BF16")?;
            let initial = if let Some(clean) = &clean {
                let one_minus = p
                    .graph
                    .rust_scalar(Dtype::Float32, "img2img.rs interpolation one-minus-sigma")?;
                let sigma = p
                    .graph
                    .rust_scalar(Dtype::Float32, "img2img.rs interpolation sigma")?;
                let a =
                    p.graph
                        .binary(clean, &one_minus, "img2img.rs interpolation clean product")?;
                let b = p
                    .graph
                    .binary(&noise, &sigma, "img2img.rs interpolation noise product")?;
                p.graph.binary(&a, &b, "img2img.rs interpolation sum")?
            } else {
                noise
            };
            p.mark_at(q::LATENTS, 0)?;
            let prep = prepare(&mut p, model, route, &mut dit, &mut control, &caption)?;
            (initial, prep)
        };
        let mut callback = |x: &Tensor, sigma: f32| -> mlx_gen::gen_core::Result<Tensor> {
            let out = mlx_gen::gen_core::sampling::denoise(&ops, &ms, x, sigma, |input, _time| {
                forward(
                    &mut program.borrow_mut(),
                    model,
                    route,
                    &mut dit,
                    &mut control,
                    &prep,
                    input,
                )
                .map_err(mlx_gen::gen_core::Error::from)
            })?;
            program
                .borrow_mut()
                .mark_at(mlx_gen::sampler::SAMPLER_STEP_QUANTUM, 0)
                .map_err(mlx_gen::gen_core::Error::from)?;
            Ok(out)
        };
        let final_latent = sampling
            .solver()
            .boxed::<SolverOps>()
            .sample(&ops, &ms, &mut callback, initial, sampling.sigmas(), 0)
            .map_err(Error::from)?;
        program
            .borrow_mut()
            .mark_at(mlx_gen::sampler::SAMPLER_STEP_QUANTUM, 0)?;
        latents.push(final_latent);
    }
    *p = program.borrow().clone();
    let tiling = crate::pipeline::decode_tiling_for(
        Some(route.execution),
        route.width,
        route.height,
        route.execution.stage_residency,
    );
    for latent in latents {
        p.reset_phase_points(IosPhase::Decode);
        let latent = p.graph.reshape(
            &latent,
            &[1, 16, h / 8, w / 8],
            "pipeline.rs unpack_latents batch/frame views",
        )?;
        let decoded = super::vae_graph::decode(p, &mut vae, &latent, tiling.as_ref())?;
        super::vae_graph::output_image(p, &decoded)?;
    }
    Ok(())
}
