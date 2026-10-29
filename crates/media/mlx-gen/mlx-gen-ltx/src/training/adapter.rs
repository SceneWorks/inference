//! The LTX-local LoRA adapter seam both trainers share: target resolution, install, block
//! grouping and save.

use super::*;

/// `to_out.0` → `to_out`, the only diffusers→checkpoint rename in the attention LoRA surface (the
/// inference loader does the same in `adapters::normalize`); other suffixes pass through.
pub(super) fn resolve_segments(save_path: &str) -> Vec<String> {
    crate::adapters::normalize_ltx_key(save_path)
        .split('.')
        .map(String::from)
        .collect()
}

pub(super) fn build_targets_from_paths(
    dit: &mut impl LtxAdaptable,
    paths: impl IntoIterator<Item = String>,
    rank: i32,
    seed: u64,
) -> Result<(Vec<LtxLoraTarget>, LoraParams)> {
    let mut targets = Vec::new();
    let mut params = LoraParams::new();
    let small = Array::from_slice(&[0.02f32], &[1]);
    for (idx, save_path) in paths.into_iter().enumerate() {
        let segs = resolve_segments(&save_path);
        let seg_refs: Vec<&str> = segs.iter().map(String::as_str).collect();
        let Some(lin) = dit.adaptable_mut(&seg_refs) else {
            continue;
        };
        let shape = lin.base_shape();
        let (out_f, in_f) = (shape[0], shape[1]);
        let a_key: Rc<str> = Rc::from(format!("{save_path}.lora_a"));
        let b_key: Rc<str> = Rc::from(format!("{save_path}.lora_b"));
        let ka = random::key(seed.wrapping_add(2 * idx as u64 + 1))?;
        let a = multiply(
            &random::normal::<f32>(&[rank, in_f], None, None, Some(&ka))?,
            &small,
        )?;
        let b = Array::zeros::<f32>(&[out_f, rank])?;
        eval([&a, &b])?;
        params.insert(a_key.clone(), a);
        params.insert(b_key.clone(), b);
        targets.push(LtxLoraTarget {
            save_path,
            segs,
            a_key,
            b_key,
        });
    }
    Ok((targets, params))
}

/// Inject the current trainable factors as one LoRA residual per target via the LTX training seam —
/// transpose `[r,in]`→`[in,r]` and `[out,r]`→`[r,out]`, fold `alpha/rank` into `b` — so the residual
/// is `(x·Aᵀ·Bᵀ)·(alpha/rank)`, matching the reference `_MlxLoRALinear`. Differentiable.
pub(super) fn install_train_lora(
    dit: &mut impl LtxAdaptable,
    params: &LoraParams,
    targets: &[LtxLoraTarget],
    alpha: f32,
    rank: f32,
    lora_dtype: Option<Dtype>,
) -> MlxResult<()> {
    for t in targets {
        let a = params[&t.a_key].t(); // [r,in] -> [in,r]
        let b = params[&t.b_key]
            .t()
            .multiply(Array::from_slice(&[alpha / rank], &[1]))?; // [out,r] -> [r,out] · (α/r)

        // sc-4942 — under the bf16 training cast the f32 factors must join the bf16 stream, or every
        // adapted Linear re-promotes its block to f32 (defeating the activation saving). No-op in f32.
        let (a, b) = match lora_dtype {
            Some(dt) => (a.as_dtype(dt)?, b.as_dtype(dt)?),
            None => (a, b),
        };
        let seg_refs: Vec<&str> = t.segs.iter().map(String::as_str).collect();
        let lin = dit
            .adaptable_mut(&seg_refs)
            .ok_or_else(|| Exception::custom(format!("LoRA target not found: {}", t.save_path)))?;
        lin.set_train_lora(a, b);
    }
    Ok(())
}

/// Group resolved targets by their owning block (sc-4942) — `block_targets[i]` lists block `i`'s
/// trainable LoRA targets as the block-local path (`segs` minus the `transformer_blocks.{i}` prefix)
/// plus the factor-map keys, for the gradient-checkpoint closure. Every target lives in a
/// `transformer_blocks.{i}.attn{1,2}` leaf, so the grouping is exhaustive.
pub(super) fn group_block_targets(
    targets: &[LtxLoraTarget],
    n_layers: usize,
) -> Vec<Vec<BlockLoraRef>> {
    let mut out: Vec<Vec<BlockLoraRef>> = (0..n_layers).map(|_| Vec::new()).collect();
    for t in targets {
        // segs = ["transformer_blocks", "{i}", attn, suffix...]; the block-local path is segs[2..].
        if t.segs.len() < 3 || t.segs[0] != "transformer_blocks" {
            continue;
        }
        let Ok(i) = t.segs[1].parse::<usize>() else {
            continue;
        };
        if i >= n_layers {
            continue;
        }
        out[i].push(BlockLoraRef {
            local: t.segs[2..].to_vec(),
            a_key: t.a_key.to_string(),
            b_key: t.b_key.to_string(),
        });
    }
    out
}

/// Write the trained LoRA as safetensors keyed by the LTX module paths — `{module}.lora_A.weight`
/// `[rank,in]`, `{module}.lora_B.weight` `[out,rank]`, scalar `{module}.alpha` (= `alpha`) — the
/// reference `_save_lora` format, reloadable by [`crate::apply_ltx_adapters`] (which folds
/// `scale = alpha/rank`). `networkType`/`rank`/`alpha` metadata mirrors the other family trainers.
pub(super) fn save_lora(
    params: &LoraParams,
    targets: &[LtxLoraTarget],
    alpha: f32,
    rank: u32,
    path: &Path,
) -> Result<()> {
    let alphas: Vec<(String, Array)> = targets
        .iter()
        .map(|t| {
            (
                format!("{}.alpha", t.save_path),
                Array::from_slice(&[alpha], &[1]),
            )
        })
        .collect();
    let mut entries: Vec<(String, &Array)> = Vec::with_capacity(targets.len() * 3);
    for t in targets {
        entries.push((format!("{}.lora_A.weight", t.save_path), &params[&t.a_key]));
        entries.push((format!("{}.lora_B.weight", t.save_path), &params[&t.b_key]));
    }
    for (k, v) in &alphas {
        entries.push((k.clone(), v));
    }
    let mut meta: HashMap<String, String> = HashMap::new();
    meta.insert("networkType".to_string(), "lora".to_string());
    meta.insert("rank".to_string(), rank.to_string());
    meta.insert("alpha".to_string(), alpha.to_string());
    // The split LTX-2.5 provider consumes this explicit file-wide contract and deliberately does
    // not infer a rank from a factor.  Retain the legacy keys above so LTX-2.3/third-party loaders
    // remain byte-for-byte compatible while a newly trained adapter can be selected by either route.
    meta.insert("lora_rank".to_string(), rank.to_string());
    meta.insert("lora_alpha".to_string(), alpha.to_string());
    Array::save_safetensors(entries, Some(&meta), path)?;
    Ok(())
}
