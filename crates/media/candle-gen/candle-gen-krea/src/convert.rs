//! Krea 2 transformer **architecture validation** — prove the on-disk tensor set exactly matches the
//! architecture implied by [`Krea2Config`] before the DiT forward trusts it. Port of `mlx-gen-krea`'s
//! `convert.rs` validation half (the Q4/Q8 turnkey assembly is the worker-wiring story sc-7581).
//!
//! The published `krea/Krea-2-Turbo` diffusers checkpoint uses dotted keys that map 1:1 onto the
//! `Krea2Transformer2DModel` module tree, so [`crate::loader::Weights::from_dir`] loads them directly
//! — there is no key remap. [`validate_transformer`] catches a wrong variant / truncated download /
//! config-weight mismatch loudly at load instead of as garbage latents.

use std::collections::BTreeSet;

use candle_gen::candle_core::Result;

use crate::config::Krea2Config;
use crate::loader::Weights;

/// GQA / full attention Linear weights: `to_q/to_k/to_v/to_gate/to_out.0` + per-head `norm_q/norm_k`.
fn attn_keys(prefix: &str) -> Vec<String> {
    [
        "norm_q", "norm_k", "to_q", "to_k", "to_v", "to_gate", "to_out.0",
    ]
    .iter()
    .map(|p| format!("{prefix}.{p}.weight"))
    .collect()
}

/// SwiGLU feed-forward (`gate`/`up` in, `down` out), all bias-free.
fn ff_keys(prefix: &str) -> Vec<String> {
    ["gate", "up", "down"]
        .iter()
        .map(|p| format!("{prefix}.{p}.weight"))
        .collect()
}

/// A text-fusion block (`layerwise_blocks` / `refiner_blocks`): RMSNorm-attn(+gate)-RMSNorm-SwiGLU,
/// no per-block modulation table.
fn text_block_keys(prefix: &str) -> Vec<String> {
    let mut k = attn_keys(&format!("{prefix}.attn"));
    k.extend(ff_keys(&format!("{prefix}.ff")));
    k.push(format!("{prefix}.norm1.weight"));
    k.push(format!("{prefix}.norm2.weight"));
    k
}

/// A single-stream `transformer_block`: a text-fusion-shaped block plus the per-block 6-factor
/// `scale_shift_table` (the `DoubleSharedModulation` offset added to the shared `time_mod_proj`).
fn single_block_keys(prefix: &str) -> Vec<String> {
    let mut k = text_block_keys(prefix);
    k.push(format!("{prefix}.scale_shift_table"));
    k
}

/// The complete set of transformer tensor keys implied by `cfg` (= the published 430 for Turbo/Raw).
pub fn expected_transformer_keys(cfg: &Krea2Config) -> Vec<String> {
    let mut keys = Vec::new();

    // Image patch embed.
    keys.push("img_in.weight".into());
    keys.push("img_in.bias".into());

    // Text input projection: RMSNorm(text) → Linear(text→hidden) → Linear(hidden→hidden).
    keys.push("txt_in.norm.weight".into());
    for n in ["linear_1", "linear_2"] {
        keys.push(format!("txt_in.{n}.weight"));
        keys.push(format!("txt_in.{n}.bias"));
    }

    // Timestep embed + the shared 6-factor modulation projection.
    for n in ["linear_1", "linear_2"] {
        keys.push(format!("time_embed.{n}.weight"));
        keys.push(format!("time_embed.{n}.bias"));
    }
    keys.push("time_mod_proj.weight".into());
    keys.push("time_mod_proj.bias".into());

    // text_fusion: layerwise (cross-layer-axis aggregator) → projector(12→1) → refiner (token-axis).
    for i in 0..cfg.num_layerwise_text_blocks {
        keys.extend(text_block_keys(&format!(
            "text_fusion.layerwise_blocks.{i}"
        )));
    }
    keys.push("text_fusion.projector.weight".into());
    for i in 0..cfg.num_refiner_text_blocks {
        keys.extend(text_block_keys(&format!("text_fusion.refiner_blocks.{i}")));
    }

    // The single-stream stack.
    for i in 0..cfg.num_layers {
        keys.extend(single_block_keys(&format!("transformer_blocks.{i}")));
    }

    // Continuous-AdaLN output (2-factor scale/shift table).
    keys.push("final_layer.linear.weight".into());
    keys.push("final_layer.linear.bias".into());
    keys.push("final_layer.norm.weight".into());
    keys.push("final_layer.scale_shift_table".into());

    keys
}

/// Validate a loaded transformer against `cfg`: exact key coverage (no missing, no extra) and the
/// shapes of the dimension-bearing entry points.
///
/// **Native-mmdit-keyed (sc-9300 ConvRot / sc-14022 dense):** a native-keyed checkpoint stores the
/// reference names, so the exact diffusers key-set diff would spuriously report every key missing +
/// every native key extra. Instead validate that each expected diffusers key **resolves** to a present
/// native tensor (via the loader's diffusers→native remap, which `w.contains` applies), then run the
/// same shape checks (which also resolve). This proves the file covers the full
/// `Krea2Transformer2DModel` surface without asserting a 1:1 native key match. (The dense single-file
/// path additionally rejects unmapped/foreign on-disk tensors — see
/// [`validate_native_transformer`].)
pub fn validate_transformer(w: &Weights, cfg: &Krea2Config) -> Result<()> {
    if w.uses_native_keys() {
        let missing: Vec<String> = expected_transformer_keys(cfg)
            .into_iter()
            .filter(|k| !w.contains(k))
            .collect();
        if !missing.is_empty() {
            let head = missing
                .iter()
                .take(8)
                .map(String::as_str)
                .collect::<Vec<_>>()
                .join(", ");
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "krea native-keyed DiT: {} expected key(s) do not resolve to a native tensor [{head}]",
                missing.len(),
            )));
        }
        return validate_shapes(w, cfg);
    }
    let expected: BTreeSet<String> = expected_transformer_keys(cfg).into_iter().collect();
    // A pre-quantized snapshot replaces each Linear `{base}.weight` with the packed triple
    // `{base}.weight` + `{base}.scales` + `{base}.biases`; drop the two quant-only artifacts before
    // the coverage diff (the `{base}.weight` key is still present).
    let actual: BTreeSet<String> = w
        .keys()
        .into_iter()
        .filter(|k| !k.ends_with(".scales") && !k.ends_with(".biases"))
        .collect();

    let missing: Vec<&String> = expected.difference(&actual).collect();
    let extra: Vec<&String> = actual.difference(&expected).collect();
    if !missing.is_empty() || !extra.is_empty() {
        let head = |v: &[&String]| {
            v.iter()
                .take(8)
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        };
        return Err(candle_gen::candle_core::Error::Msg(format!(
            "krea transformer key mismatch vs config: {} missing [{}], {} extra [{}]",
            missing.len(),
            head(&missing),
            extra.len(),
            head(&extra),
        )));
    }

    validate_shapes(w, cfg)
}

/// Validate a **dense-bf16 or plain-int8 native-keyed single file** (sc-14022/sc-14023) — the candle
/// sibling of the MLX `remap_native_dit_to_diffusers` + `validate_transformer` fail-closed pair.
/// Stricter than the ConvRot branch of [`validate_transformer`]: every model tensor must form a
/// bijection onto the module tree. A descriptor-validated plain-int8 projection additionally carries
/// exactly its `.weight_scale` and `.comfy_quant` companions; those are the only allowed surplus.
///
/// Fails closed (typed error) — never a silent gap:
/// 1. **coverage** — any expected diffusers key that does not resolve to a present native tensor (a
///    truncated download, wrong family, or a bad prefix) is reported by name; then
/// 2. **no foreign/extra** — the on-disk tensor count must equal the expected model count plus two
///    companions per validated plain-int8 projection. With the remap pinned injective
///    (`loader::tests::convrot_remap_pins_the_key_map` + the sc-14022 coverage test) and full coverage
///    from step 1, count-equality proves there is no foreign or duplicate tensor; then
/// 3. the shared `validate_shapes` (resolving, flat-`scale_shift_table` aware).
///
/// The ConvRot file is validated by [`validate_transformer`]'s coverage-only native branch instead — it
/// legitimately carries extra `.weight_scale`/`.comfy_quant` siblings, so it is NOT a bijection.
pub fn validate_native_transformer(w: &Weights, cfg: &Krea2Config) -> Result<()> {
    validate_native_key_surface(w, cfg)?;
    validate_shapes(w, cfg)
}

/// Validate coverage and the exact allowed on-disk key surface independently of tensor dimensions.
/// Kept separate so a tiny, real safetensors fixture can prove the plain-int8 companion accounting
/// without allocating the multi-gigabyte tensors required by Krea's representative shape checks.
fn validate_native_key_surface(w: &Weights, cfg: &Krea2Config) -> Result<()> {
    let expected = expected_transformer_keys(cfg);
    // (1) Coverage: every expected diffusers key must resolve (via the native remap) to a present tensor.
    let missing: Vec<&String> = expected.iter().filter(|k| !w.contains(k)).collect();
    if !missing.is_empty() {
        let head = missing
            .iter()
            .take(8)
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        return Err(candle_gen::candle_core::Error::Msg(format!(
            "krea native single-file: {} expected DiT key(s) do not resolve to a native tensor \
             (truncated download, wrong family, or unexpected namespace) [{head}]",
            missing.len(),
        )));
    }
    // (2) No foreign/extra: dense has exactly one on-disk tensor per expected model tensor. Plain int8
    // has exactly two already-validated companions for each I8 projection. `from_native_file` proves
    // descriptor contents, I8 rank, scale dtype/shape, and the 1:1 descriptor-to-I8 relationship before
    // constructing `Weights`, so descriptor count is the authoritative quantized projection count here.
    let keys = w.keys();
    let plain_int8_projections = if w.is_plain_int8() {
        keys.iter()
            .filter(|key| key.ends_with(".comfy_quant"))
            .count()
    } else {
        0
    };
    let expected_on_disk = expected.len() + plain_int8_projections * 2;
    let on_disk = keys.len();
    if on_disk != expected_on_disk {
        let quant_note = if w.is_plain_int8() {
            format!(
                " plus two companions for each of {plain_int8_projections} validated plain-int8 projection(s)"
            )
        } else {
            String::new()
        };
        return Err(candle_gen::candle_core::Error::Msg(format!(
            "krea native single-file: {on_disk} on-disk DiT tensor(s), expected exactly {} model \
             tensor(s){quant_note} — the file carries an unmapped/foreign or duplicate tensor",
            expected.len()
        )));
    }
    Ok(())
}

/// Shape checks on the dimension-bearing entry points (Linear weight = `[out, in]`). Shared by the
/// dense/packed path and the native-keyed path (`check_shape` resolves the diffusers key to the native
/// key and skips a quantized weight whose on-disk shape differs — packed u32 codes or int8 codes).
fn validate_shapes(w: &Weights, cfg: &Krea2Config) -> Result<()> {
    let h = cfg.hidden_size;
    check_shape(w, "img_in.weight", &[h, cfg.in_channels])?;
    check_shape(w, "final_layer.linear.weight", &[cfg.in_channels, h])?;
    check_shape(w, "final_layer.scale_shift_table", &[2, h])?;
    check_shape(w, "txt_in.linear_1.weight", &[h, cfg.text_hidden_dim])?;
    check_shape(w, "txt_in.linear_2.weight", &[h, h])?;
    check_shape(
        w,
        "time_embed.linear_1.weight",
        &[h, cfg.timestep_embed_dim],
    )?;
    check_shape(w, "time_mod_proj.weight", &[cfg.time_mod_dim(), h])?;
    check_shape(w, "text_fusion.projector.weight", &[1, cfg.num_text_layers])?;
    // A representative text-fusion block (full attention, text width).
    let th = cfg.text_hidden_dim;
    check_shape(
        w,
        "text_fusion.layerwise_blocks.0.attn.to_q.weight",
        &[th, th],
    )?;
    check_shape(
        w,
        "text_fusion.layerwise_blocks.0.ff.gate.weight",
        &[cfg.text_intermediate_size, th],
    )?;
    // A representative single-stream block: GQA + the SwiGLU FFN + the 6-factor modulation table.
    check_shape(
        w,
        "transformer_blocks.0.attn.to_q.weight",
        &[cfg.q_dim(), h],
    )?;
    check_shape(
        w,
        "transformer_blocks.0.attn.to_k.weight",
        &[cfg.kv_dim(), h],
    )?;
    check_shape(
        w,
        "transformer_blocks.0.attn.to_gate.weight",
        &[cfg.q_dim(), h],
    )?;
    check_shape(
        w,
        "transformer_blocks.0.ff.gate.weight",
        &[cfg.intermediate_size, h],
    )?;
    check_shape(
        w,
        "transformer_blocks.0.scale_shift_table",
        &[Krea2Config::MOD_FACTORS, h],
    )?;
    Ok(())
}

fn check_shape(w: &Weights, key: &str, expected: &[usize]) -> Result<()> {
    // A packed (quantized) `{base}.weight` is u32-codes with a different on-disk shape; skip the
    // dense-shape check when a sibling `{base}.scales` marks it pre-quantized.
    if let Some(base) = key.strip_suffix(".weight") {
        if w.contains(&format!("{base}.scales")) {
            return Ok(());
        }
    }
    // A native-keyed file's per-block `scale_shift_table` is stored 1-D (`mod.lin` `[6·h]`) rather than
    // `[6, h]` (both the ConvRot export and the dense single file); the DiT reshapes it identically
    // row-major, so the flat form is correct. `w.shape` resolves to the native key, so compare against
    // the flattened `[6·h]` for a native-keyed checkpoint.
    if w.uses_native_keys() && key.ends_with(".scale_shift_table") {
        if let Some(shape) = w.shape(key) {
            let flat: usize = expected.iter().product();
            if shape == expected || shape == [flat] {
                return Ok(());
            }
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "krea native-keyed DiT: {key} shape {shape:?}, expected {expected:?} or [{flat}]"
            )));
        }
    }
    match w.shape(key) {
        Some(shape) if shape == expected => Ok(()),
        Some(shape) => Err(candle_gen::candle_core::Error::Msg(format!(
            "krea: {key} shape {shape:?}, expected {expected:?}"
        ))),
        None => Err(candle_gen::candle_core::Error::Msg(format!(
            "krea: missing tensor {key}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::{safetensors, DType, Device, Tensor};
    use serde_json::{json, Map, Value};
    use std::collections::HashMap;
    use std::path::Path;

    use crate::loader::{convrot_diffusers_to_native, Weights};

    #[test]
    fn expected_key_count_matches_published_turbo() {
        let cfg = Krea2Config::turbo();
        let keys = expected_transformer_keys(&cfg);
        let unique: BTreeSet<_> = keys.iter().collect();
        assert_eq!(keys.len(), unique.len(), "no duplicate expected keys");
        // 17 top-level + 49 text_fusion (2×12 layerwise + 1 projector + 2×12 refiner) + 364 blocks
        // (28×13) = 430, matching the published safetensors index exactly.
        assert_eq!(keys.len(), 430);
    }

    /// The real native-mmdit key set captured from `kreamania_variant5.safetensors` (430 tensors,
    /// prefixed `model.diffusion_model.`) — the committed fixture; the 26 GB weights file is NOT committed.
    fn variant5_native_keys() -> Vec<String> {
        let raw = include_str!("../tests/fixtures/variant5_native_keys.txt");
        raw.lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(str::to_string)
            .collect()
    }

    /// Write a native-mmdit-keyed single file of 1-element f32 stubs (key coverage/count is what the
    /// validation reads; values/shapes are irrelevant until the shape check, which these tests reach only
    /// on the happy path they deliberately avoid). Mirrors the MLX `stub()` fixture approach.
    fn write_native_stub_file(path: &Path, keys: &[String]) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let dev = Device::Cpu;
        let mut map: HashMap<String, Tensor> = HashMap::new();
        for k in keys {
            map.insert(k.clone(), Tensor::from_vec(vec![0f32], (1,), &dev).unwrap());
        }
        safetensors::save(&map, path).unwrap();
    }

    /// Write a tiny but structurally real native safetensors file with one I8 projection and its
    /// sc-14023 companions. The remaining model tensors are f32 scalar stubs: key-surface validation
    /// reads their actual headers while deliberately avoiding Krea-sized allocations.
    fn write_plain_int8_native_stub_file(path: &Path, keys: &[String], foreign: bool) {
        const I8_WEIGHT: &str = "model.diffusion_model.blocks.0.attn.wq.weight";
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();

        let mut entries: Vec<(String, &'static str, Vec<usize>, Vec<u8>)> = keys
            .iter()
            .map(|key| {
                if key == I8_WEIGHT {
                    (key.clone(), "I8", vec![2, 3], vec![1, 254, 3, 252, 5, 250])
                } else {
                    (key.clone(), "F32", vec![1], 0f32.to_le_bytes().to_vec())
                }
            })
            .collect();
        let base = I8_WEIGHT.strip_suffix(".weight").unwrap();
        entries.push((
            format!("{base}.weight_scale"),
            "F32",
            vec![2],
            [0.5f32.to_le_bytes(), 2.0f32.to_le_bytes()].concat(),
        ));
        let descriptor = br#"{"format":"int8_tensorwise","per_row":true}"#.to_vec();
        entries.push((
            format!("{base}.comfy_quant"),
            "U8",
            vec![descriptor.len()],
            descriptor,
        ));
        if foreign {
            entries.push((
                "model.diffusion_model.blocks.0.attn.bogus".to_string(),
                "F32",
                vec![1],
                0f32.to_le_bytes().to_vec(),
            ));
        }
        entries.sort_by(|a, b| a.0.cmp(&b.0));

        let mut header = Map::new();
        let mut payload = Vec::new();
        for (name, dtype, shape, data) in entries {
            let start = payload.len();
            payload.extend_from_slice(&data);
            header.insert(
                name,
                json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [start, payload.len()]
                }),
            );
        }
        let mut header_bytes = serde_json::to_vec(&Value::Object(header)).unwrap();
        while !header_bytes.len().is_multiple_of(8) {
            header_bytes.push(b' ');
        }
        let mut file = (header_bytes.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(&header_bytes);
        file.extend_from_slice(&payload);
        std::fs::write(path, file).unwrap();
    }

    /// **The candle diffusers→native remap covers EXACTLY the real variant5 native key set (sc-14022).**
    /// Every one of the 430 expected diffusers keys maps (no `None`), and the set of native keys it
    /// produces equals the variant5 header's native keys (prefix stripped) — a bijection over the real
    /// dense single-file header, the candle mirror of mlx-gen-krea's
    /// `remap_covers_every_variant5_key_and_matches_expected_module_keys`.
    #[test]
    fn native_remap_covers_every_variant5_key() {
        let mapped: BTreeSet<String> = expected_transformer_keys(&Krea2Config::turbo())
            .iter()
            .map(|k| {
                convrot_diffusers_to_native(k)
                    .unwrap_or_else(|| panic!("expected diffusers key has no native mapping: {k}"))
            })
            .collect();

        let fixture: BTreeSet<String> = variant5_native_keys()
            .iter()
            .map(|k| {
                k.strip_prefix("model.diffusion_model.")
                    .unwrap_or_else(|| panic!("fixture key lacks the DiT prefix: {k}"))
                    .to_string()
            })
            .collect();

        let missing: Vec<&String> = fixture.difference(&mapped).collect();
        let extra: Vec<&String> = mapped.difference(&fixture).collect();
        assert!(
            missing.is_empty() && extra.is_empty(),
            "candle remap ≠ variant5 native keys: uncovered {missing:?}, stray {extra:?}"
        );
    }

    /// **Fail-closed on a missing required key (sc-14022).** A native single file missing one DiT tensor
    /// (`first.weight` → `img_in.weight`) is rejected by `validate_native_transformer`'s coverage half,
    /// naming the key that does not resolve — never a silent gap.
    #[test]
    fn validate_native_missing_key_fails_closed() -> Result<()> {
        let dev = Device::Cpu;
        let keys: Vec<String> = variant5_native_keys()
            .into_iter()
            .filter(|k| k != "model.diffusion_model.first.weight")
            .collect();
        let path = std::env::temp_dir()
            .join(format!("sc14022_missing_{}", std::process::id()))
            .join("variant5.safetensors");
        write_native_stub_file(&path, &keys);

        let w = Weights::from_native_file(&path, &dev, DType::F32)?;
        let err = crate::convert::validate_native_transformer(&w, &Krea2Config::turbo())
            .expect_err("a missing DiT tensor must fail closed")
            .to_string();
        assert!(
            err.contains("do not resolve") && err.contains("img_in.weight"),
            "error must name the unresolved key: {err}"
        );
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
        Ok(())
    }

    /// **Fail-closed on an unmapped/foreign key (sc-14022).** A native single file carrying the full 430
    /// plus one foreign tensor is rejected: coverage passes, but the on-disk count exceeds the expected
    /// count, so the bijection check rejects the surplus (the candle analogue of the MLX remap's
    /// `unmapped_key_fails_closed`). No stray weight loads silently.
    #[test]
    fn validate_native_foreign_key_fails_closed() -> Result<()> {
        let dev = Device::Cpu;
        let mut keys = variant5_native_keys();
        keys.push("model.diffusion_model.blocks.0.attn.bogus".to_string());
        let path = std::env::temp_dir()
            .join(format!("sc14022_foreign_{}", std::process::id()))
            .join("variant5.safetensors");
        write_native_stub_file(&path, &keys);

        let w = Weights::from_native_file(&path, &dev, DType::F32)?;
        let err = crate::convert::validate_native_transformer(&w, &Krea2Config::turbo())
            .expect_err("an unmapped/foreign tensor must fail closed")
            .to_string();
        assert!(
            err.contains("unmapped/foreign") || err.contains("bijection"),
            "error must flag the foreign tensor: {err}"
        );
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
        Ok(())
    }

    /// The native validator admits the exact two descriptor-validated companions of a plain-I8 model
    /// tensor, while retaining the same fail-closed behavior for every other surplus key.
    #[test]
    fn validate_native_plain_int8_companions_are_exactly_accounted() -> Result<()> {
        let dev = Device::Cpu;
        let keys = variant5_native_keys();
        let root = std::env::temp_dir().join(format!(
            "sc14023_plain_int8_native_surface_{}",
            std::process::id()
        ));
        let valid_path = root.join("variant4.safetensors");
        write_plain_int8_native_stub_file(&valid_path, &keys, false);

        let valid = Weights::from_native_file(&valid_path, &dev, DType::F32)?;
        assert!(valid.is_plain_int8());
        validate_native_key_surface(&valid, &Krea2Config::turbo())?;

        let foreign_path = root.join("variant4_foreign.safetensors");
        write_plain_int8_native_stub_file(&foreign_path, &keys, true);
        let foreign = Weights::from_native_file(&foreign_path, &dev, DType::F32)?;
        let err = validate_native_key_surface(&foreign, &Krea2Config::turbo())
            .expect_err("a non-companion surplus tensor must still fail closed")
            .to_string();
        assert!(
            err.contains("unmapped/foreign"),
            "error must flag the foreign tensor: {err}"
        );

        std::fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    #[ignore = "requires a local KREAMANIA_VARIANT4 checkpoint"]
    fn validate_real_variant4_checkpoint() -> Result<()> {
        let path = std::env::var("KREAMANIA_VARIANT4")
            .expect("set KREAMANIA_VARIANT4 to kreamania_variant4.safetensors");
        let weights = Weights::from_native_file(Path::new(&path), &Device::Cpu, DType::F32)?;
        assert!(weights.is_plain_int8());
        validate_native_transformer(&weights, &Krea2Config::turbo())
    }
}
