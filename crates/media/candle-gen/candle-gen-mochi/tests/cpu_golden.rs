//! CPU CI-green tokenizer id-exactness for Mochi 1 (A5, sc-11989) — the candle twin of the
//! non-ignored parts of `mlx-gen-mochi`'s `te_parity.rs`. Exercises the vendored tokenizer (the same
//! `t5_tokenizer.json` the MLX crate ships) with no model weights: pad-to-256, determinism, and the
//! EOS `</s>`(1)-then-pad(0) structure Mochi's `_get_t5_prompt_embeds` produces. The scheduler sigma
//! values, RoPE positions/geometry, and small-weight VAE-decoder + DiT-block shape/determinism CPU
//! gates live as in-module unit tests (`scheduler`, `rope`, `vae`, `transformer`).

use candle_gen::candle_core::Device;
use candle_gen_mochi::text_encoder::tokenize;
use candle_gen_mochi::{load_tokenizer, MAX_SEQUENCE_LENGTH};

/// The exact prompt the A1 dump harness blessed the golden with.
const PROMPT: &str = "A calico kitten batting a ball of red yarn across a sunlit wooden floor.";

#[test]
fn tokenizer_pads_to_max_len_and_is_deterministic() {
    let dev = Device::Cpu;
    let tok = load_tokenizer().unwrap();

    let (ids_a, mask_a) = tokenize(&tok, PROMPT, &dev).unwrap();
    let (ids_b, mask_b) = tokenize(&tok, PROMPT, &dev).unwrap();
    let a = ids_a.flatten_all().unwrap().to_vec1::<u32>().unwrap();
    let b = ids_b.flatten_all().unwrap().to_vec1::<u32>().unwrap();
    // Determinism.
    assert_eq!(a, b, "tokenization must be deterministic");
    assert_eq!(mask_a, mask_b);

    // Padded to max_length = 256, shape [1, 256].
    assert_eq!(a.len(), MAX_SEQUENCE_LENGTH);
    assert_eq!(ids_a.dims(), &[1, MAX_SEQUENCE_LENGTH]);
    assert_eq!(mask_a.len(), MAX_SEQUENCE_LENGTH);

    // Real tokens are a contiguous non-pad prefix ending in EOS `</s>`(1), then pad(0). The 0/1 mask
    // marks exactly that prefix.
    let first_pad = a.iter().position(|&id| id == 0).expect("some padding");
    assert!(first_pad >= 2, "content + EOS present");
    assert_eq!(a[first_pad - 1], 1, "content ends with EOS </s>=1");
    assert!(a[first_pad..].iter().all(|&id| id == 0), "tail is all pad");
    // mask01: 1 over the real prefix, 0 over the pad tail.
    assert!(mask_a[..first_pad].iter().all(|&m| m == 1));
    assert!(mask_a[first_pad..].iter().all(|&m| m == 0));
}

#[test]
fn empty_prompt_is_eos_then_pad() {
    let dev = Device::Cpu;
    let tok = load_tokenizer().unwrap();
    let (ids, mask) = tokenize(&tok, "", &dev).unwrap();
    let ids = ids.flatten_all().unwrap().to_vec1::<u32>().unwrap();
    assert_eq!(ids.len(), MAX_SEQUENCE_LENGTH);
    // T5 encodes "" (add_special_tokens) to just the EOS, then pads.
    assert_eq!(ids[0], 1, "empty prompt starts with EOS </s>=1");
    assert!(ids[1..].iter().all(|&id| id == 0), "then all pad");
    assert_eq!(mask[0], 1, "EOS is a real token");
    assert!(mask[1..].iter().all(|&m| m == 0));
}
