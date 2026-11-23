use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::candle_nn::{Module, VarBuilder, VarMap};
use candle_gen::testkit::tensor_max_abs_diff_f32;
use candle_gen::text_encoder::{ClipConfig, ClipTextEncoder, T5Config, T5Encoder};

/// **Vendored CLIP ≡ stock candle-transformers CLIP (dense, 1e-4).** Both models load from the SAME
/// random `VarMap` (identical `text_model.*` diffusers keys), so a byte-for-byte match of their
/// pooled `[B, 768]` outputs proves the vendored packed-detect CLIP is a faithful dense port — the
/// dense path is unchanged, and (since the packed forward = dequant-then-dense-matmul) a packed load
/// of the same weights would track it too. A tiny 2-layer config keeps it GPU-free and fast.
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn vendored_clip_matches_stock_dense() -> Result<()> {
    use candle_transformers::models::clip::text_model::{
        Activation, ClipTextConfig, ClipTextTransformer,
    };
    let dev = Device::Cpu;
    let cfg = ClipConfig {
        vocab_size: 64,
        embed_dim: 32,
        intermediate_size: 64,
        max_position_embeddings: 16,
        num_hidden_layers: 2,
        num_attention_heads: 4,
    };
    let stock_cfg = ClipTextConfig {
        vocab_size: cfg.vocab_size,
        embed_dim: cfg.embed_dim,
        projection_dim: cfg.embed_dim,
        activation: Activation::QuickGelu,
        intermediate_size: cfg.intermediate_size,
        max_position_embeddings: cfg.max_position_embeddings,
        pad_with: None,
        num_hidden_layers: cfg.num_hidden_layers,
        num_attention_heads: cfg.num_attention_heads,
    };

    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
    // Build the vendored model (forces the `text_model.*` keys to be created in the VarMap).
    let vendored = ClipTextEncoder::new(&cfg, vb.pp("text_model"))?;
    // Build the stock model from the SAME VarMap → identical weights.
    let vb2 = VarBuilder::from_varmap(&vm, DType::F32, &dev);
    let stock = ClipTextTransformer::new(vb2.pp("text_model"), &stock_cfg)?;

    let ids = Tensor::from_vec(vec![3u32, 8, 15, 2, 0, 0], (1, 6), &dev)?;
    let v = vendored.forward(&ids)?;
    let s = __model_proof_output!(stock.forward(&ids))?;
    assert_eq!(v.dims(), s.dims());
    let d = tensor_max_abs_diff_f32(&v, &s);
    assert!(d < 1e-4, "packed-detect CLIP vs stock max|Δ| = {d}");
    let native = ClipTextEncoder::new_bfl(&cfg, vb2.pp("text_model"))?;
    let native_output = native.forward(&ids)?;
    assert_eq!(native_output.dims(), s.dims());
    assert_eq!(tensor_max_abs_diff_f32(&native_output, &s), 0.0);
    #[cfg(feature = "model-inference-proof")]
    println!(
        "\nMODEL_INFERENCE_BODY_COMPLETED text_encoder_tests::vendored_clip_matches_stock_dense"
    );
    Ok(())
}

/// **Vendored T5 encoder ≡ stock candle-transformers T5 encoder (dense, 1e-4).** Same-VarMap parity
/// on the encoder path (`shared`, `encoder.block.*`, `encoder.final_layer_norm`, gated-GELU FF,
/// block-0 relative bias). A tiny 2-block config keeps it GPU-free.
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn vendored_t5_matches_stock_dense() -> Result<()> {
    use candle_transformers::models::t5::{
        ActivationWithOptionalGating, Config as StockT5Config, T5EncoderModel,
    };
    let dev = Device::Cpu;
    let cfg = T5Config {
        vocab_size: 64,
        d_model: 32,
        d_kv: 8,
        d_ff: 64,
        num_layers: 2,
        num_heads: 4,
        relative_attention_num_buckets: 8,
        relative_attention_max_distance: 128,
        layer_norm_epsilon: 1e-6,
    };
    let stock_cfg = StockT5Config {
        vocab_size: cfg.vocab_size,
        d_model: cfg.d_model,
        d_kv: cfg.d_kv,
        d_ff: cfg.d_ff,
        num_layers: cfg.num_layers,
        num_decoder_layers: None,
        num_heads: cfg.num_heads,
        relative_attention_num_buckets: cfg.relative_attention_num_buckets,
        relative_attention_max_distance: cfg.relative_attention_max_distance,
        dropout_rate: 0.0,
        layer_norm_epsilon: cfg.layer_norm_epsilon,
        initializer_factor: 1.0,
        feed_forward_proj: ActivationWithOptionalGating {
            gated: true,
            activation: candle_gen::candle_nn::Activation::NewGelu,
        },
        tie_word_embeddings: false,
        is_decoder: false,
        is_encoder_decoder: true,
        use_cache: false,
        pad_token_id: 0,
        eos_token_id: 1,
        decoder_start_token_id: Some(0),
    };

    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
    let vendored = T5Encoder::new(&cfg, vb)?;
    let vb2 = VarBuilder::from_varmap(&vm, DType::F32, &dev);
    let mut stock = T5EncoderModel::load(vb2, &stock_cfg)?;

    let ids = Tensor::from_vec(vec![3u32, 8, 15, 2, 7, 1], (1, 6), &dev)?;
    let v = vendored.forward(&ids, DType::F32)?;
    let s = __model_proof_output!(stock.forward(&ids))?;
    assert_eq!(v.dims(), s.dims());
    let d = tensor_max_abs_diff_f32(&v, &s);
    assert!(d < 1e-4, "packed-detect T5 vs stock max|Δ| = {d}");
    let native = T5Encoder::new_bfl(
        &cfg,
        VarBuilder::from_varmap(&vm, DType::F32, &dev),
        Some(candle_gen::candle_nn::Activation::NewGelu),
    )?;
    assert_eq!(
        tensor_max_abs_diff_f32(&native.forward(&ids, DType::F32)?, &s),
        0.0
    );

    // The BFL config can select a different gated activation or an ungated `wi`.
    // Use the encoder token-table fallback so the same inputs exercise both choices.
    for gated_activation in [Some(candle_gen::candle_nn::Activation::Silu), None] {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let native = T5Encoder::new_bfl(&cfg, vb.clone(), gated_activation)?;
        let mut stock_cfg = stock_cfg.clone();
        stock_cfg.feed_forward_proj = ActivationWithOptionalGating {
            gated: gated_activation.is_some(),
            activation: gated_activation.unwrap_or(candle_gen::candle_nn::Activation::Relu),
        };
        let mut stock = T5EncoderModel::load(vb, &stock_cfg)?;
        let actual = native.forward(&ids, DType::F32)?;
        let expected = __model_proof_output!(stock.forward(&ids))?;
        assert_eq!(actual.dims(), expected.dims());
        assert_eq!(tensor_max_abs_diff_f32(&actual, &expected), 0.0);
    }
    #[cfg(feature = "model-inference-proof")]
    println!(
        "\nMODEL_INFERENCE_BODY_COMPLETED text_encoder_tests::vendored_t5_matches_stock_dense"
    );
    Ok(())
}

/// The masked forward (added for Mochi's `_get_t5_prompt_embeds`) is **inert** with an all-zero
/// additive key-padding mask: `forward_masked(ids, dt, Some(zeros))` is byte-identical to the
/// unmasked `forward(ids, dt)`. This guards the FLUX dense path's byte-exactness — the added
/// `Some(mask)` branch must not perturb the computation when the mask is all-zero. (The mask's
/// *numeric* effect on padded keys needs non-trivial weights, so it is validated with real weights
/// in the Mochi `te_parity` gate, not here — this test's `VarMap` zero-inits every projection.)
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn masked_forward_zero_mask_is_byte_identical_to_unmasked() -> Result<()> {
    let dev = Device::Cpu;
    let cfg = T5Config {
        vocab_size: 64,
        d_model: 32,
        d_kv: 8,
        d_ff: 64,
        num_layers: 2,
        num_heads: 4,
        relative_attention_num_buckets: 8,
        relative_attention_max_distance: 128,
        layer_norm_epsilon: 1e-6,
    };
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
    let enc = T5Encoder::new(&cfg, vb)?;

    let ids = Tensor::from_vec(vec![3u32, 8, 15, 2, 7, 1], (1, 6), &dev)?;
    let unmasked = enc.forward(&ids, DType::F32)?;
    let zeros = Tensor::zeros((1, 1, 1, 6), DType::F32, &dev)?;
    let masked_zero = enc.forward_masked(&ids, DType::F32, Some(&zeros))?;
    assert_eq!(tensor_max_abs_diff_f32(&unmasked, &masked_zero), 0.0);
    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED text_encoder_tests::masked_forward_zero_mask_is_byte_identical_to_unmasked");
    Ok(())
}
