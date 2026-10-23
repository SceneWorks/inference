#![allow(clippy::too_many_arguments)] // Header-only source-shape builders.
use super::*;

struct Headers(Inventory);
impl Headers {
    fn new() -> Self {
        Self(Inventory::default())
    }
    fn put(&mut self, key: impl Into<String>, shape: &[u64], dtype: Dtype) {
        let elements = shape.iter().copied().product::<u64>();
        self.0 .0.insert(
            key.into(),
            Tensor {
                dtype,
                shape: shape.to_vec(),
                elements,
                bytes: elements
                    * if dtype == Dtype::Float16 || dtype == Dtype::Bfloat16 {
                        2
                    } else {
                        4
                    },
            },
        );
    }
    fn linear(&mut self, key: &str, input: u64, out: u64, bias: bool, dtype: Dtype) {
        self.put(format!("{key}.weight"), &[out, input], dtype);
        if bias {
            self.put(format!("{key}.bias"), &[out], dtype);
        }
    }
    fn conv(
        &mut self,
        key: &str,
        input: u64,
        out: u64,
        k: u64,
        groups: u64,
        bias: bool,
        dtype: Dtype,
    ) {
        self.put(format!("{key}.weight"), &[out, input / groups, k, k], dtype);
        if bias {
            self.put(format!("{key}.bias"), &[out], dtype);
        }
    }
    fn norm(&mut self, key: &str, c: u64, bias: bool, dtype: Dtype) {
        self.put(format!("{key}.weight"), &[c], dtype);
        if bias {
            self.put(format!("{key}.bias"), &[c], dtype);
        }
    }
    fn glu(&mut self, p: &str, c: u64, norm: bool, dtype: Dtype) {
        self.conv(&format!("{p}.conv_inverted"), c, 8 * c, 1, 1, true, dtype);
        self.conv(
            &format!("{p}.conv_depth"),
            8 * c,
            8 * c,
            3,
            8 * c,
            true,
            dtype,
        );
        self.conv(&format!("{p}.conv_point"), 4 * c, c, 1, 1, false, dtype);
        if norm {
            self.norm(&format!("{p}.norm"), c, true, dtype);
        }
    }
    fn vae_block(&mut self, p: &str, c: u64, kind: BlockType, cfg: &DcAeConfig) {
        let dt = Dtype::Float32;
        match kind {
            BlockType::Res => {
                self.conv(&format!("{p}.conv1"), c, c, 3, 1, true, dt);
                self.conv(&format!("{p}.conv2"), c, c, 3, 1, false, dt);
                self.norm(&format!("{p}.norm"), c, true, dt);
            }
            BlockType::EfficientVit => {
                for name in ["to_q", "to_k", "to_v"] {
                    self.linear(&format!("{p}.attn.{name}"), c, c, false, dt);
                }
                self.linear(
                    &format!("{p}.attn.to_out"),
                    c * (1 + cfg.qkv_multiscales.len() as u64),
                    c,
                    false,
                    dt,
                );
                self.norm(&format!("{p}.attn.norm_out"), c, true, dt);
                for (i, &k) in cfg.qkv_multiscales.iter().enumerate() {
                    self.conv(
                        &format!("{p}.attn.to_qkv_multiscale.{i}.proj_in"),
                        3 * c,
                        3 * c,
                        k as u64,
                        3 * c,
                        false,
                        dt,
                    );
                    self.conv(
                        &format!("{p}.attn.to_qkv_multiscale.{i}.proj_out"),
                        3 * c,
                        3 * c,
                        1,
                        3 * c / cfg.attention_head_dim as u64,
                        false,
                        dt,
                    );
                }
                self.glu(&format!("{p}.conv_out"), c, true, dt);
            }
        }
    }
}
pub(super) fn facts(sprint: bool) -> SanaPricingFacts {
    let mut dcae = DcAeConfig::sana_f32c32();
    dcae.block_out_channels = vec![32; 6];
    dcae.layers_per_block = vec![1; 6];
    dcae.encoder_layers_per_block = vec![1; 6];
    let mut transformer = if sprint {
        SanaTransformerConfig::sana_sprint_1600m()
    } else {
        SanaTransformerConfig::sana_1600m()
    };
    transformer.num_attention_heads = 4;
    transformer.attention_head_dim = 16;
    transformer.num_cross_attention_heads = 4;
    transformer.cross_attention_head_dim = 16;
    transformer.num_layers = 2;
    transformer.caption_channels = 64;
    transformer.mlp_ratio = 4.0;
    let mut gemma = Gemma2Config::gemma_2_2b();
    gemma.hidden_size = 64;
    gemma.num_layers = 2;
    gemma.num_heads = 4;
    gemma.num_kv_heads = 2;
    gemma.head_dim = 16;
    gemma.intermediate_size = 128;
    let mut te = Headers::new();
    let dt = Dtype::Bfloat16;
    te.put("model.embed_tokens.weight", &[256, 64], dt);
    te.norm("model.norm", 64, false, dt);
    for i in 0..gemma.num_layers {
        let p = format!("model.layers.{i}");
        for name in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            te.norm(&format!("{p}.{name}"), 64, false, dt);
        }
        for (name, input, out) in [
            ("q_proj", 64, 64),
            ("k_proj", 64, 32),
            ("v_proj", 64, 32),
            ("o_proj", 64, 64),
        ] {
            te.linear(&format!("{p}.self_attn.{name}"), input, out, false, dt);
        }
        for (name, input, out) in [
            ("gate_proj", 64, 128),
            ("up_proj", 64, 128),
            ("down_proj", 128, 64),
        ] {
            te.linear(&format!("{p}.mlp.{name}"), input, out, false, dt);
        }
    }
    let mut trunk = Headers::new();
    let dt = Dtype::Float16;
    trunk.conv("patch_embed.proj", 32, 64, 1, 1, true, dt);
    let ts = if sprint {
        "time_embed.timestep_embedder"
    } else {
        "time_embed.emb.timestep_embedder"
    };
    trunk.linear(&format!("{ts}.linear_1"), 256, 64, true, dt);
    trunk.linear(&format!("{ts}.linear_2"), 64, 64, true, dt);
    if sprint {
        trunk.linear("time_embed.guidance_embedder.linear_1", 256, 64, true, dt);
        trunk.linear("time_embed.guidance_embedder.linear_2", 64, 64, true, dt);
    }
    trunk.linear("time_embed.linear", 64, 384, true, dt);
    trunk.linear("caption_projection.linear_1", 64, 64, true, dt);
    trunk.linear("caption_projection.linear_2", 64, 64, true, dt);
    trunk.norm("caption_norm", 64, false, dt);
    trunk.put("scale_shift_table", &[2, 64], dt);
    trunk.linear("proj_out", 64, 32, true, dt);
    for i in 0..transformer.num_layers {
        let p = format!("transformer_blocks.{i}");
        trunk.put(format!("{p}.scale_shift_table"), &[6, 64], dt);
        for (attn, bias) in [("attn1", false), ("attn2", true)] {
            for name in ["to_q", "to_k", "to_v"] {
                trunk.linear(&format!("{p}.{attn}.{name}"), 64, 64, bias, dt);
            }
            trunk.linear(&format!("{p}.{attn}.to_out.0"), 64, 64, true, dt);
            if sprint {
                trunk.norm(&format!("{p}.{attn}.norm_q"), 64, false, dt);
                trunk.norm(&format!("{p}.{attn}.norm_k"), 64, false, dt);
            }
        }
        trunk.glu(&format!("{p}.ff"), 64, false, dt);
    }
    let mut vae = Headers::new();
    let dt = Dtype::Float32;
    vae.conv("encoder.conv_in", 3, 32, 3, 1, true, dt);
    vae.conv("encoder.conv_out", 32, 32, 3, 1, true, dt);
    vae.conv("decoder.conv_in", 32, 32, 3, 1, true, dt);
    vae.conv("decoder.conv_out", 32, 3, 3, 1, true, dt);
    vae.norm("decoder.norm_out", 32, true, dt);
    for i in 0..6 {
        vae.vae_block(
            &format!("encoder.down_blocks.{i}.0"),
            32,
            dcae.block_types[i],
            &dcae,
        );
        vae.vae_block(
            &format!("decoder.up_blocks.{i}.{}", u8::from(i < 5)),
            32,
            dcae.block_types[i],
            &dcae,
        );
        if i < 5 {
            vae.conv(
                &format!("encoder.down_blocks.{i}.1.conv"),
                32,
                32,
                3,
                1,
                true,
                dt,
            );
            vae.conv(
                &format!("decoder.up_blocks.{i}.0.conv"),
                32,
                32,
                3,
                1,
                true,
                dt,
            );
        }
    }
    SanaPricingFacts {
        root: "unused-header-only-native-walk".into(),
        transformer,
        dc_ae: dcae,
        gemma,
        gemma_tokens: 24,
        caption_tokens: 16,
        text_encoder: te.0,
        trunk: trunk.0,
        vae: vae.0,
    }
}
pub(super) fn plan(sprint: bool, img2img: bool, tiled: bool) -> SanaPricingPlan {
    SanaPricingPlan {
        sprint,
        sequential: tiled,
        img2img,
        width: 256,
        height: 256,
        count: 1,
        decode_tiling: tiled.then_some((192, 32)),
        encode_tile: tiled.then_some(192),
        attention: AttentionBudget::from_score_elements(128, true),
    }
}
pub(super) fn sampling(sprint: bool, img2img: bool, steps: u32) -> NativeSampling {
    let generation = mlx_gen::GenerationRequest {
        steps: Some(steps),
        strength: img2img.then_some(0.9),
        ..Default::default()
    };
    super::super::native_certificate::fixture_sampling(&generation, sprint, img2img, 1 << 20)
        .unwrap()
}
pub(super) fn packed(mut facts: SanaPricingFacts, bits: u64) -> SanaPricingFacts {
    for inv in [&mut facts.text_encoder, &mut facts.trunk] {
        let names: Vec<_> = inv
            .0
            .iter()
            .filter(|(name, t)| {
                name.ends_with(".weight") && t.shape.len() == 2 && t.shape[1].is_multiple_of(64)
            })
            .map(|(name, _)| name.clone())
            .collect();
        for name in names {
            let t = inv.0[&name].clone();
            let rows = t.shape[0];
            let input = t.shape[1];
            let base = name.strip_suffix(".weight").unwrap();
            let mut h = Headers(std::mem::take(inv));
            h.put(name.clone(), &[rows, input * bits / 32], Dtype::Uint32);
            h.put(
                format!("{base}.scales"),
                &[rows, input / 64],
                Dtype::Bfloat16,
            );
            h.put(
                format!("{base}.biases"),
                &[rows, input / 64],
                Dtype::Bfloat16,
            );
            *inv = h.0;
        }
    }
    facts
}
#[test]
fn closed_walk_covers_variants_routes_tiers_and_staged_tiled_regimes() {
    for sprint in [false, true] {
        for img2img in [false, true] {
            for tiled in [false, true] {
                for bits in [0, 4, 8] {
                    let f = if bits == 0 {
                        facts(sprint)
                    } else {
                        packed(facts(sprint), bits)
                    };
                    let p = plan(sprint, img2img, tiled);
                    let s = sampling(sprint, img2img, 2);
                    for backend in [NativeBackend::Cpu, NativeBackend::Metal] {
                        let upper = cumulative_upper(&f, &p, &s, 16_384, backend).unwrap();
                        assert!(upper > f.text_encoder.stored_where(|_| true));
                        println!("sana_native_upper sprint={sprint} reference={img2img} tiled={tiled} bits={bits} backend={backend:?} bytes={upper}");
                    }
                }
            }
        }
    }
}
#[test]
fn request_count_and_actual_solver_cadence_are_not_one_forward() {
    let f = facts(false);
    let p = plan(false, false, false);
    let two = cumulative_upper(
        &f,
        &p,
        &sampling(false, false, 2),
        16_384,
        NativeBackend::Cpu,
    )
    .unwrap();
    let four = cumulative_upper(
        &f,
        &p,
        &sampling(false, false, 4),
        16_384,
        NativeBackend::Cpu,
    )
    .unwrap();
    assert!(four > two);
    let mut batch = p;
    batch.count = 3;
    let three = cumulative_upper(
        &f,
        &batch,
        &sampling(false, false, 2),
        16_384,
        NativeBackend::Cpu,
    )
    .unwrap();
    assert!(
        three > two * 2,
        "constructor is shared, but each image actually executes its sampler and decoder"
    );
}
#[test]
fn missing_or_changed_source_headers_cannot_be_certified() {
    let mut f = facts(false);
    let p = plan(false, true, true);
    let s = sampling(false, true, 2);
    let key = "encoder.down_blocks.3.0.attn.to_q.weight";
    let saved = f.vae.0.remove(key).unwrap();
    assert!(cumulative_upper(&f, &p, &s, 16_384, NativeBackend::Cpu).is_err());
    f.vae.0.insert(key.into(), saved);
    f.trunk
        .0
        .get_mut("transformer_blocks.0.attn2.to_k.weight")
        .unwrap()
        .shape[1] += 1;
    assert!(cumulative_upper(&f, &p, &s, 16_384, NativeBackend::Metal).is_err());
}
#[test]
fn packed_embedding_cpu_expansion_is_not_the_metal_kernel() {
    let f = packed(facts(false), 4);
    let mut cpu = Walk::new(&f, 16_384, NativeBackend::Cpu).unwrap();
    let mut metal = Walk::new(&f, 16_384, NativeBackend::Metal).unwrap();
    cpu.gemma_embedding(24, 64).unwrap();
    metal.gemma_embedding(24, 64).unwrap();
    assert!(cpu.ledger.total > metal.ledger.total);
    let operand = |elements| {
        let mut ledger = NativeLedger::new(16_384, NativeBackend::Cpu).unwrap();
        ledger.array(4, elements).unwrap();
        ledger.total
    };
    // Q4 CPU affine source: eight left/right field pairs, not the Metal
    // dequantizer. All operands are individually passed through allocator U.
    let expected_cpu = 7 * operand(24)
        + 17 * operand(24 * 8)
        + 7 * operand(24 * 64)
        + 20 * operand(1)
        + 2 * operand(24 * 16)
        + 2 * operand(24 * 24);
    assert_eq!(cpu.ledger.total, expected_cpu);
}

#[test]
fn load_coherence_prices_cpu_reference_and_every_gpu_retry() {
    let f = facts(false);
    let mut headers = Headers::new();
    headers.put("odd-byte-norm.weight", &[1], Dtype::Float16);
    let mut cpu = Walk::new(&f, 16_384, NativeBackend::Cpu).unwrap();
    let mut metal = Walk::new(&f, 16_384, NativeBackend::Metal).unwrap();
    cpu.inventory(&headers.0).unwrap();
    metal.inventory(&headers.0).unwrap();
    let operand = |bytes| {
        let mut l = NativeLedger::new(16_384, NativeBackend::Cpu).unwrap();
        l.bytes(bytes).unwrap();
        l.total
    };
    // 2-byte source + 4-byte constructor envelope. Odd byte count widens two
    // U8 lanes to U32, then reduces: independently allocated 8-byte + 4-byte.
    let load = operand(2) + operand(4);
    let checksum = operand(8) + operand(4);
    let complete = load + checksum * (2 + u64::from(mlx_gen::coherence::MAX_RETRIES));
    // Compute stream CPU does NOT prove explicit default-GPU load checks are
    // absent. Both lowerings price the same complete possible checksum union.
    assert_eq!(cpu.ledger.total, complete);
    assert_eq!(metal.ledger.total, complete);
}

#[test]
fn every_actual_flow_solver_factory_is_walked() {
    use mlx_gen::gen_core::sampling::Solver;
    let f = facts(false);
    let p = plan(false, false, false);
    let mut totals = BTreeMap::new();
    for solver in Solver::ALL {
        let generation = mlx_gen::GenerationRequest {
            steps: Some(5),
            sampler: Some(solver.name().into()),
            ..Default::default()
        };
        let s =
            super::super::native_certificate::fixture_sampling(&generation, false, false, 1 << 20)
                .unwrap();
        let upper = cumulative_upper(&f, &p, &s, 16_384, NativeBackend::Cpu).unwrap();
        totals.insert(solver.name(), upper);
    }
    assert_eq!(totals.len(), Solver::ALL.len());
    assert!(
        totals["heun"] > totals["euler"],
        "actual corrector forwards must not collapse to one-per-step"
    );
}
