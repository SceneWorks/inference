use super::*;
use mlx_gen::gen_core::{
    MemoryBudget, MemoryCacheState, MemoryProviderContract, MemoryRunOutcome, MemorySelection,
    MemoryStrategyParameters, Precision, ProviderRegistryBuilder,
};
use mlx_gen::GenerationRequest;
use std::path::Path;

/// One named, independently applied mutation of a known-good contract.
type ContractMutation = (&'static str, Box<dyn Fn(&mut MemoryProviderContract)>);

/// The catalog conformance spec: a directory that does not exist, and a load shape this
/// provider deliberately does **not** implement.
fn weightless_spec() -> LoadSpec {
    LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
        .with_load_shape(LoadShape::DeferredMaterialization)
}

fn declared() -> MemoryProviderContract {
    weights_free_contract(&weightless_spec()).expect("weights-free contract")
}

#[test]
fn selected_source_partition_is_immune_to_unselected_weights_cache_constructor_and_patch() {
    let root = tempfile::tempdir().unwrap();
    gen_core_testkit::h3_memory::assert_selected_task_contract(root.path(), contract_for, true);
}

/// AC (SC-22662): the MiniMax-H3 contract publishes the axes of the DiT and video VAE this crate
/// declares, and passes the shared facts conformance check.
#[test]
fn architecture_facts_follow_the_crate_dit_and_vae_constants() {
    let contract = declared();
    assert_eq!(
        contract.architecture_facts,
        mlx_gen::gen_core::MemoryArchitectureFacts {
            attention_heads: Some(56),
            // The config's declared head width. `hidden_size` 5376 / 56 is NOT it: this DiT's
            // `inner_dim` (7168) is deliberately wider than its `hidden_size`.
            head_dim: Some(128),
            transformer_blocks: Some(50),
            // `patch_size` is `[1, 2, 2]`: the square spatial patch is 2.
            patch_size: Some(2),
            latent_channels: Some(24),
            vae_spatial_scale: Some(16),
            // A video autoencoder: four frames per latent unit.
            vae_temporal_scale: Some(4),
            // The bf16 denoise loop. SC-22667 also pins the `Precision::Fp32` leg below.
            activation_dtype_width: Some(2),
        }
    );
    assert!(contract.architecture_facts.has_declared_architecture_axis());
    gen_core_testkit::assert_memory_contract_facts_conform(&contract);
}

/// Feature-end review (SC-22667, E2): `activation_dtype_width` is the DENOISE-phase width, and
/// `model.rs` selects `Dtype::Float32` for the DiT under `Precision::Fp32` and
/// `Dtype::Bfloat16` otherwise. The literal used to be an unconditional 2 while its own comment
/// named that exception, so an Fp32 load published half the width its denoise loop computes at.
///
/// Mutation that fails this: hardcoding `HALF_ACTIVATION_WIDTH` again — the Fp32 leg reads back
/// 2. Hardcoding `FLOAT32_ACTIVATION_WIDTH` instead reds the bf16 leg, which is what stops the
/// fix from being "declare the widest phase": the f32 audio VAE belongs to the decode phase,
/// which this axis does not describe.
#[test]
fn the_denoise_activation_width_follows_the_loaded_precision() {
    let bf16 = weightless_spec();
    assert_eq!(
        architecture_facts(&bf16).activation_dtype_width,
        Some(mlx_gen::architecture_facts::HALF_ACTIVATION_WIDTH)
    );
    let mut fp32 = bf16.clone();
    fp32.precision = mlx_gen::Precision::Fp32;
    assert_eq!(
        architecture_facts(&fp32).activation_dtype_width,
        Some(mlx_gen::architecture_facts::FLOAT32_ACTIVATION_WIDTH)
    );
    // Nothing else on the axis set moves with the precision.
    let widened = architecture_facts(&fp32);
    let narrow = architecture_facts(&bf16);
    assert_eq!(widened.attention_heads, narrow.attention_heads);
    assert_eq!(widened.head_dim, narrow.head_dim);
    assert_eq!(widened.transformer_blocks, narrow.transformer_blocks);
    assert_eq!(widened.latent_channels, narrow.latent_channels);
}

/// The published `transformer/config.json` layout, emitted from a config value so the fixture
/// cannot drift from the struct it mirrors. Every key is required by `from_diffusers_json`.
fn dit_config_json(cfg: &crate::dit::config::MiniMaxH3DitConfig) -> serde_json::Value {
    serde_json::json!({
        "_class_name": "MiniMaxH3Transformer3DModel",
        "num_attention_heads": cfg.num_attention_heads,
        "attention_head_dim": cfg.attention_head_dim,
        "hidden_size": cfg.hidden_size,
        "num_layers": cfg.num_layers,
        "num_refiner_layers": cfg.num_refiner_layers,
        "ffn_dim": cfg.ffn_dim,
        "in_channels": cfg.in_channels,
        "audio_in_channels": cfg.audio_in_channels,
        "patch_size": cfg.patch_size,
        "text_dim": cfg.text_dim,
        "freq_dim": cfg.freq_dim,
        "time_embed_hidden_dim": cfg.time_embed_hidden_dim,
        "time_embed_dim": cfg.time_embed_dim,
        "rope_freq_dim": cfg.rope_freq_dim,
        "rope_theta": cfg.rope_theta,
        "norm_eps": cfg.norm_eps,
        "qk_norm_eps": cfg.qk_norm_eps,
        "final_norm_eps": cfg.final_norm_eps,
    })
}

fn spec_for_dit_config(dir: &Path, config: &serde_json::Value) -> LoadSpec {
    let transformer = dir.join(crate::model::BASE_DIT_PARTITION);
    std::fs::create_dir_all(&transformer).unwrap();
    std::fs::write(transformer.join("config.json"), config.to_string()).unwrap();
    LoadSpec::new(mlx_gen::gen_core::WeightsSource::Dir(dir.to_path_buf()))
}

/// AC (SC-22662, review follow-up): on the **materialized** path the trunk axes are read out of
/// the snapshot's own `transformer/config.json` — the file `DitBlockStream` parses at load —
/// rather than published from the compile-time preset. The mirror fixture agrees with the
/// weights-free path; a fixture with mutated keys publishes the mutated axes, which is the
/// assertion the unconditional `architecture_facts()` this replaced would fail.
#[test]
fn materialized_dit_axes_come_from_the_snapshot_rather_than_the_preset() {
    let preset = crate::dit::config::MiniMaxH3DitConfig::default();

    let mirror = tempfile::tempdir().unwrap();
    assert_eq!(
        architecture_facts(&spec_for_dit_config(
            mirror.path(),
            &dit_config_json(&preset)
        )),
        architecture_facts(&weightless_spec()),
        "a snapshot mirroring the published config must publish the preset's axes"
    );

    let mutated_dir = tempfile::tempdir().unwrap();
    let mut mutated = dit_config_json(&preset);
    mutated["num_layers"] = serde_json::json!(7);
    // 160, not a smaller width: `MiniMaxH3DitConfig::validate` requires the partial rotary
    // (`2 * 3 * rope_freq_dim` = 96) to fit inside a head, and a rejected config would fall
    // back to the preset and hide the mutation rather than publish it.
    mutated["attention_head_dim"] = serde_json::json!(160);
    let mutated_facts = architecture_facts(&spec_for_dit_config(mutated_dir.path(), &mutated));
    assert_eq!(
        (mutated_facts.transformer_blocks, mutated_facts.head_dim),
        (Some(7), Some(160)),
        "the materialized path must publish the snapshot's geometry, not the preset's"
    );
}

/// Feature-end review (SC-22667, E2): the DiT is the one tiered component, so a split `q4`
/// install stages [`DIT_COMPONENT`] OUTSIDE the snapshot and carries no `root/transformer` at
/// all — `model.rs` says exactly that where it probes the same file, and
/// `ComponentBytes::resolve` already honours the staged override for the DiT and the text
/// encoder. `dit_config` read only under the root, so every split install fell back to the
/// preset although the loaded config was one `spec.components` lookup away.
///
/// Mutation that fails this: dropping the staged leg from `resolved_dit_config` — the split
/// fixture then publishes the preset's 50 blocks instead of the staged config's 7.
#[test]
fn a_split_tier_install_reads_the_dit_config_at_its_staged_location() {
    let preset = crate::dit::config::MiniMaxH3DitConfig::default();
    let mut staged_config = dit_config_json(&preset);
    staged_config["num_layers"] = serde_json::json!(7);

    // A snapshot root with NO `transformer/` at all, plus the DiT staged elsewhere.
    let root = tempfile::tempdir().unwrap();
    let staged = tempfile::tempdir().unwrap();
    std::fs::write(staged.path().join("config.json"), staged_config.to_string()).unwrap();
    let mut spec = LoadSpec::new(mlx_gen::gen_core::WeightsSource::Dir(
        root.path().to_path_buf(),
    ));
    spec.components.insert(
        DIT_COMPONENT.to_owned(),
        mlx_gen::gen_core::WeightsSource::Dir(staged.path().to_path_buf()),
    );
    assert_eq!(
        architecture_facts(&spec).transformer_blocks,
        Some(7),
        "a staged DiT partition must be read where `resolve_dit_dir` would read it"
    );

    // The staged partition WINS over a root partition, exactly as `resolve_dit_dir` prefers it.
    let both = tempfile::tempdir().unwrap();
    let transformer = both.path().join(crate::model::BASE_DIT_PARTITION);
    std::fs::create_dir_all(&transformer).unwrap();
    std::fs::write(
        transformer.join("config.json"),
        dit_config_json(&preset).to_string(),
    )
    .unwrap();
    let mut both_spec = LoadSpec::new(mlx_gen::gen_core::WeightsSource::Dir(
        both.path().to_path_buf(),
    ));
    both_spec.components.insert(
        DIT_COMPONENT.to_owned(),
        mlx_gen::gen_core::WeightsSource::Dir(staged.path().to_path_buf()),
    );
    assert_eq!(architecture_facts(&both_spec).transformer_blocks, Some(7));

    // A flat snapshot with no staged component is unchanged.
    let flat = tempfile::tempdir().unwrap();
    let flat_spec = spec_for_dit_config(flat.path(), &dit_config_json(&preset));
    assert_eq!(
        architecture_facts(&flat_spec).transformer_blocks,
        architecture_facts(&weightless_spec()).transformer_blocks
    );
}

/// Feature-end review (SC-22667, E2): a spec that HAS a DiT to load but no readable
/// `transformer/config.json` must declare the trunk axes ABSENT, not publish the preset.
/// `MiniMaxH3Dit::load_dir` errors on exactly that file, so the preset would describe a model
/// this load can never build — the same rule Mage's `dit_config` follows. The VAE axes survive,
/// so the contract still declares a real architecture axis.
///
/// Mutations that fail this:
/// * restoring `resolved_dit_config(spec).unwrap_or_default()` — every leg below reads back the
///   preset's 50 blocks instead of `None`;
/// * chaining a root fallback after the staged directory in `resolved_dit_config` — the third
///   leg reads back `Some(50)` from the root partition that `resolve_dit_dir` never opens.
#[test]
fn an_unreadable_dit_config_declares_the_trunk_absent_rather_than_the_preset() {
    let preset = crate::dit::config::MiniMaxH3DitConfig::default();
    let vae_preset = crate::config::MiniMaxH3VaeConfig::default();

    // A materialized snapshot with no `transformer/config.json` at all.
    let bare = tempfile::tempdir().unwrap();
    let bare_facts = architecture_facts(&LoadSpec::new(WeightsSource::Dir(
        bare.path().to_path_buf(),
    )));
    assert_eq!(
        (
            bare_facts.attention_heads,
            bare_facts.head_dim,
            bare_facts.transformer_blocks,
            bare_facts.patch_size
        ),
        (None, None, None, None),
        "a materialized snapshot the loader would reject must not publish the preset trunk"
    );
    assert_eq!(
        bare_facts.latent_channels,
        mlx_gen::architecture_facts::axis(vae_preset.latent_channels),
        "the VAE axes are their own read and must survive the absent trunk"
    );

    // A materialized snapshot whose config is present but unparseable.
    let partial = tempfile::tempdir().unwrap();
    let transformer = partial.path().join(crate::model::BASE_DIT_PARTITION);
    std::fs::create_dir_all(&transformer).unwrap();
    let mut missing_key = dit_config_json(&preset);
    missing_key
        .as_object_mut()
        .unwrap()
        .remove("num_attention_heads");
    std::fs::write(transformer.join("config.json"), missing_key.to_string()).unwrap();
    assert_eq!(
        architecture_facts(&LoadSpec::new(WeightsSource::Dir(
            partial.path().to_path_buf()
        )))
        .transformer_blocks,
        None,
        "a partial config declines whole; `from_diffusers_json` requires every key"
    );

    // A STAGED DiT whose config is unreadable does not fall back to a readable root partition:
    // `resolve_dit_dir` returns the staged directory outright and `load_dir` fails there.
    let both = tempfile::tempdir().unwrap();
    let root_transformer = both.path().join(crate::model::BASE_DIT_PARTITION);
    std::fs::create_dir_all(&root_transformer).unwrap();
    std::fs::write(
        root_transformer.join("config.json"),
        dit_config_json(&preset).to_string(),
    )
    .unwrap();
    let staged = tempfile::tempdir().unwrap();
    let mut staged_spec = LoadSpec::new(WeightsSource::Dir(both.path().to_path_buf()));
    staged_spec.components.insert(
        DIT_COMPONENT.to_owned(),
        WeightsSource::Dir(staged.path().to_path_buf()),
    );
    assert_eq!(
        architecture_facts(&staged_spec).transformer_blocks,
        None,
        "a staged DiT wins outright; the root partition is not a fallback for it"
    );

    // And with NO load to describe at all, the preset is still what the registry publishes.
    assert_eq!(
        architecture_facts(&weightless_spec()).transformer_blocks,
        mlx_gen::architecture_facts::axis(preset.num_layers),
    );
}

/// The published `vae/config.json` layout, emitted from a config value so the fixture cannot
/// drift from the struct it mirrors. Every key is required by
/// `MiniMaxH3VaeConfig::from_diffusers_json`, which also re-derives `patch_size` /
/// `patch_size_t` as the products of the two per-level factor lists.
fn vae_config_json(cfg: &crate::config::MiniMaxH3VaeConfig) -> serde_json::Value {
    serde_json::json!({
        "latent_channels": cfg.latent_channels,
        "out_channels": cfg.out_channels,
        "decoder_num_layers": cfg.num_layers,
        "decoder_num_attention_heads": cfg.num_heads,
        "decoder_attention_head_dim": cfg.head_dim,
        "decoder_num_register_tokens": cfg.num_register_tokens,
        "decoder_ffn_mult": cfg.ffn_mult,
        "decoder_rope_theta": cfg.rope_theta,
        "decoder_rope_dim_ratio": cfg.rope_dim_ratio,
        "decoder_norm_eps": cfg.norm_eps,
        "clip_length": cfg.clip_length,
        "token_drop": cfg.token_drop,
        "latents_mean": cfg.latents_mean,
        "latents_std": cfg.latents_std,
        "in_channels": cfg.in_channels,
        "block_out_channels": cfg.block_out_channels,
        "layers_per_block": cfg.layers_per_block,
        "spatial_downsample_factors": cfg.spatial_downsample_factors,
        "temporal_downsample_factors": cfg.temporal_downsample_factors,
        "norm_num_groups": cfg.norm_num_groups,
        "norm_eps": cfg.encoder_norm_eps,
    })
}

/// SC-22667: the VAE axes are read out of the snapshot's own `vae/config.json` — the file the
/// VAE loader parses — rather than published from the `VAE_RATIO` / `LATENT_CHANNELS` preset
/// constants. The mirror fixture agrees with the weights-free path; a fixture with mutated
/// factor lists and latent width publishes the mutated axes.
///
/// Mutation that fails this: `latent_channels: axis(LATENT_CHANNELS)`,
/// `vae_spatial_scale: axis(VAE_RATIO)`, `vae_temporal_scale: axis(VAE_RATIO_T)` (the preset
/// shape under review) — the mutated fixture then reads back 24 / 16 / 4.
#[test]
fn materialized_vae_axes_come_from_the_snapshot_rather_than_the_preset() {
    let preset = crate::config::MiniMaxH3VaeConfig::default();
    // SC-22667 review: both fixtures below mirror the DiT config too. A materialized snapshot
    // with no readable `transformer/config.json` now declares its TRUNK axes absent (the loader
    // errors on that file), so a VAE-only fixture can no longer be compared against the
    // weights-free preset — and an absent trunk would say nothing about the VAE axes this test
    // is for.
    let mirror_dit = |root: &std::path::Path| {
        let transformer = root.join(crate::model::BASE_DIT_PARTITION);
        std::fs::create_dir_all(&transformer).unwrap();
        std::fs::write(
            transformer.join("config.json"),
            dit_config_json(&crate::dit::config::MiniMaxH3DitConfig::default()).to_string(),
        )
        .unwrap();
    };

    let mirror = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(mirror.path().join("vae")).unwrap();
    std::fs::write(
        mirror.path().join("vae/config.json"),
        vae_config_json(&preset).to_string(),
    )
    .unwrap();
    mirror_dit(mirror.path());
    let mirror_spec = LoadSpec::new(mlx_gen::gen_core::WeightsSource::Dir(
        mirror.path().to_path_buf(),
    ));
    assert_eq!(
        architecture_facts(&mirror_spec),
        architecture_facts(&weightless_spec()),
        "a snapshot mirroring the published VAE config must publish the preset's axes"
    );

    // Halve the latent width and the spatial compression, keep everything the validator
    // cross-checks consistent (`latents_mean/std` per channel, one factor per level).
    let mut mutated = preset.clone();
    mutated.latent_channels = 12;
    mutated.latents_mean = vec![0.0; 12];
    mutated.latents_std = vec![1.0; 12];
    let levels = mutated.spatial_downsample_factors.len();
    mutated.spatial_downsample_factors = vec![1; levels];
    mutated.spatial_downsample_factors[0] = 8;
    mutated.temporal_downsample_factors = vec![1; levels];
    mutated.temporal_downsample_factors[0] = 2;
    let mutated_dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(mutated_dir.path().join("vae")).unwrap();
    std::fs::write(
        mutated_dir.path().join("vae/config.json"),
        vae_config_json(&mutated).to_string(),
    )
    .unwrap();
    mirror_dit(mutated_dir.path());
    let facts = architecture_facts(&LoadSpec::new(mlx_gen::gen_core::WeightsSource::Dir(
        mutated_dir.path().to_path_buf(),
    )));
    assert_eq!(
        (
            facts.latent_channels,
            facts.vae_spatial_scale,
            facts.vae_temporal_scale
        ),
        (Some(12), Some(8), Some(2)),
        "the materialized path must publish the snapshot's VAE geometry, not the preset's"
    );
    // The trunk axes are untouched by a VAE-only fixture.
    assert_eq!(
        facts.transformer_blocks,
        architecture_facts(&weightless_spec()).transformer_blocks
    );
}

/// SC-22667 review: the VAE branch now mirrors the DiT branch. `MiniMaxH3Vae::load` reads
/// `vae/config.json` and propagates both its read and its parse failure, so a file that is
/// PRESENT and unparseable describes no load — its axes are declared absent rather than filled
/// in from this crate's mirror. An ABSENT file keeps the mirror: that is the weights-free shape
/// and the shape of a snapshot whose VAE component is not materialized yet, and nothing there
/// contradicts the shipped geometry.
///
/// Mutation that fails this: restoring `.unwrap_or_default()` in `vae_config` — the unparseable
/// leg then reads back the mirror's `latent_channels` instead of `None`.
#[test]
fn an_unparseable_vae_config_declines_the_vae_axes_instead_of_the_mirror() {
    let preset_facts = architecture_facts(&weightless_spec());
    let mirror_dit = |root: &std::path::Path| {
        let transformer = root.join(crate::model::BASE_DIT_PARTITION);
        std::fs::create_dir_all(&transformer).unwrap();
        std::fs::write(
            transformer.join("config.json"),
            dit_config_json(&crate::dit::config::MiniMaxH3DitConfig::default()).to_string(),
        )
        .unwrap();
    };

    // Present but missing a required key: `from_diffusers_json` declines it whole.
    let broken = tempfile::tempdir().unwrap();
    mirror_dit(broken.path());
    std::fs::create_dir_all(broken.path().join("vae")).unwrap();
    let mut partial = vae_config_json(&crate::config::MiniMaxH3VaeConfig::default());
    partial.as_object_mut().unwrap().remove("latent_channels");
    std::fs::write(broken.path().join("vae/config.json"), partial.to_string()).unwrap();
    let facts = architecture_facts(&LoadSpec::new(WeightsSource::Dir(
        broken.path().to_path_buf(),
    )));
    assert_eq!(
        (
            facts.latent_channels,
            facts.vae_spatial_scale,
            facts.vae_temporal_scale
        ),
        (None, None, None),
        "an unparseable vae/config.json must not publish the crate mirror's geometry"
    );
    assert_eq!(
        facts.transformer_blocks, preset_facts.transformer_blocks,
        "the DiT read is independent and survives"
    );
    assert!(facts.has_declared_architecture_axis());

    // Absent: the mirror still stands, which is what the weights-free surface publishes.
    let absent = tempfile::tempdir().unwrap();
    mirror_dit(absent.path());
    assert_eq!(
        architecture_facts(&LoadSpec::new(WeightsSource::Dir(
            absent.path().to_path_buf()
        ))),
        preset_facts,
        "no vae/config.json to read keeps the shipped mirror"
    );
}

fn support(contract: &MemoryProviderContract, strategy: MemoryStrategy) -> MemoryStrategySupport {
    contract
        .capability(strategy)
        .unwrap_or_else(|| panic!("{strategy:?} must appear in the strategy table"))
        .support
        .clone()
}

/// DiT shards have valid headers at unchanged exact file sizes for header-aware admission.
/// Other component fixtures retain their existing opaque, stat-only byte semantics.
#[test]
fn native_vae_facts_follow_selected_config_readsets_and_actual_compute_dtypes() {
    let root = tempfile::tempdir().unwrap();
    gen_core_testkit::h3_memory::write_native_component(&root.path().join("transformer"), 2);
    gen_core_testkit::h3_memory::write_vae_configs(root.path());
    gen_core_testkit::h3_memory::write_float_operands(
        &root.path().join("vae"),
        [
            "encoder.conv_in.weight",
            "decoder.patch_embed.weight",
            "post_quant_conv.weight",
        ]
        .map(str::to_owned),
        "F16",
    );
    let a = crate::audio_config::MiniMaxH3AudioVaeConfig::default();
    let mut names = crate::audio_vae_encoder::MiniMaxH3AudioVaeEncoder::tensor_names(&a);
    names.extend(crate::audio_vae::MiniMaxH3AudioVae::tensor_names(&a));
    gen_core_testkit::h3_memory::write_float_operands(
        &root.path().join("audio_vae"),
        names,
        "BF16",
    );
    let mut spec = LoadSpec::new(WeightsSource::Dir(root.path().into()));
    spec.precision = mlx_gen::Precision::Fp32;
    let contract = contract_for(&spec).unwrap();
    let native = contract
        .phase_facts
        .as_ref()
        .unwrap()
        .native_quant
        .as_ref()
        .unwrap();
    let vae = native.conditioning.as_ref().unwrap();
    assert_eq!(
        vae.decoder_layers, 3,
        "source config, not stock preset depth"
    );
    assert_eq!(vae.video_element_bytes, 4);
    assert_eq!(
        vae.audio_decode_element_bytes, 4,
        "MLX audio decoder and encoder both remain F32"
    );
    assert!(vae.unified_memory);
    assert!(native.conditioning_resident_bytes > native.video_decode_resident_bytes);
    assert!(
        native.audio_decode_load_bytes > native.audio_decode_resident_bytes,
        "weightnorm source/derived constructor overlap is not permanent residency"
    );
    assert_eq!(
        contract.asset_facts.decoder_bytes,
        native.video_decode_resident_bytes + native.audio_decode_resident_bytes
    );
    let req = mlx_gen::gen_core::GenerationRequest {
        width: 1344,
        height: 768,
        frames: Some(124),
        ..Default::default()
    };
    let workspace =
        mlx_gen::gen_core::h3_conditioning_memory::workspace(&(&req).into(), vae).unwrap();
    assert!(workspace.decode_bytes > 0);
    assert!(
        vae.video_forward_extra_bytes > 0,
        "lazy source F16 to compute F32 cast overlaps first decode"
    );
    assert!(
        vae.audio_decode_forward_extra_bytes > 0,
        "lazy weightnorm parameters overlap decode"
    );
    let mut without_cast = vae.clone();
    without_cast.video_forward_extra_bytes = 0;
    let no_cast =
        mlx_gen::gen_core::h3_conditioning_memory::workspace(&(&req).into(), &without_cast)
            .unwrap();
    assert_eq!(
        workspace.video_decode_bytes - no_cast.video_decode_bytes,
        vae.video_forward_extra_bytes,
        "real provider-mapped forward phase, not a separate constructor MAX"
    );
}

#[test]
fn packed_admission_bound_covers_native_stereo_target_and_long_reference_audio() {
    let geometry = crate::denoise::JointGeometry::new(
        345,
        768 / crate::config::VAE_RATIO,
        1344 / crate::config::VAE_RATIO,
    )
    .unwrap();
    let actual = crate::cost::packed_seq_len(
        &geometry,
        crate::pipeline::PATCH_SIZE,
        1,
        0,
        crate::audio_config::AUDIO_OUTPUT_CHANNELS as i32,
    )
    .unwrap() as u64;
    gen_core_testkit::h3_memory::assert_stereo_packing_bound(actual);
}

#[test]
fn dense_source_keeps_actual_dtype_accounting_without_integer_descriptors() {
    let root = tempfile::tempdir().unwrap();
    gen_core_testkit::h3_memory::assert_dense_contract(root.path(), contract_for, true);
}

#[test]
fn native_formats_price_actual_payloads_shared_buffers_and_staged_task() {
    let root = tempfile::tempdir().unwrap();
    let tier = tempfile::tempdir().unwrap();
    let staged = tier.path().join("transformer");
    gen_core_testkit::h3_memory::assert_native_contract(root.path(), &staged, contract_for, true);
}

/// Write a snapshot tree whose component directories hold **sparse** `.safetensors` files of
/// the exact measured sizes. Valid headers exercise tier detection without materializing the
/// sparse payload; stat-based component accounting still sees every original measured byte.
fn sparse_snapshot(root: &Path, sizes: &[(&str, u64)]) {
    for (component, bytes) in sizes {
        let dir = root.join(component);
        std::fs::create_dir_all(&dir).expect("component dir");
        let mut file = std::fs::File::create(dir.join("model.safetensors")).expect("shard");
        use std::io::Write;
        // Header and both payload intervals cover the exact sparse file, including odd
        // measured lengths. A real BF16 tensor proves dense storage; U8 padding is not a
        // pretend floating parameter and must not alter stat-derived component charges.
        let mut padding = 0_u64;
        let header = loop {
            let header = format!(
                r#"{{"w":{{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}},"padding":{{"dtype":"U8","shape":[{padding:20}],"data_offsets":[2,{:20}]}}}}"#,
                padding + 2
            );
            let remaining = (*bytes)
                .checked_sub(8 + header.len() as u64 + 2)
                .expect("fixture fits header");
            if padding == remaining {
                break header;
            }
            padding = remaining;
        };
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header.as_bytes()).unwrap();
        file.set_len(*bytes).expect("sparse shard");
    }
}

/// Write the DiT's own `quantization` marker into a snapshot tree — the same `config.json`
/// `mlx_gen::quant::packed_quant_bits_at` parses on the load path, and therefore the file
/// `resolved_artifact_tier` reads. `None` writes a DENSE marker (a real `config.json` with no
/// `quantization` block), which is what makes a bf16 root distinguishable from an absent one.
fn write_tier_marker(root: &Path, bits: Option<i32>) {
    let dir = root.join(DIT_COMPONENT);
    std::fs::create_dir_all(&dir).expect("component dir");
    let body = match bits {
        Some(bits) => format!(
            "{{\"quantization\":{{\"bits\":{bits},\"group_size\":{}}}}}",
            crate::quant::GROUP_SIZE
        ),
        None => "{}".to_owned(),
    };
    std::fs::write(dir.join("config.json"), body).expect("quantization marker");
}

// ------------------------------------------------------------------------------------------
// sc-22737 (epic sc-22723 E1/E4): the identity is per ARTIFACT-PROVEN tier, and the
// weights-free surface may not publish it.
// ------------------------------------------------------------------------------------------

/// A snapshot root whose DiT carries a real tier marker.
fn minimax_tier_root(bits: Option<i32>) -> tempfile::TempDir {
    let root = tempfile::tempdir().expect("tempdir");
    sparse_snapshot(root.path(), &[("transformer", DIT_BF16_BYTES)]);
    write_tier_marker(root.path(), bits);
    root
}

/// Three shipped tiers, three DISTINCT identities, each reaching the production contract — and
/// the measured bf16 cell keeping its retained key. Before sc-22737 all three published the
/// SAME string, so an anchor measured on bf16 would have been matched against q4 and q8 loads.
#[test]
fn each_shipped_tier_publishes_its_own_identity() {
    let mut seen = std::collections::BTreeSet::new();
    for (bits, expected_tier) in [(None, "bf16"), (Some(4), "q4"), (Some(8), "q8")] {
        let root = minimax_tier_root(bits);
        let spec = LoadSpec::new(WeightsSource::Dir(root.path().into()));
        let fingerprint = production_calibration_fingerprint(&spec)
            .unwrap_or_else(|| panic!("{expected_tier} publishes an identity"));
        assert!(
            fingerprint.contains(expected_tier) || expected_tier == "bf16",
            "{fingerprint} must name {expected_tier}"
        );
        assert!(
            seen.insert(fingerprint.clone()),
            "{expected_tier} reuses {fingerprint}"
        );
        assert_ne!(fingerprint, STATIC_CALIBRATION_FINGERPRINT);
        assert_eq!(
            contract_for(&spec)
                .unwrap()
                .calibration
                .as_ref()
                .map(|identity| identity.fingerprint.as_str()),
            Some(fingerprint.as_str()),
            "{expected_tier}: the identity must reach the CONTRACT, not just the table"
        );
    }
    assert_eq!(seen.len(), 3);
    // The measured cell keeps its retained key byte-for-byte.
    let bf16 = minimax_tier_root(None);
    assert_eq!(
        production_calibration_fingerprint(&LoadSpec::new(WeightsSource::Dir(bf16.path().into())))
            .as_deref(),
        Some(MEMORY_CALIBRATION_FINGERPRINT)
    );
}

/// **The tier is proven, not requested.** A dense root asked for as q4 is a load-time
/// requantization no anchor measured, and an absent snapshot proves nothing at all. Both
/// withhold — and neither fails the load.
#[test]
fn a_tier_the_artifact_does_not_prove_is_withheld() {
    let dense = minimax_tier_root(None);
    let crossed =
        LoadSpec::new(WeightsSource::Dir(dense.path().into())).with_quant(mlx_gen::Quant::Q4);
    assert!(production_calibration_fingerprint(&crossed).is_none());
    assert!(
        contract_for(&crossed).unwrap().calibration.is_none(),
        "a withheld identity must not fail the load"
    );

    // An honest request for the tier the artifact carries does publish.
    let packed = minimax_tier_root(Some(8));
    let honest =
        LoadSpec::new(WeightsSource::Dir(packed.path().into())).with_quant(mlx_gen::Quant::Q8);
    assert_eq!(
        production_calibration_fingerprint(&honest).as_deref(),
        Some("minimax-h3-q8-mlx-staged-joint-av-eager-abi3-v1")
    );

    // No marker at all is an UNREADABLE tier, not a dense one — fail closed.
    assert!(production_calibration_fingerprint(&weightless_spec()).is_none());
    let unmarked = tempfile::tempdir().expect("tempdir");
    sparse_snapshot(unmarked.path(), &[("transformer", DIT_BF16_BYTES)]);
    assert!(
        production_calibration_fingerprint(&LoadSpec::new(WeightsSource::Dir(
            unmarked.path().into()
        )))
        .is_none(),
        "a snapshot with no config.json marker must not read as the dense tier"
    );
}

/// The weights-free declaration may never publish a production string — the leak sc-22737
/// closed. A plan row naming a measured cell could otherwise be satisfied by a contract that
/// loaded no weights at all.
#[test]
fn the_weights_free_declaration_cannot_publish_a_production_string() {
    for bits in [None, Some(4), Some(8)] {
        let root = minimax_tier_root(bits);
        let spec = LoadSpec::new(WeightsSource::Dir(root.path().into()));
        let declared = weights_free_contract(&spec)
            .unwrap()
            .calibration
            .unwrap()
            .fingerprint;
        assert_eq!(declared, STATIC_CALIBRATION_FINGERPRINT);
        assert_ne!(
            production_calibration_fingerprint(&spec).unwrap(),
            declared,
            "q{bits:?}"
        );
    }
}

// --- AC1: the resolved contract is the DECLARED one, never the compatibility default ---------

/// A Resident-only assertion is a false green here: `compatibility_default` also reports
/// Resident. This distinguishes the two by the things a fallback contract **cannot** produce.
#[test]
fn resolved_contract_is_declared_and_not_the_compatibility_default() {
    let spec = weightless_spec();
    let registry = crate::provider_registry().expect("registry");
    let registration = registry
        .memory_strategy_registrations()
        .find(|registration| registration.provider_id == MODEL_ID)
        .expect("minimax_h3 must be registered on the memory ladder");
    let contract = (registration.contract)(&spec).expect("registered contract");

    let fallback = MemoryProviderContract::compatibility_default(
        MODEL_ID,
        MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: true,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
    );
    assert_ne!(
        contract, fallback,
        "the resolved contract must not be the resident-only fallback"
    );

    // Each of these is impossible for `compatibility_default`, so each independently proves the
    // declaration was resolved rather than the fallback.
    assert_eq!(
        support(&contract, MemoryStrategy::StagedResidency),
        MemoryStrategySupport::Implemented,
        "the fallback declares every optimized rung Missing"
    );
    // sc-22737: `contract_for` on a weightless spec proves no artifact tier and so publishes no
    // identity, which the compatibility fallback also does — so this is no longer a
    // discriminator on THIS spec. The claim it was making is kept, against a spec that resolves
    // a real tier: a resolved declaration publishes a per-tier identity, the fallback never
    // publishes one at all.
    assert_eq!(fallback.calibration, None);
    let resolved_root = tempfile::tempdir().expect("tempdir");
    sparse_snapshot(resolved_root.path(), &[("transformer", DIT_BF16_BYTES)]);
    write_tier_marker(resolved_root.path(), None);
    let resolved = (registration.contract)(&LoadSpec::new(WeightsSource::Dir(
        resolved_root.path().into(),
    )))
    .expect("registered contract");
    assert!(
        resolved.calibration.is_some(),
        "a resolved declaration publishes a per-tier identity; the fallback never does"
    );
    assert!(
        matches!(
            contract.formula,
            MemoryFormulaKind::ComponentPhaseEnvelope { .. }
        ),
        "the fallback formula is AssetBytesPlusHeadroom, got {:?}",
        contract.formula
    );
    assert_eq!(contract.lifecycle.phases, phases());
    assert!(contract.lifecycle.synchronized_phase_release);
}

/// The registration is wired end to end: removing any one of the three lines, or misspelling
/// the provider id on any of them, fails `build()`.
#[test]
fn the_three_memory_registrations_are_present_and_id_matched() {
    let registry = crate::provider_registry().expect("registry");
    for (kind, found) in [
        (
            "memory strategy",
            registry
                .memory_strategy_registrations()
                .any(|r| r.provider_id == MODEL_ID),
        ),
        (
            "contract fixture",
            registry
                .memory_contract_fixture_registrations()
                .any(|r| r.provider_id == MODEL_ID),
        ),
        (
            "behavior",
            registry
                .memory_behavior_registrations()
                .any(|r| r.provider_id == MODEL_ID),
        ),
    ] {
        assert!(found, "{MODEL_ID} is missing its {kind} registration");
    }

    // A misspelled id has no matching generator and must be rejected at build time rather than
    // becoming a contract with no executable owner.
    let typo = ProviderRegistryBuilder::new()
        .register_generator(crate::model::REGISTRATION)
        .register_memory_strategy(mlx_gen::gen_core::MemoryRegistration {
            provider_id: "minimax_h4",
            contract: contract_for,
            safety_check,
        })
        .build();
    assert!(
        typo.is_err(),
        "a misspelled memory-strategy provider id must fail registry build"
    );
}

/// The contract a caller resolves reports the declared id, not a neighbour's.
#[test]
fn contract_reports_its_own_provider_id() {
    assert_eq!(declared().provider_id, MODEL_ID);
    assert_eq!(MEMORY_REGISTRATION.provider_id, MODEL_ID);
    assert_eq!(MEMORY_CONTRACT_FIXTURE.provider_id, MODEL_ID);
    assert_eq!(MEMORY_BEHAVIOR.provider_id, MODEL_ID);
}

// --- AC2 + AC3: weights-free conformance drives every implemented rung ----------------------

/// The shipped registry conformance, run in this crate's own default lane rather than only in
/// the catalog. It resolves the **fixture** contract, then for every optimized rung declared
/// `Implemented` it drives `valid_fixtures` → `safety_check` → `begin_request` →
/// `configure_request` → `enter_phase` → `leave_phase` → `finish`, and proves the safety check
/// is blind to none of the calibration ABI, fingerprint, load shape, numeric tier or budget.
#[test]
fn registry_conformance_drives_every_implemented_rung_weights_free() {
    let registry = crate::provider_registry().expect("registry");
    gen_core_testkit::memory_strategy_registry_conformance(&registry, &weightless_spec());
}

/// The fixture contract is genuinely weights-free: zero asset facts, no filesystem traversal,
/// and the identical route declaration.
#[test]
fn fixture_contract_is_weights_free_and_matches_the_route_declaration() {
    let spec = weightless_spec();
    let fixture = (MEMORY_CONTRACT_FIXTURE.contract)(&spec).expect("fixture contract");
    assert_eq!(fixture.asset_facts, MemoryAssetFacts::default());
    let production = (MEMORY_REGISTRATION.contract)(&spec).expect("production contract");
    assert_eq!(fixture.strategies, production.strategies);
    assert_eq!(fixture.lifecycle, production.lifecycle);
    assert_eq!(fixture.formula, production.formula);
    // sc-22737: the two identities must DIFFER. This asserted they were equal, which is what
    // let a contract that read no weights report the identity of a measured production cell.
    // On this weightless spec the production side proves no tier at all and so publishes
    // nothing; the declaration publishes its own registry-behaviour key.
    assert_eq!(
        fixture
            .calibration
            .as_ref()
            .map(|id| id.fingerprint.as_str()),
        Some(STATIC_CALIBRATION_FINGERPRINT)
    );
    assert_eq!(production.calibration, None);
    assert_eq!(fixture.load_shape, production.load_shape);
    assert_eq!(fixture.backend, production.backend);
}

/// Rung 1's mechanism is reachable in the provider's own request surface: the scope writes the
/// canonical memory block, and the resulting request survives the generator's real validator.
///
/// This is the step registry conformance cannot take — it checks the block against the
/// contract, not against the model. A `Resident` selection is included as the control arm, so
/// the test proves the two selections are *distinguishable* rather than that some block exists.
#[test]
fn staged_residency_reaches_the_generators_own_validator() {
    let spec = weightless_spec();
    let contract = declared();
    let caps = crate::model::descriptor().capabilities;

    for (strategy, expect_staged) in [
        (MemoryStrategy::StagedResidency, true),
        (MemoryStrategy::Resident, false),
    ] {
        let mut context = mlx_gen::gen_core::standard_memory_behavior_context(
            &contract,
            strategy,
            mlx_gen::request_scope::loaded_tier(&spec),
            MemoryBehaviorRoute {
                mode: MemoryMode::TextToImage,
                reference_count: 0,
                use_pid: false,
                has_phases: true,
                overlay: None,
            },
        )
        .expect("behavior context");
        context.geometry = MemoryGeometry {
            width: FIXTURE_WIDTH,
            height: FIXTURE_HEIGHT,
            batch: 1,
            frames: FIXTURE_FRAMES,
            reference_count: 0,
        };
        assert_eq!(
            safety_check(&spec, &contract, &context),
            MemorySafetyDecision::Accept,
            "{strategy:?} must admit its own legal geometry"
        );

        let mut scope = registered_begin_request(&spec, &contract, &context)
            .expect("begin_request")
            .expect("scope");
        let mut request = GenerationRequest {
            prompt: "a slow pan across a rainy street at night".into(),
            width: FIXTURE_WIDTH,
            height: FIXTURE_HEIGHT,
            frames: Some(FIXTURE_FRAMES),
            ..Default::default()
        };
        scope.configure_request(&mut request).expect("configure");
        assert_eq!(
            request.memory.is_some_and(|memory| memory.stage_residency),
            expect_staged,
            "{strategy:?} must map to stage_residency={expect_staged}"
        );
        crate::model::validate_request(&caps, &request).unwrap_or_else(|error| {
            panic!("{strategy:?} produced a request the generator rejects: {error}")
        });
        scope.finish(MemoryRunOutcome::Complete).expect("finish");
    }
}

// --- sc-18660 AC4: rung 2 is REACHABLE, not merely declared --------------------------------

/// **The whole rung-2 chain, executed.** Contract → selection → `safety_check` → scope →
/// `configure_decode` → `GenerationRequest.memory` → the pipeline's own reader.
///
/// The chain ends at [`crate::pipeline::decode_tiling_for`], which is the function
/// `MiniMaxH3::decode_video` calls. Nothing here asserts that a *default* arrived — the
/// admitted edge equals the default, so such an assertion would pass with the plumbing deleted
/// (`test_asserting_a_default_value_is_a_false_green`). What is asserted instead is that the
/// selection's values travel intact **and** that the two negative directions are refused.
#[test]
fn bounded_decode_reaches_the_decode_call_through_the_request() {
    let spec = weightless_spec();
    let contract = declared();
    let tier = mlx_gen::request_scope::loaded_tier(&spec);
    let mut context = mlx_gen::gen_core::standard_memory_behavior_context(
        &contract,
        MemoryStrategy::BoundedDecode,
        tier,
        MemoryBehaviorRoute {
            mode: MemoryMode::TextToImage,
            reference_count: 0,
            use_pid: false,
            has_phases: true,
            overlay: None,
        },
    )
    .expect("behavior context");
    context.geometry = MemoryGeometry {
        width: FIXTURE_WIDTH,
        height: FIXTURE_HEIGHT,
        batch: 1,
        frames: FIXTURE_FRAMES,
        reference_count: 0,
    };

    // The representative selection carries the published domain, not a hardcoded pair.
    assert_eq!(
        (
            context.selection.parameters.decode_tile_edge,
            context.selection.parameters.decode_overlap
        ),
        (Some(DECODE_TILE_EDGE), Some(DECODE_OVERLAP)),
        "the representative rung-2 selection must carry the published domain"
    );
    assert_eq!(
        safety_check(&spec, &contract, &context),
        MemorySafetyDecision::Accept,
        "rung 2 must admit its own published geometry"
    );

    let mut scope = registered_begin_request(&spec, &contract, &context)
        .expect("begin_request")
        .expect("scope");
    // The scope's own decode hook admits it — this is the seam registry conformance drives.
    scope
        .configure_decode(DECODE_TILE_EDGE, DECODE_OVERLAP, context.geometry)
        .expect("the published geometry must configure");
    // …and refuses a budgeted substitute, which is the direction that carries information.
    let refused = scope
        .configure_decode(128, 32, context.geometry)
        .expect_err("a smaller tile must be refused, not silently clamped");
    assert!(
        refused.to_string().contains("output-correctness"),
        "the refusal must name WHY the geometry is pinned, got: {refused}"
    );

    let mut request = GenerationRequest {
        prompt: "a slow pan across a rainy street at night".into(),
        width: FIXTURE_WIDTH,
        height: FIXTURE_HEIGHT,
        frames: Some(FIXTURE_FRAMES),
        ..Default::default()
    };
    scope.configure_request(&mut request).expect("configure");
    let memory = request
        .memory
        .expect("a rung-2 request carries a memory block");
    assert!(memory.tile_vae_decode, "rung 2 must set the decode signal");
    assert_eq!(
        (memory.decode_tile_edge, memory.decode_overlap),
        (Some(DECODE_TILE_EDGE), Some(DECODE_OVERLAP)),
        "the selected geometry must survive onto the request"
    );

    // The pipeline reader accepts the block the scope built, and produces the reference
    // geometry — the same one `decode_clip` executes.
    let tiling = crate::pipeline::decode_tiling_for(&request).expect("admitted block resolves");
    assert!(tiling.enabled);
    assert_eq!(
        (tiling.tile_height, tiling.tile_width),
        (DECODE_TILE_EDGE as i32, DECODE_TILE_EDGE as i32)
    );
    assert_eq!(
        (tiling.overlap_height, tiling.overlap_width),
        (DECODE_OVERLAP as i32, DECODE_OVERLAP as i32)
    );
    scope.finish(MemoryRunOutcome::Complete).expect("finish");
}

/// **The reader is not a constant folder.** A hand-built block naming an unadmitted geometry —
/// including the starved zero overlap — must be refused by the same predicate admission uses.
///
/// This is the half that goes red if `decode_tiling_for` is replaced by
/// `Ok(SpatialTiling::default())`: every positive assertion above would still pass, because the
/// admitted value *is* the default.
#[test]
fn the_pipeline_reader_refuses_every_unadmitted_decode_geometry() {
    use mlx_gen::gen_core::GenerationMemory;
    let request = |edge: Option<u32>, overlap: Option<u32>| GenerationRequest {
        prompt: "p".into(),
        width: FIXTURE_WIDTH,
        height: FIXTURE_HEIGHT,
        frames: Some(FIXTURE_FRAMES),
        memory: Some(GenerationMemory {
            tile_vae_decode: true,
            decode_tile_edge: edge,
            decode_overlap: overlap,
            ..Default::default()
        }),
        ..Default::default()
    };
    // A budgeted tile — the substitution that would reintroduce sc-18786's defect.
    for edge in [64u32, 128, 192, 320, 512, 2048] {
        assert!(
            crate::pipeline::decode_tiling_for(&request(Some(edge), Some(DECODE_OVERLAP))).is_err(),
            "a {edge}px tile must be refused: it changes the decode"
        );
    }
    // **A starved overlap**, the failure mode this rung must not be able to select.
    for overlap in [0u32, 16, 32, 128] {
        assert!(
            crate::pipeline::decode_tiling_for(&request(Some(DECODE_TILE_EDGE), Some(overlap)))
                .is_err(),
            "a {overlap}px overlap must be refused"
        );
    }
    // The admitted pair resolves, so the guard rejects the geometry and not the request shape.
    assert!(crate::pipeline::decode_tiling_for(&request(
        Some(DECODE_TILE_EDGE),
        Some(DECODE_OVERLAP)
    ))
    .is_ok());
    // A request that engages no rung, or names no geometry, gets the shipped default — the
    // pre-sc-18660 behaviour, byte for byte.
    for plain in [
        GenerationRequest::default(),
        GenerationRequest {
            memory: Some(GenerationMemory::default()),
            ..Default::default()
        },
    ] {
        assert_eq!(
            crate::pipeline::decode_tiling_for(&plain).expect("plain request resolves"),
            crate::spatial_tiling::SpatialTiling::default()
        );
    }
}

/// A zero overlap cannot be published, independent of the reader: `validate_ranges` forbids a
/// zero entry in any declared range. Two guards, one failure mode, neither relying on the other.
#[test]
fn a_starved_overlap_cannot_be_published_as_a_candidate() {
    let mut starved = declared();
    for entry in &mut starved.strategies {
        if entry.strategy == MemoryStrategy::BoundedDecode {
            entry.parameters.decode_overlaps = vec![0];
        }
    }
    let errors = starved.conformance_errors();
    assert!(
        !errors.is_empty(),
        "a zero decode overlap must fail conformance, not ship as a candidate"
    );
    // And the shipped domain is the geometry the decode actually runs — read back through
    // `SpatialTiling::default()`, which is what `decode_clip` consults, rather than off the
    // constants, since a constant compared with itself proves nothing.
    let tiling = crate::spatial_tiling::SpatialTiling::default();
    assert!(tiling.enabled);
    assert_eq!(
        (DECODE_TILE_EDGE as i32, DECODE_OVERLAP as i32),
        (tiling.tile_height, tiling.overlap_height),
        "the published domain must be the geometry decode_clip executes"
    );
    assert_eq!(
        (DECODE_TILE_EDGE as i32, DECODE_OVERLAP as i32),
        (tiling.tile_width, tiling.overlap_width),
        "the domain is square; a per-axis split would need two more candidates"
    );
    // The audio VAE's exclusion is a size-and-shape fact, not a preference: an order of
    // magnitude smaller, and a waveform has no spatial grid a tile edge could mean anything on.
    // Both operands are constants, so this is a compile-time guard — which is the stronger
    // form, since the premise cannot then drift into a runtime-only check nobody runs.
    const _: () = assert!(
        VIDEO_VAE_BYTES > AUDIO_VAE_BYTES * 10,
        "the audio-VAE exclusion assumes it is an order of magnitude smaller"
    );
}

/// Reachability has a negative half: a rung this provider does not implement must be refused by
/// the same admission path, not silently accepted.
#[test]
fn unimplemented_rungs_are_refused_at_admission() {
    let spec = weightless_spec();
    let contract = declared();
    // sc-18660 landed rung 2, so it is no longer in this list. The two remaining rungs are
    // unavailable for **different** reasons and the distinction is asserted rather than
    // flattened: rung 3 is measured-inapplicable (sc-18661), rung 4 is simply unbuilt
    // (sc-18662). Both must be refused identically at admission.
    match support(&contract, MemoryStrategy::BoundedAttention) {
        MemoryStrategySupport::StructurallyNotApplicable { reason } => {
            assert_eq!(
                reason,
                rung3_inapplicability_reason(),
                "the declared reason must be the one derived from the measurement"
            );
            assert!(
                reason.contains("4·B·H·S·D") && reason.contains("MLX-scoped"),
                "the reason must carry the measured prediction and its backend scope: {reason}"
            );
        }
        other => panic!("sc-18661 measured rung 3 inapplicable, got {other:?}"),
    }
    // sc-18662 landed rung 4, so it is no longer refused on a DEFERRED contract — its refusal
    // moved to the resident one, which `load_shape_follows_the_spec_because_both_loaders_now_exist`
    // owns. Rung 3 is the only rung this contract still refuses, and it refuses it for a
    // measured reason rather than an unbuilt one.
    assert_eq!(
        support(&contract, MemoryStrategy::BoundedTransformerResidency),
        MemoryStrategySupport::Implemented,
        "rung 4 is implemented on a DeferredMaterialization load (sc-18662)"
    );
    {
        let strategy = MemoryStrategy::BoundedAttention;
        assert!(
            contract
                .validate_selection(&MemorySelection {
                    strategy,
                    tier: mlx_gen::request_scope::loaded_tier(&spec),
                    parameters: MemoryStrategyParameters::default(),
                })
                .is_err(),
            "{strategy:?} must not be selectable"
        );
        assert!(
            !contract.engages(MemoryStrategy::StagedResidency, strategy),
            "{strategy:?} must not be engaged by a rung-1 selection"
        );
    }
}

/// The geometry gate is real: the shared fixture default this family had to override is
/// refused, and so is every other illegal shape — one mutation at a time, so each clause is
/// proven independently rather than as a set.
#[test]
fn each_geometry_clause_refuses_its_own_illegal_shape() {
    let spec = weightless_spec();
    let contract = declared();
    let legal = MemoryGeometry {
        width: FIXTURE_WIDTH,
        height: FIXTURE_HEIGHT,
        batch: 1,
        frames: FIXTURE_FRAMES,
        reference_count: 0,
    };
    let context = |geometry: MemoryGeometry, use_pid: bool| MemoryRunContext {
        reference_resources: None,
        // The optimized StagedResidency probe must clear the shared authority gate to reach
        // the geometry clauses under test; Calibrated matches the fixture's real calibration
        // handshake above.
        optimization_authority: mlx_gen::gen_core::MemoryOptimizationAuthority::Calibrated,
        selection: MemorySelection {
            strategy: MemoryStrategy::StagedResidency,
            tier: mlx_gen::request_scope::loaded_tier(&spec),
            parameters: MemoryStrategyParameters::default(),
        },
        calibration_abi: mlx_gen::gen_core::MEMORY_CALIBRATION_ABI,
        // sc-22737: read off the contract under test for exactly the reason `load_shape` is.
        // The identity is per-(artifact tier) now, and the weights-free declaration carries its
        // own key, so a context pinned to the production constant fails the handshake and makes
        // every geometry rejection below vacuous.
        calibration_fingerprint: contract
            .calibration
            .as_ref()
            .map(|identity| identity.fingerprint.clone())
            .unwrap_or_default(),
        // sc-18662: the contract's shape now follows the spec, so a context pinned to
        // `LOAD_SHAPE` would fail the calibration handshake on a deferred spec and make every
        // geometry rejection below vacuous. Read it off the contract under test.
        load_shape: contract.load_shape,
        mode: MemoryMode::TextToImage,
        has_reference: geometry.reference_count > 0,
        use_pid,
        has_phases: true,
        geometry,
        overlay: None,
        budget: MemoryBudget {
            total_bytes: 128 * 1024 * 1024 * 1024,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        predicted_peak_bytes: CONDITIONING_STAGE_PEAK_BYTES,
        cache_state: MemoryCacheState::Cold,
        evidence_revision: "unit-test".to_owned(),
    };

    assert_eq!(
        safety_check(&spec, &contract, &context(legal, false)),
        MemorySafetyDecision::Accept,
        "the control arm must be admitted, or every rejection below is vacuous"
    );

    // 1. the shared 1024x1024 single-frame default: over the canvas budget AND off-lattice.
    let shared_default = MemoryGeometry {
        width: 1024,
        height: 1024,
        frames: 1,
        ..legal
    };
    // 2. off-lattice frame count, canvas legal.
    let off_lattice = MemoryGeometry {
        frames: 125,
        ..legal
    };
    // 3. T=1, which does not render at all.
    let single_frame = MemoryGeometry { frames: 1, ..legal };
    // 4. stride violation, canvas and lattice legal.
    let off_stride = MemoryGeometry {
        width: 1330,
        ..legal
    };
    // 5. canvas budget, stride and lattice legal.
    let over_canvas = MemoryGeometry {
        width: 1344,
        height: 1024,
        ..legal
    };
    // 6. batch, everything else legal.
    let batched = MemoryGeometry { batch: 2, ..legal };

    for (name, geometry, use_pid) in [
        ("shared 1024x1024 T=1 default", shared_default, false),
        ("off-lattice frames", off_lattice, false),
        ("T=1", single_frame, false),
        ("off-stride width", off_stride, false),
        ("over-budget canvas", over_canvas, false),
        ("batch > 1", batched, false),
        ("PiD decode", legal, true),
    ] {
        assert!(
            matches!(
                safety_check(&spec, &contract, &context(geometry, use_pid)),
                MemorySafetyDecision::Reject { .. }
            ),
            "{name} must be refused"
        );
    }
}

// --- AC3: a malformed or incomplete contract fails, one mutation at a time -------------------

/// Each mutation is applied **alone** to a known-good contract. Mutating them together would
/// prove the set conforms, not that each guard independently detects its own breakage.
#[test]
fn each_contract_mutation_is_independently_detected() {
    assert!(
        gen_core_testkit::check_memory_strategy_contract(&declared()).is_ok(),
        "the shipped contract must conform, or every mutation below is vacuous"
    );

    let mutations: Vec<ContractMutation> = vec![
        (
            "a dropped strategy entry",
            Box::new(|c| {
                c.strategies
                    .retain(|entry| entry.strategy != MemoryStrategy::BoundedAttention);
            }),
        ),
        (
            "a duplicated strategy entry",
            Box::new(|c| {
                let first = c.strategies[0].clone();
                c.strategies.push(first);
            }),
        ),
        (
            "an empty StructurallyNotApplicable reason",
            Box::new(|c| {
                for entry in &mut c.strategies {
                    if entry.strategy == MemoryStrategy::BoundedDecode {
                        entry.support = MemoryStrategySupport::StructurallyNotApplicable {
                            reason: "   ".to_owned(),
                        };
                    }
                }
            }),
        ),
        (
            "StagedResidency without synchronized phase release",
            Box::new(|c| c.lifecycle.synchronized_phase_release = false),
        ),
        (
            "StagedResidency with a missing lifecycle phase",
            Box::new(|c| c.lifecycle.phases.retain(|p| *p != MemoryPhase::Decode)),
        ),
        (
            "BoundedDecode implemented with no tile domain",
            Box::new(|c| implement_without_range(c, MemoryStrategy::BoundedDecode)),
        ),
        (
            "BoundedAttention implemented with no chunk domain",
            Box::new(|c| implement_without_range(c, MemoryStrategy::BoundedAttention)),
        ),
        (
            "BoundedTransformerResidency implemented with no window domain",
            Box::new(|c| implement_without_range(c, MemoryStrategy::BoundedTransformerResidency)),
        ),
        (
            "base_bytes that does not equal its components",
            Box::new(|c| c.asset_facts.base_bytes += 1),
        ),
        (
            "a malformed calibration fingerprint",
            Box::new(|c| {
                c.calibration = Some(MemoryCalibrationIdentity::new("No_Version", LOAD_SHAPE));
            }),
        ),
        (
            "a calibration load shape that disagrees with the contract",
            // sc-18662: flipped RELATIVE to the contract rather than pinned to one shape. The
            // contract's shape now follows the spec, and a hardcoded
            // `DeferredMaterialization` here silently became the agreeing value — an inert
            // mutation that would have kept passing with the conformance rule deleted.
            Box::new(|c| {
                let opposite = match c.load_shape {
                    LoadShape::DeferredMaterialization => LoadShape::EagerMaterialization,
                    _ => LoadShape::DeferredMaterialization,
                };
                c.calibration = Some(MemoryCalibrationIdentity::new(
                    MEMORY_CALIBRATION_FINGERPRINT,
                    opposite,
                ));
            }),
        ),
        (
            "an overlay charged with no typed component",
            Box::new(|c| {
                c.asset_facts.overlay_bytes = 1;
                c.asset_facts.base_bytes = c.asset_facts.base_bytes.saturating_add(1);
            }),
        ),
    ];

    for (name, mutate) in mutations {
        let mut contract = declared();
        mutate(&mut contract);
        assert!(
            gen_core_testkit::check_memory_strategy_contract(&contract).is_err(),
            "conformance must reject {name}"
        );
    }
}

/// Flip one rung to `Implemented` while leaving its parameter domain empty — the exact shape
/// AC5 asks a test to catch.
fn implement_without_range(contract: &mut MemoryProviderContract, strategy: MemoryStrategy) {
    for entry in &mut contract.strategies {
        if entry.strategy == strategy {
            entry.support = MemoryStrategySupport::Implemented;
            entry.parameters = MemoryParameterRanges::default();
        }
    }
    match strategy {
        MemoryStrategy::BoundedDecode => contract.lifecycle.decode_tiling = true,
        MemoryStrategy::BoundedAttention => contract.lifecycle.attention_chunking = true,
        MemoryStrategy::BoundedTransformerResidency => {
            contract.lifecycle.transformer_window_materialization = true
        }
        _ => {}
    }
}

// --- AC5: parameter ranges are declared exactly where they are owned ------------------------

/// Both directions: an implemented lever publishes its domain, and a rung that does not own a
/// parameter publishes none.
///
/// Since sc-18660 the positive half is real rather than vacuous — rung 2 publishes exactly the
/// reference geometry — while every other rung is still legitimately empty.
#[test]
fn parameter_ranges_are_owned_by_the_rung_that_consumes_them() {
    let contract = declared();
    assert!(contract.conformance_errors().is_empty());
    for capability in &contract.strategies {
        if capability.strategy == MemoryStrategy::BoundedTransformerResidency {
            // sc-18662. Rung 4 owns the window domain on a deferred contract, and `declared()`
            // is resolved at `DeferredMaterialization`.
            assert_eq!(
                capability.parameters,
                MemoryParameterRanges {
                    transformer_window_sizes: vec![TRANSFORMER_WINDOW_SIZE],
                    transformer_window_components: vec![TransformerComponent::Both],
                    ..Default::default()
                },
                "rung 4 publishes the measured singleton window and both component arms"
            );
            continue;
        }
        if capability.strategy == MemoryStrategy::BoundedDecode {
            assert_eq!(
                capability.parameters,
                MemoryParameterRanges {
                    decode_tile_edges: vec![DECODE_TILE_EDGE],
                    decode_overlaps: vec![DECODE_OVERLAP],
                    ..Default::default()
                },
                "rung 2 publishes the reference geometry and nothing else"
            );
            continue;
        }
        assert_eq!(
            capability.parameters,
            MemoryParameterRanges::default(),
            "{:?} owns no numeric parameters until its rung lands",
            capability.strategy
        );
    }
    // And an implemented lever with an empty domain is an error, per rung.
    for strategy in [
        MemoryStrategy::BoundedDecode,
        MemoryStrategy::BoundedAttention,
        MemoryStrategy::BoundedTransformerResidency,
    ] {
        let mut mutated = declared();
        implement_without_range(&mut mutated, strategy);
        assert!(
            !mutated.conformance_errors().is_empty(),
            "{strategy:?} implemented with no MemoryParameterRanges must fail"
        );
    }
}

// --- AC4: asset facts, and the stage the other measured numbers belong to -------------------

/// Tolerance for the GB figures recorded on sc-18659, in bytes. The story quotes the text
/// encoder as 66.73 GB; the measured on-disk total is 66,714,912,872 B = 66.715 GB, so the byte
/// constants are authoritative and the GB figures are held only to this window.
const GB_TOLERANCE_BYTES: u64 = 20_000_000;

fn assert_within(measured: u64, story_gb: f64, what: &str) {
    let story_bytes = (story_gb * 1e9) as u64;
    let delta = measured.abs_diff(story_bytes);
    assert!(
        delta <= GB_TOLERANCE_BYTES,
        "{what}: measured {measured} B ({:.3} GB) is {delta} B from the recorded {story_gb} GB, \
             outside the {GB_TOLERANCE_BYTES} B tolerance",
        measured as f64 / 1e9
    );
}

/// The four measured on-disk footprints, held to the figures recorded on the story.
#[test]
fn measured_component_bytes_match_the_recorded_footprints() {
    assert_within(DIT_BF16_BYTES, 66.28, "33 B DiT partition at bf16");
    assert_within(TEXT_ENCODER_BYTES, 66.73, "Qwen3-VL-32B text encoder");
    assert_within(VIDEO_VAE_BYTES, 10.42, "video VAE");
    assert_within(AUDIO_VAE_BYTES, 0.61, "audio VAE");
}

/// The contract charges each measured component to the right field, resolved off a real
/// directory tree rather than from a constant.
#[test]
fn asset_facts_charge_each_component_to_its_own_field() {
    let root = tempfile::tempdir().expect("tempdir");
    sparse_snapshot(
        root.path(),
        &[
            ("text_encoder", TEXT_ENCODER_BYTES),
            ("transformer", DIT_BF16_BYTES),
            ("vae", VIDEO_VAE_BYTES),
            ("audio_vae", AUDIO_VAE_BYTES),
            // `transformer_ref` is byte-identical and a render loads exactly one partition, so
            // it must NOT be added on top.
            ("transformer_ref", DIT_BF16_BYTES),
        ],
    );
    // Audio VAE loads EVERY tensor as F32, including the fixture's U8 padding. Derive
    // its resident charge from actual header elements, not the sparse file's stat size.
    let audio_headers = mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(
        root.path().join("audio_vae"),
    )
    .expect("valid audio fixture headers");
    let audio_f32: u64 = audio_headers
        .iter()
        .map(|h| h.shape.iter().map(|&d| d as u64).product::<u64>() * 4)
        .sum();
    assert!(
        audio_f32 > AUDIO_VAE_BYTES,
        "U8 padding must materialize wider"
    );
    let contract =
        contract_for(&LoadSpec::new(WeightsSource::Dir(root.path().into()))).expect("contract");
    let facts = contract.asset_facts;
    assert_eq!(facts.conditioning_bytes, TEXT_ENCODER_BYTES);
    assert_eq!(facts.transformer_bytes, DIT_BF16_BYTES);
    assert_eq!(
        facts.decoder_bytes,
        VIDEO_VAE_BYTES + audio_f32,
        "both decoders are charged, and only the decoders"
    );
    assert_eq!(facts.overlay_bytes, 0);
    assert_eq!(
        facts.base_bytes,
        TEXT_ENCODER_BYTES + DIT_BF16_BYTES + VIDEO_VAE_BYTES + audio_f32
    );
    assert!(contract.conformance_errors().is_empty());
}

/// **A staged text encoder is charged at its own size** (sc-19120 / sc-20267).
///
/// The sibling of `a_staged_dit_component_is_charged_at_its_own_size`, one component over, and it
/// closes the same defect: `resolve` sized `root.join("text_encoder")` unconditionally, so a split
/// install — which sc-19120 made ordinary by publishing per-tier text encoders staged
/// **independently** of the DiT — was charged **zero** for the largest single component this
/// family loads. An under-declared floor admits a render that then OOMs.
///
/// The unstaged control is what stops this being vacuous: the snapshot really does carry no
/// `text_encoder/`, so a resolver that ignored the override would report 0 here.
#[test]
fn a_staged_text_encoder_is_charged_at_its_own_size() {
    // A split install: the root holds the DiT but no `text_encoder/` at all.
    let root = tempfile::tempdir().expect("tempdir");
    sparse_snapshot(root.path(), &[("transformer", DIT_BF16_BYTES)]);
    let staged = tempfile::tempdir().expect("tempdir");
    // The q4 text encoder's hosted size (`SceneWorks/minimax-h3-mlx` @ `137ce668`).
    const TE_Q4_BYTES: u64 = 18_722_713_964;
    sparse_snapshot(staged.path(), &[("text_encoder", TE_Q4_BYTES)]);

    let unstaged =
        contract_for(&LoadSpec::new(WeightsSource::Dir(root.path().into()))).expect("contract");
    assert_eq!(
        unstaged.asset_facts.conditioning_bytes, 0,
        "the snapshot genuinely carries no text encoder, so the staged case below is not vacuous"
    );

    let contract = contract_for(
        &LoadSpec::new(WeightsSource::Dir(root.path().into())).with_component(
            TEXT_ENCODER_COMPONENT,
            WeightsSource::Dir(staged.path().join(TEXT_ENCODER_COMPONENT)),
        ),
    )
    .expect("contract");
    assert_eq!(
        contract.asset_facts.conditioning_bytes, TE_Q4_BYTES,
        "a staged packed encoder must be charged, not measured as absent"
    );
    assert_eq!(
        contract.asset_facts.base_bytes,
        TE_Q4_BYTES + DIT_BF16_BYTES,
        "and it must reach the floor the formula reads"
    );
    assert!(
        contract.conformance_errors().is_empty(),
        "{:?}",
        contract.conformance_errors()
    );
}

/// A tiered install stages the DiT elsewhere; the contract must charge the staged tier's bytes,
/// not the flat snapshot's. This is the field that would silently report a bf16 floor for a q4
/// render.
#[test]
fn a_staged_dit_component_is_charged_at_its_own_size() {
    let root = tempfile::tempdir().expect("tempdir");
    sparse_snapshot(root.path(), &[("transformer", DIT_BF16_BYTES)]);
    let staged = tempfile::tempdir().expect("tempdir");
    const Q4_BYTES: u64 = 18_779_970_678;
    sparse_snapshot(staged.path(), &[("transformer", Q4_BYTES)]);

    let contract = contract_for(
        &LoadSpec::new(WeightsSource::Dir(root.path().into())).with_component(
            DIT_COMPONENT,
            WeightsSource::Dir(staged.path().join(DIT_COMPONENT)),
        ),
    )
    .expect("contract");
    assert_eq!(contract.asset_facts.transformer_bytes, Q4_BYTES);
    // The original blocker was a contract that resolved its asset facts to the staged tier while
    // still declaring the bf16 sub-stack against them — a sub-stack larger than the stack
    // containing it, which `Registry::memory_strategy_contract` turns into a q4 render that
    // cannot resolve a contract at all. Reading the field without grading the contract cannot
    // see that, so this is the assertion that would have caught it.
    assert!(
        contract.conformance_errors().is_empty(),
        "a staged tier must resolve a conformant contract: {:?}",
        contract.conformance_errors()
    );
    // This snapshot carries no `config.json`, so the marker leg is absent and the **footprint**
    // leg is what sized the sub-stack. Pinned so the two legs cannot be confused for one another:
    // the marker leg is driven separately by
    // `the_marker_leg_sizes_the_sub_stack_from_the_staged_tiers_own_quantization_block`.
    assert_eq!(
        contract.resident_components()[0].resident_bytes,
        Q4_FOOTPRINT_LEG_BYTES,
        "with no quantization marker the footprint leg must be what wins"
    );
}

/// The q4 sub-stack the **footprint** leg derives: `ADALN_EVICTED_BYTES · Q4_BYTES /
/// DIT_BF16_BYTES`.
///
/// It sits 0.648 % above the marker leg's exact 7,325,337,600 B because `crate::convert` leaves
/// the f32 I/O heads, `context_embedder` and the norms dense in every tier, so the whole-DiT
/// ratio shrinks slightly more slowly than the projections' own. That gap is the entire reason
/// `resolved_adaln_bytes` takes the **minimum** rather than either leg alone.
const Q4_FOOTPRINT_LEG_BYTES: u64 = 7_372_786_768;

/// The stage-attributed peaks, and the relationships between them that the epic measured. The
/// point of this test is the **attribution**: it fails if someone re-labels the ~53 GB floor as
/// the DiT's, or drops the tiering win the flat process-wide peak once hid.
///
/// Two layers, and both are needed. The **literals** pin each stage peak to the number that
/// stage was actually measured at; the **identities** pin the relationships the epic's argument
/// rests on. Identities alone are not sufficient: they are relative, so a consistent re-basing
/// of the constants — or an edit that moves a constant and its own identity literal together —
/// satisfies every one of them while silently re-writing a measurement. These are the three
/// numbers most exposed to the mis-attribution class, so they get the same literal anchor the
/// AdaLN figure already had.
#[test]
fn measured_peaks_stay_attributed_to_the_stage_that_produced_them() {
    // Each stage peak against the literal it was measured at, independent of the others.
    assert_eq!(
        CONDITIONING_STAGE_PEAK_BYTES, 53_070_000_000,
        "conditioning stage: the dense Qwen3-VL-32B text encoder measured in isolation at \
             53.07 GB"
    );
    assert_eq!(
        DENOISE_RESIDENT_BF16_BYTES, 40_430_000_000,
        "denoise stage: bf16 DiT residency after the AdaLN precompute-and-evict, 40.43 GB"
    );
    assert_eq!(
        DENOISE_RESIDENT_Q4_BYTES, 11_630_000_000,
        "denoise stage: the same residency at q4, 11.63 GB"
    );
    // The floor is the conditioning stage, not the denoise one.
    const {
        assert!(
            CONDITIONING_STAGE_PEAK_BYTES > DENOISE_RESIDENT_BF16_BYTES,
            "the ~53 GB floor is the text encoder; the DiT's post-evict residency sits below it"
        )
    };
    assert_eq!(
        CONDITIONING_STAGE_PEAK_BYTES - DENOISE_RESIDENT_BF16_BYTES,
        12_640_000_000,
        "measured gap between the conditioning mark and bf16 denoise residency"
    );
    // The DiT is genuinely tiered: the win is real, it was only masked.
    assert_eq!(
        DENOISE_RESIDENT_BF16_BYTES - DENOISE_RESIDENT_Q4_BYTES,
        28_800_000_000,
        "bf16 -> q4 denoise-resident reduction"
    );
    // And the AdaLN eviction is the loader's own figure, not a contract-local copy.
    assert_eq!(
        ADALN_EVICTED_BYTES, 26_020_915_200,
        "the exact bytes crate::dit::adaln releases"
    );

    // --- the packed text-encoder tiers (sc-19120) ---------------------------------------
    assert_eq!(
        CONDITIONING_STAGE_PEAK_Q4_BYTES, 14_430_000_000,
        "conditioning stage on the packed q4 text-encoder tier, measured in isolation"
    );
    assert_eq!(
        CONDITIONING_STAGE_PEAK_Q8_BYTES, 26_940_000_000,
        "conditioning stage on the packed q8 text-encoder tier, measured in isolation"
    );
    assert_eq!(
        CONDITIONING_STAGE_PEAK_BYTES - CONDITIONING_STAGE_PEAK_Q4_BYTES,
        38_640_000_000,
        "dense -> q4 conditioning-stage reduction: the largest single memory change in this \
             model, and larger than the DiT's own 28.80 GB"
    );
    // The tier ladder is monotone and the stage really is tiered — a packed TE that measured
    // at or above the dense datum would mean the pack never engaged.
    const {
        assert!(
            CONDITIONING_STAGE_PEAK_Q4_BYTES < CONDITIONING_STAGE_PEAK_Q8_BYTES
                && CONDITIONING_STAGE_PEAK_Q8_BYTES < CONDITIONING_STAGE_PEAK_BYTES,
            "the conditioning stage must fall monotonically q4 < q8 < dense"
        )
    };
    // **What the pair still owes.** Packed q4 puts the conditioner (14.43 GB) within 2.8 GB of
    // the q4 DiT's own denoise residency (11.63 GB), so the floor is now the DiT's — which
    // sc-18662's bounded transformer residency is what bounds. Neither story reaches a
    // consumer-GPU footprint alone, and this assertion is the reason why, in executable form.
    const {
        assert!(
            CONDITIONING_STAGE_PEAK_Q4_BYTES > DENOISE_RESIDENT_Q4_BYTES,
            "if the packed conditioner ever fell below the q4 denoise residency, the DiT would \
                 be the sole floor and this comment would be stale"
        )
    };
    const {
        assert!(
            ADALN_EVICTED_BYTES < DIT_BF16_BYTES,
            "the evicted projections are part of the DiT partition"
        )
    };
}

// --- sc-18665: the AdaLN evict as a typed intra-transformer exclusion -----------------------

/// The declared component, resolved off the shipped contract rather than off `adaln_component`,
/// so every assertion below travels the same seam a consumer does.
fn adaln_declaration() -> mlx_gen::gen_core::MemoryResidentComponent {
    let components = declared().resident_components().to_vec();
    assert_eq!(
        components.len(),
        1,
        "the contract declares exactly one resident component: the AdaLN sub-stack"
    );
    assert_eq!(components[0].id, ADALN_COMPONENT_ID);
    components[0].clone()
}

/// **AC1.** The 26.02 GB drop is in the declared formula, and the steady-state transformer
/// charge is lower than the load-exact one by exactly the net exclusion.
///
/// Two literals, because a ratio alone would survive a consistent re-basing: the gross figure is
/// the loader's own [`ADALN_EVICTED_BYTES`] and the retained figure is
/// [`ADALN_MODULATION_TABLE_MAX_BYTES`]. Deleting the component, or flipping the formula back to
/// a plain `PhaseEnvelope`, takes the delta to zero and this red.
#[test]
fn the_declared_formula_carries_the_adaln_exclusion_as_a_typed_sub_stack() {
    let root = tempfile::tempdir().expect("tempdir");
    sparse_snapshot(root.path(), &[("transformer", DIT_BF16_BYTES)]);
    let contract =
        contract_for(&LoadSpec::new(WeightsSource::Dir(root.path().into()))).expect("contract");
    assert!(contract.conformance_errors().is_empty());

    let component = adaln_declaration();
    assert_eq!(component.resident_bytes, ADALN_EVICTED_BYTES);
    assert_eq!(
        component.steady_state_bytes(),
        ADALN_MODULATION_TABLE_MAX_BYTES
    );
    assert_eq!(
        component.kind,
        MemoryComponentKind::TransformerSubStack(TransformerComponent::Dit),
        "these bytes are INSIDE transformer_bytes; a whole-Transformer kind charges them twice"
    );

    // The number the ladder's arithmetic sees change.
    let net = ADALN_EVICTED_BYTES - ADALN_MODULATION_TABLE_MAX_BYTES;
    assert_eq!(contract.evicted_component_bytes(), net);
    assert_eq!(
        contract.asset_facts.transformer_bytes - contract.steady_state_transformer_bytes(),
        net,
        "the post-precompute charge must be the load-exact one minus the net exclusion"
    );
    assert_eq!(
        contract.steady_state_transformer_bytes(),
        DIT_BF16_BYTES - net
    );
    // The exclusion is not free, and the contract says by how much.
    assert!(
        net < ADALN_EVICTED_BYTES,
        "declaring the gross {ADALN_EVICTED_BYTES} B would claim a saving the precompute does \
             not deliver: it keeps a modulation table in the projections' place"
    );
    // A sub-stack must not leak into the auxiliary legs — those are for networks beside the
    // base model, and `overlay_bytes` would charge these bytes a second time.
    assert_eq!(contract.auxiliary_resident_bytes(), 0);
    assert_eq!(contract.asset_facts.overlay_bytes, 0);
    assert_eq!(
        contract.total_resident_bytes(),
        contract.asset_facts.base_bytes
    );
}

/// **AC1's negative half.** With the component removed the delta collapses to zero, so the
/// assertions above are grading the declaration rather than an accessor's arithmetic.
#[test]
fn removing_the_component_takes_the_declared_exclusion_to_zero() {
    let mut stripped = declared();
    let MemoryFormulaKind::ComponentPhaseEnvelope {
        phases,
        variables,
        resident_components,
    } = stripped.formula.clone()
    else {
        panic!("the shipped formula is a component envelope");
    };
    assert!(!resident_components.is_empty());
    stripped.formula = MemoryFormulaKind::PhaseEnvelope { phases, variables };
    assert_eq!(stripped.evicted_component_bytes(), 0);
    assert_ne!(declared().evicted_component_bytes(), 0);

    // The steady-state half must be graded on a **resolved** contract. `declared()` is the
    // weights-free one, whose `transformer_bytes` is 0, so asserting the two equal there is
    // `0 == 0` — it passes whether or not the accessor consulted the formula at all.
    let mut resolved = staged_contract(DIT_BF16_BYTES, None);
    let net = ADALN_EVICTED_BYTES - ADALN_MODULATION_TABLE_MAX_BYTES;
    assert_ne!(resolved.asset_facts.transformer_bytes, 0);
    assert_eq!(
        resolved.steady_state_transformer_bytes(),
        DIT_BF16_BYTES - net,
        "the resolved contract charges the post-precompute steady state"
    );
    let MemoryFormulaKind::ComponentPhaseEnvelope {
        phases, variables, ..
    } = resolved.formula.clone()
    else {
        panic!("the shipped formula is a component envelope");
    };
    resolved.formula = MemoryFormulaKind::PhaseEnvelope { phases, variables };
    assert_eq!(
        resolved.steady_state_transformer_bytes(),
        DIT_BF16_BYTES,
        "a plain PhaseEnvelope has nowhere to record the drop — this is the pre-sc-18665 state, \
             and the delta against the assertion above is the whole of what sc-18665 adds"
    );
    assert_eq!(resolved.evicted_component_bytes(), 0);
}

/// A contract resolved off a staged DiT of `dit_bytes`, optionally carrying a real packed
/// `quantization` marker of `bits`.
///
/// Those are exactly the two inputs [`resolved_adaln_bytes`] reads, and the marker is written as
/// a genuine `config.json` in the staged component directory — the same file
/// `mlx_gen::quant::packed_quant_bits_at` parses on the load path. Nothing here reaches the bit
/// width through a test-only seam, which is the point: before this helper existed no test wrote a
/// `quantization` block at all, so `packed_quant_bits_at` always returned `Ok(None)` and the
/// marker leg was never executed.
fn staged_contract(dit_bytes: u64, bits: Option<i32>) -> MemoryProviderContract {
    let root = tempfile::tempdir().expect("tempdir");
    sparse_snapshot(root.path(), &[("transformer", DIT_BF16_BYTES)]);
    let staged = tempfile::tempdir().expect("tempdir");
    sparse_snapshot(staged.path(), &[("transformer", dit_bytes)]);
    if let Some(bits) = bits {
        std::fs::write(
            staged.path().join(DIT_COMPONENT).join("config.json"),
            format!(
                "{{\"quantization\":{{\"bits\":{bits},\"group_size\":{}}}}}",
                crate::quant::GROUP_SIZE
            ),
        )
        .expect("quantization marker");
    }
    contract_for(
        &LoadSpec::new(WeightsSource::Dir(root.path().into())).with_component(
            DIT_COMPONENT,
            WeightsSource::Dir(staged.path().join(DIT_COMPONENT)),
        ),
    )
    .expect("contract")
}

/// The staged q4 DiT partition's measured on-disk footprint — 18.78 GB.
const Q4_DIT_BYTES: u64 = 18_779_970_678;

/// The three published tier figures are [`adaln_stack_bytes`]'s own output at the three shipped
/// widths, derived from the shipped configuration at the group size the **loader** loads at.
///
/// Before this guard, [`ADALN_EVICTED_Q8_BYTES`] and [`ADALN_EVICTED_Q4_BYTES`] were referenced
/// by nothing but their own doc comments: two 8- and 13-digit literals no execution could
/// contradict. The bindings here are the ones that actually bite —
///
/// * the constants against the function, so a re-typed digit reds;
/// * the function's three **inputs** against `MiniMaxH3DitConfig::default()` and
///   [`crate::quant::GROUP_SIZE`], so a config or group-size change moves the constants or reds
///   (the function reads them, so grading its output alone would not see a config edit);
/// * the packed figures against the **code-buffer-only** total, which is the distinct claim the
///   docs make at [`ADALN_EVICTED_Q8_BYTES`]: the published figure carries the per-group bf16
///   scale and bias that `crate::quant::nbytes` sums and the packed triple cannot run without,
///   so it must NOT equal the 13.01 GB code buffer.
///
/// The bf16 arm is anchored independently of all of this: `ADALN_EVICTED_BYTES` is asserted
/// against the sum over the 50 real `adaln_proj` tensors by
/// `tests/adaln_evict_real_weights.rs`, so binding the packed arms to the same function carries
/// that anchor onto them.
#[test]
fn the_packed_tier_stack_sizes_are_the_loaders_own_accounting() {
    let config = crate::dit::MiniMaxH3DitConfig::default();
    let out = config.adaln_out_features() as u64;
    let inp = config.time_embed_dim as u64;

    // The derivation's inputs, pinned to the shipped configuration and the loader's group size.
    assert_eq!(out, 96_768, "6 modulation params x 3 x hidden_size 5376");
    assert_eq!(inp, 2688, "time_embed_dim");
    assert_eq!(DIT_BLOCKS as i32, config.num_layers);
    assert_eq!(
        crate::quant::GROUP_SIZE,
        64,
        "adaln_stack_bytes charges one bf16 scale and one bf16 bias per input group; a loader \
             that packed at a different group size would hold a different number of them"
    );

    // The three published figures ARE this function's output.
    assert_eq!(adaln_stack_bytes(16), ADALN_EVICTED_BYTES);
    assert_eq!(adaln_stack_bytes(8), ADALN_EVICTED_Q8_BYTES);
    assert_eq!(adaln_stack_bytes(4), ADALN_EVICTED_Q4_BYTES);

    // The packed figures count the group metadata, and that is what separates them from the
    // code-buffer figure `crate::quant` quotes.
    for (bits, published) in [(8_i32, ADALN_EVICTED_Q8_BYTES), (4, ADALN_EVICTED_Q4_BYTES)] {
        let code_only = DIT_BLOCKS as u64 * out * inp * bits as u64 / 8;
        assert!(
            published > code_only,
            "the q{bits} figure must carry the per-group scales and biases on top of the \
                 {code_only} B code buffer, or it is not what is resident"
        );
    }
    assert_eq!(
        DIT_BLOCKS as u64 * out * inp,
        13_005_619_200,
        "the 13.01 GB code buffer the crate::quant docs quote for q8"
    );
    assert_ne!(
        ADALN_EVICTED_Q8_BYTES, 13_005_619_200,
        "the published q8 figure is the resident triple, not the code buffer alone"
    );

    // Monotone in the tier, and every packed tier strictly below the bf16 stack it replaces.
    const {
        assert!(
            ADALN_EVICTED_Q4_BYTES < ADALN_EVICTED_Q8_BYTES
                && ADALN_EVICTED_Q8_BYTES < ADALN_EVICTED_BYTES,
            "the sub-stack must fall monotonically q4 < q8 < bf16"
        )
    };
    // `packed_quant_bits_at` admits only 4 and 8, so those are the only packed widths this
    // function is ever called at; 16 is the dense arm.
    assert_eq!(adaln_stack_bytes(16), adaln_stack_bytes(32));
}

/// **The marker leg, driven.** The staged tier's own `config.json` `quantization` block sizes
/// the sub-stack, at q4 **and** q8.
///
/// This closes the coverage hole that made the tier fix unproven where it matters: no test wrote
/// a `quantization` block, so `mlx_gen::quant::packed_quant_bits_at` always returned `Ok(None)`,
/// `adaln_stack_bytes(bits < 16)` never executed, and the q4 case passed on the **footprint**
/// leg instead. Both legs are now driven against the same footprint, and they differ, so the
/// assertions grade which leg ran rather than only what it returned.
///
/// **The q8 cell is deliberately stated as a leg property, not as a shipped-artifact figure.**
/// No q8 DiT on-disk footprint constant exists in this repo, so which leg wins for the real q8
/// artifact cannot be settled by reading — it needs a q8 snapshot. What is settled here is the
/// crossover: above a staged footprint of 35.22 GB the q8 marker leg wins and below it the
/// footprint leg does, and both sides are executed. The shipped q8 artifact's own size is not
/// claimed in either direction.
#[test]
fn the_marker_leg_sizes_the_sub_stack_from_the_staged_tiers_own_quantization_block() {
    // --- q4: the shipped tier, where the marker leg genuinely wins ---------------------------
    let marked = staged_contract(Q4_DIT_BYTES, Some(4));
    assert!(
        marked.conformance_errors().is_empty(),
        "a marked q4 tier must resolve a conformant contract: {:?}",
        marked.conformance_errors()
    );
    assert_eq!(
        marked.resident_components()[0].resident_bytes,
        ADALN_EVICTED_Q4_BYTES,
        "the q4 marker leg's exact stack must win over the footprint leg"
    );
    // The same footprint with the marker removed resolves HIGHER, which is what proves the
    // marker leg is what decided the number above rather than the footprint leg agreeing by
    // coincidence.
    assert_eq!(
        staged_contract(Q4_DIT_BYTES, None).resident_components()[0].resident_bytes,
        Q4_FOOTPRINT_LEG_BYTES
    );
    const {
        assert!(
            ADALN_EVICTED_Q4_BYTES < Q4_FOOTPRINT_LEG_BYTES,
            "the marker leg is exact and the footprint leg over-declares by the dense f32 \
                 heads and norms crate::convert leaves in every tier"
        )
    };

    // --- q8: both sides of the crossover, executed ------------------------------------------
    // Above the crossover the marker leg wins, so `adaln_stack_bytes(8)` is what produced this.
    let q8_marked = staged_contract(DIT_BF16_BYTES, Some(8));
    assert!(
        q8_marked.conformance_errors().is_empty(),
        "a marked q8 tier must resolve a conformant contract: {:?}",
        q8_marked.conformance_errors()
    );
    assert_eq!(
        q8_marked.resident_components()[0].resident_bytes,
        ADALN_EVICTED_Q8_BYTES
    );
    // Below it the footprint leg wins, which is the `min` doing its job rather than the marker
    // leg simply always being taken.
    let q8_small = staged_contract(20_000_000_000, Some(8));
    assert_eq!(
        q8_small.resident_components()[0].resident_bytes,
        7_851_755_356,
        "under the crossover the scaled footprint is the smaller, and safety takes the minimum"
    );
    assert!(q8_small.conformance_errors().is_empty());
    // u128: the numerator is ~9.2e20 and overflows u64, the same reason `resolved_adaln_bytes`
    // widens its own footprint-leg multiply.
    assert_eq!(
        ADALN_EVICTED_Q8_BYTES as u128 * DIT_BF16_BYTES as u128 / ADALN_EVICTED_BYTES as u128 + 1,
        35_223_071_970,
        "the staged footprint above which the q8 marker leg wins; the shipped q8 artifact's own \
             footprint is not recorded in this repo, so which side it falls on is UNDETERMINED"
    );

    // An unreadable marker must not be mistaken for a dense tier: `packed_quant_bits_at` errors,
    // `resolved_adaln_bytes` falls back to the bf16 figure, and the footprint leg is what keeps
    // that from declaring 26 GB against an 18.78 GB DiT.
    let root = tempfile::tempdir().expect("tempdir");
    sparse_snapshot(root.path(), &[("transformer", DIT_BF16_BYTES)]);
    let staged = tempfile::tempdir().expect("tempdir");
    sparse_snapshot(staged.path(), &[("transformer", Q4_DIT_BYTES)]);
    std::fs::write(
        staged.path().join(DIT_COMPONENT).join("config.json"),
        "{ not json",
    )
    .expect("damaged marker");
    let damaged = contract_for(
        &LoadSpec::new(WeightsSource::Dir(root.path().into())).with_component(
            DIT_COMPONENT,
            WeightsSource::Dir(staged.path().join(DIT_COMPONENT)),
        ),
    )
    .expect("contract");
    assert_eq!(
        damaged.resident_components()[0].resident_bytes,
        Q4_FOOTPRINT_LEG_BYTES,
        "a damaged marker must fall through to the footprint leg, never to the bf16 constant"
    );
    assert!(damaged.conformance_errors().is_empty());
}

/// Every shipped tier resolves an eviction that still excludes something, and the floor below
/// which it would not is real rather than rhetorical.
///
/// `conformance_errors` refuses a `PrecomputedThenEvicted` component whose retained table is at
/// or above its resident bytes — the eviction would declare a saving of zero or less. The scaled
/// footprint leg falls below [`ADALN_MODULATION_TABLE_MAX_BYTES`] once the resolved DiT drops
/// under ~9.86 GB, so that floor is a property of the arithmetic and not a hypothetical. The
/// shipped tiers sit clear of it: q4's 18.78 GB is 1.90x above.
#[test]
fn the_shipped_tiers_sit_clear_of_the_floor_where_the_eviction_stops_excluding_anything() {
    for (what, dit_bytes, bits, expected) in [
        ("bf16", DIT_BF16_BYTES, None, ADALN_EVICTED_BYTES),
        ("q4", Q4_DIT_BYTES, Some(4), ADALN_EVICTED_Q4_BYTES),
        ("q8", DIT_BF16_BYTES, Some(8), ADALN_EVICTED_Q8_BYTES),
    ] {
        let contract = staged_contract(dit_bytes, bits);
        let component = &contract.resident_components()[0];
        assert_eq!(component.resident_bytes, expected, "{what}");
        assert!(
            ADALN_MODULATION_TABLE_MAX_BYTES < component.resident_bytes,
            "{what}: the retained table must stay strictly below the resident stack, or the \
                 declared eviction excludes nothing"
        );
        assert!(
            component.resident_bytes <= contract.asset_facts.transformer_bytes,
            "{what}: a sub-stack cannot exceed the stack that contains it"
        );
        assert!(
            contract.conformance_errors().is_empty(),
            "{what}: {:?}",
            contract.conformance_errors()
        );
    }

    // The floor is real: a DiT under it resolves a sub-stack below the retained table, and
    // conformance says exactly that rather than passing quietly.
    const FLOOR_DIT_BYTES: u64 = 9_859_502_300;
    assert_eq!(
        ADALN_MODULATION_TABLE_MAX_BYTES as u128 * DIT_BF16_BYTES as u128
            / ADALN_EVICTED_BYTES as u128,
        FLOOR_DIT_BYTES as u128,
        "below this resolved DiT footprint the scaled stack falls under the retained table"
    );
    let starved = staged_contract(9_000_000_000, None);
    assert_eq!(
        starved.resident_components()[0].resident_bytes,
        3_533_289_910
    );
    let errors = starved.conformance_errors();
    assert!(
        errors.iter().any(|e| e.contains("excludes nothing")),
        "a sub-stack under the retained table must be refused, got {errors:?}"
    );
    // ...and the shipped tier's margin above that floor, stated as a number.
    const {
        assert!(
            Q4_DIT_BYTES > FLOOR_DIT_BYTES * 19 / 10,
            "the shipped q4 DiT must sit at least 1.90x above the floor"
        )
    };
}

/// **AC2.** The saving the contract claims may not exceed the saving the runtime was measured
/// to deliver, and may not fall short of it by more than the table the precompute retains.
///
/// The tolerance is [`ADALN_MODULATION_TABLE_MAX_BYTES`] — a quantity the schedule domain fixes,
/// pinned to the real schedule by `the_retained_table_is_the_worst_case_over_the_admitted_schedule`
/// — not a window sized to fit the numbers. Both directions carry information:
///
/// * **upper** — declaring the gross [`ADALN_EVICTED_BYTES`] as if the evict were free puts the
///   claim 170_410_984 B above the measured drop, and this red. That is the overstatement the
///   story's own measurement warned about.
/// * **lower** — declaring no eviction, or retaining more than the table, puts the claim below
///   the measured drop by more than the table, and this red.
#[test]
fn the_declared_saving_does_not_exceed_the_measured_residency_drop() {
    let contract = declared();
    let claimed = contract.evicted_component_bytes();
    // What the measurement says the drop was worth: the load-exact DiT against the residency
    // measured through the denoise loop, after the precompute-and-evict.
    let measured_drop = DIT_BF16_BYTES - DENOISE_RESIDENT_BF16_BYTES;

    assert!(
        claimed <= measured_drop,
        "the contract claims a {claimed} B saving against a measured {measured_drop} B drop \
             ({} B over). A declaration may under-claim; over-claiming turns a suppressed \
             configuration into an OOM",
        claimed.saturating_sub(measured_drop)
    );
    assert!(
        claimed + ADALN_MODULATION_TABLE_MAX_BYTES >= measured_drop,
        "the contract claims only {claimed} B of a measured {measured_drop} B drop; the \
             shortfall {} B exceeds the {ADALN_MODULATION_TABLE_MAX_BYTES} B modulation table, \
             which is the only thing that is supposed to account for it",
        measured_drop.saturating_sub(claimed)
    );

    // The gross figure is bracketed the same way, which is the sharper statement: the whole
    // disagreement between the projections' exact bytes and the measured residency drop is the
    // retained table, and nothing else.
    assert!(
        ADALN_EVICTED_BYTES >= measured_drop,
        "the evict cannot have released fewer bytes ({ADALN_EVICTED_BYTES}) than the residency \
             it caused to drop ({measured_drop})"
    );
    assert!(
        ADALN_EVICTED_BYTES - measured_drop <= ADALN_MODULATION_TABLE_MAX_BYTES,
        "{} B of the eviction is unaccounted for once the retained table is allowed for",
        ADALN_EVICTED_BYTES - measured_drop
    );

    // --- the same bracket on the q4 tier ------------------------------------------------------
    //
    // Grading bf16 alone leaves the tier the story exists to fix ungraded, and the q4 bracket is
    // the *tight* one: the shortfall against the measured q4 drop is 175,366,922 B, i.e. the
    // retained table covers it with only 4.5 % of itself to spare, where bf16 has 22.15 GB of
    // slack. A tier-scaling error that bf16 cannot see lands here.
    let q4 = staged_contract(Q4_DIT_BYTES, Some(4));
    let q4_claimed = q4.evicted_component_bytes();
    assert_eq!(
        q4_claimed,
        ADALN_EVICTED_Q4_BYTES - ADALN_MODULATION_TABLE_MAX_BYTES,
        "the q4 claim is the resolved stack net of the table, which does NOT scale with the tier"
    );
    let q4_measured_drop = Q4_DIT_BYTES - DENOISE_RESIDENT_Q4_BYTES;
    assert!(
        q4_claimed <= q4_measured_drop,
        "q4: the contract claims {q4_claimed} B against a measured {q4_measured_drop} B drop"
    );
    assert!(
        q4_claimed + ADALN_MODULATION_TABLE_MAX_BYTES >= q4_measured_drop,
        "q4: the shortfall {} B exceeds the {ADALN_MODULATION_TABLE_MAX_BYTES} B table",
        q4_measured_drop.saturating_sub(q4_claimed)
    );
    assert_eq!(
        q4_claimed + ADALN_MODULATION_TABLE_MAX_BYTES - q4_measured_drop,
        175_366_922,
        "the q4 margin, pinned: this is the tightest cell in the table and the one a tier \
             mis-scaling moves first"
    );
}

/// **AC2's constants are bound to the code, not typed twice.**
///
/// [`ADALN_EVICTED_BYTES`] is re-derived from the shipped DiT configuration the way
/// `crate::dit::adaln` derives it — `num_layers · (out_features · time_embed_dim + out_features)`
/// at 2 B — so a config change moves both together or this goes red. And
/// [`ADALN_MODULATION_TABLE_MAX_BYTES`] is read off a **real** [`crate::model::MAX_STEPS`]
/// schedule rather than from a row count guessed by hand.
#[test]
fn the_retained_table_is_the_worst_case_over_the_admitted_schedule() {
    let config = crate::dit::MiniMaxH3DitConfig::default();
    let out = config.adaln_out_features() as u64;
    let derived = DIT_BLOCKS as u64 * (out * config.time_embed_dim as u64 + out) * 2;
    assert_eq!(
        derived, ADALN_EVICTED_BYTES,
        "the declared eviction must be the shipped config's own projection bytes"
    );

    // `num_inference_steps` counts the terminal sigma = 0, at which the model is never
    // evaluated, so the longest admitted run is `MAX_STEPS + 1` inference steps.
    let longest = crate::denoise::JointSchedule::new(crate::model::MAX_STEPS as usize + 1)
        .expect("the longest admitted schedule");
    assert_eq!(longest.num_evals(), crate::model::MAX_STEPS as usize);
    let rows = crate::denoise::adaln_schedule(&longest)
        .expect("adaln schedule")
        .modulation_rows() as u64;
    let widest = crate::dit::config::MODULATION_PARAMS as u64
        * rows
        * config.hidden_size as u64
        * DIT_BLOCKS as u64
        * 2;
    assert_eq!(
        widest, ADALN_MODULATION_TABLE_MAX_BYTES,
        "the declared retained table must be the widest one the admitted schedule can produce"
    );

    // …and it really is the worst case: a shorter schedule keeps strictly less. Without this
    // the constant could be any figure at all and the equality above would still hold.
    let default = crate::denoise::JointSchedule::new(crate::model::DEFAULT_STEPS as usize + 1)
        .expect("the default schedule");
    let default_rows = crate::denoise::adaln_schedule(&default)
        .expect("adaln schedule")
        .modulation_rows() as u64;
    assert!(
        default_rows < rows,
        "the default schedule's {default_rows} rows must sit below the admitted maximum's \
             {rows}, or the declaration is not conservative"
    );
}

/// **AC5.** The evict is reachable on the real generate path, and the declaration is not merely
/// a statement about a mechanism nobody calls.
///
/// This constructs what it watches rather than describing it: `AdaLnResidency` is the enum
/// `crate::model::generate_impl` passes, and `AdaLnCache::precompute_and_evict` is the function
/// that releases the bytes. It refuses [`crate::dit::adaln::AdaLnResidency::Resident`]
/// **before** touching the block stack, which is why an empty stack reaches the residency guard
/// here and not the "empty stack" one — so the arm that keeps 26 GB resident cannot be selected
/// by accident. The end-to-end drop itself is measured by `tests/adaln_evict_memory.rs` and
/// `tests/adaln_evict_real_weights.rs` through `common::assert_adaln_phase_envelope`.
#[test]
fn the_declared_eviction_is_reachable_on_the_generate_path() {
    use crate::dit::adaln::{AdaLnCache, AdaLnResidency};

    let schedule =
        crate::denoise::adaln_schedule(&crate::denoise::JointSchedule::new(9).expect("joint"))
            .expect("adaln schedule");
    let refused =
        AdaLnCache::precompute_and_evict(&mut [], schedule, AdaLnResidency::Resident, |_| {
            unreachable!("the residency guard runs before the stack is touched")
        })
        .expect_err("Resident must not precompute");
    assert!(
        refused.to_string().contains("does not precompute"),
        "the Resident arm must refuse rather than silently evict: {refused}"
    );

    // The contract's evidence string names the tests that drive the positive direction, so a
    // reader can follow the declaration to an executable measurement.
    let MemoryComponentResidency::PrecomputedThenEvicted { evidence, .. } =
        adaln_declaration().residency
    else {
        panic!("the AdaLN component declares an eviction");
    };
    for named in ["adaln_evict_real_weights.rs", "assert_adaln_phase_envelope"] {
        assert!(
            evidence.contains(named),
            "the declared evidence must name {named}, got: {evidence}"
        );
    }
}

/// **AC6.** The evict is **unconditional**, not opt-in and not rung-selected, and the contract
/// says so in the only two places that can carry it.
///
/// `default_engagement_exclusions` is asserted empty on purpose: it is the seam the story
/// offered as an alternative, and it is not applicable. An exclusion removes a *rung* from a
/// selection's engaged composition — conformance requires both ends to be implemented
/// strategies — where this removes *bytes* from a formula. The last arm proves that is not
/// merely an opinion: attempting to express the eviction there is refused.
#[test]
fn adaln_eviction_is_unconditional_not_rung_selected() {
    let root = tempfile::tempdir().expect("tempdir");
    sparse_snapshot(root.path(), &[("transformer", DIT_BF16_BYTES)]);
    let contract =
        contract_for(&LoadSpec::new(WeightsSource::Dir(root.path().into()))).expect("contract");
    assert!(
        contract.default_engagement_exclusions.is_empty(),
        "an engagement exclusion removes a rung, not bytes"
    );
    assert_eq!(
        adaln_declaration().bounded_by,
        None,
        "no rung bounds this: crate::model passes PrecomputeAndEvict as a literal, so there is \
             no selection that keeps the projections loaded"
    );
    // The charge is the same whatever a request selects, including the plain resident rung —
    // which is what "unconditional" means, and would be false if the drop were a lever. Read
    // through `engaged_composition` so the loop covers compositions a caller can actually
    // select rather than an enum this accessor never sees.
    for strategy in MemoryStrategy::ALL {
        assert!(
            !contract
                .engaged_composition(strategy)
                .iter()
                .any(|rung| adaln_declaration().bounded_by == Some(*rung)),
            "{strategy:?} must not engage a rung that bounds the AdaLN stack — there is none"
        );
        assert_eq!(
            contract.steady_state_transformer_bytes(),
            DIT_BF16_BYTES - (ADALN_EVICTED_BYTES - ADALN_MODULATION_TABLE_MAX_BYTES),
            "{strategy:?} must not change the AdaLN charge"
        );
    }

    // And the alternative seam is genuinely closed rather than merely unused.
    let mut misdeclared = declared();
    misdeclared.default_engagement_exclusions.push(
        mlx_gen::gen_core::MemoryStrategyEngagementExclusion {
            selection: MemoryStrategy::StagedResidency,
            excluded_rung: MemoryStrategy::BoundedAttention,
            evidence: "the AdaLN evict".to_owned(),
        },
    );
    assert!(
        !misdeclared.conformance_errors().is_empty(),
        "an exclusion naming a rung this provider does not implement must fail conformance"
    );
}

// --- the declaration facts that are easy to get wrong ----------------------------------------

/// The load shape is pinned to the loader, not mirrored from the request. The spec asks for
/// `DeferredMaterialization` and this provider still reports `EagerMaterialization`, because
/// `MiniMaxH3Dit::load_dir` builds the whole block stack (sc-18662 changes that).
#[test]
fn load_shape_follows_the_spec_because_both_loaders_now_exist() {
    // sc-18662 inverted this test's subject, deliberately. It used to assert that the contract
    // PINNED `EagerMaterialization` whatever the spec asked, because `load_dir` was the only
    // loader and answering otherwise would declare a shape the provider did not have. Both
    // loaders exist now (`load_dir_deferred`, `MiniMaxH3TextEncoder::from_dir_deferred`), so the
    // honest answer is the spec's — and the test that has to exist is that the two shapes give
    // DIFFERENT contracts rather than one silently winning.
    // sc-22737: a weightless spec proves no artifact tier and so publishes no identity at all,
    // and the claim under test is about the identity's shape — so it needs a resolvable root.
    let root = tempfile::tempdir().expect("tempdir");
    sparse_snapshot(root.path(), &[("transformer", DIT_BF16_BYTES)]);
    write_tier_marker(root.path(), None);
    let deferred = contract_for(
        &LoadSpec::new(WeightsSource::Dir(root.path().into()))
            .with_load_shape(LoadShape::DeferredMaterialization),
    )
    .expect("contract");
    assert_eq!(deferred.load_shape, LoadShape::DeferredMaterialization);
    assert_eq!(
        deferred.calibration.as_ref().unwrap().load_shape,
        LoadShape::DeferredMaterialization,
        "the calibration identity must carry the shape it was resolved at, or an evidence \
             record would be matched against measurements of the other residency"
    );
    assert_eq!(
        support(&deferred, MemoryStrategy::BoundedTransformerResidency),
        MemoryStrategySupport::Implemented
    );

    let resident_spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
        .with_load_shape(LoadShape::EagerMaterialization);
    let resident = contract_for(&resident_spec).expect("contract");
    assert_eq!(resident.load_shape, LoadShape::EagerMaterialization);
    assert_eq!(resident.load_shape, LOAD_SHAPE);
    assert_eq!(
        support(&resident, MemoryStrategy::BoundedTransformerResidency),
        MemoryStrategySupport::Missing,
        "a resident load genuinely does not have rung 4's declared prerequisite"
    );
    assert!(!resident.lifecycle.transformer_window_materialization);
    assert!(deferred.lifecycle.transformer_window_materialization);
    let contract = deferred;
    // ...and the spec IS read: the asset facts come off it.
    let root = tempfile::tempdir().expect("tempdir");
    sparse_snapshot(root.path(), &[("vae", VIDEO_VAE_BYTES)]);
    let resolved =
        contract_for(&LoadSpec::new(WeightsSource::Dir(root.path().into()))).expect("contract");
    assert_eq!(resolved.asset_facts.decoder_bytes, VIDEO_VAE_BYTES);
    assert_eq!(contract.asset_facts.decoder_bytes, 0);
}

/// The rung-4 precondition is unmet today, and that is why rung 4 is `Missing` rather than a
/// declaration waiting on a flag.
#[test]
fn rung_four_publishes_both_component_arms_and_a_singleton_window() {
    // sc-18662. The predecessor asserted rung 4 was `Missing` for want of a deferred loader.
    let contract = declared();
    assert_eq!(contract.load_shape, LoadShape::DeferredMaterialization);
    assert!(contract.lifecycle.transformer_window_materialization);
    assert_eq!(
        support(&contract, MemoryStrategy::BoundedTransformerResidency),
        MemoryStrategySupport::Implemented
    );
    let ranges = &contract
        .capability(MemoryStrategy::BoundedTransformerResidency)
        .expect("rung 4")
        .parameters;
    // **`Both`, not `Dit`.** Declaring the DiT alone would leave the conditioning phase — the
    // taller stage at every tier — with no lever, which is what AC3 refuses. The TE arm is
    // measured separately: 53.07 -> 4.50 GB (dense).
    assert_eq!(
        ranges.transformer_window_components,
        vec![TransformerComponent::Both]
    );
    assert!(ranges.transformer_window_components[0].includes_dit());
    assert!(ranges.transformer_window_components[0].includes_text_encoder());
    // The window domain is a measured singleton, not a placeholder — see
    // `TRANSFORMER_WINDOW_SIZE`.
    assert_eq!(
        ranges.transformer_window_sizes,
        vec![TRANSFORMER_WINDOW_SIZE]
    );
    assert_eq!(TRANSFORMER_WINDOW_SIZE, 1);

    // **The declared-versus-measured guard is deliberately NOT here.** A ratio between two
    // `const`s is decided at compile time — clippy says so — and would prove nothing at run
    // time, which is the false-green shape this epic keeps refusing. It lives in
    // `tests/block_window_real.rs`, where the constants are checked against a fresh measurement
    // on real weights and go red when they drift.
}

/// `Precision::Fp32` is a different numeric tier and must not be admitted against a bf16 load.
#[test]
fn a_numeric_tier_mismatch_is_refused() {
    let spec = weightless_spec();
    let contract = declared();
    let mut context = mlx_gen::gen_core::standard_memory_behavior_context(
        &contract,
        MemoryStrategy::StagedResidency,
        mlx_gen::request_scope::loaded_tier(&spec),
        MemoryBehaviorRoute {
            mode: MemoryMode::TextToImage,
            reference_count: 0,
            use_pid: false,
            has_phases: true,
            overlay: None,
        },
    )
    .expect("context");
    context.geometry = MemoryGeometry {
        width: FIXTURE_WIDTH,
        height: FIXTURE_HEIGHT,
        batch: 1,
        frames: FIXTURE_FRAMES,
        reference_count: 0,
    };
    assert_eq!(
        safety_check(&spec, &contract, &context),
        MemorySafetyDecision::Accept
    );
    context.selection.tier.precision = Precision::Fp32;
    assert!(matches!(
        safety_check(&spec, &contract, &context),
        MemorySafetyDecision::Reject { .. }
    ));
}

/// The block count the request scope validates windows against is the DiT's real depth.
#[test]
fn declared_block_count_matches_the_dit_configuration() {
    assert_eq!(
        DIT_BLOCKS as i32,
        crate::dit::MiniMaxH3DitConfig::default().num_layers
    );
}

/// Every render route gets its own driven fixture — `t2va`, `fl2va` and `ref2va` are three
/// admission shapes, and `ref2va` reads a different 66 GB checkpoint.
#[test]
fn every_render_route_has_a_driven_fixture() {
    let spec = weightless_spec();
    let contract = declared();
    let fixtures =
        registered_fixture(&spec, &contract, MemoryStrategy::StagedResidency).expect("fixtures");
    let modes: Vec<String> = fixtures
        .iter()
        .map(|fixture| fixture.context.mode.as_key().to_owned())
        .collect();
    assert_eq!(modes, vec!["text_to_image", "image_to_image", "ref2va"]);
    for fixture in &fixtures {
        assert_eq!(fixture.context.geometry.frames, FIXTURE_FRAMES);
        assert_eq!(fixture.context.geometry.width, FIXTURE_WIDTH);
    }
    // A rung that is not implemented yields no fixture at all, rather than an empty-but-present
    // one the conformance harness would flag as a missing context.
    assert!(
        registered_fixture(&spec, &contract, MemoryStrategy::BoundedAttention)
            .expect("fixtures")
            .is_empty()
    );
}

/// sc-19591. The conformance walk feeds `fixture.request` back through `configure_request`,
/// which re-grades it against the geometry the same fixture admitted. This family's lattice
/// excludes 1, so a request left at `GenerationRequest::default()`'s `None` is refused as
/// off-lattice — see the `LEGAL_FRAME_COUNTS` gate in `safety_check`. The frame count reaches
/// the request from `context.geometry` through `gen_core::MemoryBehaviorFixture::new`; this
/// asserts that propagation actually reaches this provider, with no post-hoc override here.
#[test]
fn the_fixture_request_declares_the_lattice_frame_count() {
    // The premise: 1 is off this family's lattice, so the propagated value can never be
    // confused with the one-frame default it replaces.
    assert!(!LEGAL_FRAME_COUNTS.contains(&1));
    assert_eq!(GenerationRequest::default().frames, None);

    let spec = weightless_spec();
    let contract = declared();
    let fixtures =
        registered_fixture(&spec, &contract, MemoryStrategy::StagedResidency).expect("fixtures");
    assert!(!fixtures.is_empty(), "every render route drives a fixture");
    for fixture in &fixtures {
        assert_eq!(
            fixture.request.frames,
            Some(fixture.context.geometry.frames),
            "the request must restate the geometry it was admitted against"
        );
        assert_eq!(fixture.request.frames, Some(FIXTURE_FRAMES));
        assert!(LEGAL_FRAME_COUNTS.contains(&(fixture.request.frames.expect("frames") as i32)));
    }
}

// A header-only synthetic source. No downloaded table, tensor runtime, or accelerator is used.
fn pruning_memory_headers() -> serde_json::Value {
    let mut headers = serde_json::Map::new();
    let mut add = |name: String, dtype: &str, shape: Vec<usize>| {
        headers.insert(
            name,
            serde_json::json!({"dtype":dtype,"shape":shape,"data_offsets":[0,0]}),
        );
    };
    add("adaln_t_table".into(), "F32", vec![1025, 8]);
    add("body".into(), "F32", vec![257]);
    for layer in 0..2 {
        add(
            format!("transformer_blocks.{layer}.adaln_proj.linear.weight"),
            "BF16",
            vec![18 * 256, 8],
        );
        add(
            format!("transformer_blocks.{layer}.adaln_proj.linear.bias"),
            "F16",
            vec![18 * 256],
        );
    }
    add("norm_out.linear.weight".into(), "BF16", vec![2 * 256, 8]);
    add("norm_out.linear.bias".into(), "F16", vec![2 * 256]);
    serde_json::Value::Object(headers)
}

fn write_pruning_memory_headers(dir: &Path, mut headers: serde_json::Value) -> u64 {
    use std::io::Write;
    std::fs::create_dir_all(dir).unwrap();
    let mut offset = 0u64;
    for header in headers.as_object_mut().unwrap().values_mut() {
        let width = match header["dtype"].as_str().unwrap() {
            "F32" | "U32" => 4,
            "BF16" | "F16" => 2,
            _ => unreachable!(),
        };
        let size = header["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_u64().unwrap())
            .product::<u64>()
            * width;
        header["data_offsets"] = serde_json::json!([offset, offset + size]);
        offset += size;
    }
    let mut header = headers.to_string();
    header.extend(std::iter::repeat_n(' ', (8 - header.len() % 8) % 8));
    let path = dir.join("model.safetensors");
    let mut file = std::fs::File::create(&path).unwrap();

    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(header.as_bytes()).unwrap();
    let bytes = 8 + header.len() as u64 + offset;
    file.set_len(bytes).unwrap();
    let config = serde_json::json!({
        "_class_name":"MiniMaxH3Transformer3DModel", "hidden_size":256,"num_layers":2,
        "num_refiner_layers":2,"num_attention_heads":2,"attention_head_dim":128,
        "ffn_dim":256,"in_channels":24,"audio_in_channels":32,"patch_size":[1,2,2],
        "text_dim":5120,"freq_dim":256,"time_embed_hidden_dim":256,"time_embed_dim":64,
        "rope_freq_dim":16,"rope_theta":10000.0,"norm_eps":1e-5,"qk_norm_eps":1e-5,"final_norm_eps":1e-5
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    offset
}

// Independent dtype/shape arithmetic. Two generated streams contribute at most N values each;
// conditioned video can add one fixed clamp value and conditioned audio one fixed clean value.
fn expected_pruning_memory() -> (u64, u64, u64) {
    let table = 1025 * 8 * 4;
    let output_width = 256 * (18 * 2 + 2);
    let source = table + output_width * 8 * 2 + output_width * 2;
    let compute = table + output_width * 8 * 4 + output_width * 4;
    let retained = (2 * crate::model::MAX_STEPS as u64 + 2) * (output_width + 8) * 4;
    (source, compute, retained)
}

#[test]
fn source_pruned_native_mixed_checkpoint_never_earns_dense_eviction_credit() {
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join(crate::model::BASE_DIT_PARTITION);
    let stored = write_pruning_memory_headers(&dir, pruning_memory_headers());
    let other = tempfile::tempdir().unwrap();
    let (native_bytes, groups) =
        gen_core_testkit::h3_memory::write_native_component(other.path(), 2);
    std::fs::copy(
        other.path().join("model.safetensors"),
        dir.join("native.safetensors"),
    )
    .unwrap();
    let (source, compute, retained) = expected_pruning_memory();
    let shared = groups.iter().map(|g| g * g * 4).sum::<u64>();
    let contract = contract_for(&LoadSpec::new(WeightsSource::Dir(root.path().into()))).unwrap();
    assert_eq!(
        contract.asset_facts.transformer_bytes,
        stored - source + compute + retained + native_bytes + shared
    );
    assert_eq!(contract.evicted_component_bytes(), 0);
    let native = contract
        .phase_facts
        .as_ref()
        .unwrap()
        .native_quant
        .as_ref()
        .unwrap();
    assert!(native.tasks.iter().all(|t| t.evicted_bytes == 0));
    let rows = mlx_gen::gen_core::h3_memory::NativeRows {
        task: mlx_gen::gen_core::h3_memory::NativeTaskSelection::All,
        packed: 17,
        conditioning: 9,
        schedule: 5,
        visual_elements: 0,
        global: 0,
    };
    assert!(native.denoise_peak_bytes(rows) > contract.asset_facts.transformer_bytes);
}

#[test]
fn source_pruned_static_and_loaded_memory_plan_keeps_f32_operands_and_cache() {
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join(crate::model::BASE_DIT_PARTITION);
    let stored = write_pruning_memory_headers(&dir, pruning_memory_headers());
    let (source, compute, retained) = expected_pruning_memory();
    let sizing = source_prune_sizing(&dir).unwrap().unwrap();
    assert_eq!(
        (
            sizing.source_bytes,
            sizing.compute_bytes,
            sizing.retained_bytes
        ),
        (source, compute, retained)
    );
    let spec = LoadSpec::new(WeightsSource::Dir(root.path().into()));
    let contract = contract_for(&spec).unwrap();
    assert_eq!(
        contract.asset_facts.transformer_bytes,
        stored - source + compute + retained
    );
    assert_eq!(
        contract.evicted_component_bytes(),
        0,
        "reduced operands are not released by precompute"
    );
    assert_eq!(
        contract.steady_state_transformer_bytes(),
        contract.asset_facts.transformer_bytes
    );
    let adaln = &contract.resident_components()[0];
    assert_eq!(adaln.resident_bytes, compute + retained);
    assert_eq!(adaln.residency, MemoryComponentResidency::WholeRender);
    assert!(
        contract.conformance_errors().is_empty(),
        "{:?}",
        contract.conformance_errors()
    );
    assert_eq!(
        contract.asset_facts.base_bytes,
        contract.asset_facts.conditioning_bytes
            + contract.asset_facts.transformer_bytes
            + contract.asset_facts.decoder_bytes,
        "the typed sub-stack is not a second additive charge"
    );
    // Companions need only config probes for this metadata-only constructor; the selected
    // DiT already has its source headers. A zero-byte shard is malformed, not a placeholder.
    for (component, document) in [
        (crate::model::REFERENCE_DIT_PARTITION, "config.json"),
        ("text_encoder", "config.json"),
        ("vae", "config.json"),
        ("audio_vae", "config.json"),
        ("tokenizer", "tokenizer.json"),
        ("FL2VA/audio_vae", "config.json"),
        ("FL2VA/audio_vae", "config.yaml"),
        ("FL2VA/audio_vae", "metadata.json"),
    ] {
        let directory = root.path().join(component);
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join(document), "{}").unwrap();
    }
    let loaded = crate::model::load(&spec).unwrap();
    let loaded = loaded
        .memory_strategy_contract()
        .expect("loaded provider publishes actual contract");
    assert_eq!(loaded.asset_facts, contract.asset_facts);
    assert_eq!(loaded.evicted_component_bytes(), 0);
    let mut fp32 = spec.clone();
    fp32.precision = mlx_gen::Precision::Fp32;
    let wide = contract_for(&fp32).unwrap();
    assert_eq!(
        wide.asset_facts.transformer_bytes,
        257 * 4 + compute + retained,
        "a pre-widened materialization total must not charge widening twice"
    );
    assert_eq!(wide.evicted_component_bytes(), 0);
}

#[test]
fn source_pruned_sibling_override_wins_after_correction_without_dense_eviction_credit() {
    let root = tempfile::tempdir().unwrap();
    let staged = tempfile::tempdir().unwrap();
    let reference = staged.path().join(crate::model::REFERENCE_DIT_PARTITION);
    let stored = write_pruning_memory_headers(&reference, pruning_memory_headers());
    let (source, compute, retained) = expected_pruning_memory();
    let dense_gross = stored + retained / 2;
    sparse_snapshot(
        staged.path(),
        &[(crate::model::BASE_DIT_PARTITION, dense_gross)],
    );
    std::fs::write(
        staged
            .path()
            .join(crate::model::BASE_DIT_PARTITION)
            .join("config.json"),
        "{}",
    )
    .unwrap();
    // A much larger root-relative source must not override the actual staged task paths.
    sparse_snapshot(
        root.path(),
        &[(crate::model::BASE_DIT_PARTITION, DIT_BF16_BYTES)],
    );
    let spec = LoadSpec::new(WeightsSource::Dir(root.path().into())).with_component(
        crate::model::BASE_DIT_PARTITION,
        WeightsSource::Dir(staged.path().join(crate::model::BASE_DIT_PARTITION)),
    );
    let corrected = stored - source + compute + retained;
    assert!(
        stored < dense_gross && dense_gross < corrected,
        "the corrected peak winner differs from the gross-file winner"
    );
    let contract = contract_for(&spec).unwrap();
    assert_eq!(
        contract.asset_facts.transformer_bytes, corrected,
        "Ref2VA-only source pruning is priced at the actual staged sibling"
    );
    assert_eq!(contract.evicted_component_bytes(), 0);
    assert_eq!(contract.steady_state_transformer_bytes(), corrected);
    assert!(
        contract.conformance_errors().is_empty(),
        "{:?}",
        contract.conformance_errors()
    );
}

#[test]
fn source_pruned_missing_data_or_config_refuses_instead_of_stock_pricing() {
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join(crate::model::BASE_DIT_PARTITION);
    for missing in [
        "adaln_t_table",
        "transformer_blocks.0.adaln_proj.linear.bias",
        "norm_out.linear.weight",
    ] {
        let mut headers = pruning_memory_headers();
        headers.as_object_mut().unwrap().remove(missing);
        write_pruning_memory_headers(&dir, headers);
        assert!(
            contract_for(&LoadSpec::new(WeightsSource::Dir(root.path().into()))).is_err(),
            "{missing}"
        );
    }
    write_pruning_memory_headers(&dir, pruning_memory_headers());
    std::fs::remove_file(dir.join("config.json")).unwrap();
    assert!(contract_for(&LoadSpec::new(WeightsSource::Dir(root.path().into()))).is_err());
    std::fs::write(dir.join("config.json"), "not JSON").unwrap();
    assert!(contract_for(&LoadSpec::new(WeightsSource::Dir(root.path().into()))).is_err());
    std::fs::write(dir.join("model.safetensors"), b"broken header").unwrap();
    assert!(contract_for(&LoadSpec::new(WeightsSource::Dir(root.path().into()))).is_err());
}

#[test]
fn packed_q4_q8_headers_preserve_stock_contract_and_loaded_constructor() {
    for bits in [4usize, 8] {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join(crate::model::BASE_DIT_PARTITION);
        let mut headers = pruning_memory_headers();
        headers.as_object_mut().unwrap().remove("adaln_t_table");
        // Real MLX-affine code-column geometry, not an opaque stand-in. Large payload is
        // sparse: metadata only, never mapped/evaluated or downloaded by this test.
        headers["body"]["shape"] = serde_json::json!([3_000_000_000u64]);
        for layer in 0..2 {
            let prefix = format!("transformer_blocks.{layer}.adaln_proj.linear");
            headers[format!("{prefix}.weight")] = serde_json::json!({"dtype":"U32","shape":[18*256,64/(32/bits)],"data_offsets":[0,0]});
            headers[format!("{prefix}.scales")] =
                serde_json::json!({"dtype":"BF16","shape":[18*256,1],"data_offsets":[0,0]});
            headers[format!("{prefix}.biases")] =
                serde_json::json!({"dtype":"BF16","shape":[18*256,1],"data_offsets":[0,0]});
        }
        headers["norm_out.linear.weight"]["shape"] = serde_json::json!([2 * 256, 64]);
        for leaf in ["linear_1", "linear_2"] {
            for suffix in ["weight", "bias"] {
                headers[format!("time_embedder.{leaf}.{suffix}")] =
                    serde_json::json!({"dtype":"F32","shape":[1],"data_offsets":[0,0]});
            }
        }
        let stored = write_pruning_memory_headers(&dir, headers);
        let config_path = dir.join("config.json");
        let mut config: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&config_path).unwrap()).unwrap();
        config["quantization"] = serde_json::json!({"bits":bits,"group_size":64});
        std::fs::write(config_path, config.to_string()).unwrap();
        assert!(
            source_prune_sizing(&dir).unwrap().is_none(),
            "Q{bits} is full-width packed, not a missing curve table"
        );
        let quant = if bits == 4 {
            mlx_gen::gen_core::Quant::Q4
        } else {
            mlx_gen::gen_core::Quant::Q8
        };
        let spec = LoadSpec::new(WeightsSource::Dir(root.path().into())).with_quant(quant);
        let contract = contract_for(&spec).unwrap();
        let out = 18 * 256u64;
        let expected = 2 * (out * 64 * bits as u64 / 8 + out * 4 + out * 2);
        assert_eq!(contract.asset_facts.transformer_bytes, stored);
        let retained = 2 * out * 2 * (2 * crate::model::MAX_STEPS as u64 + 2);
        if expected > retained {
            assert_eq!(
                contract.resident_components()[0].resident_bytes,
                expected,
                "actual affine payload, not a ratio against the stock model population"
            );
        } else {
            assert!(
                contract
                    .resident_components()
                    .iter()
                    .all(|c| c.id != ADALN_COMPONENT_ID),
                "a cache larger than its projection stack earns no claimed eviction credit"
            );
        }
        assert_eq!(
            contract.evicted_component_bytes(),
            expected.saturating_sub(retained)
        );
        assert!(
            contract.conformance_errors().is_empty(),
            "{:?}",
            contract.conformance_errors()
        );
        // Keep companions config-only; the selected DiT's packed headers were written above.
        for (component, document) in [
            (crate::model::REFERENCE_DIT_PARTITION, "config.json"),
            ("text_encoder", "config.json"),
            ("vae", "config.json"),
            ("audio_vae", "config.json"),
            ("tokenizer", "tokenizer.json"),
            ("FL2VA/audio_vae", "config.json"),
            ("FL2VA/audio_vae", "config.yaml"),
            ("FL2VA/audio_vae", "metadata.json"),
        ] {
            let directory = root.path().join(component);
            std::fs::create_dir_all(&directory).unwrap();
            std::fs::write(directory.join(document), "{}").unwrap();
        }
        std::fs::write(
            root.path()
                .join(crate::model::REFERENCE_DIT_PARTITION)
                .join("config.json"),
            serde_json::json!({"quantization":{"bits":bits,"group_size":64}}).to_string(),
        )
        .unwrap();
        let loaded = crate::model::load(&spec).unwrap();
        let loaded = loaded.memory_strategy_contract().unwrap();
        assert_eq!(loaded.asset_facts, contract.asset_facts);
        assert_eq!(
            loaded.evicted_component_bytes(),
            contract.evicted_component_bytes()
        );
    }
}

#[test]
fn source_pruned_retained_bound_covers_nondefault_scheduler_shifts() {
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join(crate::model::BASE_DIT_PARTITION);
    write_pruning_memory_headers(&dir, pruning_memory_headers());
    let sizing = source_prune_sizing(&dir).unwrap().unwrap();
    let default = crate::denoise::JointSchedule::new(crate::model::MAX_STEPS as usize + 1).unwrap();
    let default_rows = crate::denoise::adaln_schedule(&default)
        .unwrap()
        .distinct_timesteps()
        .len();
    let mut exceeds_default = false;
    for video_shift in [0.1, 1.0, 5.0, 12.01, 17.0, 100.0] {
        for audio_shift in [0.125, 3.0, 7.0] {
            let schedule = crate::denoise::JointSchedule::with_shifts(
                crate::model::MAX_STEPS as usize + 1,
                video_shift,
                audio_shift,
            )
            .unwrap();
            let actual = crate::denoise::adaln_schedule(&schedule).unwrap();
            let distinct = actual.distinct_timesteps().len();
            exceeds_default |= distinct > default_rows;
            let actual_retained = distinct as u64 * (256 * (18 * 2 + 2) + 8) * 4;
            assert!(actual_retained <= sizing.retained_bytes,
                    "admitted shifts {video_shift}/{audio_shift}: {distinct} rows must fit the declared F32 cache bound");
        }
    }
    assert!(
        exceeds_default,
        "the fixture must discriminate the old default-grid underbound"
    );
    let contract = contract_for(&LoadSpec::new(WeightsSource::Dir(root.path().into()))).unwrap();
    assert_eq!(
        contract.resident_components()[0].resident_bytes,
        sizing.compute_bytes + sizing.retained_bytes
    );
    assert_eq!(contract.evicted_component_bytes(), 0);
}
