//! Public weights-free catalog checks.
#[test]
fn registers_both_native_routes() {
    let r = candle_gen_moss_image::register_providers(
        candle_gen::gen_core::ProviderRegistryBuilder::new(),
    )
    .build()
    .unwrap();
    assert!(r.descriptor_conformance_errors().is_empty());
    let ids: Vec<_> = r.generators().map(|r| (r.descriptor)().id).collect();
    assert_eq!(ids, vec!["moss_image_base", "moss_image_sft"]);
}

#[test]
fn checkpoint_import_reuses_the_existing_base_loader() {
    use candle_gen::gen_core::{
        ImportedModelOperation, ImportedModelSource, LoadSpec, WeightsSource,
    };
    let registry = candle_gen_moss_image::register_providers(
        candle_gen::gen_core::ProviderRegistryBuilder::new(),
    )
    .build()
    .unwrap();
    let descriptor = registry
        .imported_model_descriptor(
            "moss-image",
            ImportedModelSource::TransformerFile,
            ImportedModelOperation::Generate,
        )
        .unwrap();
    assert_eq!(descriptor.id, "moss_image_base");
    assert!(!descriptor.capabilities.supports_lora && !descriptor.capabilities.supports_lokr);
    let spec = LoadSpec::new(WeightsSource::Dir("unused-snapshot".into()));
    let error = registry
        .load_imported(
            "moss-image",
            ImportedModelSource::TransformerFile,
            ImportedModelOperation::Generate,
            &spec,
        )
        .err()
        .unwrap();
    assert!(error.to_string().contains("missing imported"));
}

#[test]
fn imported_transformer_cannot_be_discarded_by_dmd_selection() {
    use candle_gen::gen_core::{LoadSpec, WeightsSource};
    use gen_core_moss_image::moss_image::{
        MossDmdPrecision, MossImageAcceleration, MossImageVariant,
    };
    let spec = LoadSpec::new(WeightsSource::Dir("unused-snapshot".into()))
        .with_checkpoint_component(
            "transformer",
            WeightsSource::File("caller-owned.safetensors".into()),
        );
    for precision in [MossDmdPrecision::UnmergedBf16, MossDmdPrecision::FullFp32] {
        let error = candle_gen_moss_image::memory_strategy::selected_asset_facts(
            MossImageVariant::Sft,
            &spec,
            MossImageAcceleration::Dmd(precision),
        )
        .err()
        .unwrap();
        assert!(error.to_string().contains("does not support published DMD"));
    }
}
