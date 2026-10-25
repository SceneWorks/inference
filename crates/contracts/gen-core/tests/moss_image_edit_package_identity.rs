//! CPU-only binding to captured public Hub metadata; never downloads weights.
use gen_core::moss_image_edit::{
    EDIT_CONFIG_SHA256, EDIT_REPOSITORY, EDIT_REVISION, EDIT_WEIGHTS_SHA256,
};
use sha2::{Digest, Sha256};

#[test]
fn edit_package_matches_accessible_upstream_identity_without_changing_content() {
    let identity: serde_json::Value = serde_json::from_str(include_str!(
        "fixtures/moss_image/tiny_edit/upstream_identity.json"
    ))
    .unwrap();
    assert_eq!(identity["repo"], EDIT_REPOSITORY);
    assert_eq!(identity["revision"], EDIT_REVISION);
    assert_eq!(identity["treeStatus"], 200);
    assert_eq!(identity["configResponseCommit"], EDIT_REVISION);
    assert_eq!(identity["oldRevisionStatus"], 404);
    assert_eq!(identity["oldRevisionError"], "RevisionNotFound");
    assert_ne!(identity["formerRevision"], EDIT_REVISION);
    assert_eq!(identity["weights"]["sha256"], EDIT_WEIGHTS_SHA256);
    assert_eq!(identity["config"]["sha256"], EDIT_CONFIG_SHA256);
    let original_config = include_bytes!("fixtures/moss_image/tiny_edit/adapter_config.json");
    assert_eq!(identity["config"]["byteLength"], original_config.len());
    assert_eq!(
        format!("{:x}", Sha256::digest(original_config)),
        EDIT_CONFIG_SHA256
    );
}
