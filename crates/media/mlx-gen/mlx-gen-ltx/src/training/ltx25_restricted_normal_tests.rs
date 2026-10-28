#![cfg(test)]

use super::*;

/// sc-24832: LTX-2.5 refuses the subject-restricted normal loss (no image aligned to its
/// latent bundles) and still admits the unrestricted one. Mutation: drop the
/// `PerceptualSubjectMasks::needed` refusal ⇒ red.
#[test]
fn ltx25_refuses_subject_restricted_normals() {
    let mut req = super::validate_request_tests::request(1);
    req.config.body_losses.normal.weight = 0.1;
    assert!(refuse_ltx25_subject_mask(MODEL_25_ID, &req).is_ok());
    req.config.body_losses.normal_restrict_to_subject = true;
    match refuse_ltx25_subject_mask(MODEL_25_ID, &req) {
        Err(gen_core::Error::Unsupported(m)) => {
            assert!(m.contains("normal_restrict_to_subject"))
        }
        other => panic!("expected Unsupported, got {other:?}"),
    }
    assert!(
        refuse_ltx25_subject_mask(MODEL_ID, &req).is_ok(),
        "LTX-2.3 has aligned images"
    );
}
