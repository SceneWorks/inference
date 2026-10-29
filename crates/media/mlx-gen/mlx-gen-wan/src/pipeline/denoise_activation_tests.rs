use super::*;

/// The admission estimate prices exactly the activation the generate-time fit gate does.
#[test]
fn integer_activation_matches_the_fit_gate() {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    for (tokens, dim, cfg) in [
        (13_824, 5_120, true),
        (1_280, 1_536, false),
        (32_560, 3_072, true),
    ] {
        let bytes = denoise_activation_bytes(tokens, dim, cfg).unwrap();
        let gib = estimated_denoise_peak_gib(0, tokens, dim, cfg);
        assert!(
            (bytes as f64 / GIB - gib).abs() < 1e-9,
            "{tokens} {dim} {cfg}"
        );
    }
    assert_eq!(denoise_activation_bytes(usize::MAX, usize::MAX, true), None);
}
