//! The exact SC-21028 trainer artifact: its pinned identity, the operator-supplied paths and the
//! QKV probe.

use super::*;

// ─── exact external trainer receipt (sc-21028) ─────────────────────────────────────────────────

pub(super) const TRAINER_LORA_ENV: &str = "MINIMAX_H3_TRAINER_LORA";

pub(super) const TRAINER_LORA_SHA256_ENV: &str = "MINIMAX_H3_TRAINER_LORA_SHA256";

pub(super) const TRAINER_LORA_BYTES_ENV: &str = "MINIMAX_H3_TRAINER_LORA_BYTES";

pub(super) const TRAINER_DIT_ENV: &str = "MINIMAX_H3_DIT";

pub(super) const SC21028_TRAINER_LORA_SHA256: &str =
    "1fd239662f6290255b0bb3a220764fb53aab2859378f7fd3024030c1e1991cb2";

pub(super) const SC21028_TRAINER_LORA_BYTES: u64 = 298_263_792;

pub(super) const SC21028_PROBE_DOWN: &str = "lora_unet_blocks_0_attn_qkv_proj.lora_down.weight";

pub(super) const SC21028_PROBE_UP: &str = "lora_unet_blocks_0_attn_qkv_proj.lora_up.weight";

pub(super) const SC21028_PROBE_ALPHA: &str = "lora_unet_blocks_0_attn_qkv_proj.alpha";

pub(super) const SC21028_RUNTIME_ADAPTER_SCALE: f32 = 0.5;

pub(super) const SC21028_RUNTIME_RESIDUAL_REL_MAX: f32 = 1e-3;

/// An explicitly selected exact-file receipt must fail closed when the proprietary file was not
/// supplied. The optional digest and byte count bind a local receipt to a known artifact when the
/// operator has them; absent values are reported, never invented.
pub(super) fn trainer_lora_path() -> PathBuf {
    let raw = std::env::var(TRAINER_LORA_ENV).unwrap_or_else(|_| {
        panic!(
            "{TRAINER_LORA_ENV}=<exact networks.lora_minimax_h3 safetensors file> is required \
             when this ignored receipt test is selected"
        )
    });
    let path = PathBuf::from(raw);
    assert!(
        path.is_file(),
        "{TRAINER_LORA_ENV}={} is not a readable safetensors file",
        path.display()
    );
    path
}

pub(super) fn sha256_of(path: &Path) -> String {
    let mut file =
        File::open(path).unwrap_or_else(|error| panic!("open {}: {error}", path.display()));
    let mut hasher = Sha256::new();
    io::copy(&mut file, &mut hasher)
        .unwrap_or_else(|error| panic!("hash {}: {error}", path.display()));
    format!("{:x}", hasher.finalize())
}

pub(super) fn assert_optional_trainer_artifact_identity(path: &Path, bytes: u64, sha256: &str) {
    if let Ok(expected) = std::env::var(TRAINER_LORA_BYTES_ENV) {
        let expected = expected.parse::<u64>().unwrap_or_else(|error| {
            panic!("{TRAINER_LORA_BYTES_ENV}={expected:?} is not an unsigned byte count: {error}")
        });
        assert_eq!(bytes, expected, "{} byte count", path.display());
    }
    if let Ok(expected) = std::env::var(TRAINER_LORA_SHA256_ENV) {
        let expected = expected.trim().to_ascii_lowercase();
        assert!(
            expected.len() == 64 && expected.bytes().all(|byte| byte.is_ascii_hexdigit()),
            "{TRAINER_LORA_SHA256_ENV} must be a 64-character SHA-256 hex digest"
        );
        assert_eq!(sha256, expected, "{} SHA-256", path.display());
    }
}

pub(super) fn assert_sc21028_trainer_artifact_identity(path: &Path, bytes: u64, sha256: &str) {
    assert_eq!(
        bytes,
        SC21028_TRAINER_LORA_BYTES,
        "{} is not the exact SC-21028 trainer artifact: byte count",
        path.display()
    );
    assert_eq!(
        sha256,
        SC21028_TRAINER_LORA_SHA256,
        "{} is not the exact SC-21028 trainer artifact: SHA-256",
        path.display()
    );
    assert_optional_trainer_artifact_identity(path, bytes, sha256);
}

pub(super) fn trainer_dit_path() -> PathBuf {
    let raw = std::env::var(TRAINER_DIT_ENV).unwrap_or_else(|_| {
        panic!(
            "{TRAINER_DIT_ENV}=<real MiniMax-H3 transformer component directory> is required; \
             this ignored runtime receipt never skips"
        )
    });
    let path = PathBuf::from(raw);
    assert!(
        path.is_dir() && path.join("config.json").is_file(),
        "{TRAINER_DIT_ENV}={} must be a real transformer component directory with config.json",
        path.display()
    );
    path
}

/// Assert the runtime this receipt will actually dispatch through. MLX's public device model names
/// the Apple accelerator `Gpu`; on macOS that backend is Metal. Checking only availability would
/// still let a caller set the process default to CPU and publish a false Metal receipt.
pub(super) fn require_default_mlx_metal_device() -> (String, i32) {
    #[cfg(not(target_os = "macos"))]
    panic!("the MLX/Metal receipt is only valid on macOS");
    let device = Device::try_default().expect("query the MLX process-default device");
    let device_type = device
        .get_type()
        .expect("query the MLX default device type");
    assert!(
        matches!(device_type, DeviceType::Gpu),
        "the MLX process-default device must be GPU/Metal, got {device}"
    );
    let index = device
        .get_index()
        .expect("query the MLX default device index");
    (device.to_string(), index)
}

/// Independent oracle for the exact probe residual. It reads the trainer's raw fused-QKV factors,
/// takes the Q row block itself, and evaluates
/// `x·down^T·up_q^T·alpha/rank·SC21028_RUNTIME_ADAPTER_SCALE`; it never calls either trainer
/// conversion or the adapter installer under test. The exact file's alpha/rank is the identity
/// 16/16, so the non-identity alpha-formula proof remains the independent mutation fixture
/// `trainer_namespace_converts_to_independent_expected_factors_at_relative_max_abs`; this oracle's
/// independent non-unit term is the runtime adapter scale.
pub(super) fn exact_trainer_q_probe_residual(
    path: &Path,
    x: &Array,
    cfg: &MiniMaxH3DitConfig,
) -> Array {
    let raw = Weights::from_file(path).expect("read exact trainer factors for independent oracle");
    let down = raw
        .require(SC21028_PROBE_DOWN)
        .expect("exact QKV down factor");
    let fused_up = raw
        .require(SC21028_PROBE_UP)
        .expect("exact fused QKV up factor");
    let alpha = raw
        .require(SC21028_PROBE_ALPHA)
        .expect("exact fused QKV alpha")
        .as_dtype(Dtype::Float32)
        .unwrap()
        .item::<f32>();
    let rank = down.shape()[0];
    assert_eq!(down.shape(), &[16, cfg.hidden_size], "raw QKV down shape");
    assert_eq!(
        fused_up.shape(),
        &[3 * cfg.inner_dim(), 16],
        "raw fused QKV up shape"
    );
    assert_eq!((rank, alpha), (16, 16.0), "raw probe rank/alpha");
    let q_up = mlx_gen_minimax_h3::tensor::slice_axis(fused_up, 0, 0, cfg.inner_dim())
        .expect("take the Q row block independently");
    let unscaled = matmul(matmul(x, down.t()).unwrap(), q_up.t()).unwrap();
    unscaled
        .multiply(Array::from_slice(
            &[(alpha / rank as f32) * SC21028_RUNTIME_ADAPTER_SCALE],
            &[1],
        ))
        .expect("scale independent exact trainer residual")
}

pub(super) fn converted_lora_module(key: &str) -> Option<&str> {
    key.strip_suffix(".lora_A.weight")
        .or_else(|| key.strip_suffix(".lora_B.weight"))
        .or_else(|| key.strip_suffix(".alpha"))
}
