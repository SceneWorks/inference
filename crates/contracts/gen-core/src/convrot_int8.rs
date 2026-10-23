//! Native ConvRot INT8 projection contract (sc-26683).
//!
//! This is deliberately separate from the plain `int8-per-row-v1` dense decoder: the
//! stored codes are in the rotated basis and require dynamic, per-token INT8 execution.
//! Authority: Comfy Kitchen f8b7e97a1418d2e529bd5405fbf86312be198ae6 and
//! comfy-model-tools d6797787e6bdb1a1fb0094d588a26f8e71a1c757.

use serde::Deserialize;

/// Stable source/storage and native execution identity of this validated operand format.
///
/// Consumers may persist this identity only after validating the descriptor and operands with
/// [`ConvRotInt8`]. It is distinct from the unrotated `int8-per-row-v1` codec: reporting that
/// codec for these rotated codes would describe the wrong basis and activation arithmetic.
/// This identifier does not register a generic dense decoder or authorize a dense fallback;
/// native MLX and CUDA projections consume the original signed codes and F32 row scales.
pub const CODEC_ID: &str = "convrot-int8-native";

/// Maximum descriptor payload; checked before copying a descriptor off a device.
pub const MAX_DESCRIPTOR_BYTES: usize = 4096;

/// Validated logical geometry. `rotation_group` is NOT a scale group: weight scales
/// are F32 `[out, 1]`, activation scales are dynamic `[tokens, 1]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConvRotInt8 {
    pub out_features: usize,
    pub in_features: usize,
    pub rotation_group: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Descriptor {
    format: String,
    convrot: bool,
    convrot_groupsize: usize,
}

impl ConvRotInt8 {
    /// Parse the authoritative U8 `.comfy_quant` payload and validate its operands.
    /// Only the actual H3 INT8 dialect is admitted; affine Q8 and W4A8 are different codecs.
    pub fn parse(payload: &[u8], weight: &[usize], scales: &[usize]) -> Result<Self, String> {
        if payload.is_empty() || payload.len() > MAX_DESCRIPTOR_BYTES {
            return Err("ConvRot INT8 descriptor length is invalid".into());
        }
        let d: Descriptor =
            serde_json::from_slice(payload).map_err(|e| format!("ConvRot INT8 descriptor: {e}"))?;
        if d.format != "int8_tensorwise" || !d.convrot {
            return Err("expected int8_tensorwise with convrot:true".into());
        }
        if weight.len() != 2 || weight.contains(&0) {
            return Err("ConvRot INT8 weight must be nonempty [out, in]".into());
        }
        let (n, k) = (weight[0], weight[1]);
        if !matches!(d.convrot_groupsize, 64 | 256)
            || !k.is_multiple_of(d.convrot_groupsize)
            || k > i32::MAX as usize / (128 * 128)
            || n > i32::MAX as usize
            || n.checked_mul(k).is_none()
        {
            return Err(
                "ConvRot INT8 invalid rotation group, tail, or integer GEMM geometry".into(),
            );
        }
        if scales != [n, 1] {
            return Err("ConvRot INT8 weight_scale must be F32 [out, 1] (not group scales)".into());
        }
        Ok(Self {
            out_features: n,
            in_features: k,
            rotation_group: d.convrot_groupsize,
        })
    }

    /// Value checks shared by both native backends, before a projection is executable.
    pub fn validate_scales(&self, scales: &[f32]) -> Result<(), String> {
        self.validate_scale_values(scales.iter().copied())
    }

    /// Streaming load-boundary check: scan mapped F32 operands without allocating a scale copy.
    pub fn validate_scale_values(
        &self,
        scales: impl IntoIterator<Item = f32>,
    ) -> Result<(), String> {
        let mut count = 0usize;
        for scale in scales {
            if !scale.is_finite() || scale <= 0.0 || count == self.out_features {
                return Err("ConvRot INT8 requires finite positive per-output F32 scales".into());
            }
            count += 1;
        }
        if count != self.out_features {
            return Err("ConvRot INT8 requires finite positive per-output F32 scales".into());
        }
        Ok(())
    }

    /// Resident codes and row scales, excluding an optional dense bias. No dense base exists.
    pub fn resident_bytes(&self) -> usize {
        self.out_features * self.in_features + self.out_features * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn actual_h3_groups_and_row_scales() {
        for (k, g) in [(2688, 64), (5376, 256), (7168, 256), (14336, 256)] {
            let payload =
                format!(r#"{{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":{g}}}"#);
            let d = ConvRotInt8::parse(payload.as_bytes(), &[16, k], &[16, 1]).unwrap();
            assert_eq!(d.rotation_group, g);
            assert_eq!(d.resident_bytes(), 16 * k + 64);
            d.validate_scales(&[1e-30; 16]).unwrap();
        }
    }

    #[test]
    fn refuses_malformed_contradictory_and_partial_metadata() {
        let good = r#"{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":64}"#;
        for bad in [
            good.replace("true", "false"),
            good.replace("64", "256"),
            good.replace("64", "32"),
            good.replace("int8_tensorwise", "asym_w4a8_int8"),
            good.replace("true,", "true,\"per_row\":true,"),
            "{}".into(),
            "[]".into(),
        ] {
            assert!(
                ConvRotInt8::parse(bad.as_bytes(), &[16, 192], &[16, 1]).is_err(),
                "{bad}"
            );
        }
        for shape in [&[16][..], &[16, 4][..], &[1, 16][..]] {
            assert!(ConvRotInt8::parse(good.as_bytes(), &[16, 192], shape).is_err());
        }
        assert!(ConvRotInt8::parse(good.as_bytes(), &[16, 65], &[16, 1]).is_err());
        let d = ConvRotInt8::parse(good.as_bytes(), &[1, 64], &[1, 1]).unwrap();
        for s in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(d.validate_scales(&[s]).is_err());
        }
        assert!(d.validate_scales(&[]).is_err());
        assert!(
            ConvRotInt8::parse(&vec![b' '; MAX_DESCRIPTOR_BYTES + 1], &[1, 64], &[1, 1]).is_err()
        );
    }
}
