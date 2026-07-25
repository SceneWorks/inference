//! The SAME patched stereo pretransform.

use candle_audio::candle_core::{bail, Result, Tensor};

/// Reversible channel-to-patch transform used by SAME.
#[derive(Debug, Clone, Copy)]
pub struct PatchedPretransform {
    channels: usize,
    patch_size: usize,
}

impl PatchedPretransform {
    pub fn new(channels: usize, patch_size: usize) -> Result<Self> {
        if channels == 0 || patch_size == 0 {
            bail!("patched pretransform channels and patch_size must be non-zero")
        }
        Ok(Self {
            channels,
            patch_size,
        })
    }

    pub fn encoded_channels(&self) -> usize {
        self.channels * self.patch_size
    }

    /// Zero-pad the temporal tail to a patch boundary and return `[B, C*P, ceil(T/P)]`.
    pub fn encode(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, channels, length) = x.dims3()?;
        if channels != self.channels {
            bail!(
                "patched pretransform expected {} channels, got {channels}",
                self.channels
            )
        }
        let padded = length.div_ceil(self.patch_size) * self.patch_size;
        let x = if padded == length {
            x.clone()
        } else {
            let zeros = Tensor::zeros((batch, channels, padded - length), x.dtype(), x.device())?;
            Tensor::cat(&[x, &zeros], 2)?
        };
        x.reshape((batch, channels, padded / self.patch_size, self.patch_size))?
            .transpose(2, 3)?
            .reshape((batch, channels * self.patch_size, padded / self.patch_size))
    }

    /// Decode every patch. For a non-divisible original length this intentionally returns the
    /// padded length; callers that know the source length may crop the valid prefix.
    pub fn decode(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, channels, patches) = x.dims3()?;
        if channels != self.encoded_channels() {
            bail!(
                "patched pretransform expected {} encoded channels, got {channels}",
                self.encoded_channels()
            )
        }
        x.reshape((batch, self.channels, self.patch_size, patches))?
            .transpose(2, 3)?
            .reshape((batch, self.channels, patches * self.patch_size))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_audio::candle_core::Device;

    #[test]
    fn exact_at_256_and_zero_padded_otherwise() {
        let dev = Device::Cpu;
        let p = PatchedPretransform::new(2, 256).unwrap();
        for len in [256usize, 259] {
            let values: Vec<f32> = (0..2 * len).map(|v| v as f32).collect();
            let x = Tensor::from_vec(values, (1, 2, len), &dev).unwrap();
            let y = p.decode(&p.encode(&x).unwrap()).unwrap();
            assert_eq!(
                y.narrow(2, 0, len).unwrap().to_vec3::<f32>().unwrap(),
                x.to_vec3::<f32>().unwrap()
            );
            if len % 256 != 0 {
                assert_eq!(
                    y.narrow(2, len, y.dim(2).unwrap() - len)
                        .unwrap()
                        .abs()
                        .unwrap()
                        .max_all()
                        .unwrap()
                        .to_scalar::<f32>()
                        .unwrap(),
                    0.0
                );
            }
        }
    }
}
