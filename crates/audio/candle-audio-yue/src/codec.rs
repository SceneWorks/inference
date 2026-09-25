//! **Seam: xcodec decode** — one track's 8-codebook grid → its 16 kHz waveform plus the quantized
//! codec embedding the Vocos upsampler consumes.
//!
//! A native candle port (sc-19377) of the decode half of xcodec's `SoundStream`
//! (`xcodec_mini_infer/models/soundstream_hubert_new.py`; pinned revisions in
//! `tests/fixtures/xcodec_decode_reference.json`):
//!
//! ```text
//!   codes [8, T] ── quantizer.decode (Σ_q embed_q[code_q]) ──▶ embedding [1, 1024, T]   = get_embed
//!   embedding ─▶ fc_post2 Linear(1024 → 256) ─▶ [1, 256, T]
//!             ─▶ decoder_2  DAC Decoder(256, 1024, rates [8, 5, 4, 2])  ─▶ [1, 1, 320·T]  (16 kHz)
//! ```
//!
//! **The HuBERT / semantic branch is not on this path.** Upstream `SoundStream.decode` and
//! `get_embed` read only `quantizer`, `fc_post2` and `decoder_2`; the semantic model,
//! `encoder_semantic` / `decoder_semantic`, `fc_post1` and `fc_prior` serve `encode` (the ICL
//! reference encoder, sc-19379) and training. The loader therefore materializes only the decode
//! tensors of the checkpoint (the first [`NUM_CODEBOOKS`] of its 12 quantizers).
//!
//! The quantizer's `EuclideanCodebook`s are `[1024, 1024]` with `codebook_dim == dim`, so the
//! reference's `project_out` is the identity and the dequant is a plain lookup-sum — the shared
//! [`candle_audio::neural_codec::rvq_dequantize`]. `decoder_2` is the descript-audio-codec decoder
//! (its stride-5 block carries `output_padding = 1`, which is the shared `r mod 2` rule), ported
//! once in [`candle_audio::neural_codec::DacDecoder`]. Upstream's module is named after SEANet, but
//! the checkpoint's decode path is this DAC decoder (`SEANetDecoder` is never constructed).
//!
//! Precision: the codec runs in **float32** at every LM tier (the approved epic-R2 carve-out — the
//! staged `xcodec-mini-infer` checkpoint is float32 and is never tiered). [`StubCodec`] stays as
//! the end-to-end seam test's weights-free double.

use std::collections::HashMap;
use std::path::Path;

use candle_audio::candle_core::safetensors::MmapedSafetensors;
use candle_audio::candle_core::{DType, Device, Module, Tensor};
use candle_audio::gen_core::{self, CancelFlag};
use candle_audio::neural_codec::{resolve_weight_norm, rvq_dequantize, DacDecoder};
use candle_audio::AudioError;
use candle_nn::Linear;

use crate::config::Assets;
use crate::tokens::{CodecFrames, CODEBOOK_SIZE, FRAMES_PER_SECOND, NUM_CODEBOOKS};

/// The codec's native output rate.
pub const SAMPLE_RATE: u32 = 16_000;
/// Waveform samples per codec frame at [`SAMPLE_RATE`].
pub const SAMPLES_PER_FRAME: usize = (SAMPLE_RATE / FRAMES_PER_SECOND) as usize;
/// The DAC decoder's upsampling rates (`generator.config.ratios` of `final_ckpt/config.yaml`);
/// their product is [`SAMPLES_PER_FRAME`].
pub const DECODER_RATES: [usize; 4] = [8, 5, 4, 2];
/// The codec checkpoint inside the `xcodec_mini_infer` snapshot (the SceneWorks safetensors
/// rehost of `final_ckpt/ckpt_00360000.pth`'s `codec_model` state dict).
pub const CHECKPOINT: &str = "final_ckpt/ckpt_00360000.safetensors";

/// One decoded track.
#[derive(Clone, Debug)]
pub struct DecodedTrack {
    /// Mono waveform at [`SAMPLE_RATE`].
    pub wave: Vec<f32>,
    /// The RVQ-decoded codec embedding, `[1, dim, frames]` — the Vocos input.
    pub embedding: Tensor,
}

/// The codec-decoder seam.
pub trait CodecDecoder: Send {
    /// Decode one track's grid. Must honor `cancel` by returning [`gen_core::Error::Canceled`].
    fn decode(&self, frames: &CodecFrames, cancel: &CancelFlag) -> gen_core::Result<DecodedTrack>;
}

/// Production loader: the xcodec decoder from the staged snapshot's [`CHECKPOINT`], on the audio
/// lane's default device (CPU, or Metal / CUDA under the `metal` / `cuda` features).
pub fn load(assets: &Assets) -> gen_core::Result<Box<dyn CodecDecoder>> {
    let device = candle_audio::default_device()?;
    let decoder = XcodecDecoder::load(&assets.xcodec_root().join(CHECKPOINT), &device)?;
    Ok(Box::new(decoder))
}

/// The weights-free stub loader (the seam test's double).
pub fn load_stub(_assets: &Assets) -> gen_core::Result<Box<dyn CodecDecoder>> {
    Ok(Box::new(StubCodec))
}

/// The native xcodec decoder: RVQ dequant ([`Self::get_embed`]) → `fc_post2` → DAC `decoder_2`.
pub struct XcodecDecoder {
    /// The first [`NUM_CODEBOOKS`] quantizers' `[CODEBOOK_SIZE, dim]` codebooks.
    codebooks: Vec<Tensor>,
    fc_post2: Linear,
    decoder: DacDecoder,
    device: Device,
}

impl XcodecDecoder {
    /// Load the decode path from a codec checkpoint (safetensors, the `codec_model` state dict).
    /// Only `quantizer.vq.layers.{0..8}._codebook.embed`, `fc_post2.*` and `decoder_2.*` are read;
    /// the semantic branch and the encoders are never paged in. Widths come from the tensors and
    /// are cross-checked (codebook size, codebook width → `fc_post2` → decoder input).
    pub fn load(checkpoint: &Path, device: &Device) -> Result<Self, AudioError> {
        // SAFETY: the checkpoint is a caller-staged, read-only weight file; nothing in this process
        // writes it while the map is alive (the contract every candle mmap loader relies on).
        let st = unsafe { MmapedSafetensors::new(checkpoint) }
            .map_err(|e| AudioError::Msg(format!("xcodec: open {}: {e}", checkpoint.display())))?;
        let load = |name: &str| -> Result<Tensor, AudioError> {
            Ok(st
                .load(name, device)
                .map_err(|e| AudioError::Msg(format!("xcodec: tensor {name:?}: {e}")))?
                .to_dtype(DType::F32)?)
        };

        let codebooks = (0..NUM_CODEBOOKS)
            .map(|q| load(&format!("quantizer.vq.layers.{q}._codebook.embed")))
            .collect::<Result<Vec<_>, _>>()?;
        let (size, dim) = codebooks[0].dims2()?;
        if size != CODEBOOK_SIZE as usize || codebooks.iter().any(|c| c.dims() != [size, dim]) {
            return Err(AudioError::Msg(format!(
                "xcodec: codebooks must all be [{CODEBOOK_SIZE}, {dim}] (got {:?})",
                codebooks
                    .iter()
                    .map(|c| c.dims().to_vec())
                    .collect::<Vec<_>>()
            )));
        }

        let fc_w = load("fc_post2.weight")?;
        let fc_b = load("fc_post2.bias")?;
        let (latent, fc_in) = fc_w.dims2()?;
        if fc_in != dim {
            return Err(AudioError::Msg(format!(
                "xcodec: fc_post2 takes {fc_in} channels but the codebooks are {dim} wide"
            )));
        }

        let mut raw = HashMap::new();
        for (name, _) in st.tensors() {
            if name.starts_with("decoder_2.") {
                let t = load(&name)?;
                raw.insert(name, t);
            }
        }
        let map = resolve_weight_norm(raw)?;
        let decoder = DacDecoder::load(&map, "decoder_2", &DECODER_RATES)?;
        let dec_in = decoder.input_dim()?;
        if dec_in != latent {
            return Err(AudioError::Msg(format!(
                "xcodec: decoder_2 takes {dec_in} channels but fc_post2 emits {latent}"
            )));
        }
        Ok(Self {
            codebooks,
            fc_post2: Linear::new(fc_w, Some(fc_b)),
            decoder,
            device: device.clone(),
        })
    }

    /// Upstream `SoundStream.get_embed`: the RVQ-dequantized embedding `[1, dim, frames]` (the
    /// Vocos upsampler's input). The grid must pass [`check_grid`].
    pub fn get_embed(&self, frames: &CodecFrames) -> Result<Tensor, AudioError> {
        check_grid(frames)?;
        Ok(rvq_dequantize(
            &self.codebooks,
            &frames.codebooks,
            &self.device,
        )?)
    }

    /// Decode an embedding from [`Self::get_embed`] to the `[1, 1, 320·frames]` waveform
    /// (`fc_post2` → `decoder_2`). `None` when `cancel` trips between decoder blocks.
    pub fn decode_embedding(
        &self,
        embedding: &Tensor,
        cancel: &dyn Fn() -> bool,
    ) -> Result<Option<Tensor>, AudioError> {
        let x = self
            .fc_post2
            .forward(&embedding.transpose(1, 2)?.contiguous()?)?
            .transpose(1, 2)?
            .contiguous()?;
        Ok(self.decoder.decode(&x, cancel)?)
    }
}

impl CodecDecoder for XcodecDecoder {
    fn decode(&self, frames: &CodecFrames, cancel: &CancelFlag) -> gen_core::Result<DecodedTrack> {
        if cancel.is_cancelled() {
            return Err(gen_core::Error::Canceled);
        }
        let embedding = self.get_embed(frames)?;
        if frames.frames() == 0 {
            return Ok(DecodedTrack {
                wave: Vec::new(),
                embedding,
            });
        }
        let wave = self
            .decode_embedding(&embedding, &|| cancel.is_cancelled())?
            .ok_or(gen_core::Error::Canceled)?;
        let wave = wave
            .flatten_all()
            .and_then(|w| w.to_vec1::<f32>())
            .map_err(AudioError::from)?;
        Ok(DecodedTrack { wave, embedding })
    }
}

/// The grid the codec accepts: exactly [`NUM_CODEBOOKS`] equal-length rows of codes in
/// `0..CODEBOOK_SIZE`. Out-of-range codes are refused, never clamped (the reference's
/// `F.embedding` would raise).
pub fn check_grid(frames: &CodecFrames) -> Result<(), AudioError> {
    if frames.codebooks.len() != NUM_CODEBOOKS {
        return Err(AudioError::Msg(format!(
            "xcodec: expected {NUM_CODEBOOKS} codebooks, got {}",
            frames.codebooks.len()
        )));
    }
    let t = frames.frames();
    if frames.codebooks.iter().any(|row| row.len() != t) {
        return Err(AudioError::Msg(
            "xcodec: every codebook row must hold the same number of frames".into(),
        ));
    }
    if let Some(&c) = frames
        .codebooks
        .iter()
        .flatten()
        .find(|&&c| c >= CODEBOOK_SIZE)
    {
        return Err(AudioError::Msg(format!(
            "xcodec: code {c} is outside 0..{CODEBOOK_SIZE}"
        )));
    }
    Ok(())
}

/// Embedding width the stub reports.
const STUB_DIM: usize = 4;

/// **Stub xcodec decoder** — the end-to-end seam test's weights-free double. Renders
/// [`SAMPLES_PER_FRAME`] samples of a code-keyed tone per frame, and an embedding holding each
/// frame's first four codes scaled to `[0, 1)`.
#[derive(Clone, Copy, Debug, Default)]
pub struct StubCodec;

impl CodecDecoder for StubCodec {
    fn decode(&self, frames: &CodecFrames, cancel: &CancelFlag) -> gen_core::Result<DecodedTrack> {
        if cancel.is_cancelled() {
            return Err(gen_core::Error::Canceled);
        }
        let n = frames.frames();
        let mut wave = Vec::with_capacity(n * SAMPLES_PER_FRAME);
        for t in 0..n {
            let key = frames.codebooks.iter().map(|row| row[t] as u64).sum();
            wave.extend(crate::stub::tone(key, SAMPLES_PER_FRAME, SAMPLE_RATE));
        }
        let mut data = Vec::with_capacity(STUB_DIM * n);
        for row in frames.codebooks.iter().take(STUB_DIM) {
            data.extend(row.iter().map(|&c| c as f32 / CODEBOOK_SIZE as f32));
        }
        let embedding = Tensor::from_vec(data, (1, STUB_DIM, n), &Device::Cpu)
            .map_err(|e| gen_core::Error::Msg(format!("stub codec embedding: {e}")))?;
        Ok(DecodedTrack { wave, embedding })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_audio::candle_core::safetensors::save;

    /// Tiny synthetic widths — the production loader reads widths from the tensors, so a
    /// few-hundred-KB checkpoint exercises the real load + decode path.
    const E: usize = 4;
    const LATENT: usize = 3;
    const CH: usize = 16;

    fn det(shape: &[usize], seed: u32) -> Tensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|i| (((i as u32).wrapping_mul(2_654_435_761) ^ seed) % 1000) as f32 / 1000.0 - 0.5)
            .collect();
        Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
    }

    /// A weight-norm pair plus bias. `transposed` convs store `[in, out, k]` (bias `[out]`).
    fn wn(t: &mut HashMap<String, Tensor>, name: &str, shape: [usize; 3], transposed: bool) {
        let seed = t.len() as u32;
        t.insert(format!("{name}.weight_v"), det(&shape, seed));
        t.insert(format!("{name}.weight_g"), det(&[shape[0], 1, 1], seed + 1));
        let out = if transposed { shape[1] } else { shape[0] };
        t.insert(format!("{name}.bias"), det(&[out], seed + 2));
    }

    /// Write a tiny xcodec-shaped checkpoint at [`CHECKPOINT`] under a snapshot root.
    fn tiny_snapshot() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let mut t: HashMap<String, Tensor> = HashMap::new();
        for q in 0..12 {
            t.insert(
                format!("quantizer.vq.layers.{q}._codebook.embed"),
                det(&[CODEBOOK_SIZE as usize, E], q),
            );
        }
        t.insert("fc_post2.weight".into(), det(&[LATENT, E], 50));
        t.insert("fc_post2.bias".into(), det(&[LATENT], 51));
        wn(&mut t, "decoder_2.model.0", [CH, LATENT, 7], false);
        let mut c = CH;
        for (i, &r) in DECODER_RATES.iter().enumerate() {
            let base = format!("decoder_2.model.{}", i + 1);
            t.insert(
                format!("{base}.block.0.alpha"),
                det(&[1, c, 1], 7 + i as u32),
            );
            wn(&mut t, &format!("{base}.block.1"), [c, c / 2, 2 * r], true);
            for j in 0..3 {
                let ru = format!("{base}.block.{}", j + 2);
                t.insert(format!("{ru}.block.0.alpha"), det(&[1, c / 2, 1], 20 + j));
                wn(&mut t, &format!("{ru}.block.1"), [c / 2, c / 2, 7], false);
                t.insert(format!("{ru}.block.2.alpha"), det(&[1, c / 2, 1], 30 + j));
                wn(&mut t, &format!("{ru}.block.3"), [c / 2, c / 2, 1], false);
            }
            c /= 2;
        }
        t.insert("decoder_2.model.5.alpha".into(), det(&[1, c, 1], 40));
        wn(&mut t, "decoder_2.model.6", [1, c, 7], false);
        // A semantic-branch tensor the decode path must never need.
        t.insert("semantic_model.unused".into(), det(&[2], 99));
        let path = dir.path().join(CHECKPOINT);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        save(&t, &path).unwrap();
        dir
    }

    fn assets(xcodec: &Path) -> Assets {
        Assets {
            stage1: "/unused/s1".into(),
            stage2: "/unused/s2".into(),
            xcodec: xcodec.to_path_buf(),
        }
    }

    fn grid(frames: usize) -> CodecFrames {
        CodecFrames {
            codebooks: (0..NUM_CODEBOOKS)
                .map(|k| {
                    (0..frames)
                        .map(|t| ((t * 131 + k * 97) % 1024) as u32)
                        .collect()
                })
                .collect(),
        }
    }

    #[test]
    fn decoder_rates_upsample_one_frame_to_its_sample_count() {
        assert_eq!(DECODER_RATES.iter().product::<usize>(), SAMPLES_PER_FRAME);
        assert_eq!(SAMPLES_PER_FRAME, 320);
    }

    #[test]
    fn production_load_decodes_a_staged_checkpoint_instead_of_refusing() {
        let snap = tiny_snapshot();
        let codec = load(&assets(snap.path())).expect("production codec loads staged weights");
        let out = codec.decode(&grid(3), &CancelFlag::new()).unwrap();
        assert_eq!(out.wave.len(), 3 * SAMPLES_PER_FRAME);
        assert!(out.wave.iter().all(|s| s.is_finite()));
        assert_eq!(out.embedding.dims(), [1, E, 3]);
    }

    #[test]
    fn get_embed_is_the_sum_of_the_first_eight_codebook_rows() {
        let snap = tiny_snapshot();
        let dec = XcodecDecoder::load(&snap.path().join(CHECKPOINT), &Device::Cpu).unwrap();
        let frames = grid(2);
        let got = dec
            .get_embed(&frames)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        for t in 0..2 {
            for (d, row) in got.iter().enumerate() {
                let want: f32 = (0..NUM_CODEBOOKS)
                    .map(|q| {
                        let code = frames.codebooks[q][t] as usize;
                        det(&[CODEBOOK_SIZE as usize, E], q as u32)
                            .get(code)
                            .and_then(|r| r.get(d))
                            .and_then(|v| v.to_scalar::<f32>())
                            .unwrap()
                    })
                    .sum();
                assert!((row[t] - want).abs() < 1e-6, "frame {t} dim {d}");
            }
        }
    }

    #[test]
    fn decode_refuses_malformed_grids_and_honors_cancel() {
        let snap = tiny_snapshot();
        let codec = load(&assets(snap.path())).unwrap();
        let mut bad = grid(2);
        bad.codebooks[3][1] = CODEBOOK_SIZE;
        assert!(codec.decode(&bad, &CancelFlag::new()).is_err());
        let mut short = grid(2);
        short.codebooks.pop();
        assert!(codec.decode(&short, &CancelFlag::new()).is_err());
        let cancel = CancelFlag::new();
        cancel.cancel();
        assert!(matches!(
            codec.decode(&grid(2), &cancel),
            Err(gen_core::Error::Canceled)
        ));
        let empty = codec.decode(&grid(0), &CancelFlag::new()).unwrap();
        assert!(empty.wave.is_empty());
        assert_eq!(empty.embedding.dims(), [1, E, 0]);
    }

    /// max |got − want| / max |want|.
    fn max_rel(got: &[f32], want: &[f32]) -> f64 {
        assert_eq!(got.len(), want.len(), "length mismatch");
        let peak = want.iter().fold(0f64, |m, &v| m.max(v.abs() as f64));
        let diff = got
            .iter()
            .zip(want)
            .fold(0f64, |m, (&a, &b)| m.max((a as f64 - b as f64).abs()));
        diff / peak
    }

    /// Always-run numeric parity against the **upstream** decode modules at toy widths
    /// (`scripts/reference/yue_xcodec_reference.py tiny`): the fixture is one safetensors file
    /// holding a `SoundStream`-layout state dict (random codebooks, `fc_post2`, a DAC
    /// `Decoder(3, 16, [8, 5, 4, 2])` with randomized weight-norm `g` and Snake `α`) plus a code
    /// grid and torch-CPU `get_embed` / `decode` outputs. It loads through the production
    /// [`XcodecDecoder::load`], so a wrong dilation, activation, padding or weight-norm fold is
    /// caught without real weights. Measured max relative error: embed 0, wave 5.4e-7.
    #[test]
    fn tiny_upstream_decoder_matches_the_torch_reference() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("xcodec_tiny_reference.safetensors");
        let fx = candle_audio::candle_core::safetensors::load(&path, &Device::Cpu).unwrap();
        let flat = |k: &str| -> Vec<f32> { fx[k].flatten_all().unwrap().to_vec1().unwrap() };
        let frames = CodecFrames {
            codebooks: fx["ref.codes"]
                .to_vec2::<i64>()
                .unwrap()
                .into_iter()
                .map(|r| r.into_iter().map(|c| c as u32).collect())
                .collect(),
        };
        let dec = XcodecDecoder::load(&path, &Device::Cpu).unwrap();
        let embed = dec.get_embed(&frames).unwrap();
        let embed_rel = max_rel(
            &embed.flatten_all().unwrap().to_vec1().unwrap(),
            &flat("ref.embed"),
        );
        assert!(embed_rel <= 1e-6, "get_embed diverges: {embed_rel:.3e}");
        let wave = dec.decode(&frames, &CancelFlag::new()).unwrap().wave;
        let wave_rel = max_rel(&wave, &flat("ref.wave"));
        println!("tiny parity: embed {embed_rel:.3e}, wave {wave_rel:.3e}");
        assert!(wave_rel <= 1e-5, "decode diverges: {wave_rel:.3e}");
    }

    #[test]
    fn a_missing_checkpoint_is_an_error_not_a_refusal() {
        let dir = tempfile::tempdir().unwrap();
        match load(&assets(dir.path())) {
            Err(gen_core::Error::Unsupported(m)) => panic!("still refusing: {m}"),
            Err(_) => {}
            Ok(_) => panic!("loaded without a checkpoint"),
        }
    }
}
