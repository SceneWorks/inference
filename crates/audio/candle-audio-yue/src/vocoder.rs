//! **Seam: Vocos upsampler** — a track's codec embedding → its 44.1 kHz waveform, through the
//! track's own decoder checkpoint (separate vocal and instrumental decoders in
//! `xcodec_mini_infer/decoders/`).
//!
//! A native candle port (sc-19378) of upstream's `VocosDecoder` (`xcodec_mini_infer/vocos`, config
//! `decoders/config.yaml`; pinned revisions in `tests/fixtures/vocos_splice_reference.json`) on the
//! shared [`candle_audio::vocos::Vocos`]:
//!
//! ```text
//!   embedding [1, 1024, T]  (= SoundStream.get_embed, what `vocoder.process_audio` feeds it)
//!     ─▶ ConvNeXt backbone (1024 → 512, 8 blocks, MLP 1536) ─▶ ISTFT head (n_fft 3528, hop 882)
//!     ─▶ [T · 882] samples at 44.1 kHz
//! ```
//!
//! The vocal stem runs through `decoder_131000`, the instrumental stem through `decoder_151000`
//! ([`VOCAL_CHECKPOINT`] / [`INSTRUMENTAL_CHECKPOINT`], the SceneWorks safetensors rehosts of the
//! upstream `.pth` files). Widths are read from the tensors; the synthesis window is the
//! checkpoint's own `head.istft.window` buffer. Precision: **float32** at every LM tier (the
//! approved epic-R2 carve-out — the staged decoders are float32 and never tiered).
//! [`StubVocoder`] stays as the end-to-end seam test's weights-free double.

use std::collections::HashMap;
use std::path::Path;

use candle_audio::candle_core::safetensors::MmapedSafetensors;
use candle_audio::candle_core::{DType, Device, Tensor};
use candle_audio::gen_core::{self, CancelFlag};
use candle_audio::vocos::{Vocos, VocosConfig};
use candle_audio::AudioError;
use candle_nn::VarBuilder;

use crate::config::Assets;
use crate::tokens::FRAMES_PER_SECOND;

/// The upsampled output rate (the final mix and stem rate).
pub const SAMPLE_RATE: u32 = 44_100;
/// Waveform samples per codec frame at [`SAMPLE_RATE`] — the ISTFT hop.
pub const SAMPLES_PER_FRAME: usize = (SAMPLE_RATE / FRAMES_PER_SECOND) as usize;
/// The vocal decoder inside the `xcodec_mini_infer` snapshot.
pub const VOCAL_CHECKPOINT: &str = "decoders/decoder_131000.safetensors";
/// The instrumental decoder inside the `xcodec_mini_infer` snapshot.
pub const INSTRUMENTAL_CHECKPOINT: &str = "decoders/decoder_151000.safetensors";
/// The backbone's LayerNorm epsilon (`VocosBackbone`'s `1e-6`).
const LN_EPS: f64 = 1e-6;

/// Which stem a decoder renders (each has its own Vocos checkpoint).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Track {
    /// The vocal stem.
    Vocals,
    /// The instrumental stem.
    Instrumental,
}

impl Track {
    /// The [`gen_core::AudioStem::name`] this track is published under.
    pub fn stem_name(self) -> &'static str {
        match self {
            Self::Vocals => "vocals",
            Self::Instrumental => "instrumental",
        }
    }
}

/// The vocoder seam (both decoders).
pub trait Vocoder: Send {
    /// Upsample one track's `[1, dim, frames]` embedding to mono [`SAMPLE_RATE`] audio. Must honor
    /// `cancel` by returning [`gen_core::Error::Canceled`].
    fn decode(
        &self,
        track: Track,
        embedding: &Tensor,
        cancel: &CancelFlag,
    ) -> gen_core::Result<Vec<f32>>;
}

/// Production loader: both Vocos decoders from the staged `xcodec_mini_infer` snapshot, on the
/// audio lane's default device (CPU, or Metal / CUDA under the `metal` / `cuda` features).
pub fn load(assets: &Assets) -> gen_core::Result<Box<dyn Vocoder>> {
    let device = candle_audio::default_device()?;
    Ok(Box::new(VocosUpsampler::load(
        assets.xcodec_root(),
        &device,
    )?))
}

/// The weights-free stub loader (the seam test's double).
pub fn load_stub(_assets: &Assets) -> gen_core::Result<Box<dyn Vocoder>> {
    Ok(Box::new(StubVocoder))
}

/// The two native Vocos decoders.
pub struct VocosUpsampler {
    vocals: Vocos,
    instrumental: Vocos,
    device: Device,
}

impl VocosUpsampler {
    /// Load [`VOCAL_CHECKPOINT`] and [`INSTRUMENTAL_CHECKPOINT`] under an `xcodec_mini_infer`
    /// snapshot root.
    pub fn load(root: &Path, device: &Device) -> Result<Self, AudioError> {
        Ok(Self {
            vocals: load_decoder(&root.join(VOCAL_CHECKPOINT), device)?,
            instrumental: load_decoder(&root.join(INSTRUMENTAL_CHECKPOINT), device)?,
            device: device.clone(),
        })
    }

    /// The decoder a track renders through.
    pub fn decoder(&self, track: Track) -> &Vocos {
        match track {
            Track::Vocals => &self.vocals,
            Track::Instrumental => &self.instrumental,
        }
    }
}

/// Load one Vocos decoder checkpoint (safetensors of upstream's `VocosDecoder` state dict), its
/// widths read from the tensors and the hop fixed at [`SAMPLES_PER_FRAME`].
pub fn load_decoder(checkpoint: &Path, device: &Device) -> Result<Vocos, AudioError> {
    let err = |m: String| AudioError::Msg(format!("vocos {}: {m}", checkpoint.display()));
    // SAFETY: the checkpoint is a caller-staged, read-only weight file; nothing in this process
    // writes it while the map is alive (the contract every candle mmap loader relies on).
    let st =
        unsafe { MmapedSafetensors::new(checkpoint) }.map_err(|e| err(format!("open: {e}")))?;
    let mut tensors = HashMap::new();
    for (name, _) in st.tensors() {
        let t = st
            .load(&name, device)
            .and_then(|t| t.to_dtype(DType::F32))
            .map_err(|e| err(format!("tensor {name:?}: {e}")))?;
        tensors.insert(name, t);
    }
    let get = |name: &str| {
        tensors
            .get(name)
            .ok_or_else(|| err(format!("missing tensor {name:?}")))
    };
    let (dim, input_channels, _) = get("backbone.embed.weight")?.dims3()?;
    let (intermediate_dim, _) = get("backbone.convnext.0.pwconv1.weight")?.dims2()?;
    let num_layers = (0..)
        .take_while(|i| tensors.contains_key(&format!("backbone.convnext.{i}.gamma")))
        .count();
    let window = get("head.istft.window")?.to_vec1::<f32>()?;
    let cfg = VocosConfig {
        input_channels,
        dim,
        intermediate_dim,
        num_layers,
        n_fft: window.len(),
        hop: SAMPLES_PER_FRAME,
        ln_eps: LN_EPS,
    };
    let vb = VarBuilder::from_tensors(tensors, DType::F32, device);
    Vocos::load(&vb, cfg, window, device).map_err(|e| err(e.to_string()))
}

impl Vocoder for VocosUpsampler {
    fn decode(
        &self,
        track: Track,
        embedding: &Tensor,
        cancel: &CancelFlag,
    ) -> gen_core::Result<Vec<f32>> {
        if cancel.is_cancelled() {
            return Err(gen_core::Error::Canceled);
        }
        let decoder = self.decoder(track);
        let want = decoder.config().input_channels;
        let (b, c, frames) = embedding.dims3().map_err(AudioError::from)?;
        if b != 1 || c != want {
            return Err(gen_core::Error::Msg(format!(
                "vocos {track:?}: expected a [1, {want}, frames] embedding, got {:?}",
                embedding.dims()
            )));
        }
        if frames == 0 {
            return Ok(Vec::new());
        }
        let x = embedding
            .to_device(&self.device)
            .and_then(|x| x.to_dtype(DType::F32))
            .map_err(AudioError::from)?;
        decoder
            .forward_cancellable(&x, &|| cancel.is_cancelled())
            .map_err(AudioError::from)?
            .ok_or(gen_core::Error::Canceled)
    }
}

/// **Stub Vocos** — the end-to-end seam test's weights-free double. Renders
/// [`SAMPLES_PER_FRAME`] samples per embedding frame of a tone keyed by the frame's embedding and
/// the track.
#[derive(Clone, Copy, Debug, Default)]
pub struct StubVocoder;

impl Vocoder for StubVocoder {
    fn decode(
        &self,
        track: Track,
        embedding: &Tensor,
        cancel: &CancelFlag,
    ) -> gen_core::Result<Vec<f32>> {
        if cancel.is_cancelled() {
            return Err(gen_core::Error::Canceled);
        }
        let rows = embedding
            .squeeze(0)
            .and_then(|e| e.to_vec2::<f32>())
            .map_err(|e| gen_core::Error::Msg(format!("stub vocoder: {e}")))?;
        let frames = rows.first().map_or(0, Vec::len);
        let mut out = Vec::with_capacity(frames * SAMPLES_PER_FRAME);
        for t in 0..frames {
            let key = rows.iter().map(|r| (r[t] * 1024.0) as u64).sum::<u64>() + track as u64;
            out.extend(crate::stub::tone(key, SAMPLES_PER_FRAME, SAMPLE_RATE));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_audio::candle_core::safetensors::save;

    /// Tiny synthetic widths with the production transform (n_fft 3528, hop 882) — the loader
    /// reads widths from the tensors, so a small checkpoint exercises the real load + decode path.
    const IN: usize = 5;
    const DIM: usize = 4;
    const MLP: usize = 6;
    const N_FFT: usize = 3528;

    fn det(shape: &[usize], seed: u32) -> Tensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|i| {
                (((i as u32).wrapping_mul(2_654_435_761) ^ seed) % 1000) as f32 / 10_000.0 - 0.05
            })
            .collect();
        Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
    }

    /// Write a tiny Vocos-shaped checkpoint (`layers` ConvNeXt blocks) at `path`.
    fn tiny_decoder(path: &Path, layers: usize, seed: u32) {
        let mut t: HashMap<String, Tensor> = HashMap::new();
        let mut put = |name: String, shape: &[usize]| {
            let s = seed.wrapping_add(t.len() as u32 * 7919);
            t.insert(name, det(shape, s));
        };
        put("backbone.embed.weight".into(), &[DIM, IN, 7]);
        put("backbone.embed.bias".into(), &[DIM]);
        for ln in ["backbone.norm", "backbone.final_layer_norm"] {
            put(format!("{ln}.weight"), &[DIM]);
            put(format!("{ln}.bias"), &[DIM]);
        }
        for i in 0..layers {
            let p = format!("backbone.convnext.{i}");
            put(format!("{p}.dwconv.weight"), &[DIM, 1, 7]);
            put(format!("{p}.dwconv.bias"), &[DIM]);
            put(format!("{p}.norm.weight"), &[DIM]);
            put(format!("{p}.norm.bias"), &[DIM]);
            put(format!("{p}.pwconv1.weight"), &[MLP, DIM]);
            put(format!("{p}.pwconv1.bias"), &[MLP]);
            put(format!("{p}.pwconv2.weight"), &[DIM, MLP]);
            put(format!("{p}.pwconv2.bias"), &[DIM]);
            put(format!("{p}.gamma"), &[DIM]);
        }
        put("head.out.weight".into(), &[N_FFT + 2, DIM]);
        put("head.out.bias".into(), &[N_FFT + 2]);
        t.insert(
            "head.istft.window".into(),
            Tensor::new(candle_audio::vocos::hann_window(N_FFT), &Device::Cpu).unwrap(),
        );
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        save(&t, path).unwrap();
    }

    fn tiny_snapshot() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        tiny_decoder(&dir.path().join(VOCAL_CHECKPOINT), 2, 1);
        tiny_decoder(&dir.path().join(INSTRUMENTAL_CHECKPOINT), 3, 2);
        dir
    }

    fn assets(xcodec: &Path) -> Assets {
        Assets {
            stage1: "/unused/s1".into(),
            stage2: "/unused/s2".into(),
            xcodec: xcodec.to_path_buf(),
        }
    }

    fn embedding(frames: usize) -> Tensor {
        det(&[1, IN, frames], 77)
    }

    #[test]
    fn the_hop_is_one_codec_frame_at_44k1() {
        assert_eq!(SAMPLES_PER_FRAME, 882);
    }

    #[test]
    fn production_load_upsamples_each_track_through_its_own_decoder_instead_of_refusing() {
        let snap = tiny_snapshot();
        let vocoder = load(&assets(snap.path())).expect("production vocoder loads staged weights");
        let cancel = CancelFlag::new();
        let v = vocoder
            .decode(Track::Vocals, &embedding(3), &cancel)
            .unwrap();
        let i = vocoder
            .decode(Track::Instrumental, &embedding(3), &cancel)
            .unwrap();
        assert_eq!(v.len(), 3 * SAMPLES_PER_FRAME);
        assert_eq!(i.len(), 3 * SAMPLES_PER_FRAME);
        assert!(v.iter().chain(&i).all(|s| s.is_finite()));
        assert_ne!(v, i, "the two tracks use different checkpoints");

        let up = VocosUpsampler::load(snap.path(), &Device::Cpu).unwrap();
        assert_eq!(up.decoder(Track::Vocals).config().num_layers, 2);
        assert_eq!(up.decoder(Track::Instrumental).config().num_layers, 3);
        assert_eq!(up.decoder(Track::Vocals).config().n_fft, N_FFT);
    }

    #[test]
    fn decode_refuses_malformed_embeddings_and_honors_cancel() {
        let snap = tiny_snapshot();
        let vocoder = load(&assets(snap.path())).unwrap();
        let ok = CancelFlag::new();
        assert!(vocoder
            .decode(Track::Vocals, &det(&[1, IN + 1, 2], 3), &ok)
            .is_err());
        assert!(vocoder
            .decode(Track::Vocals, &det(&[2, IN, 2], 3), &ok)
            .is_err());
        assert!(vocoder
            .decode(Track::Vocals, &embedding(0), &ok)
            .unwrap()
            .is_empty());
        let cancel = CancelFlag::new();
        cancel.cancel();
        assert!(matches!(
            vocoder.decode(Track::Vocals, &embedding(2), &cancel),
            Err(gen_core::Error::Canceled)
        ));
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
