//! The gen-core [`AudioEmbedder`] adapter for `laion/clap-htsat-unfused` (sc-12851): descriptor,
//! pinned-SHA hub resolution, weights load, and the joint audio/text embedding path (both towers →
//! `ClapProjectionLayer` → L2-normalized `Vec<f32>` in one CLIP-style space).

use std::path::PathBuf;
use std::sync::Mutex;

use candle_audio::candle_core::{DType, Device, Tensor};
use candle_audio::gen_core::runtime::{LoadSpec, WeightsSource};
use candle_audio::gen_core::{self, AudioEmbedder, AudioEmbedderDescriptor};
use candle_audio::hub::hf_get_pinned;
use candle_audio::{AudioError, Result as AudioResult};
use candle_nn::{linear, Linear, Module, VarBuilder};
use tokenizers::Tokenizer;

use crate::audio::AudioTower;
use crate::config;
use crate::mel;
use crate::text::TextTower;

/// The routing id.
pub const MODEL_ID: &str = "clap_htsat_unfused";
/// The provider family.
pub const FAMILY: &str = "audio-embed";
/// The joint embedding-space identifier.
pub const SPACE: &str = "clap-htsat-unfused";
/// Backend tag.
pub const BACKEND: &str = "candle";
/// HF repo of the pinned checkpoint.
pub const HUB_REPO: &str = "laion/clap-htsat-unfused";
/// Immutable commit SHA of the pinned checkpoint (Apache-2.0).
pub const HUB_REVISION: &str = "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a";

/// The license of the pinned LAION CLAP weight checkpoint (sc-13332) — surfaced for SceneWorks'
/// end-product licenses page. Apache-2.0 (permissive), verified against the
/// `laion/clap-htsat-unfused` model card.
pub const WEIGHT_LICENSE: candle_audio::gen_core::WeightLicense =
    candle_audio::gen_core::WeightLicense {
        spdx_id: "Apache-2.0",
        name: "Apache License 2.0",
        source_url: "https://huggingface.co/laion/clap-htsat-unfused",
        attribution: Some("CLAP (HTSAT-unfused) © LAION — licensed under Apache-2.0"),
        commercial_use: true,
        restriction: None,
    };

/// This provider's weight-license entry (keyed by [`MODEL_ID`]) for catalog aggregation.
pub const WEIGHT_LICENSE_ENTRY: candle_audio::gen_core::WeightLicenseEntry =
    candle_audio::gen_core::WeightLicenseEntry {
        provider_id: MODEL_ID,
        license: WEIGHT_LICENSE,
    };
/// Checkpoint file (pytorch pickle; loaded via `VarBuilder::from_pth`).
pub const WEIGHTS_FILE: &str = "pytorch_model.bin";
/// RoBERTa BPE tokenizer.
pub const TOKENIZER_FILE: &str = "tokenizer.json";
/// Architecture config (used by the preparer probe).
pub const CONFIG_FILE: &str = "config.json";

/// Stable identity + advertised shape, constructible without loading weights.
pub fn descriptor() -> AudioEmbedderDescriptor {
    AudioEmbedderDescriptor {
        id: MODEL_ID,
        family: FAMILY,
        backend: BACKEND,
        embedding_dim: config::PROJECTION_DIM,
        space: SPACE,
        mac_only: false,
    }
}

/// A `ClapProjectionLayer`: `linear1(768→512) → relu → linear2(512→512)`.
struct Projection {
    linear1: Linear,
    linear2: Linear,
}

impl Projection {
    fn load(vb: VarBuilder) -> AudioResult<Self> {
        Ok(Self {
            linear1: linear(
                config::TEXT_HIDDEN,
                config::PROJECTION_DIM,
                vb.pp("linear1"),
            )?,
            linear2: linear(
                config::PROJECTION_DIM,
                config::PROJECTION_DIM,
                vb.pp("linear2"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> AudioResult<Tensor> {
        let x = self.linear1.forward(x)?;
        let x = x.relu()?;
        Ok(self.linear2.forward(&x)?)
    }
}

struct Loaded {
    audio_tower: AudioTower,
    text_tower: TextTower,
    audio_proj: Projection,
    text_proj: Projection,
    tokenizer: Tokenizer,
    filterbank: Vec<f32>,
    device: Device,
}

/// The loaded (or lazily loadable) CLAP embedder.
pub struct ClapEmbedder {
    descriptor: AudioEmbedderDescriptor,
    root: PathBuf,
    loaded: Mutex<Option<Loaded>>,
}

fn lock_recover<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match m.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    }
}

/// L2-normalize a host vector (CLAP's native retrieval feature) — infallible; a zero vector is
/// returned unchanged.
fn l2_normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

impl ClapEmbedder {
    fn build(&self) -> AudioResult<Loaded> {
        let device = candle_audio::default_device()?;
        let weights = self.root.join(WEIGHTS_FILE);
        if !weights.exists() {
            return Err(AudioError::Msg(format!(
                "{MODEL_ID}: missing {WEIGHTS_FILE} under {}",
                self.root.display()
            )));
        }
        let tokenizer = Tokenizer::from_file(self.root.join(TOKENIZER_FILE))
            .map_err(|e| AudioError::Msg(format!("{MODEL_ID}: tokenizer load failed: {e}")))?;

        // Load the whole pickle checkpoint as f32.
        let vb = VarBuilder::from_pth(&weights, DType::F32, &device).map_err(AudioError::from)?;

        let audio_tower = AudioTower::load(vb.pp("audio_model").pp("audio_encoder"), &device)
            .map_err(AudioError::from)?;
        let text_tower = TextTower::load(vb.pp("text_model")).map_err(AudioError::from)?;
        let audio_proj = Projection::load(vb.pp("audio_projection"))?;
        let text_proj = Projection::load(vb.pp("text_projection"))?;
        let filterbank = mel::slaney_filterbank();

        Ok(Loaded {
            audio_tower,
            text_tower,
            audio_proj,
            text_proj,
            tokenizer,
            filterbank,
            device,
        })
    }

    fn with_loaded<T>(&self, f: impl FnOnce(&Loaded) -> AudioResult<T>) -> AudioResult<T> {
        let mut guard = lock_recover(&self.loaded);
        if guard.is_none() {
            *guard = Some(self.build()?);
        }
        f(guard.as_ref().expect("just loaded"))
    }

    fn embed_audio(&self, audio: &gen_core::media::AudioTrack) -> AudioResult<Vec<f32>> {
        self.with_loaded(|m| {
            let mel = mel::log_mel(
                &audio.samples,
                audio.sample_rate,
                audio.channels,
                &m.filterbank,
            )?;
            let pooled = m
                .audio_tower
                .forward(&mel, &m.device)
                .map_err(AudioError::from)?;
            let feats = m.audio_proj.forward(&pooled)?;
            let v = feats.flatten_all()?.to_vec1::<f32>()?;
            Ok(l2_normalize(v))
        })
    }

    fn embed_query(&self, text: &str) -> AudioResult<Vec<f32>> {
        self.with_loaded(|m| {
            let encoding = m
                .tokenizer
                .encode(text, true)
                .map_err(|e| AudioError::Msg(format!("{MODEL_ID}: tokenize failed: {e}")))?;
            let mut ids: Vec<u32> = encoding.get_ids().to_vec();
            if ids.is_empty() {
                return Err(AudioError::Msg(format!("{MODEL_ID}: empty text query")));
            }
            ids.truncate(config::TEXT_MAX_TOKENS);
            let pooled = m
                .text_tower
                .forward(&ids, &m.device)
                .map_err(AudioError::from)?;
            let feats = m.text_proj.forward(&pooled)?;
            let v = feats.flatten_all()?.to_vec1::<f32>()?;
            Ok(l2_normalize(v))
        })
    }
}

impl AudioEmbedder for ClapEmbedder {
    fn descriptor(&self) -> &AudioEmbedderDescriptor {
        &self.descriptor
    }

    fn embed(&self, audio: &gen_core::media::AudioTrack) -> gen_core::Result<Vec<f32>> {
        Ok(self.embed_audio(audio)?)
    }

    fn embed_text(&self, text: &str) -> gen_core::Result<Vec<f32>> {
        Ok(self.embed_query(text)?)
    }
}

/// Construct the provider from a [`LoadSpec`] (a pinned-SHA snapshot directory).
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn AudioEmbedder>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(format!(
                "{MODEL_ID} expects a snapshot directory (config.json + tokenizer.json + \
                 {WEIGHTS_FILE}), not a single file"
            )))
        }
    };
    if spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID} does not support on-the-fly quantization"
        )));
    }
    Ok(Box::new(ClapEmbedder {
        descriptor: descriptor(),
        root,
        loaded: Mutex::new(None),
    }))
}

candle_audio::gen_core::register_audio_embedder! {
    pub const REGISTRATION = descriptor => load
}

/// Resolve the pinned-SHA CLAP snapshot over the HF cache/network, returning its directory.
pub fn resolve_pinned_snapshot() -> AudioResult<WeightsSource> {
    let cfg = hf_get_pinned(HUB_REPO, HUB_REVISION, CONFIG_FILE)?;
    hf_get_pinned(HUB_REPO, HUB_REVISION, TOKENIZER_FILE)?;
    hf_get_pinned(HUB_REPO, HUB_REVISION, WEIGHTS_FILE)?;
    let dir = cfg
        .parent()
        .ok_or_else(|| AudioError::Msg("pinned snapshot has no parent dir".into()))?;
    Ok(WeightsSource::Dir(dir.to_path_buf()))
}
