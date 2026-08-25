//! Shared golden-fixture loader for the Bernini planner CPU parity tests. Reads the committed
//! `tests/fixtures/*.safetensors` goldens (dumped from the reference by the MLX lane's
//! `tools/dump_bernini_*_golden.py`, reused byte-for-byte here) without depending on candle's
//! supported-dtype set — the goldens carry `I32`/`I8` tensors candle can't natively load, so we parse
//! the safetensors container by hand and expose typed accessors + a f32-only `VarBuilder`.

#![allow(dead_code)]

use std::collections::HashMap;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;

struct Entry {
    dtype: String,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

pub struct Golden {
    data: Vec<u8>,
    data_start: usize,
    entries: HashMap<String, Entry>,
    meta: HashMap<String, String>,
}

impl Golden {
    /// Load a fixture from `tests/fixtures/<name>.safetensors`.
    pub fn load(name: &str) -> Golden {
        let path = format!(
            "{}/tests/fixtures/{}.safetensors",
            env!("CARGO_MANIFEST_DIR"),
            name
        );
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        let hlen = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let header: serde_json::Value =
            serde_json::from_slice(&bytes[8..8 + hlen]).expect("parse safetensors header");
        let obj = header.as_object().expect("header object");

        let mut entries = HashMap::new();
        let mut meta = HashMap::new();
        for (k, v) in obj {
            if k == "__metadata__" {
                for (mk, mv) in v.as_object().expect("metadata object") {
                    meta.insert(mk.clone(), mv.as_str().unwrap_or_default().to_string());
                }
                continue;
            }
            let dtype = v["dtype"].as_str().unwrap().to_string();
            let shape: Vec<usize> = v["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect();
            let offs = v["data_offsets"].as_array().unwrap();
            entries.insert(
                k.clone(),
                Entry {
                    dtype,
                    shape,
                    start: offs[0].as_u64().unwrap() as usize,
                    end: offs[1].as_u64().unwrap() as usize,
                },
            );
        }
        Golden {
            data_start: 8 + hlen,
            data: bytes,
            entries,
            meta,
        }
    }

    pub fn has(&self, key: &str) -> bool {
        self.entries.contains_key(key)
    }

    pub fn meta(&self, key: &str) -> Option<&str> {
        self.meta.get(key).map(|s| s.as_str())
    }

    pub fn meta_req(&self, key: &str) -> &str {
        self.meta(key)
            .unwrap_or_else(|| panic!("missing metadata {key}"))
    }

    pub fn shape(&self, key: &str) -> Vec<usize> {
        self.entries
            .get(key)
            .unwrap_or_else(|| panic!("missing tensor {key}"))
            .shape
            .clone()
    }

    fn raw(&self, key: &str) -> &[u8] {
        let e = self
            .entries
            .get(key)
            .unwrap_or_else(|| panic!("missing tensor {key}"));
        &self.data[self.data_start + e.start..self.data_start + e.end]
    }

    fn dtype(&self, key: &str) -> &str {
        &self.entries.get(key).unwrap().dtype
    }

    pub fn f32(&self, key: &str) -> Vec<f32> {
        assert_eq!(self.dtype(key), "F32", "{key} is not F32");
        self.raw(key)
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    pub fn i32(&self, key: &str) -> Vec<i32> {
        assert_eq!(self.dtype(key), "I32", "{key} is not I32");
        self.raw(key)
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    /// i64 view of an integer tensor (accepts I32 or I64 storage).
    pub fn i64(&self, key: &str) -> Vec<i64> {
        match self.dtype(key) {
            "I32" => self.i32(key).into_iter().map(|x| x as i64).collect(),
            "I64" => self
                .raw(key)
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            d => panic!("{key} is {d}, not an integer tensor"),
        }
    }

    pub fn i8(&self, key: &str) -> Vec<i8> {
        assert_eq!(self.dtype(key), "I8", "{key} is not I8");
        self.raw(key).iter().map(|&b| b as i8).collect()
    }

    pub fn bools_from_i32(&self, key: &str) -> Vec<bool> {
        self.i32(key).into_iter().map(|x| x != 0).collect()
    }

    /// A candle F32 [shape] tensor for an F32 golden entry.
    pub fn tensor(&self, key: &str, dev: &Device) -> Tensor {
        let v = self.f32(key);
        let shape = self.shape(key);
        Tensor::from_vec(v, shape, dev).unwrap()
    }

    /// A `VarBuilder` over **all F32 tensors** in the fixture (weights + f32 io), rooted at the file
    /// top level. Navigate to the model / connector namespace with `.pp("w.model")` etc.
    pub fn var_builder(&self, dev: &Device) -> VarBuilder<'static> {
        self.var_builder_dtype(dev, DType::F32)
    }

    /// Like [`Golden::var_builder`], but every weight is cast to `dtype`. Used to exercise the
    /// production bf16 weight layout against f32 inputs (sc-11150 — the vision-tower dtype contract).
    pub fn var_builder_dtype(&self, dev: &Device, dtype: DType) -> VarBuilder<'static> {
        let mut map: HashMap<String, Tensor> = HashMap::new();
        for (k, e) in &self.entries {
            if e.dtype == "F32" {
                map.insert(k.clone(), self.tensor(k, dev).to_dtype(dtype).unwrap());
            }
        }
        VarBuilder::from_tensors(map, dtype, dev)
    }
}

/// (peak abs diff, peak-relative `max|Δ|/max|b|`).
pub fn errors(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    let peak = b.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-12);
    let max_diff = a
        .iter()
        .zip(b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    (max_diff, max_diff / peak)
}

pub fn flat_f32(t: &Tensor) -> Vec<f32> {
    t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

// --- Cross-backend fixture geometry (sc-19496) ---------------------------------------------------
//
// Ten of the fixture files under `tests/fixtures/` here are committed **byte-identical** to the file
// `mlx-gen-bernini` commits under the same name (`assembly`, `clip_diff`, `handoff`, `mar`,
// `process`, `qwen_backbone`, `template`, `vision_tower`, `vit_guidance` and `vit_preprocess`
// goldens). Both lanes load the same bytes, so a drift in either lane's hand-typed geometry leaves
// both lanes internally consistent and both parity suites green while the two backends compare
// tensors dumped at one shape against a model built at another. Nothing could see that: the two
// crates cannot import each other, because `mlx-gen-*` builds on macOS only.
//
// Most of this family's fixture geometry does not need pinning here at all, and deliberately is not:
// the goldens carry their own `__metadata__` and both lanes parse it (`Golden::meta_req` here,
// `Weights::metadata` there), so the fixture's own bytes are the single source and drift is
// impossible by construction. What follows is the remainder — the values both lanes genuinely
// hand-type because the golden does not record them. `check_cross_backend_geometry` in
// `scripts/check-workspace.py` compares every `SHARED_FIXTURE_*` declaration under this crate's
// `tests/` against the MLX crate's, by name set and by value.

/// Assembly-fixture backbone depth: 0 layers, so only the token embedding is exercised.
pub const SHARED_FIXTURE_ASSEMBLY_NUM_LAYERS: usize = 0;
/// Assembly-fixture attention heads.
pub const SHARED_FIXTURE_ASSEMBLY_NUM_HEADS: usize = 2;
/// Assembly-fixture key/value heads (GQA).
pub const SHARED_FIXTURE_ASSEMBLY_NUM_KV_HEADS: usize = 1;
/// Assembly-fixture per-head width.
pub const SHARED_FIXTURE_ASSEMBLY_HEAD_DIM: usize = 8;
/// Assembly-fixture feed-forward width.
pub const SHARED_FIXTURE_ASSEMBLY_INTERMEDIATE_SIZE: usize = 32;
/// Assembly-fixture RMSNorm epsilon.
pub const SHARED_FIXTURE_ASSEMBLY_RMS_NORM_EPS: f64 = 1e-6;
/// Assembly-fixture RoPE base.
pub const SHARED_FIXTURE_ASSEMBLY_ROPE_THETA: f64 = 1_000_000.0;
/// Assembly-fixture MRoPE per-axis (T/H/W) frequency counts.
pub const SHARED_FIXTURE_ASSEMBLY_MROPE_SECTION: [usize; 3] = [1, 2, 1];

/// ViT-guidance fixture: the image-conditioned guidance weight the golden was dumped at.
pub const SHARED_FIXTURE_VIT_GUIDANCE_W_IMG: f32 = 4.5;
/// ViT-guidance fixture: the text-conditioned guidance weight.
pub const SHARED_FIXTURE_VIT_GUIDANCE_W_TXT: f32 = 4.0;
/// ViT-guidance fixture: the target-conditioned guidance weight.
pub const SHARED_FIXTURE_VIT_GUIDANCE_W_TGT: f32 = 3.0;
/// ViT-guidance fixture: the video-conditioned guidance weight (the `rv2v` chain's first rung).
pub const SHARED_FIXTURE_VIT_GUIDANCE_W_VID: f32 = 1.25;
/// ViT-guidance fixture: `apg_delta`'s eta (the parallel-component retention).
pub const SHARED_FIXTURE_VIT_GUIDANCE_APG_ETA: f32 = 0.2;
/// ViT-guidance fixture: `apg_delta`'s norm threshold.
pub const SHARED_FIXTURE_VIT_GUIDANCE_APG_NORM_THRESHOLD: f32 = 1.0;

/// Template-fixture task mixes, in the order the golden dumps them.
pub const SHARED_FIXTURE_TEMPLATE_TASKS: [&str; 4] = ["t2i", "i2i", "r2v", "rv2v"];
/// Template-fixture prompts, one per task.
pub const SHARED_FIXTURE_TEMPLATE_PROMPTS: [&str; 4] = ["a cat", "edit", "subj", "edit v"];
/// Template-fixture input reference images `(h, w)`, one list per task.
pub const SHARED_FIXTURE_TEMPLATE_INPUT_IMAGE_HW: [&[(i64, i64)]; 4] =
    [&[], &[(48, 72)], &[(72, 48)], &[]];
/// Template-fixture input reference-video counts, one per task.
pub const SHARED_FIXTURE_TEMPLATE_INPUT_VIDEO_COUNT: [usize; 4] = [0, 0, 0, 1];
/// Template-fixture output frame counts, one per task.
pub const SHARED_FIXTURE_TEMPLATE_OUTPUT_T: [i64; 4] = [1, 1, 9, 9];
/// Template-fixture output height.
pub const SHARED_FIXTURE_TEMPLATE_OUTPUT_H: i64 = 64;
/// Template-fixture output width.
pub const SHARED_FIXTURE_TEMPLATE_OUTPUT_W: i64 = 64;
/// Template-fixture image token counts, one list per task — grids match the process golden, so
/// `token_num = t·(h/2)·(w/2)`.
pub const SHARED_FIXTURE_TEMPLATE_IMAGE_TOKEN_NUMS: [&[i64]; 4] = [&[4], &[6, 4], &[6], &[]];
/// Template-fixture video token counts, one list per task.
pub const SHARED_FIXTURE_TEMPLATE_VIDEO_TOKEN_NUMS: [&[i64]; 4] = [&[], &[], &[12], &[12, 20]];
