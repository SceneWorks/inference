//! The frozen **REPA teacher**, native MLX (story sc-25689): the `facebookresearch/dinov2`
//! hub ViT (`DinoVisionTransformer.forward_features()["x_norm_patchtokens"]` @
//! [`gen_core::iris::train::repa::DINOV2_CODE_REVISION`]) and `repa.py`'s input path, in fp32 and
//! outside every gradient trace.
//!
//! Input: the clean `[B, 3, H, W]` training images in `[−1, 1]` → `(x + 1)/2` → bicubic
//! (`align_corners=False`, no antialias) to `grid · 14` pixels → ImageNet normalization → patch
//! embed (the `14 × 14` stride-14 conv as an unfold + matmul) → class token (+ registers) + the
//! learned position table (bicubic-resized exactly as `interpolate_pos_encoding`, including the
//! hub's `+0.1` scale-factor kludge, or the antialiased resize of the `_reg` entrypoints) → `depth`
//! pre-norm blocks with LayerScale (fused-qkv attention, GELU MLP or SwiGLU) → final LayerNorm →
//! the patch tokens. iREPA's spatial normalization is applied by [`spatial_normalize`].

use std::collections::HashMap;
use std::path::Path;

use gen_core::iris::train::repa::{
    self as contract, hf_sources, TeacherArch, TeacherFfn, TeacherNaming, IMAGENET_MEAN,
    IMAGENET_STD, SPATIAL_NORM_EPS, TEACHER_LN_EPS,
};
use mlx_gen::gen_core;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_rs::fast::{layer_norm, scaled_dot_product_attention};
use mlx_rs::nn::silu;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{concatenate_axis, matmul, mean_axes, sqrt, var_axes};
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype, Stream};

/// The caller's stream (`with_new_default_stream`'s, e.g. the CPU stream), else the default
/// device's: mlx-rs's plain `index` ignores the task-local stream and always targets the default
/// device.
fn cur() -> Stream {
    Stream::task_local_or_default()
}

/// The loaded teacher: every tensor under its hub name, f32.
pub struct Teacher {
    arch: TeacherArch,
    w: HashMap<String, Array>,
}

/// Resolve a teacher resource path: a directory holding `model.safetensors`, or the file.
pub fn teacher_file(path: &Path) -> std::path::PathBuf {
    if path.is_dir() {
        path.join(contract::TEACHER_WEIGHTS_FILE)
    } else {
        path.to_path_buf()
    }
}

impl Teacher {
    /// Load `arch`'s tensors from a hub `state_dict` or a Hugging Face `Dinov2Model` safetensors,
    /// checking every tensor's presence and shape.
    pub fn load(path: &Path, arch: TeacherArch) -> Result<Self> {
        let file = teacher_file(path);
        let src = Weights::from_file(&file).map_err(|e| {
            Error::Msg(format!(
                "iris training: load the REPA teacher {}: {e}",
                file.display()
            ))
        })?;
        Self::from_weights(src, arch)
    }

    /// Build from tensors already in memory (hub or Hugging Face names).
    pub fn from_tensors(arch: TeacherArch, tensors: HashMap<String, Array>) -> Result<Self> {
        Self::from_weights(Weights::from_map(tensors), arch)
    }

    fn from_weights(mut src: Weights, arch: TeacherArch) -> Result<Self> {
        let keys: Vec<String> = src.keys().map(str::to_string).collect();
        let naming = TeacherNaming::detect(keys.iter().map(String::as_str))?;
        let mut w = HashMap::new();
        for (key, shape) in arch.tensor_shapes() {
            let a = match naming {
                TeacherNaming::Hub => src.remove(&key).ok_or_else(|| {
                    Error::Msg(format!("iris training: the REPA teacher lacks {key}"))
                })?,
                TeacherNaming::HuggingFace => {
                    let parts: Vec<Array> = hf_sources(&key)
                        .iter()
                        .map(|k| {
                            src.remove(k).ok_or_else(|| {
                                Error::Msg(format!(
                                    "iris training: the REPA teacher lacks {k} (for {key})"
                                ))
                            })
                        })
                        .collect::<Result<_>>()?;
                    if parts.len() == 1 {
                        parts.into_iter().next().expect("one")
                    } else {
                        concatenate_axis(&parts, 0)?
                    }
                }
            }
            .as_dtype(Dtype::Float32)?;
            let have: Vec<usize> = a.shape().iter().map(|&d| d as usize).collect();
            if have != shape {
                return Err(Error::Msg(format!(
                    "iris training: REPA teacher tensor {key} is {have:?}, {} implies {shape:?}",
                    arch.entrypoint
                )));
            }
            eval([&a])?;
            w.insert(key, a);
        }
        Ok(Self { arch, w })
    }

    pub fn arch(&self) -> &TeacherArch {
        &self.arch
    }

    fn get(&self, k: &str) -> &Array {
        &self.w[k]
    }

    /// `_teacher_tokens(images, grid)`: `[B, 3, H, W]` clean images in `[−1, 1]` → `[B, gh·gw, D]`
    /// fp32 final-norm patch tokens.
    pub fn tokens(&self, images: &Array, grid: (usize, usize)) -> Result<Array> {
        let a = &self.arch;
        let sh = images.shape();
        let (b, h, w) = (sh[0], sh[2] as usize, sh[3] as usize);
        let p = a.patch_size;
        let (gh, gw) = grid;
        let (mh, mw) = contract::teacher_image_resize(h, w, grid, p);
        let mh = Array::from_slice(&mh, &[(gh * p) as i32, h as i32]);
        let mw = Array::from_slice(&mw, &[(gw * p) as i32, w as i32]);
        let one = Array::from_f32(1.0);
        let x = images
            .as_dtype(Dtype::Float32)?
            .add(&one)?
            .divide(Array::from_f32(2.0))?;
        // Mh · x · Mwᵀ over the last two axes.
        let x = matmul(&matmul(&mh, &x)?, &mw.t())?;
        let mean = Array::from_slice(&IMAGENET_MEAN, &[1, 3, 1, 1]);
        let std = Array::from_slice(&IMAGENET_STD, &[1, 3, 1, 1]);
        let x = x.subtract(&mean)?.divide(&std)?;
        // patch embed: conv(k = s = p) ≡ unfold (channel-major patch vectors) · Wᵀ + b.
        let d = a.embed_dim as i32;
        let (ghi, gwi, pi) = (gh as i32, gw as i32, p as i32);
        let patches = x
            .reshape(&[b, 3, ghi, pi, gwi, pi])?
            .transpose_axes(&[0, 2, 4, 1, 3, 5])?
            .reshape(&[b, ghi * gwi, 3 * pi * pi])?;
        let pw = self.get("patch_embed.proj.weight").reshape(&[d, -1])?;
        let mut t = matmul(&patches, &pw.t())?.add(self.get("patch_embed.proj.bias"))?;
        let cls = mlx_rs::ops::broadcast_to(self.get("cls_token"), &[b, 1, d])?;
        t = concatenate_axis(&[&cls, &t], 1)?;
        t = t.add(&self.pos_embed(grid)?)?;
        if a.num_register_tokens > 0 {
            let reg = mlx_rs::ops::broadcast_to(
                self.get("register_tokens"),
                &[b, a.num_register_tokens as i32, d],
            )?;
            let head = t.index_device((.., ..1, ..), cur());
            let tail = t.index_device((.., 1.., ..), cur());
            t = concatenate_axis(&[&head, &reg, &tail], 1)?;
        }
        for i in 0..a.depth {
            t = self.block(i, &t)?;
        }
        let t = layer_norm(
            &t,
            Some(self.get("norm.weight")),
            Some(self.get("norm.bias")),
            TEACHER_LN_EPS as f32,
        )?;
        Ok(t.index_device((.., (1 + a.num_register_tokens) as i32.., ..), cur()))
    }

    /// `interpolate_pos_encoding` for a `gh × gw` patch grid → `[1, 1 + gh·gw, D]`.
    fn pos_embed(&self, grid: (usize, usize)) -> Result<Array> {
        let table = self.get("pos_embed");
        let Some((mh, mw)) = contract::pos_embed_resize(&self.arch, grid) else {
            return Ok(table.clone());
        };
        let m = self.arch.pos_grid as i32;
        let d = self.arch.embed_dim as i32;
        let (gh, gw) = (grid.0 as i32, grid.1 as i32);
        let cls = table.index_device((.., ..1, ..), cur());
        // [1, M², D] → [D, M, M] → Mh · P · Mwᵀ → [D, gh, gw] → [1, gh·gw, D]
        let patch = table
            .index_device((.., 1.., ..), cur())
            .reshape(&[m, m, d])?
            .transpose_axes(&[2, 0, 1])?;
        let mh = Array::from_slice(&mh, &[gh, m]);
        let mw = Array::from_slice(&mw, &[gw, m]);
        let r = matmul(&matmul(&mh, &patch)?, &mw.t())?
            .transpose_axes(&[1, 2, 0])?
            .reshape(&[1, gh * gw, d])?;
        Ok(concatenate_axis(&[&cls, &r], 1)?)
    }

    fn block(&self, i: usize, x: &Array) -> Result<Array> {
        let a = &self.arch;
        let g = |leaf: &str| self.get(&format!("blocks.{i}.{leaf}"));
        let sh = x.shape();
        let (b, n) = (sh[0], sh[1]);
        let heads = a.num_heads as i32;
        let hd = (a.embed_dim / a.num_heads) as i32;
        let lin = |x: &Array, w: &str| -> Result<Array> {
            Ok(matmul(x, &g(&format!("{w}.weight")).t())?.add(g(&format!("{w}.bias")))?)
        };
        let h = layer_norm(
            x,
            Some(g("norm1.weight")),
            Some(g("norm1.bias")),
            TEACHER_LN_EPS as f32,
        )?;
        let qkv = lin(&h, "attn.qkv")?
            .reshape(&[b, n, 3, heads, hd])?
            .transpose_axes(&[2, 0, 3, 1, 4])?;
        let (q, k, v) = (
            qkv.index_device(0, cur()),
            qkv.index_device(1, cur()),
            qkv.index_device(2, cur()),
        );
        let attn = scaled_dot_product_attention(&q, &k, &v, (hd as f32).powf(-0.5), None, None)?
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, n, heads * hd])?;
        let attn = lin(&attn, "attn.proj")?.multiply(g("ls1.gamma"))?;
        let x = x.add(&attn)?;
        let h = layer_norm(
            &x,
            Some(g("norm2.weight")),
            Some(g("norm2.bias")),
            TEACHER_LN_EPS as f32,
        )?;
        let m = match a.ffn {
            TeacherFfn::Mlp => {
                let u = lin(&h, "mlp.fc1")?;
                lin(&mlx_rs::nn::gelu(&u)?, "mlp.fc2")?
            }
            TeacherFfn::SwiGlu => {
                let u = lin(&h, "mlp.w12")?;
                let halves = u.split(2, -1)?;
                lin(&silu(&halves[0])?.multiply(&halves[1])?, "mlp.w3")?
            }
        };
        Ok(x.add(&m.multiply(g("ls2.gamma"))?)?)
    }
}

/// iREPA's `_spatial_normalize`: `(t − γ·mean_N t) / √(var_N t + 1e−6)` (biased variance over the
/// token axis, per channel).
pub fn spatial_normalize(tokens: &Array, gamma: f64) -> Result<Array> {
    let mean = mean_axes(tokens, &[1], true)?;
    let var = var_axes(tokens, &[1], true, 0)?;
    Ok(tokens
        .subtract(&mean.multiply(Array::from_f32(gamma as f32))?)?
        .divide(&sqrt(&var.add(Array::from_f32(SPATIAL_NORM_EPS as f32))?)?)?)
}
