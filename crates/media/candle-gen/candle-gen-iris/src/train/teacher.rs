//! The frozen **REPA teacher**, native Candle (story sc-25689) — the twin of
//! `mlx_gen_iris::train::teacher`: the `facebookresearch/dinov2` hub ViT's
//! `forward_features()["x_norm_patchtokens"]` and `repa.py`'s input path, in fp32 with no autograd
//! tape (the teacher never trains). See the MLX twin for the step-by-step definition; both read
//! the same backend-neutral resampling matrices from `gen_core::iris::train::repa`.

use std::collections::HashMap;
use std::path::Path;

use candle_gen::candle_core::{DType, Device, Tensor, D};
use candle_gen::candle_nn::ops::softmax_last_dim;
use candle_gen::gen_core::iris::train::repa::{
    self as contract, hf_sources, TeacherArch, TeacherFfn, TeacherNaming, IMAGENET_MEAN,
    IMAGENET_STD, SPATIAL_NORM_EPS, TEACHER_LN_EPS,
};
use candle_gen::{CandleError as Error, Result};

/// The loaded teacher: every tensor under its hub name, f32 on the training device.
pub struct Teacher {
    arch: TeacherArch,
    w: HashMap<String, Tensor>,
}

pub use candle_gen::gen_core::iris::train::repa::teacher_file;

fn mat(m: Vec<f32>, rows: usize, cols: usize, device: &Device) -> Result<Tensor> {
    Ok(Tensor::from_vec(m, (rows, cols), device)?)
}

impl Teacher {
    /// Load `arch`'s tensors from a hub `state_dict` or a Hugging Face `Dinov2Model` safetensors.
    pub fn load(path: &Path, arch: TeacherArch, device: &Device) -> Result<Self> {
        let file = teacher_file(path);
        let src = candle_gen::candle_core::safetensors::load(&file, device).map_err(|e| {
            Error::Msg(format!(
                "iris training: load the REPA teacher {}: {e}",
                file.display()
            ))
        })?;
        Self::from_tensors(arch, src)
    }

    /// Build from tensors already in memory (hub or Hugging Face names).
    pub fn from_tensors(arch: TeacherArch, mut src: HashMap<String, Tensor>) -> Result<Self> {
        let naming = TeacherNaming::detect(src.keys().map(String::as_str))?;
        let mut w = HashMap::new();
        for (key, shape) in arch.tensor_shapes() {
            let a = match naming {
                TeacherNaming::Hub => src.remove(&key).ok_or_else(|| {
                    Error::Msg(format!("iris training: the REPA teacher lacks {key}"))
                })?,
                TeacherNaming::HuggingFace => {
                    let parts: Vec<Tensor> = hf_sources(&key)
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
                        Tensor::cat(&parts, 0)?
                    }
                }
            }
            .to_dtype(DType::F32)?;
            if a.dims() != shape.as_slice() {
                return Err(Error::Msg(format!(
                    "iris training: REPA teacher tensor {key} is {:?}, {} implies {shape:?}",
                    a.dims(),
                    arch.entrypoint
                )));
            }
            w.insert(key, a.detach());
        }
        Ok(Self { arch, w })
    }

    pub fn arch(&self) -> &TeacherArch {
        &self.arch
    }

    fn get(&self, k: &str) -> &Tensor {
        &self.w[k]
    }

    fn layer_norm(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let mean = x.mean_keepdim(D::Minus1)?;
        let xc = x.broadcast_sub(&mean)?;
        let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
        let y = xc.broadcast_div(&(var + TEACHER_LN_EPS)?.sqrt()?)?;
        Ok(y.broadcast_mul(self.get(&format!("{prefix}.weight")))?
            .broadcast_add(self.get(&format!("{prefix}.bias")))?)
    }

    fn linear(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        Ok(
            x.broadcast_matmul(&self.get(&format!("{prefix}.weight")).t()?)?
                .broadcast_add(self.get(&format!("{prefix}.bias")))?,
        )
    }

    /// `_teacher_tokens(images, grid)`: `[B, 3, H, W]` clean images in `[−1, 1]` → `[B, gh·gw, D]`
    /// fp32 final-norm patch tokens.
    pub fn tokens(&self, images: &Tensor, grid: (usize, usize)) -> Result<Tensor> {
        let a = &self.arch;
        let device = images.device().clone();
        let (b, _, h, w) = images.dims4()?;
        let p = a.patch_size;
        let (gh, gw) = grid;
        let (mh, mw) = contract::teacher_image_resize(h, w, grid, p);
        let mh = mat(mh, gh * p, h, &device)?;
        let mw = mat(mw, gw * p, w, &device)?;
        let x = ((images.to_dtype(DType::F32)?.detach() + 1.0)? / 2.0)?;
        let x = mh.broadcast_matmul(&x)?.broadcast_matmul(&mw.t()?)?;
        let mean = Tensor::from_vec(IMAGENET_MEAN.to_vec(), (1, 3, 1, 1), &device)?;
        let std = Tensor::from_vec(IMAGENET_STD.to_vec(), (1, 3, 1, 1), &device)?;
        let x = x.broadcast_sub(&mean)?.broadcast_div(&std)?;
        let d = a.embed_dim;
        let patches = x
            .reshape((b, 3, gh, p, gw, p))?
            .permute((0, 2, 4, 1, 3, 5))?
            .contiguous()?
            .reshape((b, gh * gw, 3 * p * p))?;
        let pw = self
            .get("patch_embed.proj.weight")
            .reshape((d, 3 * p * p))?;
        let mut t = patches
            .broadcast_matmul(&pw.t()?)?
            .broadcast_add(self.get("patch_embed.proj.bias"))?;
        let cls = self
            .get("cls_token")
            .broadcast_as((b, 1, d))?
            .contiguous()?;
        t = Tensor::cat(&[&cls, &t], 1)?;
        t = t.broadcast_add(&self.pos_embed(grid, &device)?)?;
        if a.num_register_tokens > 0 {
            let reg = self
                .get("register_tokens")
                .broadcast_as((b, a.num_register_tokens, d))?
                .contiguous()?;
            let n = t.dim(1)?;
            t = Tensor::cat(&[&t.narrow(1, 0, 1)?, &reg, &t.narrow(1, 1, n - 1)?], 1)?;
        }
        for i in 0..a.depth {
            t = self.block(i, &t)?;
        }
        let t = self.layer_norm(&t, "norm")?;
        let skip = 1 + a.num_register_tokens;
        let n = t.dim(1)?;
        Ok(t.narrow(1, skip, n - skip)?)
    }

    fn pos_embed(&self, grid: (usize, usize), device: &Device) -> Result<Tensor> {
        let table = self.get("pos_embed");
        let Some((mh, mw)) = contract::pos_embed_resize(&self.arch, grid) else {
            return Ok(table.clone());
        };
        let m = self.arch.pos_grid;
        let d = self.arch.embed_dim;
        let (gh, gw) = grid;
        let cls = table.narrow(1, 0, 1)?;
        let patch = table
            .narrow(1, 1, m * m)?
            .reshape((m, m, d))?
            .permute((2, 0, 1))?
            .contiguous()?;
        let mh = mat(mh, gh, m, device)?;
        let mw = mat(mw, gw, m, device)?;
        let r = mh
            .broadcast_matmul(&patch)?
            .broadcast_matmul(&mw.t()?)?
            .permute((1, 2, 0))?
            .contiguous()?
            .reshape((1, gh * gw, d))?;
        Ok(Tensor::cat(&[&cls, &r], 1)?)
    }

    fn block(&self, i: usize, x: &Tensor) -> Result<Tensor> {
        let a = &self.arch;
        let pre = format!("blocks.{i}");
        let (b, n, _) = x.dims3()?;
        let heads = a.num_heads;
        let hd = a.embed_dim / heads;
        let h = self.layer_norm(x, &format!("{pre}.norm1"))?;
        let qkv = self
            .linear(&h, &format!("{pre}.attn.qkv"))?
            .reshape((b, n, 3, heads, hd))?
            .permute((2, 0, 3, 1, 4))?
            .contiguous()?;
        let (q, k, v) = (qkv.get(0)?, qkv.get(1)?, qkv.get(2)?);
        let scores = (q.matmul(&k.t()?.contiguous()?)? * (hd as f64).powf(-0.5))?;
        let attn = softmax_last_dim(&scores)?
            .matmul(&v)?
            .permute((0, 2, 1, 3))?
            .contiguous()?
            .reshape((b, n, heads * hd))?;
        let attn = self
            .linear(&attn, &format!("{pre}.attn.proj"))?
            .broadcast_mul(self.get(&format!("{pre}.ls1.gamma")))?;
        let x = (x + attn)?;
        let h = self.layer_norm(&x, &format!("{pre}.norm2"))?;
        let m = match a.ffn {
            TeacherFfn::Mlp => {
                let u = self.linear(&h, &format!("{pre}.mlp.fc1"))?.gelu_erf()?;
                self.linear(&u, &format!("{pre}.mlp.fc2"))?
            }
            TeacherFfn::SwiGlu => {
                let u = self.linear(&h, &format!("{pre}.mlp.w12"))?;
                let half = u.dim(D::Minus1)? / 2;
                let x1 = u.narrow(D::Minus1, 0, half)?;
                let x2 = u.narrow(D::Minus1, half, half)?;
                self.linear(&(x1.silu()? * x2)?, &format!("{pre}.mlp.w3"))?
            }
        };
        Ok((x + m.broadcast_mul(self.get(&format!("{pre}.ls2.gamma")))?)?)
    }
}

/// iREPA's `_spatial_normalize`: `(t − γ·mean_N t) / √(var_N t + 1e−6)` (biased variance over the
/// token axis, per channel).
pub fn spatial_normalize(tokens: &Tensor, gamma: f64) -> Result<Tensor> {
    let mean = tokens.mean_keepdim(1)?;
    let var = tokens.broadcast_sub(&mean)?.sqr()?.mean_keepdim(1)?;
    Ok(tokens
        .broadcast_sub(&(mean * gamma)?)?
        .broadcast_div(&(var + SPATIAL_NORM_EPS)?.sqrt()?)?)
}
