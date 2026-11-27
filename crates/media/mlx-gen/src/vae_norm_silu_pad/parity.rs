//! SW-6142 parity harness for [`super`]: the fused routes against [`reference_conv_input`] over
//! the Wan z16 / LTX conv-input geometries, with a mismatch report (count, first index, max abs /
//! relative error, max ULP). Shared by the registered MLX parity tests (this crate, `mlx-gen-wan`,
//! `mlx-gen-ltx`) and the real-weight `vae_norm_silu_pad_proof` model proofs, so every gate runs
//! the same cases. Running it executes MLX kernels — callers are `#[ignore]`d GPU tests.

use mlx_rs::ops::split;
use mlx_rs::{Array, Dtype, Stream};

use super::{
    compiled_conv_input, launch_metal, plan_for, reference_conv_input, seeded, ChannelNorm,
    NormSiluPadOut, NormSiluPadRoute, PadSpec, TemporalPad, TOLERANCE_F32,
};
use crate::{Error, Result};

/// Physical layout of the case's `x`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// Row-contiguous NCTHW (a residual-add output).
    Ncthw,
    /// A channels-last buffer viewed as NCTHW (a conv output after its transpose back).
    NdhwcView,
}

/// The normalization a case runs.
#[derive(Clone, Copy, Debug)]
pub enum NormCase {
    /// Wan `RMS_norm` (eps `1e-12`), seeded γ.
    ChannelL2,
    /// LTX `pixel_norm` with this eps.
    Pixel(f32),
}

/// One parity geometry.
#[derive(Clone, Copy, Debug)]
pub struct Case {
    pub name: &'static str,
    /// `[B, C, T, H, W]`.
    pub shape: [i32; 5],
    pub dtype: Dtype,
    pub layout: Layout,
    pub norm: NormCase,
    pub temporal: TemporalPad,
    /// Cached prefix frames (Zeros only).
    pub cache_frames: i32,
    /// Pass the cache as an offset view (the second half of a split) instead of its own buffer.
    pub cache_view: bool,
    pub spatial: (i32, i32),
    pub tail: i32,
    /// Zero every channel of voxel (0, 0, 0, 0) so the eps clamp is exercised.
    pub zero_voxel: bool,
}

const WAN_EPS: f32 = 1e-12;

fn wan(name: &'static str, shape: [i32; 5], temporal: TemporalPad) -> Case {
    Case {
        name,
        shape,
        dtype: Dtype::Float32,
        layout: Layout::Ncthw,
        norm: NormCase::ChannelL2,
        temporal,
        cache_frames: 0,
        cache_view: false,
        spatial: (1, 1),
        tail: 0,
        zero_voxel: false,
    }
}

fn ltx(name: &'static str, shape: [i32; 5], eps: f32, temporal: TemporalPad) -> Case {
    Case {
        name,
        shape,
        dtype: Dtype::Float32,
        layout: Layout::NdhwcView,
        norm: NormCase::Pixel(eps),
        temporal,
        cache_frames: 0,
        cache_view: false,
        spatial: (1, 1),
        tail: 0,
        zero_voxel: false,
    }
}

const CAUSAL: TemporalPad = TemporalPad::Zeros { front: 2, back: 0 };

/// The Wan z16 geometries: decode residual / head convs (no cache, causal 2-frame zero pad) at
/// every decoder width and both input layouts, the chunked encoder's first chunk (`T = 1`, tail
/// 1), a cached 4-frame chunk (2 cached frames, no zero frames left, tail 2) and the 1-frame-cache
/// case (zero frame + cached frame), bf16/f16 inputs, a batch of 2 with odd spatial dims, the eps
/// clamp, and a sub-8-channel norm (γ in MLX `constant` memory).
pub fn wan_cases() -> Vec<Case> {
    vec![
        wan("wan_decode_c96_ncthw", [1, 96, 4, 9, 11], CAUSAL),
        Case {
            layout: Layout::NdhwcView,
            ..wan("wan_decode_c96_ndhwc_view", [1, 96, 4, 9, 11], CAUSAL)
        },
        Case {
            zero_voxel: true,
            ..wan("wan_decode_c192_eps_clamp", [1, 192, 2, 6, 5], CAUSAL)
        },
        Case {
            layout: Layout::NdhwcView,
            ..wan(
                "wan_decode_c384_latent_ndhwc_view",
                [1, 384, 3, 6, 7],
                CAUSAL,
            )
        },
        Case {
            tail: 1,
            ..wan(
                "wan_encode_first_chunk_t1_tail1",
                [1, 96, 1, 12, 10],
                CAUSAL,
            )
        },
        Case {
            cache_frames: 2,
            tail: 2,
            ..wan(
                "wan_encode_cached_chunk_cache2_tail2",
                [1, 192, 4, 6, 8],
                TemporalPad::Zeros { front: 0, back: 0 },
            )
        },
        Case {
            cache_frames: 1,
            tail: 2,
            layout: Layout::NdhwcView,
            ..wan(
                "wan_encode_cache1_zero1_tail2_ndhwc_view",
                [1, 96, 4, 5, 6],
                TemporalPad::Zeros { front: 1, back: 0 },
            )
        },
        Case {
            cache_frames: 2,
            cache_view: true,
            tail: 2,
            ..wan(
                "wan_encode_cached_chunk_cache_view",
                [1, 96, 4, 4, 6],
                TemporalPad::Zeros { front: 0, back: 0 },
            )
        },
        Case {
            dtype: Dtype::Bfloat16,
            ..wan("wan_decode_bf16_input", [1, 96, 3, 5, 7], CAUSAL)
        },
        Case {
            dtype: Dtype::Float16,
            layout: Layout::NdhwcView,
            ..wan("wan_decode_f16_input_ndhwc_view", [1, 192, 2, 5, 7], CAUSAL)
        },
        wan("wan_decode_batch2_odd", [2, 96, 3, 7, 5], CAUSAL),
        Case {
            spatial: (0, 0),
            ..wan(
                "wan_pointwise_no_spatial_pad",
                [1, 96, 2, 4, 4],
                TemporalPad::Zeros { front: 0, back: 0 },
            )
        },
        wan("wan_c4_constant_gamma", [1, 4, 3, 5, 5], CAUSAL),
    ]
}

/// The LTX geometries: decoder non-causal 3³ convs (one replicated frame each side) across the
/// decoder widths, the encoder's causal convs (two replicated front frames), a temporal-1 conv
/// (no temporal pad), f32 + bf16 inputs, both layouts and the eps floor.
pub fn ltx_cases() -> Vec<Case> {
    let symmetric = TemporalPad::Replicate { front: 1, back: 1 };
    let causal = TemporalPad::Replicate { front: 2, back: 0 };
    vec![
        ltx(
            "ltx_decode_c1024_symmetric",
            [1, 1024, 2, 3, 4],
            1e-8,
            symmetric,
        ),
        Case {
            layout: Layout::Ncthw,
            ..ltx(
                "ltx_decode_c512_symmetric_ncthw",
                [1, 512, 3, 5, 4],
                1e-8,
                symmetric,
            )
        },
        ltx(
            "ltx_decode_c128_symmetric",
            [1, 128, 5, 9, 8],
            1e-8,
            symmetric,
        ),
        Case {
            zero_voxel: true,
            ..ltx(
                "ltx_decode_c256_eps_floor",
                [1, 256, 2, 4, 6],
                1e-8,
                symmetric,
            )
        },
        ltx("ltx_encode_c128_causal", [1, 128, 3, 8, 6], 1e-6, causal),
        Case {
            layout: Layout::Ncthw,
            ..ltx("ltx_encode_c512_causal_t1", [1, 512, 1, 4, 4], 1e-6, causal)
        },
        ltx(
            "ltx_kt1_no_temporal_pad",
            [1, 128, 3, 4, 5],
            1e-8,
            TemporalPad::Replicate { front: 0, back: 0 },
        ),
        Case {
            dtype: Dtype::Bfloat16,
            ..ltx("ltx_decode_bf16_input", [1, 256, 2, 5, 3], 1e-8, symmetric)
        },
        ltx("ltx_decode_batch2", [2, 128, 2, 3, 5], 1e-8, symmetric),
    ]
}

/// Every Wan and LTX case.
pub fn all_cases() -> Vec<Case> {
    let mut cases = wan_cases();
    cases.extend(ltx_cases());
    cases
}

/// The cases whose temporal pad is asymmetric, for the wrong-side mutation control (a one-frame
/// replicate case is excluded: every replicated frame is that frame, so the side cannot matter).
pub fn mutation_cases() -> Vec<Case> {
    all_cases()
        .into_iter()
        .filter(|case| {
            let (front, back, replicate) = case.temporal.frames();
            front != back && case.cache_frames == 0 && !(replicate && case.shape[2] == 1)
        })
        .collect()
}

/// A case's materialized inputs.
pub struct Inputs {
    pub x: Array,
    pub gamma: Option<Array>,
    pub cache: Option<Array>,
}

impl Inputs {
    pub fn norm(&self, case: &Case) -> ChannelNorm<'_> {
        match case.norm {
            NormCase::ChannelL2 => ChannelNorm::ChannelL2 {
                gamma: self.gamma.as_ref().expect("ChannelL2 cases carry gamma"),
                eps: WAN_EPS,
            },
            NormCase::Pixel(eps) => ChannelNorm::Pixel { eps },
        }
    }

    pub fn spec(&self, case: &Case) -> PadSpec<'_> {
        PadSpec {
            temporal: case.temporal,
            cache: self.cache.as_ref(),
            ph: case.spatial.0,
            pw: case.spatial.1,
        }
    }
}

/// Build (and evaluate) a case's deterministic inputs.
pub fn inputs(case: &Case) -> Result<Inputs> {
    let [b, c, t, h, w] = case.shape;
    let n = (b * c * t * h * w) as usize;
    let mut data: Vec<f32> = (0..n)
        .map(|i| ((i as f32) * 0.618 + 0.25).sin() * 3.0)
        .collect();
    let voxels = (t * h * w) as usize;
    let x = match case.layout {
        Layout::Ncthw => {
            if case.zero_voxel {
                for ch in 0..c as usize {
                    data[ch * voxels] = 0.0;
                }
            }
            Array::from_slice(&data, &case.shape).as_dtype(case.dtype)?
        }
        Layout::NdhwcView => {
            if case.zero_voxel {
                for value in data.iter_mut().take(c as usize) {
                    *value = 0.0;
                }
            }
            Array::from_slice(&data, &[b, t, h, w, c])
                .as_dtype(case.dtype)?
                .transpose_axes(&[0, 4, 1, 2, 3])?
        }
    };
    let gamma = match case.norm {
        NormCase::ChannelL2 => Some(seeded(&[c, 1, 1, 1], 0.41, 0.7)?),
        NormCase::Pixel(_) => None,
    };
    let cache = if case.cache_frames > 0 {
        let tc = case.cache_frames;
        let make = |frames: i32| -> Result<Array> {
            crate::nn::silu(&seeded(&[b, c, frames, h, w], 0.29, -0.3)?)
        };
        Some(if case.cache_view {
            let whole = make(2 * tc)?;
            let mut halves = split(&whole, 2, 2)?;
            halves.swap_remove(1)
        } else {
            make(tc)?
        })
    } else {
        None
    };
    let mut live: Vec<&Array> = vec![&x];
    live.extend(gamma.iter());
    live.extend(cache.iter());
    mlx_rs::transforms::eval(live)?;
    Ok(Inputs { x, gamma, cache })
}

/// Elementwise comparison of two equally-shaped arrays (read back as f32 in logical order).
#[derive(Clone, Copy, Debug)]
pub struct Comparison {
    pub elements: usize,
    /// Whether dtype and shape agree (a dtype change is a mismatch even if the values agree).
    pub same_dtype_shape: bool,
    pub mismatches: usize,
    pub first_mismatch: Option<usize>,
    pub max_abs: f32,
    /// `max |got − want| / max |want|` (the E2 metric, [`crate::nn::max_rel_diff`]'s definition).
    pub max_rel: f32,
    pub max_ulp: u64,
}

impl Comparison {
    pub fn bitwise(&self) -> bool {
        self.same_dtype_shape && self.mismatches == 0
    }

    pub fn within_tolerance(&self) -> bool {
        self.same_dtype_shape && self.max_rel <= TOLERANCE_F32
    }
}

/// `x` as f32 host values in logical (row-major) order.
pub fn host_f32(x: &Array) -> Result<Vec<f32>> {
    let y = crate::array::contiguous(&x.as_dtype(Dtype::Float32)?)?;
    y.eval()?;
    Ok(y.as_slice::<f32>().to_vec())
}

fn ordered(v: f32) -> i64 {
    let bits = v.to_bits() as i32;
    if bits < 0 {
        -i64::from(bits & 0x7fff_ffff)
    } else {
        i64::from(bits)
    }
}

/// Compare `got` against `want`.
pub fn compare(got: &Array, want: &Array) -> Result<Comparison> {
    let same_dtype_shape = got.dtype() == want.dtype() && got.shape() == want.shape();
    if got.shape() != want.shape() {
        return Ok(Comparison {
            elements: want.size(),
            same_dtype_shape,
            mismatches: want.size(),
            first_mismatch: Some(0),
            max_abs: f32::INFINITY,
            max_rel: f32::INFINITY,
            max_ulp: u64::MAX,
        });
    }
    let g = host_f32(got)?;
    let w = host_f32(want)?;
    let peak = w.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let (mut mismatches, mut first, mut max_abs, mut max_ulp) = (0usize, None, 0.0f32, 0u64);
    for (i, (a, b)) in g.iter().zip(&w).enumerate() {
        let same = a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
        if !same {
            mismatches += 1;
            first.get_or_insert(i);
            let d = (a - b).abs();
            max_abs = if d.is_nan() {
                f32::INFINITY
            } else {
                max_abs.max(d)
            };
            max_ulp = max_ulp.max(ordered(*a).abs_diff(ordered(*b)));
        }
    }
    Ok(Comparison {
        elements: w.len(),
        same_dtype_shape,
        mismatches,
        first_mismatch: first,
        max_abs,
        max_rel: max_abs / peak.max(1e-12),
        max_ulp,
    })
}

/// One case × route result.
#[derive(Clone, Copy, Debug)]
pub struct Report {
    pub case: &'static str,
    pub route: NormSiluPadRoute,
    pub out: Comparison,
    pub tail: Option<Comparison>,
}

impl Report {
    pub fn bitwise(&self) -> bool {
        self.out.bitwise() && self.tail.as_ref().is_none_or(Comparison::bitwise)
    }

    pub fn within_tolerance(&self) -> bool {
        self.out.within_tolerance() && self.tail.as_ref().is_none_or(Comparison::within_tolerance)
    }

    /// One parse-friendly line.
    pub fn line(&self) -> String {
        let tail = self.tail.map_or_else(
            || "tail=-".to_string(),
            |t| {
                format!(
                    "tail_bitwise={} tail_mismatches={} tail_max_rel={:.3e}",
                    t.bitwise(),
                    t.mismatches,
                    t.max_rel
                )
            },
        );
        format!(
            "SW6142_PARITY case={} route={} bitwise={} elements={} mismatches={} first={:?} \
             max_abs={:.3e} max_rel={:.3e} max_ulp={} {tail}",
            self.case,
            self.route.label(),
            self.bitwise(),
            self.out.elements,
            self.out.mismatches,
            self.out.first_mismatch,
            self.out.max_abs,
            self.out.max_rel,
            self.out.max_ulp,
        )
    }
}

/// Produce the conv input of `case` by a fused `route` directly (bypassing the switch and the
/// probe, so a parity failure is the kernel's, not a silent fallback).
pub fn fused(
    inputs: &Inputs,
    case: &Case,
    spec: &PadSpec<'_>,
    route: NormSiluPadRoute,
) -> Result<NormSiluPadOut> {
    let norm = inputs.norm(case);
    let plan = plan_for(&inputs.x, &norm, spec, case.tail, true)
        .map_err(|r| Error::Msg(format!("{}: plan refused: {}", case.name, r.label())))?;
    match route {
        NormSiluPadRoute::Metal => {
            let denom = norm.denominator(&inputs.x)?;
            launch_metal(
                &inputs.x,
                &denom,
                &norm,
                spec,
                &plan,
                &Stream::task_local_or_default(),
            )
        }
        NormSiluPadRoute::Compiled => compiled_conv_input(&inputs.x, &norm, spec, &plan),
        NormSiluPadRoute::Reference => Err(Error::Msg(
            "parity::fused compares a fused route against the reference".into(),
        )),
    }
}

/// Run `case` through `route` and compare with [`reference_conv_input`].
pub fn run(case: &Case, route: NormSiluPadRoute) -> Result<Report> {
    let inputs = inputs(case)?;
    let spec = inputs.spec(case);
    let got = fused(&inputs, case, &spec, route)?;
    let want = reference_conv_input(&inputs.x, &inputs.norm(case), &spec, case.tail)?;
    let tail = match (&got.tail, &want.tail) {
        (Some(g), Some(w)) => Some(compare(g, w)?),
        (None, None) => None,
        _ => return Err(Error::Msg(format!("{}: tail presence differs", case.name))),
    };
    Ok(Report {
        case: case.name,
        route,
        out: compare(&got.ndhwc, &want.ndhwc)?,
        tail,
    })
}

/// The wrong-temporal-side mutation: the Metal kernel with the case's front/back pads swapped,
/// compared against the reference with the real pads. Parity **must** fail.
pub fn run_mutation(case: &Case) -> Result<Comparison> {
    let inputs = inputs(case)?;
    let spec = inputs.spec(case);
    let swapped = PadSpec {
        temporal: match case.temporal {
            TemporalPad::Zeros { front, back } => TemporalPad::Zeros {
                front: back,
                back: front,
            },
            TemporalPad::Replicate { front, back } => TemporalPad::Replicate {
                front: back,
                back: front,
            },
        },
        ..spec
    };
    let got = fused(&inputs, case, &swapped, NormSiluPadRoute::Metal)?;
    let want = reference_conv_input(&inputs.x, &inputs.norm(case), &spec, case.tail)?;
    compare(&got.ndhwc, &want.ndhwc)
}

/// Every case through `route`, printing one [`Report::line`] each.
pub fn run_all(cases: &[Case], route: NormSiluPadRoute) -> Result<Vec<Report>> {
    cases
        .iter()
        .map(|case| {
            let report = run(case, route)?;
            println!("{}", report.line());
            Ok(report)
        })
        .collect()
}
