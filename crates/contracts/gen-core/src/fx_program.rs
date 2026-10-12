//! The backend-neutral structure of a **SceneWorks fx-program** checkpoint
//! (`sceneworks-fx-program/1`, epic 2123 sc-24831): a straight-line list of NCHW-semantics ops
//! lowered from a torch `GraphModule` by `crates/media/mlx-gen/tools/fx_program.py` (which documents
//! the format) — today the MediaPipe FaceMesh-v2 landmark detector of the face-landmark loss. The
//! program travels as JSON in the checkpoint's safetensors `__metadata__.program`; the parameters are
//! the checkpoint's tensors (torch layout). Both native executors (`mlx-gen-face` / `candle-gen-face`
//! `program`) run this one parsed form, so an op is understood identically on both backends and an
//! unknown op is refused by name before anything runs.

use serde_json::Value;

use crate::{Error, Result};

/// The format tag a program checkpoint's `__metadata__.format` must carry.
pub const FORMAT: &str = "sceneworks-fx-program/1";

/// An op operand.
#[derive(Clone, Debug, PartialEq)]
pub enum Operand {
    /// A value produced by an earlier node (or the program input).
    Value(String),
    /// A stored tensor (`param:<key>`).
    Param(String),
    /// A literal (`scalar:<float>`).
    Scalar(f32),
}

/// Elementwise binary arithmetic (torch broadcasting).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// One op (all tensor semantics torch NCHW).
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    /// 2-D convolution; `weight` is OIHW.
    Conv {
        weight: String,
        bias: Option<String>,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
    },
    /// Per-channel PReLU; `slope` is `[C]`.
    Prelu {
        slope: String,
    },
    Relu,
    Sigmoid,
    Binary(BinaryOp),
    /// Max-pool (floor mode, `-inf` padding).
    MaxPool {
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    },
    /// Constant pad, `(before, after)` per N, C, H, W axis.
    Pad {
        pads: [(usize, usize); 4],
        value: f32,
    },
    /// torch `reshape` (one `-1` allowed).
    Reshape(Vec<i64>),
    /// Concatenate along a torch axis.
    Concat(i64),
}

/// One program node.
#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub op: Op,
    pub out: String,
    pub inputs: Vec<Operand>,
}

/// A parsed program.
#[derive(Clone, Debug, PartialEq)]
pub struct ProgramSpec {
    /// The single input's name.
    pub input: String,
    pub outputs: Vec<Operand>,
    pub nodes: Vec<Node>,
}

fn msg(m: impl Into<String>) -> Error {
    Error::Msg(format!("fx-program: {}", m.into()))
}

/// Largest padding (conv / max-pool / pad op) a program may ask for, per side.
pub const MAX_PAD: usize = 4096;

/// A non-negative integer that fits `i32` (the MLX executor's index type) and is `>= min`.
fn uint_min(v: &Value, what: &str, min: usize) -> Result<usize> {
    v.as_u64()
        .filter(|&n| n <= i32::MAX as u64 && n as usize >= min)
        .map(|n| n as usize)
        .ok_or_else(|| {
            msg(format!(
                "{what} must be an integer in [{min}, {}], got {v}",
                i32::MAX
            ))
        })
}

fn pair_min(v: &Value, what: &str, min: usize) -> Result<(usize, usize)> {
    let a = v
        .as_array()
        .filter(|a| a.len() == 2)
        .ok_or_else(|| msg(format!("{what} must be a pair, got {v}")))?;
    Ok((uint_min(&a[0], what, min)?, uint_min(&a[1], what, min)?))
}

fn pair(v: &Value, what: &str) -> Result<(usize, usize)> {
    pair_min(v, what, 0)
}

/// A padding pair, each side at most [`MAX_PAD`].
fn pad_pair(v: &Value, what: &str) -> Result<(usize, usize)> {
    let p = pair(v, what)?;
    if p.0 > MAX_PAD || p.1 > MAX_PAD {
        return Err(msg(format!("{what} {p:?} exceeds the {MAX_PAD} cap")));
    }
    Ok(p)
}

fn string(v: &Value, key: &str) -> Result<String> {
    v.get(key)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| msg(format!("node field '{key}' missing or not a string")))
}

fn operand(s: &str) -> Result<Operand> {
    Ok(if let Some(k) = s.strip_prefix("param:") {
        Operand::Param(k.to_owned())
    } else if let Some(v) = s.strip_prefix("scalar:") {
        Operand::Scalar(
            v.parse::<f32>()
                .map_err(|e| msg(format!("bad scalar '{v}': {e}")))?,
        )
    } else {
        Operand::Value(s.to_owned())
    })
}

fn operands(v: &Value, key: &str) -> Result<Vec<Operand>> {
    v.get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| msg(format!("'{key}' must be a list")))?
        .iter()
        .map(|s| {
            operand(
                s.as_str()
                    .ok_or_else(|| msg(format!("'{key}' entries are strings")))?,
            )
        })
        .collect()
}

impl ProgramSpec {
    /// Parse the program JSON (a checkpoint's `__metadata__.program`).
    pub fn parse(json: &str) -> Result<Self> {
        let v: Value =
            serde_json::from_str(json).map_err(|e| msg(format!("invalid program JSON: {e}")))?;
        Self::from_value(&v)
    }

    /// Parse an already-decoded program object.
    pub fn from_value(v: &Value) -> Result<Self> {
        let inputs = operands(v, "inputs")?;
        let input = match inputs.as_slice() {
            [Operand::Value(n)] => n.clone(),
            _ => {
                return Err(msg(format!(
                    "exactly one named input is supported, got {inputs:?}"
                )))
            }
        };
        let outputs = operands(v, "outputs")?;
        let mut nodes = Vec::new();
        for n in v
            .get("nodes")
            .and_then(Value::as_array)
            .ok_or_else(|| msg("'nodes' must be a list"))?
        {
            let kind = string(n, "op")?;
            let op = match kind.as_str() {
                "conv2d" => Op::Conv {
                    weight: string(n, "weight")?,
                    bias: n.get("bias").and_then(Value::as_str).map(str::to_owned),
                    stride: pair_min(&n["stride"], "stride", 1)?,
                    padding: pad_pair(&n["padding"], "padding")?,
                    dilation: pair_min(&n["dilation"], "dilation", 1)?,
                    groups: uint_min(&n["groups"], "groups", 1)?,
                },
                "prelu" => Op::Prelu {
                    slope: string(n, "slope")?,
                },
                "relu" => Op::Relu,
                "sigmoid" => Op::Sigmoid,
                "add" => Op::Binary(BinaryOp::Add),
                "sub" => Op::Binary(BinaryOp::Sub),
                "mul" => Op::Binary(BinaryOp::Mul),
                "div" => Op::Binary(BinaryOp::Div),
                "maxpool2d" => Op::MaxPool {
                    kernel: pair_min(&n["kernel"], "kernel", 1)?,
                    stride: pair_min(&n["stride"], "stride", 1)?,
                    padding: pad_pair(&n["padding"], "padding")?,
                },
                "pad" => {
                    let p = n["pads"]
                        .as_array()
                        .filter(|p| p.len() == 4)
                        .ok_or_else(|| msg("pad needs 4 axis pairs"))?;
                    let mut pads = [(0, 0); 4];
                    for (dst, src) in pads.iter_mut().zip(p) {
                        *dst = pad_pair(src, "pads")?;
                    }
                    Op::Pad {
                        pads,
                        value: n["value"].as_f64().unwrap_or(0.0) as f32,
                    }
                }
                "reshape" => Op::Reshape(
                    n["shape"]
                        .as_array()
                        .ok_or_else(|| msg("reshape needs a shape"))?
                        .iter()
                        .map(|d| {
                            d.as_i64()
                                .filter(|&d| d >= -1 && d <= i32::MAX as i64)
                                .ok_or_else(|| msg(format!("bad reshape dim {d}")))
                        })
                        .collect::<Result<Vec<_>>>()?,
                ),
                "concat" => Op::Concat(
                    n["axis"]
                        .as_i64()
                        .ok_or_else(|| msg("concat needs an integer axis"))?,
                ),
                other => {
                    return Err(msg(format!(
                        "unsupported op '{other}' (node {})",
                        string(n, "out").unwrap_or_default()
                    )))
                }
            };
            nodes.push(Node {
                op,
                out: string(n, "out")?,
                inputs: operands(n, "inputs")?,
            });
        }
        Ok(Self {
            input,
            outputs,
            nodes,
        })
    }

    /// Every parameter key the program reads, paired with whether it is a conv kernel (OIHW).
    pub fn param_keys(&self) -> Vec<(String, bool)> {
        let mut keys = Vec::new();
        for n in &self.nodes {
            match &n.op {
                Op::Conv { weight, bias, .. } => {
                    keys.push((weight.clone(), true));
                    if let Some(b) = bias {
                        keys.push((b.clone(), false));
                    }
                }
                Op::Prelu { slope } => keys.push((slope.clone(), false)),
                _ => {}
            }
            for i in &n.inputs {
                if let Operand::Param(k) = i {
                    keys.push((k.clone(), false));
                }
            }
        }
        keys.sort();
        keys.dedup();
        keys
    }

    /// Resolve a reshape `shape` (one `-1`) for a tensor of `elems` elements.
    pub fn resolve_shape(shape: &[i64], elems: usize) -> Result<Vec<usize>> {
        let known = shape
            .iter()
            .filter(|&&d| d != -1)
            .try_fold(1i64, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| msg(format!("reshape {shape:?} overflows")))?;
        let holes = shape.iter().filter(|&&d| d == -1).count();
        if holes > 1 || known <= 0 || shape.iter().any(|&d| d < -1) {
            return Err(msg(format!("bad reshape {shape:?}")));
        }
        let fill = elems as i64 / known;
        if fill * known != elems as i64 {
            return Err(msg(format!("cannot reshape {elems} elements to {shape:?}")));
        }
        Ok(shape
            .iter()
            .map(|&d| if d == -1 { fill as usize } else { d as usize })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_ops_and_refuses_unknown_ones_by_name() {
        let p = ProgramSpec::parse(
            r#"{"inputs":["x"],"outputs":["y","param:c"],"nodes":[
              {"op":"conv2d","out":"h","inputs":["x"],"weight":"w","bias":null,"stride":[2,2],
               "padding":[1,0],"dilation":[1,1],"groups":4},
              {"op":"add","out":"y","inputs":["h","scalar:0.5"]}]}"#,
        )
        .unwrap();
        assert_eq!(p.input, "x");
        assert_eq!(p.param_keys(), vec![("w".to_string(), true)]);
        assert_eq!(p.nodes[1].inputs[1], Operand::Scalar(0.5));
        assert_eq!(p.outputs[1], Operand::Param("c".into()));
        let err = ProgramSpec::parse(
            r#"{"inputs":["x"],"outputs":["y"],"nodes":[{"op":"gelu","out":"y","inputs":["x"]}]}"#,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("gelu"), "{err}");
        assert_eq!(
            ProgramSpec::resolve_shape(&[-1, 1, 1, 6], 12).unwrap(),
            [2, 1, 1, 6]
        );
        assert!(ProgramSpec::resolve_shape(&[-1, 5], 12).is_err());
    }

    /// Malformed numeric fields are refused at parse time, never left to panic an executor:
    /// zero stride / dilation / groups / kernel, integers past i32, oversized pads, and reshape
    /// products that overflow. Mutation: accept a zero stride (`pair` instead of `pair_min(.., 1)`)
    /// ⇒ red.
    #[test]
    fn malformed_numeric_fields_are_refused() {
        let conv = |stride: &str, pad: &str, dil: &str, groups: &str| {
            format!(
                r#"{{"inputs":["x"],"outputs":["y"],"nodes":[{{"op":"conv2d","out":"y","inputs":["x"],
                "weight":"w","bias":null,"stride":{stride},"padding":{pad},"dilation":{dil},
                "groups":{groups}}}]}}"#
            )
        };
        assert!(ProgramSpec::parse(&conv("[1,1]", "[0,0]", "[1,1]", "1")).is_ok());
        for bad in [
            conv("[0,1]", "[0,0]", "[1,1]", "1"),
            conv("[1,1]", "[0,0]", "[0,1]", "1"),
            conv("[1,1]", "[0,0]", "[1,1]", "0"),
            conv("[1,1]", "[4097,0]", "[1,1]", "1"),
            conv("[3000000000,1]", "[0,0]", "[1,1]", "1"),
        ] {
            assert!(ProgramSpec::parse(&bad).is_err(), "{bad}");
        }
        let pool = |k: &str| {
            format!(
                r#"{{"inputs":["x"],"outputs":["y"],"nodes":[{{"op":"maxpool2d","out":"y",
                "inputs":["x"],"kernel":{k},"stride":[1,1],"padding":[0,0]}}]}}"#
            )
        };
        assert!(ProgramSpec::parse(&pool("[0,2]")).is_err());
        let pad = r#"{"inputs":["x"],"outputs":["y"],"nodes":[{"op":"pad","out":"y","inputs":["x"],
            "pads":[[0,0],[0,0],[0,5000],[0,0]],"value":0}]}"#;
        assert!(ProgramSpec::parse(pad).is_err());
        let reshape = r#"{"inputs":["x"],"outputs":["y"],"nodes":[{"op":"reshape","out":"y",
            "inputs":["x"],"shape":[-2,4]}]}"#;
        assert!(ProgramSpec::parse(reshape).is_err());
        assert!(ProgramSpec::resolve_shape(&[i64::MAX, 4], 8).is_err());
    }
}
