//! Bounded packed-attention reference used by the Metal reader contract.
//!
//! This is deliberately a small, allocation-bounded reference: it reconstructs one K/V value at
//! a time while accumulating fp32 dot/softmax state. The device implementation must preserve this
//! semantic order without materialising dense K/V or an S_q×S_kv score matrix.
use crate::error::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedAttentionShape {
    pub batch: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub query_len: usize,
    pub kv_len: usize,
    pub head_dim: usize,
}

pub fn validate(shape: PackedAttentionShape) -> Result<()> {
    if shape.batch == 0
        || shape.query_heads == 0
        || shape.kv_heads == 0
        || shape.query_len == 0
        || shape.kv_len == 0
        || shape.head_dim == 0
        || !shape.query_heads.is_multiple_of(shape.kv_heads)
    {
        return Err(Error::Config("unsupported packed attention shape".into()));
    }
    Ok(())
}

/// Reference decode for already-dequantized scalar accessors. `key` and `value` are called for
/// each tile element, so this API is suitable for a packed reader that reconstructs registers only.
pub fn attention_f32(
    shape: PackedAttentionShape,
    query: &[f32],
    key: impl Fn(usize, usize, usize, usize) -> f32,
    value: impl Fn(usize, usize, usize, usize) -> f32,
    scale: f32,
) -> Result<Vec<f32>> {
    validate(shape)?;
    if query.len() != shape.batch * shape.query_heads * shape.query_len * shape.head_dim {
        return Err(Error::Config(
            "packed attention query shape mismatch".into(),
        ));
    }
    let mut out = vec![0.0; query.len()];
    let groups = shape.query_heads / shape.kv_heads;
    for b in 0..shape.batch {
        for qh in 0..shape.query_heads {
            let kh = qh / groups;
            for qi in 0..shape.query_len {
                let base = ((b * shape.query_heads + qh) * shape.query_len + qi) * shape.head_dim;
                let mut max = f32::NEG_INFINITY;
                let mut norm = 0.0;
                let mut weighted = vec![0.0f32; shape.head_dim];
                for ks in 0..shape.kv_len {
                    let mut dot = 0.0;
                    for d in 0..shape.head_dim {
                        dot += query[base + d] * key(b, kh, ks, d);
                    }
                    let score = dot * scale;
                    if score > max {
                        max = score;
                    }
                }
                for ks in 0..shape.kv_len {
                    let mut dot = 0.0;
                    for d in 0..shape.head_dim {
                        dot += query[base + d] * key(b, kh, ks, d);
                    }
                    norm += (dot * scale - max).exp();
                }
                for ks in 0..shape.kv_len {
                    let mut dot = 0.0;
                    for d in 0..shape.head_dim {
                        dot += query[base + d] * key(b, kh, ks, d);
                    }
                    let weight = (dot * scale - max).exp() / norm;
                    for (d, output) in weighted.iter_mut().enumerate() {
                        *output += weight * value(b, kh, ks, d);
                    }
                }
                out[base..base + shape.head_dim].copy_from_slice(&weighted);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gqa_and_tail_shape_are_accepted_without_score_matrix() {
        let shape = PackedAttentionShape {
            batch: 1,
            query_heads: 4,
            kv_heads: 2,
            query_len: 3,
            kv_len: 5,
            head_dim: 3,
        };
        let q = vec![1.0; 36];
        let out = attention_f32(shape, &q, |_, _, _, _| 1.0, |_, _, _, d| d as f32, 0.5).unwrap();
        assert_eq!(out.len(), q.len());
    }
    #[test]
    fn invalid_gqa_fails_closed() {
        assert!(validate(PackedAttentionShape {
            batch: 1,
            query_heads: 3,
            kv_heads: 2,
            query_len: 1,
            kv_len: 1,
            head_dim: 64
        })
        .is_err());
    }
}
