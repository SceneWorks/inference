//! The shared block-draft tree builder ([`core_llm::build_lattice_tree`]) driven by MLX's draft
//! sampler over a synthetic lattice — host arithmetic on hand-written scores, no model: every
//! sampled row carries exactly the `q` of its parent's own branch (its sender column, the
//! penalties over its own path, the constraint advanced along it), the sampled child is tried
//! before its siblings, siblings are distinct point masses, and a greedy tree's first child is
//! the draft's choice for its own sender.

use mlx_rs::Array;

use core_llm::{build_lattice_tree, DraftLattice, DraftTree};

use crate::decode::engine::{DraftSampler, MlxSampler, RewindableConstraintMask, TokenSampler};
use crate::decode::stream::{ConstraintMask, GenerationConfig};
use crate::primitives::sampler::SamplingParams;

const WIDTH: usize = 12;
const ROWS: usize = 3;
const K: usize = 4;

/// Three rows of four candidates each, every row's scores different for every sender.
fn lattice() -> (Vec<Vec<i32>>, Vec<f32>) {
    let candidates: Vec<Vec<i32>> = (0..ROWS)
        .map(|r| {
            (0..K as i32)
                .map(|c| (r as i32 + 2 * c) % WIDTH as i32)
                .collect()
        })
        .map(|mut row: Vec<i32>| {
            row.sort_unstable();
            row
        })
        .collect();
    let table = (0..ROWS * K * K)
        .map(|i| ((i * 7 + 3) % 11) as f32 * 0.3 - 1.0)
        .collect();
    (candidates, table)
}

/// A constraint allowing no token twice along a branch (from the proposal's start).
struct NoRepeat {
    accepted: Vec<i32>,
    allow: Vec<bool>,
}

impl ConstraintMask for NoRepeat {
    fn allowed(&mut self) -> &[bool] {
        self.allow = (0..WIDTH as i32)
            .map(|t| !self.accepted.contains(&t))
            .collect();
        &self.allow
    }
    fn accept(&mut self, token: i32) {
        self.accepted.push(token);
    }
}

impl RewindableConstraintMask for NoRepeat {
    fn checkpoint(&self) -> usize {
        self.accepted.len()
    }
    fn rewind(&mut self, checkpoint: usize) {
        self.accepted.truncate(checkpoint);
    }
}

/// Build the tree of [`lattice`] after `history` under `params`, with the no-repeat
/// constraint when `constrained`.
fn build(params: SamplingParams, constrained: bool, history: &[i32]) -> DraftTree {
    let (candidates, table) = lattice();
    let lattice = DraftLattice {
        candidates: &candidates,
        table: &table,
        width: WIDTH,
    };
    let config = GenerationConfig {
        max_new_tokens: 16,
        sampling: params,
        seed: Some(3),
        stop_tokens: Vec::new(),
    };
    let mut token_sampler = MlxSampler::new(params, 0x7198);
    let mut constraint = NoRepeat {
        accepted: Vec::new(),
        allow: Vec::new(),
    };
    let mut sampler = DraftSampler::for_tests(
        &config,
        &mut token_sampler,
        constrained.then_some(&mut constraint as &mut dyn RewindableConstraintMask),
    );
    let tree = build_lattice_tree(&lattice, history, 16, &mut sampler).unwrap();
    assert!(
        constraint.accepted.is_empty(),
        "the constraint is left as found"
    );
    tree
}

/// Row `row`'s path from the anchor (tokens) and its sender column in its draft row.
fn path_and_sender(tree: &DraftTree, row: usize) -> (Vec<i32>, usize) {
    let (candidates, _) = lattice();
    let path: Vec<i32> = tree
        .path(row)
        .into_iter()
        .filter(|&r| r > 0)
        .map(|r| tree.node(r).token)
        .collect();
    let sender = match path.last() {
        None => 0,
        Some(&t) => candidates[path.len() - 1]
            .iter()
            .position(|&c| c == t)
            .unwrap(),
    };
    (path, sender)
}

/// The `q` of tree row `row`'s children: its draft row's candidates for its own sender,
/// shaped under `params` over the history plus its own path, masked by the no-repeat rule.
fn expected_q(
    tree: &DraftTree,
    row: usize,
    params: &SamplingParams,
    constrained: bool,
    history: &[i32],
) -> Vec<(i32, f32)> {
    let (candidates, table) = lattice();
    let (path, sender) = path_and_sender(tree, row);
    let depth = path.len();
    let scores = &table[(depth * K + sender) * K..(depth * K + sender + 1) * K];
    let mut dense = vec![f32::NEG_INFINITY; WIDTH];
    for (&t, &s) in candidates[depth].iter().zip(scores) {
        dense[t as usize] = s;
    }
    let logits = Array::from_slice(&dense, &[1, WIDTH as i32]);
    let mut h = history.to_vec();
    h.extend(&path);
    let mask: Vec<bool> = (0..WIDTH as i32).map(|t| !path.contains(&t)).collect();
    MlxSampler::new(*params, 0)
        .distribution(&logits, &h, constrained.then_some(mask.as_slice()))
        .unwrap()
}

#[test]
fn a_sampled_tree_draws_each_child_from_its_own_branch() {
    let history = [5, 1, 5, 3];
    let plain = SamplingParams {
        temperature: 1.0,
        top_p: 1.0,
        ..SamplingParams::default()
    };
    let penalized = SamplingParams {
        repetition_penalty: 1.8,
        repetition_context: 16,
        presence_penalty: 0.7,
        ..plain
    };
    for (params, constrained) in [(plain, false), (penalized, false), (plain, true)] {
        let label = format!("{params:?} constrained {constrained}");
        let tree = build(params, constrained, &history);
        assert_eq!(tree.len(), 16, "{label}: the row bound is filled");
        assert!(!tree.is_chain(), "{label}");
        assert!(tree.max_depth() <= ROWS, "{label}");
        for row in 0..tree.rows() {
            let children = tree.children(row);
            let Some((&first, siblings)) = children.split_first() else {
                continue;
            };
            let q = expected_q(&tree, row, &params, constrained, &history);
            // The sampled child: tried first, carrying exactly its parent's own `q`.
            let drawn = tree.node(first);
            assert_eq!(drawn.dist.as_ref(), Some(&q), "{label}: row {row}'s q");
            let support: Vec<i32> = q.iter().filter(|x| x.1 > 0.0).map(|x| x.0).collect();
            assert!(support.contains(&drawn.token), "{label}: a draw from q");
            // Siblings: point masses, distinct, other than the sampled child, from q's
            // support, most probable first.
            let tokens: Vec<i32> = siblings.iter().map(|&r| tree.node(r).token).collect();
            assert!(
                siblings.iter().all(|&r| tree.node(r).dist.is_none()),
                "{label}"
            );
            assert!(!tokens.contains(&drawn.token), "{label}");
            let mut unique = tokens.clone();
            unique.sort_unstable();
            unique.dedup();
            assert_eq!(unique.len(), tokens.len(), "{label}: distinct siblings");
            assert!(tokens.iter().all(|t| support.contains(t)), "{label}");
            let weight = |t: i32| q.iter().find(|x| x.0 == t).unwrap().1;
            assert!(
                tokens.windows(2).all(|w| weight(w[0]) >= weight(w[1])),
                "{label}: siblings by probability"
            );
            if constrained {
                let (path, _) = path_and_sender(&tree, row);
                for &r in &children {
                    assert!(!path.contains(&tree.node(r).token), "{label}: no repeat");
                }
            }
        }
    }
}

#[test]
fn a_greedy_tree_puts_the_drafts_choice_first_for_each_sender() {
    let history = [5, 1, 5, 3];
    let tree = build(SamplingParams::default(), false, &history);
    let (candidates, table) = lattice();
    assert_eq!(tree.len(), 16);
    assert!(
        tree.nodes().iter().all(|n| n.dist.is_none()),
        "point masses"
    );
    for row in 0..tree.rows() {
        let children = tree.children(row);
        let Some(&first) = children.first() else {
            continue;
        };
        let (path, sender) = path_and_sender(&tree, row);
        let depth = path.len();
        let scores = &table[(depth * K + sender) * K..(depth * K + sender + 1) * K];
        let best = candidates[depth]
            .iter()
            .zip(scores)
            .fold(None, |b: Option<(i32, f32)>, (&t, &s)| match b {
                Some((_, bs)) if s <= bs => b,
                _ => Some((t, s)),
            })
            .unwrap()
            .0;
        assert_eq!(tree.node(first).token, best, "row {row}'s first child");
    }
}
