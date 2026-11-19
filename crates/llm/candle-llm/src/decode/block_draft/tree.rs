//! The block draft's candidate tree on Candle (SW-7198), the port of mlx-llm's SW-7196 builder:
//! one draft forward's candidate lattice — each scored row's top-`K` candidates (DFlash2's
//! selector candidates, DSpine's per-row top candidates of its logits) and the
//! predecessor-conditioned score table — grown best first into the tree the engine verifies in one
//! target forward.
//!
//! Neither format's block reads a drafted token, so every branch has its exact proposal
//! distribution in the one table — a row's candidates scored for its own parent's column, never
//! the greedy path's — and no branch needs a second draft forward.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use core_llm::DraftTree;

use crate::decode::engine::{DraftSampler, ProposeContext};
use crate::error::{Error, Result};

/// One draft forward's candidate lattice: each scored row's candidates (ascending) and the
/// `[rows, K, K]` predecessor-conditioned score table, over a `width`-wide vocabulary.
pub(super) struct Lattice<'a> {
    pub(super) candidates: &'a [Vec<i32>],
    pub(super) table: &'a [f32],
    pub(super) width: usize,
}

impl Lattice<'_> {
    /// Row `row`'s scores for the sender at column `sender` of row `row - 1`.
    fn scores(&self, row: usize, sender: usize) -> &[f32] {
        let k = self.candidates[row].len();
        &self.table[(row * k + sender) * k..(row * k + sender + 1) * k]
    }
}

/// A tree row under construction: the tokens from the anchor to it, the draft's estimate of
/// reaching it, and — once expanded — its children's proposal.
struct TreeNode {
    /// Tokens from the anchor (exclusive) to this row (inclusive).
    path: Vec<i32>,
    /// This row's column among its draft row's candidates: the sender its children are scored
    /// for (`0` for the anchor, whose sender column is the same for every candidate).
    sender: usize,
    /// The draft's probability of the path: the product of each step's proposal probability.
    mass: f64,
    /// A stochastic run's proposal `q` for this row's children; its sampled child is drawn from it.
    q: Option<Vec<(i32, f32)>>,
    /// The children's tokens, most probable first, with their normalized proposal probability.
    order: Vec<(i32, f64)>,
    /// The first child: the sampled child once drawn (stochastic), the draft's choice (greedy).
    first: Option<i32>,
}

/// What a [`Slot`] adds once popped.
#[derive(Clone, Copy)]
enum Child {
    /// A stochastic run's sampled child: drawn from the parent's `q` when popped.
    Drawn,
    /// A deterministic child with this token.
    Fixed(i32),
    /// A stochastic run's `n`-th most probable token of the parent's `q` other than its sampled
    /// child — deterministic once that child is drawn.
    Ranked(usize),
}

/// A child the best-first build may add under tree row `parent`, ranked by `priority` (the
/// parent's path mass times the child's proposal probability), then by insertion (`seq`).
struct Slot {
    priority: f64,
    seq: u64,
    parent: usize,
    child: Child,
}

impl PartialEq for Slot {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Slot {}

impl PartialOrd for Slot {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Slot {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .total_cmp(&other.priority)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

/// Build the candidate tree of one draft forward, best first: starting at the anchor, repeatedly
/// add the child of highest estimated path probability under the draft (each row's children
/// scored for its own sender column, so a branch's distribution is conditioned on its own
/// parent), until `ctx.max_rows` candidates or nothing left; no candidate deeper than the draft's
/// scored rows or under a stop token. Each node's children see the request's penalties over the
/// node's own path and the constraint advanced along it.
///
/// **Exactness** (`core_llm`'s recursive rejection over siblings): a greedy run's children are
/// all deterministic (`dist: None`) — the draft's choice first, then the other allowed candidates
/// by score — and any tree commits exactly the target's greedy tokens. A stochastic run gives
/// each node **one sampled child** drawn from the node's `q` (the shaped distribution
/// [`DraftSampler::sample_among`] draws a linear draft from), tried first, then deterministic
/// siblings: the most probable tokens of `q` other than the sampled one. The sampled child is a
/// true draw from its own `q`, made when its slot is popped — whether the slot exists and where
/// it sits depend only on draws made before it, never on its own value, and it outranks its
/// siblings so it is tried first — and each later sibling is a point mass fixed once the sampled
/// child is known. After the sampled child's rejection the residual is `norm(max(0, p − q))`
/// whatever was drawn, and a fixed list of distinct point masses is exact against any residual,
/// so the committed token is distributed exactly as the target's. Each node's sampled child is
/// drawn independently of every other node's (fresh uniforms of the run's seeded stream).
pub(super) fn build_tree(
    lattice: &Lattice<'_>,
    ctx: &ProposeContext<'_>,
    sampler: &mut DraftSampler<'_, '_>,
) -> Result<DraftTree> {
    let max_rows = ctx.max_rows.min(core_llm::MAX_TREE_ROWS as usize);
    let checkpoint = sampler.constraint_checkpoint();
    let mut tree = DraftTree::new();
    let mut nodes = vec![TreeNode {
        path: Vec::new(),
        sender: 0,
        mass: 1.0,
        q: None,
        order: Vec::new(),
        first: None,
    }];
    let mut slots = BinaryHeap::new();
    let mut seq = 0u64;
    let mut builder = Builder {
        lattice,
        ctx,
        checkpoint,
        slots: &mut slots,
        seq: &mut seq,
    };
    builder.expand(sampler, &mut nodes, 0)?;
    while tree.len() < max_rows {
        let Some(slot) = builder.slots.pop() else {
            break;
        };
        let (token, dist) = match slot.child {
            Child::Drawn => {
                let q = nodes[slot.parent].q.clone().ok_or_else(|| {
                    Error::Msg("block draft tree: a sampled child without its q".into())
                })?;
                let token = sampler.draw(&q);
                nodes[slot.parent].first = Some(token);
                (token, Some(q))
            }
            Child::Fixed(token) => (token, None),
            Child::Ranked(n) => {
                let parent = &nodes[slot.parent];
                let first = parent.first;
                match parent
                    .order
                    .iter()
                    .map(|&(t, _)| t)
                    .filter(|&t| Some(t) != first)
                    .nth(n)
                {
                    Some(token) => (token, None),
                    None => continue,
                }
            }
        };
        let parent = &nodes[slot.parent];
        let depth = parent.path.len();
        let probability = parent
            .order
            .iter()
            .find(|&&(t, _)| t == token)
            .map_or(0.0, |&(_, p)| p);
        let sender = lattice.candidates[depth]
            .iter()
            .position(|&c| c == token)
            .ok_or_else(|| {
                Error::Msg(format!(
                    "block draft tree: {token} is not a candidate of draft row {depth}"
                ))
            })?;
        let mut path = parent.path.clone();
        path.push(token);
        let mass = parent.mass * probability;
        let row = tree
            .push(slot.parent, token, dist)
            .map_err(|e| Error::Msg(e.to_string()))?;
        debug_assert_eq!(row, nodes.len(), "one node per tree row");
        nodes.push(TreeNode {
            path,
            sender,
            mass,
            q: None,
            order: Vec::new(),
            first: None,
        });
        if depth + 1 < lattice.candidates.len() && !sampler.is_stop(token) {
            builder.expand(sampler, &mut nodes, row)?;
        }
    }
    // Leave the constraint where the proposal found it.
    sampler.enter_branch(checkpoint, &[]);
    Ok(tree)
}

/// The build's fixed inputs and its slot queue.
struct Builder<'b, 'l> {
    lattice: &'b Lattice<'l>,
    ctx: &'b ProposeContext<'b>,
    checkpoint: Option<usize>,
    slots: &'b mut BinaryHeap<Slot>,
    seq: &'b mut u64,
}

impl Builder<'_, '_> {
    fn queue(&mut self, parent: usize, priority: f64, child: Child) {
        self.slots.push(Slot {
            priority,
            seq: *self.seq,
            parent,
            child,
        });
        *self.seq += 1;
    }

    /// Score tree row `row`'s children ([`build_tree`]) — its draft row's candidates for its own
    /// sender, under the penalties and constraint of its own path — and queue their slots.
    fn expand(
        &mut self,
        sampler: &mut DraftSampler<'_, '_>,
        nodes: &mut [TreeNode],
        row: usize,
    ) -> Result<()> {
        let node = &nodes[row];
        let depth = node.path.len();
        let candidates = &self.lattice.candidates[depth];
        let scores = self.lattice.scores(depth, node.sender);
        let mass = node.mass;
        sampler.enter_branch(self.checkpoint, &node.path);
        let mut history = self.ctx.history.to_vec();
        history.extend(&node.path);
        let width = self.lattice.width;
        if sampler.greedy() {
            let allowed = sampler.allowed_among(candidates);
            let Some((first, _)) = sampler.sample_among(candidates, scores, width, &history)?
            else {
                return Ok(());
            };
            // The draft's own probabilities (a softmax of the allowed candidates' scores) rank the
            // deterministic children; the draft's choice comes first.
            let scored: Vec<(i32, f32)> = candidates
                .iter()
                .zip(scores)
                .filter(|(t, _)| allowed.contains(t))
                .map(|(&t, &s)| (t, s))
                .collect();
            let top = scored.iter().map(|x| x.1).fold(f32::NEG_INFINITY, f32::max);
            let mut order: Vec<(i32, f64)> = scored
                .iter()
                .map(|&(t, s)| (t, f64::from((s - top).exp())))
                .collect();
            normalize(&mut order);
            let p_first = order.iter().find(|x| x.0 == first).map_or(1.0, |x| x.1);
            self.queue(row, mass * order[0].1.max(p_first), Child::Fixed(first));
            for &(token, p) in order.iter().filter(|x| x.0 != first) {
                self.queue(row, mass * p, Child::Fixed(token));
            }
            let node = &mut nodes[row];
            node.order = order;
            node.first = Some(first);
        } else {
            let Some(q) = sampler.among_distribution(candidates, scores, width, &history)? else {
                return Ok(());
            };
            let mut order: Vec<(i32, f64)> = q
                .iter()
                .filter(|x| x.1 > 0.0)
                .map(|&(t, w)| (t, f64::from(w)))
                .collect();
            if order.is_empty() {
                return Ok(());
            }
            normalize(&mut order);
            // The sampled child's slot outranks (or ties, and precedes) every sibling's, so it is
            // drawn and tried first; sibling `n` is the `n`-th most probable token after it.
            self.queue(row, mass * order[0].1, Child::Drawn);
            for (n, &(_, p)) in order.iter().enumerate().skip(1) {
                self.queue(row, mass * p, Child::Ranked(n - 1));
            }
            let node = &mut nodes[row];
            node.q = Some(q);
            node.order = order;
        }
        Ok(())
    }
}

/// Normalize `dist`'s weights and order it most probable first (lower id first on ties).
fn normalize(dist: &mut [(i32, f64)]) {
    let total: f64 = dist.iter().map(|x| x.1).sum();
    if total > 0.0 {
        for x in dist.iter_mut() {
            x.1 /= total;
        }
    }
    dist.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
}

/// The tree builder over a synthetic lattice — host arithmetic on hand-written scores, no model:
/// every sampled row carries exactly the `q` of its parent's own branch (its sender column, the
/// penalties over its own path, the constraint advanced along it), the sampled child is tried
/// before its siblings, siblings are distinct point masses, and a greedy tree's first child is
/// the draft's choice for its own sender.
#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use super::*;
    use crate::decode::engine::RewindableConstraintMask;
    use crate::decode::stream::{ConstraintMask, GenerationConfig};
    use crate::primitives::sampler::{shaped_candidates, SamplingParams, SplitMix64};

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
        let lattice = Lattice {
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
        let mut rng = SplitMix64::new(0x7198);
        let mut constraint = NoRepeat {
            accepted: Vec::new(),
            allow: Vec::new(),
        };
        let mut sampler = DraftSampler::for_tests(
            &config,
            &mut rng,
            constrained.then_some(&mut constraint as &mut dyn RewindableConstraintMask),
        );
        let cur_ids = Tensor::new(&[[0u32]], &Device::Cpu).unwrap();
        let ctx = ProposeContext {
            cur: *history.last().unwrap(),
            cur_ids: &cur_ids,
            history,
            previous_hidden: None,
            position: history.len() as i32,
            max_drafts: ROWS,
            budget_drafts: ROWS,
            max_rows: 16,
        };
        let tree = build_tree(&lattice, &ctx, &mut sampler).unwrap();
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
        let logits = Tensor::from_vec(dense, (1, WIDTH), &Device::Cpu).unwrap();
        let mut h = history.to_vec();
        h.extend(&path);
        let mask: Vec<bool> = (0..WIDTH as i32).map(|t| !path.contains(&t)).collect();
        shaped_candidates(&logits, &h, params, constrained.then_some(mask.as_slice())).unwrap()
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
}
