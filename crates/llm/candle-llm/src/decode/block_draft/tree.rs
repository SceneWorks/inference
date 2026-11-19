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
