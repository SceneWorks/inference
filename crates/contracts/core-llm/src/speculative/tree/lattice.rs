//! A block draft's candidate tree (SW-7196 on MLX, SW-7198 on Candle): one draft forward's
//! candidate lattice — each scored row's top-`K` candidates (DFlash2's selector candidates,
//! DSpine's per-row top candidates of its logits) and the predecessor-conditioned score table —
//! grown best first into the [`DraftTree`] the engine verifies in one target forward.
//!
//! Neither format's block reads a drafted token, so every branch has its exact proposal
//! distribution in the one table — a row's candidates scored for its own parent's column, never
//! the greedy path's — and no branch needs a second draft forward.
//!
//! The builder is host arithmetic over candidate ids and scores. The backend supplies the
//! request's sampling policy (the knobs and penalties, the seeded draws, the rewindable
//! constraint) through [`LatticeSampler`], so both backends grow the same tree from the same
//! lattice and the same policy answers.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fmt;

use super::{DraftTree, MAX_TREE_ROWS};

/// One draft forward's candidate lattice: each scored row's candidates (ascending) and the
/// `[rows, K, K]` predecessor-conditioned score table (row-major: row `i`'s candidates scored for
/// each candidate sender of row `i - 1`), over a `width`-wide vocabulary.
#[derive(Clone, Copy, Debug)]
pub struct DraftLattice<'a> {
    /// Each scored row's candidate ids, ascending.
    pub candidates: &'a [Vec<i32>],
    /// The `[rows, K, K]` score table.
    pub table: &'a [f32],
    /// The vocabulary width the candidates are drawn from.
    pub width: usize,
}

impl DraftLattice<'_> {
    /// Row `row`'s scores for the sender at column `sender` of row `row - 1`.
    pub fn scores(&self, row: usize, sender: usize) -> &[f32] {
        let k = self.candidates[row].len();
        &self.table[(row * k + sender) * k..(row * k + sender + 1) * k]
    }
}

/// The request's draft-sampling policy as the tree builder sees it — each backend's draft sampler
/// (its knobs, seeded stream and rewindable constraint) behind host-side ids and scores.
pub trait LatticeSampler {
    /// The backend's error; a builder failure converts into it.
    type Error: From<LatticeTreeError>;

    /// Whether the run is greedy (`temperature <= 0`).
    fn greedy(&self) -> bool;

    /// Whether `token` is a stop token: no candidate grows under one.
    fn is_stop(&self, token: i32) -> bool;

    /// The constraint's checkpoint at the proposal's start (`None` without a constraint).
    fn constraint_checkpoint(&self) -> Option<usize>;

    /// Put the constraint in its state at a tree node: rewound to `checkpoint`, then advanced by
    /// the node's `path` from the anchor (stop tokens excepted).
    fn enter_branch(&mut self, checkpoint: Option<usize>, path: &[i32]);

    /// The `candidates` the constraint (if any) allows in its current state, in order.
    fn allowed_among(&mut self, candidates: &[i32]) -> Vec<i32>;

    /// The draft's choice among `candidates` scored `scores` after `history` — a greedy run's
    /// first child — or `None` when the constraint allows no candidate.
    fn choose_among(
        &mut self,
        candidates: &[i32],
        scores: &[f32],
        width: usize,
        history: &[i32],
    ) -> Result<Option<i32>, Self::Error>;

    /// The stochastic proposal distribution `q` over `candidates` scored `scores` after
    /// `history` (unnormalized `(token, weight)` pairs), or `None` when the constraint allows no
    /// candidate.
    fn among_distribution(
        &mut self,
        candidates: &[i32],
        scores: &[f32],
        width: usize,
        history: &[i32],
    ) -> Result<Option<Vec<(i32, f32)>>, Self::Error>;

    /// One draw from `dist` with the run's seeded stream. The constraint is not advanced.
    fn draw(&mut self, dist: &[(i32, f32)]) -> i32;
}

/// Why [`build_lattice_tree`] failed: the lattice and the sampler's answers disagree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LatticeTreeError(String);

impl fmt::Display for LatticeTreeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for LatticeTreeError {}

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

impl TreeNode {
    fn new(path: Vec<i32>, sender: usize, mass: f64) -> Self {
        Self {
            path,
            sender,
            mass,
            q: None,
            order: Vec::new(),
            first: None,
        }
    }
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

/// Build the candidate tree of one draft forward after `history`, best first: starting at the
/// anchor, repeatedly add the child of highest estimated path probability under the draft (each
/// row's children scored for its own sender column, so a branch's distribution is conditioned on
/// its own parent), until `max_rows` candidates (at most [`MAX_TREE_ROWS`]) or nothing left; no
/// candidate deeper than the lattice's scored rows or under a stop token. Each node's children see
/// the request's penalties over the node's own path and the constraint advanced along it; the
/// constraint is left where the build found it.
///
/// **Exactness** (recursive rejection over siblings, [`accept_among_siblings`](super::accept_among_siblings)):
/// a greedy run's children are all deterministic (`dist: None`) — the draft's choice first, then
/// the other allowed candidates by score — and any tree commits exactly the target's greedy
/// tokens. A stochastic run gives each node **one sampled child** drawn from the node's `q`
/// ([`LatticeSampler::among_distribution`]), tried first, then deterministic siblings: the most
/// probable tokens of `q` other than the sampled one. The sampled child is a true draw from its
/// own `q`, made when its slot is popped — whether the slot exists and where it sits depend only
/// on draws made before it, never on its own value, and it outranks its siblings so it is tried
/// first — and each later sibling is a point mass fixed once the sampled child is known. After
/// the sampled child's rejection the residual is `norm(max(0, p − q))` whatever was drawn, and a
/// fixed list of distinct point masses is exact against any residual, so the committed token is
/// distributed exactly as the target's. Each node's sampled child is drawn independently of every
/// other node's (fresh draws of the run's seeded stream).
pub fn build_lattice_tree<S: LatticeSampler>(
    lattice: &DraftLattice<'_>,
    history: &[i32],
    max_rows: usize,
    sampler: &mut S,
) -> Result<DraftTree, S::Error> {
    let max_rows = max_rows.min(MAX_TREE_ROWS as usize);
    let checkpoint = sampler.constraint_checkpoint();
    let mut builder = Builder {
        lattice,
        history,
        checkpoint,
        nodes: vec![TreeNode::new(Vec::new(), 0, 1.0)],
        slots: BinaryHeap::new(),
        seq: 0,
    };
    let mut tree = DraftTree::new();
    builder.expand(sampler, 0)?;
    while tree.len() < max_rows {
        let Some(slot) = builder.slots.pop() else {
            break;
        };
        let (token, dist) = match slot.child {
            Child::Drawn => {
                let q = builder.nodes[slot.parent].q.clone().ok_or_else(|| {
                    LatticeTreeError("block draft tree: a sampled child without its q".into())
                })?;
                let token = sampler.draw(&q);
                builder.nodes[slot.parent].first = Some(token);
                (token, Some(q))
            }
            Child::Fixed(token) => (token, None),
            Child::Ranked(n) => {
                let parent = &builder.nodes[slot.parent];
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
        let parent = &builder.nodes[slot.parent];
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
                LatticeTreeError(format!(
                    "block draft tree: {token} is not a candidate of draft row {depth}"
                ))
            })?;
        let mut path = parent.path.clone();
        path.push(token);
        let mass = parent.mass * probability;
        let row = tree
            .push(slot.parent, token, dist)
            .map_err(|e| LatticeTreeError(e.to_string()))?;
        debug_assert_eq!(row, builder.nodes.len(), "one node per tree row");
        builder.nodes.push(TreeNode::new(path, sender, mass));
        if depth + 1 < lattice.candidates.len() && !sampler.is_stop(token) {
            builder.expand(sampler, row)?;
        }
    }
    // Leave the constraint where the proposal found it.
    sampler.enter_branch(checkpoint, &[]);
    Ok(tree)
}

/// The build's fixed inputs, its rows and its slot queue.
struct Builder<'b, 'l> {
    lattice: &'b DraftLattice<'l>,
    history: &'b [i32],
    checkpoint: Option<usize>,
    nodes: Vec<TreeNode>,
    slots: BinaryHeap<Slot>,
    seq: u64,
}

impl Builder<'_, '_> {
    fn queue(&mut self, parent: usize, priority: f64, child: Child) {
        self.slots.push(Slot {
            priority,
            seq: self.seq,
            parent,
            child,
        });
        self.seq += 1;
    }

    /// Score tree row `row`'s children ([`build_lattice_tree`]) — its draft row's candidates for
    /// its own sender, under the penalties and constraint of its own path — and queue their slots.
    fn expand<S: LatticeSampler>(&mut self, sampler: &mut S, row: usize) -> Result<(), S::Error> {
        let lattice = self.lattice;
        let node = &self.nodes[row];
        let depth = node.path.len();
        let candidates = &lattice.candidates[depth];
        let scores = lattice.scores(depth, node.sender);
        let mass = node.mass;
        sampler.enter_branch(self.checkpoint, &node.path);
        let mut history = self.history.to_vec();
        history.extend(&node.path);
        let width = lattice.width;
        if sampler.greedy() {
            let allowed = sampler.allowed_among(candidates);
            let Some(first) = sampler.choose_among(candidates, scores, width, &history)? else {
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
            let lead = order.first().map_or(p_first, |x| x.1.max(p_first));
            self.queue(row, mass * lead, Child::Fixed(first));
            for &(token, p) in order.iter().filter(|x| x.0 != first) {
                self.queue(row, mass * p, Child::Fixed(token));
            }
            let node = &mut self.nodes[row];
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
            let node = &mut self.nodes[row];
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

#[cfg(test)]
mod tests;
