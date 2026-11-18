//! Tree-shaped speculative verification (feature SW-F328, story SW-6914).
//!
//! A linear speculative step verifies one proposed continuation; a **tree** step verifies a
//! bounded tree of candidate continuations in one target forward: row `0` is the anchor (the last
//! committed token, not yet in the cache) and every further row is one candidate token whose
//! `parent` row precedes it. Each row attends only to the cached prefix and its own ancestors, and
//! sits at the RoPE position of its depth below the anchor, so the target scores every branch
//! exactly as a separate linear verify of that branch would. The step then walks the tree from the
//! anchor, accepting one child per level, and keeps only the accepted branch's state.
//!
//! Like the linear policy in the parent module this is tensor-free and RNG-free: the backend
//! supplies the per-row target decision (an argmax, or a shaped distribution) and its uniform
//! draws, so every rule here is exhaustively unit-testable.
//!
//! ## Distribution preservation with siblings
//! With several candidates under one node the stochastic rule is **recursive rejection sampling**
//! (SpecInfer; multi-draft speculative sampling): try the children in order; accept child `c`
//! with probability `min(1, p_res(c) / q_c(c))`, where `q_c` is the distribution `c` was drawn
//! from; on rejection replace the residual `p_res` with `norm(max(0, p_res − q_c))` and try the
//! next child; when every child is rejected, draw the committed token from the residual. The
//! committed token is then distributed exactly as the target `p` whenever each child was drawn
//! from its own `q_c` independently of its earlier siblings. A child without a distribution is a
//! deterministic candidate (a point mass at its token: prompt lookup, a block draft's top-K row),
//! for which the same rule accepts with probability `p_res(c)` and removes `c` from the residual —
//! exact for any fixed list of distinct candidates.

use std::fmt;

/// The most candidate rows (beyond the anchor) one tree verify pass carries on either backend:
/// the anchor plus sixteen rows, the bound the tree workspace and the hybrid recurrent per-row
/// states are priced for.
pub const MAX_TREE_ROWS: u32 = 16;

/// One candidate row of a [`DraftTree`].
#[derive(Clone, Debug, PartialEq)]
pub struct DraftNode {
    /// The row this candidate continues: `0` is the anchor, `r >= 1` the node pushed `r`-th.
    /// Always below the node's own row.
    pub parent: usize,
    /// The candidate token.
    pub token: i32,
    /// For a sampled candidate, the proposal distribution `q` it was drawn from (unnormalized
    /// `(token, weight)` pairs); `None` for a deterministic candidate (a point mass).
    pub dist: Option<Vec<(i32, f32)>>,
}

/// A bounded tree of candidate continuations after the anchor. Rows are topologically ordered:
/// row `0` is the anchor and node `i` is row `i + 1`, whose parent row is below it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DraftTree {
    nodes: Vec<DraftNode>,
}

/// Why a [`DraftTree::push`] was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TreeShapeError(String);

impl fmt::Display for TreeShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for TreeShapeError {}

impl DraftTree {
    /// An empty tree (the anchor only).
    pub fn new() -> Self {
        Self::default()
    }

    /// A one-branch tree: `drafts` in order, each the child of the one before it, with each
    /// draft's proposal distribution from `dists` (missing entries are point masses). This is
    /// how a linear proposal enters the tree path.
    pub fn chain(drafts: &[i32], dists: &[Vec<(i32, f32)>]) -> Self {
        Self {
            nodes: drafts
                .iter()
                .enumerate()
                .map(|(i, &token)| DraftNode {
                    parent: i,
                    token,
                    dist: dists.get(i).cloned(),
                })
                .collect(),
        }
    }

    /// Append a candidate under `parent` (a row already in the tree) and return its row.
    pub fn push(
        &mut self,
        parent: usize,
        token: i32,
        dist: Option<Vec<(i32, f32)>>,
    ) -> Result<usize, TreeShapeError> {
        if parent > self.nodes.len() {
            return Err(TreeShapeError(format!(
                "draft tree: parent row {parent} is not in a tree of {} rows",
                self.rows()
            )));
        }
        self.nodes.push(DraftNode {
            parent,
            token,
            dist,
        });
        Ok(self.nodes.len())
    }

    /// Candidate rows (the anchor excluded).
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the tree holds no candidate.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Verify rows, the anchor included (`len() + 1`).
    pub fn rows(&self) -> usize {
        self.nodes.len() + 1
    }

    /// The candidate at `row` (`row >= 1`).
    pub fn node(&self, row: usize) -> &DraftNode {
        &self.nodes[row - 1]
    }

    /// The candidates in row order.
    pub fn nodes(&self) -> &[DraftNode] {
        &self.nodes
    }

    /// The parent row of `row`, `None` for the anchor.
    pub fn parent(&self, row: usize) -> Option<usize> {
        (row > 0).then(|| self.nodes[row - 1].parent)
    }

    /// Depth of every row below the anchor (`0` for the anchor): the RoPE offset each row takes
    /// from the anchor's position.
    pub fn depths(&self) -> Vec<i32> {
        let mut depths = vec![0i32; self.rows()];
        for (i, node) in self.nodes.iter().enumerate() {
            depths[i + 1] = depths[node.parent] + 1;
        }
        depths
    }

    /// The deepest row's depth (`0` for an empty tree).
    pub fn max_depth(&self) -> usize {
        self.depths().into_iter().max().unwrap_or(0) as usize
    }

    /// The child rows of `row`, in row order.
    pub fn children(&self, row: usize) -> Vec<usize> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.parent == row)
            .map(|(i, _)| i + 1)
            .collect()
    }

    /// Whether every candidate continues the one before it — a linear proposal, which a backend
    /// may verify through its linear path with the same outcome.
    pub fn is_chain(&self) -> bool {
        self.nodes.iter().enumerate().all(|(i, n)| n.parent == i)
    }

    /// The rows from the anchor to `row`, inclusive.
    pub fn path(&self, row: usize) -> Vec<usize> {
        let mut path = vec![row];
        let mut at = row;
        while let Some(parent) = self.parent(at) {
            path.push(parent);
            at = parent;
        }
        path.reverse();
        path
    }

    /// Ancestor-only visibility, row-major `rows × rows`: entry `i * rows + j` says whether row
    /// `i` may attend row `j` — `j` is `i` or one of its ancestors.
    pub fn visibility(&self) -> Vec<bool> {
        let n = self.rows();
        let mut visible = vec![false; n * n];
        for i in 0..n {
            for j in self.path(i) {
                visible[i * n + j] = true;
            }
        }
        visible
    }

    /// The tree a verify pass may carry: candidates at most `max_depth` below the anchor, none
    /// under a stop token (nothing after one can be committed), and at most `max_nodes` of them —
    /// kept in row order, so every kept row's parent is kept. Rows are renumbered.
    pub fn pruned(
        &self,
        max_depth: usize,
        max_nodes: usize,
        is_stop: impl Fn(i32) -> bool,
    ) -> Self {
        let depths = self.depths();
        // `kept[row]`: the row's index in the pruned tree.
        let mut kept: Vec<Option<usize>> = vec![None; self.rows()];
        kept[0] = Some(0);
        let mut out = DraftTree::new();
        for (i, node) in self.nodes.iter().enumerate() {
            let row = i + 1;
            if out.len() >= max_nodes || depths[row] as usize > max_depth {
                continue;
            }
            let Some(parent) = kept[node.parent] else {
                continue;
            };
            if node.parent > 0 && is_stop(self.node(node.parent).token) {
                continue;
            }
            kept[row] = Some(
                out.push(parent, node.token, node.dist.clone())
                    .expect("a kept parent precedes its child"),
            );
        }
        out
    }
}

/// One decision of a tree walk at a row: move to an accepted child row, or end the walk with a
/// target-chosen token (the bonus after a full accept, or the correction after a rejection).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TreeStep {
    /// The child row the target accepted.
    Child(usize),
    /// The target's own token; the walk ends.
    Stop(i32),
}

/// A finished tree walk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TreeWalk {
    /// The accepted rows from the anchor, inclusive: `path[0] == 0`, then one row per accepted
    /// candidate. The target keeps exactly these rows' state.
    pub path: Vec<usize>,
    /// The committed tokens: the accepted candidates, then the target's token unless the walk
    /// ended on an accepted stop token.
    pub committed: Vec<i32>,
    /// Accepted candidates (`path.len() - 1`).
    pub accepted: usize,
}

/// Walk `tree` from the anchor: at each row `step` decides from the row's target output and its
/// `children` (row order). An accepted stop token ends the walk with nothing after it. `step`
/// must answer [`TreeStep::Stop`] at a row without children.
pub fn walk_tree<E>(
    tree: &DraftTree,
    is_stop: impl Fn(i32) -> bool,
    mut step: impl FnMut(usize, &[usize]) -> Result<TreeStep, E>,
) -> Result<TreeWalk, E> {
    let mut walk = TreeWalk {
        path: vec![0],
        committed: Vec::new(),
        accepted: 0,
    };
    let mut row = 0;
    loop {
        let children = tree.children(row);
        match step(row, &children)? {
            TreeStep::Child(child) => {
                debug_assert!(children.contains(&child), "a walk moves to a child");
                let token = tree.node(child).token;
                walk.path.push(child);
                walk.committed.push(token);
                walk.accepted += 1;
                if is_stop(token) {
                    return Ok(walk);
                }
                row = child;
            }
            TreeStep::Stop(token) => {
                walk.committed.push(token);
                return Ok(walk);
            }
        }
    }
}

/// The greedy decision at a row: the child whose token is the target's own choice `target`
/// (the first such child), else the walk ends on `target`. Every committed token is the target's
/// greedy choice at its position, so a greedy tree run commits exactly what token-at-a-time
/// greedy decoding chooses.
pub fn greedy_tree_step(tree: &DraftTree, children: &[usize], target: i32) -> TreeStep {
    children
        .iter()
        .copied()
        .find(|&c| tree.node(c).token == target)
        .map_or(TreeStep::Stop(target), TreeStep::Child)
}

/// The stochastic decision among the `children` of one row (see the module docs): `target` is
/// the row's shaped target distribution `p`, `uniform` the backend's seeded `[0, 1)` draws (one
/// per child tried, plus one when every child is rejected).
pub fn sampled_tree_step(
    tree: &DraftTree,
    target: &[(i32, f32)],
    children: &[usize],
    uniform: &mut dyn FnMut() -> f32,
) -> TreeStep {
    let candidates: Vec<SiblingCandidate<'_>> = children
        .iter()
        .map(|&c| {
            let node = tree.node(c);
            (node.token, node.dist.as_deref())
        })
        .collect();
    match accept_among_siblings(target, &candidates, uniform) {
        SiblingAcceptance::Accepted(i) => TreeStep::Child(children[i]),
        SiblingAcceptance::Resampled(token) => TreeStep::Stop(token),
    }
}

/// One sibling candidate for [`accept_among_siblings`]: its token and the proposal distribution
/// it was drawn from (`None`: a point mass at the token).
pub type SiblingCandidate<'a> = (i32, Option<&'a [(i32, f32)]>);

/// The outcome of [`accept_among_siblings`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SiblingAcceptance {
    /// The candidate at this index was accepted.
    Accepted(usize),
    /// Every candidate was rejected; this token was drawn from the final residual.
    Resampled(i32),
}

/// Recursive rejection sampling over sibling candidates `(token, q)` against the target `p`
/// (`q = None` is a point mass at the token). The committed token is distributed exactly as the
/// normalized `target` when each sampled candidate was drawn from its own `q` independently of
/// the candidates before it.
pub fn accept_among_siblings(
    target: &[(i32, f32)],
    candidates: &[SiblingCandidate<'_>],
    uniform: &mut dyn FnMut() -> f32,
) -> SiblingAcceptance {
    let mut residual = normalized(target);
    for (i, &(token, q)) in candidates.iter().enumerate() {
        let point = [(token, 1.0f32)];
        let mut q = normalized(q.unwrap_or(&point));
        let mut q_t = weight_of(&q, token);
        if q_t <= 0.0 {
            // A token its own distribution cannot have drawn: treat it as deterministic.
            q = normalized(&point);
            q_t = 1.0;
        }
        let p_t = weight_of(&residual, token);
        if uniform() < (p_t / q_t).min(1.0) {
            return SiblingAcceptance::Accepted(i);
        }
        let next: Vec<(i32, f32)> = residual
            .iter()
            .map(|&(t, p)| (t, (p - weight_of(&q, t)).max(0.0)))
            .filter(|&(_, w)| w > 0.0)
            .collect();
        // An empty residual has probability zero (p_res == q accepts surely); on rounding noise
        // keep the previous residual rather than drawing from nothing.
        if next.iter().map(|x| x.1).sum::<f32>() > 0.0 {
            residual = normalized(&next);
        }
    }
    let fallback = residual.first().map_or(0, |x| x.0);
    SiblingAcceptance::Resampled(super::sample_weighted(&residual, uniform(), fallback))
}

/// `dist` with non-positive weights dropped, duplicate tokens merged and the rest normalized.
fn normalized(dist: &[(i32, f32)]) -> Vec<(i32, f32)> {
    let mut out: Vec<(i32, f32)> = Vec::with_capacity(dist.len());
    for &(t, w) in dist {
        if w > 0.0 {
            match out.iter_mut().find(|x| x.0 == t) {
                Some(x) => x.1 += w,
                None => out.push((t, w)),
            }
        }
    }
    let total: f32 = out.iter().map(|x| x.1).sum();
    if total > 0.0 {
        for x in &mut out {
            x.1 /= total;
        }
    }
    out
}

fn weight_of(dist: &[(i32, f32)], token: i32) -> f32 {
    dist.iter().find(|x| x.0 == token).map_or(0.0, |x| x.1)
}

/// Prompt lookup as a **tree** (feature SW-F328): every earlier occurrence of the trailing
/// n-gram (longest n-gram first, most recent first — [`ngram_propose`](super::ngram_propose)'s
/// order) contributes the continuation that followed it, up to `max_depth` tokens, merged into one
/// tree of at most `max_rows` candidates (shared prefixes share rows). The first branch is
/// exactly `ngram_propose`'s proposal, so a context with one match proposes the linear chain.
pub fn ngram_propose_tree(
    tokens: &[i32],
    max_ngram: usize,
    max_depth: usize,
    max_rows: usize,
) -> DraftTree {
    let mut tree = DraftTree::new();
    let len = tokens.len();
    if len < 2 || max_ngram == 0 || max_depth == 0 || max_rows == 0 {
        return tree;
    }
    for n in (1..=max_ngram.min(len - 1)).rev() {
        let suffix = &tokens[len - n..];
        for start in (0..=len - n - 1).rev() {
            if &tokens[start..start + n] != suffix {
                continue;
            }
            let from = start + n;
            let to = (from + max_depth).min(len);
            let mut row = 0;
            for &token in &tokens[from..to] {
                row = match tree
                    .children(row)
                    .into_iter()
                    .find(|&c| tree.node(c).token == token)
                {
                    Some(child) => child,
                    None if tree.len() < max_rows => {
                        tree.push(row, token, None).expect("row is in the tree")
                    }
                    None => break,
                };
            }
            if tree.len() >= max_rows {
                return tree;
            }
        }
    }
    tree
}

/// What a request's [`SpeculativeVerify`](crate::SpeculativeVerify) resolves to against the
/// loaded model and the resolved speculative plan.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TreeResolution {
    /// Candidate rows per tree pass; `0` verifies linearly.
    pub max_rows: u32,
    /// Why the tree runs with fewer rows than asked, or not at all (`speculative: …`).
    pub fallback: Option<String>,
}

/// Resolve the request's verify layout (E5/E8): `linear` is always linear; `tree` runs only on a
/// plan with a proposer and a model advertising tree verification, at most at the advertised row
/// bound — otherwise it falls back with the reason named, never a failure. `auto` speculation
/// never turns a tree on by itself: the tree is the request's explicit opt-in.
pub fn resolve_tree_verify(
    verify: crate::SpeculativeVerify,
    plan: super::SpeculativePlan,
    capabilities: &crate::TextLlmCapabilities,
) -> TreeResolution {
    resolve_tree_verify_with(verify, plan, capabilities.tree_verify)
}

/// [`resolve_tree_verify`] over the advertisement alone — for a provider (a captioner) that has
/// no tree advertisement and no proposer, so the fallback naming why is the shared one.
pub fn resolve_tree_verify_with(
    verify: crate::SpeculativeVerify,
    plan: super::SpeculativePlan,
    tree_verify: Option<crate::TreeVerifyCapabilities>,
) -> TreeResolution {
    let crate::SpeculativeVerify::Tree { max_rows } = verify else {
        return TreeResolution::default();
    };
    let linear = |why: String| TreeResolution {
        max_rows: 0,
        fallback: Some(why),
    };
    if plan == super::SpeculativePlan::Off {
        return linear(
            "speculative: tree verification needs a proposer and none runs for this request".into(),
        );
    }
    let Some(cap) = tree_verify else {
        return linear(format!("speculative: {TREE_VERIFY_NOT_ADVERTISED}"));
    };
    let bound = cap.max_rows.min(MAX_TREE_ROWS);
    let rows = max_rows.clamp(1, bound.max(1));
    TreeResolution {
        max_rows: rows,
        fallback: (rows != max_rows).then(|| {
            format!(
                "speculative: tree verification at {max_rows} rows clamped to {rows} \
                 (advertised 1..={bound})"
            )
        }),
    }
}

/// Why a model without a tree advertisement verifies linearly.
pub const TREE_VERIFY_NOT_ADVERTISED: &str =
    "tree verification is not available for this model; verified linearly";

#[cfg(test)]
mod tests;
