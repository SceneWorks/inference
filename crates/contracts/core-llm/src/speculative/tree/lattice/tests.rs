//! The shared lattice tree builder over hand-written lattices and a host policy — no tensors, no
//! model: best-first growth, each branch scored for its own sender, sampled children tried first
//! with their own `q`, distinct point-mass siblings, stop tokens, the row bound, the constraint
//! advanced per branch and restored, and the errors a disagreeing sampler raises.

use super::*;

const WIDTH: usize = 12;
const ROWS: usize = 3;
const K: usize = 4;

/// Three rows of four candidates each, every row's scores different for every sender.
fn lattice() -> (Vec<Vec<i32>>, Vec<f32>) {
    let candidates: Vec<Vec<i32>> = (0..ROWS)
        .map(|r| {
            let mut row: Vec<i32> = (0..K as i32)
                .map(|c| (r as i32 + 2 * c) % WIDTH as i32)
                .collect();
            row.sort_unstable();
            row
        })
        .collect();
    let table = (0..ROWS * K * K)
        .map(|i| ((i * 7 + 3) % 11) as f32 * 0.3 - 1.0)
        .collect();
    (candidates, table)
}

/// A proposal distribution: `(token, weight)` pairs.
type Dist = Vec<(i32, f32)>;

/// A host policy: a softmax of the allowed candidates' scores (temperature 1) for a stochastic
/// run, the allowed argmax (lowest id on ties) for a greedy one, an optional no-repeat-along-the-
/// branch constraint, and a deterministic uniform stream.
struct HostSampler {
    greedy: bool,
    stops: Vec<i32>,
    /// `Some(accepted)` when constrained: the tokens accepted since the start.
    constraint: Option<Vec<i32>>,
    state: u64,
    /// Every `(history, q)` a distribution was asked for, in order.
    asked: Vec<(Vec<i32>, Dist)>,
}

impl HostSampler {
    fn new(greedy: bool, constrained: bool) -> Self {
        Self {
            greedy,
            stops: Vec::new(),
            constraint: constrained.then(Vec::new),
            state: 0x7196_7198,
            asked: Vec::new(),
        }
    }

    fn uniform(&mut self) -> f32 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        let x = self.state.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((x >> 40) as f32) / (1u64 << 24) as f32
    }
}

impl LatticeSampler for HostSampler {
    type Error = LatticeTreeError;

    fn greedy(&self) -> bool {
        self.greedy
    }

    fn is_stop(&self, token: i32) -> bool {
        self.stops.contains(&token)
    }

    fn constraint_checkpoint(&self) -> Option<usize> {
        self.constraint.as_ref().map(Vec::len)
    }

    fn enter_branch(&mut self, checkpoint: Option<usize>, path: &[i32]) {
        let stops = self.stops.clone();
        if let (Some(c), Some(checkpoint)) = (self.constraint.as_mut(), checkpoint) {
            c.truncate(checkpoint);
            c.extend(path.iter().filter(|t| !stops.contains(t)));
        }
    }

    fn allowed_among(&mut self, candidates: &[i32]) -> Vec<i32> {
        match &self.constraint {
            Some(c) => candidates
                .iter()
                .copied()
                .filter(|t| !c.contains(t))
                .collect(),
            None => candidates.to_vec(),
        }
    }

    fn choose_among(
        &mut self,
        candidates: &[i32],
        scores: &[f32],
        _: usize,
        _: &[i32],
    ) -> Result<Option<i32>, LatticeTreeError> {
        let allowed = self.allowed_among(candidates);
        Ok(candidates
            .iter()
            .zip(scores)
            .filter(|(t, _)| allowed.contains(t))
            .fold(None, |best: Option<(i32, f32)>, (&t, &s)| match best {
                Some((_, b)) if s <= b => best,
                _ => Some((t, s)),
            })
            .map(|(t, _)| t))
    }

    fn among_distribution(
        &mut self,
        candidates: &[i32],
        scores: &[f32],
        _: usize,
        history: &[i32],
    ) -> Result<Option<Vec<(i32, f32)>>, LatticeTreeError> {
        let allowed = self.allowed_among(candidates);
        if allowed.is_empty() {
            return Ok(None);
        }
        let q: Vec<(i32, f32)> = candidates
            .iter()
            .zip(scores)
            .map(|(&t, &s)| (t, if allowed.contains(&t) { s.exp() } else { 0.0 }))
            .collect();
        self.asked.push((history.to_vec(), q.clone()));
        Ok(Some(q))
    }

    fn draw(&mut self, dist: &[(i32, f32)]) -> i32 {
        let total: f32 = dist.iter().map(|x| x.1).sum();
        let mut u = self.uniform() * total;
        for &(t, w) in dist {
            if w > 0.0 && u < w {
                return t;
            }
            u -= w;
        }
        dist.iter().rev().find(|x| x.1 > 0.0).map_or(0, |x| x.0)
    }
}

fn build(sampler: &mut HostSampler, history: &[i32], max_rows: usize) -> DraftTree {
    let (candidates, table) = lattice();
    let lattice = DraftLattice {
        candidates: &candidates,
        table: &table,
        width: WIDTH,
    };
    build_lattice_tree(&lattice, history, max_rows, sampler).unwrap()
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

/// Row `row`'s children's scores: its draft row's candidates for its own sender.
fn branch_scores(tree: &DraftTree, row: usize) -> (Vec<i32>, Vec<f32>, Vec<i32>) {
    let (candidates, table) = lattice();
    let (path, sender) = path_and_sender(tree, row);
    let depth = path.len();
    let scores = table[(depth * K + sender) * K..(depth * K + sender + 1) * K].to_vec();
    (candidates[depth].clone(), scores, path)
}

#[test]
fn lattice_scores_index_the_senders_column() {
    let (candidates, table) = lattice();
    let lattice = DraftLattice {
        candidates: &candidates,
        table: &table,
        width: WIDTH,
    };
    assert_eq!(lattice.scores(0, 0), &table[0..K]);
    assert_eq!(lattice.scores(1, 2), &table[(K + 2) * K..(K + 3) * K]);
    assert_eq!(
        lattice.scores(2, 3),
        &table[(2 * K + 3) * K..(2 * K + 4) * K]
    );
}

#[test]
fn a_greedy_tree_puts_the_drafts_choice_first_for_each_sender() {
    let mut sampler = HostSampler::new(true, false);
    let tree = build(&mut sampler, &[5, 1, 5, 3], 16);
    assert_eq!(tree.len(), 16, "the row bound is filled");
    assert!(!tree.is_chain());
    assert!(tree.max_depth() <= ROWS);
    assert!(
        tree.nodes().iter().all(|n| n.dist.is_none()),
        "point masses"
    );
    for row in 0..tree.rows() {
        let children = tree.children(row);
        let Some(&first) = children.first() else {
            continue;
        };
        let (candidates, scores, _) = branch_scores(&tree, row);
        let mut ranked: Vec<(i32, f32)> = candidates.iter().copied().zip(scores).collect();
        // Most probable first, lower id first on ties: the order the children are queued in.
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        assert_eq!(
            tree.node(first).token,
            ranked[0].0,
            "row {row}'s first child"
        );
        let tokens: Vec<i32> = children.iter().map(|&r| tree.node(r).token).collect();
        let expected: Vec<i32> = ranked.iter().take(tokens.len()).map(|x| x.0).collect();
        assert_eq!(tokens, expected, "row {row}'s children by score");
    }
}

#[test]
fn a_greedy_tree_grows_best_first() {
    // Two rows of two candidates. Row 0 (anchor sender): token 1 at p = e/(1+e) ≈ 0.731, token 2
    // at ≈ 0.269. Row 1 under token 1 (sender 0): token 3 at ≈ 0.881, token 4 at ≈ 0.119; under
    // token 2 (sender 1): token 4 at ≈ 0.731, token 3 at ≈ 0.269. Path masses: [1] 0.731,
    // [1,3] 0.644, [2] 0.269, [2,4] 0.197, [1,4] 0.087, [2,3] 0.072.
    let candidates = vec![vec![1, 2], vec![3, 4]];
    let table = vec![
        1.0, 0.0, 1.0, 0.0, // row 0, either sender
        2.0, 0.0, 0.0, 1.0, // row 1: sender 0 (token 1), sender 1 (token 2)
    ];
    let lattice = DraftLattice {
        candidates: &candidates,
        table: &table,
        width: 8,
    };
    let mut sampler = HostSampler::new(true, false);
    let tree = build_lattice_tree(&lattice, &[0], 16, &mut sampler).unwrap();
    let rows: Vec<(usize, i32)> = tree.nodes().iter().map(|n| (n.parent, n.token)).collect();
    assert_eq!(rows, vec![(0, 1), (1, 3), (0, 2), (3, 4), (1, 4), (3, 3)]);
    // A tighter row bound keeps the most probable prefix of the same order.
    let mut sampler = HostSampler::new(true, false);
    let short = build_lattice_tree(&lattice, &[0], 3, &mut sampler).unwrap();
    let rows: Vec<(usize, i32)> = short.nodes().iter().map(|n| (n.parent, n.token)).collect();
    assert_eq!(rows, vec![(0, 1), (1, 3), (0, 2)]);
}

#[test]
fn a_sampled_tree_draws_each_child_from_its_own_branch() {
    let history = [5, 1, 5, 3];
    for constrained in [false, true] {
        let mut sampler = HostSampler::new(false, constrained);
        let tree = build(&mut sampler, &history, 16);
        assert_eq!(
            tree.len(),
            16,
            "constrained {constrained}: the row bound is filled"
        );
        assert!(!tree.is_chain());
        assert!(tree.max_depth() <= ROWS);
        assert_eq!(
            sampler.constraint,
            constrained.then(Vec::new),
            "the constraint is left as found"
        );
        for row in 0..tree.rows() {
            let children = tree.children(row);
            let Some((&first, siblings)) = children.split_first() else {
                continue;
            };
            let (candidates, scores, path) = branch_scores(&tree, row);
            let q: Vec<(i32, f32)> = candidates
                .iter()
                .zip(&scores)
                .map(|(&t, &s)| {
                    (
                        t,
                        if path.contains(&t) && constrained {
                            0.0
                        } else {
                            s.exp()
                        },
                    )
                })
                .collect();
            // The sampled child: tried first, carrying exactly its parent's own `q`, asked for
            // over the history plus the parent's own path.
            let drawn = tree.node(first);
            assert_eq!(drawn.dist.as_ref(), Some(&q), "row {row}'s q");
            let mut h = history.to_vec();
            h.extend(&path);
            assert!(
                sampler.asked.contains(&(h, q.clone())),
                "row {row}'s history"
            );
            let support: Vec<i32> = q.iter().filter(|x| x.1 > 0.0).map(|x| x.0).collect();
            assert!(support.contains(&drawn.token), "a draw from q");
            // Siblings: distinct point masses other than the sampled child, from q's support,
            // most probable first.
            let tokens: Vec<i32> = siblings.iter().map(|&r| tree.node(r).token).collect();
            assert!(siblings.iter().all(|&r| tree.node(r).dist.is_none()));
            assert!(!tokens.contains(&drawn.token));
            let mut unique = tokens.clone();
            unique.sort_unstable();
            unique.dedup();
            assert_eq!(unique.len(), tokens.len(), "distinct siblings");
            assert!(tokens.iter().all(|t| support.contains(t)));
            let weight = |t: i32| q.iter().find(|x| x.0 == t).unwrap().1;
            assert!(
                tokens.windows(2).all(|w| weight(w[0]) >= weight(w[1])),
                "siblings by probability"
            );
            if constrained {
                for &r in &children {
                    assert!(!path.contains(&tree.node(r).token), "no repeat on a branch");
                }
            }
        }
    }
}

#[test]
fn nothing_grows_under_a_stop_token_or_past_the_lattice() {
    let (candidates, _) = lattice();
    for greedy in [true, false] {
        let mut sampler = HostSampler::new(greedy, false);
        sampler.stops = candidates[0].clone();
        let tree = build(&mut sampler, &[5], 16);
        assert_eq!(
            tree.max_depth(),
            1,
            "greedy {greedy}: every first row stops"
        );
        assert_eq!(tree.len(), K);
    }
    let mut sampler = HostSampler::new(true, false);
    let tree = build(&mut sampler, &[5], 64);
    assert_eq!(
        tree.len(),
        MAX_TREE_ROWS as usize,
        "the row bound is capped at the shared maximum"
    );
}

#[test]
fn a_fully_masked_row_proposes_nothing() {
    // One row whose only candidate the constraint already refuses.
    let candidates = vec![vec![7]];
    let table = vec![0.0];
    let lattice = DraftLattice {
        candidates: &candidates,
        table: &table,
        width: 8,
    };
    for greedy in [true, false] {
        let mut sampler = HostSampler::new(greedy, true);
        sampler.constraint = Some(vec![7]);
        let tree = build_lattice_tree(&lattice, &[0], 16, &mut sampler).unwrap();
        assert!(tree.is_empty(), "greedy {greedy}");
        assert_eq!(sampler.constraint, Some(vec![7]), "left as found");
    }
}

/// A policy whose greedy choice is not among the row's candidates.
struct Stray(HostSampler);

impl LatticeSampler for Stray {
    type Error = LatticeTreeError;
    fn greedy(&self) -> bool {
        true
    }
    fn is_stop(&self, _: i32) -> bool {
        false
    }
    fn constraint_checkpoint(&self) -> Option<usize> {
        None
    }
    fn enter_branch(&mut self, _: Option<usize>, _: &[i32]) {}
    fn allowed_among(&mut self, candidates: &[i32]) -> Vec<i32> {
        candidates.to_vec()
    }
    fn choose_among(
        &mut self,
        _: &[i32],
        _: &[f32],
        _: usize,
        _: &[i32],
    ) -> Result<Option<i32>, LatticeTreeError> {
        Ok(Some(11))
    }
    fn among_distribution(
        &mut self,
        c: &[i32],
        s: &[f32],
        w: usize,
        h: &[i32],
    ) -> Result<Option<Vec<(i32, f32)>>, LatticeTreeError> {
        self.0.among_distribution(c, s, w, h)
    }
    fn draw(&mut self, dist: &[(i32, f32)]) -> i32 {
        self.0.draw(dist)
    }
}

#[test]
fn a_choice_outside_the_row_is_refused() {
    let (candidates, table) = lattice();
    let lattice = DraftLattice {
        candidates: &candidates,
        table: &table,
        width: WIDTH,
    };
    let err = build_lattice_tree(
        &lattice,
        &[5],
        16,
        &mut Stray(HostSampler::new(true, false)),
    )
    .unwrap_err();
    assert_eq!(
        err.to_string(),
        "block draft tree: 11 is not a candidate of draft row 0"
    );
}
