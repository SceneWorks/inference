use super::*;
use crate::{SpeculativeProposer, SpeculativeVerify, TextLlmCapabilities, TreeVerifyCapabilities};

/// A small deterministic uniform stream (xorshift64*), independent of any backend RNG.
struct Draws(u64);

impl Draws {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let x = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((x >> 40) as f32) / (1u64 << 24) as f32
    }
}

/// The tree used by most shape tests:
/// ```text
/// 0 ─┬─ 1(10) ─┬─ 3(30) ── 5(50)
///    │         └─ 4(31)
///    └─ 2(11) ──── 6(32)
/// ```
fn sample_tree() -> DraftTree {
    let mut t = DraftTree::new();
    assert_eq!(t.push(0, 10, None).unwrap(), 1);
    assert_eq!(t.push(0, 11, None).unwrap(), 2);
    assert_eq!(t.push(1, 30, None).unwrap(), 3);
    assert_eq!(t.push(1, 31, None).unwrap(), 4);
    assert_eq!(t.push(3, 50, None).unwrap(), 5);
    assert_eq!(t.push(2, 32, None).unwrap(), 6);
    t
}

#[test]
fn a_tree_reports_its_shape() {
    let mut t = sample_tree();
    assert_eq!(t.rows(), 7);
    assert_eq!(t.depths(), vec![0, 1, 1, 2, 2, 3, 2]);
    assert_eq!(t.max_depth(), 3);
    assert_eq!(t.children(0), vec![1, 2]);
    assert_eq!(t.children(1), vec![3, 4]);
    assert!(t.children(5).is_empty());
    assert_eq!(t.path(5), vec![0, 1, 3, 5]);
    assert_eq!(t.path(6), vec![0, 2, 6]);
    assert!(!t.is_chain());
    assert!(
        t.push(9, 1, None).is_err(),
        "a parent must already be a row"
    );

    let v = t.visibility();
    let n = t.rows();
    let sees = |i: usize, j: usize| v[i * n + j];
    for i in 0..n {
        assert!(
            sees(i, 0) && sees(i, i),
            "every row sees the anchor and itself"
        );
    }
    assert!(sees(5, 3) && sees(5, 1) && !sees(5, 2) && !sees(5, 4) && !sees(5, 6));
    assert!(!sees(3, 4) && !sees(4, 3), "siblings never see each other");
    assert!(!sees(1, 3), "no row sees a descendant");
}

/// The parent-row contract every producer and consumer shares: a parent is a row index where
/// `0` is the anchor and `r >= 1` the `r`-th candidate pushed; the next row is `len() + 1`, so
/// `0..=len()` are accepted and anything else is refused. The MLX tree proofs' shapes (chain,
/// star, two interleaved branches, binary, random) are built here from the same parent lists, so
/// a malformed fixture fails in ordinary CI instead of inside the model-proof workflow.
#[test]
fn a_parent_is_a_row_index_from_the_anchor() {
    let mut t = DraftTree::new();
    let refused = t.push(1, 7, None).unwrap_err().to_string();
    assert!(refused.contains("parent row 1"), "{refused}");
    assert_eq!(t.push(0, 7, None).unwrap(), 1, "the anchor is row 0");
    assert_eq!(
        t.push(1, 8, None).unwrap(),
        2,
        "the last row is a valid parent"
    );
    assert!(t.push(3, 9, None).is_err(), "a row cannot parent itself");
    assert!(t.push(4, 9, None).is_err(), "nor a row after it");
    assert_eq!(t.len(), 2, "a refused push adds nothing");
    assert_eq!(t.parent(2), Some(1));
    assert_eq!(t.path(2), vec![0, 1, 2]);

    let build = |parents: &[usize]| {
        let mut t = DraftTree::new();
        for (i, &p) in parents.iter().enumerate() {
            assert_eq!(t.push(p, i as i32, None).unwrap(), i + 1, "{parents:?}");
        }
        t
    };
    let chain: Vec<usize> = (0..16).collect();
    let star = vec![0; 16];
    let two: Vec<usize> = [0, 0].into_iter().chain(1..=14).collect();
    let binary: Vec<usize> = (1..=15).map(|r| (r - 1) / 2).collect();
    let max = MAX_TREE_ROWS as usize;

    let t = build(&chain);
    assert!(t.is_chain());
    assert_eq!(t.depths(), (0..=16).collect::<Vec<i32>>());

    let t = build(&star);
    assert_eq!(t.max_depth(), 1);
    assert_eq!(t.children(0), (1..=16).collect::<Vec<_>>());

    let t = build(&two);
    assert_eq!(t.len(), max);
    assert_eq!(t.children(0), vec![1, 2]);
    assert_eq!(t.path(15), vec![0, 1, 3, 5, 7, 9, 11, 13, 15]);
    assert_eq!(t.path(16), vec![0, 2, 4, 6, 8, 10, 12, 14, 16]);
    assert!((1..=16).all(|r| t.children(r).len() <= 1));

    let t = build(&binary);
    // Heap order with the anchor as the root: rows 0..=6 have two children, row 7 one.
    assert_eq!(t.max_depth(), 4);
    assert!((0..=6).all(|r| t.children(r).len() == 2));
    assert_eq!(t.children(7), vec![15]);
    assert!((8..=15).all(|r| t.children(r).is_empty()));

    // Random parents in `0..r` for row `r` are always valid.
    let mut draws = Draws(0x5eed_6914);
    for _ in 0..64 {
        let parents: Vec<usize> = (1..=max)
            .map(|r| ((draws.next() * r as f32) as usize).min(r - 1))
            .collect();
        let t = build(&parents);
        let depths = t.depths();
        for r in 1..t.rows() {
            let p = t.parent(r).unwrap();
            assert!(p < r && depths[r] == depths[p] + 1);
            assert_eq!(t.path(r).len(), depths[r] as usize + 1);
        }
    }
}

#[test]
fn a_chain_is_a_linear_proposal() {
    let t = DraftTree::chain(&[4, 5, 6], &[vec![(4, 1.0)]]);
    assert!(t.is_chain());
    assert_eq!(t.depths(), vec![0, 1, 2, 3]);
    assert_eq!(t.node(1).dist, Some(vec![(4, 1.0)]));
    assert_eq!(
        t.node(2).dist,
        None,
        "a missing distribution is a point mass"
    );
    assert!(DraftTree::new().is_chain());
}

#[test]
fn pruning_bounds_depth_rows_and_drops_rows_after_a_stop_token() {
    let t = sample_tree();
    let shallow = t.pruned(1, 16, |_| false);
    assert_eq!(shallow.len(), 2);
    assert_eq!(shallow.max_depth(), 1);

    let few = t.pruned(16, 3, |_| false);
    assert_eq!(few.len(), 3);
    assert_eq!(few.node(3).parent, 1, "kept in row order, parents kept");

    let stop_at_10 = t.pruned(16, 16, |tok| tok == 10);
    assert_eq!(
        stop_at_10
            .nodes()
            .iter()
            .map(|n| n.token)
            .collect::<Vec<_>>(),
        vec![10, 11, 32],
        "nothing below an accepted stop token can commit"
    );
    // Renumbered rows keep their parents.
    assert_eq!(stop_at_10.node(3).parent, 2);
}

#[test]
fn a_greedy_walk_follows_the_target_through_the_tree() {
    let t = sample_tree();
    // The target's argmax per row: anchor → 10, row 1 → 31, row 4 → 99.
    let argmax = |row: usize| match row {
        0 => 10,
        1 => 31,
        4 => 99,
        _ => unreachable!("the walk visits only accepted rows"),
    };
    let walk = walk_tree(
        &t,
        |_| false,
        |row, children| Ok::<_, ()>(greedy_tree_step(&t, children, argmax(row))),
    )
    .unwrap();
    assert_eq!(walk.path, vec![0, 1, 4]);
    assert_eq!(walk.committed, vec![10, 31, 99]);
    assert_eq!(walk.accepted, 2);

    // A rejection at the anchor commits the target's own token only.
    let walk = walk_tree(
        &t,
        |_| false,
        |_, children| Ok::<_, ()>(greedy_tree_step(&t, children, 7)),
    )
    .unwrap();
    assert_eq!(
        (walk.path, walk.committed, walk.accepted),
        (vec![0], vec![7], 0)
    );

    // A leaf accepted all the way down ends on the bonus from the leaf's row.
    let leaf = |row: usize| [10, 30, 0, 50, 0, 77][row];
    let walk = walk_tree(
        &t,
        |_| false,
        |row, children| Ok::<_, ()>(greedy_tree_step(&t, children, leaf(row))),
    )
    .unwrap();
    assert_eq!(walk.path, vec![0, 1, 3, 5]);
    assert_eq!(walk.committed, vec![10, 30, 50, 77]);
}

#[test]
fn an_accepted_stop_token_ends_the_walk_without_a_bonus() {
    let t = sample_tree();
    let walk = walk_tree(
        &t,
        |tok| tok == 30,
        |row, children| Ok::<_, ()>(greedy_tree_step(&t, children, [10, 30][row.min(1)])),
    )
    .unwrap();
    assert_eq!(walk.path, vec![0, 1, 3]);
    assert_eq!(walk.committed, vec![10, 30]);
}

#[test]
fn a_walk_propagates_the_backends_error() {
    let t = sample_tree();
    let out = walk_tree(&t, |_| false, |_, _| Err::<TreeStep, _>("device lost"));
    assert_eq!(out.unwrap_err(), "device lost");
}

// ---- Distribution preservation ----

fn norm(d: &[(i32, f32)]) -> Vec<f64> {
    let total: f64 = d.iter().map(|x| f64::from(x.1.max(0.0))).sum();
    let mut out = vec![0.0; VOCAB];
    for &(t, w) in d {
        out[t as usize] += f64::from(w.max(0.0)) / total;
    }
    out
}

const VOCAB: usize = 5;

/// The committed-token distribution of recursive rejection sampling over fixed candidates,
/// computed in closed form (independently of the draw-based implementation): accept candidate
/// `c` with probability `min(1, r(c)/q(c))`, else recurse on `norm(max(0, r − q))`; the residual
/// after the last candidate is the draw.
fn exact(residual: Vec<f64>, candidates: &[(i32, Vec<f64>)]) -> Vec<f64> {
    let Some(((c, q), rest)) = candidates.split_first() else {
        return residual;
    };
    let c = *c as usize;
    let a = if q[c] > 0.0 {
        (residual[c] / q[c]).min(1.0)
    } else {
        residual[c]
    };
    let mut next: Vec<f64> = residual
        .iter()
        .zip(q)
        .map(|(r, q)| (r - q).max(0.0))
        .collect();
    let total: f64 = next.iter().sum();
    if total > 0.0 {
        next.iter_mut().for_each(|x| *x /= total);
    } else {
        next = residual.clone();
    }
    let tail = exact(next, rest);
    (0..VOCAB)
        .map(|t| a * f64::from(u8::from(t == c)) + (1.0 - a) * tail[t])
        .collect()
}

/// Every pair of draft tokens drawn independently from `q1`, `q2`, weighted by its probability.
fn exact_sampled(p: &[f64], q1: &[f64], q2: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; VOCAB];
    for c1 in 0..VOCAB {
        for c2 in 0..VOCAB {
            let w = q1[c1] * q2[c2];
            if w == 0.0 {
                continue;
            }
            let d = exact(
                p.to_vec(),
                &[(c1 as i32, q1.to_vec()), (c2 as i32, q2.to_vec())],
            );
            for t in 0..VOCAB {
                out[t] += w * d[t];
            }
        }
    }
    out
}

fn random_dist(draws: &mut Draws, zeros: bool) -> Vec<(i32, f32)> {
    (0..VOCAB as i32)
        .map(|t| {
            let w = draws.next();
            (t, if zeros && w < 0.3 { 0.0 } else { w + 0.01 })
        })
        .collect()
}

fn assert_close(a: &[f64], b: &[f64], tol: f64, what: &str) {
    for t in 0..VOCAB {
        assert!(
            (a[t] - b[t]).abs() <= tol,
            "{what}: token {t}: {} vs {} (tolerance {tol})\n{a:?}\n{b:?}",
            a[t],
            b[t]
        );
    }
}

/// The rule itself is exact: over many random targets and proposal distributions (sparse ones
/// included), two independently sampled siblings, and fixed deterministic siblings, the closed
/// form commits exactly the target distribution.
#[test]
fn recursive_rejection_is_exactly_the_target_distribution() {
    let mut draws = Draws(0x6914_0001);
    for case in 0..300 {
        let sparse = case % 3 == 0;
        let p = norm(&random_dist(&mut draws, sparse));
        let q1 = norm(&random_dist(&mut draws, sparse && case % 2 == 0));
        let q2 = norm(&random_dist(&mut draws, false));
        assert_close(
            &exact_sampled(&p, &q1, &q2),
            &p,
            1e-9,
            "two sampled siblings",
        );
        assert_close(
            &exact_sampled(&p, &q1, &q1),
            &p,
            1e-9,
            "two draws from one q",
        );

        // Deterministic distinct candidates (point masses), in any order, up to the whole vocab.
        for k in 1..=VOCAB {
            let start = case % VOCAB;
            let cands: Vec<(i32, Vec<f64>)> = (0..k)
                .map(|i| {
                    let t = (start + i) % VOCAB;
                    let mut q = vec![0.0; VOCAB];
                    q[t] = 1.0;
                    (t as i32, q)
                })
                .collect();
            assert_close(&exact(p.clone(), &cands), &p, 1e-9, "point-mass siblings");
        }
    }
}

/// The draw-based implementation follows that rule: its empirical committed distribution over a
/// long seeded stream matches the target within five standard errors, for sampled siblings,
/// deterministic siblings and a single child (the linear case).
#[test]
fn sibling_acceptance_draws_commit_the_target_distribution() {
    const N: usize = 200_000;
    let mut draws = Draws(0x6914_0002);
    let p = vec![(0, 0.05f32), (1, 0.4), (2, 0.25), (3, 0.0), (4, 0.3)];
    let q1 = vec![(0, 0.5f32), (1, 0.1), (2, 0.1), (3, 0.2), (4, 0.1)];
    let q2 = vec![(0, 0.1f32), (1, 0.1), (2, 0.6), (3, 0.1), (4, 0.1)];
    let target = norm(&p);
    let tol = |pt: f64| 5.0 * (pt * (1.0 - pt) / N as f64).sqrt() + 1e-4;
    let sample_from = |d: &[(i32, f32)], u: f32| crate::speculative::sample_weighted(d, u, 0);

    // Two siblings, each drawn from its own q, then accepted or rejected against p.
    let mut counts = [0usize; VOCAB];
    for _ in 0..N {
        let c1 = sample_from(&q1, draws.next());
        let c2 = sample_from(&q2, draws.next());
        let cands = [(c1, Some(&q1[..])), (c2, Some(&q2[..]))];
        let token = match accept_among_siblings(&p, &cands, &mut || draws.next()) {
            SiblingAcceptance::Accepted(i) => cands[i].0,
            SiblingAcceptance::Resampled(t) => t,
        };
        counts[token as usize] += 1;
    }
    for t in 0..VOCAB {
        let f = counts[t] as f64 / N as f64;
        assert!(
            (f - target[t]).abs() <= tol(target[t]),
            "sampled: token {t}: {f} vs {}",
            target[t]
        );
    }
    assert_eq!(
        counts[3], 0,
        "a token the target excludes is never committed"
    );

    // Deterministic top-2 siblings (a block draft's candidates).
    let mut counts = [0usize; VOCAB];
    for _ in 0..N {
        let cands = [(1, None), (2, None)];
        let token = match accept_among_siblings(&p, &cands, &mut || draws.next()) {
            SiblingAcceptance::Accepted(i) => cands[i].0,
            SiblingAcceptance::Resampled(t) => t,
        };
        counts[token as usize] += 1;
    }
    for t in 0..VOCAB {
        let f = counts[t] as f64 / N as f64;
        assert!(
            (f - target[t]).abs() <= tol(target[t]),
            "point: token {t}: {f} vs {}",
            target[t]
        );
    }
}

/// With one child the sibling rule is the linear rule: the same accept probability
/// `min(1, p/q)` as [`accept_token`](crate::accept_token), and on rejection a draw from the same
/// residual.
#[test]
fn a_single_child_is_the_linear_acceptance_rule() {
    let p = vec![(0, 0.2f32), (1, 0.5), (2, 0.3)];
    let q = vec![(0, 0.6f32), (1, 0.2), (2, 0.2)];
    for i in 0..=100 {
        let u_accept = i as f32 / 100.0 * 0.999;
        for j in 0..=20 {
            let u_resample = j as f32 / 20.0 * 0.999;
            let linear = crate::accept_token(&p, &q, 0, u_accept, u_resample);
            let mut us = [u_accept, u_resample].into_iter();
            let tree = accept_among_siblings(&p, &[(0, Some(&q[..]))], &mut || us.next().unwrap());
            match (linear, tree) {
                (crate::Acceptance::Accepted(t), SiblingAcceptance::Accepted(0)) => {
                    assert_eq!(t, 0)
                }
                (crate::Acceptance::Rejected(a), SiblingAcceptance::Resampled(b)) => {
                    assert_eq!(a, b, "u_accept {u_accept} u_resample {u_resample}")
                }
                other => panic!("u_accept {u_accept}: {other:?}"),
            }
        }
    }
}

#[test]
fn a_sampled_walk_accepts_down_the_tree_and_resamples_at_the_end() {
    let t = sample_tree();
    // Every row's target puts all its mass on one token: the walk is forced.
    let target_at = |row: usize| match row {
        0 => vec![(11, 1.0f32)],
        2 => vec![(32, 1.0)],
        6 => vec![(8, 1.0)],
        _ => unreachable!(),
    };
    let mut draws = Draws(3);
    let walk = walk_tree(
        &t,
        |_| false,
        |row, children| {
            Ok::<_, ()>(sampled_tree_step(
                &t,
                &target_at(row),
                children,
                &mut || draws.next(),
            ))
        },
    )
    .unwrap();
    assert_eq!(walk.path, vec![0, 2, 6]);
    assert_eq!(walk.committed, vec![11, 32, 8]);
}

// ---- Request resolution ----

fn caps(max_rows: Option<u32>) -> TextLlmCapabilities {
    TextLlmCapabilities {
        tree_verify: max_rows.map(|max_rows| TreeVerifyCapabilities { max_rows }),
        ..Default::default()
    }
}

#[test]
fn tree_verification_is_an_explicit_opt_in_that_falls_back_by_name() {
    let run = crate::SpeculativePlan::Run {
        proposer: SpeculativeProposer::PromptLookup,
        depth: 4,
    };
    let tree = |max_rows| SpeculativeVerify::Tree { max_rows };

    assert_eq!(
        resolve_tree_verify(SpeculativeVerify::Linear, run, &caps(Some(16))),
        TreeResolution::default(),
        "linear stays linear"
    );
    assert_eq!(
        resolve_tree_verify(tree(8), run, &caps(Some(16))).max_rows,
        8
    );
    assert_eq!(
        resolve_tree_verify(tree(8), run, &caps(Some(16))).fallback,
        None
    );

    let clamped = resolve_tree_verify(tree(40), run, &caps(Some(16)));
    assert_eq!(clamped.max_rows, 16);
    assert!(clamped.fallback.unwrap().contains("clamped to 16"));

    let unadvertised = resolve_tree_verify(tree(8), run, &caps(None));
    assert_eq!(unadvertised.max_rows, 0);
    assert!(unadvertised
        .fallback
        .unwrap()
        .contains(TREE_VERIFY_NOT_ADVERTISED));

    let off = resolve_tree_verify(tree(8), crate::SpeculativePlan::Off, &caps(Some(16)));
    assert_eq!(off.max_rows, 0);
    assert!(off
        .fallback
        .unwrap()
        .starts_with("speculative: tree verification needs"));

    // An advertisement above the shared bound never runs past it.
    assert_eq!(
        resolve_tree_verify(tree(40), run, &caps(Some(64))).max_rows,
        MAX_TREE_ROWS
    );
}

#[test]
fn prompt_lookup_merges_every_earlier_match_into_one_tree() {
    // "1 2 3 4 | 1 2 5 6 | 1 2" — the trailing bigram "1 2" was followed by "5 6 1" and "3 4 1".
    let ctx = [1, 2, 3, 4, 1, 2, 5, 6, 1, 2];
    let tree = ngram_propose_tree(&ctx, 3, 3, 16);
    let linear = crate::ngram_propose(&ctx, 3, 3);
    assert_eq!(
        tree.path(linear.len())[1..]
            .iter()
            .map(|&r| tree.node(r).token)
            .collect::<Vec<_>>(),
        linear,
        "the first branch is the linear proposal"
    );
    assert_eq!(
        tree.children(0).len(),
        2,
        "two continuations branch at the anchor"
    );
    assert!(tree.max_depth() <= 3);
    // The row bound holds and keeps parents.
    let small = ngram_propose_tree(&ctx, 3, 3, 4);
    assert_eq!(small.len(), 4);
    // One match is a chain; no match is empty.
    assert!(ngram_propose_tree(&[7, 8, 9, 7], 3, 4, 16).is_chain());
    assert!(ngram_propose_tree(&[1, 2, 3], 3, 4, 16).is_empty());
}

#[test]
fn a_zero_row_tree_request_is_refused() {
    let mut req = crate::TextLlmRequest::new(vec![crate::Message::user("hi")], 4);
    req.speculative_verify = SpeculativeVerify::Tree { max_rows: 0 };
    let err = TextLlmCapabilities::default()
        .validate_request("t", &req)
        .unwrap_err();
    assert!(err.to_string().contains("max_rows >= 1"));
    req.speculative_verify = SpeculativeVerify::Tree { max_rows: 4 };
    TextLlmCapabilities::default()
        .validate_request("t", &req)
        .unwrap();
}

#[test]
fn the_verify_layout_has_a_stable_wire_form() {
    assert_eq!(
        serde_json::to_string(&SpeculativeVerify::Linear).unwrap(),
        "\"linear\""
    );
    let tree: SpeculativeVerify = serde_json::from_str(r#"{"tree":{"max_rows":8}}"#).unwrap();
    assert_eq!(tree, SpeculativeVerify::Tree { max_rows: 8 });
    assert_eq!(SpeculativeVerify::default(), SpeculativeVerify::Linear);
}

/// E6: only a proposer that builds branching trees (prompt lookup, and a block draft since
/// SW-7196) carries tree workspace; the MTP head and a draft model propose one branch, verified on
/// the linear path, so a tree request on them prices no tree rows — and no tree prices nothing for
/// anyone.
#[test]
fn only_a_tree_building_proposer_prices_tree_workspace() {
    use crate::ProposerKind;
    for kind in [ProposerKind::PromptLookup, ProposerKind::BlockDraft] {
        assert!(kind.builds_trees(), "{}", kind.label());
        assert_eq!(tree_workspace_rows(kind, 16), 16, "{}", kind.label());
        assert_eq!(tree_workspace_rows(kind, 0), 0, "{}", kind.label());
    }
    for kind in [
        ProposerKind::None,
        ProposerKind::Mtp,
        ProposerKind::DraftModel,
    ] {
        assert!(!kind.builds_trees(), "{}", kind.label());
        assert_eq!(tree_workspace_rows(kind, 16), 0, "{}", kind.label());
    }
    // Every request proposer resolves to the same capability.
    assert!(ProposerKind::from(SpeculativeProposer::PromptLookup).builds_trees());
    assert!(ProposerKind::from(SpeculativeProposer::BlockDraft).builds_trees());
    assert!(!ProposerKind::from(SpeculativeProposer::Mtp).builds_trees());
    assert!(!ProposerKind::from(SpeculativeProposer::DraftModel).builds_trees());
}
