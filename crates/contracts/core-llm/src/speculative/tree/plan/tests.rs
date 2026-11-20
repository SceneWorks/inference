//! The tree execution plan on host data: parents and depths, conv windows (a chain's linear
//! windows, branch taps, pre-step history above the anchor) and depth groups.

use super::*;

/// ```text
/// 0 ─┬─ 1 ─── 4
///    └─ 2 ─┬─ 3
///          └─ 5 ── 6
/// ```
fn branching() -> TreePlan {
    TreePlan::from_parents(&[0, 0, 2, 1, 2, 5])
}

#[test]
fn parents_and_depths_follow_the_tree() {
    let plan = branching();
    assert_eq!(plan.rows(), 7);
    assert_eq!(
        plan.parents,
        vec![None, Some(0), Some(0), Some(2), Some(1), Some(2), Some(5)]
    );
    assert_eq!(plan.depths, vec![0, 1, 1, 2, 2, 2, 3]);
    let mut tree = DraftTree::new();
    for (p, t) in [(0, 10), (0, 11), (2, 12), (1, 13), (2, 14), (5, 15)] {
        tree.push(p, t, None).unwrap();
    }
    assert_eq!(
        TreePlan::new(&tree),
        plan,
        "a DraftTree plans as its parents do"
    );
    let anchor = TreePlan::new(&DraftTree::new());
    assert_eq!((anchor.rows(), anchor.depths.clone()), (1, vec![0]));
}

#[test]
fn a_chain_gives_the_linear_conv_windows() {
    let chain = TreePlan::from_parents(&[0, 1, 2]);
    for k in 1..=5 {
        let linear: Vec<Vec<usize>> = (0..4).map(|s| (s..s + k).collect()).collect();
        assert_eq!(chain.conv_windows(k), linear, "k = {k}");
    }
}

#[test]
fn a_branch_reads_its_ancestors_then_the_pre_step_history() {
    let plan = branching();
    // `[h0 h1 h2 | r0 .. r6]` (k = 4): indices 0..3 are the pre-step history, 3 + r row r.
    let windows = plan.conv_windows(4);
    assert_eq!(windows[0], vec![0, 1, 2, 3], "the anchor reads all history");
    assert_eq!(windows[4], vec![2, 3, 4, 7], "row 4: h2, r0, r1, r4");
    assert_eq!(windows[3], vec![2, 3, 5, 6], "row 3: h2, r0, r2, r3");
    assert_eq!(windows[6], vec![3, 5, 8, 9], "row 6: r0, r2, r5, r6");
    // A shorter kernel never reaches the history once the branch covers it (indices shift with
    // the history's length, `k - 1`).
    assert_eq!(
        plan.conv_windows(2)[6],
        vec![6, 7],
        "`[h0 | r0 ..]`: r5, r6"
    );
    assert_eq!(plan.conv_windows(1)[6], vec![6]);
    // A longer one reaches past the anchor into the history.
    assert_eq!(plan.conv_windows(6)[3], vec![2, 3, 4, 5, 7, 8]);
}

#[test]
fn levels_group_rows_by_depth_in_row_order() {
    assert_eq!(
        branching().levels(),
        vec![vec![0], vec![1, 2], vec![3, 4, 5], vec![6]]
    );
    assert_eq!(TreePlan::from_parents(&[]).levels(), vec![vec![0]]);
}
