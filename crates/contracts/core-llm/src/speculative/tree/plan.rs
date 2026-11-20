//! The host-index plan of one tree verify forward (feature SW-F328) as a hybrid model's
//! recurrent (DeltaNet) layers see it: each row's parent and depth, every row's short-conv input
//! window over `[conv_state ‖ tree rows]`, and the rows grouped by depth for the level-by-level
//! recurrence. Both backends consume the same plan and convert its indices to their tensor index
//! type at the tensor boundary.

use super::DraftTree;

/// The shape of one tree verify forward: each row's parent row (`None` for the anchor, row `0`)
/// and its depth below the anchor. Rows are topologically ordered (a parent precedes its
/// children).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TreePlan {
    /// The parent row of every row; `parents[0]` is `None`.
    pub parents: Vec<Option<usize>>,
    /// Every row's depth below the anchor.
    pub depths: Vec<usize>,
}

impl TreePlan {
    /// The plan of `tree`'s rows (the anchor, then every candidate).
    pub fn new(tree: &DraftTree) -> Self {
        let parents: Vec<usize> = tree.nodes().iter().map(|n| n.parent).collect();
        Self::from_parents(&parents)
    }

    /// The plan of a tree whose row `r >= 1` continues row `node_parents[r - 1]` (each parent
    /// below its child's row).
    pub fn from_parents(node_parents: &[usize]) -> Self {
        let mut parents = vec![None];
        let mut depths = vec![0];
        for &p in node_parents {
            depths.push(depths[p] + 1);
            parents.push(Some(p));
        }
        Self { parents, depths }
    }

    /// Rows, the anchor included.
    pub fn rows(&self) -> usize {
        self.parents.len()
    }

    /// The short conv's input rows for every tree row under a `k`-tap kernel (`k >= 1`):
    /// `windows[r][j]` indexes `[conv_state ‖ tree rows]` (`k - 1 + rows` rows) with the row tap
    /// `j` (`0` the oldest) of row `r` reads — the row itself for the newest tap, then its
    /// ancestors, then the pre-step conv history once the branch above the anchor runs out. A
    /// chain gives exactly the linear conv's contiguous windows.
    pub fn conv_windows(&self, k: usize) -> Vec<Vec<usize>> {
        let tail = k - 1;
        (0..self.rows())
            .map(|r| {
                let mut back = Vec::with_capacity(k); // newest first
                let mut at = Some(r);
                for m in 0..k {
                    match at {
                        Some(row) => {
                            back.push(tail + row);
                            at = self.parents[row];
                        }
                        // `m - depth - 1` rows into the history before the anchor.
                        None => back.push(tail + self.depths[r] - m),
                    }
                }
                back.reverse();
                back
            })
            .collect()
    }

    /// Rows grouped by depth, shallowest first, each group in row order.
    pub fn levels(&self) -> Vec<Vec<usize>> {
        let max = self.depths.iter().copied().max().unwrap_or(0);
        let mut levels = vec![Vec::new(); max + 1];
        for (r, &d) in self.depths.iter().enumerate() {
            levels[d].push(r);
        }
        levels
    }
}

#[cfg(test)]
mod tests;
