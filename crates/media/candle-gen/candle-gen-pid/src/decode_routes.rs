//! The per-route bounded-decode domain a PiD-eligible Candle provider declares, and its
//! enforcement (SC-15775 mirror for Candle).

use crate::budget;

fn route_label(use_pid: bool) -> &'static str {
    if use_pid {
        "PiD overlay"
    } else {
        "native VAE"
    }
}

fn conformance_errors(provider_id: &str, native_edges: &[u32], native_overlap: u32) -> Vec<String> {
    let mut errors = Vec::new();
    if native_edges.is_empty() {
        errors.push(format!(
            "{provider_id}: declares no native decode tile-edge candidates, so the native route has \
             no domain to be disjoint from the PiD one"
        ));
    }
    for &edge in native_edges {
        if edge == 0 {
            errors.push(format!(
                "{provider_id}: native decode tile edge 0 is degenerate"
            ));
            continue;
        }
        if budget::is_tile_edge_candidate(edge as i32) {
            errors.push(format!(
                "{provider_id}: native decode tile edge {edge} is ALSO a legal PiD tile edge ({:?}). \
                 The two routes' domains must be disjoint.",
                budget::tile_edge_candidates()
            ));
        }
        if native_overlap >= edge {
            errors.push(format!(
                "{provider_id}: native decode overlap {native_overlap} is not smaller than \
                 candidate edge {edge}, which makes the tile split degenerate"
            ));
        }
    }
    if native_overlap == 0 {
        errors.push(format!(
            "{provider_id}: native decode overlap must be positive (a zero-overlap split seams)"
        ));
    }
    let pid_edges = DecodeRoutes::pid_edges();
    if pid_edges.is_empty() {
        errors.push(format!(
            "{provider_id}: the PiD route publishes no candidates, so a bounded-decode selection on \
             it can never validate"
        ));
    }
    for edge in pid_edges {
        if !budget::is_tile_edge_candidate(edge as i32) {
            errors.push(format!(
                "{provider_id}: published PiD tile edge {edge} is not one of \
                 `budget::tile_edge_candidates` {:?}",
                budget::tile_edge_candidates()
            ));
        }
        if DecodeRoutes::pid_overlap() == 0 || DecodeRoutes::pid_overlap() >= edge {
            errors.push(format!(
                "{provider_id}: the PiD overlap {} must be in 1..{edge}",
                DecodeRoutes::pid_overlap()
            ));
        }
    }
    errors
}

/// The two bounded-decode routes a PiD-eligible Candle provider runs, and the single place their
/// candidate domains are reconciled.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodeRoutes {
    provider_id: String,
    native_edges: Vec<u32>,
    native_overlap: u32,
}

impl DecodeRoutes {
    pub fn new(
        provider_id: &str,
        native_edges: impl IntoIterator<Item = u32>,
        native_overlap: u32,
    ) -> Result<Self, Vec<String>> {
        let mut native_edges: Vec<u32> = native_edges.into_iter().collect();
        native_edges.sort_unstable_by(|a, b| b.cmp(a));
        native_edges.dedup();
        let errors = conformance_errors(provider_id, &native_edges, native_overlap);
        if !errors.is_empty() {
            return Err(errors);
        }
        Ok(Self {
            provider_id: provider_id.to_owned(),
            native_edges,
            native_overlap,
        })
    }

    pub fn new_core(
        provider_id: &str,
        native_edges: impl IntoIterator<Item = u32>,
        native_overlap: u32,
    ) -> candle_gen::gen_core::Result<Self> {
        Self::new(provider_id, native_edges, native_overlap)
            .map_err(|errors| candle_gen::gen_core::Error::Unsupported(errors.join("; ")))
    }

    pub fn native_edges(&self) -> &[u32] {
        &self.native_edges
    }

    pub fn pid_edges() -> Vec<u32> {
        vec![budget::PREFERRED_TILE_EDGE as u32]
    }

    pub fn pid_overlap() -> u32 {
        budget::DEFAULT_TILE_OVERLAP as u32
    }

    pub fn domain(&self, use_pid: bool) -> (Vec<u32>, u32) {
        if use_pid {
            (Self::pid_edges(), Self::pid_overlap())
        } else {
            (self.native_edges.clone(), self.native_overlap)
        }
    }

    pub fn published_edges(&self) -> Vec<u32> {
        let mut edges: Vec<u32> = self
            .native_edges
            .iter()
            .copied()
            .chain(Self::pid_edges())
            .collect();
        edges.sort_unstable_by(|a, b| b.cmp(a));
        edges.dedup();
        edges
    }

    pub fn published_overlaps(&self) -> Vec<u32> {
        let mut overlaps = vec![Self::pid_overlap(), self.native_overlap];
        overlaps.sort_unstable_by(|a, b| b.cmp(a));
        overlaps.dedup();
        overlaps
    }

    pub fn validate(
        &self,
        use_pid: bool,
        edge: Option<u32>,
        overlap: Option<u32>,
    ) -> Result<(), String> {
        let (edges, expected_overlap) = self.domain(use_pid);
        let label = route_label(use_pid);
        let provider_id = &self.provider_id;
        match edge {
            Some(edge) if edges.contains(&edge) => {}
            Some(edge) => {
                return Err(format!(
                    "{provider_id}: decode tile edge {edge} is not a {label} route candidate \
                     {edges:?}"
                ));
            }
            None => {
                return Err(format!(
                    "{provider_id}: a bounded decode on the {label} route named no tile edge; the \
                     candidates are {edges:?}"
                ));
            }
        }
        match overlap {
            Some(overlap) if overlap == expected_overlap => {}
            Some(overlap) => {
                return Err(format!(
                    "{provider_id}: the {label} route's decode overlap is {expected_overlap}, got \
                     {overlap}"
                ));
            }
            None => {
                return Err(format!(
                    "{provider_id}: a bounded decode on the {label} route named no overlap; it must \
                     be {expected_overlap}"
                ));
            }
        }
        Ok(())
    }
}

pub fn assert_decode_routes(
    provider_id: &str,
    native_edges: impl IntoIterator<Item = u32>,
    native_overlap: u32,
) -> DecodeRoutes {
    match DecodeRoutes::new(provider_id, native_edges, native_overlap) {
        Ok(routes) => routes,
        Err(errors) => panic!(
            "PiD decode-route conformance FAILED for '{provider_id}':\n- {}",
            errors.join("\n- ")
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_routes_enforces_disjointness_and_route_domains() {
        let routes = assert_decode_routes("z_image_turbo", [768, 640, 512], 64);
        assert_eq!(routes.published_edges(), vec![2048, 768, 640, 512]);
        assert!(routes.validate(false, Some(512), Some(64)).is_ok());
        assert!(routes.validate(true, Some(512), Some(64)).is_err());
        assert!(routes.validate(true, Some(2048), Some(256)).is_ok());
        assert!(routes.validate(false, Some(2048), Some(256)).is_err());
        assert!(DecodeRoutes::new("bad", [2048, 768], 64).is_err());
    }
}
