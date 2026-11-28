//! Exact `SceneWorks/<repository>@<revision>` identities a provider binds a tier root to (SW-7285).
//!
//! A shipped q4/q8 tier has two published homes. The MLX tier tree (`SceneWorks/<family>-mlx`)
//! stores MLX-affine packed weights; its Candle-native twin (`SceneWorks/<family>-candle`) mirrors
//! that tree file for file but stores each quantized projection as GGML blocks, and it is what
//! SceneWorks resolves the q4/q8 tier to on Windows/Linux. A provider that pins its artifact to an
//! exact identity therefore carries both rows in one [`TierSources`] and admits exactly those two:
//! the primary identity for every tier it always admitted, and the twin only for the `q4`/`q8`
//! tiers the twin publishes. Matching is exact repository + exact revision, never a suffix or a
//! wildcard, so every other repository and every other revision of the twin stays refused.
//!
//! A family whose twin is not published yet carries `candle_twin: None` and stays primary-only;
//! publishing it is the one-line change of naming its [`SceneWorksSnapshot`].

use std::path::Path;

/// The tier directories a Candle twin publishes. Its dense `bf16` tier (if the mirrored tree has
/// one) is never what SceneWorks resolves, so the twin identity is not admitted for it.
pub const CANDLE_TWIN_TIERS: &[&str] = &["q4", "q8"];

/// One exact published `SceneWorks/<repository>@<revision>` snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SceneWorksSnapshot {
    /// The repository name under the `SceneWorks` owner, without the `SceneWorks/` prefix.
    pub repository: &'static str,
    /// The exact commit the manifest pins.
    pub revision: &'static str,
}

impl SceneWorksSnapshot {
    pub const fn new(repository: &'static str, revision: &'static str) -> Self {
        Self {
            repository,
            revision,
        }
    }

    /// `models--SceneWorks--<repository>`: the Hugging Face cache directory.
    pub fn hf_cache_dir(&self) -> String {
        format!("models--SceneWorks--{}", self.repository)
    }

    /// `SceneWorks__<repository>`: the app-owned cache directory.
    pub fn app_cache_dir(&self) -> String {
        format!("SceneWorks__{}", self.repository)
    }

    /// The three on-disk spellings of this snapshot followed by `trailing`: the Hugging Face cache
    /// (`models--SceneWorks--<repo>/snapshots/<rev>/…`), the app-owned cache
    /// (`SceneWorks__<repo>/<rev>/…`) and a plain local mirror (`<repo>/<rev>/…`).
    pub fn layouts(&self, trailing: &[&str]) -> [Vec<String>; 3] {
        let with = |prefix: Vec<String>| {
            let mut suffix = prefix;
            suffix.extend(trailing.iter().map(|part| (*part).to_owned()));
            suffix
        };
        [
            with(vec![
                self.hf_cache_dir(),
                "snapshots".to_owned(),
                self.revision.to_owned(),
            ]),
            with(vec![self.app_cache_dir(), self.revision.to_owned()]),
            with(vec![self.repository.to_owned(), self.revision.to_owned()]),
        ]
    }

    /// The length of the layout `components` ends in (see [`Self::layouts`]), or `None` when the
    /// path spells no layout of this exact repository and revision.
    pub fn suffix_len(&self, components: &[String], trailing: &[&str]) -> Option<usize> {
        self.layouts(trailing)
            .into_iter()
            .find(|suffix| components.ends_with(suffix))
            .map(|suffix| suffix.len())
    }
}

impl std::fmt::Display for SceneWorksSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SceneWorks/{}@{}", self.repository, self.revision)
    }
}

/// The exact identities one route's tier root may be bound to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TierSources {
    /// The identity the route has always pinned (the MLX tier tree, or a dense re-host).
    pub primary: SceneWorksSnapshot,
    /// The published Candle-native GGML twin of `primary`'s q4/q8 tiers, once it exists.
    pub candle_twin: Option<SceneWorksSnapshot>,
}

impl TierSources {
    pub const fn primary_only(primary: SceneWorksSnapshot) -> Self {
        Self {
            primary,
            candle_twin: None,
        }
    }

    pub const fn with_candle_twin(primary: SceneWorksSnapshot, twin: SceneWorksSnapshot) -> Self {
        Self {
            primary,
            candle_twin: Some(twin),
        }
    }

    /// The identities a root ending in `tier` may come from: the primary for every tier, the twin
    /// only for [`CANDLE_TWIN_TIERS`].
    pub fn admitted(&self, tier: &str) -> impl Iterator<Item = SceneWorksSnapshot> {
        let twin = self
            .candle_twin
            .filter(|_| CANDLE_TWIN_TIERS.contains(&tier));
        std::iter::once(self.primary).chain(twin)
    }

    /// The admitted identity whose exact layout `components` end in, followed by `tier`, with the
    /// number of trailing components that layout spans.
    pub fn bind(&self, components: &[String], tier: &str) -> Option<(SceneWorksSnapshot, usize)> {
        self.admitted(tier).find_map(|source| {
            source
                .suffix_len(components, &[tier])
                .map(|depth| (source, depth))
        })
    }

    /// `SceneWorks/a@x` or `SceneWorks/a@x or SceneWorks/b@y`, for refusal messages.
    pub fn describe(&self, tier: &str) -> String {
        self.admitted(tier)
            .map(|source| source.to_string())
            .collect::<Vec<_>>()
            .join(" or ")
    }
}

/// `path`'s components as lossy UTF-8 strings, the shape [`TierSources::bind`] compares.
pub fn path_components(path: &Path) -> Vec<String> {
    path.components()
        .map(|part| part.as_os_str().to_string_lossy().into_owned())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const MLX: SceneWorksSnapshot =
        SceneWorksSnapshot::new("demo-mlx", "1111111111111111111111111111111111111111");
    const TWIN: SceneWorksSnapshot =
        SceneWorksSnapshot::new("demo-candle", "2222222222222222222222222222222222222222");

    fn parts(path: &str) -> Vec<String> {
        path.split('/').map(str::to_owned).collect()
    }

    #[test]
    fn primary_and_twin_bind_only_their_own_exact_identity() {
        let sources = TierSources::with_candle_twin(MLX, TWIN);
        for (path, tier, expected) in [
            (
                "hub/models--SceneWorks--demo-mlx/snapshots/1111111111111111111111111111111111111111/q4",
                "q4",
                Some(MLX),
            ),
            (
                "hub/models--SceneWorks--demo-candle/snapshots/2222222222222222222222222222222222222222/q4",
                "q4",
                Some(TWIN),
            ),
            (
                "app/SceneWorks__demo-candle/2222222222222222222222222222222222222222/q8",
                "q8",
                Some(TWIN),
            ),
            (
                "mirror/demo-candle/2222222222222222222222222222222222222222/q8",
                "q8",
                Some(TWIN),
            ),
            // The twin never stands in for the dense tier.
            (
                "hub/models--SceneWorks--demo-candle/snapshots/2222222222222222222222222222222222222222/bf16",
                "bf16",
                None,
            ),
            // Repository and revision are paired: neither crosses to the other identity.
            (
                "hub/models--SceneWorks--demo-candle/snapshots/1111111111111111111111111111111111111111/q4",
                "q4",
                None,
            ),
            (
                "hub/models--SceneWorks--demo-mlx/snapshots/2222222222222222222222222222222222222222/q4",
                "q4",
                None,
            ),
            (
                "hub/models--SceneWorks--demo-candle-ggml/snapshots/2222222222222222222222222222222222222222/q4",
                "q4",
                None,
            ),
            (
                "hub/models--Other--demo-candle/snapshots/2222222222222222222222222222222222222222/q4",
                "q4",
                None,
            ),
        ] {
            assert_eq!(
                sources.bind(&parts(path), tier).map(|(source, _)| source),
                expected,
                "{path}"
            );
        }
    }

    #[test]
    fn an_unpublished_twin_admits_only_the_primary() {
        let sources = TierSources::primary_only(MLX);
        assert!(sources
            .bind(
                &parts("hub/models--SceneWorks--demo-candle/snapshots/2222222222222222222222222222222222222222/q4"),
                "q4"
            )
            .is_none());
        assert_eq!(
            sources.bind(
                &parts("hub/models--SceneWorks--demo-mlx/snapshots/1111111111111111111111111111111111111111/q4"),
                "q4"
            ),
            Some((MLX, 4))
        );
        assert_eq!(
            sources.describe("q4"),
            "SceneWorks/demo-mlx@1111111111111111111111111111111111111111"
        );
        assert_eq!(
            TierSources::with_candle_twin(MLX, TWIN).describe("q8"),
            "SceneWorks/demo-mlx@1111111111111111111111111111111111111111 or \
             SceneWorks/demo-candle@2222222222222222222222222222222222222222"
        );
    }
}
