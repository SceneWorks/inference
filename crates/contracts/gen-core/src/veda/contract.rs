//! The Veda tile shape that keys a [`WorkspaceBound`](crate::veda::workspace::WorkspaceBound) chunk table.

use crate::{Error, Result};

pub const TILE: usize = 128;
fn bad(message: impl Into<String>) -> Error {
    Error::Msg(format!("Veda predictor: {}", message.into()))
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct TileShape(pub [usize; 3]);
impl TileShape {
    pub fn parse(s: &str) -> Result<Self> {
        let v: Vec<usize> = s
            .split('x')
            .map(str::parse)
            .collect::<std::result::Result<_, _>>()
            .map_err(|_| bad("invalid tile shape"))?;
        if v.len() != 3
            || v.contains(&0)
            || v.iter().try_fold(1usize, |n, &v| n.checked_mul(v)) != Some(TILE)
        {
            return Err(bad("tile shape must hold 128 rows"));
        }
        Ok(Self([v[0], v[1], v[2]]))
    }
    pub fn tiles(self, g: [usize; 3]) -> usize {
        (0..3).map(|i| g[i].div_ceil(self.0[i])).product()
    }
    pub fn transpose(self) -> Self {
        Self([self.0[0], self.0[2], self.0[1]])
    }
}
