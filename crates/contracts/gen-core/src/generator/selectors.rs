//! Typed boundary selectors for [`GenerationRequest`](super::GenerationRequest) (`REG-4`).
//!
//! Each selector enum gives callers a strongly typed vocabulary at the contract boundary while
//! round-tripping losslessly with canonical lowercase strings for wire compatibility, recipes,
//! and explicit refusal of unrecognized values during validation.

use std::fmt;
use std::ops::Deref;

use serde::{Deserialize, Serialize};

use crate::guidance::GuidanceMethod;
use crate::sampling::{Scheduler, Solver};

/// Typed sampler/solver selector at the request boundary (`REG-4`).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(from = "String", into = "String")]
pub enum SolverId {
    Euler,
    EulerAncestral,
    Heun,
    Dpmpp2m,
    DpmppSde,
    UniPc,
    Lcm,
    Ddim,
    ErSde,
    Dpmpp2mSde,
    /// Model-native flow-matching sampler (`"flow_match"`).
    FlowMatch,
    /// Model-native DPM++ sampler alias (`"dpmpp"`).
    Dpmpp,
    /// Compact DPM++ 2M spelling (`"dpmpp2m"`).
    Dpmpp2mCompact,
    /// Compact UniPC spelling (`"unipc"`).
    UniPcCompact,
    /// Ping-pong flow sampler (`"pingpong"`).
    PingPong,
    /// Runge-Kutta 4th order sampler (`"rk4"`).
    Rk4,
    /// SeedVR2 Euler sampler (`"seedvr2_euler"`).
    SeedVr2Euler,
    /// Explicit default sentinel (`"default"`).
    Default,
    /// Unrecognized or provider-specific sampler name preserved losslessly for validation refusal.
    Custom(String),
}

impl SolverId {
    /// Parse a canonical lowercase sampler name into its typed variant, preserving unrecognized
    /// names in [`Self::Custom`] so validation can reject them with the exact input string.
    pub fn from_name(name: &str) -> Self {
        match name {
            "euler" => Self::Euler,
            "euler_ancestral" => Self::EulerAncestral,
            "heun" => Self::Heun,
            "dpmpp_2m" => Self::Dpmpp2m,
            "dpmpp_sde" => Self::DpmppSde,
            "uni_pc" => Self::UniPc,
            "lcm" => Self::Lcm,
            "ddim" => Self::Ddim,
            "er_sde" => Self::ErSde,
            "dpmpp_2m_sde" => Self::Dpmpp2mSde,
            "flow_match" => Self::FlowMatch,
            "dpmpp" => Self::Dpmpp,
            "dpmpp2m" => Self::Dpmpp2mCompact,
            "unipc" => Self::UniPcCompact,
            "pingpong" => Self::PingPong,
            "rk4" => Self::Rk4,
            "seedvr2_euler" => Self::SeedVr2Euler,
            "default" => Self::Default,
            other => Self::Custom(other.to_owned()),
        }
    }

    /// Canonical lowercase string form (round-trips with [`Self::from_name`]).
    pub fn as_str(&self) -> &str {
        match self {
            Self::Euler => "euler",
            Self::EulerAncestral => "euler_ancestral",
            Self::Heun => "heun",
            Self::Dpmpp2m => "dpmpp_2m",
            Self::DpmppSde => "dpmpp_sde",
            Self::UniPc => "uni_pc",
            Self::Lcm => "lcm",
            Self::Ddim => "ddim",
            Self::ErSde => "er_sde",
            Self::Dpmpp2mSde => "dpmpp_2m_sde",
            Self::FlowMatch => "flow_match",
            Self::Dpmpp => "dpmpp",
            Self::Dpmpp2mCompact => "dpmpp2m",
            Self::UniPcCompact => "unipc",
            Self::PingPong => "pingpong",
            Self::Rk4 => "rk4",
            Self::SeedVr2Euler => "seedvr2_euler",
            Self::Default => "default",
            Self::Custom(s) => s.as_str(),
        }
    }

    /// Map to the shared unified [`Solver`] vocabulary when applicable.
    pub fn as_solver(&self) -> Option<Solver> {
        Solver::from_name(self.as_str())
    }
}

impl Deref for SolverId {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        self.as_str()
    }
}

impl AsRef<str> for SolverId {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for SolverId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<Solver> for SolverId {
    fn from(value: Solver) -> Self {
        Self::from_name(value.name())
    }
}

impl From<&str> for SolverId {
    fn from(value: &str) -> Self {
        Self::from_name(value)
    }
}

impl From<String> for SolverId {
    fn from(value: String) -> Self {
        Self::from_name(&value)
    }
}

impl From<SolverId> for String {
    fn from(value: SolverId) -> Self {
        match value {
            SolverId::Custom(s) => s,
            other => other.as_str().to_owned(),
        }
    }
}

impl PartialEq<&str> for SolverId {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

impl PartialEq<str> for SolverId {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

/// Typed sigma-schedule / scheduler selector at the request boundary (`REG-4`).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(from = "String", into = "String")]
pub enum ScheduleId {
    Normal,
    Simple,
    Karras,
    Exponential,
    SgmUniform,
    Beta,
    DdimUniform,
    Beta57,
    /// Linear flow schedule (`"linear"`).
    Linear,
    /// Flow-match Euler schedule (`"flow_match_euler"`).
    FlowMatchEuler,
    /// Training-grid schedule (`"moss_training_grid"`).
    MossTrainingGrid,
    /// SeedVR2 Euler schedule (`"seedvr2_euler"`).
    SeedVr2Euler,
    /// Explicit default sentinel (`"default"`).
    Default,
    /// Unrecognized or provider-specific schedule name preserved losslessly for validation refusal.
    Custom(String),
}

impl ScheduleId {
    /// Parse a canonical lowercase schedule name into its typed variant.
    pub fn from_name(name: &str) -> Self {
        match name {
            "normal" => Self::Normal,
            "simple" => Self::Simple,
            "karras" => Self::Karras,
            "exponential" => Self::Exponential,
            "sgm_uniform" => Self::SgmUniform,
            "beta" => Self::Beta,
            "ddim_uniform" => Self::DdimUniform,
            "beta57" => Self::Beta57,
            "linear" => Self::Linear,
            "flow_match_euler" => Self::FlowMatchEuler,
            "moss_training_grid" => Self::MossTrainingGrid,
            "seedvr2_euler" => Self::SeedVr2Euler,
            "default" => Self::Default,
            other => Self::Custom(other.to_owned()),
        }
    }

    /// Canonical lowercase string form (round-trips with [`Self::from_name`]).
    pub fn as_str(&self) -> &str {
        match self {
            Self::Normal => "normal",
            Self::Simple => "simple",
            Self::Karras => "karras",
            Self::Exponential => "exponential",
            Self::SgmUniform => "sgm_uniform",
            Self::Beta => "beta",
            Self::DdimUniform => "ddim_uniform",
            Self::Beta57 => "beta57",
            Self::Linear => "linear",
            Self::FlowMatchEuler => "flow_match_euler",
            Self::MossTrainingGrid => "moss_training_grid",
            Self::SeedVr2Euler => "seedvr2_euler",
            Self::Default => "default",
            Self::Custom(s) => s.as_str(),
        }
    }

    /// Map to the shared unified [`Scheduler`] vocabulary when applicable.
    pub fn as_scheduler(&self) -> Option<Scheduler> {
        Scheduler::from_name(self.as_str())
    }
}

impl Deref for ScheduleId {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        self.as_str()
    }
}

impl AsRef<str> for ScheduleId {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for ScheduleId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<Scheduler> for ScheduleId {
    fn from(value: Scheduler) -> Self {
        Self::from_name(value.name())
    }
}

impl From<&str> for ScheduleId {
    fn from(value: &str) -> Self {
        Self::from_name(value)
    }
}

impl From<String> for ScheduleId {
    fn from(value: String) -> Self {
        Self::from_name(&value)
    }
}

impl From<ScheduleId> for String {
    fn from(value: ScheduleId) -> Self {
        match value {
            ScheduleId::Custom(s) => s,
            other => other.as_str().to_owned(),
        }
    }
}

impl PartialEq<&str> for ScheduleId {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

impl PartialEq<str> for ScheduleId {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

/// Typed guidance-method selector at the request boundary (`REG-4`).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(from = "String", into = "String")]
pub enum GuidanceMethodId {
    Cfg,
    CfgRescale,
    Apg,
    CfgPp,
    /// Unrecognized guidance method preserved losslessly for validation refusal.
    Custom(String),
}

impl GuidanceMethodId {
    /// Parse a canonical lowercase guidance method name into its typed variant.
    pub fn from_name(name: &str) -> Self {
        match name {
            "cfg" => Self::Cfg,
            "cfg_rescale" => Self::CfgRescale,
            "apg" => Self::Apg,
            "cfg_pp" => Self::CfgPp,
            other => Self::Custom(other.to_owned()),
        }
    }

    /// Canonical lowercase string form (round-trips with [`Self::from_name`]).
    pub fn as_str(&self) -> &str {
        match self {
            Self::Cfg => "cfg",
            Self::CfgRescale => "cfg_rescale",
            Self::Apg => "apg",
            Self::CfgPp => "cfg_pp",
            Self::Custom(s) => s.as_str(),
        }
    }

    /// Map to the closed [`GuidanceMethod`] operator vocabulary when known.
    pub fn as_guidance_method(&self) -> Option<GuidanceMethod> {
        GuidanceMethod::from_name(self.as_str())
    }
}

impl Deref for GuidanceMethodId {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        self.as_str()
    }
}

impl AsRef<str> for GuidanceMethodId {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for GuidanceMethodId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<GuidanceMethod> for GuidanceMethodId {
    fn from(value: GuidanceMethod) -> Self {
        Self::from_name(value.name())
    }
}

impl From<&str> for GuidanceMethodId {
    fn from(value: &str) -> Self {
        Self::from_name(value)
    }
}

impl From<String> for GuidanceMethodId {
    fn from(value: String) -> Self {
        Self::from_name(&value)
    }
}

impl From<GuidanceMethodId> for String {
    fn from(value: GuidanceMethodId) -> Self {
        match value {
            GuidanceMethodId::Custom(s) => s,
            other => other.as_str().to_owned(),
        }
    }
}

impl PartialEq<&str> for GuidanceMethodId {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

impl PartialEq<str> for GuidanceMethodId {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

/// Typed video-mode selector at the request boundary (`REG-4`).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(from = "String", into = "String")]
pub enum VideoMode {
    TextToVideo,
    ImageToVideo,
    VideoToVideo,
    FirstLastFrame,
    ExtendClip,
    VideoBridge,
    Animation,
    Replacement,
    ReferenceToVideo,
    NoAudio,
    T2i,
    I2i,
    V2v,
    R2v,
    Rv2v,
    Mv2v,
    Ads2v,
    T2vApg,
    R2vApg,
    Rv2vWapg,
    /// Unrecognized or provider-specific mode name preserved losslessly for validation refusal.
    Custom(String),
}

impl VideoMode {
    /// Parse a canonical lowercase video mode name into its typed variant.
    pub fn from_name(name: &str) -> Self {
        match name {
            "text_to_video" => Self::TextToVideo,
            "image_to_video" => Self::ImageToVideo,
            "video_to_video" => Self::VideoToVideo,
            "first_last_frame" => Self::FirstLastFrame,
            "extend_clip" => Self::ExtendClip,
            "video_bridge" => Self::VideoBridge,
            "animation" => Self::Animation,
            "replacement" => Self::Replacement,
            "reference_to_video" => Self::ReferenceToVideo,
            "no_audio" => Self::NoAudio,
            "t2i" => Self::T2i,
            "i2i" => Self::I2i,
            "v2v" => Self::V2v,
            "r2v" => Self::R2v,
            "rv2v" => Self::Rv2v,
            "mv2v" => Self::Mv2v,
            "ads2v" => Self::Ads2v,
            "t2v_apg" => Self::T2vApg,
            "r2v_apg" => Self::R2vApg,
            "rv2v_wapg" => Self::Rv2vWapg,
            other => Self::Custom(other.to_owned()),
        }
    }

    /// Canonical lowercase string form (round-trips with [`Self::from_name`]).
    pub fn as_str(&self) -> &str {
        match self {
            Self::TextToVideo => "text_to_video",
            Self::ImageToVideo => "image_to_video",
            Self::VideoToVideo => "video_to_video",
            Self::FirstLastFrame => "first_last_frame",
            Self::ExtendClip => "extend_clip",
            Self::VideoBridge => "video_bridge",
            Self::Animation => "animation",
            Self::Replacement => "replacement",
            Self::ReferenceToVideo => "reference_to_video",
            Self::NoAudio => "no_audio",
            Self::T2i => "t2i",
            Self::I2i => "i2i",
            Self::V2v => "v2v",
            Self::R2v => "r2v",
            Self::Rv2v => "rv2v",
            Self::Mv2v => "mv2v",
            Self::Ads2v => "ads2v",
            Self::T2vApg => "t2v_apg",
            Self::R2vApg => "r2v_apg",
            Self::Rv2vWapg => "rv2v_wapg",
            Self::Custom(s) => s.as_str(),
        }
    }
}

impl Deref for VideoMode {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        self.as_str()
    }
}

impl AsRef<str> for VideoMode {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for VideoMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<&str> for VideoMode {
    fn from(value: &str) -> Self {
        Self::from_name(value)
    }
}

impl From<String> for VideoMode {
    fn from(value: String) -> Self {
        Self::from_name(&value)
    }
}

impl From<VideoMode> for String {
    fn from(value: VideoMode) -> Self {
        match value {
            VideoMode::Custom(s) => s,
            other => other.as_str().to_owned(),
        }
    }
}

impl PartialEq<&str> for VideoMode {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

impl PartialEq<str> for VideoMode {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}
