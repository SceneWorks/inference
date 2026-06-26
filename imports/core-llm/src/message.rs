//! The multimodal, multi-turn message model.
//!
//! Generalises gen-core's flat `system` + `prompt` strings into roles + content blocks carrying
//! **text and images**, which is what a chat + vision contract needs (and what the chat templates
//! render). Images are carried as raw RGB8 bytes so the contract stays tensor-free; a backend lifts
//! them into its own tensors.

/// The author of a message turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// System / developer instructions.
    System,
    /// An end-user turn.
    User,
    /// A model turn (prior assistant output, for multi-turn context).
    Assistant,
    /// A tool / function result turn.
    Tool,
}

impl Role {
    /// The lowercase wire name (`"system"`, `"user"`, `"assistant"`, `"tool"`).
    pub fn as_str(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

/// A decoded RGB image (row-major, 3 bytes per pixel). Tensor-free so the contract carries no
/// backend types; a provider lifts it into a tensor at its boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageRef {
    /// Pixel width.
    pub width: u32,
    /// Pixel height.
    pub height: u32,
    /// `width * height * 3` RGB8 bytes.
    pub pixels: Vec<u8>,
}

impl ImageRef {
    /// Construct, validating that `pixels` is exactly `width * height * 3` bytes.
    pub fn new(width: u32, height: u32, pixels: Vec<u8>) -> Result<Self, String> {
        let expected = width as usize * height as usize * 3;
        if pixels.len() != expected {
            return Err(format!(
                "ImageRef: {}x{} needs {expected} RGB bytes, got {}",
                width,
                height,
                pixels.len()
            ));
        }
        Ok(Self { width, height, pixels })
    }
}

/// A sampled video: an ordered list of decoded RGB frames plus each frame's timestamp (seconds).
/// Tensor-free, like [`ImageRef`] — the host samples/decodes the video into frames at its boundary
/// and a provider lifts the frames into tensors and renders the per-frame timestamps.
///
/// **Text–Timestamp Alignment.** Qwen3-VL grounds temporal questions by interleaving each frame with
/// its wall-clock timestamp in the prompt (`<{t:.1f} seconds>` before each frame's vision block). The
/// per-frame [`timestamps`](Self::timestamps) carry exactly that data, so a provider can reproduce
/// the reference timestamp tokens without re-deriving them from a frame index + fps. They are
/// **already merged** by the model's `temporal_patch_size` (one timestamp per emitted vision frame,
/// i.e. per merged temporal patch), matching `Qwen3VLProcessor.replace_video_token`.
#[derive(Clone, Debug, PartialEq)]
pub struct VideoRef {
    /// The sampled frames, in temporal order. Each is a decoded RGB8 [`ImageRef`].
    pub frames: Vec<ImageRef>,
    /// Per-frame timestamp in seconds (one per [`frames`](Self::frames) entry), in temporal order.
    /// Drives Text–Timestamp Alignment. `f32` (not `Eq`) so frame-rate math round-trips faithfully.
    pub timestamps: Vec<f32>,
}

impl VideoRef {
    /// Construct, validating that there is exactly one timestamp per frame and at least one frame.
    pub fn new(frames: Vec<ImageRef>, timestamps: Vec<f32>) -> Result<Self, String> {
        if frames.is_empty() {
            return Err("VideoRef: needs at least one frame".to_string());
        }
        if frames.len() != timestamps.len() {
            return Err(format!(
                "VideoRef: {} frames but {} timestamps (need one timestamp per frame)",
                frames.len(),
                timestamps.len()
            ));
        }
        Ok(Self { frames, timestamps })
    }

    /// The number of sampled frames.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Whether the video has no frames (never true for a `VideoRef` built via [`Self::new`]).
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }
}

/// A single piece of message content.
///
/// Not `Eq` (only `PartialEq`): [`Content::Video`] carries `f32` timestamps.
#[derive(Clone, Debug, PartialEq)]
pub enum Content {
    /// Text content.
    Text(String),
    /// Image content (vision input).
    Image(ImageRef),
    /// Video content (sampled frames + per-frame timestamps for Text–Timestamp Alignment).
    Video(VideoRef),
}

impl Content {
    /// Convenience: text content from anything string-like.
    pub fn text(s: impl Into<String>) -> Self {
        Content::Text(s.into())
    }

    /// Whether this is image content.
    pub fn is_image(&self) -> bool {
        matches!(self, Content::Image(_))
    }

    /// Whether this is video content.
    pub fn is_video(&self) -> bool {
        matches!(self, Content::Video(_))
    }
}

/// One turn in a conversation: a role and its ordered content blocks.
///
/// Not `Eq` (only `PartialEq`): a [`tool_calls`](Self::tool_calls) argument is a `serde_json::Value`,
/// which is `PartialEq` but not `Eq` (it can hold a float).
#[derive(Clone, Debug, PartialEq)]
pub struct Message {
    /// Who authored the turn.
    pub role: Role,
    /// The ordered content of the turn (text and/or images).
    pub content: Vec<Content>,
    /// An assistant turn's prior reasoning ("thinking"), separated from [`content`](Self::content) —
    /// the multi-turn input dual of [`TextLlmOutput::thinking`](crate::TextLlmOutput::thinking).
    /// `None` for non-reasoning turns. When set, it is exposed to a chat template as the standard
    /// `reasoning_content` (and `thinking`) message field, so a reasoning model's template can
    /// re-render or strip prior-turn reasoning per its own policy (e.g. Qwen3 keeps it only for the
    /// most recent turn). Carry back a previous turn's `output.thinking` here to round-trip faithfully.
    pub thinking: Option<String>,
    /// An assistant turn's tool / function calls — the multi-turn input dual of
    /// [`TextLlmOutput::tool_calls`](crate::TextLlmOutput::tool_calls). Empty for non-tool turns. When
    /// non-empty it is exposed to a chat template as the standard `tool_calls` message field, so a
    /// tool-capable model's template re-renders the prior call(s) (e.g. Qwen3.6's `<tool_call>` XML).
    /// Carry back a previous turn's `output.tool_calls` here, paired with the [`Role::Tool`] result
    /// turn(s), to continue a multi-step tool exchange faithfully.
    pub tool_calls: Vec<crate::tool::ToolCall>,
}

impl Message {
    /// A message with a single text block.
    pub fn text(role: Role, text: impl Into<String>) -> Self {
        Self {
            role,
            content: vec![Content::Text(text.into())],
            thinking: None,
            tool_calls: Vec::new(),
        }
    }

    /// Attach prior reasoning ("thinking") to this turn (builder style); typically an assistant turn
    /// carrying a previous generation's [`TextLlmOutput::thinking`](crate::TextLlmOutput::thinking).
    pub fn with_thinking(mut self, thinking: impl Into<String>) -> Self {
        self.thinking = Some(thinking.into());
        self
    }

    /// Attach tool / function calls to this turn (builder style); typically an assistant turn carrying
    /// a previous generation's [`TextLlmOutput::tool_calls`](crate::TextLlmOutput::tool_calls) for a
    /// multi-step tool exchange.
    pub fn with_tool_calls(mut self, tool_calls: Vec<crate::tool::ToolCall>) -> Self {
        self.tool_calls = tool_calls;
        self
    }

    /// A system text turn.
    pub fn system(text: impl Into<String>) -> Self {
        Self::text(Role::System, text)
    }

    /// A user text turn.
    pub fn user(text: impl Into<String>) -> Self {
        Self::text(Role::User, text)
    }

    /// An assistant text turn.
    pub fn assistant(text: impl Into<String>) -> Self {
        Self::text(Role::Assistant, text)
    }

    /// Concatenated text of this turn (image and video blocks omitted).
    pub fn text_content(&self) -> String {
        self.content
            .iter()
            .filter_map(|c| match c {
                Content::Text(t) => Some(t.as_str()),
                Content::Image(_) | Content::Video(_) => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Whether the turn contains any image content.
    pub fn has_image(&self) -> bool {
        self.content.iter().any(Content::is_image)
    }

    /// Whether the turn contains any video content.
    pub fn has_video(&self) -> bool {
        self.content.iter().any(Content::is_video)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_validates_pixel_count() {
        assert!(ImageRef::new(2, 2, vec![0u8; 12]).is_ok());
        assert!(ImageRef::new(2, 2, vec![0u8; 10]).is_err());
    }

    #[test]
    fn video_validates_frame_timestamp_pairing() {
        let frame = ImageRef::new(2, 2, vec![0u8; 12]).unwrap();
        // One frame, one timestamp: ok.
        assert!(VideoRef::new(vec![frame.clone()], vec![0.0]).is_ok());
        // Two frames, two timestamps: ok.
        let v = VideoRef::new(vec![frame.clone(), frame.clone()], vec![0.0, 0.5]).unwrap();
        assert_eq!(v.len(), 2);
        assert!(!v.is_empty());
        // Mismatched counts: err.
        assert!(VideoRef::new(vec![frame.clone()], vec![0.0, 0.5]).is_err());
        // No frames: err.
        assert!(VideoRef::new(vec![], vec![]).is_err());
    }

    #[test]
    fn message_video_helpers() {
        let frame = ImageRef::new(1, 1, vec![1, 2, 3]).unwrap();
        let video = VideoRef::new(vec![frame], vec![0.0]).unwrap();
        let m = Message {
            role: Role::User,
            content: vec![Content::Video(video), Content::text("describe")],
            thinking: None,
            tool_calls: Vec::new(),
        };
        assert!(m.has_video());
        assert!(!m.has_image());
        // Video blocks are omitted from the flattened text, like images.
        assert_eq!(m.text_content(), "describe");
    }

    #[test]
    fn role_names() {
        assert_eq!(Role::Assistant.as_str(), "assistant");
    }

    #[test]
    fn message_text_helpers() {
        let m = Message::user("hi");
        assert_eq!(m.role, Role::User);
        assert_eq!(m.text_content(), "hi");
        assert!(!m.has_image());
        assert_eq!(m.thinking, None);
        assert!(m.tool_calls.is_empty());
    }

    #[test]
    fn with_thinking_attaches_reasoning() {
        let m = Message::assistant("the answer").with_thinking("the reasoning");
        assert_eq!(m.role, Role::Assistant);
        assert_eq!(m.text_content(), "the answer");
        assert_eq!(m.thinking.as_deref(), Some("the reasoning"));
    }

    #[test]
    fn with_tool_calls_attaches_calls() {
        let call = crate::tool::ToolCall::new("get_weather", serde_json::Map::new());
        let m = Message::assistant("").with_tool_calls(vec![call.clone()]);
        assert_eq!(m.role, Role::Assistant);
        assert_eq!(m.tool_calls, vec![call]);
    }
}
