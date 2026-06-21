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

/// A single piece of message content.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Content {
    /// Text content.
    Text(String),
    /// Image content (vision input).
    Image(ImageRef),
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
}

/// One turn in a conversation: a role and its ordered content blocks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    /// Who authored the turn.
    pub role: Role,
    /// The ordered content of the turn (text and/or images).
    pub content: Vec<Content>,
}

impl Message {
    /// A message with a single text block.
    pub fn text(role: Role, text: impl Into<String>) -> Self {
        Self {
            role,
            content: vec![Content::Text(text.into())],
        }
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

    /// Concatenated text of this turn (image blocks omitted).
    pub fn text_content(&self) -> String {
        self.content
            .iter()
            .filter_map(|c| match c {
                Content::Text(t) => Some(t.as_str()),
                Content::Image(_) => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Whether the turn contains any image content.
    pub fn has_image(&self) -> bool {
        self.content.iter().any(Content::is_image)
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
    fn role_names() {
        assert_eq!(Role::Assistant.as_str(), "assistant");
    }

    #[test]
    fn message_text_helpers() {
        let m = Message::user("hi");
        assert_eq!(m.role, Role::User);
        assert_eq!(m.text_content(), "hi");
        assert!(!m.has_image());
    }
}
