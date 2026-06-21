//! Chat-template rendering: a conversation → the model's prompt string.
//!
//! [`ChatTemplate`] is the host-policy seam. This module ships the common **typed** templates
//! (Llama-3, ChatML); the `minijinja`-backed renderer that reads a model's own `chat_template`
//! lands in story 7164 behind the same trait. Templates render the text of each turn; the vision
//! image-placeholder splice is the backend VLM path's concern (story 7157).

use crate::error::Result;
use crate::message::Message;

/// Renders a conversation into a single prompt string the tokenizer then encodes.
pub trait ChatTemplate {
    /// Render `messages`. When `add_generation_prompt` is set, append the opening of an assistant
    /// turn so the model continues as the assistant.
    fn render(&self, messages: &[Message], add_generation_prompt: bool) -> Result<String>;
}

/// The Llama-3 instruct chat format.
#[derive(Clone, Copy, Debug, Default)]
pub struct Llama3Template;

impl ChatTemplate for Llama3Template {
    fn render(&self, messages: &[Message], add_generation_prompt: bool) -> Result<String> {
        let mut out = String::from("<|begin_of_text|>");
        for m in messages {
            out.push_str("<|start_header_id|>");
            out.push_str(m.role.as_str());
            out.push_str("<|end_header_id|>\n\n");
            out.push_str(&m.text_content());
            out.push_str("<|eot_id|>");
        }
        if add_generation_prompt {
            out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
        }
        Ok(out)
    }
}

/// The ChatML format (`<|im_start|>role … <|im_end|>`), used by Qwen and others.
#[derive(Clone, Copy, Debug, Default)]
pub struct ChatMlTemplate;

impl ChatTemplate for ChatMlTemplate {
    fn render(&self, messages: &[Message], add_generation_prompt: bool) -> Result<String> {
        let mut out = String::new();
        for m in messages {
            out.push_str("<|im_start|>");
            out.push_str(m.role.as_str());
            out.push('\n');
            out.push_str(&m.text_content());
            out.push_str("<|im_end|>\n");
        }
        if add_generation_prompt {
            out.push_str("<|im_start|>assistant\n");
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{Message, Role};

    #[test]
    fn llama3_renders_turns_and_generation_prompt() {
        let msgs = vec![Message::system("be brief"), Message::user("hi")];
        let s = Llama3Template.render(&msgs, true).unwrap();
        assert!(s.starts_with("<|begin_of_text|>"));
        assert!(s.contains("<|start_header_id|>system<|end_header_id|>\n\nbe brief<|eot_id|>"));
        assert!(s.contains("<|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>"));
        assert!(s.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn chatml_renders_turns() {
        let msgs = vec![Message::text(Role::User, "hi")];
        let s = ChatMlTemplate.render(&msgs, true).unwrap();
        assert_eq!(s, "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n");
    }
}
