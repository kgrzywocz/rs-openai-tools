//! Conversation item types for the Realtime API.

use serde::{Deserialize, Serialize};

/// A conversation item in the Realtime API.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ConversationItem {
    /// A message item (user, assistant, or system).
    Message(MessageItem),
    /// A function call made by the assistant.
    FunctionCall(FunctionCallItem),
    /// The output of a function call.
    FunctionCallOutput(FunctionCallOutputItem),
}

/// A message in the conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageItem {
    /// Unique identifier for this item.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    /// Role of the message sender.
    pub role: MessageRole,

    /// Content parts of the message.
    pub content: Vec<ContentPart>,
}

impl MessageItem {
    /// Create a new user text message.
    pub fn user_text(text: impl Into<String>) -> Self {
        Self { id: None, role: MessageRole::User, content: vec![ContentPart::InputText { text: text.into() }] }
    }

    /// Create a new assistant text message.
    pub fn assistant_text(text: impl Into<String>) -> Self {
        Self { id: None, role: MessageRole::Assistant, content: vec![ContentPart::Text { text: text.into() }] }
    }

    /// Create a new system message.
    pub fn system(text: impl Into<String>) -> Self {
        Self { id: None, role: MessageRole::System, content: vec![ContentPart::InputText { text: text.into() }] }
    }

    /// Create a new user audio message.
    pub fn user_audio(audio_base64: impl Into<String>) -> Self {
        Self { id: None, role: MessageRole::User, content: vec![ContentPart::InputAudio { audio: audio_base64.into(), transcript: None }] }
    }
}

/// Role of a message sender.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    /// System message (instructions).
    System,
    /// User message.
    User,
    /// Assistant message.
    Assistant,
}

/// A content part within a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// Text input from user.
    InputText { text: String },
    /// Audio input from user (base64 encoded).
    InputAudio {
        audio: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        transcript: Option<String>,
    },
    /// Text output from assistant.
    Text { text: String },
    /// Audio output from assistant.
    Audio {
        #[serde(skip_serializing_if = "Option::is_none")]
        audio: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        transcript: Option<String>,
    },
    /// Reference to another item.
    ItemReference { id: String },
}

impl ContentPart {
    /// Create a text input content part.
    pub fn input_text(text: impl Into<String>) -> Self {
        Self::InputText { text: text.into() }
    }

    /// Create an audio input content part.
    pub fn input_audio(audio_base64: impl Into<String>) -> Self {
        Self::InputAudio { audio: audio_base64.into(), transcript: None }
    }

    /// Create a text output content part.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }
}

/// A function call made by the assistant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallItem {
    /// Unique identifier for this item.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    /// The function call ID (used to match with output).
    pub call_id: String,

    /// Name of the function being called.
    pub name: String,

    /// JSON string of function arguments.
    pub arguments: String,
}

/// The output of a function call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallOutputItem {
    /// Unique identifier for this item.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    /// The function call ID this is responding to.
    pub call_id: String,

    /// The output value (usually JSON string).
    pub output: String,
}

impl FunctionCallOutputItem {
    /// Create a new function call output.
    pub fn new(call_id: impl Into<String>, output: impl Into<String>) -> Self {
        Self { id: None, call_id: call_id.into(), output: output.into() }
    }
}

/// Item status in the conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ItemStatus {
    /// Item is being processed.
    InProgress,
    /// Item processing completed.
    Completed,
    /// Item processing was incomplete.
    Incomplete,
    /// A status this version of the library does not know about.
    ///
    /// OpenAI can add lifecycle states at any time; capturing the raw value
    /// keeps the surrounding response parseable.
    #[serde(untagged)]
    Other(String),
}
