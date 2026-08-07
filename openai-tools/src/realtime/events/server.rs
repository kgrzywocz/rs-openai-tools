//! Server-to-client events for the Realtime API.

use serde::{Deserialize, Serialize};

use crate::realtime::audio::{AudioFormat, Voice};
use crate::realtime::conversation::ItemStatus;
use crate::realtime::session::{MaxTokens, Modality, RealtimeTool, ToolChoice};
use crate::realtime::vad::TurnDetection;

/// Server events received from the OpenAI Realtime API.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum ServerEvent {
    // ==================== Session Events ====================
    /// Session was created (first event after connection).
    #[serde(rename = "session.created")]
    SessionCreated(SessionCreatedEvent),

    /// Session configuration was updated.
    #[serde(rename = "session.updated")]
    SessionUpdated(SessionUpdatedEvent),

    // ==================== Conversation Events ====================
    /// Conversation was created.
    #[serde(rename = "conversation.created")]
    ConversationCreated(ConversationCreatedEvent),

    /// Conversation item was created.
    #[serde(rename = "conversation.item.created")]
    ConversationItemCreated(ConversationItemCreatedEvent),

    /// Conversation item was retrieved.
    #[serde(rename = "conversation.item.retrieved")]
    ConversationItemRetrieved(ConversationItemRetrievedEvent),

    /// Conversation item was deleted.
    #[serde(rename = "conversation.item.deleted")]
    ConversationItemDeleted(ConversationItemDeletedEvent),

    /// Conversation item was truncated.
    #[serde(rename = "conversation.item.truncated")]
    ConversationItemTruncated(ConversationItemTruncatedEvent),

    /// Input audio transcription completed.
    #[serde(rename = "conversation.item.input_audio_transcription.completed")]
    InputAudioTranscriptionCompleted(InputAudioTranscriptionCompletedEvent),

    /// Input audio transcription failed.
    #[serde(rename = "conversation.item.input_audio_transcription.failed")]
    InputAudioTranscriptionFailed(InputAudioTranscriptionFailedEvent),

    // ==================== Input Audio Buffer Events ====================
    /// Input audio buffer was committed.
    #[serde(rename = "input_audio_buffer.committed")]
    InputAudioBufferCommitted(InputAudioBufferCommittedEvent),

    /// Input audio buffer was cleared.
    #[serde(rename = "input_audio_buffer.cleared")]
    InputAudioBufferCleared(InputAudioBufferClearedEvent),

    /// Speech started in input audio.
    #[serde(rename = "input_audio_buffer.speech_started")]
    InputAudioBufferSpeechStarted(SpeechStartedEvent),

    /// Speech stopped in input audio.
    #[serde(rename = "input_audio_buffer.speech_stopped")]
    InputAudioBufferSpeechStopped(SpeechStoppedEvent),

    // ==================== Output Audio Buffer Events (WebRTC) ====================
    /// Output audio buffer playback started.
    #[serde(rename = "output_audio_buffer.started")]
    OutputAudioBufferStarted(OutputAudioBufferEvent),

    /// Output audio buffer playback stopped.
    #[serde(rename = "output_audio_buffer.stopped")]
    OutputAudioBufferStopped(OutputAudioBufferStoppedEvent),

    /// Output audio buffer was cleared.
    #[serde(rename = "output_audio_buffer.cleared")]
    OutputAudioBufferCleared(OutputAudioBufferEvent),

    // ==================== Response Events ====================
    /// Response was created.
    #[serde(rename = "response.created")]
    ResponseCreated(ResponseCreatedEvent),

    /// Response generation completed.
    #[serde(rename = "response.done")]
    ResponseDone(ResponseDoneEvent),

    /// Output item was added to response.
    #[serde(rename = "response.output_item.added")]
    ResponseOutputItemAdded(ResponseOutputItemEvent),

    /// Output item completed.
    #[serde(rename = "response.output_item.done")]
    ResponseOutputItemDone(ResponseOutputItemEvent),

    /// Content part was added.
    #[serde(rename = "response.content_part.added")]
    ResponseContentPartAdded(ResponseContentPartEvent),

    /// Content part completed.
    #[serde(rename = "response.content_part.done")]
    ResponseContentPartDone(ResponseContentPartEvent),

    /// Text delta received.
    #[serde(rename = "response.text.delta")]
    ResponseTextDelta(ResponseTextDeltaEvent),

    /// Text output completed.
    #[serde(rename = "response.text.done")]
    ResponseTextDone(ResponseTextDoneEvent),

    /// Audio delta received.
    #[serde(rename = "response.audio.delta")]
    ResponseAudioDelta(ResponseAudioDeltaEvent),

    /// Audio output completed.
    #[serde(rename = "response.audio.done")]
    ResponseAudioDone(ResponseAudioDoneEvent),

    /// Audio transcript delta received.
    #[serde(rename = "response.audio_transcript.delta")]
    ResponseAudioTranscriptDelta(ResponseAudioTranscriptDeltaEvent),

    /// Audio transcript completed.
    #[serde(rename = "response.audio_transcript.done")]
    ResponseAudioTranscriptDone(ResponseAudioTranscriptDoneEvent),

    /// Function call arguments delta.
    #[serde(rename = "response.function_call_arguments.delta")]
    ResponseFunctionCallArgumentsDelta(ResponseFunctionCallArgumentsDeltaEvent),

    /// Function call arguments completed.
    #[serde(rename = "response.function_call_arguments.done")]
    ResponseFunctionCallArgumentsDone(ResponseFunctionCallArgumentsDoneEvent),

    // ==================== Rate Limits ====================
    /// Rate limits updated.
    #[serde(rename = "rate_limits.updated")]
    RateLimitsUpdated(RateLimitsUpdatedEvent),

    // ==================== Error ====================
    /// Error occurred.
    #[serde(rename = "error")]
    Error(ErrorEvent),

    // ==================== Unknown ====================
    /// An event type this version of the library does not know about.
    ///
    /// OpenAI adds realtime events regularly. Without this arm a single
    /// unrecognised `type` would fail deserialization and tear down the
    /// session, so unknown events are surfaced here instead.
    ///
    /// Note that the payload is not preserved: `serde`'s `other` attribute is
    /// only allowed on a unit variant.
    #[serde(other)]
    Unknown,
}

// ==================== Session Event Types ====================

/// Session created event payload.
#[derive(Debug, Clone, Deserialize)]
pub struct SessionCreatedEvent {
    pub event_id: String,
    pub session: SessionInfo,
}

/// Session updated event payload.
#[derive(Debug, Clone, Deserialize)]
pub struct SessionUpdatedEvent {
    pub event_id: String,
    pub session: SessionInfo,
}

/// Session information.
#[derive(Debug, Clone, Deserialize)]
pub struct SessionInfo {
    pub id: String,
    pub object: String,
    pub model: String,
    #[serde(default)]
    pub modalities: Vec<Modality>,
    #[serde(default)]
    pub instructions: String,
    pub voice: Option<Voice>,
    pub input_audio_format: Option<AudioFormat>,
    pub output_audio_format: Option<AudioFormat>,
    pub turn_detection: Option<TurnDetection>,
    #[serde(default)]
    pub tools: Vec<RealtimeTool>,
    pub tool_choice: Option<ToolChoice>,
    pub temperature: Option<f32>,
    pub max_response_output_tokens: Option<MaxTokens>,
}

// ==================== Conversation Event Types ====================

/// Conversation created event.
#[derive(Debug, Clone, Deserialize)]
pub struct ConversationCreatedEvent {
    pub event_id: String,
    pub conversation: ConversationInfo,
}

/// Conversation information.
#[derive(Debug, Clone, Deserialize)]
pub struct ConversationInfo {
    pub id: String,
    pub object: String,
}

/// Conversation item created event.
#[derive(Debug, Clone, Deserialize)]
pub struct ConversationItemCreatedEvent {
    pub event_id: String,
    #[serde(default)]
    pub previous_item_id: Option<String>,
    pub item: ResponseItem,
}

/// Conversation item retrieved event.
#[derive(Debug, Clone, Deserialize)]
pub struct ConversationItemRetrievedEvent {
    pub event_id: String,
    pub item: ResponseItem,
}

/// Conversation item deleted event.
#[derive(Debug, Clone, Deserialize)]
pub struct ConversationItemDeletedEvent {
    pub event_id: String,
    pub item_id: String,
}

/// Conversation item truncated event.
#[derive(Debug, Clone, Deserialize)]
pub struct ConversationItemTruncatedEvent {
    pub event_id: String,
    pub item_id: String,
    pub content_index: u32,
    pub audio_end_ms: u32,
}

/// Input audio transcription completed event.
#[derive(Debug, Clone, Deserialize)]
pub struct InputAudioTranscriptionCompletedEvent {
    pub event_id: String,
    pub item_id: String,
    pub content_index: u32,
    pub transcript: String,
}

/// Input audio transcription failed event.
#[derive(Debug, Clone, Deserialize)]
pub struct InputAudioTranscriptionFailedEvent {
    pub event_id: String,
    pub item_id: String,
    pub content_index: u32,
    pub error: RealtimeError,
}

// ==================== Input Audio Buffer Event Types ====================

/// Input audio buffer committed event.
#[derive(Debug, Clone, Deserialize)]
pub struct InputAudioBufferCommittedEvent {
    pub event_id: String,
    #[serde(default)]
    pub previous_item_id: Option<String>,
    pub item_id: String,
}

/// Input audio buffer cleared event.
#[derive(Debug, Clone, Deserialize)]
pub struct InputAudioBufferClearedEvent {
    pub event_id: String,
}

/// Speech started event.
#[derive(Debug, Clone, Deserialize)]
pub struct SpeechStartedEvent {
    pub event_id: String,
    pub audio_start_ms: u32,
    pub item_id: String,
}

/// Speech stopped event.
#[derive(Debug, Clone, Deserialize)]
pub struct SpeechStoppedEvent {
    pub event_id: String,
    pub audio_end_ms: u32,
    #[serde(default)]
    pub item_id: Option<String>,
}

// ==================== Output Audio Buffer Event Types ====================

/// Output audio buffer event (started/cleared).
#[derive(Debug, Clone, Deserialize)]
pub struct OutputAudioBufferEvent {
    pub event_id: String,
    pub response_id: String,
}

/// Output audio buffer stopped event.
#[derive(Debug, Clone, Deserialize)]
pub struct OutputAudioBufferStoppedEvent {
    pub event_id: String,
    pub response_id: String,
    pub audio_end_ms: u32,
    pub item_id: String,
}

// ==================== Response Event Types ====================

/// Response created event.
#[derive(Debug, Clone, Deserialize)]
pub struct ResponseCreatedEvent {
    pub event_id: String,
    pub response: ResponseInfo,
}

/// Response done event.
#[derive(Debug, Clone, Deserialize)]
pub struct ResponseDoneEvent {
    pub event_id: String,
    pub response: ResponseInfo,
}

/// Response information.
#[derive(Debug, Clone, Deserialize)]
pub struct ResponseInfo {
    pub id: String,
    pub object: String,
    pub status: ResponseStatus,
    #[serde(default)]
    pub status_details: Option<serde_json::Value>,
    #[serde(default)]
    pub output: Vec<ResponseItem>,
    #[serde(default)]
    pub usage: Option<RealtimeUsage>,
}

/// Response status.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ResponseStatus {
    InProgress,
    Completed,
    Cancelled,
    Incomplete,
    Failed,
    /// A status this version of the library does not know about.
    ///
    /// OpenAI can add lifecycle states at any time; capturing the raw value
    /// keeps the surrounding response parseable.
    #[serde(untagged)]
    Other(String),
}

/// Response item (output item in a response).
#[derive(Debug, Clone, Deserialize)]
pub struct ResponseItem {
    pub id: String,
    pub object: String,
    #[serde(rename = "type")]
    pub item_type: String,
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Vec<ResponseContentPart>,
    #[serde(default)]
    pub status: Option<ItemStatus>,
    // Function call fields
    #[serde(default)]
    pub call_id: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
    #[serde(default)]
    pub output: Option<String>,
}

/// Response content part.
#[derive(Debug, Clone, Deserialize)]
pub struct ResponseContentPart {
    #[serde(rename = "type")]
    pub content_type: String,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub audio: Option<String>,
    #[serde(default)]
    pub transcript: Option<String>,
}

/// Response output item event.
#[derive(Debug, Clone, Deserialize)]
pub struct ResponseOutputItemEvent {
    pub event_id: String,
    pub response_id: String,
    pub output_index: u32,
    pub item: ResponseItem,
}

/// Response content part event.
#[derive(Debug, Clone, Deserialize)]
pub struct ResponseContentPartEvent {
    pub event_id: String,
    pub response_id: String,
    pub item_id: String,
    pub output_index: u32,
    pub content_index: u32,
    pub part: ResponseContentPart,
}

/// Response text delta event.
#[derive(Debug, Clone, Deserialize)]
pub struct ResponseTextDeltaEvent {
    pub event_id: String,
    pub response_id: String,
    pub item_id: String,
    pub output_index: u32,
    pub content_index: u32,
    pub delta: String,
}

/// Response text done event.
#[derive(Debug, Clone, Deserialize)]
pub struct ResponseTextDoneEvent {
    pub event_id: String,
    pub response_id: String,
    pub item_id: String,
    pub output_index: u32,
    pub content_index: u32,
    pub text: String,
}

/// Response audio delta event.
#[derive(Debug, Clone, Deserialize)]
pub struct ResponseAudioDeltaEvent {
    pub event_id: String,
    pub response_id: String,
    pub item_id: String,
    pub output_index: u32,
    pub content_index: u32,
    /// Base64-encoded audio chunk.
    pub delta: String,
}

/// Response audio done event.
#[derive(Debug, Clone, Deserialize)]
pub struct ResponseAudioDoneEvent {
    pub event_id: String,
    pub response_id: String,
    pub item_id: String,
    pub output_index: u32,
    pub content_index: u32,
}

/// Response audio transcript delta event.
#[derive(Debug, Clone, Deserialize)]
pub struct ResponseAudioTranscriptDeltaEvent {
    pub event_id: String,
    pub response_id: String,
    pub item_id: String,
    pub output_index: u32,
    pub content_index: u32,
    pub delta: String,
}

/// Response audio transcript done event.
#[derive(Debug, Clone, Deserialize)]
pub struct ResponseAudioTranscriptDoneEvent {
    pub event_id: String,
    pub response_id: String,
    pub item_id: String,
    pub output_index: u32,
    pub content_index: u32,
    pub transcript: String,
}

/// Response function call arguments delta event.
#[derive(Debug, Clone, Deserialize)]
pub struct ResponseFunctionCallArgumentsDeltaEvent {
    pub event_id: String,
    pub response_id: String,
    pub item_id: String,
    pub output_index: u32,
    pub call_id: String,
    pub delta: String,
}

/// Response function call arguments done event.
#[derive(Debug, Clone, Deserialize)]
pub struct ResponseFunctionCallArgumentsDoneEvent {
    pub event_id: String,
    pub response_id: String,
    pub item_id: String,
    pub output_index: u32,
    pub call_id: String,
    pub name: String,
    pub arguments: String,
}

// ==================== Rate Limits ====================

/// Rate limits updated event.
#[derive(Debug, Clone, Deserialize)]
pub struct RateLimitsUpdatedEvent {
    pub event_id: String,
    pub rate_limits: Vec<RateLimit>,
}

/// Rate limit information.
#[derive(Debug, Clone, Deserialize)]
pub struct RateLimit {
    pub name: String,
    pub limit: u32,
    pub remaining: u32,
    pub reset_seconds: f32,
}

// ==================== Error ====================

/// Error event.
#[derive(Debug, Clone, Deserialize)]
pub struct ErrorEvent {
    pub event_id: String,
    pub error: RealtimeError,
}

/// Realtime API error.
#[derive(Debug, Clone, Deserialize)]
pub struct RealtimeError {
    #[serde(rename = "type")]
    pub error_type: Option<String>,
    pub code: Option<String>,
    pub message: String,
    #[serde(default)]
    pub param: Option<String>,
    #[serde(default)]
    pub event_id: Option<String>,
}

// ==================== Usage ====================

/// Token usage information.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct RealtimeUsage {
    pub total_tokens: u32,
    pub input_tokens: u32,
    pub output_tokens: u32,
    #[serde(default)]
    pub input_token_details: Option<InputTokenDetails>,
    #[serde(default)]
    pub output_token_details: Option<OutputTokenDetails>,
}

/// Input token details.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct InputTokenDetails {
    #[serde(default)]
    pub cached_tokens: u32,
    #[serde(default)]
    pub text_tokens: u32,
    #[serde(default)]
    pub audio_tokens: u32,
}

/// Output token details.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct OutputTokenDetails {
    #[serde(default)]
    pub text_tokens: u32,
    #[serde(default)]
    pub audio_tokens: u32,
}

impl ServerEvent {
    /// Check if this is an error event.
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error(_))
    }

    /// Get the event ID if available.
    pub fn event_id(&self) -> Option<&str> {
        match self {
            Self::SessionCreated(e) => Some(&e.event_id),
            Self::SessionUpdated(e) => Some(&e.event_id),
            Self::ConversationCreated(e) => Some(&e.event_id),
            Self::ConversationItemCreated(e) => Some(&e.event_id),
            Self::ConversationItemRetrieved(e) => Some(&e.event_id),
            Self::ConversationItemDeleted(e) => Some(&e.event_id),
            Self::ConversationItemTruncated(e) => Some(&e.event_id),
            Self::InputAudioTranscriptionCompleted(e) => Some(&e.event_id),
            Self::InputAudioTranscriptionFailed(e) => Some(&e.event_id),
            Self::InputAudioBufferCommitted(e) => Some(&e.event_id),
            Self::InputAudioBufferCleared(e) => Some(&e.event_id),
            Self::InputAudioBufferSpeechStarted(e) => Some(&e.event_id),
            Self::InputAudioBufferSpeechStopped(e) => Some(&e.event_id),
            Self::OutputAudioBufferStarted(e) => Some(&e.event_id),
            Self::OutputAudioBufferStopped(e) => Some(&e.event_id),
            Self::OutputAudioBufferCleared(e) => Some(&e.event_id),
            Self::ResponseCreated(e) => Some(&e.event_id),
            Self::ResponseDone(e) => Some(&e.event_id),
            Self::ResponseOutputItemAdded(e) => Some(&e.event_id),
            Self::ResponseOutputItemDone(e) => Some(&e.event_id),
            Self::ResponseContentPartAdded(e) => Some(&e.event_id),
            Self::ResponseContentPartDone(e) => Some(&e.event_id),
            Self::ResponseTextDelta(e) => Some(&e.event_id),
            Self::ResponseTextDone(e) => Some(&e.event_id),
            Self::ResponseAudioDelta(e) => Some(&e.event_id),
            Self::ResponseAudioDone(e) => Some(&e.event_id),
            Self::ResponseAudioTranscriptDelta(e) => Some(&e.event_id),
            Self::ResponseAudioTranscriptDone(e) => Some(&e.event_id),
            Self::ResponseFunctionCallArgumentsDelta(e) => Some(&e.event_id),
            Self::ResponseFunctionCallArgumentsDone(e) => Some(&e.event_id),
            Self::RateLimitsUpdated(e) => Some(&e.event_id),
            Self::Error(e) => Some(&e.event_id),
            // The payload of an unrecognised event is not retained.
            Self::Unknown => None,
        }
    }
}
